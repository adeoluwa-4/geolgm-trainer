from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import AppConfig
from ..data.dataset import GeoImageDataset
from ..data.buckets import region_bucket, time_bucket
from ..models.resnet_head import build_resnet18
from ..models.simple_cnn import build_simple_cnn
from ..tracking.db import insert_metric
from ..tracking.artifacts import save_plots, save_confusion_matrix
from .checkpoint import save_checkpoint, load_checkpoint, latest_checkpoint
from .profiler import StepTimer


def _build_model(cfg: AppConfig):
    if cfg.model.name == "resnet18":
        return build_resnet18(cfg.model.num_classes, cfg.model.pretrained)
    return build_simple_cnn(cfg.model.num_classes)


def _evaluate(model, loader, device, median_lat: float, criterion):
    model.eval()
    correct = 0
    total = 0
    region_correct = {"north": 0, "south": 0}
    region_total = {"north": 0, "south": 0}
    time_correct = {"day": 0, "night": 0, "unknown": 0}
    time_total = {"day": 0, "night": 0, "unknown": 0}
    all_preds = []
    all_labels = []

    total_loss = 0.0
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                lat = float(meta["lat"][i])
                ts = meta["timestamp"][i]
                region = region_bucket(lat, median_lat)
                tb = time_bucket(ts)

                region_total[region] += 1
                if preds[i] == labels[i]:
                    region_correct[region] += 1

                time_total[tb] += 1
                if preds[i] == labels[i]:
                    time_correct[tb] += 1

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, len(loader))
    region_acc = {k: region_correct[k] / max(1, region_total[k]) for k in region_total}
    time_acc = {k: time_correct[k] / max(1, time_total[k]) for k in time_total}

    return avg_loss, acc, region_acc, time_acc, np.array(all_labels), np.array(all_preds)


def train_model(cfg: AppConfig, run_id: str, run_dir: Path, jsonl_logger, resume: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = GeoImageDataset(
        data_root=Path(cfg.dataset.root),
        index_path=Path(cfg.dataset.index_path),
        split="train",
        image_size=cfg.dataset.image_size,
        cache=cfg.dataset.cache,
        cache_dir=Path(cfg.dataset.cache_dir),
    )
    val_ds = GeoImageDataset(
        data_root=Path(cfg.dataset.root),
        index_path=Path(cfg.dataset.index_path),
        split="val",
        image_size=cfg.dataset.image_size,
        cache=False,
        cache_dir=Path(cfg.dataset.cache_dir),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = _build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.mixed_precision)

    start_epoch = 0
    best_acc = 0.0
    checkpoints_dir = run_dir / "checkpoints"

    if resume:
        ckpt_path = latest_checkpoint(checkpoints_dir)
        if ckpt_path:
            start_epoch, best_acc = load_checkpoint(ckpt_path, model, optimizer)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "images_per_sec": []}

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_images = 0
        epoch_time = 0.0

        for step, (images, labels, _) in enumerate(train_loader):
            try:
                timer = StepTimer()
                images = images.to(device)
                labels = labels.to(device)
                timer.mark_loader_done()

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=cfg.train.mixed_precision):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                prof = timer.mark_step_done()
            except Exception:
                if cfg.train.fail_skip_batch:
                    continue
                raise

            batch_size = labels.size(0)
            batch_time = (prof.dataloader_ms + prof.step_ms) / 1000
            images_per_sec = batch_size / max(1e-6, batch_time)

            epoch_loss += loss.item()
            epoch_steps += 1
            epoch_images += batch_size
            epoch_time += batch_time

            if step % cfg.logging.log_interval == 0:
                metric = {
                    "run_id": run_id,
                    "step": epoch * len(train_loader) + step,
                    "split": "train",
                    "loss": loss.item(),
                    "acc": None,
                    "images_per_sec": images_per_sec,
                    "dataloader_ms": prof.dataloader_ms,
                    "step_ms": prof.step_ms,
                    "cpu_mem_mb": prof.cpu_mem_mb,
                    "gpu_mem_mb": prof.gpu_mem_mb,
                }
                jsonl_logger.log(metric)
                insert_metric(Path(cfg.logging.sqlite_path), metric)

        train_loss = epoch_loss / max(1, epoch_steps)
        avg_images_per_sec = epoch_images / max(1e-6, epoch_time)

        if (epoch + 1) % cfg.train.eval_every == 0:
            val_loss, val_acc, region_acc, time_acc, labels, preds = _evaluate(
                model, val_loader, device, val_ds.median_lat, criterion
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["images_per_sec"].append(avg_images_per_sec)

            metric = {
                "run_id": run_id,
                "step": epoch,
                "split": "val",
                "loss": val_loss,
                "acc": val_acc,
                "images_per_sec": avg_images_per_sec,
                "dataloader_ms": None,
                "step_ms": None,
                "cpu_mem_mb": None,
                "gpu_mem_mb": None,
            }
            jsonl_logger.log(metric)
            insert_metric(Path(cfg.logging.sqlite_path), metric)

            if cfg.artifacts.confusion_matrix:
                save_confusion_matrix(run_dir, labels, preds, Path(cfg.logging.sqlite_path))

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, best_acc)

            # log region/time metrics as JSONL only
            jsonl_logger.log({"run_id": run_id, "step": epoch, "split": "val", **region_acc})
            jsonl_logger.log({"run_id": run_id, "step": epoch, "split": "val", **time_acc})

        if (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(checkpoints_dir / f"epoch-{epoch+1}.pt", model, optimizer, epoch, best_acc)

    if cfg.artifacts.save_plots:
        save_plots(run_dir, history, Path(cfg.logging.sqlite_path))
