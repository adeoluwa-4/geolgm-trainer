from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..config import AppConfig, load_config
from ..data.dataset import GeoImageDataset
from ..data.buckets import region_bucket, time_bucket
from ..models.resnet_head import build_resnet18
from ..models.simple_cnn import build_simple_cnn


def _build_model(cfg: AppConfig):
    if cfg.model.name == "resnet18":
        return build_resnet18(cfg.model.num_classes, cfg.model.pretrained)
    return build_simple_cnn(cfg.model.num_classes)


def evaluate_split(cfg: AppConfig, run_dir: Path, split: str = "val") -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GeoImageDataset(
        data_root=Path(cfg.dataset.root),
        index_path=Path(cfg.dataset.index_path),
        split=split,
        image_size=cfg.dataset.image_size,
        cache=False,
        cache_dir=Path(cfg.dataset.cache_dir),
    )
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False)

    model = _build_model(cfg).to(device)
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])

    model.eval()
    correct = 0
    total = 0
    region_correct = {"north": 0, "south": 0}
    region_total = {"north": 0, "south": 0}
    time_correct = {"day": 0, "night": 0, "unknown": 0}
    time_total = {"day": 0, "night": 0, "unknown": 0}

    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                lat = float(meta["lat"][i])
                ts = meta["timestamp"][i]
                region = region_bucket(lat, dataset.median_lat)
                tb = time_bucket(ts)

                region_total[region] += 1
                if preds[i] == labels[i]:
                    region_correct[region] += 1

                time_total[tb] += 1
                if preds[i] == labels[i]:
                    time_correct[tb] += 1

    acc = correct / max(1, total)
    region_acc = {k: region_correct[k] / max(1, region_total[k]) for k in region_total}
    time_acc = {k: time_correct[k] / max(1, time_total[k]) for k in time_total}

    metrics = {"acc": acc}
    metrics.update({f"region_{k}_acc": v for k, v in region_acc.items()})
    metrics.update({f"time_{k}_acc": v for k, v in time_acc.items()})
    return metrics


def evaluate_run(run_id: str, split: str = "val") -> None:
    run_dir = Path("runs") / run_id
    cfg_path = run_dir / "config_snapshot.yaml"
    cfg = load_config(cfg_path).config
    metrics = evaluate_split(cfg, run_dir, split=split)
    print(metrics)
