from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .db import insert_artifact


def ensure_artifacts_dir(run_dir: Path) -> Path:
    out = run_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_plots(run_dir: Path, history: Dict[str, List[float]], db_path: Path):
    out_dir = ensure_artifacts_dir(run_dir)

    if history.get("train_loss"):
        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.legend()
        path = out_dir / "loss_curve.png"
        plt.savefig(path)
        plt.close()
        insert_artifact(db_path, run_dir.name, "loss_curve", str(path))

    if history.get("val_acc"):
        plt.figure()
        plt.plot(history["val_acc"], label="val_acc")
        plt.legend()
        path = out_dir / "accuracy_curve.png"
        plt.savefig(path)
        plt.close()
        insert_artifact(db_path, run_dir.name, "accuracy_curve", str(path))

    if history.get("images_per_sec"):
        plt.figure()
        plt.plot(history["images_per_sec"], label="images_per_sec")
        plt.legend()
        path = out_dir / "throughput_curve.png"
        plt.savefig(path)
        plt.close()
        insert_artifact(db_path, run_dir.name, "throughput_curve", str(path))


def save_confusion_matrix(run_dir: Path, labels: np.ndarray, preds: np.ndarray, db_path: Path):
    out_dir = ensure_artifacts_dir(run_dir)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    num_classes = int(max(labels.max(), preds.max()) + 1) if len(labels) else 1
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for y, p in zip(labels, preds):
        mat[y, p] += 1

    plt.figure()
    plt.imshow(mat, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    insert_artifact(db_path, run_dir.name, "confusion_matrix", str(path))
