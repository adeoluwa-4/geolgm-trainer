from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_acc", 0.0)


def latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(checkpoints_dir.glob("epoch-*.pt"))
    return candidates[-1] if candidates else None
