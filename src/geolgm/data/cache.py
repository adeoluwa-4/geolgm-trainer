from __future__ import annotations

from pathlib import Path
import torch


def cache_path(cache_dir: Path, rel_path: str) -> Path:
    safe_name = rel_path.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_name}.pt"


def load_cached(cache_dir: Path, rel_path: str):
    path = cache_path(cache_dir, rel_path)
    if path.exists():
        return torch.load(path)
    return None


def save_cached(cache_dir: Path, rel_path: str, tensor):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, rel_path)
    torch.save(tensor, path)
    return path
