from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cache import load_cached, save_cached
from .transforms import build_transforms


class GeoImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        index_path: Path,
        split: str,
        image_size: int,
        cache: bool,
        cache_dir: Path,
    ) -> None:
        self.data_root = data_root
        self.index = pd.read_csv(index_path)
        self.meta = pd.read_csv(data_root / "metadata.csv")
        self.index = self.index[self.index["split"] == split].reset_index(drop=True)
        self.transforms = build_transforms(image_size)
        self.cache = cache
        self.cache_dir = cache_dir
        self.cache_hits = 0
        self.cache_misses = 0
        self.median_lat = float(self.meta["lat"].median()) if len(self.meta) else 0.0

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]
        rel_path = row["image_path"]
        label = int(row["label"])

        cached = None
        if self.cache:
            cached = load_cached(self.cache_dir, rel_path)
        if cached is not None:
            self.cache_hits += 1
            image_tensor = cached
        else:
            self.cache_misses += 1
            image = Image.open(self.data_root / rel_path).convert("RGB")
            image_tensor = self.transforms(image)
            if self.cache:
                save_cached(self.cache_dir, rel_path, image_tensor)

        meta_row = self.meta[self.meta["image_path"] == rel_path].iloc[0].to_dict()
        return image_tensor, label, meta_row
