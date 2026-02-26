from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms


def _process_row(args):
    data_root, cache_dir, image_size, rel_path = args
    img_path = data_root / rel_path
    image = Image.open(img_path).convert("RGB")
    t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    tensor = t(image)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = rel_path.replace("/", "_").replace("\\", "_")
    out_path = cache_dir / f"{safe}.pt"
    import torch

    torch.save(tensor, out_path)


def preprocess_parallel(data_root: Path, out_dir: Path, workers: int = 4):
    meta = pd.read_csv(data_root / "metadata.csv")
    image_size = 64
    args = [(data_root, out_dir, image_size, rel) for rel in meta["image_path"].tolist()]
    with Pool(processes=workers) as pool:
        pool.map(_process_row, args)
