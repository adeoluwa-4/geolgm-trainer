from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir = Path(args.out)
    meta = pd.read_csv(data_root / "metadata.csv")
    rows = meta["image_path"].tolist()

    with Pool(processes=args.workers) as pool:
        pool.map(
            _process_row,
            [(data_root, out_dir, args.image_size, rel) for rel in rows],
        )


if __name__ == "__main__":
    main()
