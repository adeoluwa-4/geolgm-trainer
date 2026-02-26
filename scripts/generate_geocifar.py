from __future__ import annotations

import argparse
from pathlib import Path
import random
from datetime import datetime, timedelta

import pandas as pd
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out)
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    dataset = CIFAR10(root=out_root, download=True, train=True)
    to_pil = ToPILImage()

    rows = []
    base_lat, base_lon = 37.7749, -122.4194
    devices = ["iphone", "pixel", "gopro"]

    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    for idx in range(total):
        image, label = dataset[idx]
        img_path = img_dir / f"img_{idx}.png"
        if hasattr(image, "save"):
            image.save(img_path)
        else:
            to_pil(image).save(img_path)

        lat = base_lat + random.uniform(-0.05, 0.05)
        lon = base_lon + random.uniform(-0.05, 0.05)
        heading = random.uniform(0, 360)
        ts = datetime.now() - timedelta(days=random.randint(0, 365))
        split = "train" if idx % 10 < 8 else "val" if idx % 10 == 8 else "test"

        rows.append(
            {
                "image_path": str(Path("images") / img_path.name),
                "label": int(label),
                "lat": lat,
                "lon": lon,
                "timestamp": ts.isoformat(),
                "heading": heading,
                "device": random.choice(devices),
                "split": split,
            }
        )

    df = pd.DataFrame(rows)
    meta_path = out_root / "metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"Wrote {len(df)} rows to {meta_path}")


if __name__ == "__main__":
    main()
