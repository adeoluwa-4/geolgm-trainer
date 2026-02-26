from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "image_path",
    "label",
    "lat",
    "lon",
    "timestamp",
    "heading",
    "device",
    "split",
]


def validate_dataset(data_root: Path) -> dict[str, Any]:
    meta_path = data_root / "metadata.csv"
    report: dict[str, Any] = {
        "meta_path": str(meta_path),
        "errors": [],
        "counts": {},
    }

    if not meta_path.exists():
        report["errors"].append("metadata.csv not found")
        return report

    df = pd.read_csv(meta_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["errors"].append(f"missing_columns: {missing}")
        return report

    report["counts"]["rows"] = int(len(df))
    report["counts"]["missing_files"] = 0
    report["counts"]["invalid_lat_lon"] = 0
    report["counts"]["invalid_label"] = 0

    for _, row in df.iterrows():
        img_path = data_root / row["image_path"]
        if not img_path.exists():
            report["counts"]["missing_files"] += 1

        lat = float(row["lat"])
        lon = float(row["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            report["counts"]["invalid_lat_lon"] += 1

        label = int(row["label"])
        if label < 0:
            report["counts"]["invalid_label"] += 1

    return report
