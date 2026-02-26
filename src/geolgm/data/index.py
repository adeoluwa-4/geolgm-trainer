from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def build_index(
    data_root: Path,
    out_path: Path,
    limit: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> None:
    meta_path = data_root / "metadata.csv"
    df = pd.read_csv(meta_path)

    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    if num_shards > 1:
        df = df.iloc[shard_id::num_shards]

    if limit is not None:
        df = df.head(limit)

    out_df = df[["image_path", "label", "split"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
