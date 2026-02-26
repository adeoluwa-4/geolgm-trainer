from pathlib import Path

import pandas as pd

from geolgm.data.validate import validate_dataset


def test_validate_lat_lon(tmp_path: Path):
    data_root = tmp_path
    (data_root / "images").mkdir()
    # dummy image path
    img_path = data_root / "images" / "img_0.png"
    img_path.write_bytes(b"dummy")

    df = pd.DataFrame(
        [
            {
                "image_path": "images/img_0.png",
                "label": 0,
                "lat": 999,
                "lon": 0,
                "timestamp": "2024-01-01T00:00:00",
                "heading": 0,
                "device": "iphone",
                "split": "train",
            }
        ]
    )
    df.to_csv(data_root / "metadata.csv", index=False)

    report = validate_dataset(data_root)
    assert report["counts"]["invalid_lat_lon"] == 1
