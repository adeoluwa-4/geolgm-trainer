from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, record: Dict):
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
