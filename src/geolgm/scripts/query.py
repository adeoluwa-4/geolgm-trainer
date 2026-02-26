from __future__ import annotations

from pathlib import Path

from ..tracking.db import query_runs as _query


def query_runs(db_path: Path, limit: int = 10):
    return _query(db_path, limit=limit)
