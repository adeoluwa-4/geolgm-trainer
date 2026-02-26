from __future__ import annotations

import argparse
from pathlib import Path

from geolgm.tracking.db import query_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = query_runs(Path("runs.db"), limit=args.limit)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
