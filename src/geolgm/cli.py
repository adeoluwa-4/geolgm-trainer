from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import load_config, snapshot_config
from .data.validate import validate_dataset
from .data.index import build_index
from .train.trainer import train_model
from .train.evaluate import evaluate_run
from .tracking.db import init_db, insert_run, update_run_status
from .tracking.logger import JsonlLogger
from .tracking.artifacts import ensure_artifacts_dir
from .services.api import launch_dashboard
from .scripts.query import query_runs


def _print_header(run_id: str, cfg: dict[str, Any]) -> None:
    print(f"run_id: {run_id}")
    print("config:")
    print(json.dumps(cfg, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="geolgm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--distributed", action="store_true")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--split", default="val")

    p_dash = sub.add_parser("dashboard")
    p_dash.add_argument("--port", type=int, default=8501)

    p_val = sub.add_parser("validate-data")
    p_val.add_argument("--data", required=True)
    p_val.add_argument("--out", required=True)

    p_index = sub.add_parser("build-index")
    p_index.add_argument("--data", required=True)
    p_index.add_argument("--out", required=True)
    p_index.add_argument("--limit", type=int, default=None)
    p_index.add_argument("--shuffle-seed", type=int, default=None)
    p_index.add_argument("--shard-id", type=int, default=0)
    p_index.add_argument("--num-shards", type=int, default=1)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--data", required=True)
    p_pre.add_argument("--out", required=True)
    p_pre.add_argument("--workers", type=int, default=4)

    p_query = sub.add_parser("query-runs")
    p_query.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "validate-data":
        report = validate_dataset(Path(args.data))
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"validation_report: {args.out}")
        return

    if args.cmd == "build-index":
        build_index(
            data_root=Path(args.data),
            out_path=Path(args.out),
            limit=args.limit,
            shuffle_seed=args.shuffle_seed,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        print(f"index_path: {args.out}")
        return

    if args.cmd == "preprocess":
        from .scripts.preprocess import preprocess_parallel

        preprocess_parallel(Path(args.data), Path(args.out), args.workers)
        return

    if args.cmd == "query-runs":
        rows = query_runs(Path("runs.db"), limit=args.limit)
        for row in rows:
            print(row)
        return

    if args.cmd == "dashboard":
        launch_dashboard(args.port)
        return

    if args.cmd == "eval":
        evaluate_run(args.run_id, split=args.split)
        return

    if args.cmd == "train":
        cfg_result = load_config(args.config)
        cfg = cfg_result.config
        run_id = cfg.run.name + "-" + str(os.getpid())

        run_dir = Path(cfg.run.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ensure_artifacts_dir(run_dir)
        snapshot_config(cfg_result.raw, run_dir / "config_snapshot.yaml")

        init_db(Path(cfg.logging.sqlite_path))
        insert_run(
            Path(cfg.logging.sqlite_path),
            run_id=run_id,
            config_json=json.dumps(cfg_result.raw),
            status="running",
            notes=cfg.run.notes,
        )

        _print_header(run_id, cfg_result.raw)
        jsonl_logger = JsonlLogger(run_dir / cfg.logging.jsonl_path)
        try:
            train_model(cfg, run_id, run_dir, jsonl_logger, resume=args.resume)
            update_run_status(Path(cfg.logging.sqlite_path), run_id, "completed")
        except Exception:
            update_run_status(Path(cfg.logging.sqlite_path), run_id, "failed")
            raise

        return


if __name__ == "__main__":
    main()
