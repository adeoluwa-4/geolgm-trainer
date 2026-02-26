from pathlib import Path

from geolgm.tracking.db import init_db, insert_run, insert_metric, query_runs


def test_db_insert(tmp_path: Path):
    db = tmp_path / "runs.db"
    init_db(db)
    insert_run(db, "run-1", "{}", "running", None)
    insert_metric(
        db,
        {
            "run_id": "run-1",
            "step": 0,
            "split": "train",
            "loss": 1.0,
            "acc": None,
            "images_per_sec": 10.0,
            "dataloader_ms": 1.0,
            "step_ms": 2.0,
            "cpu_mem_mb": 100.0,
            "gpu_mem_mb": None,
        },
    )
    rows = list(query_runs(db))
    assert rows[0][0] == "run-1"
