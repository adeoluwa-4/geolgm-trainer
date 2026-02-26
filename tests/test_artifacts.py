from pathlib import Path

from geolgm.tracking.db import init_db
from geolgm.tracking.artifacts import save_confusion_matrix


def test_artifact_register(tmp_path: Path):
    db = tmp_path / "runs.db"
    init_db(db)

    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    labels = [0, 1, 0]
    preds = [0, 1, 1]
    save_confusion_matrix(run_dir, labels, preds, db)
    assert (run_dir / "artifacts" / "confusion_matrix.png").exists()
