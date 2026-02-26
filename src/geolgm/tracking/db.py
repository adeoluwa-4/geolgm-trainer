from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        status TEXT,
        config_json TEXT,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        run_id TEXT,
        step INTEGER,
        split TEXT,
        loss REAL,
        acc REAL,
        images_per_sec REAL,
        dataloader_ms REAL,
        step_ms REAL,
        cpu_mem_mb REAL,
        gpu_mem_mb REAL,
        ts TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        run_id TEXT,
        type TEXT,
        path TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


def _connect(db_path: Path):
    return sqlite3.connect(db_path)


def init_db(db_path: Path) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    for stmt in SCHEMA:
        cur.execute(stmt)
    conn.commit()
    conn.close()


def insert_run(db_path: Path, run_id: str, config_json: str, status: str, notes: str | None):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO runs(run_id, status, config_json, notes) VALUES (?, ?, ?, ?)",
        (run_id, status, config_json, notes),
    )
    conn.commit()
    conn.close()


def update_run_status(db_path: Path, run_id: str, status: str):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
    conn.commit()
    conn.close()


def insert_metric(db_path: Path, metric: Dict):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO metrics(run_id, step, split, loss, acc, images_per_sec, dataloader_ms, step_ms, cpu_mem_mb, gpu_mem_mb)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metric.get("run_id"),
            metric.get("step"),
            metric.get("split"),
            metric.get("loss"),
            metric.get("acc"),
            metric.get("images_per_sec"),
            metric.get("dataloader_ms"),
            metric.get("step_ms"),
            metric.get("cpu_mem_mb"),
            metric.get("gpu_mem_mb"),
        ),
    )
    conn.commit()
    conn.close()


def insert_artifact(db_path: Path, run_id: str, type_: str, path: str):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO artifacts(run_id, type, path) VALUES (?, ?, ?)",
        (run_id, type_, path),
    )
    conn.commit()
    conn.close()


def query_runs(db_path: Path, limit: int = 10) -> Iterable[tuple]:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT run_id, created_at, status FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
