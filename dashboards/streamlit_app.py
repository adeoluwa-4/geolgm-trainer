from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path("runs.db")


def load_runs():
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["run_id", "created_at", "status"])
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT run_id, created_at, status FROM runs ORDER BY created_at DESC", conn)
    conn.close()
    return df


def load_metrics(run_id: str):
    if not DB_PATH.exists():
        return pd.DataFrame()
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM metrics WHERE run_id = ? ORDER BY step ASC", conn, params=(run_id,)
    )
    conn.close()
    return df


def load_config(run_id: str):
    cfg_path = Path("runs") / run_id / "config_snapshot.yaml"
    if not cfg_path.exists():
        return {}
    import yaml

    return yaml.safe_load(cfg_path.read_text())


def load_artifacts(run_id: str):
    if not DB_PATH.exists():
        return pd.DataFrame()
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at DESC", conn, params=(run_id,)
    )
    conn.close()
    return df


def main():
    st.set_page_config(page_title="GeoLGM Trainer", layout="wide")
    st.title("GeoLGM Trainer Dashboard")

    runs = load_runs()

    tab_runs, tab_compare = st.tabs(["Runs", "Compare"])

    with tab_runs:
        st.subheader("Runs")
        st.dataframe(runs, use_container_width=True)

        if len(runs) > 0:
            run_id = st.selectbox("Select run", runs["run_id"].tolist())
            st.subheader("Run Detail")
            cfg = load_config(run_id)
            st.json(cfg)

            metrics = load_metrics(run_id)
            if not metrics.empty:
                st.line_chart(metrics[metrics["split"] == "train"]["loss"], height=200)
                st.line_chart(metrics[metrics["split"] == "val"]["acc"], height=200)
                st.line_chart(metrics[metrics["split"] == "val"]["images_per_sec"], height=200)

            artifacts = load_artifacts(run_id)
            if not artifacts.empty:
                st.subheader("Artifacts")
                st.dataframe(artifacts, use_container_width=True)

                for _, row in artifacts.iterrows():
                    path = row["path"]
                    if Path(path).exists() and path.endswith(".png"):
                        st.image(path)

    with tab_compare:
        st.subheader("Compare Runs")
        if len(runs) > 1:
            selected = st.multiselect("Select runs", runs["run_id"].tolist())
            if selected:
                for run_id in selected:
                    metrics = load_metrics(run_id)
                    if metrics.empty:
                        continue
                    val = metrics[metrics["split"] == "val"]
                    st.write(f"Run: {run_id}")
                    st.line_chart(val["acc"], height=150)
                    st.line_chart(val["images_per_sec"], height=150)


if __name__ == "__main__":
    main()
