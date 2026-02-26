from __future__ import annotations

import subprocess
from pathlib import Path


def launch_dashboard(port: int = 8501) -> None:
    app_path = Path(__file__).resolve().parents[3] / "dashboards" / "streamlit_app.py"
    subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(port)], check=False)
