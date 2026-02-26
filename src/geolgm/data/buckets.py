from __future__ import annotations

from datetime import datetime


def region_bucket(lat: float, median_lat: float) -> str:
    return "north" if lat >= median_lat else "south"


def time_bucket(ts: str) -> str:
    # Expect ISO timestamp; simple split into day/night
    try:
        dt = datetime.fromisoformat(ts)
        return "day" if 6 <= dt.hour < 18 else "night"
    except Exception:
        return "unknown"
