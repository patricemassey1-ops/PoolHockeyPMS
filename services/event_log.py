# services/event_log.py
from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd


def event_log_path(data_dir: str, season: str) -> str:
    season = str(season or "").strip() or "season"
    data_dir = str(data_dir or "data")
    return os.path.join(data_dir, f"event_log_{season}.csv")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def append_event(
    *,
    data_dir: str,
    season: str,
    owner: str = "",
    event_type: str = "",
    summary: str = "",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Append un événement au journal CSV. Ne lève pas d'exception.
    """
    path = event_log_path(data_dir, season)
    _ensure_parent(path)

    row = {
        "timestamp": _now_iso(),
        "season": str(season or ""),
        "owner": str(owner or ""),
        "type": str(event_type or ""),
        "summary": str(summary or ""),
        "payload_json": json.dumps(payload or {}, ensure_ascii=False),
    }

    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=list(row.keys()))
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)
        return {"ok": True, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e), "path": path}


def read_events(data_dir: str, season: str) -> pd.DataFrame:
    path = event_log_path(data_dir, season)
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame(columns=["timestamp", "season", "owner", "type", "summary", "payload_json"])

