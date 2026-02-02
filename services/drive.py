# services/backup_drive.py
# -------------------------------------------------
# Auto backups to Google Drive (noon + midnight) with retention
# - Uses st.secrets["gdrive_oauth"] (same as Admin drive OAuth)
# - Writes policy in data/backup_policy.json
# - Writes state in data/backup_state.json (last run per period)
# Notes:
# - Streamlit can't run true cron in the background.
# - This runs when the app is opened/refreshed around those times.
# -------------------------------------------------

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

import streamlit as st

# Optional Google Drive libs
try:
    from google.oauth2.credentials import Credentials  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaIoBaseUpload  # type: ignore
except Exception:
    Credentials = None  # type: ignore
    build = None  # type: ignore
    MediaIoBaseUpload = None  # type: ignore


@dataclass
class BackupPolicy:
    enabled: bool = True
    retention_days: int = 30
    tz_offset_hours: int = -5  # Montreal default
    folder_id: str = ""        # if empty: use secrets gdrive_oauth.folder_id
    window_minutes: int = 45   # time window after 00:00 / 12:00
    include_patterns: Tuple[str, ...] = ("*.csv", "*.json")  # data files to include


def _data_dir(data_dir: str | None) -> str:
    return str(data_dir or os.getenv("DATA_DIR") or "data")


def _policy_path(data_dir: str) -> str:
    return os.path.join(data_dir, "backup_policy.json")


def _state_path(data_dir: str) -> str:
    return os.path.join(data_dir, "backup_state.json")


def load_policy(data_dir: str) -> BackupPolicy:
    data_dir = _data_dir(data_dir)
    p = _policy_path(data_dir)
    pol = BackupPolicy()
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            pol.enabled = bool(d.get("enabled", pol.enabled))
            pol.retention_days = int(d.get("retention_days", pol.retention_days))
            pol.tz_offset_hours = int(d.get("tz_offset_hours", pol.tz_offset_hours))
            pol.folder_id = str(d.get("folder_id", pol.folder_id) or "")
            pol.window_minutes = int(d.get("window_minutes", pol.window_minutes))
            inc = d.get("include_patterns")
            if isinstance(inc, list) and inc:
                pol.include_patterns = tuple(str(x) for x in inc)
    except Exception:
        pass
    # clamp
    pol.retention_days = max(1, min(pol.retention_days, 365))
    pol.window_minutes = max(10, min(pol.window_minutes, 180))
    pol.tz_offset_hours = max(-12, min(pol.tz_offset_hours, 14))
    return pol


def save_policy(data_dir: str, pol: BackupPolicy) -> Tuple[bool, str]:
    data_dir = _data_dir(data_dir)
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(_policy_path(data_dir), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "enabled": pol.enabled,
                    "retention_days": pol.retention_days,
                    "tz_offset_hours": pol.tz_offset_hours,
                    "folder_id": pol.folder_id,
                    "window_minutes": pol.window_minutes,
                    "include_patterns": list(pol.include_patterns),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return True, ""
    except Exception as e:
        return False, str(e)


def _load_state(data_dir: str) -> Dict[str, Any]:
    data_dir = _data_dir(data_dir)
    p = _state_path(data_dir)
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _save_state(data_dir: str, state: Dict[str, Any]) -> None:
    data_dir = _data_dir(data_dir)
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(_state_path(data_dir), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _drive_oauth_available() -> bool:
    try:
        sec = st.secrets
        g = sec.get("gdrive_oauth", {}) or {}
        return bool(g.get("client_id")) and bool(g.get("client_secret")) and bool(g.get("refresh_token"))
    except Exception:
        return False


def _drive_service():
    if Credentials is None or build is None:
        raise RuntimeError("Libs Google Drive manquantes (google-api-python-client / google-auth).")

    sec = st.secrets
    g = sec.get("gdrive_oauth", {}) or {}
    client_id = str(g.get("client_id") or "").strip()
    client_secret = str(g.get("client_secret") or "").strip()
    refresh_token = str(g.get("refresh_token") or "").strip()
    token_uri = str(g.get("token_uri") or "https://oauth2.googleapis.com/token").strip()
    scopes = g.get("scopes")  # optional list

    if not (client_id and client_secret and refresh_token):
        raise RuntimeError("Secrets OAuth Drive incomplets (client_id/client_secret/refresh_token).")

    if scopes:
        creds = Credentials(token=None, refresh_token=refresh_token, token_uri=token_uri, client_id=client_id, client_secret=client_secret, scopes=scopes)
    else:
        # Do NOT force scopes; use what the refresh token already has.
        creds = Credentials(token=None, refresh_token=refresh_token, token_uri=token_uri, client_id=client_id, client_secret=client_secret)
    return build("drive", "v3", credentials=creds)


def _drive_folder_id(policy_folder_id: str) -> str:
    try:
        sec = st.secrets
        g = sec.get("gdrive_oauth", {}) or {}
        fid = str(policy_folder_id or g.get("folder_id") or "").strip()
        return fid
    except Exception:
        return str(policy_folder_id or "").strip()


def _drive_upload_bytes(folder_id: str, name: str, data: bytes, mime: str = "application/zip") -> Tuple[bool, str]:
    try:
        svc = _drive_service()
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=True)
        meta = {"name": name, "parents": [folder_id]}
        svc.files().create(body=meta, media_body=media, fields="id,name", supportsAllDrives=True).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


def _drive_list_backups(folder_id: str) -> list[dict]:
    svc = _drive_service()
    q = f"'{folder_id}' in parents and trashed=false and name contains 'backup_'"
    res = []
    token = None
    while True:
        resp = svc.files().list(
            q=q,
            pageSize=200,
            fields="nextPageToken, files(id,name,createdTime,modifiedTime)",
            supportsAllDrives=True,
            pageToken=token,
        ).execute()
        res.extend(resp.get("files", []) or [])
        token = resp.get("nextPageToken")
        if not token:
            break
    return res


def _drive_delete_file(file_id: str) -> Tuple[bool, str]:
    try:
        svc = _drive_service()
        svc.files().delete(fileId=file_id, supportsAllDrives=True).execute()
        return True, ""
    except Exception as e:
        return False, str(e)


def _zip_data_dir(data_dir: str, include_patterns: Tuple[str, ...]) -> bytes:
    # Zip all matching files in data_dir root (not recursive)
    data_dir = _data_dir(data_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pat in include_patterns:
            # simple glob in root
            import glob
            for p in glob.glob(os.path.join(data_dir, pat)):
                if os.path.isfile(p):
                    z.write(p, arcname=os.path.basename(p))
    return buf.getvalue()


def cleanup_old_backups(data_dir: str, folder_id: str, retention_days: int) -> Tuple[int, int]:
    """
    Returns (deleted, kept).
    """
    deleted = 0
    kept = 0
    try:
        files = _drive_list_backups(folder_id)
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        for f in files:
            ct = f.get("createdTime") or f.get("modifiedTime") or ""
            try:
                # createdTime is RFC3339
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            except Exception:
                dt = None
            if dt and dt < cutoff:
                ok, _ = _drive_delete_file(f.get("id",""))
                if ok:
                    deleted += 1
            else:
                kept += 1
    except Exception:
        pass
    return deleted, kept


def _period_due(now_local: datetime, state: Dict[str, Any], period: str, window_minutes: int) -> bool:
    """
    period: 'midnight' or 'noon'
    """
    today = now_local.strftime("%Y-%m-%d")
    key = f"last_{period}"
    last = str(state.get(key, "")).strip()

    if period == "midnight":
        anchor = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        anchor = now_local.replace(hour=12, minute=0, second=0, microsecond=0)

    if now_local < anchor:
        return False
    if (now_local - anchor) > timedelta(minutes=window_minutes):
        return False
    if last == today:
        return False
    return True


def run_backup_now(data_dir: str, season: str, policy: BackupPolicy, label: str) -> Tuple[bool, str]:
    """
    Upload a zip backup of data/* to Drive folder.
    """
    if not policy.enabled:
        return False, "backups disabled"

    if not _drive_oauth_available():
        return False, "drive oauth not configured"

    folder_id = _drive_folder_id(policy.folder_id)
    if not folder_id:
        return False, "folder_id missing"

    # build name
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_season = str(season).replace("/", "-").replace("\\", "-").replace(" ", "")
    name = f"backup_{safe_season}_{label}_{ts}.zip"

    # zip
    payload = _zip_data_dir(data_dir, policy.include_patterns)

    ok, err = _drive_upload_bytes(folder_id, name, payload, mime="application/zip")
    if not ok:
        return False, err

    # cleanup
    cleanup_old_backups(data_dir, folder_id, policy.retention_days)

    return True, name


def scheduled_backup_tick(data_dir: str, season: str, owner: str, show_debug: bool = False) -> Tuple[bool, str]:
    """
    Call on each run (cheap). If due, performs backup.
    Safety: run backups only when owner == 'Whalers' (admin).
    """
    if str(owner) != "Whalers":
        return False, "owner not whalers"

    pol = load_policy(data_dir)
    if not pol.enabled:
        return False, "disabled"

    if not _drive_oauth_available():
        return False, "drive oauth missing"

    folder_id = _drive_folder_id(pol.folder_id)
    if not folder_id:
        return False, "folder id missing"

    state = _load_state(data_dir)

    # local time = utc + offset
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc + timedelta(hours=pol.tz_offset_hours)

    did = False
    msg = "no backup"

    if _period_due(now_local, state, "midnight", pol.window_minutes):
        ok, res = run_backup_now(data_dir, season, pol, label="midnight")
        if ok:
            state["last_midnight"] = now_local.strftime("%Y-%m-%d")
            state["last_run_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S")
            _save_state(data_dir, state)
            did = True
            msg = f"backup ok: {res}"
        else:
            msg = f"backup failed: {res}"

    if (not did) and _period_due(now_local, state, "noon", pol.window_minutes):
        ok, res = run_backup_now(data_dir, season, pol, label="noon")
        if ok:
            state["last_noon"] = now_local.strftime("%Y-%m-%d")
            state["last_run_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S")
            _save_state(data_dir, state)
            did = True
            msg = f"backup ok: {res}"
        else:
            msg = f"backup failed: {res}"

    if show_debug:
        state2 = _load_state(data_dir)
        msg += f" | state={state2}"

    return did, msg
