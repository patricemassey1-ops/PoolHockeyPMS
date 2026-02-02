from __future__ import annotations

import glob
import io
import json
import os
import zipfile
from datetime import datetime, timedelta, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


DATA_DIR = os.getenv("DATA_DIR", "data")

# GitHub Secrets (required)
CLIENT_ID = os.getenv("GDRIVE_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("GDRIVE_CLIENT_SECRET", "").strip()
REFRESH_TOKEN = os.getenv("GDRIVE_REFRESH_TOKEN", "").strip()
FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", "").strip()

# Optional overrides (Secrets) — if empty, policy file drives them.
OVR_TZ = os.getenv("BACKUP_TZ_OFFSET_HOURS", "").strip()
OVR_RET = os.getenv("BACKUP_RETENTION_DAYS", "").strip()
OVR_WIN = os.getenv("BACKUP_WINDOW_MINUTES", "").strip()


def load_policy() -> dict:
    """
    Reads your app policy if present: data/backup_policy.json
    so you can control retention days in-app.
    """
    p = os.path.join(DATA_DIR, "backup_policy.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}


def get_int(val: str, default: int) -> int:
    try:
        return int(str(val).strip())
    except Exception:
        return default


def drive_service():
    if not (CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN):
        raise RuntimeError("Missing secrets: GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET / GDRIVE_REFRESH_TOKEN")

    creds = Credentials(
        token=None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return build("drive", "v3", credentials=creds)


def zip_data_dir(include_patterns: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pat in include_patterns:
            for fp in glob.glob(os.path.join(DATA_DIR, pat)):
                if os.path.isfile(fp):
                    z.write(fp, arcname=os.path.basename(fp))
    return buf.getvalue()


def upload_zip(svc, folder_id: str, name: str, payload: bytes) -> None:
    media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/zip", resumable=True)
    meta = {"name": name, "parents": [folder_id]}
    svc.files().create(body=meta, media_body=media, fields="id,name", supportsAllDrives=True).execute()


def list_backups(svc, folder_id: str) -> list[dict]:
    q = f"'{folder_id}' in parents and trashed=false and name contains 'backup_'"
    out = []
    token = None
    while True:
        resp = svc.files().list(
            q=q,
            pageSize=200,
            fields="nextPageToken, files(id,name,createdTime,modifiedTime)",
            supportsAllDrives=True,
            pageToken=token,
        ).execute()
        out.extend(resp.get("files", []) or [])
        token = resp.get("nextPageToken")
        if not token:
            break
    return out


def delete_file(svc, file_id: str) -> None:
    svc.files().delete(fileId=file_id, supportsAllDrives=True).execute()


def cleanup_old(svc, folder_id: str, retention_days: int) -> tuple[int, int]:
    deleted = 0
    kept = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for f in list_backups(svc, folder_id):
        ct = f.get("createdTime") or f.get("modifiedTime") or ""
        try:
            dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except Exception:
            dt = None
        if dt and dt < cutoff:
            delete_file(svc, f.get("id", ""))
            deleted += 1
        else:
            kept += 1
    return deleted, kept


def period_due(now_local: datetime, period: str, window_minutes: int) -> bool:
    """
    True if now is within window after midnight/noon local time.
    """
    if period == "midnight":
        anchor = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        anchor = now_local.replace(hour=12, minute=0, second=0, microsecond=0)

    if now_local < anchor:
        return False
    if (now_local - anchor) > timedelta(minutes=window_minutes):
        return False
    return True


def load_state() -> dict:
    p = os.path.join(DATA_DIR, "backup_state.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, "backup_state.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    if not FOLDER_ID:
        raise RuntimeError("Missing secret: GDRIVE_FOLDER_ID")

    pol = load_policy()
    tz_offset = get_int(OVR_TZ or pol.get("tz_offset_hours", -5), -5)
    retention_days = get_int(OVR_RET or pol.get("retention_days", 30), 30)
    window_minutes = get_int(OVR_WIN or pol.get("window_minutes", 60), 60)

    include_patterns = pol.get("include_patterns") or ["*.csv", "*.json"]
    if not isinstance(include_patterns, list) or not include_patterns:
        include_patterns = ["*.csv", "*.json"]

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc + timedelta(hours=tz_offset)

    state = load_state()
    today = now_local.strftime("%Y-%m-%d")

    # Decide which period we are in
    do_midnight = period_due(now_local, "midnight", window_minutes) and state.get("last_midnight") != today
    do_noon = period_due(now_local, "noon", window_minutes) and state.get("last_noon") != today

    if not (do_midnight or do_noon):
        print(f"[SKIP] Not in window. now_local={now_local.isoformat()} tz={tz_offset} win={window_minutes}m")
        return

    svc = drive_service()

    label = "midnight" if do_midnight else "noon"
    season = (pol.get("season") or os.getenv("SEASON") or "2025-2026").strip().replace(" ", "")
    ts = now_utc.strftime("%Y%m%d_%H%M%S")
    name = f"backup_{season}_{label}_{ts}.zip"

    payload = zip_data_dir(include_patterns)
    upload_zip(svc, FOLDER_ID, name, payload)

    if do_midnight:
        state["last_midnight"] = today
    if do_noon:
        state["last_noon"] = today
    state["last_run_utc"] = now_utc.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)

    deleted, kept = cleanup_old(svc, FOLDER_ID, retention_days)
    print(f"[OK] Uploaded {name} | retention_days={retention_days} | deleted={deleted} kept={kept}")


if __name__ == "__main__":
    main()
