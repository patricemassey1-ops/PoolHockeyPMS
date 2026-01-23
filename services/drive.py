import os
import io
import streamlit as st
from typing import Optional

def resolve_drive_folder_id(default: str = "") -> str:
    fid = str(st.secrets.get("gdrive_folder_id", "") or "").strip()
    return fid or default

def drive_ready() -> bool:
    try:
        g = st.secrets.get("gdrive_oauth", None)
        if not isinstance(g, dict):
            return False
        if not g.get("client_id") or not g.get("client_secret") or not g.get("refresh_token"):
            return False
        return True
    except Exception:
        return False

@st.cache_resource(show_spinner=False)
def drive_service():
    if not drive_ready():
        return None
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except Exception:
        return None

    g = st.secrets["gdrive_oauth"]
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = Credentials(
        token=None,
        refresh_token=g.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=g.get("client_id"),
        client_secret=g.get("client_secret"),
        scopes=scopes,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def drive_list_files(folder_id: str, *, name_contains: str = "") -> list[dict]:
    svc = drive_service()
    if svc is None:
        return []
    folder_id = str(folder_id or "").strip()
    if not folder_id:
        return []

    q = [f"'{folder_id}' in parents", "trashed=false"]
    if name_contains:
        safe = name_contains.replace("'", "\\'")
        q.append(f"name contains '{safe}'")

    files = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=" and ".join(q),
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,size)",
            pageToken=page_token,
            pageSize=200,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    def _ts(x):
        return x.get("modifiedTime") or ""

    return sorted(files, key=_ts, reverse=True)

def drive_download_file(file_id: str, dest_path: str) -> dict:
    svc = drive_service()
    if svc is None:
        return {"ok": False, "error": "Drive not ready"}

    try:
        from googleapiclient.http import MediaIoBaseDownload
    except Exception as e:
        return {"ok": False, "error": f"Missing googleapiclient http: {e}"}

    try:
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
        fh = io.FileIO(dest_path, "wb")
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.close()
        return {"ok": True, "path": dest_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def drive_upload_file(folder_id: str, local_path: str, *, name: Optional[str] = None) -> dict:
    svc = drive_service()
    if svc is None:
        return {"ok": False, "error": "Drive not ready"}

    try:
        from googleapiclient.http import MediaFileUpload
    except Exception as e:
        return {"ok": False, "error": f"Missing googleapiclient http: {e}"}

    if not os.path.exists(local_path):
        return {"ok": False, "error": f"Local file not found: {local_path}"}

    folder_id = str(folder_id or "").strip()
    if not folder_id:
        return {"ok": False, "error": "Missing folder_id"}

    fname = name or os.path.basename(local_path)
    media = MediaFileUpload(local_path, resumable=True)
    body = {"name": fname, "parents": [folder_id]}

    try:
        created = svc.files().create(
            body=body,
            media_body=media,
            fields="id,name,webViewLink,createdTime",
            supportsAllDrives=True,
        ).execute()
        return {"ok": True, **created}
    except Exception as e:
        return {"ok": False, "error": str(e)}
