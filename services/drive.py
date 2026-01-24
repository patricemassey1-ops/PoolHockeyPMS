# services/drive.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st


def drive_ready() -> bool:
    try:
        _ = st.secrets.get("gdrive_folder_id", "")
        oauth = st.secrets.get("gdrive_oauth", {})
        return bool(_) and bool(oauth.get("client_id")) and bool(oauth.get("client_secret")) and bool(oauth.get("refresh_token"))
    except Exception:
        return False


def _get_folder_id() -> str:
    try:
        return str(st.secrets.get("gdrive_folder_id", "")).strip()
    except Exception:
        return ""


def _build_service():
    # Lazy imports (évite crash si libs manquent)
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    oauth = st.secrets.get("gdrive_oauth", {})
    creds = Credentials(
        token=None,
        refresh_token=oauth.get("refresh_token"),
        token_uri=oauth.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=oauth.get("client_id"),
        client_secret=oauth.get("client_secret"),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_in_folder(service, folder_id: str, name_contains: str = "", page_size: int = 200) -> List[Dict[str, Any]]:
    q_parts = [f"'{folder_id}' in parents", "trashed=false"]
    if name_contains:
        # contains is not directly supported; use name contains via fullText OR just filter client-side
        # We'll do client-side filter to be safe.
        pass

    q = " and ".join(q_parts)
    res = service.files().list(
        q=q,
        pageSize=page_size,
        fields="files(id,name,modifiedTime,size,mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", []) or []
    if name_contains:
        nc = name_contains.lower().strip()
        files = [f for f in files if nc in str(f.get("name", "")).lower()]
    return files


def drive_list_files(folder_id: Optional[str] = None, name_contains: str = "", limit: int = 200) -> List[Dict[str, Any]]:
    if not drive_ready():
        return []
    folder_id = str(folder_id or _get_folder_id()).strip()
    if not folder_id:
        return []
    try:
        service = _build_service()
        files = _list_in_folder(service, folder_id, name_contains=name_contains, page_size=min(1000, max(10, limit)))
        return files[:limit]
    except Exception:
        return []


def _find_file_by_name(service, folder_id: str, filename: str) -> Optional[Dict[str, Any]]:
    q = f"'{folder_id}' in parents and trashed=false and name='{filename}'"
    res = service.files().list(
        q=q,
        pageSize=10,
        fields="files(id,name,modifiedTime,size,mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", []) or []
    return files[0] if files else None


def drive_download_file(file_id: str, dest_path: str) -> Dict[str, Any]:
    if not drive_ready():
        return {"ok": False, "error": "Drive OAuth not ready."}
    if not file_id:
        return {"ok": False, "error": "Missing file_id."}
    if not dest_path:
        return {"ok": False, "error": "Missing dest_path."}

    try:
        from googleapiclient.http import MediaIoBaseDownload
        import io

        service = _build_service()
        req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

        fh = io.FileIO(dest_path, "wb")
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        return {"ok": True, "path": dest_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def drive_upload_file(folder_id: Optional[str], local_path: str, drive_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload en mode UPSERT:
    - si un fichier de même nom existe dans le folder -> update
    - sinon -> create
    """
    if not drive_ready():
        return {"ok": False, "error": "Drive OAuth not ready."}

    folder_id = str(folder_id or _get_folder_id()).strip()
    if not folder_id:
        return {"ok": False, "error": "Missing folder_id."}

    local_path = str(local_path or "").strip()
    if not local_path or not os.path.exists(local_path):
        return {"ok": False, "error": f"Local file not found: {local_path}"}

    filename = str(drive_name or os.path.basename(local_path)).strip()
    if not filename:
        return {"ok": False, "error": "Missing drive_name/filename."}

    try:
        from googleapiclient.http import MediaFileUpload

        service = _build_service()
        media = MediaFileUpload(local_path, resumable=True)

        existing = _find_file_by_name(service, folder_id, filename)
        if existing:
            file_id = existing["id"]
            updated = service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id,name,modifiedTime",
                supportsAllDrives=True,
            ).execute()
            return {"ok": True, "id": updated.get("id"), "name": updated.get("name"), "mode": "update"}

        created = service.files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=media,
            fields="id,name,modifiedTime",
            supportsAllDrives=True,
        ).execute()
        return {"ok": True, "id": created.get("id"), "name": created.get("name"), "mode": "create"}

    except Exception as e:
        return {"ok": False, "error": str(e)}
