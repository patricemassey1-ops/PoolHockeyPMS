# tabs/admin.py — NHL_ID Tools (SAFE) + Source Dropdown Fix + No "écran noir"
# -----------------------------------------------------------------------------
# Goals:
# 1) Never crash UI (global try/except in render)
# 2) Fix StreamlitDuplicateElementId (unique keys everywhere)
# 3) Source dropdown lists ALL CSVs in /data AND always includes nhl_search_players.csv + equipes_joueurs_*.csv
# 4) Associer NHL_ID works even if NHL_ID column is absent in target (it will be created)
# 5) Supports french column "Joueur" (and many others) as name column
# 6) SAFE MODE prevents catastrophic write (0 NHL_ID after operation)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os

DATA_DIR = os.getenv("DATA_DIR", "data")

def _to_str(x):
    try:
        if x is None:
            return ""
        s = str(x)
        return s.strip()
    except Exception:
        return ""


# Google Drive (optional)
try:
    from google.oauth2.credentials import Credentials  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaIoBaseDownload  # type: ignore
except Exception:
    Credentials = None  # type: ignore
    build = None  # type: ignore
    MediaIoBaseDownload = None  # type: ignore


def _drive_oauth_available() -> bool:
    try:
        sec = getattr(st, "secrets", {}) or {}
        # expected keys in secrets:
        # [gdrive_oauth] client_id, client_secret, refresh_token, token_uri (optional), folder_id (optional)
        return ("gdrive_oauth" in sec) and bool(sec["gdrive_oauth"].get("client_id")) and bool(sec["gdrive_oauth"].get("client_secret")) and bool(sec["gdrive_oauth"].get("refresh_token"))
    except Exception:
        return False

def _drive_get_folder_id_default() -> str:
    try:
        sec = getattr(st, "secrets", {}) or {}
        fid = (sec.get("gdrive_oauth", {}) or {}).get("folder_id") or ""
        fid = str(fid).strip()
        if fid:
            return fid
    except Exception:
        pass
    # fallback hardcoded default (user folder)
    return "1hIJovsHid2L1cY_wKM_sY-wVZKXAwrh1"

def _drive_service():
    """
    Returns Google Drive service using OAuth refresh token from st.secrets[gdrive_oauth].

    IMPORTANT (invalid_scope fix):
      - Ne force pas un scope différent de celui utilisé pour générer le refresh_token.
      - Si tu forces un scope qui ne correspond pas, Google renvoie: invalid_scope: Bad Request.
    Stratégie:
      1) Si secrets[gdrive_oauth].scopes est fourni -> on l'utilise
      2) Sinon -> on NE PASSE PAS scopes (on utilise les scopes du refresh_token)
      3) Si la liste échoue par permission, tu pourras ajuster scopes dans secrets.
    """
    if Credentials is None or build is None:
        raise RuntimeError("Libs Google Drive manquantes (google-api-python-client / google-auth).")

    sec = getattr(st, "secrets", {}) or {}
    g = sec.get("gdrive_oauth", {}) or {}

    client_id = str(g.get("client_id") or "").strip()
    client_secret = str(g.get("client_secret") or "").strip()
    refresh_token = str(g.get("refresh_token") or "").strip()
    token_uri = str(g.get("token_uri") or "https://oauth2.googleapis.com/token").strip()

    # optional override scopes in secrets: a list of urls or a comma-separated string
    scopes_val = g.get("scopes")
    scopes = None
    if isinstance(scopes_val, (list, tuple)) and scopes_val:
        scopes = [str(s).strip() for s in scopes_val if str(s).strip()]
    elif isinstance(scopes_val, str) and scopes_val.strip():
        scopes = [s.strip() for s in scopes_val.split(",") if s.strip()]

    if not (client_id and client_secret and refresh_token):
        raise RuntimeError("Secrets OAuth Drive incomplets (client_id/client_secret/refresh_token).")

    if scopes:
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=token_uri,
            client_id=client_id,
            client_secret=client_secret,
            scopes=scopes,
        )
    else:
        # Do NOT force scopes; use what the refresh token already has.
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=token_uri,
            client_id=client_id,
            client_secret=client_secret,
        )

    return build("drive", "v3", credentials=creds)


def _drive_list_csv_files(folder_id: str, page_size: int = 200) -> list[dict]:
    """
    List CSV files in a Drive folder. Returns list of dicts: id,name,modifiedTime,size,webViewLink (if available).
    """
    svc = _drive_service()
    q = f"'{folder_id}' in parents and trashed=false and mimeType='text/csv'"
    res = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=q,
            fields="nextPageToken, files(id,name,modifiedTime,size,webViewLink)",
            pageSize=page_size,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        res.extend(resp.get("files", []) or [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return res

def _score_drive_file(name: str) -> int:
    n = (name or "").lower()
    score = 0
    if "nhl_search" in n:
        score += 80
    if "nhl" in n and "id" in n:
        score += 60
    if "hockey.players" in n and "master" not in n:
        score += 50
    if "puckpedia" in n:
        score += 40
    if "master" in n:
        score += 15
    if "report" in n or "audit" in n:
        score += 10
    if n.endswith(".csv"):
        score += 5
    return score

def _drive_pick_auto(files: list[dict]) -> dict | None:
    """
    Pick best file automatically by (score, modifiedTime).
    """
    if not files:
        return None
    def parse_time(t):
        try:
            # drive uses RFC3339
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            return datetime(1970,1,1,tzinfo=timezone.utc)
    ranked = sorted(
        files,
        key=lambda f: (_score_drive_file(f.get("name","")), parse_time(f.get("modifiedTime",""))),
        reverse=True,
    )
    return ranked[0]

def _drive_download_file(file_id: str, out_path: str) -> tuple[bool, str]:
    """
    Download a Drive file to out_path.
    """
    try:
        svc = _drive_service()
        request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        return True, ""
    except Exception as e:
        return False, str(e)

import glob
import difflib
import re
import io
import json
import time
import tempfile
from datetime import datetime, timezone
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st


def _read_file_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""



def _quick_nhl_id_stats(csv_path: str) -> dict:
    out = {"path": csv_path, "rows": 0, "with_id": 0, "missing": 0, "ok": False, "error": ""}
    if not csv_path or not os.path.exists(csv_path):
        out["error"] = "introuvable"
        return out
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        out["rows"] = int(len(df))
        if "NHL_ID" not in df.columns:
            out["with_id"] = 0
            out["missing"] = int(out["rows"])
        else:
            s = df["NHL_ID"].astype(str).str.strip()
            out["with_id"] = int(s.ne("").sum())
            out["missing"] = int(out["rows"] - out["with_id"])
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


WKEY = "admin_nhlid_"  # widget key prefix (unique)


# =========================
# I/O
# =========================
def load_csv(path: str) -> Tuple[pd.DataFrame, str | None]:
    try:
        if not path:
            return pd.DataFrame(), "Chemin CSV vide."
        if not os.path.exists(path):
            return pd.DataFrame(), f"Fichier introuvable: {path}"
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Erreur lecture CSV: {type(e).__name__}: {e}"


def save_csv(df: pd.DataFrame, path: str, *, safe_mode: bool = True, allow_zero: bool = False) -> str | None:
    """SAFE MODE: refuse to write if NHL_ID exists but would be 0% filled (unless allow_zero)."""
    try:
        if safe_mode and not allow_zero:
            id_col = _resolve_nhl_id_col(df)
            if id_col and id_col in df.columns:
                s = pd.to_numeric(df[id_col], errors="coerce")
                if int(s.notna().sum()) == 0:
                    return "SAFE MODE: Refus d'écrire — NHL_ID serait 0/100% (colonne vide)."
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        return None
    except Exception as e:
        return f"Erreur écriture CSV: {type(e).__name__}: {e}"


# =========================
# Column resolution
# =========================

# =========================
# Master Builder helpers
# =========================
def _atomic_write_df(df: pd.DataFrame, out_path: str) -> Tuple[bool, str | None]:
    """Atomic CSV write on the SAME filesystem as out_path (avoids Errno 18 cross-device link)."""
    try:
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            suffix=".csv",
            dir=out_dir,
            encoding="utf-8",
            newline="",
        ) as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _list_csv_files(data_dir: str) -> list[str]:
    try:
        paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        return [p for p in paths if os.path.isfile(p)]
    except Exception:
        return []

def _auto_detect_nhl_id_sources(data_dir: str) -> list[str]:
    """
    Heuristic: prefer files with both 'nhl' and 'id' in filename, then anything with 'nhl' or 'id'.
    """
    files = _list_csv_files(data_dir)
    scored = []
    for p in files:
        name = os.path.basename(p).lower()
        score = 0
        if "nhl" in name and "id" in name:
            score += 50
        if "nhl" in name:
            score += 10
        if "id" in name:
            score += 8
        if "source" in name:
            score += 4
        if "cache" in name:
            score -= 2
        if "report" in name:
            score -= 5
        if "master" in name:
            score -= 10
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for s, p in scored if s >= 10]
    return top if top else [p for _, p in scored]

def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols.get(cand.lower())
        if c:
            return c
    return None

def _normalize_player_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).fillna("").str.strip().str.lower()
    s2 = s2.str.replace(r"\s+", " ", regex=True)
    s2 = s2.str.replace(r"[^a-z0-9 \-'.]", "", regex=True)
    return s2

def _apply_nhl_id_source_to_players(players_path: str, source_path: str) -> tuple[bool, str]:
    """
    Merge NHL_ID from a detected source CSV into hockey.players.csv (by Player name).
    Returns (ok, message).
    """
    if not os.path.exists(players_path):
        return False, f"players file introuvable: {players_path}"
    if not source_path or not os.path.exists(source_path):
        return False, f"source introuvable: {source_path}"

    try:
        players = pd.read_csv(players_path, low_memory=False)
        src = pd.read_csv(source_path, low_memory=False)
    except Exception as e:
        return False, f"lecture CSV impossible: {e}"

    p_name = _detect_col(players, ["Player", "Skaters", "Name", "Full Name"])
    s_name = _detect_col(src, ["Player", "Skaters", "Name", "Full Name"])
    s_id = _detect_col(src, ["NHL_ID", "nhl_id", "NHL ID", "playerId", "player_id", "nhlPlayerId", "id"])
    if s_id is None:
        return False, "colonne NHL_ID introuvable dans la source (ex: NHL_ID / playerId)."
    if p_name is None or s_name is None:
        return False, "impossible de matcher: colonne Player manquante dans players ou la source."

    if "NHL_ID" not in players.columns:
        players["NHL_ID"] = ""

    before_filled = int(players["NHL_ID"].astype(str).str.strip().ne("").sum())

    src2 = src.copy()
    src2["__sname__"] = _normalize_player_series(src2[s_name])
    src2["__sid__"] = src2[s_id].astype(str).str.strip()
    src2 = src2[src2["__sid__"].ne("") & src2["__sid__"].str.lower().ne("nan")]
    src_map = src2.drop_duplicates(subset=["__sname__"])[["__sname__", "__sid__"]]

    merged = players.copy()
    merged["__pname__"] = _normalize_player_series(merged[p_name])
    merged = merged.merge(src_map, how="left", left_on="__pname__", right_on="__sname__")

    blank = merged["NHL_ID"].astype(str).str.strip().eq("")
    got = merged["__sid__"].astype(str).str.strip()
    okmask = got.ne("") & got.str.lower().ne("nan")
    merged.loc[blank & okmask, "NHL_ID"] = got[blank & okmask]

    merged = merged.drop(columns=[c for c in ["__pname__", "__sname__", "__sid__"] if c in merged.columns], errors="ignore")

    after_filled = int(merged["NHL_ID"].astype(str).str.strip().ne("").sum())
    added = after_filled - before_filled

    ok, err = _atomic_write_df(merged, players_path)
    if not ok:
        return False, f"écriture players impossible: {err}"

    return True, f"NHL_ID appliqués: +{added} (total NHL_ID non vides: {after_filled})"



def _nhl_id_coverage_report(target_path: str, source_path: str) -> dict:
    """
    Compare target players DB vs NHL search source and report NHL_ID coverage + missing names.
    """
    out = {
        "target_path": target_path,
        "source_path": source_path,
        "target_rows": 0,
        "target_with_id": 0,
        "target_missing_id": 0,
        "source_rows": 0,
        "matched_by_name": 0,
        "missing_names": [],
        "ambiguous_names": [],
    }

    if not os.path.exists(target_path) or not os.path.exists(source_path):
        out["error"] = "target/source introuvable"
        return out

    try:
        tdf = pd.read_csv(target_path, low_memory=False)
        sdf = pd.read_csv(source_path, low_memory=False)
    except Exception as e:
        out["error"] = str(e)
        return out

    t_name = _detect_col(tdf, ["Player", "Skaters", "Name", "Full Name"])
    if t_name is None:
        out["error"] = "colonne Player introuvable dans target"
        return out

    if "NHL_ID" not in tdf.columns:
        tdf["NHL_ID"] = ""

    s_name = _detect_col(sdf, ["Player", "Skaters", "Name", "Full Name"])
    s_id = _detect_col(sdf, ["NHL_ID", "nhl_id", "NHL ID", "playerId", "player_id", "id"])
    if s_name is None or s_id is None:
        out["error"] = "colonne Player/NHL_ID introuvable dans source"
        return out

    tdf["__pname__"] = _normalize_player_series(tdf[t_name])
    sdf["__sname__"] = _normalize_player_series(sdf[s_name])
    sdf["__sid__"] = sdf[s_id].astype(str).str.strip()

    sdf = sdf[sdf["__sid__"].ne("") & sdf["__sid__"].str.lower().ne("nan")]

    # detect ambiguous names in source (same name -> multiple NHL_ID)
    amb = sdf.groupby("__sname__")["__sid__"].nunique()
    ambiguous = set(amb[amb > 1].index.tolist())

    # mapping (first occurrence)
    src_map = sdf.drop_duplicates(subset=["__sname__"])[["__sname__", "__sid__"]]

    # coverage
    out["target_rows"] = int(len(tdf))
    out["source_rows"] = int(len(sdf))
    out["target_with_id"] = int(tdf["NHL_ID"].astype(str).str.strip().ne("").sum())
    out["target_missing_id"] = int(out["target_rows"] - out["target_with_id"])

    missing_names = []
    matched_by_name = 0
    for name, nhl_id in zip(tdf["__pname__"].tolist(), tdf["NHL_ID"].astype(str).str.strip().tolist()):
        if nhl_id:
            continue
        if name in ambiguous:
            continue
        # match exists?
        if (src_map["__sname__"] == name).any():
            matched_by_name += 1
        else:
            missing_names.append(name)

    out["matched_by_name"] = int(matched_by_name)
    out["missing_names"] = missing_names[:200]  # keep UI light
    out["ambiguous_names"] = [n for n in tdf["__pname__"].tolist() if n in ambiguous][:200]
    return out


def _fill_missing_nhl_ids_from_source(target_path: str, source_path: str) -> tuple[bool, str, dict]:
    """
    Fill missing NHL_ID in target_path by name from source_path.
    Skips ambiguous names (where source has >1 NHL_ID for same name).
    Writes target_path atomically. Returns (ok, message, stats).
    """
    stats = {"filled": 0, "skipped_ambiguous": 0, "still_missing": 0}
    if not os.path.exists(target_path) or not os.path.exists(source_path):
        return False, "target/source introuvable", stats

    try:
        tdf = pd.read_csv(target_path, low_memory=False)
        sdf = pd.read_csv(source_path, low_memory=False)
    except Exception as e:
        return False, f"lecture CSV impossible: {e}", stats

    t_name = _detect_col(tdf, ["Player", "Skaters", "Name", "Full Name"])
    if t_name is None:
        return False, "colonne Player introuvable dans target", stats

    if "NHL_ID" not in tdf.columns:
        tdf["NHL_ID"] = ""

    s_name = _detect_col(sdf, ["Player", "Skaters", "Name", "Full Name"])
    s_id = _detect_col(sdf, ["NHL_ID", "nhl_id", "NHL ID", "playerId", "player_id", "id"])
    if s_name is None or s_id is None:
        return False, "colonne Player/NHL_ID introuvable dans source", stats

    tdf["__pname__"] = _normalize_player_series(tdf[t_name])
    sdf["__sname__"] = _normalize_player_series(sdf[s_name])
    sdf["__sid__"] = sdf[s_id].astype(str).str.strip()
    sdf = sdf[sdf["__sid__"].ne("") & sdf["__sid__"].str.lower().ne("nan")]

    amb = sdf.groupby("__sname__")["__sid__"].nunique()
    ambiguous = set(amb[amb > 1].index.tolist())
    src_map = sdf.drop_duplicates(subset=["__sname__"])[["__sname__", "__sid__"]]

    # merge
    merged = tdf.merge(src_map, how="left", left_on="__pname__", right_on="__sname__")

    blank = merged["NHL_ID"].astype(str).str.strip().eq("")
    got = merged["__sid__"].astype(str).str.strip()
    # skip ambiguous
    is_amb = merged["__pname__"].isin(list(ambiguous))
    fill_mask = blank & (~is_amb) & got.ne("") & got.str.lower().ne("nan")

    stats["filled"] = int(fill_mask.sum())
    stats["skipped_ambiguous"] = int((blank & is_amb).sum())

    merged.loc[fill_mask, "NHL_ID"] = got[fill_mask]

    # cleanup
    merged = merged.drop(columns=[c for c in ["__pname__", "__sname__", "__sid__"] if c in merged.columns], errors="ignore")
    stats["still_missing"] = int(merged["NHL_ID"].astype(str).str.strip().eq("").sum())

    ok, err = _atomic_write_df(merged, target_path)
    if not ok:
        return False, f"écriture impossible: {err}", stats

    return True, f"✅ NHL_ID remplis: +{stats['filled']} (ambigus ignorés: {stats['skipped_ambiguous']}, encore manquants: {stats['still_missing']})", stats



def _preview_fill_missing_nhl_ids(target_path: str, source_path: str) -> tuple[bool, str, dict]:
    """
    Dry-run: compute how many NHL_ID would be filled (blank -> value) from source into target.
    Does NOT write anything.
    Returns (ok, message, stats).
    """
    stats = {"rows": 0, "would_fill": 0, "skipped_ambiguous": 0, "still_missing": 0}
    if not os.path.exists(target_path) or not os.path.exists(source_path):
        return False, "target/source introuvable", stats
    try:
        tdf = pd.read_csv(target_path, low_memory=False)
        sdf = pd.read_csv(source_path, low_memory=False)
    except Exception as e:
        return False, f"lecture CSV impossible: {e}", stats

    t_name = _detect_col(tdf, ["Player", "Skaters", "Name", "Full Name"])
    if t_name is None:
        return False, "colonne Player introuvable dans target", stats

    if "NHL_ID" not in tdf.columns:
        tdf["NHL_ID"] = ""

    s_name = _detect_col(sdf, ["Player", "Skaters", "Name", "Full Name"])
    s_id = _detect_col(sdf, ["NHL_ID", "nhl_id", "NHL ID", "playerId", "player_id", "nhlPlayerId", "id"])
    if s_name is None or s_id is None:
        return False, "colonne Player/NHL_ID introuvable dans source", stats

    tdf["__pname__"] = _normalize_player_series(tdf[t_name])
    sdf["__sname__"] = _normalize_player_series(sdf[s_name])
    sdf["__sid__"] = sdf[s_id].astype(str).str.strip()
    sdf = sdf[sdf["__sid__"].ne("") & sdf["__sid__"].str.lower().ne("nan")]

    amb = sdf.groupby("__sname__")["__sid__"].nunique()
    ambiguous = set(amb[amb > 1].index.tolist())
    src_map = sdf.drop_duplicates(subset=["__sname__"])[["__sname__", "__sid__"]]

    merged = tdf.merge(src_map, how="left", left_on="__pname__", right_on="__sname__")

    blank = merged["NHL_ID"].astype(str).str.strip().eq("")
    got = merged["__sid__"].astype(str).str.strip()
    is_amb = merged["__pname__"].isin(list(ambiguous))
    would_fill_mask = blank & (~is_amb) & got.ne("") & got.str.lower().ne("nan")

    stats["rows"] = int(len(merged))
    stats["would_fill"] = int(would_fill_mask.sum())
    stats["skipped_ambiguous"] = int((blank & is_amb).sum())
    stats["still_missing"] = int((blank & (~would_fill_mask)).sum())

    return True, f"would_fill={stats['would_fill']} (ambigus={stats['skipped_ambiguous']})", stats


def _build_fill_preview_table(target_path: str, source_path: str, limit: int = 50) -> pd.DataFrame:
    """
    Build a small preview table (top N) of players who would receive a NHL_ID (blank -> value).
    """
    if (not target_path) or (not source_path) or (not os.path.exists(target_path)) or (not os.path.exists(source_path)):
        return pd.DataFrame(columns=["Player", "NHL_ID_before", "NHL_ID_after", "reason"])

    try:
        tdf = pd.read_csv(target_path, low_memory=False)
        sdf = pd.read_csv(source_path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=["Player", "NHL_ID_before", "NHL_ID_after", "reason"])

    t_name = _detect_col(tdf, ["Player", "Skaters", "Name", "Full Name"])
    s_name = _detect_col(sdf, ["Player", "Skaters", "Name", "Full Name"])
    s_id = _detect_col(sdf, ["NHL_ID", "nhl_id", "NHL ID", "playerId", "player_id", "nhlPlayerId", "id"])
    if t_name is None or s_name is None or s_id is None:
        return pd.DataFrame(columns=["Player", "NHL_ID_before", "NHL_ID_after", "reason"])

    if "NHL_ID" not in tdf.columns:
        tdf["NHL_ID"] = ""

    tdf["__pname__"] = _normalize_player_series(tdf[t_name])
    sdf["__sname__"] = _normalize_player_series(sdf[s_name])
    sdf["__sid__"] = sdf[s_id].astype(str).str.strip()
    sdf = sdf[sdf["__sid__"].ne("") & sdf["__sid__"].str.lower().ne("nan")]

    amb = sdf.groupby("__sname__")["__sid__"].nunique()
    ambiguous = set(amb[amb > 1].index.tolist())
    src_map = sdf.drop_duplicates(subset=["__sname__"])[["__sname__", "__sid__"]]

    merged = tdf[[t_name, "NHL_ID", "__pname__"]].copy()
    merged = merged.merge(src_map, how="left", left_on="__pname__", right_on="__sname__")

    before = merged["NHL_ID"].astype(str).str.strip()
    after = merged["__sid__"].astype(str).str.strip()
    is_blank = before.eq("")
    is_amb = merged["__pname__"].isin(list(ambiguous))
    can_fill = is_blank & (~is_amb) & after.ne("") & after.str.lower().ne("nan")

    out = pd.DataFrame({
        "Player": merged[t_name].astype(str),
        "NHL_ID_before": before,
        "NHL_ID_after": after.where(can_fill, ""),
        "reason": ["fill" if cf else ("ambiguous" if (ib and ia) else ("no_match" if ib else "already_has_id"))
                   for cf, ib, ia in zip(can_fill.tolist(), is_blank.tolist(), is_amb.tolist())]
    })
    out = out[out["reason"].eq("fill")].head(int(limit))
    return out


def _audit_nhl_id_suspects(players_path: str, master_path: str, nhl_search_path: str, after_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Audit NHL_ID suspect (idiot-proof):
      - même NHL_ID attribué à plusieurs noms (duplicate id)
      - même nom attribué à plusieurs NHL_ID (duplicate name)
      - mismatch Team/Position/Jersey entre master et nhl_search (si dispo)
    Retourne un DataFrame (toutes anomalies) avec une colonne 'issue'.
    """
    def _safe_read(path: str) -> pd.DataFrame:
        try:
            if path and os.path.exists(path):
                return pd.read_csv(path, low_memory=False)
        except Exception:
            pass
        return pd.DataFrame()

    dfp = _safe_read(players_path)
    dfm = after_df if isinstance(after_df, pd.DataFrame) and (not after_df.empty) else _safe_read(master_path)
    dfs = _safe_read(nhl_search_path)

    # pick best table for checking attributes (prefer master)
    df = dfm if not dfm.empty else dfp
    if df.empty:
        return pd.DataFrame(columns=["issue","NHL_ID","Player","detail"])

    # standardize cols
    name_col = _detect_col(df, ["Player","Skaters","Name","Full Name"]) or "Player"
    if name_col not in df.columns:
        df[name_col] = ""
    if "NHL_ID" not in df.columns:
        df["NHL_ID"] = ""

    out_rows = []

    # 1) Duplicate NHL_ID -> multiple players
    tmp = df[[name_col, "NHL_ID"]].copy()
    tmp["NHL_ID"] = tmp["NHL_ID"].astype(str).str.strip()
    tmp[name_col] = tmp[name_col].astype(str).str.strip()
    tmp = tmp[tmp["NHL_ID"].ne("") & tmp["NHL_ID"].str.lower().ne("nan")]
    if not tmp.empty:
        g = tmp.groupby("NHL_ID")[name_col].nunique().reset_index(name="unique_players")
        dup_ids = g[g["unique_players"] > 1]["NHL_ID"].tolist()
        if dup_ids:
            for nhl_id in dup_ids[:5000]:
                players = tmp[tmp["NHL_ID"] == nhl_id][name_col].dropna().astype(str).tolist()
                players = sorted(list(dict.fromkeys([p for p in players if p.strip()])))[:25]
                out_rows.append({
                    "issue": "duplicate_nhl_id",
                    "NHL_ID": nhl_id,
                    "Player": " | ".join(players),
                    "detail": f"{len(players)}+ noms pour le même NHL_ID"
                })

    # 2) Duplicate name -> multiple NHL_ID
    tmp2 = df[[name_col, "NHL_ID"]].copy()
    tmp2["NHL_ID"] = tmp2["NHL_ID"].astype(str).str.strip()
    tmp2[name_col] = tmp2[name_col].astype(str).str.strip()
    tmp2 = tmp2[tmp2[name_col].ne("") & tmp2[name_col].str.lower().ne("nan")]
    if not tmp2.empty:
        g2 = tmp2.groupby(name_col)["NHL_ID"].nunique().reset_index(name="unique_ids")
        dup_names = g2[g2["unique_ids"] > 1][name_col].tolist()
        if dup_names:
            for nm in dup_names[:5000]:
                ids = tmp2[tmp2[name_col] == nm]["NHL_ID"].dropna().astype(str).tolist()
                ids = sorted(list(dict.fromkeys([i for i in ids if i.strip()])))[:25]
                out_rows.append({
                    "issue": "duplicate_player_name",
                    "NHL_ID": " | ".join(ids),
                    "Player": nm,
                    "detail": f"{len(ids)}+ NHL_ID pour le même nom"
                })

    # 3) Mismatch Team/Position/Jersey between master and nhl_search (if available)
    if not dfm.empty and not dfs.empty:
        # detect cols
        team_m = _detect_col(dfm, ["Team","TeamAbbrev","teamAbbrev","NHL Team"])
        pos_m  = _detect_col(dfm, ["Position","Pos"])
        jer_m  = _detect_col(dfm, ["Jersey#","Jersey","SweaterNumber"])
        dob_m  = _detect_col(dfm, ["DOB","BirthDate","birthDate"])

        team_s = _detect_col(dfs, ["Team","team","teamAbbrev"])
        pos_s  = _detect_col(dfs, ["Position","Pos","position"])
        jer_s  = _detect_col(dfs, ["Jersey#","Jersey","sweaterNumber"])
        dob_s  = _detect_col(dfs, ["DOB","BirthDate","birthDate"])

        # only if we have NHL_ID
        if "NHL_ID" in dfm.columns and "NHL_ID" in dfs.columns:
            msub = dfm.copy()
            ssub = dfs.copy()
            msub["NHL_ID"] = msub["NHL_ID"].astype(str).str.strip()
            ssub["NHL_ID"] = ssub["NHL_ID"].astype(str).str.strip()
            msub = msub[msub["NHL_ID"].ne("") & msub["NHL_ID"].str.lower().ne("nan")]
            ssub = ssub[ssub["NHL_ID"].ne("") & ssub["NHL_ID"].str.lower().ne("nan")]

            join_cols = ["NHL_ID"]
            keep_m = ["NHL_ID", name_col]
            keep_s = ["NHL_ID"]
            if team_m: keep_m.append(team_m)
            if pos_m: keep_m.append(pos_m)
            if jer_m: keep_m.append(jer_m)
            if dob_m: keep_m.append(dob_m)
            if team_s: keep_s.append(team_s)
            if pos_s: keep_s.append(pos_s)
            if jer_s: keep_s.append(jer_s)
            if dob_s: keep_s.append(dob_s)

            msub = msub[keep_m].copy()
            ssub = ssub[keep_s].copy()

            j = msub.merge(ssub, how="inner", on="NHL_ID", suffixes=("_m","_s"))

            def _cmp(a, b):
                aa = a.astype(str).str.strip()
                bb = b.astype(str).str.strip()
                ok = aa.ne("") & aa.str.lower().ne("nan") & bb.ne("") & bb.str.lower().ne("nan")
                return ok & (aa != bb)

            # Team mismatch
            if team_m and team_s:
                mism = j[_cmp(j[f"{team_m}_m"], j[f"{team_s}_s"])].head(5000)
                for _, r in mism.iterrows():
                    out_rows.append({
                        "issue": "mismatch_team",
                        "NHL_ID": r["NHL_ID"],
                        "Player": r.get(name_col, ""),
                        "detail": f"master={r.get(team_m + '_m','')} vs search={r.get(team_s + '_s','')}"
                    })
            # Position mismatch
            if pos_m and pos_s:
                mism = j[_cmp(j[f"{pos_m}_m"], j[f"{pos_s}_s"])].head(5000)
                for _, r in mism.iterrows():
                    out_rows.append({
                        "issue": "mismatch_position",
                        "NHL_ID": r["NHL_ID"],
                        "Player": r.get(name_col, ""),
                        "detail": f"master={r.get(pos_m + '_m','')} vs search={r.get(pos_s + '_s','')}"
                    })
            # Jersey mismatch
            if jer_m and jer_s:
                mism = j[_cmp(j[f"{jer_m}_m"], j[f"{jer_s}_s"])].head(5000)
                for _, r in mism.iterrows():
                    out_rows.append({
                        "issue": "mismatch_jersey",
                        "NHL_ID": r["NHL_ID"],
                        "Player": r.get(name_col, ""),
                        "detail": f"master={r.get(jer_m + '_m','')} vs search={r.get(jer_s + '_s','')}"
                    })
            # DOB mismatch (rare but strong signal)
            if dob_m and dob_s:
                mism = j[_cmp(j[f"{dob_m}_m"], j[f"{dob_s}_s"])].head(5000)
                for _, r in mism.iterrows():
                    out_rows.append({
                        "issue": "mismatch_dob",
                        "NHL_ID": r["NHL_ID"],
                        "Player": r.get(name_col, ""),
                        "detail": f"master={r.get(dob_m + '_m','')} vs search={r.get(dob_s + '_s','')}"
                    })

    out = pd.DataFrame(out_rows) if out_rows else pd.DataFrame(columns=["issue","NHL_ID","Player","detail"])
    return out


def _norm_person_name(s: str) -> str:
    s = _to_str(s).lower()
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-'.]", "", s)
    return s.strip()

def _name_variants(s: str) -> list[str]:
    raw = _to_str(s)
    n = _norm_person_name(raw)
    out = {n}
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) >= 2:
            last = _norm_person_name(parts[0])
            first = _norm_person_name(parts[1])
            if first and last:
                out.add(f"{first} {last}".strip())
                out.add(f"{last} {first}".strip())
    return [x for x in out if x]

def _similarity(a: str, b: str) -> float:
    a = _norm_person_name(a)
    b = _norm_person_name(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def _autofix_duplicate_nhl_ids(players_path: str, nhl_search_path: str) -> tuple[bool, str, pd.DataFrame, pd.DataFrame]:
    """
    Auto-fix duplicate NHL_ID in hockey.players.csv:
      - For each NHL_ID assigned to multiple different player names,
        keep the row whose name best matches the official name from nhl_search_players.csv (by NHL_ID),
        and CLEAR NHL_ID for the others.
    Returns: (ok, message, fixed_players_df, fix_report_df)
    """
    if not os.path.exists(players_path):
        return False, f"players introuvable: {players_path}", pd.DataFrame(), pd.DataFrame()
    if not os.path.exists(nhl_search_path):
        return False, f"source introuvable: {nhl_search_path}", pd.DataFrame(), pd.DataFrame()

    try:
        players = pd.read_csv(players_path, low_memory=False)
        src = pd.read_csv(nhl_search_path, low_memory=False)
    except Exception as e:
        return False, f"lecture CSV impossible: {e}", pd.DataFrame(), pd.DataFrame()

    p_name = _detect_col(players, ["Player", "Skaters", "Name", "Full Name"]) or "Player"
    if p_name not in players.columns:
        players[p_name] = ""
    if "NHL_ID" not in players.columns:
        return False, "players n'a pas de colonne NHL_ID", pd.DataFrame(), pd.DataFrame()

    s_name = _detect_col(src, ["Player", "Skaters", "Name", "Full Name"]) or "Player"
    s_id = _detect_col(src, ["NHL_ID", "nhl_id", "NHL ID", "playerId", "player_id", "nhlPlayerId", "id"])
    if s_id is None:
        return False, "source n'a pas de colonne NHL_ID", pd.DataFrame(), pd.DataFrame()
    if s_name not in src.columns:
        src[s_name] = ""

    src_map = src[[s_id, s_name]].copy()
    src_map[s_id] = src_map[s_id].astype(str).str.strip()
    src_map[s_name] = src_map[s_name].astype(str).str.strip()
    src_map = src_map[src_map[s_id].ne("") & src_map[s_id].str.lower().ne("nan")]
    id2name = dict(zip(src_map[s_id].tolist(), src_map[s_name].tolist()))

    df = players.copy()
    df["NHL_ID"] = df["NHL_ID"].astype(str).str.strip()
    df[p_name] = df[p_name].astype(str).str.strip()

    g = df[df["NHL_ID"].ne("") & df["NHL_ID"].str.lower().ne("nan")].groupby("NHL_ID")[p_name].nunique()
    dup_ids = g[g > 1].index.tolist()

    if not dup_ids:
        return True, "Aucun NHL_ID dupliqué détecté.", df, pd.DataFrame(columns=["NHL_ID","official_name","kept_player","kept_score","cleared_players","cleared_count"])

    report_rows = []
    cleared_total = 0

    for nhl_id in dup_ids:
        rows = df.index[df["NHL_ID"] == nhl_id].tolist()
        names = [df.at[i, p_name] for i in rows]
        official = id2name.get(str(nhl_id), "")

        best_i = rows[0] if rows else None
        best_score = -1.0
        for i, nm in zip(rows, names):
            if official:
                score = max([_similarity(official, v) for v in _name_variants(nm)] + [0.0])
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_i = i

        kept_player = df.at[best_i, p_name] if best_i is not None else (names[0] if names else "")
        cleared = []
        for i, nm in zip(rows, names):
            if best_i is not None and i == best_i:
                continue
            df.at[i, "NHL_ID"] = ""
            cleared.append(nm)

        cleared_total += len(cleared)
        report_rows.append({
            "NHL_ID": nhl_id,
            "official_name": official,
            "kept_player": kept_player,
            "kept_score": round(float(best_score), 4) if best_score >= 0 else 0.0,
            "cleared_players": " | ".join([c for c in cleared if _to_str(c)])[:5000],
            "cleared_count": int(len(cleared)),
        })

    rep = pd.DataFrame(report_rows).sort_values(by=["cleared_count","kept_score"], ascending=[False, True])
    msg = f"Auto-fix terminé: {len(dup_ids)} NHL_ID dupliqués traités, {cleared_total} NHL_ID retirés."
    return True, msg, df, rep

def _pick_name_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["player", "joueur", "skaters", "name", "full_name", "fullname"]:
        if key in cmap:
            return cmap[key]
    return None

def _make_row_key(df: pd.DataFrame) -> pd.Series:
    """Stable key for diff: prefer NHL_ID when present, else normalized player name."""
    if df is None or df.empty:
        return pd.Series([], dtype=str)
    name_col = _pick_name_col(df) or df.columns[0]
    names = df[name_col].astype(str).map(_normalize_player_name)

    if "NHL_ID" in df.columns:
        ids = df["NHL_ID"].astype(str).str.strip()
    else:
        ids = pd.Series([""] * len(df))

    # NHL_ID dominates when present
    key = np.where(ids.astype(str).str.strip() != "", "NHL:" + ids.astype(str).str.strip(), "NAME:" + names)
    return pd.Series(key, index=df.index, dtype=str)

def _safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})

def _build_diff_and_audit(before: pd.DataFrame, after: pd.DataFrame, max_rows: int = 50000, compare_cols: Optional[List[str]] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Returns (summary_dict, audit_df).
    audit_df columns: key, change_type, field, before, after
    """
    summary: Dict[str, Any] = {
        "before_rows": int(len(before)) if isinstance(before, pd.DataFrame) else 0,
        "after_rows": int(len(after)) if isinstance(after, pd.DataFrame) else 0,
        "added": 0,
        "removed": 0,
        "modified_rows": 0,
        "audit_truncated": False,
    }

    if not isinstance(after, pd.DataFrame) or after.empty:
        return summary, pd.DataFrame(columns=["key", "change_type", "field", "before", "after"])

    before = before if isinstance(before, pd.DataFrame) else pd.DataFrame()
    after = after.copy()

    bkey = _make_row_key(before) if not before.empty else pd.Series([], dtype=str)
    akey = _make_row_key(after)

    if not before.empty:
        b = before.copy()
        b["_row_key"] = bkey.values
    else:
        b = pd.DataFrame(columns=["_row_key"])

    a = after.copy()
    a["_row_key"] = akey.values

    # Key sets
    bset = set(b["_row_key"].dropna().astype(str).tolist()) if "_row_key" in b.columns else set()
    aset = set(a["_row_key"].dropna().astype(str).tolist()) if "_row_key" in a.columns else set()

    added_keys = sorted(list(aset - bset))
    removed_keys = sorted(list(bset - aset))
    common_keys = sorted(list(aset & bset))

    summary["added"] = len(added_keys)
    summary["removed"] = len(removed_keys)

    # Choose columns to compare (important ones first)
    default_cols_interest = [
        "Player", "Joueur", "Team", "Équipe", "Position", "Jersey#", "Country",
        "Level", "Cap Hit", "Length", "Start Year", "Signing Status", "Expiry Year", "Expiry Status",
        "Status",
    ]
    # If user selected specific columns in Admin, use them; else use defaults.
    cols_interest = list(compare_cols) if (compare_cols and len(compare_cols) > 0) else default_cols_interest
    # Keep only columns that exist in either frame (case-sensitive)
    cols_existing = []
    for c in cols_interest:
        if c in after.columns or (not before.empty and c in before.columns):
            cols_existing.append(c)

    # Also include NHL_ID if present
    if "NHL_ID" in after.columns or (not before.empty and "NHL_ID" in before.columns):
        cols_existing = ["NHL_ID"] + [c for c in cols_existing if c != "NHL_ID"]

    # Build index by key for before/after (dedupe by first occurrence)
    if not b.empty:
        b_idx = b.drop_duplicates("_row_key", keep="first").set_index("_row_key", drop=True)
    else:
        b_idx = pd.DataFrame().set_index(pd.Index([], name="_row_key"))

    a_idx = a.drop_duplicates("_row_key", keep="first").set_index("_row_key", drop=True)

    audit_rows: List[Dict[str, Any]] = []

    def _row_json(df_row: pd.Series) -> str:
        try:
            d = {k: ("" if pd.isna(v) else v) for k, v in df_row.to_dict().items()}
            # don't include huge blobs
            d.pop("_row_key", None)
            return json.dumps(d, ensure_ascii=False)
        except Exception:
            return ""

    # Added / Removed
    for k in added_keys:
        if len(audit_rows) >= max_rows:
            summary["audit_truncated"] = True
            break
        row = a_idx.loc[k] if k in a_idx.index else None
        audit_rows.append({"key": k, "change_type": "added", "field": "__row__", "before": "", "after": _row_json(row)})

    for k in removed_keys:
        if len(audit_rows) >= max_rows:
            summary["audit_truncated"] = True
            break
        row = b_idx.loc[k] if (not b.empty and k in b_idx.index) else None
        audit_rows.append({"key": k, "change_type": "removed", "field": "__row__", "before": _row_json(row), "after": ""})

    # Modified fields (common)
    modified_rows_set = set()
    for k in common_keys:
        if len(audit_rows) >= max_rows:
            summary["audit_truncated"] = True
            break
        brow = b_idx.loc[k] if (not b.empty and k in b_idx.index) else None
        arow = a_idx.loc[k] if k in a_idx.index else None
        if brow is None or arow is None:
            continue

        for col in cols_existing:
            if len(audit_rows) >= max_rows:
                summary["audit_truncated"] = True
                break
            bval = _to_str(brow.get(col, ""))
            aval = _to_str(arow.get(col, ""))
            # Normalize common empties
            if bval.lower() in ["nan", "none"]:
                bval = ""
            if aval.lower() in ["nan", "none"]:
                aval = ""
            if bval != aval:
                modified_rows_set.add(k)
                audit_rows.append({
                    "key": k,
                    "change_type": "modified",
                    "field": col,
                    "before": bval,
                    "after": aval,
                })

    summary["modified_rows"] = len(modified_rows_set)

    audit_df = pd.DataFrame(audit_rows, columns=["key", "change_type", "field", "before", "after"])
    return summary, audit_df


def _norm_col(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _resolve_nhl_id_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in [
        "nhl_id", "nhlid", "id_nhl", "player_id", "playerid", "nhlplayerid",
        "nhl_id_api", "nhl_id_nhl", "nhl_player_id",
    ]:
        if key in cmap:
            return cmap[key]
    # common variants with spaces
    for c in df.columns:
        if str(c).strip().lower() in ("nhl_id", "nhl id", "nhl-id"):
            return c
    return None


def _resolve_player_name_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    # include french 'joueur'
    for key in ["joueur", "player", "player_name", "nom", "name", "full_name", "playername"]:
        if key in cmap:
            return cmap[key]
    return None


def _resolve_team_col(df: pd.DataFrame) -> str | None:
    """Prefer nul_id as team column if present (per user requirement)."""
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["nul_id", "nulid"]:
        if key in cmap:
            return cmap[key]
    for key in ["team", "equipe", "club", "nhl_team", "team_abbrev", "owner", "proprietaire"]:
        if key in cmap:
            return cmap[key]
    return None


def _normalize_player_name(x: Any) -> str:
    x = str(x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    # handle "Last, First"
    if "," in x:
        a, b = [p.strip() for p in x.split(",", 1)]
        if a and b:
            x = f"{b} {a}"
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# =========================
# Source dropdown discovery
# =========================
def list_data_csvs(data_dir: str = "data") -> List[str]:
    """Return sorted list of CSV paths in data_dir, ensuring key files are present if they exist."""
    paths: List[str] = []
    try:
        if os.path.isdir(data_dir):
            for fn in os.listdir(data_dir):
                if fn.lower().endswith(".csv"):
                    paths.append(os.path.join(data_dir, fn))
    except Exception:
        pass

    # ensure important sources are present if exist
    must = [
        os.path.join(data_dir, "nhl_search_players.csv"),
        os.path.join(data_dir, "nhl_search_players_2025-2026.csv"),
        os.path.join(data_dir, "equipes_joueurs_2025-2026.csv"),
        os.path.join(data_dir, "hockey.players.csv"),
    ]
    for p in must:
        if os.path.exists(p) and p not in paths:
            paths.append(p)

    # stable sort: important first, then alpha
    def _rank(p: str) -> Tuple[int, str]:
        base = os.path.basename(p).lower()
        if base.startswith("nhl_search_players"):
            r = 0
        elif base.startswith("hockey.players"):
            r = 1
        elif base.startswith("equipes_joueurs"):
            r = 2
        else:
            r = 9
        return (r, base)

    paths = sorted(set(paths), key=_rank)
    return paths


# =========================
# Matching from source
# =========================
def recover_from_source(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    target_id_col: str,
    target_name_col: str,
    source_id_col: str,
    source_name_col: str,
    conf: float,
    source_tag: str,
    max_fill: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = target_df.copy()

    # ensure NHL_ID exists
    if target_id_col not in out.columns:
        out[target_id_col] = np.nan

    # tracking cols
    if "nhl_id_source" not in out.columns:
        out["nhl_id_source"] = ""
    if "nhl_id_confidence" not in out.columns:
        out["nhl_id_confidence"] = np.nan

    s = source_df.copy()
    s["_k_name"] = s[source_name_col].map(_normalize_player_name)
    s["_id"] = pd.to_numeric(s[source_id_col], errors="coerce")
    s = s.dropna(subset=["_id"]).drop_duplicates(subset=["_k_name"], keep="first")

    out["_k_name"] = out[target_name_col].map(_normalize_player_name)
    cur = pd.to_numeric(out[target_id_col], errors="coerce")

    miss_mask = cur.isna()
    idx_miss = out.index[miss_mask].tolist()
    if max_fill and len(idx_miss) > int(max_fill):
        idx_miss = idx_miss[: int(max_fill)]

    miss_slice = out.loc[idx_miss, ["_k_name"]].merge(s[["_k_name", "_id"]], on="_k_name", how="left")
    fill_mask = miss_slice["_id"].notna()

    filled = int(fill_mask.sum())
    if filled:
        out.loc[miss_slice.index[fill_mask], target_id_col] = miss_slice.loc[fill_mask, "_id"].values
        out.loc[miss_slice.index[fill_mask], "nhl_id_source"] = source_tag
        out.loc[miss_slice.index[fill_mask], "nhl_id_confidence"] = float(conf)

    out = out.drop(columns=["_k_name"], errors="ignore")
    return out, {"filled": filled}


def audit_nhl_ids(df: pd.DataFrame, id_col: str) -> Dict[str, Any]:
    s = pd.to_numeric(df[id_col], errors="coerce")
    total = int(len(df))
    with_id = int(s.notna().sum())
    missing = total - with_id
    dup_cnt = int(s.dropna().duplicated().sum())
    dup_pct = (dup_cnt / max(with_id, 1)) * 100.0
    miss_pct = (missing / max(total, 1)) * 100.0
    return {
        "total": total,
        "with_id": with_id,
        "missing": missing,
        "missing_pct": miss_pct,
        "dup_cnt": dup_cnt,
        "dup_pct": dup_pct,
    }


# =========================
# NHL Search API generator (optional)
# =========================
def _http_get_json(url: str, timeout: int = 20) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (PoolHockeyPMS)", "Accept": "application/json,text/plain,*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return json.loads(raw)


def _extract_items(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ["data", "players", "items", "results"]:
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def generate_nhl_search_source(
    out_path: str,
    *,
    active_only: bool = True,
    limit: int = 1000,
    timeout_s: int = 20,
    max_pages: int = 20,
    culture: str = "en-us",
    q: str = "*",
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
    base = "https://search.d3.nhle.com/api/v1/search/player"
    all_rows: List[dict] = []
    seen_ids: set[int] = set()
    pages = 0
    start = 0
    used_url = ""

    try:
        while pages < max_pages:
            params = {
                "culture": culture,
                "limit": str(int(limit)),
                "q": q,
                "active": "True" if active_only else "False",
                "start": str(int(start)),
            }
            url = f"{base}?{urllib.parse.urlencode(params)}"
            used_url = url

            payload = _http_get_json(url, timeout=timeout_s)
            items = _extract_items(payload)

            if pages == 0 and active_only and len(items) == 0:
                active_only = False
                continue

            if not items:
                break

            new_count = 0
            for it in items:
                nhl_id = it.get("playerId", it.get("id", it.get("player_id", it.get("NHL_ID"))))
                try:
                    nhl_id_int = int(nhl_id)
                except Exception:
                    continue
                if nhl_id_int in seen_ids:
                    continue
                seen_ids.add(nhl_id_int)
                new_count += 1

                all_rows.append(
                    {
                        "NHL_ID": nhl_id_int,
                        "Player": it.get("name", it.get("fullName", it.get("playerName", ""))),
                        "Team": it.get("teamAbbrev", it.get("team", "")),
                        "Position": it.get("positionCode", it.get("position", "")),
                        "Jersey#": it.get("sweaterNumber", it.get("jerseyNumber", "")),
                        "DOB": it.get("birthDate", it.get("dob", "")),
                        "_source": "nhl_search_api",
                    }
                )

            pages += 1
            if new_count == 0:
                break
            if len(items) < int(limit):
                break
            start += int(limit)
            time.sleep(0.05)

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df["NHL_ID"] = pd.to_numeric(df["NHL_ID"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["NHL_ID"]).drop_duplicates(subset=["NHL_ID"], keep="first")

        errw = save_csv(df, out_path, safe_mode=False, allow_zero=True)
        if errw:
            return pd.DataFrame(), {"url": used_url, "pages": pages, "rows": int(len(df))}, errw

        return df, {"url": used_url, "pages": pages, "rows_saved": int(len(df)), "out_path": out_path}, None

    except Exception as e:
        return pd.DataFrame(), {"url": used_url, "pages": pages}, f"Erreur NHL Search API: {type(e).__name__}: {e}"


# =========================
# UI (render)
# =========================
def render(*args, **kwargs):
    try:
        return _render_impl(*args, **kwargs)
    except Exception as e:
        st.error("Une erreur a été détectée (évite l’écran noir).")
        st.exception(e)
        return None


def _render_impl(ctx: Optional[Dict[str, Any]] = None):
    ctx = ctx or {}
    season = str(ctx.get("season") or "2025-2026")
    data_dir = str(ctx.get("data_dir") or "data")

    st.subheader("🛠️ Outils — synchros (NHL_ID)")

    # =====================================================
    # 🧱 Master Builder (hockey.players_master.csv)
    #   - Fusion hockey.players.csv + PuckPedia2025_26.csv + NHL API (optionnel)
    #   - Aperçu diff avant/après
    #   - Rapport audit CSV: data/master_build_report.csv
    # =====================================================
    with st.expander("🧱 Master Builder", expanded=False):

        # =====================================================
        # 🧭 Guide (Étapes) + Auto-détection NHL_ID
        # =====================================================
        show_guide = st.toggle("🧭 Guide (Étapes 1 → 5)", value=False, key="mb_show_guide")
        if show_guide:
            st.markdown("""
**Étape 1 — Vérifier la source**
- **Ce que ça fait :** vérifie si `data/hockey.players.csv` contient déjà des `NHL_ID`.
- **Résultat :** si `NHL_ID` est vide → l’enrichissement NHL ne démarre pas (normal).

**Étape 2 — Générer une source NHL_ID**
- **Où ça écrit :** toujours dans **`data/hockey.players.csv`** (forcé).

**Important (dummy-proof)**
- Pour que le Master Builder voie tes NHL_ID, le fichier cible doit être **`data/hockey.players.csv`**.
- Si tu écris les NHL_ID dans `equipes_joueurs_*.csv`, tu devras ensuite les **appliquer** dans `hockey.players.csv`.

- **Ce que ça fait :** l’outil “NHL Search” construit une table `Player → NHL_ID`.
- **Résultat :** un fichier CSV de correspondance (ou une mise à jour d’un fichier cible).

**Étape 3 — Appliquer la source NHL_ID (auto)**
- **Dans la section NHL Search :** choisis `data/nhl_search_players.csv`, puis clique le **gros bouton rouge 🟥 ASSOCIER NHL_ID** (et coche la case ✅ si demandée).
- **Ce que ça fait :** détecte automatiquement la meilleure source NHL_ID dans `data/` et la fusionne dans `hockey.players.csv`.
- **Résultat :** `NHL_ID` non vides augmentent (ex: +1234).

**Étape 4 — Construire le master**
- **Ce que ça fait :** fusionne `hockey.players.csv + PuckPedia2025_26.csv (+ NHL API si NHL_ID)`.
- **Résultat :** `data/hockey.players_master.csv` est créé / mis à jour.

**Étape 5 — Audit & téléchargements**
- **Ce que ça fait :** écrit `data/master_build_report.csv` (ajouts / suppressions / champs modifiés).
- **Résultat :** tu peux télécharger le rapport et le master directement ici.
""")

        show_autodetect = st.toggle("🔎 Auto-détection source NHL_ID", value=False, key="mb_show_autodetect")
        if show_autodetect:
            data_dir = DATA_DIR
            players_path = os.path.join(DATA_DIR, "hockey.players.csv")

            candidates = _auto_detect_nhl_id_sources(data_dir)
            if candidates:
                default_src = candidates[0]
                options = ["(auto) " + os.path.basename(default_src)] + [os.path.basename(p) for p in candidates[1:]]
                paths = [default_src] + candidates[1:]
            else:
                options = ["(auto) aucune source trouvée"]
                paths = [""]

            sel = st.selectbox(
                "Source NHL_ID détectée (tu peux en choisir une autre)",
                options=list(range(len(paths))),
                format_func=lambda i: options[i],
                key="mb_src_sel",
            )
            src_path = paths[sel] if sel < len(paths) else (paths[0] if paths else "")

            c1, c2 = st.columns([1, 2])
            with c1:
                do_apply = st.button(
                    "🧷 Appliquer NHL_ID → hockey.players.csv",
                    use_container_width=True,
                    disabled=(not src_path),
                    key="mb_apply_src",
                )
            with c2:
                st.caption(f"Source sélectionnée: `{os.path.basename(src_path) if src_path else 'aucune'}`")

            if do_apply:
                ok2, msg2 = _apply_nhl_id_source_to_players(players_path, src_path)
                if ok2:
                    st.success("✅ " + msg2)
                else:
                    st.error("❌ " + msg2)
        st.divider()

    # =====================================================
    # 📥 Import Drive (CSV) — AUTO + choix manuel (dummy-proof)
    # =====================================================
    with st.expander("📥 Importer un CSV depuis Google Drive (AUTO)", expanded=False):
        st.caption("Import **lecture seule**: liste les CSV d'un dossier Drive et télécharge dans `data/`. Mode AUTO choisit le meilleur fichier (ex: nhl_search_players).")
        if not _drive_oauth_available():
            st.warning("Drive OAuth non configuré. Ajoute `st.secrets[gdrive_oauth]` (client_id, client_secret, refresh_token, folder_id).")
        else:
            folder_id = st.text_input("Folder ID (Drive)", value=_drive_get_folder_id_default(), key="drive_folder_id")
            if not folder_id.strip():
                st.warning("Entre le Folder ID (ou mets-le dans secrets: gdrive_oauth.folder_id).")
            else:
                col1, col2 = st.columns([1,1])
                with col1:
                    do_list = st.button("🔄 Lister les CSV Drive", use_container_width=True, key="drive_list")
                with col2:
                    auto_mode = st.checkbox("AUTO (recommandé)", value=True, key="drive_auto")

                files = st.session_state.get("drive_files_cache", None)
                if do_list or files is None:
                    with st.spinner("Liste des fichiers Drive…"):
                        try:
                            files = _drive_list_csv_files(folder_id)
                            st.session_state["drive_files_cache"] = files
                        except Exception as e:
                            st.error("❌ Drive list error: " + str(e))
                            files = []

                if files:
                    st.success(f"✅ {len(files)} CSV trouvés dans le dossier.")
                    auto_pick = _drive_pick_auto(files)
                    if auto_pick:
                        st.info(f"AUTO sélection: **{auto_pick.get('name','')}** (modifié: {auto_pick.get('modifiedTime','')})")

                    # manual choice (still available)
                    names = [f.get("name","") for f in files]
                    idx = 0
                    if auto_pick and auto_pick.get("name") in names:
                        idx = names.index(auto_pick.get("name"))
                    choice = st.selectbox("Choisir un fichier (optionnel)", options=list(range(len(files))), index=idx, format_func=lambda i: files[i].get("name",""), key="drive_choice")

                    selected = auto_pick if auto_mode else files[choice]
                    if not selected:
                        st.warning("Aucun fichier sélectionné.")
                    else:
                                                # Option idiot-proof: renommer automatiquement
                        auto_rename = st.checkbox("Renommer automatiquement en nhl_search_players.csv (recommandé)", value=True, key="drive_auto_rename")
                        if auto_rename:
                            out_name = "nhl_search_players.csv"
                            st.caption("Nom local forcé: `data/nhl_search_players.csv`")
                        else:
                            out_name = st.text_input("Nom local (dans data/)", value=selected.get("name","download.csv"), key="drive_out_name")
                        out_path = os.path.join(DATA_DIR, os.path.basename(out_name))
                        st.caption(f"Destination: `{out_path}`")

                        if st.button("⬇️ Télécharger dans data/", type="primary", use_container_width=True, key="drive_download"):
                            with st.spinner("Téléchargement Drive → data/ …"):
                                ok, err = _drive_download_file(selected.get("id",""), out_path)
                            if ok:
                                st.success(f"✅ Téléchargé: {out_path}")
                                st.info("Prochain step: retourne dans Master Builder — la source `data/nhl_search_players.csv` sera détectée automatiquement.")
                            else:
                                st.error("❌ Download error: " + err)
                else:
                    st.info("Aucun fichier listé (clique ‘Lister’).")

        master_path = os.path.join(data_dir, "hockey.players_master.csv")
        report_path = os.path.join(data_dir, "master_build_report.csv")

        st.caption("Fusionne **hockey.players.csv + PuckPedia2025_26.csv + NHL API** → **hockey.players_master.csv**.")
        st.caption("📄 Audit écrit dans **data/master_build_report.csv** (ajouts / suppressions / champs modifiés).")

        before_df = pd.DataFrame()
        if os.path.exists(master_path):
            before_df, err = load_csv(master_path)
            if err:
                st.warning(f"Avant: {err}")
            elif isinstance(before_df, pd.DataFrame) and (not before_df.empty):
                st.info(f"Avant: master existant ✅ ({len(before_df)} lignes)")
                st.info("Avant: master existant ✅")

        # -------------------------------------------------
        # ✅ Statut NHL_ID (idiot-proof) (idiot-proof)
        # -------------------------------------------------
        players_path_dbg = os.path.join(DATA_DIR, "hockey.players.csv")
        master_path_dbg = os.path.join(DATA_DIR, "hockey.players_master.csv")
        pstat = _quick_nhl_id_stats(players_path_dbg)
        mstat = _quick_nhl_id_stats(master_path_dbg)

        cA, cB, cC, cD = st.columns([1.2, 1, 1, 1])
        with cA:
            st.markdown("#### 🧾 Statut NHL_ID")
        with cB:
            st.metric("Players (rows)", pstat.get("rows", 0))
        with cC:
            st.metric("Players avec NHL_ID", pstat.get("with_id", 0))
        with cD:
            st.metric("Players manquants", pstat.get("missing", 0))

        if pstat.get("ok") and pstat.get("missing", 0) == 0 and pstat.get("rows", 0) > 0:
            st.success("✅ Tous tes joueurs ont un NHL_ID dans hockey.players.csv — l’enrichissement NHL peut fonctionner.")
        elif pstat.get("ok") and pstat.get("rows", 0) > 0:
            st.warning(f"⚠️ Il manque {pstat.get('missing', 0)} NHL_ID dans hockey.players.csv. Génère une source NHL_ID puis applique-la.")
            # Actions rapides (1 clic)
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("⬇️ Aller à Générer source NHL_ID", use_container_width=True, key="go_nhl_search"):
                    st.session_state["open_nhl_cov"] = True
            with c2:
                st.markdown("🔗 [Aller à la section NHL Search](#nhl_search_section)")
            with c3:
                # 1-clic: remplir depuis data/nhl_search_players.csv si présent
                src_auto = os.path.join(DATA_DIR, "nhl_search_players.csv")
                can_fill = os.path.exists(src_auto)
                threshold_pct = st.slider("Seuil de protection: bloquer si > X% des joueurs reçoivent un NHL_ID", 1, 50, 10, 1, key="mb_guard_pct")
                confirm_big = st.checkbox("Je confirme appliquer un gros changement NHL_ID (au-delà du seuil)", value=False, key="mb_guard_confirm")

                if st.button("🧩 Remplir maintenant", use_container_width=True, disabled=(not can_fill), key="mb_fill_now"):
                    okp, msgp, pstats = _preview_fill_missing_nhl_ids(players_path_dbg, src_auto)
                    if okp and pstats.get("rows", 0) > 0:
                        rate = (pstats.get("would_fill", 0) / max(1, pstats.get("rows", 1))) * 100.0
                        if rate > float(threshold_pct) and not bool(confirm_big):
                            st.error(f"🛑 Protection activée: {pstats.get('would_fill',0)}/{pstats.get('rows',0)} joueurs (≈{rate:.1f}%) recevraient un NHL_ID. Coche la confirmation pour continuer.")
                            prev_df = _build_fill_preview_table(players_path_dbg, src_auto, limit=50)
                            if isinstance(prev_df, pd.DataFrame) and (not prev_df.empty):
                                st.markdown("**Aperçu (top 50) — joueurs qui vont recevoir un NHL_ID**")
                                st.dataframe(prev_df, use_container_width=True, height=280)
                            st.stop()
                    okf, msgf, stats = _fill_missing_nhl_ids_from_source(players_path_dbg, src_auto)
                    if okf:
                        st.success(msgf)
                        # refresh stats after fill
                        st.rerun()
                    else:
                        st.error("❌ " + msgf)
                if not can_fill:
                    st.caption("⚠️ Source manquante: data/nhl_search_players.csv (génère-la plus bas).")
        else:
            st.info("ℹ️ Statut NHL_ID: players non lisible (ou fichier manquant).")

        if mstat.get("ok") and mstat.get("rows", 0) > 0:
            st.caption(f"Master NHL_ID: {mstat.get('with_id', 0)}/{mstat.get('rows', 0)} non vides (fichier: hockey.players_master.csv)")
        st.divider()

        # Colonnes à comparer (diff) — mode lisible
        default_compare = ["Level", "Cap Hit", "Expiry Year", "Expiry Status", "Team", "Position", "Jersey#", "Country", "Status", "NHL_ID"]
        opts_base = []
        try:
            if isinstance(before_df, pd.DataFrame) and (not before_df.empty):
                opts_base = list(before_df.columns)
        except Exception:
            opts_base = []
        opts = sorted(set(opts_base) | set(default_compare) | {"Player"})
        compare_cols = st.multiselect(
            "Colonnes à comparer dans le diff (laisser vide = colonnes clés par défaut)",
            options=opts,
            default=[c for c in default_compare if c in opts],
            key=WKEY + "mb_compare_cols",
        )

        colA, colB, colC = st.columns([1.1, 1.1, 1.2])
        with colA:
            enrich = st.checkbox("Enrichir via NHL API", value=True, key=WKEY + "mb_enrich")
        with colB:
            max_calls = st.number_input("Max appels NHL", min_value=0, max_value=5000, value=250, step=50, key=WKEY + "mb_max_calls")
        with colC:
            st.write("")
            st.write("")
            run_all_btn = st.button("🚀 Tout faire automatiquement", use_container_width=True, key=WKEY + "mb_run_all", help="Génère (si besoin) la source NHL_ID, remplit hockey.players.csv, construit le master, et écrit l’audit.")
            run_btn = st.button("🧱 Construire / Mettre à jour Master", type="primary", use_container_width=True, key=WKEY + "mb_build")


        if run_all_btn:
            st.info("🚀 Pipeline automatique: (1) Source NHL_ID → (2) Remplir NHL_ID → (3) Master → (4) Audit")
            data_dir = data_dir  # (déjà défini plus haut)
            src_auto = os.path.join(data_dir, "nhl_search_players.csv")

            # (1) Générer la source NHL_ID si absente
            if not os.path.exists(src_auto):
                with st.spinner("Étape 1/4 — Génération source NHL_ID (NHL Search API)…"):
                    df_src, meta, err = generate_nhl_search_source(
                        src_auto,
                        active_only=True,
                        limit=1000,
                        timeout_s=20,
                        max_pages=25,
                    )
                if err:
                    st.error("❌ Étape 1/4 échouée: " + err)
                    st.stop()
                st.success(f"✅ Étape 1/4 OK — source créée: {os.path.basename(src_auto)} (rows_saved={meta.get('rows_saved', 0)}, pages={meta.get('pages', 0)})")

            # (2) Remplir NHL_ID manquants dans hockey.players.csv
            with st.spinner("Étape 2/4 — Remplissage NHL_ID manquants dans hockey.players.csv…"):
                okp, msgp, pstats = _preview_fill_missing_nhl_ids(players_path_dbg, src_auto)
            if okp and pstats.get("rows", 0) > 0:
                rate = (pstats.get("would_fill", 0) / max(1, pstats.get("rows", 1))) * 100.0
                guard_pct = float(st.session_state.get("mb_guard_pct", 10))
                guard_ok = bool(st.session_state.get("mb_guard_confirm", False))
                if rate > guard_pct and not guard_ok:
                    st.error(f"🛑 Protection activée: {pstats.get('would_fill',0)}/{pstats.get('rows',0)} joueurs (≈{rate:.1f}%) recevraient un NHL_ID. Coche la confirmation dans Master Builder pour continuer.")
                    prev_df = _build_fill_preview_table(players_path_dbg, src_auto, limit=50)
                    if isinstance(prev_df, pd.DataFrame) and (not prev_df.empty):
                        st.markdown("**Aperçu (top 50) — joueurs qui vont recevoir un NHL_ID**")
                        st.dataframe(prev_df, use_container_width=True, height=280)
                    st.stop()
            okf, msgf, stats = _fill_missing_nhl_ids_from_source(players_path_dbg, src_auto)
            if not okf:
                st.error("❌ Étape 2/4 échouée: " + msgf)
                st.stop()
            st.success("✅ Étape 2/4 OK — " + msgf)

            # (3) Construire master
            with st.spinner("Étape 3/4 — Construction du master (fusion + enrichissement)…"):
                try:
                    from services.master_builder import build_master, MasterBuildConfig
                except Exception as e:
                    st.error("Impossible d'importer services.master_builder. Assure-toi que le fichier est dans /services/master_builder.py.")
                    st.exception(e)
                    st.stop()

                cfg = MasterBuildConfig(
                    data_dir=data_dir,
                    enrich_from_nhl=bool(enrich),
                    max_nhl_calls=int(max_calls),
                )
                after_df, rep = build_master(cfg, write_output=False)

                # -------------------------------------------------
                # 🔎 Audit NHL_ID suspect + Mode review (bloque si duplicates)
                # -------------------------------------------------
                nhl_search_path = os.path.join(DATA_DIR, "nhl_search_players.csv")
                suspects_df = _audit_nhl_id_suspects(players_path_dbg, master_path, nhl_search_path, after_df=after_df)

                dup_ids = int((suspects_df["issue"] == "duplicate_nhl_id").sum()) if (isinstance(suspects_df, pd.DataFrame) and "issue" in suspects_df.columns) else 0
                mism = int(suspects_df["issue"].astype(str).str.startswith("mismatch").sum()) if (isinstance(suspects_df, pd.DataFrame) and "issue" in suspects_df.columns) else 0

                # Always write suspects report (safe)
                suspects_path = os.path.join(DATA_DIR, "nhl_id_suspects.csv")
                _atomic_write_df(suspects_df, suspects_path)

                # Block overwrite if duplicate NHL_ID detected (idiot-proof)
                block_overwrite = st.toggle("🔒 Bloquer l'écriture si NHL_ID suspects", value=True, key=WKEY + "block_suspects")
                if block_overwrite and dup_ids > 0:
                    st.error(f"🛑 Écriture bloquée: {dup_ids} cas de NHL_ID dupliqués détectés. Corrige ou confirme manuellement.")
                    st.caption(f"Rapport: {suspects_path}")

                    # Preview duplicates
                    st.markdown("**Aperçu (top 200) — NHL_ID suspects**")
                    st.dataframe(suspects_df.head(200), use_container_width=True, height=320)
                    # 🧹 Auto-fix (idiot-proof)
                    st.markdown("### 🧹 Réparer automatiquement (recommandé)")
                    st.caption("But: enlever le NHL_ID pour les mauvais joueurs (doublons). Ensuite tu relances le pipeline.")
                    if st.button("🧹 Auto-corriger les doublons NHL_ID (retire les mauvais IDs)", type="primary", use_container_width=True, key=WKEY + "autofix_dups"):
                        okx, msgx, fixed_players, fix_report = _autofix_duplicate_nhl_ids(players_path_dbg, nhl_search_path)
                        if not okx:
                            st.error("❌ " + msgx)
                        else:
                            st.success("✅ " + msgx)
                            if isinstance(fix_report, pd.DataFrame) and (not fix_report.empty):
                                st.markdown("**Aperçu (top 50) — corrections**")
                                st.dataframe(fix_report.head(50), use_container_width=True, height=280)
                                fix_path = os.path.join(DATA_DIR, "nhl_id_autofix_report.csv")
                                _atomic_write_df(fix_report, fix_path)
                                b = _read_file_bytes(fix_path)
                                if b:
                                    st.download_button("📥 Télécharger rapport auto-fix (CSV)", data=b, file_name=os.path.basename(fix_path), mime="text/csv", use_container_width=True, key=WKEY + "dl_autofix")
                            okp, errp = _atomic_write_df(fixed_players, players_path_dbg)
                            if okp:
                                st.success("✅ hockey.players.csv mis à jour. Relance le pipeline.")
                                st.rerun()
                            else:
                                st.error("❌ Écriture hockey.players.csv échouée: " + str(errp))


                    sus_bytes = _read_file_bytes(suspects_path)
                    if sus_bytes:
                        st.download_button(
                            "📥 Télécharger audit NHL_ID suspect (CSV)",
                            data=sus_bytes,
                            file_name=os.path.basename(suspects_path),
                            mime="text/csv",
                            use_container_width=True,
                            key=WKEY + "dl_suspects_block",
                        )

                    # Write pending files instead of overwriting master
                    blocked, msgb, pending_master, pending_report = _write_pending_and_gate(after_df, suspects_df, master_path, report_path)
                    if blocked and msgb not in ("ok", ""):
                        if msgb not in ("blocked_for_review",):
                            st.error("❌ " + msgb)

                    confirm_force = st.checkbox("Je confirme sauvegarder le master malgré des NHL_ID suspects", value=False, key=WKEY + "confirm_force_suspects")
                    if st.button("✅ Sauver master malgré suspects", type="primary", use_container_width=True, disabled=(not confirm_force), key=WKEY + "force_save_master"):
                        # write the real master now
                        okm, erm = _atomic_write_df(after_df, master_path)
                        if okm:
                            try:
                                rep["master_written"] = True
                            except Exception:
                                pass
                            st.success(f"✅ Master écrit malgré suspects: {master_path}")
                        else:
                            st.error(f"❌ Échec écriture master: {erm}")
                    st.stop()

                # If not blocked, proceed to write master
                okm, erm = _atomic_write_df(after_df, master_path)
                if not okm:
                    st.error(f"❌ Échec écriture master: {erm}")
                    st.stop()

            st.success("✅ Étape 3/4 OK — Master généré: " + master_path)

            # (4) Audit / Diff
            with st.spinner("Étape 4/4 — Calcul diff + écriture audit…"):
                summary, audit_df = _build_diff_and_audit(before_df, after_df, max_rows=50000, compare_cols=compare_cols)
                ok, werr = _atomic_write_df(audit_df, report_path)

            if ok:
                st.success(f"🧾 Audit écrit: {report_path} ({len(audit_df)} lignes)")
                # Téléchargements
                rep_bytes = _read_file_bytes(report_path)
                if rep_bytes:
                    st.download_button(
                        "📥 Télécharger rapport CSV (audit fusion)",
                        data=rep_bytes,
                        file_name=os.path.basename(report_path),
                        mime="text/csv",
                        use_container_width=True,
                        key=WKEY + "dl_report_all",
                    )
                if os.path.exists(master_path):
                    master_bytes = _read_file_bytes(master_path)
                    if master_bytes:
                        st.download_button(
                            "📥 Télécharger hockey.players_master.csv",
                            data=master_bytes,
                            file_name=os.path.basename(master_path),
                            mime="text/csv",
                            use_container_width=True,
                            key=WKEY + "dl_master_all",
                        )
                st.error(f"❌ Échec écriture audit: {werr}")

            # Mini résumé (simple)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avant", summary.get("before_rows", 0))
            c2.metric("Après", summary.get("after_rows", 0))
            c3.metric("Ajouts", summary.get("added", 0))
            c4.metric("Modifiés", summary.get("modified_rows", 0))

            st.info("✅ Pipeline terminé. (Si tu veux, refais juste ‘Construire’ pour recalculer l’audit après d’autres edits.)")

        if run_btn:
            # lazy import (évite crash si module absent)
            # lazy import (évite crash si module absent)
            try:
                from services.master_builder import build_master, MasterBuildConfig
            except Exception as e:
                st.error("Impossible d'importer services.master_builder. Assure-toi que le fichier est dans /services/master_builder.py.")
                st.exception(e)
                st.stop()

            cfg = MasterBuildConfig(
                data_dir=data_dir,
                enrich_from_nhl=bool(enrich),
                max_nhl_calls=int(max_calls),
            )
            with st.spinner("Fusion + enrichissement…"):
                after_df, rep = build_master(cfg, write_output=False)

                cfg = MasterBuildConfig(
                    data_dir=data_dir,
                    enrich_from_nhl=bool(enrich),
                    max_nhl_calls=int(max_calls),
                )
                with st.spinner("Fusion + enrichissement…"):
                    after_df, rep = build_master(cfg, write_output=False)

                st.success("✅ Master écrit: data/hockey.players_master.csv")
                st.json(rep)

                # Diff + audit
                summary, audit_df = _build_diff_and_audit(before_df, after_df, max_rows=50000, compare_cols=compare_cols)

                # Write audit report CSV
                ok, werr = _atomic_write_df(audit_df, report_path)
                if ok:
                    st.success(f"🧾 Audit écrit: {report_path} ({len(audit_df)} lignes)")
                    rep_bytes = _read_file_bytes(report_path)
                    if rep_bytes:
                        st.download_button(
                            "📥 Télécharger rapport CSV (audit fusion)",
                            data=rep_bytes,
                            file_name=os.path.basename(report_path),
                            mime="text/csv",
                            use_container_width=True,
                        )

                    # 📥 Télécharger le master (si dispo)
                    if os.path.exists(master_path):
                        master_bytes = _read_file_bytes(master_path)
                        if master_bytes:
                            st.download_button(
                                "📥 Télécharger hockey.players_master.csv",
                                data=master_bytes,
                                file_name=os.path.basename(master_path),
                                mime="text/csv",
                                use_container_width=True,
                            )
                else:
                    st.error(f"❌ Échec écriture audit: {werr}")

                # Preview diff (avant/après)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lignes avant", summary.get("before_rows", 0))
                c2.metric("Lignes après", summary.get("after_rows", 0))
                c3.metric("Ajouts", summary.get("added", 0))
                c4.metric("Suppressions", summary.get("removed", 0))

                st.metric("Lignes modifiées (au moins 1 champ)", summary.get("modified_rows", 0))

                if summary.get("audit_truncated"):
                    st.warning("⚠️ Audit tronqué (trop de changements). Le fichier contient la première portion seulement.")
                # Détails (plus propre) — tout est dans un seul expander
                show_details = st.toggle("📄 Afficher détails (diff + aperçus)", value=False)
                if show_details:
                    # Détails
                    if not audit_df.empty:
                        st.markdown("**Aperçu des changements (top 200)**")
                        st.dataframe(audit_df.head(200), use_container_width=True)
                    else:
                        st.info("Aucun changement détecté (ou master créé identique).")

                    st.markdown("**Aperçu du master (top 50)**")
                    st.dataframe(after_df.head(50), use_container_width=True)


    # --- Generator (optional)
    st.markdown("### 🌐 Générer source NHL_ID (NHL Search API)")
    out_src = os.path.join(DATA_DIR, "nhl_search_players.csv")

    # Statut fichier source (idiot-proof)
    if os.path.exists(out_src):
        try:
            sz = os.path.getsize(out_src)
        except Exception:
            sz = 0
        st.success(f"✅ Source présente: `{out_src}` ({sz} bytes)")
    else:
        st.warning(f"⚠️ Source absente: `{out_src}` — clique le bouton ci-dessous pour la créer.")

    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
    with c2:
        active_only = st.checkbox("Actifs seulement", value=True, key=WKEY + "gen_active")
    with c3:
        limit = st.number_input("Chunk limit", min_value=200, max_value=2000, value=1000, step=100, key=WKEY + "gen_limit")
    with c4:
        timeout_s = st.number_input("Timeout (s)", min_value=5, max_value=60, value=20, step=5, key=WKEY + "gen_timeout")

    if c1.button("🌐 Générer source NHL_ID", use_container_width=True, key=WKEY + "btn_gen"):
        with st.spinner("Génération en cours…"):
            df_out, dbg, err = generate_nhl_search_source(
                out_src, active_only=bool(active_only), limit=int(limit), timeout_s=int(timeout_s), max_pages=25
            )
        if err:
            st.error(err)
            if dbg.get("url"):
                st.caption(f"URL: {dbg.get('url')}")
        else:
            st.success(f"✅ Généré: {dbg.get('rows_saved', 0)} joueurs (pages={dbg.get('pages', 0)}).")
            st.caption(f"Sortie: {out_src}")
            if st.button("🔄 Rafraîchir (voir le fichier)", key=WKEY + "refresh_after_gen"):
                st.rerun()

            # -------------------------------------------------
            # ✅ Vérification couverture NHL_ID (simple)
            # -------------------------------------------------
            with st.expander("✅ Vérifier couverture NHL_ID (par rapport à tes joueurs)", expanded=bool(st.session_state.get("open_nhl_cov", False))):
                target_default = os.path.join(DATA_DIR, "hockey.players.csv")
                target_path2 = st.text_input("Fichier joueurs à vérifier", value=target_default, key="nhl_cov_target")
                source_path2 = os.path.join(DATA_DIR, "nhl_search_players.csv")
                st.caption(f"Source NHL Search utilisée: `{os.path.basename(source_path2)}`")

                if st.session_state.get("open_nhl_cov"):
                    st.session_state["open_nhl_cov"] = False

                if st.button("🔍 Vérifier couverture", use_container_width=True, key="nhl_cov_check"):
                    rep = _nhl_id_coverage_report(target_path2, source_path2)
                    if rep.get("error"):
                        st.error("❌ " + rep["error"])
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Joueurs (target)", rep["target_rows"])
                        c2.metric("Avec NHL_ID", rep["target_with_id"])
                        c3.metric("Manquants", rep["target_missing_id"])

                        st.caption(f"Matchables par nom (dans la source) : {rep['matched_by_name']}  •  Ambigus : {len(rep['ambiguous_names'])}")

                        if rep["missing_names"]:
                            st.warning("Exemples de joueurs sans match dans la source (top 200 normalisés) :")
                            st.write(rep["missing_names"][:50])
                        if rep["ambiguous_names"]:
                            st.info("Noms ambigus (même nom = plusieurs NHL_ID dans la source) — top 200 :")
                            st.write(rep["ambiguous_names"][:50])

                st.divider()
                st.markdown("**Option simple :** remplir automatiquement les NHL_ID manquants par *nom* (ignore les noms ambigus).")
                do_fill = st.button("🧩 Remplir NHL_ID manquants dans hockey.players.csv", use_container_width=True, key="nhl_cov_fill")
                if do_fill:
                    okf, msgf, stats = _fill_missing_nhl_ids_from_source(target_default, os.path.join(DATA_DIR, "nhl_search_players.csv"))
                    if okf:
                        st.success(msgf)
                    else:
                        st.error("❌ " + msgf)
            st.caption(f"Sortie: {out_src}")
            st.caption(f"URL: {dbg.get('url')}")
            if not df_out.empty:
                st.dataframe(df_out.head(10), use_container_width=True)

    st.markdown("---")
    # --- Target file (FORCÉ dummy-proof)
    target_path = os.path.join(data_dir, "hockey.players.csv")
    st.info(f"✅ Fichier cible NHL_ID forcé: `{target_path}` (le Master Builder lit toujours ce fichier).")

    max_per_run = st.number_input("Max par run", min_value=50, max_value=20000, value=1000, step=50, key=WKEY + "maxrun")
    dry_run = st.checkbox("Dry-run (ne sauvegarde pas)", value=False, key=WKEY + "dry")
    override_safe = st.checkbox("Override SAFE MODE (autoriser une baisse NHL_ID)", value=False, key=WKEY + "override_safe")

    st.caption(f"🔒 Prod lock: OFF (ENV/PMS_ENV).")

    # --- Source dropdown (optional)
    st.markdown("### Source de récupération (optionnel)")
    upload = st.file_uploader("Ou uploader un CSV source", type=["csv"], key=WKEY + "upload")
    uploaded_df = None
    uploaded_name = None
    if upload is not None:
        try:
            uploaded_df = pd.read_csv(upload, low_memory=False)
            uploaded_name = f"upload:{upload.name}"
            st.success(f"✅ Upload chargé: {upload.name} ({len(uploaded_df)} lignes)")
        except Exception as e:
            st.error(f"Upload invalide: {type(e).__name__}: {e}")

    # Build source options = all csvs EXCEPT target, plus (None), plus upload option label
    csvs2 = list_data_csvs(data_dir)

    # Dummy-proof: on pousse nhl_search_players.csv en haut s'il existe
    nhl_search_path = os.path.join(DATA_DIR, "nhl_search_players.csv")
    preferred = []
    if os.path.exists(nhl_search_path):
        preferred.append(nhl_search_path)

    others = [p for p in csvs2 if p != target_path and p not in preferred]
    src_opts = ["(Aucune — API NHL uniquement)"] + preferred + others

# ensure nhl_search_players is present when exists
    must_src = os.path.join(data_dir, "nhl_search_players.csv")
    if os.path.exists(must_src) and must_src != target_path and must_src not in src_opts:
        src_opts.insert(1, must_src)

    default_src = must_src if (os.path.exists(must_src) and must_src != target_path) else ("(Aucune — API NHL uniquement)")
    src_choice = st.selectbox(
        "Récupérer NHL_ID depuis…",
        options=src_opts,
        index=(src_opts.index(default_src) if default_src in src_opts else 0),
        key=WKEY + "source",
    )

    # Load target
    df_t, err_t = load_csv(target_path)
    if err_t:
        st.error(err_t)
        return

    # Resolve columns on target
    t_name_col = _resolve_player_name_col(df_t)
    if not t_name_col:
        st.error("Colonne nom joueur introuvable dans le fichier cible (ex: Joueur / Player / player_name).")
        st.caption(f"Colonnes détectées: {list(df_t.columns)}")
        return

    # Determine NHL_ID column name for target (create if missing)
    t_id_col = _resolve_nhl_id_col(df_t) or "NHL_ID"
    created_id = False
    if t_id_col not in df_t.columns:
        df_t[t_id_col] = np.nan
        created_id = True
        st.info("Colonne NHL_ID absente → créée (prête à remplir).")

    # Load source df
    source_df = None
    source_tag = ""
    if uploaded_df is not None:
        source_df = uploaded_df
        source_tag = uploaded_name or "upload"
    elif src_choice and src_choice != "(Aucune — API NHL uniquement)":
        source_df, err_s = load_csv(src_choice)
        if err_s:
            st.error(err_s)
            source_df = None
        else:
            source_tag = os.path.basename(src_choice)

    # If source==target (should not happen, but guard)
    if src_choice == target_path:
        st.warning("Source = fichier cible. Choisis une autre source (ex: nhl_search_players.csv).")
        source_df = None

    # Controls
    conf = st.slider("Score de confiance appliqué aux IDs récupérés", 0.50, 0.99, 0.85, 0.01, key=WKEY + "conf")
    dup_lock = st.slider("🔴 Seuil blocage duplication (%)", 0.5, 20.0, 5.0, 0.5, key=WKEY + "duplock")

    # Button (dummy-proof)
    expected_src = os.path.join(DATA_DIR, "nhl_search_players.csv")
    st.caption(f"Source attendue (recommandé): `{expected_src}`")

    # Simple checkbox de sécurité (empêche les clicks “par erreur”)
    # Auto-check: on coche automatiquement seulement si la source choisie est nhl_search_players.csv
    is_expected_src = False
    try:
        if src_choice and src_choice != "(Aucune — API NHL uniquement)":
            is_expected_src = (os.path.basename(str(src_choice)) == "nhl_search_players.csv")
    except Exception:
        is_expected_src = False

    if is_expected_src:
        st.success("✅ Bonne source détectée: nhl_search_players.csv")
        confirm_src = True
        st.caption("Sécurité auto: OK (pas besoin de cocher).")
    else:
        st.warning("⚠️ Pour être 100% sûr: choisis `data/nhl_search_players.csv` dans “Récupérer NHL_ID depuis…”.")
        confirm_src = st.checkbox(
            "✅ OK, je confirme que la source est correcte et je veux écrire dans hockey.players.csv",
            value=False,
            key=WKEY + "confirm_assoc",
            help="Ce bouton écrit dans data/hockey.players.csv. Coche seulement si tu es sûr.",
        )

    # Disable if no source or not confirmed
    disable_assoc = (source_df is None) or bool(getattr(source_df, "empty", True)) or (not bool(confirm_src))

    
    st.markdown("### ✅ Ici c’est simple :")
    st.markdown("1) **Choisis** `data/nhl_search_players.csv` dans **Récupérer NHL_ID depuis…**")
    st.markdown("2) **Clique** le bouton rouge **🟥 ASSOCIER NHL_ID**")
    st.caption("Ensuite, remonte dans Master Builder et clique 🚀 Tout faire automatiquement.")

    if st.button("🟥 ASSOCIER NHL_ID (écrit dans hockey.players.csv)", key=WKEY + "btn_assoc", type="primary", use_container_width=True, disabled=disable_assoc):
        if source_df is None or source_df.empty:
            st.warning("Aucune source exploitable sélectionnée. Sélectionne nhl_search_players.csv ou upload un CSV avec NHL_ID.")
            st.stop()

        s_id_col = _resolve_nhl_id_col(source_df)
        s_name_col = _resolve_player_name_col(source_df)

        if not s_id_col or s_id_col not in source_df.columns:
            st.error("Source: colonne NHL_ID introuvable.")
            st.caption(f"Colonnes source: {list(source_df.columns)}")
            return
        if not s_name_col or s_name_col not in source_df.columns:
            st.error("Source: colonne nom joueur introuvable (Player/Joueur/Name).")
            st.caption(f"Colonnes source: {list(source_df.columns)}")
            return

        # --- Stats AVANT
        a0 = audit_nhl_ids(df_t, t_id_col)
        st.caption(f"Avant: total={a0['total']}, avec NHL_ID={a0['with_id']}, manquants={a0['missing']} ({a0['missing_pct']:.1f}%), doublons={a0['dup_cnt']} ({a0['dup_pct']:.1f}%).")

        df2, stats = recover_from_source(
            df_t,
            source_df,
            target_id_col=t_id_col,
            target_name_col=t_name_col,
            source_id_col=s_id_col,
            source_name_col=s_name_col,
            conf=float(conf),
            source_tag=source_tag or "source",
            max_fill=int(max_per_run) if int(max_per_run) > 0 else 0,
        )

        a = audit_nhl_ids(df2, t_id_col)
        st.success(f"✅ Récupération terminée: +{int(stats.get('filled', 0))} IDs remplis. Doublons: {a['dup_cnt']} (~{a['dup_pct']:.1f}%).")

        if a["dup_pct"] > float(dup_lock):
            st.error(f"🛑 Blocage duplication: {a['dup_pct']:.1f}% > seuil {dup_lock:.1f}%. Ajuste la source/stratégie avant write.")
            # do not write
            return

        if dry_run:
            st.info("Dry-run: aucune écriture.")
            return

        errw = save_csv(df2, target_path, safe_mode=(not override_safe), allow_zero=False)
        if errw:
            st.error(errw)
            return

        st.success(f"💾 Sauvegardé: {target_path}")

        # Download small audit report
        audit_df = pd.DataFrame({
            "player_name": df2[t_name_col].astype(str),
            "nhl_id": pd.to_numeric(df2[t_id_col], errors="coerce"),
            "missing": pd.to_numeric(df2[t_id_col], errors="coerce").isna(),
            "source": df2.get("nhl_id_source", ""),
            "confidence": df2.get("nhl_id_confidence", np.nan),
        })
        buf = io.StringIO()
        audit_df.to_csv(buf, index=False)
        st.download_button(
            "🧾 Télécharger audit NHL_ID (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=WKEY + "dl_audit",
        )


# Backward-compat alias (if app expects _render_tools)
def _render_tools(*args, **kwargs):
    return render(*args, **kwargs)


def _write_pending_and_gate(after_df: pd.DataFrame, suspects_df: pd.DataFrame, master_path: str, report_path: str) -> tuple[bool, str, str, str]:
    """
    If suspects present, write pending master/audit instead of overwriting the real master.
    Returns (blocked, message, pending_master_path, pending_report_path)
    """
    pending_master = os.path.join(os.path.dirname(master_path) or ".", "_pending_hockey.players_master.csv")
    pending_report = os.path.join(os.path.dirname(report_path) or ".", "_pending_master_build_report.csv")

    if suspects_df is None or suspects_df.empty:
        return False, "ok", "", ""

    # Write pending master for review
    ok1, err1 = _atomic_write_df(after_df, pending_master)
    if not ok1:
        return True, f"Impossible d'écrire le master pending: {err1}", "", ""

    return True, "blocked_for_review", pending_master, pending_report

