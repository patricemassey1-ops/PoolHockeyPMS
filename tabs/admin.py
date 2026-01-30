# tabs/admin.py — PoolHockeyPMS Admin (Backups / Joueurs / Outils)
# ---------------------------------------------------------------
# ✅ Compatible with app.py that calls mod.render(ctx.as_dict()) first.
# ✅ No nested expanders (Streamlit restriction).
# ✅ Backups Google Drive (services/drive.py) + auto-backup 2x/day (00:00 / 12:00 America/Toronto)
# ✅ Joueurs: ajout/retrait/déplacement GC<->CE via un fichier roster saison (long format)
# ✅ Outils: PuckPedia -> Level (STD/ELC), NHL_ID sync (NHL free API) + audit + exports
# ---------------------------------------------------------------

from __future__ import annotations

import os
import io
import json
import time
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple

import re
import streamlit as st
import pandas as pd
import re

# ===== Teams (doit matcher app.py) =====
POOL_TEAMS = [
    "Whalers",
    "Red_Wings",
    "Predateurs",
    "Nordiques",
    "Cracheurs",
    "Canadiens",
]

TZ = ZoneInfo("America/Toronto")


# =====================================================
# Drive helpers (services/drive.py)
# =====================================================
try:
    from services.drive import (
        drive_ready,
        get_drive_folder_id,
        drive_list_files,
        drive_upload_file,
        drive_download_file,
    )
except Exception:
    def drive_ready() -> bool:  # type: ignore
        return False

    def get_drive_folder_id() -> str:  # type: ignore
        return ""

    def drive_list_files(*a, **k):  # type: ignore
        return []

    def drive_upload_file(*a, **k):  # type: ignore
        return {"ok": False, "error": "services/drive.py introuvable"}

    def drive_download_file(*a, **k):  # type: ignore
        return {"ok": False, "error": "services/drive.py introuvable"}


# =====================================================
# ctx helpers (ctx is dict from ctx.as_dict())
# =====================================================
def _get(ctx: Dict[str, Any], key: str, default=None):
    try:
        return ctx.get(key, default)
    except Exception:
        return default


def _norm_player(s: str) -> str:
    s = str(s or "").strip()
    s = " ".join(s.split())
    return s


def _is_missing_id(v: Any) -> bool:
    s = str(v or "").strip()
    if not s or s.lower() == "nan":
        return True
    if s == "0":
        return True
    return False


def _resolve_nhl_id_col(df: pd.DataFrame) -> str:
    """Return standardized NHL id column name.

    Detect common variants even if they contain spaces/hyphens or trailing spaces.
    If a non-standard column is found, copy it into a canonical 'NHL_ID' column
    (without deleting the original) and return 'NHL_ID'.
    If nothing is found, ensure 'NHL_ID' exists (empty) and return it.
    """

    def _norm(s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        return s

    colmap = {_norm(c): c for c in df.columns}

    cand_norm = [
        "nhl_id", "nhlid", "nhl_player_id", "nhlplayerid",
        "player_id", "playerid", "nhl_id_api",
    ]

    found_col = None
    for cn in cand_norm:
        if cn in colmap:
            found_col = colmap[cn]
            break

    if found_col is None:
        for nc, orig in colmap.items():
            if nc.startswith("nhl") and "id" in nc:
                found_col = orig
                break

    if found_col is None:
        if "NHL_ID" not in df.columns:
            df["NHL_ID"] = ""
        return "NHL_ID"

    # Canonicalize into NHL_ID (string)
    if found_col != "NHL_ID":
        df["NHL_ID"] = df[found_col].astype(str)
    else:
        df["NHL_ID"] = df["NHL_ID"].astype(str)

    return "NHL_ID"



def _norm_name(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ,.'-]+", "", s)
    return s.strip()

def _name_to_key(name: str) -> str:
    """Normalise un nom joueur pour matching cross-CSV.
    Supporte 'Last, First' et 'First Last'.
    """
    n = _norm_name(name)
    if not n:
        return ""
    if "," in n:
        parts = [p.strip() for p in n.split(",", 1)]
        last = parts[0]
        first = parts[1] if len(parts) > 1 else ""
    else:
        toks = [t for t in n.split(" ") if t]
        if len(toks) == 1:
            return toks[0]
        first = " ".join(toks[:-1])
        last = toks[-1]
    return f"{last},{first}".strip(",")

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    cmap = {_norm_name(c): c for c in cols}
    for cand in candidates:
        key = _norm_name(cand)
        if key in cmap:
            return cmap[key]
    return None

def recover_nhl_id_from_source(players_df: pd.DataFrame, source_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Récupère NHL_ID depuis un CSV source (roster/export) sans appeler d'API.
    Ne remplit que les valeurs manquantes dans players_df.
    Retourne (df, stats).
    """
    if players_df is None or players_df.empty:
        return players_df, {"filled": 0, "source_rows": 0}

    p = players_df.copy()
    pid_col = _resolve_nhl_id_col(p)

    # Cols nom
    p_name_col = _pick_col(p, ["player", "name", "full name", "full_name", "joueur"])
    if not p_name_col:
        return p, {"filled": 0, "source_rows": 0, "error": "colonne nom introuvable dans players DB"}

    s = source_df.copy()
    sid_col = _resolve_nhl_id_col(s)
    s_name_col = _pick_col(s, ["player", "name", "full name", "full_name", "joueur"])
    if not s_name_col:
        return p, {"filled": 0, "source_rows": len(s), "error": "colonne nom introuvable dans la source"}

    # map source key -> nhl_id (priorité: premier non-nul)
    s[sid_col] = pd.to_numeric(s[sid_col], errors="coerce")
    s["_k"] = s[s_name_col].map(_name_to_key)
    src_map = (
        s.dropna(subset=["_k", sid_col])
         .drop_duplicates(subset=["_k"])
         .set_index("_k")[sid_col]
         .to_dict()
    )

    # fill only missing
    p[pid_col] = pd.to_numeric(p[pid_col], errors="coerce")
    missing_mask = p[pid_col].isna() | (p[pid_col] == 0)
    p["_k"] = p[p_name_col].map(_name_to_key)
    before_missing = int(missing_mask.sum())

    def _fill(row):
        if not (pd.isna(row[pid_col]) or row[pid_col] == 0):
            return row[pid_col]
        k = row["_k"]
        if k and k in src_map:
            return src_map[k]
        return row[pid_col]

    p.loc[missing_mask, pid_col] = p.loc[missing_mask].apply(_fill, axis=1)
    after_missing = int((p[pid_col].isna() | (p[pid_col] == 0)).sum())
    filled = before_missing - after_missing

    p.drop(columns=["_k"], inplace=True, errors="ignore")
    return p, {"filled": filled, "source_rows": len(s), "before_missing": before_missing, "after_missing": after_missing}

def _data_path(data_dir: str, fname: str) -> str:
    return os.path.join(str(data_dir or "data"), fname)


# =====================================================
# Files
# =====================================================
def players_db_path(data_dir: str) -> str:
    return _data_path(data_dir, "hockey.players.csv")


def puckpedia_path(data_dir: str) -> str:
    return _data_path(data_dir, "PuckPedia2025_26.csv")


def roster_path(data_dir: str, season: str) -> str:
    season = str(season or "").strip() or "season"
    return _data_path(data_dir, f"roster_{season}.csv")


def autobackup_state_path(data_dir: str, season: str) -> str:
    season = str(season or "").strip() or "season"
    return _data_path(data_dir, f".autobackup_state_{season}.json")


# =====================================================
# CSV IO helpers
# =====================================================
def _count_nonempty(s: pd.Series) -> int:
    try:
        return int((s.astype(str).str.strip().replace({'nan':'', 'None':''}) != '').sum())
    except Exception:
        return 0


def _coerce_nhl_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les colonnes d'ID NHL sans perdre les valeurs existantes.

    - Accepte 'NHL_ID' ou 'nhl_id' (ou variantes).
    - Crée 'nhl_id' si seulement 'NHL_ID' existe.
    - Fusionne sans écraser une valeur non-vide.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    # Detect candidates
    cols = {c.lower(): c for c in df.columns}
    c_nhl = cols.get("nhl_id")
    c_nhl2 = cols.get("nhl id")  # rare
    c_nhl_up = cols.get("nhl_id".upper().lower())  # no-op

    # Also accept NHL_ID exactly
    c_NHL_ID = None
    for c in df.columns:
        if c == "NHL_ID":
            c_NHL_ID = c
            break

    # Choose existing lower-case id column if present
    if c_nhl is None:
        # try alternate spellings
        for c in df.columns:
            if c.lower().replace(" ", "_") in {"nhl_id", "nhl-id", "nhlid"}:
                c_nhl = c
                break

    # If NHL_ID exists but nhl_id doesn't, create nhl_id
    if c_nhl is None and c_NHL_ID is not None:
        df["nhl_id"] = df[c_NHL_ID]
        c_nhl = "nhl_id"

    # If both exist, merge safely into nhl_id
    if c_nhl is not None and c_NHL_ID is not None and c_nhl != c_NHL_ID:
        a = df[c_nhl].astype(str).str.strip().replace({'nan':'', 'None':''})
        b = df[c_NHL_ID].astype(str).str.strip().replace({'nan':'', 'None':''})
        df[c_nhl] = a.where(a != '', b)

    # Ensure string dtype-ish
    if c_nhl is not None:
        df[c_nhl] = df[c_nhl].astype(str).str.strip().replace({'nan':'', 'None':''})
        df.loc[df[c_nhl] == '', c_nhl] = pd.NA

    return df


def load_csv(path: str) -> tuple[pd.DataFrame, str | None]:
    """Lecture CSV robuste.

    Retourne (df, err). 'err' est None si OK.
    """
    if not path or not os.path.exists(path):
        return pd.DataFrame(), f"introuvable: {path}"
    try:
        df = pd.read_csv(path, low_memory=False)
        df = _coerce_nhl_id_cols(df)
        return df, None
    except Exception:
        return pd.DataFrame(), traceback.format_exc()


def save_csv(df: pd.DataFrame, path: str, *, safe_mode: bool = True, id_cols: list[str] | None = None) -> str:
    """Sauvegarde CSV avec 'SAFE MODE' (empêche d'effacer les NHL_ID par accident)."""
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            return "df invalide"
        df = _coerce_nhl_id_cols(df.copy())

        # SAFE MODE: compare couverture d'ID avec le fichier existant
        id_cols = id_cols or [c for c in df.columns if c.lower().replace(' ', '_') in {'nhl_id', 'nhl-id', 'nhlid'}] + (["NHL_ID"] if "NHL_ID" in df.columns else [])
        id_cols = [c for c in dict.fromkeys(id_cols) if c in df.columns]

        if safe_mode and os.path.exists(path) and id_cols:
            try:
                prev = pd.read_csv(path, low_memory=False)
                prev = _coerce_nhl_id_cols(prev)
                warnings = []
                for c in id_cols:
                    if c in prev.columns and c in df.columns:
                        prev_cnt = _count_nonempty(prev[c])
                        new_cnt = _count_nonempty(df[c])
                        # if we drop more than 2% coverage OR go to 0 from non-zero -> block
                        if prev_cnt > 0 and (new_cnt == 0 or new_cnt < int(prev_cnt * 0.98)):
                            warnings.append((c, prev_cnt, new_cnt))
                if warnings:
                    msg = " | ".join([f"{c}: {a} → {b}" for c, a, b in warnings])
                    return f"SAFE MODE: baisse suspecte NHL_ID ({msg}). Coche 'Override' pour forcer."
            except Exception:
                # If comparison fails, still allow save
                pass

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return ""
    except Exception:
        return traceback.format_exc()



def make_zip_bytes(files: List[str], base_dir: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = os.path.relpath(f, base_dir) if base_dir else os.path.basename(f)
            z.write(f, arcname=arc)
    return mem.getvalue()


def restore_zip_bytes(zip_bytes: bytes, data_dir: str) -> str | None:
    try:
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            z.extractall(data_dir)
        return None
    except Exception as e:
        return str(e)


def collect_backup_files(data_dir: str, season: str) -> list[str]:
    """Return a stable list of local files to include in a backup zip.

    We include common CSV/JSON assets from /data, and season-scoped files.
    This is intentionally conservative (SAFE): it never crashes if the folder is missing.
    """
    files: list[str] = []
    try:
        if not data_dir:
            return files
        os.makedirs(data_dir, exist_ok=True)

        season = str(season or "").strip()
        # include all CSV/JSON in data_dir except obvious temp/backup artifacts
        for name in sorted(os.listdir(data_dir)):
            p = os.path.join(data_dir, name)
            if not os.path.isfile(p):
                continue
            low = name.lower()
            if low.endswith((".csv", ".json")) and not low.endswith(".zip"):
                if low.startswith("backup_") or "checkpoint" in low:
                    continue
                files.append(p)

        # ensure core season roster file(s) are included if present
        if season:
            season_tokens = [season, season.replace("-", "_"), season.replace("-", "–")]
            for name in sorted(os.listdir(data_dir)):
                low = name.lower()
                if not low.endswith(".csv"):
                    continue
                if any(tok.lower() in low for tok in season_tokens):
                    p = os.path.join(data_dir, name)
                    if os.path.isfile(p) and p not in files:
                        files.append(p)

    except Exception:
        # never fail backups UI because of a listing error
        return files
    return files




# =====================================================
# AUTO BACKUP 2x/day (00:00 and 12:00 Toronto)
# Streamlit cannot run background jobs reliably; triggers when Admin is opened.
# =====================================================
def should_autobackup(now: dt.datetime, last_run_iso: str | None) -> bool:
    target_hours = {0, 12}
    if now.hour not in target_hours:
        return False
    if now.minute > 20:
        return False

    if not last_run_iso:
        return True

    try:
        last = dt.datetime.fromisoformat(last_run_iso)
        if last.date() == now.date() and last.hour == now.hour:
            return False
        return True
    except Exception:
        return True


def load_autobackup_state(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def save_autobackup_state(path: str, state: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def run_autobackup_if_due(data_dir: str, season: str, folder_id: str) -> Dict[str, Any]:
    now = dt.datetime.now(TZ)
    sp = autobackup_state_path(data_dir, season)
    state = load_autobackup_state(sp)
    last_run = state.get("last_run_iso")

    if not should_autobackup(now, last_run):
        return {"ok": True, "skipped": True, "reason": "Not in schedule window."}

    files = collect_backup_files(data_dir, season)
    if not files:
        return {"ok": False, "error": "Aucun fichier à sauvegarder."}

    zip_name = f"backup_{season}_{now.strftime('%Y%m%d_%H%M%S')}.zip"
    zip_bytes = make_zip_bytes(files, data_dir)

    tmp = _data_path(data_dir, f"__tmp__{zip_name}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(zip_bytes)
        res = drive_upload_file(folder_id, tmp, zip_name)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error", "Upload Drive échoué.")}
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    state["last_run_iso"] = now.isoformat()
    state["last_zip_name"] = zip_name
    save_autobackup_state(sp, state)
    return {"ok": True, "skipped": False, "zip_name": zip_name}


# =====================================================
# JOUEURS — roster file (long format)
# =====================================================
ROSTER_COLS = ["season", "owner", "bucket", "slot", "player"]


def ensure_roster_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ROSTER_COLS)
    for c in ROSTER_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def roster_load(data_dir: str, season: str) -> pd.DataFrame:
    p = roster_path(data_dir, season)
    if not os.path.exists(p):
        return pd.DataFrame(columns=ROSTER_COLS)
    df, _ = load_csv(p)
    return ensure_roster_schema(df)


def roster_save(df: pd.DataFrame, data_dir: str, season: str) -> str | None:
    p = roster_path(data_dir, season)
    return save_csv(df, p)


# =====================================================
# OUTILS — PuckPedia Level + NHL_ID
# =====================================================
def sync_level(players_path: str, puck_path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "updated": 0}

    missing = []
    if not os.path.exists(players_path):
        missing.append(players_path)
    if not os.path.exists(puck_path):
        missing.append(puck_path)
    if missing:
        result["error"] = "Fichier introuvable"
        result["missing"] = missing
        return result

    try:
        pdb = pd.read_csv(players_path, low_memory=False)
        pk = pd.read_csv(puck_path, low_memory=False)
    except Exception as e:
        result["error"] = f"Lecture CSV échouée: {e}"
        return result

    name_pdb = next((c for c in ["Player", "Joueur", "Name"] if c in pdb.columns), None)
    name_pk = next((c for c in ["Skaters", "Player", "Name"] if c in pk.columns), None)

    if not name_pdb or not name_pk:
        result["error"] = "Colonne joueur introuvable"
        result["players_cols"] = list(pdb.columns)
        result["puck_cols"] = list(pk.columns)
        return result

    if "Level" not in pk.columns:
        result["error"] = "Colonne Level manquante dans PuckPedia"
        result["puck_cols"] = list(pk.columns)
        return result

    if "Level" not in pdb.columns:
        pdb["Level"] = ""

    mp: Dict[str, str] = {}
    for _, r in pk.iterrows():
        nm = _norm_player(r.get(name_pk))
        lv = str(r.get("Level") or "").strip().upper()
        if nm and lv in ("ELC", "STD"):
            mp[nm] = lv

    if not mp:
        result["error"] = "Aucun Level ELC/STD trouvé dans PuckPedia"
        return result

    updated = 0
    new_col = []
    for _, r in pdb.iterrows():
        nm = _norm_player(r.get(name_pdb))
        new_lv = mp.get(nm)
        old_lv = str(r.get("Level") or "").strip().upper()
        if new_lv and old_lv != new_lv:
            updated += 1
            new_col.append(new_lv)
        else:
            new_col.append(r.get("Level"))

    pdb["Level"] = new_col
    err = save_csv(pdb, players_path)
    if err:
        result["error"] = err
        return result

    result["ok"] = True
    result["updated"] = updated
    return result


def audit_nhl_ids(df: pd.DataFrame) -> Dict[str, Any]:
    id_col = _resolve_nhl_id_col(df)

    missing_mask = df[id_col].apply(_is_missing_id)
    missing = int(missing_mask.sum())
    total = int(len(df))
    filled = total - missing
    pct = (missing / total * 100.0) if total else 0.0

    ids = df.loc[~missing_mask, id_col].astype(str).str.strip()
    dup_count = int(ids.duplicated().sum())

    return {"total": total, "filled": filled, "missing": missing, "missing_pct": pct, "duplicates": dup_count, "id_col": id_col}


def _nhl_search_player_id(name: str, session, timeout: int = 10) -> int | None:
    import requests
    q = requests.utils.quote(name)
    url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={q}"
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    data = r.json() or []
    if not data:
        return None
    pid = data[0].get("playerId")
    try:
        return int(pid)
    except Exception:
        return None


def sync_nhl_id(
    players_path: str,
    limit: int = 250,
    dry_run: bool = False,
    allow_decrease: bool = False,
    source_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Complète NHL_ID manquants dans hockey.players.csv.

    - Optionnel: récupère d'abord des NHL_ID depuis une source (roster/export).
    - Puis complète le reste via l'API NHL (search).
    - Safe mode: empêche une baisse du nombre d'IDs si allow_decrease=False.
    """
    if not os.path.exists(players_path):
        return {"ok": False, "error": f"Players DB introuvable: {players_path}"}

    try:
        df = pd.read_csv(players_path, low_memory=False)
    except Exception as e:
        return {"ok": False, "error": f"Lecture players DB échouée: {e}"}

    if df.empty:
        return {"ok": False, "error": "Players DB vide."}

    if "Player" not in df.columns:
        # tolère d'autres noms (mais l'app historique utilise Player)
        return {"ok": False, "error": "Colonne 'Player' introuvable", "players_cols": list(df.columns)}

    id_col = _resolve_nhl_id_col(df)
    before_with = int((~df[id_col].apply(_is_missing_id)).sum())

    rec_stats = None
    if source_df is not None and isinstance(source_df, pd.DataFrame) and not source_df.empty:
        df2, rec_stats = recover_nhl_id_from_source(df, source_df)
        df = df2

    missing_mask = df[id_col].apply(_is_missing_id)
    targets = df[missing_mask].head(int(limit))
    total = int(len(targets))

    # Si plus rien à faire après recover
    if total == 0:
        a = audit_nhl_ids(df)
        after_with = int((~df[id_col].apply(_is_missing_id)).sum())
        if (after_with < before_with) and (not allow_decrease):
            return {"ok": False, "error": "SAFE MODE: baisse détectée (IDs). Annulé.", "audit": a, "recover": rec_stats}
        if not dry_run:
            err = save_csv(df, players_path, safe_mode=(not allow_decrease))
            if err:
                return {"ok": False, "error": err, "recover": rec_stats}
        return {"ok": True, "added": 0, "total": 0, "audit": a, "recover": rec_stats, "summary": "✅ Aucun NHL_ID manquant."}

    bar = st.progress(0)
    txt = st.empty()

    import requests
    session = requests.Session()

    added = 0
    for n, (idx, row) in enumerate(targets.iterrows(), start=1):
        name = _norm_player(row.get("Player"))
        if not name:
            continue

        bar.progress(min(1.0, n / max(1, total)))
        txt.caption(f"Recherche NHL_ID: {n}/{total} — ajoutés: {added}")

        pid = None
        try:
            pid = _nhl_search_player_id(name, session=session, timeout=10)
        except Exception:
            pid = None

        if pid:
            added += 1
            if not dry_run:
                df.at[idx, id_col] = pid

        time.sleep(0.05)

    bar.progress(1.0)
    txt.caption(f"Terminé ✅ — ajoutés: {added}/{total}" + (" (dry-run)" if dry_run else ""))

    a = audit_nhl_ids(df)
    after_with = int((~df[id_col].apply(_is_missing_id)).sum())

    if (after_with < before_with) and (not allow_decrease):
        return {
            "ok": False,
            "error": f"SAFE MODE: baisse détectée (avant={before_with}, après={after_with}). Rien n'a été sauvegardé.",
            "audit": a,
            "recover": rec_stats,
        }

    if not dry_run:
        err = save_csv(df, players_path, safe_mode=(not allow_decrease))
        if err:
            return {"ok": False, "error": err, "recover": rec_stats}

    return {
        "ok": True,
        "added": added,
        "total": total,
        "audit": a,
        "recover": rec_stats,
        "summary": f"✅ Terminé — ajoutés: {added}/{total}" + (" (dry-run)" if dry_run else ""),
    }


def render(ctx: Dict[str, Any]) -> None:
    data_dir = str(_get(ctx, "DATA_DIR", "data"))
    season = str(_get(ctx, "season_lbl", "2025-2026")).strip() or "2025-2026"
    is_admin = bool(_get(ctx, "is_admin", False))

    st.title("🛠️ Gestion Admin")

    if not is_admin:
        st.warning("Accès admin requis.")
        return

    tab = st.radio("", ["Backups", "Joueurs", "Outils"], horizontal=True, index=2)

    if tab == "Backups":
        render_backups(data_dir, season)
    elif tab == "Joueurs":
        render_roster_admin(data_dir, season)
    else:
        render_tools(data_dir, season)


# =====================================================
# UI — Backups
# =====================================================
def render_backups(data_dir: str, season: str) -> None:
    st.subheader("📦 Backups complets (Google Drive)")

    folder_default = (get_drive_folder_id() or "").strip()
    folder_id = st.text_input("Folder ID Drive (backups)", value=folder_default).strip()

    drive_ok = bool(folder_id) and bool(drive_ready())
    if drive_ok:
        st.success("Drive OAuth prêt.")
    else:
        st.warning("Drive non prêt (secrets OAuth manquants / invalides).")

    st.markdown("### ⏱️ Auto-backup (00:00 et 12:00 — America/Toronto)")
    st.caption("Streamlit ne peut pas exécuter en arrière-plan: l’auto-backup se déclenche quand tu ouvres l’onglet Admin (dans la fenêtre horaire).")

    if drive_ok:
        auto = run_autobackup_if_due(data_dir, season, folder_id)
        if auto.get("ok") and not auto.get("skipped"):
            st.success(f"✅ Auto-backup envoyé: {auto.get('zip_name')}")
        elif auto.get("ok") and auto.get("skipped"):
            st.info("Auto-backup: rien à faire maintenant.")
        else:
            st.error(f"Auto-backup: {auto.get('error', 'Erreur')}")
    else:
        st.info("Auto-backup nécessite Drive (OAuth) + folder_id.")

    st.markdown("---")

    files = collect_backup_files(data_dir, season)
    st.markdown("### 🧾 Fichiers inclus")
    if not files:
        st.info("Aucun fichier trouvé pour ce backup.")
    else:
        st.write([os.path.relpath(f, data_dir) for f in files])

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📦 Créer un backup complet", use_container_width=True):
            if not files:
                st.error("Aucun fichier à sauvegarder.")
            else:
                now = dt.datetime.now(TZ)
                zip_name = f"backup_{season}_{now.strftime('%Y%m%d_%H%M%S')}.zip"
                zip_bytes = make_zip_bytes(files, data_dir)

                if drive_ok:
                    tmp = _data_path(data_dir, f"__tmp__{zip_name}")
                    try:
                        with open(tmp, "wb") as fh:
                            fh.write(zip_bytes)
                        res = drive_upload_file(folder_id, tmp, zip_name)
                        if res.get("ok"):
                            st.success(f"✅ Backup envoyé sur Drive: {zip_name}")
                        else:
                            st.error(f"❌ Upload Drive: {res.get('error')}")
                    finally:
                        try:
                            if os.path.exists(tmp):
                                os.remove(tmp)
                        except Exception:
                            pass
                else:
                    st.session_state["__last_zip_name__"] = zip_name
                    st.session_state["__last_zip_bytes__"] = zip_bytes
                    st.success("✅ Backup créé (local). Télécharge-le à droite.")

    with col2:
        zn = st.session_state.get("__last_zip_name__")
        zb = st.session_state.get("__last_zip_bytes__")
        if zn and zb:
            st.download_button("⬇️ Télécharger le ZIP", data=zb, file_name=zn, mime="application/zip", use_container_width=True)
        else:
            st.info("Aucun ZIP local prêt.")

    st.markdown("---")
    st.markdown("### ♻️ Restaurer depuis Drive")
    if not drive_ok:
        st.info("Connecte Drive (OAuth) pour restaurer.")
        return

    backups = drive_list_files(folder_id, name_contains="backup_", limit=200) or []
    backups = [b for b in backups if str(b.get("name", "")).lower().endswith(".zip")]
    backups = sorted(backups, key=lambda x: str(x.get("modifiedTime", "")), reverse=True)

    if not backups:
        st.info("Aucun backup trouvé dans Drive.")
        return

    name_to_id = {b["name"]: b["id"] for b in backups if b.get("name") and b.get("id")}
    sel = st.selectbox("Choisir un backup à restaurer", list(name_to_id.keys()))
    confirm = st.checkbox("Je confirme le restore (écrase data/)", value=False)

    if st.button("♻️ Restaurer", disabled=not confirm, use_container_width=True):
        file_id = name_to_id.get(sel)
        if not file_id:
            st.error("Backup invalide.")
            return

        tmp = _data_path(data_dir, "__restore__.zip")
        res = drive_download_file(file_id, tmp)
        if not res.get("ok"):
            st.error(f"❌ Download: {res.get('error')}")
            return

        try:
            with open(tmp, "rb") as fh:
                zip_bytes = fh.read()
            err = restore_zip_bytes(zip_bytes, data_dir)
            if err:
                st.error(f"❌ Restore échoué: {err}")
            else:
                st.success("✅ Restore terminé.")
                st.rerun()
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


# =====================================================
# UI — Joueurs (roster admin)
# =====================================================
def render_roster_admin(data_dir: str, season: str) -> None:
    st.subheader("👥 Joueurs — gestion roster (manuel)")

    st.caption(
        "Assigne des joueurs à une équipe + bucket/slot. "
        f"Fichier: {os.path.basename(roster_path(data_dir, season))}"
    )

    pdb_path = players_db_path(data_dir)
    pdb, err = load_csv(pdb_path)
    if err:
        st.error(err)
        return
    if "Player" not in pdb.columns:
        st.error("Colonne 'Player' introuvable dans hockey.players.csv")
        st.write(list(pdb.columns))
        return

    roster = roster_load(data_dir, season)

    st.markdown("### ➕ Ajouter joueur(s)")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    with c1:
        options = sorted({_norm_player(x) for x in pdb["Player"].dropna().astype(str).tolist() if str(x).strip()})
        picked = st.multiselect("Joueur(s)", options=options, max_selections=25)
    with c2:
        owner = st.selectbox("Équipe", options=POOL_TEAMS, index=0)
    with c3:
        bucket = st.selectbox("Bucket", options=["GC", "CE"], index=0)
    with c4:
        slot = st.selectbox("Slot", options=["Actif", "Banc", "IR", "Mineur"], index=0)

    if st.button("✅ Ajouter au roster"):
        if not picked:
            st.warning("Choisis au moins 1 joueur.")
        else:
            add_rows = []
            existing_keys = set(
                (str(r.get("season")), str(r.get("owner")), str(r.get("bucket")), str(r.get("slot")), _norm_player(r.get("player")))
                for _, r in roster.iterrows()
            )
            for p in picked:
                key = (season, owner, bucket, slot, _norm_player(p))
                if key in existing_keys:
                    continue
                add_rows.append({"season": season, "owner": owner, "bucket": bucket, "slot": slot, "player": _norm_player(p)})

            if add_rows:
                roster2 = pd.concat([roster, pd.DataFrame(add_rows)], ignore_index=True)
                err2 = roster_save(roster2, data_dir, season)
                if err2:
                    st.error(err2)
                else:
                    st.success(f"Ajoutés: {len(add_rows)}")
                    st.rerun()
            else:
                st.info("Aucun nouvel ajout (déjà présents).")

    st.markdown("---")

    st.markdown("### 🔁 Déplacer GC ↔ CE")
    c5, c6 = st.columns([2, 1])
    with c5:
        roster_owner = st.selectbox("Équipe (roster)", options=POOL_TEAMS, key="roster_owner_filter")
    with c6:
        players_for_owner = roster.loc[roster["owner"].astype(str) == roster_owner, "player"].astype(str).tolist()
        players_for_owner = sorted({_norm_player(x) for x in players_for_owner if str(x).strip()})
        move_player = st.selectbox("Joueur", options=[""] + players_for_owner, key="move_player")

    if st.button("Basculer GC↔CE"):
        if not move_player:
            st.warning("Choisis un joueur.")
        else:
            mask = (
                (roster["season"].astype(str) == season)
                & (roster["owner"].astype(str) == roster_owner)
                & (roster["player"].astype(str).apply(_norm_player) == _norm_player(move_player))
            )
            if int(mask.sum()) == 0:
                st.info("Joueur non trouvé dans le roster.")
            else:
                def _toggle(x):
                    x = str(x or "").strip().upper()
                    return "CE" if x == "GC" else "GC"
                roster.loc[mask, "bucket"] = roster.loc[mask, "bucket"].apply(_toggle)
                err3 = roster_save(roster, data_dir, season)
                if err3:
                    st.error(err3)
                else:
                    st.success("Déplacement effectué.")
                    st.rerun()

    st.markdown("---")

    st.markdown("### 🗑️ Retirer joueur(s)")
    roster_f = roster.copy()
    if not roster_f.empty:
        roster_f["player"] = roster_f["player"].astype(str).apply(_norm_player)

    filt_owner = st.selectbox("Filtrer par équipe", options=["(Toutes)"] + POOL_TEAMS, index=0, key="rm_owner")
    if filt_owner != "(Toutes)":
        roster_f = roster_f[roster_f["owner"].astype(str) == filt_owner]

    st.dataframe(roster_f.sort_values(["owner", "bucket", "slot", "player"]), use_container_width=True, height=360)

    to_remove = st.multiselect(
        "Sélectionne les lignes à retirer (format: index :: owner | bucket | slot | player)",
        options=[f"{i} :: {r['owner']} | {r['bucket']} | {r['slot']} | {r['player']}" for i, r in roster_f.iterrows()],
        max_selections=200,
    )
    confirm = st.checkbox("Je confirme la suppression", value=False)

    if st.button("🗑️ Supprimer", disabled=not (confirm and bool(to_remove))):
        idxs = []
        for s in to_remove:
            try:
                idxs.append(int(str(s).split("::")[0].strip()))
            except Exception:
                pass
        if not idxs:
            st.warning("Aucune ligne valide sélectionnée.")
        else:
            roster2 = roster.drop(index=idxs, errors="ignore").reset_index(drop=True)
            err4 = roster_save(roster2, data_dir, season)
            if err4:
                st.error(err4)
            else:
                st.success(f"Supprimés: {len(idxs)}")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📤 Export roster")
    colx1, colx2 = st.columns(2)
    with colx1:
        st.download_button(
            "⬇️ Export roster (CSV)",
            data=roster.to_csv(index=False).encode("utf-8"),
            file_name=os.path.basename(roster_path(data_dir, season)),
            mime="text/csv",
            use_container_width=True,
        )
    with colx2:
        st.download_button(
            "⬇️ Export roster filtré (CSV)",
            data=roster_f.to_csv(index=False).encode("utf-8"),
            file_name=f"roster_filtered_{season}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =====================================================
# UI — Outils
# =====================================================
def render_tools(data_dir: str, season: str) -> None:
    st.subheader("🧰 Outils — synchros")

    with st.expander("🧾 Sync PuckPedia → Level (STD/ELC)", expanded=False):
        puck = st.text_input("Fichier PuckPedia", puckpedia_path(data_dir))
        players = st.text_input("Players DB", players_db_path(data_dir))

        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("👀 Voir colonnes PuckPedia"):
                try:
                    pk0 = pd.read_csv(puck, nrows=0)
                    st.write(list(pk0.columns))
                except Exception as e:
                    st.error(f"Lecture PuckPedia échouée: {e}")
        with colp2:
            if st.button("👀 Voir colonnes Players DB"):
                try:
                    pdb0 = pd.read_csv(players, nrows=0)
                    st.write(list(pdb0.columns))
                except Exception as e:
                    st.error(f"Lecture Players DB échouée: {e}")

        if st.button("Synchroniser Level"):
            res = sync_level(players, puck)
            if res.get("ok"):
                st.success(f"Levels mis à jour: {res.get('updated', 0)}")
            else:
                st.error(res.get("error", "Erreur inconnue"))
                if res.get("missing"):
                    st.write("Chemins manquants:", res["missing"])
                if res.get("players_cols"):
                    st.write("Players DB colonnes:", res["players_cols"])
                if res.get("puck_cols"):
                    st.write("PuckPedia colonnes:", res["puck_cols"])

    with st.expander("🆔 Sync NHL_ID manquants (avec progression + audit + exports)", expanded=False):
        players2 = st.text_input("Players DB (NHL_ID)", players_db_path(data_dir), key="nhl_players")
        limit = st.number_input("Max par run", 1, 2000, 1000, step=50)
        dry = st.checkbox("Dry-run (ne sauvegarde pas)", value=False)
        override = st.checkbox("Override SAFE MODE (autoriser une baisse NHL_ID)", value=False, key="nhl_id_override")

        if st.button("🔎 Vérifier l'état des NHL_ID"):
            st.caption("Note: si tes IDs sont dans une autre colonne (ex: nhl_id), l’audit l’utilise automatiquement — il ne doit plus te retomber à 100%.")
            df, err = load_csv(players2)
            if err:
                st.error(err)
            else:
                a = audit_nhl_ids(df)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total joueurs", a["total"])
                c2.metric(f"Avec ID ({a.get('id_col','NHL_ID')})", a["filled"])
                c3.metric("Manquants", a["missing"])
                c4.metric("% manquants", f"{a['missing_pct']:.1f}%")

                if a.get("duplicates", 0):
                    st.warning(f"IDs dupliqués détectés: {a['duplicates']} (souvent normal si erreurs de match).")

                st.markdown("#### 📤 Export")
                ex1, ex2 = st.columns(2)
                with ex1:
                    st.download_button(
                        "⬇️ Exporter la liste complète (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="players_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with ex2:
                    id_col = a.get('id_col') or _resolve_nhl_id_col(df)
                    missing_df = df[df[id_col].apply(_is_missing_id)].copy()
                    st.download_button(
                        "⬇️ Exporter la liste des manquants (CSV)",
                        data=missing_df.to_csv(index=False).encode("utf-8"),
                        file_name="players_missing_nhl_id.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                if a["missing"] > 0:
                    cols = ["Player"]
                    if "Team" in df.columns:
                        cols.append("Team")
                    if "Position" in df.columns:
                        cols.append("Position")
                    st.caption("Aperçu (max 200) des joueurs sans NHL_ID :")
                    st.dataframe(missing_df[cols].head(200), use_container_width=True)

        st.markdown("**Source de récupération (optionnel)**")
        src_choice = st.selectbox(
            "Récupérer NHL_ID depuis…",
            ["Aucune (API NHL uniquement)", "Roster export (CSV dans /data)", "Uploader un CSV"],
            index=0,
            key="nhl_id_src_choice",
        )

        source_df = None
        if src_choice == "Roster export (CSV dans /data)":
            cand1 = os.path.join(data_dir, f"roster_{season}.csv")
            cand2 = os.path.join(data_dir, f"roster_filtered_{season}.csv")
            cand3 = os.path.join(data_dir, f"equipes_joueurs_{season}.csv")
            cand_list = [cand1, cand2, cand3]
            existing = [p for p in cand_list if os.path.exists(p)]
            default_p = existing[0] if existing else cand1
            src_path = st.text_input("Chemin du CSV source", value=default_p, key="nhl_id_src_path")
            if src_path and os.path.exists(src_path):
                try:
                    source_df = pd.read_csv(src_path, low_memory=False)
                    source_df = _coerce_nhl_id_cols(source_df)
                    st.caption(f"Source chargée: {os.path.basename(src_path)} — lignes: {len(source_df)}")
                except Exception:
                    st.warning("Impossible de lire la source (CSV).")
                    st.code(traceback.format_exc())
            else:
                st.info("Aucun fichier source trouvé (tu peux coller un chemin valide).")

        elif src_choice == "Uploader un CSV":
            up = st.file_uploader("Uploader un CSV (roster/export) contenant NHL_ID", type=["csv"], key="nhl_id_src_upload")
            if up is not None:
                try:
                    source_df = pd.read_csv(up, low_memory=False)
                    source_df = _coerce_nhl_id_cols(source_df)
                    st.caption(f"Source uploadée — lignes: {len(source_df)}")
                except Exception:
                    st.warning("Impossible de lire le fichier uploadé.")
                    st.code(traceback.format_exc())

        also_api = st.checkbox("Compléter aussi via l'API NHL après récupération", value=True, key="nhl_id_also_api")

        if st.button("Associer NHL_ID"):
            # Si 'also_api' est faux, on fait juste recover sans API
            if source_df is not None and not also_api:
                df0, err0 = load_csv(players2)
                if err0:
                    st.error(err0)
                else:
                    df1, stats = recover_nhl_id_from_source(df0, source_df)
                    a0 = audit_nhl_ids(df1)
                    if not dry:
                        errw = save_csv(df1, players2, safe_mode=(not override))
                        if errw:
                            st.error(errw)
                        else:
                            st.success(f"✅ Récupération terminée — +{stats.get('filled',0)} NHL_ID (sans API).")
                    else:
                        st.info(f"Dry-run: récupéré +{stats.get('filled',0)} NHL_ID (sans sauvegarde).")
                    st.caption(
                        f"État actuel — Total: {a0.get('total')} | Avec NHL_ID: {a0.get('filled')} | "
                        f"Manquants: {a0.get('missing')} ({a0.get('missing_pct',0):.1f}%)"
                    )
            else:
                res = sync_nhl_id(
                    players2,
                    int(limit),
                    dry_run=bool(dry),
                    allow_decrease=bool(override),
                    source_df=source_df,
                )
                if res.get("ok"):
                    st.success(res.get("summary", "Terminé."))
                    if res.get("recover"):
                        st.caption(f"Récupération source: +{(res['recover'] or {}).get('filled',0)} NHL_ID")
                    a = res.get("audit") or {}
                    if a:
                        st.caption(
                            f"État actuel — Total: {a.get('total')} | Avec NHL_ID: {a.get('filled')} | "
                            f"Manquants: {a.get('missing')} ({a.get('missing_pct', 0):.1f}%)"
                        )
                else:
                    st.error(res.get("error", "Erreur inconnue"))