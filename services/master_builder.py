# services/master_builder.py
# ============================================================
# PMS Pool Hockey — Master Builder
#   - Sources: data/hockey.players.csv + data/PuckPedia2025_26.csv + NHL API (optional)
#   - Output : data/hockey.players_master.csv  (single master)
#   - Key rule: Level (ELC/STD) comes from PuckPedia when available
# ============================================================

from __future__ import annotations

import os
import re
import json
import time
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

# NHL enrichment is optional; requests may not be installed in some environments
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _to_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        return ""
    return s.strip()

def _norm_nhl_id(x: Any) -> str:
    s = _to_str(x)
    s = s.replace(',', '').strip()
    # common case: pandas reads ids as float -> '8475167.0'
    s = re.sub(r"\.0$", "", s)
    # keep only digits
    s = re.sub(r"[^0-9]", "", s)
    return s


def _norm_name(x: Any) -> str:
    s = _to_str(x).lower()
    s = re.sub(r"\s+", " ", s)
    # keep letters/numbers/space/basic punctuation
    s = re.sub(r"[^a-z0-9 \-'.]", "", s)
    return s.strip()

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    # low_memory=False avoids mixed dtype warnings and weird inference issues
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        # fallback
        return pd.read_csv(path, low_memory=False, encoding="latin-1")

def _atomic_write_csv(df: pd.DataFrame, out_path: str) -> None:
    """Write CSV atomically on the SAME filesystem as out_path.

    Streamlit Cloud / some Linux setups may mount /tmp on a different device than your repo,
    causing: OSError: [Errno 18] Invalid cross-device link.
    Fix: create the temp file in the target directory, then os.replace().
    """
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

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None

def _coalesce(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return a where non-empty else b (string-aware)."""
    a_s = a.astype(str)
    b_s = b.astype(str)
    a_ok = a_s.str.strip().ne("") & a_s.str.lower().ne("nan") & a_s.str.lower().ne("none")
    out = a_s.where(a_ok, b_s)
    return out


# -----------------------------
# NHL API (best-effort)
# -----------------------------

def nhl_player_by_id(player_id: str, timeout: int = 15) -> Dict[str, Any]:
    """Best-effort NHL player landing endpoint. Returns {} on any error."""
    if not player_id or not requests:
        return {}
    pid = _to_str(player_id)
    if not pid:
        return {}
    url = f"https://api-web.nhle.com/v1/player/{pid}/landing"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return {}
        j = r.json()
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}

def _extract_nhl_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a small, stable subset of useful fields."""
    if not payload:
        return {}
    out: Dict[str, Any] = {}

    # Player name
    first = payload.get("firstName")
    last = payload.get("lastName")
    if isinstance(first, dict):
        first = first.get("default")
    if isinstance(last, dict):
        last = last.get("default")
    full = " ".join([_to_str(first), _to_str(last)]).strip()
    if full:
        out["Player"] = full

    # Team / pos / jersey
    out["Team"] = payload.get("teamAbbrev") or payload.get("currentTeamAbbrev")
    out["Position"] = payload.get("position") or payload.get("positionCode")
    out["Jersey#"] = payload.get("sweaterNumber") or payload.get("jerseyNumber")

    # Birth / country
    out["DOB"] = payload.get("birthDate")
    out["Country"] = payload.get("birthCountry") or payload.get("birthCountryCode")
    nat = payload.get("nationalityCode") or payload.get("nationality")
    if nat:
        out["Nationality"] = nat

    # Keep only non-empty
    cleaned: Dict[str, Any] = {}
    for k, v in out.items():
        if _to_str(v):
            cleaned[k] = v
    return cleaned


# -----------------------------
# Master builder
# -----------------------------

@dataclass
class MasterBuildConfig:
    data_dir: str = "data"
    players_file: str = "hockey.players.csv"
    puckpedia_file: str = "PuckPedia2025_26.csv"
    master_file: str = "hockey.players_master.csv"

    # NHL enrichment
    enrich_from_nhl: bool = True
    nhl_cache_file: str = "nhl_player_cache.json"
    max_nhl_calls: int = 250
    sleep_s: float = 0.05



def _load_nhl_cache(cache_path: str) -> dict:
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_nhl_cache(cache_path: str, cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def enrich_nhl_cache(cfg: MasterBuildConfig, nhl_ids: list[str] | None = None, max_calls: int | None = None, progress_cb=None) -> dict:
    """
    Progressive enrichment (idiot-proof):
    - Met à jour le cache NHL (cfg.nhl_cache_file) en appelant l'API uniquement pour les NHL_ID manquants.
    - Ne construit PAS le master, ne modifie PAS les CSV (safe).
    Retourne des stats: calls, hits, fetched, missing_remaining_estimate.
    """
    cache_path = os.path.join(cfg.data_dir, cfg.nhl_cache_file)
    cache = _load_nhl_cache(cache_path)

    # Determine ids
    ids = nhl_ids or []
    # Normalize ids
    ids = [_norm_nhl_id(x) for x in ids if _norm_nhl_id(x)]
    ids = list(dict.fromkeys(ids))  # unique, keep order

    calls_limit = int(max_calls if max_calls is not None else cfg.max_nhl_calls)
    calls = 0
    hits = 0
    fetched = 0

    total_ids = len(ids)
    scanned = 0
    for pid in ids:
        scanned += 1

        # Progress callback (lightweight)
        if progress_cb and (scanned == 1 or scanned % 25 == 0):
            try:
                progress_cb(scanned, total_ids, calls, fetched, hits, pid)
            except Exception:
                pass

        if calls >= calls_limit:
            break

        payload = cache.get(pid)
        if isinstance(payload, dict) and len(payload) > 0:
            hits += 1
            if progress_cb and (scanned % 100 == 0):
                try:
                    progress_cb(scanned, total_ids, calls, fetched, hits, pid)
                except Exception:
                    pass
            continue

        # fetch
        payload = nhl_player_by_id(pid)
        if isinstance(payload, dict) and len(payload) > 0:
            cache[pid] = payload
            fetched += 1

        calls += 1

        if progress_cb:
            try:
                progress_cb(scanned, total_ids, calls, fetched, hits, pid)
            except Exception:
                pass

        time.sleep(cfg.sleep_s)

    # Estimate remaining (ids missing usable cache)
    missing_remaining = 0
    for pid in ids:
        payload = cache.get(pid)
        if not (isinstance(payload, dict) and len(payload) > 0):
            missing_remaining += 1

    _save_nhl_cache(cache_path, cache)
    return {
        "cache_path": cache_path,
        "calls": calls,
        "hits": hits,
        "fetched": fetched,
        "missing_remaining_estimate": missing_remaining,
        "ids_total": len(ids),
    }


def build_master(cfg: MasterBuildConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build data/hockey.players_master.csv by merging:
      - hockey.players.csv (base)
      - PuckPedia2025_26.csv (contracts)  -> Level (ELC/STD) MUST come from here when present
      - NHL API (optional enrichment by NHL_ID)

    Returns: (master_df, report_dict)
    """
    players_path = os.path.join(cfg.data_dir, cfg.players_file)
    puck_path = os.path.join(cfg.data_dir, cfg.puckpedia_file)
    out_path = os.path.join(cfg.data_dir, cfg.master_file)
    cache_path = os.path.join(cfg.data_dir, cfg.nhl_cache_file)

    players = _safe_read_csv(players_path)
    puck = _safe_read_csv(puck_path)

    if players.empty:
        raise RuntimeError(f"Fichier base manquant ou vide: {players_path}")

    # --- Ensure base has a Player column
    base_name_col = _first_existing_col(players, ["Player", "Skaters", "Goalies", "Name", "Full Name"])
    if not base_name_col:
        raise RuntimeError("hockey.players.csv doit contenir une colonne joueur (ex: Player).")

    if base_name_col != "Player":
        players = players.rename(columns={base_name_col: "Player"})

    # --- Ensure NHL_ID exists
    if "NHL_ID" not in players.columns:
        players["NHL_ID"] = ""

    players["Player"] = players["Player"].astype(str)
    players["_norm_player"] = players["Player"].map(_norm_name)
    players["_src_players"] = os.path.basename(players_path)

    # --- Prepare PuckPedia
    puck_prepared = pd.DataFrame()
    if not puck.empty:
        puck_name_col = _first_existing_col(puck, ["Player", "Skaters", "Goalies", "Name", "Full Name"])
        if puck_name_col:
            if puck_name_col != "Player":
                puck = puck.rename(columns={puck_name_col: "Player"})
            puck["Player"] = puck["Player"].astype(str)
            puck["_norm_player"] = puck["Player"].map(_norm_name)
            puck["_src_puckpedia"] = os.path.basename(puck_path)
            puck_prepared = puck.copy()

    report: Dict[str, Any] = {
        "built_at_utc": _now_iso(),
        "players_rows": int(len(players)),
        "puckpedia_rows": int(len(puck_prepared)) if not puck_prepared.empty else 0,
        "merge_key": "norm_name (default) + NHL_ID enrich (optional)",
        "nhl_calls": 0,
        "nhl_cache_hits": 0,
        "nhl_enriched_players": 0,
    }

    merged = players.copy()

    # --- Merge PuckPedia (default: name-based)
    if not puck_prepared.empty:
        pp_cols = [c for c in puck_prepared.columns if c != "_norm_player"]
        merged = merged.merge(
            puck_prepared[["_norm_player"] + pp_cols],
            how="left",
            on="_norm_player",
            suffixes=("", "_pp"),
        )
        merged["_src_puckpedia"] = merged["_src_puckpedia"].fillna("")
    else:
        merged["_src_puckpedia"] = ""

    # -------------------------------------------------
    # Level (ELC/STD) — MUST come from PuckPedia when available
    # PuckPedia columns (from your screenshot): Level, Length, Cap Hit, Start Year, Signing Status, Expiry Year, Expiry Status, Status
    # After merge, column is likely "Level_pp".
    # -------------------------------------------------
    if "Level" not in merged.columns:
        merged["Level"] = ""

    pp_level_col = "Level_pp" if "Level_pp" in merged.columns else None
    if pp_level_col:
        pp_vals = merged[pp_level_col].astype(str).str.strip()
        pp_ok = pp_vals.ne("") & pp_vals.str.lower().ne("nan") & pp_vals.str.lower().ne("none")
        merged.loc[pp_ok, "Level"] = pp_vals[pp_ok].str.upper().str.strip()

    merged["Level"] = merged["Level"].astype(str).str.upper().str.strip()
    merged.loc[~merged["Level"].isin(["ELC", "STD"]), "Level"] = ""

    # -------------------------------------------------
    # Coalesce a few contract columns from PuckPedia into master columns
    # (Only fills blanks in base columns.)
    # -------------------------------------------------
    COALESCE_MAP = {
        "Cap Hit": "Cap Hit_pp",
        "Length": "Length_pp",
        "Start Year": "Start Year_pp",
        "Signing Status": "Signing Status_pp",
        "Expiry Year": "Expiry Year_pp",
        "Expiry Status": "Expiry Status_pp",
        "Status": "Status_pp",
    }
    for base_col, pp_col in COALESCE_MAP.items():
        if pp_col in merged.columns:
            if base_col not in merged.columns:
                merged[base_col] = ""
            merged[base_col] = _coalesce(merged[base_col], merged[pp_col])

    # -------------------------------------------------
    # NHL enrichment by NHL_ID (best-effort + cached)
    # -------------------------------------------------
    cache: Dict[str, Any] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if cfg.enrich_from_nhl and requests is not None:
        ids = merged["NHL_ID"].apply(_norm_nhl_id)
        unique_ids = [x for x in ids.unique().tolist() if x]

        calls = 0
        hits = 0
        enriched_players = 0

        for pid in unique_ids:
            if calls >= cfg.max_nhl_calls:
                break

            if pid in cache and isinstance(cache.get(pid), dict) and len(cache.get(pid) or {}) > 0:
                hits += 1
                payload = cache.get(pid) or {}
            else:
                payload = nhl_player_by_id(pid)
                # ne cache pas une réponse vide (sinon ça bloque l’enrichissement pour toujours)
                if isinstance(payload, dict) and len(payload) > 0:
                    cache[pid] = payload
                calls += 1
                time.sleep(cfg.sleep_s)

            fields = _extract_nhl_fields(payload)
            if fields:
                enriched_players += 1
                for k, v in fields.items():
                    if k not in merged.columns:
                        merged[k] = ""
                    mask = merged["NHL_ID"].astype(str).str.strip().eq(pid)
                    blank = merged[k].astype(str).str.strip().eq("")
                    merged.loc[mask & blank, k] = v

        report["nhl_calls"] = int(calls)
        report["nhl_cache_hits"] = int(hits)
        report["nhl_enriched_players"] = int(enriched_players)

        try:
            os.makedirs(cfg.data_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception:
            pass

    # -------------------------------------------------
    # Final touches
    # -------------------------------------------------
    merged["_src_nhl"] = merged["NHL_ID"].astype(str).str.strip().apply(lambda x: "NHL_API" if x else "")
    merged["_updated_at_utc"] = _now_iso()

    if "_norm_player" in merged.columns:
        merged = merged.drop(columns=["_norm_player"])

    _atomic_write_csv(merged, out_path)

    return merged, report


# -------------------------------------------------
# CLI usage (optional)
# -------------------------------------------------
if __name__ == "__main__":
    cfg = MasterBuildConfig()
    df, rep = build_master(cfg)
    print("✅ Master écrit:", os.path.join(cfg.data_dir, cfg.master_file))
    print(rep)

