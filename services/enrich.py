# services/enrich.py
# -----------------------------------------------------------------------------
# SAFE module that works BOTH ways:
#   ✅ importable by Streamlit (no argparse side-effects on import)
#   ✅ runnable as a CLI script (python services/enrich.py ...)
#
# Features:
#   1) Recover NHL_ID from a source CSV (e.g. data/nhl_search_players.csv)
#      - Matching priority: name+team -> name -> (optional) name+dob if present
#      - Collision-aware (won't fill when ambiguous)
#   2) Enrich via NHL Stats Live (api-web.nhle.com) using cache JSON + TTL
#   3) Level mapping (STD/ELC):
#      - Heuristic fallback
#      - Optional merge from PuckPedia CSV when available
#
# No extra deps beyond pandas/numpy/stdlib.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# CSV helpers
# =========================
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


# =========================
# Column detection
# =========================
def _norm_col(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def resolve_nhl_id_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["nhl_id", "nhlid", "id_nhl", "player_id", "playerid", "nhl_id_api"]:
        if key in cmap:
            return cmap[key]
    # last resort exact match
    for c in df.columns:
        if str(c).strip().lower() in ("nhl_id", "nhl id"):
            return c
    return None


def resolve_name_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["player", "joueur", "player_name", "name", "nom", "full_name"]:
        if key in cmap:
            return cmap[key]
    return None


def resolve_team_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    # IMPORTANT: nul_id is treated as team/owner bucket in your pool files
    for key in ["nul_id", "nulid"]:
        if key in cmap:
            return cmap[key]
    for key in ["team", "equipe", "team_abbrev", "teamabbrev", "club", "owner", "gm", "proprietaire"]:
        if key in cmap:
            return cmap[key]
    return None


def resolve_dob_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["dob", "birth_date", "birthdate", "date_de_naissance", "date_naissance"]:
        if key in cmap:
            return cmap[key]
    return None


# =========================
# Normalization
# =========================
def normalize_player_name(x: Any) -> str:
    x = "" if x is None else str(x)
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)

    # Convert "Last, First" -> "First Last"
    if "," in x:
        parts = [p.strip() for p in x.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            x = f"{parts[1]} {parts[0]}"

    # Remove punctuation
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_team(x: Any) -> str:
    x = "" if x is None else str(x)
    x = x.strip().upper()
    x = re.sub(r"[^A-Z0-9]", "", x)
    return x


def normalize_dob(x: Any) -> str:
    if x is None:
        return ""
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _clean_id_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x


# =========================
# NHL_ID recovery (collision-aware)
# =========================
def _build_source_maps(
    src: pd.DataFrame,
    src_id_col: str,
    src_name_col: str,
    src_team_col: Optional[str],
    src_dob_col: Optional[str],
) -> Dict[str, pd.DataFrame]:
    s = src.copy()
    s["_id"] = _clean_id_series(s[src_id_col])
    s = s.dropna(subset=["_id"])

    s["_k_name"] = s[src_name_col].map(normalize_player_name)

    if src_team_col and src_team_col in s.columns:
        s["_k_team"] = s[src_team_col].map(normalize_team)
    else:
        s["_k_team"] = ""

    if src_dob_col and src_dob_col in s.columns:
        s["_k_dob"] = s[src_dob_col].map(normalize_dob)
    else:
        s["_k_dob"] = ""

    s["_k_name_team"] = s["_k_name"] + "||" + s["_k_team"]
    s["_k_name_dob"] = s["_k_name"] + "||" + s["_k_dob"]

    def make_map(key_col: str) -> pd.DataFrame:
        tmp = s[[key_col, "_id"]].copy()
        tmp = tmp[tmp[key_col].astype(str).str.strip() != ""]
        # collision_count = nunique ids per key
        g = tmp.groupby(key_col)["_id"].nunique().rename("collision_count").reset_index()
        first = tmp.drop_duplicates(subset=[key_col], keep="first").rename(columns={"_id": "src_nhl_id"})
        out = first.merge(g, on=key_col, how="left")
        out["collision_count"] = out["collision_count"].fillna(1).astype(int)
        return out

    return {
        "name_team": make_map("_k_name_team"),
        "name_dob": make_map("_k_name_dob"),
        "name": make_map("_k_name"),
    }


def recover_nhl_id(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    target_name_col: Optional[str] = None,
    target_team_col: Optional[str] = None,
    target_dob_col: Optional[str] = None,
    target_id_col: str = "NHL_ID",
    source_name_col: Optional[str] = None,
    source_team_col: Optional[str] = None,
    source_dob_col: Optional[str] = None,
    source_id_col: Optional[str] = None,
    prefer_name_team: bool = True,
    allow_name_dob: bool = True,
    allow_name_only: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fills target_id_col only when mapping is UNIQUE (collision_count == 1).
    Returns (df, stats).
    """
    out = target_df.copy()

    # resolve cols
    t_name = target_name_col or resolve_name_col(out)
    if not t_name or t_name not in out.columns:
        raise RuntimeError("Target missing player name column (e.g., Joueur/Player/player_name).")

    t_team = target_team_col or resolve_team_col(out)
    t_dob = target_dob_col or resolve_dob_col(out)

    if target_id_col not in out.columns:
        out[target_id_col] = np.nan

    s_name = source_name_col or resolve_name_col(source_df) or "Player"
    s_id = source_id_col or resolve_nhl_id_col(source_df) or "NHL_ID"
    s_team = source_team_col or resolve_team_col(source_df)
    s_dob = source_dob_col or resolve_dob_col(source_df)

    if s_name not in source_df.columns or s_id not in source_df.columns:
        raise RuntimeError("Source missing NHL_ID and/or player name column.")

    maps = _build_source_maps(source_df, s_id, s_name, s_team, s_dob)

    # target keys
    out["_k_name"] = out[t_name].map(normalize_player_name)
    out["_k_team"] = out[t_team].map(normalize_team) if t_team and t_team in out.columns else ""
    out["_k_dob"] = out[t_dob].map(normalize_dob) if t_dob and t_dob in out.columns else ""
    out["_k_name_team"] = out["_k_name"] + "||" + out["_k_team"]
    out["_k_name_dob"] = out["_k_name"] + "||" + out["_k_dob"]

    cur = _clean_id_series(out[target_id_col])
    missing_mask = cur.isna()

    filled = 0
    collisions_target = 0

    # ensure trace cols
    if "nhl_id_source" not in out.columns:
        out["nhl_id_source"] = ""
    if "nhl_id_confidence" not in out.columns:
        out["nhl_id_confidence"] = np.nan

    def apply_map(key_col: str, map_df: pd.DataFrame, tag: str, conf: float) -> None:
        nonlocal filled, collisions_target, missing_mask

        if not missing_mask.any():
            return

        tmp = out.loc[missing_mask, [key_col]].copy()
        tmp["_idx"] = tmp.index
        joined = tmp.merge(map_df, left_on=key_col, right_on=map_df.columns[0], how="left")
        # collisions that affect target missing rows
        colmask = joined["src_nhl_id"].notna() & (joined["collision_count"].fillna(0).astype(int) > 1)
        collisions_target += int(colmask.sum())

        # fill only unique
        ok = joined["src_nhl_id"].notna() & (joined["collision_count"].fillna(0).astype(int) == 1)
        if ok.any():
            idxs = joined.loc[ok, "_idx"].tolist()
            vals = joined.loc[ok, "src_nhl_id"].astype(float).values
            out.loc[idxs, target_id_col] = vals
            out.loc[idxs, "nhl_id_source"] = tag
            out.loc[idxs, "nhl_id_confidence"] = float(conf)
            filled += int(len(idxs))

            # refresh missing mask
            missing_mask = _clean_id_series(out[target_id_col]).isna()

    # Strategy order
    if prefer_name_team and t_team and s_team:
        apply_map("_k_name_team", maps["name_team"], "source|name_team", 0.93)

    if allow_name_dob and t_dob and s_dob:
        apply_map("_k_name_dob", maps["name_dob"], "source|name_dob", 0.96)

    if allow_name_only:
        apply_map("_k_name", maps["name"], "source|name", 0.85)

    # cleanup temp
    out = out.drop(columns=[c for c in out.columns if c.startswith("_k_")], errors="ignore")

    # stats
    total = int(len(out))
    with_id = int(_clean_id_series(out[target_id_col]).notna().sum())
    missing = total - with_id

    # duplicates among present ids
    sids = _clean_id_series(out[target_id_col]).dropna()
    dup_cnt = int(sids.duplicated(keep=False).sum())
    dup_pct = (dup_cnt / max(int(sids.shape[0]), 1)) * 100.0

    return out, {
        "filled": int(filled),
        "total": total,
        "with_id": with_id,
        "missing": missing,
        "missing_pct": (missing / max(total, 1)) * 100.0,
        "dup_cnt": dup_cnt,
        "dup_pct": dup_pct,
        "collisions_target": int(collisions_target),
    }


# =========================
# NHL Stats Live enrichment
# =========================
def _http_get_json(url: str, timeout: int = 20) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PoolHockeyPMS", "Accept": "application/json,text/plain,*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return json.loads(raw)


def enrich_nhl_stats_live(
    df: pd.DataFrame,
    *,
    id_col: str = "NHL_ID",
    cache_dir: str = "data/nhl_cache",
    ttl_hours: int = 24,
    max_rows: int = 2000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()

    # ensure columns
    defaults: Dict[str, Any] = {
        "nhl_team": "",
        "nhl_position": "",
        "nhl_country": "",
        "nhl_shoots": "",
        "nhl_height_cm": np.nan,
        "nhl_weight_kg": np.nan,
        "nhl_headshot": "",
        "nhl_last_sync": "",
    }
    for c, d in defaults.items():
        if c not in out.columns:
            out[c] = d

    os.makedirs(cache_dir, exist_ok=True)
    ttl_s = int(ttl_hours) * 3600

    ids = pd.to_numeric(out[id_col], errors="coerce")
    idxs = out.index[ids.notna()].tolist()[: int(max_rows)]

    updated = 0
    used_cache = 0
    errors = 0

    for i in idxs:
        try:
            nhl_id = int(ids.loc[i])
        except Exception:
            continue

        cache_path = os.path.join(cache_dir, f"player_{nhl_id}.json")
        payload = None

        # cache valid
        if os.path.exists(cache_path):
            try:
                if (time.time() - os.path.getmtime(cache_path)) <= ttl_s:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    used_cache += 1
            except Exception:
                payload = None

        if payload is None:
            try:
                url = f"https://api-web.nhle.com/v1/player/{nhl_id}/landing"
                payload = _http_get_json(url, timeout=20)
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False)
                except Exception:
                    pass
            except Exception:
                errors += 1
                continue

        # extract defensively
        team = payload.get("currentTeamAbbrev") or payload.get("teamAbbrev") or ""
        pos = payload.get("position") or payload.get("positionCode") or ""
        country = payload.get("birthCountry") or payload.get("countryCode") or ""
        shoots = payload.get("shootsCatches") or payload.get("shoots") or ""
        headshot = payload.get("headshot") or payload.get("headshotUrl") or ""

        out.at[i, "nhl_team"] = str(team or "")
        out.at[i, "nhl_position"] = str(pos or "")
        out.at[i, "nhl_country"] = str(country or "")
        out.at[i, "nhl_shoots"] = str(shoots or "")
        out.at[i, "nhl_headshot"] = str(headshot or "")
        if payload.get("heightInCentimeters") is not None:
            out.at[i, "nhl_height_cm"] = payload.get("heightInCentimeters")
        if payload.get("weightInKilograms") is not None:
            out.at[i, "nhl_weight_kg"] = payload.get("weightInKilograms")
        out.at[i, "nhl_last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        updated += 1

    return out, {"updated": updated, "used_cache": used_cache, "errors": errors, "processed": len(idxs)}


# =========================
# Level STD / ELC
# =========================
def infer_level_std_elc(row: pd.Series) -> str:
    # 1) already present
    for c in ["Level", "level", "ContractLevel", "contract_level"]:
        if c in row.index:
            v = str(row.get(c) or "").strip().upper()
            if v in {"ELC", "STD"}:
                return v

    # 2) signing status / notes
    for c in ["Signing Status", "signing_status", "Status", "statut", "Contract", "contract", "Notes"]:
        if c in row.index:
            txt = str(row.get(c) or "").lower()
            if "entry" in txt or "elc" in txt:
                return "ELC"

    # 3) heuristic length/age if available
    try:
        age = float(row.get("Age")) if "Age" in row.index else np.nan
    except Exception:
        age = np.nan
    try:
        length = float(row.get("Length")) if "Length" in row.index else np.nan
    except Exception:
        length = np.nan

    if np.isfinite(age) and age <= 23 and np.isfinite(length) and length <= 3:
        return "ELC"

    return "STD"


def apply_level_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Level" not in out.columns:
        out["Level"] = ""
    cur = out["Level"].astype(str).str.strip().str.upper()
    mask = cur.isna() | (cur == "") | (cur == "0") | (cur == "NAN")
    out.loc[mask, "Level"] = out.loc[mask].apply(infer_level_std_elc, axis=1)
    return out


def merge_level_from_puckpedia(df: pd.DataFrame, puck_df: pd.DataFrame) -> pd.DataFrame:
    """
    Best effort:
    - if puck has NHL_ID, join on NHL_ID
    - else join on normalized name
    """
    out = df.copy()
    if "Level" not in out.columns:
        out["Level"] = ""

    if puck_df is None or puck_df.empty:
        return out

    lvl_col = None
    for c in puck_df.columns:
        if _norm_col(c) in {"level", "contract_level"}:
            lvl_col = c
            break
    if lvl_col is None:
        # guess by values
        for c in puck_df.columns:
            vals = puck_df[c].astype(str).str.upper().str.strip()
            if ((vals == "ELC") | (vals == "STD")).any():
                lvl_col = c
                break
    if lvl_col is None:
        return out

    pp = puck_df.copy()
    pp["_lvl"] = pp[lvl_col].astype(str).str.upper().str.strip()
    pp["_lvl"] = pp["_lvl"].where(pp["_lvl"].isin(["ELC", "STD"]), np.nan)

    p_id_col = resolve_nhl_id_col(pp)
    if p_id_col and "NHL_ID" in out.columns:
        pp["_id"] = pd.to_numeric(pp[p_id_col], errors="coerce")
        m = pp.dropna(subset=["_id", "_lvl"]).drop_duplicates(subset=["_id"], keep="first").set_index("_id")["_lvl"]
        out_id = pd.to_numeric(out["NHL_ID"], errors="coerce")
        fill_mask = out["Level"].astype(str).str.strip().eq("")
        out.loc[fill_mask, "Level"] = out_id.map(m)
        return out

    # name join fallback
    p_name_col = resolve_name_col(pp)
    t_name_col = resolve_name_col(out)
    if p_name_col and t_name_col:
        pp["_k"] = pp[p_name_col].map(normalize_player_name)
        pp = pp.dropna(subset=["_k", "_lvl"]).drop_duplicates(subset=["_k"], keep="first")
        out["_k"] = out[t_name_col].map(normalize_player_name)
        join = out.merge(pp[["_k", "_lvl"]], on="_k", how="left")
        mask = join["Level"].astype(str).str.strip().eq("") & join["_lvl"].notna()
        out.loc[mask, "Level"] = join.loc[mask, "_lvl"].values
        out = out.drop(columns=["_k"], errors="ignore")
    return out


# =========================
# CLI (optional) — safe because guarded by __main__
# =========================
def _cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="CSV cible (ex: data/hockey.players.csv)")
    ap.add_argument("--source", required=True, help="Source NHL_ID (ex: data/nhl_search_players.csv)")
    ap.add_argument("--puckpedia", default="", help="CSV PuckPedia optionnel (Level)")
    ap.add_argument("--cache-dir", default="data/nhl_cache", help="Cache NHL Stats Live")
    ap.add_argument("--ttl", type=int, default=24, help="TTL cache (heures)")
    ap.add_argument("--max", type=int, default=2000, help="Max lignes enrichies stats par run")
    ap.add_argument("--with-stats", action="store_true", help="Enrichir via NHL Stats Live")
    ap.add_argument("--with-level", action="store_true", help="Appliquer Level STD/ELC")
    ap.add_argument("--write", action="store_true", help="Écrire le fichier cible")
    args = ap.parse_args()

    tgt = load_csv(args.target)
    src = load_csv(args.source)

    # Recover NHL_ID
    out, st1 = recover_nhl_id(tgt, src, target_id_col="NHL_ID")
    print(f"[recover_nhl_id] filled=+{st1['filled']} with_id={st1['with_id']}/{st1['total']} dup%={st1['dup_pct']:.2f} missing={st1['missing']}")

    # Stats
    if args.with_stats:
        out, st2 = enrich_nhl_stats_live(out, id_col="NHL_ID", cache_dir=args.cache_dir, ttl_hours=args.ttl, max_rows=args.max)
        print(f"[enrich_nhl_stats_live] {st2}")

    # Level
    if args.with_level:
        out = apply_level_mapping(out)
        if args.puckpedia and os.path.exists(args.puckpedia):
            pp = load_csv(args.puckpedia)
            out = merge_level_from_puckpedia(out, pp)
        print("[level] applied")

    if args.write:
        save_csv(out, args.target)
        print(f"[write] OK -> {args.target}")
    else:
        print("[dry-run] no write")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
