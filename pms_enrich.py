# pms_enrich.py
from __future__ import annotations

import os
import re
import json
import time
import unicodedata
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd


# ============================================================
# Utils (déjà existants)
# ============================================================
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_player_key(name: str) -> str:
    """
    Normalise un nom joueur pour matcher:
    - "Last, First" / "First Last"
    - apostrophes / accents / espaces
    """
    s = str(name or "").strip()
    if not s:
        return ""
    s = _strip_accents(s).lower()
    s = s.replace(".", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Si format "Last, First"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}".strip()
    return s


def _guess_name_col(df: pd.DataFrame) -> str:
    for c in ["Joueur", "Player", "player", "Name", "name"]:
        if c in df.columns:
            return c
    # fallback: first object col
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]


def enrich_level_from_players_db(df: pd.DataFrame, players_db_path: str, *, player_col: str = "Joueur") -> pd.DataFrame:
    """
    Exemple: override df['Level'] en se basant sur hockey.players.csv.
    Garde tel quel, utile ailleurs.
    """
    if df is None or df.empty:
        return df
    if not players_db_path or not os.path.exists(players_db_path):
        return df
    if player_col not in df.columns:
        return df

    try:
        pdb = pd.read_csv(players_db_path)
        if pdb is None or pdb.empty:
            return df
        # colonnes possibles
        name_col = "Joueur" if "Joueur" in pdb.columns else ("Player" if "Player" in pdb.columns else _guess_name_col(pdb))
        lvl_col = "Level" if "Level" in pdb.columns else ""
        if not lvl_col:
            return df

        mp = {}
        for _, r in pdb.iterrows():
            k = _norm_player_key(r.get(name_col, ""))
            if not k:
                continue
            v = str(r.get(lvl_col, "") or "").strip()
            if v:
                mp[k] = v

        if not mp:
            return df

        def _resolve(x):
            k = _norm_player_key(x)
            return mp.get(k, "")

        out = df.copy()
        if "Level" not in out.columns:
            out["Level"] = ""
        out["Level"] = out[player_col].apply(_resolve).where(out[player_col].notna(), out.get("Level", ""))
        return out
    except Exception:
        return df


# ============================================================
# Players DB updater (Country fill) — utilisé par Admin
# ============================================================

def _first_existing(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


def _players_db_path_candidates(data_dir: str) -> List[str]:
    dd = str(data_dir or "data")
    return [
        os.path.join(dd, "hockey.players.csv"),
        os.path.join(dd, "Hockey.Players.csv"),
        os.path.join(dd, "Hockey.Players.CSV"),
        # fallback "data/data/..." (ancien bug)
        os.path.join(dd, "data", "hockey.players.csv"),
        os.path.join(dd, "data", "Hockey.Players.csv"),
        # fallback racine
        "hockey.players.csv",
        "Hockey.Players.csv",
    ]


def _resolve_players_db_path(data_dir: str) -> str:
    return _first_existing(_players_db_path_candidates(data_dir))


def _nhl_country_cache_path(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "nhl_country_cache.json")


def _nhl_country_checkpoint_path(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "nhl_country_checkpoint.json")


def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json(path: str, obj) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _normalize_country(val: str) -> str:
    v = str(val or "").strip()
    if not v:
        return ""
    # exemples simples (tu peux enrichir plus tard)
    v_up = v.upper()
    if v_up in ["CANADA", "CAN"]:
        return "CA"
    if v_up in ["UNITED STATES", "USA", "US"]:
        return "US"
    if v_up in ["SWEDEN", "SWE"]:
        return "SE"
    if v_up in ["FINLAND", "FIN"]:
        return "FI"
    if v_up in ["RUSSIA", "RUS"]:
        return "RU"
    if v_up in ["CZECHIA", "CZECH REPUBLIC", "CZE"]:
        return "CZ"
    if v_up in ["SLOVAKIA", "SVK"]:
        return "SK"
    if v_up in ["GERMANY", "GER"]:
        return "DE"
    if len(v_up) == 2:
        return v_up
    return v  # fallback


def _find_nhl_id(row: pd.Series) -> str:
    for c in ["NHL ID", "NHL_ID", "nhl_id", "playerId", "PlayerID", "player_id", "NHLID"]:
        if c in row.index:
            v = str(row.get(c, "") or "").strip()
            if v and v.lower() != "nan":
                return v
    return ""


def _fetch_nhl_country_by_id(nhl_id: str, timeout: int = 10) -> str:
    """
    NHL public endpoint (api-web.nhle.com).
    On tente /v1/player/{id}/landing
    """
    try:
        import requests  # lazy import

        url = f"https://api-web.nhle.com/v1/player/{nhl_id}/landing"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return ""
        j = r.json() if isinstance(r.json(), dict) else {}
        # champs possibles selon API
        for k in ["birthCountry", "birthCountryCode", "nationalityCode", "nationality"]:
            v = j.get(k)
            if v:
                return _normalize_country(v)
        # parfois nested
        bc = (j.get("birthplace") or {}).get("country")
        if bc:
            return _normalize_country(bc)
        return ""
    except Exception:
        return ""


def update_players_db(
    *,
    data_dir: str = "data",
    season: str = "2025-2026",
    mode: str = "update",  # "update" ou "resume"
    roster_only: bool = False,
    details: bool = False,
    lock: bool = False,
    reset_cache: bool = False,
    reset_progress: bool = False,
    reset_failed_only: bool = False,
) -> Dict[str, Any]:
    """
    Fonction attendue par tabs/admin.py.
    - Remplit Country dans hockey.players.csv quand possible
    - Utilise cache + checkpoint pour resume
    - Ne casse jamais l'app: retourne un dict "status"
    """
    data_dir = str(data_dir or "data")
    os.makedirs(data_dir, exist_ok=True)

    cache_path = _nhl_country_cache_path(data_dir)
    ckpt_path = _nhl_country_checkpoint_path(data_dir)

    # lock logique (simple)
    if lock:
        return {
            "error": "LOCK ON",
            "path": "",
            "phase": "Country",
            "cursor": 0,
            "total": 0,
            "processed": 0,
            "updated": 0,
            "cached": 0,
            "errors": 0,
            "done": True,
            "cache_path": cache_path,
            "checkpoint_path": ckpt_path,
        }

    # resets
    if reset_cache:
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception:
            pass

    if reset_progress:
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        except Exception:
            pass

    cache: Dict[str, str] = _load_json(cache_path, {})
    ckpt: Dict[str, Any] = _load_json(ckpt_path, {})

    # "failed only": on garde seulement cache entries non vides; ça permet de retenter les vides
    if reset_failed_only:
        try:
            cache = {k: v for k, v in cache.items() if str(v or "").strip()}
            _save_json(cache_path, cache)
        except Exception:
            pass

    players_path = _resolve_players_db_path(data_dir)
    if not players_path or not os.path.exists(players_path):
        return {
            "error": "Players DB vide ou introuvable",
            "path": players_path or "",
            "phase": "Country",
            "cursor": int(ckpt.get("cursor", 0) or 0),
            "total": 0,
            "processed": 0,
            "updated": 0,
            "cached": 0,
            "errors": 0,
            "done": True,
            "cache_path": cache_path,
            "checkpoint_path": ckpt_path,
        }

    try:
        df = pd.read_csv(players_path)
    except Exception as e:
        return {
            "error": f"CSV read error: {e}",
            "path": players_path,
            "phase": "Country",
            "cursor": 0,
            "total": 0,
            "processed": 0,
            "updated": 0,
            "cached": 0,
            "errors": 0,
            "done": True,
            "cache_path": cache_path,
            "checkpoint_path": ckpt_path,
        }

    if df is None or df.empty:
        return {
            "error": "Players DB vide",
            "path": players_path,
            "phase": "Country",
            "cursor": 0,
            "total": 0,
            "processed": 0,
            "updated": 0,
            "cached": 0,
            "errors": 0,
            "done": True,
            "cache_path": cache_path,
            "checkpoint_path": ckpt_path,
        }

    # ensure Country column
    if "Country" not in df.columns:
        df["Country"] = ""

    total = len(df)

    # cursor logic
    cursor = int(ckpt.get("cursor", 0) or 0)
    if mode == "update":
        # update = on repart de 0 mais en gardant cache
        cursor = 0
    cursor = max(0, min(cursor, total))

    # traitement par batch
    BATCH = 300  # comme ton UI
    start = cursor
    end = min(total, start + BATCH)

    processed = 0
    updated = 0
    cached_hits = 0
    errors = 0

    # roster_only : pour l’instant on laisse passer (tu brancheras plus tard via equipes_joueurs_*.csv)
    _ = roster_only

    for i in range(start, end):
        processed += 1
        try:
            row = df.iloc[i]
            cur_country = str(row.get("Country", "") or "").strip()
            if cur_country:
                continue

            nhl_id = _find_nhl_id(row)

            # cache by NHL id si possible, sinon by name key
            cache_key = nhl_id if nhl_id else _norm_player_key(row.get(_guess_name_col(df), ""))

            if cache_key and cache_key in cache and str(cache.get(cache_key) or "").strip():
                df.at[i, "Country"] = cache[cache_key]
                cached_hits += 1
                continue

            # fetch si nhl_id disponible
            country = ""
            if nhl_id:
                country = _fetch_nhl_country_by_id(nhl_id)
                if country:
                    df.at[i, "Country"] = country
                    cache[cache_key] = country
                    updated += 1
                else:
                    # marque empty dans cache pour éviter spam
                    if cache_key:
                        cache[cache_key] = ""
                    errors += 1
            else:
                # pas de nhl_id => on ne peut pas fetch maintenant
                if cache_key and cache_key not in cache:
                    cache[cache_key] = ""
                errors += 1

            # micro-throttle
            time.sleep(0.02)

        except Exception:
            errors += 1

    # save files
    try:
        df.to_csv(players_path, index=False)
    except Exception:
        pass

    _save_json(cache_path, cache)

    # update checkpoint
    new_cursor = end
    done = new_cursor >= total
    ckpt_out = {
        "path": players_path,
        "phase": "Country",
        "cursor": new_cursor,
        "total": total,
        "ts": time.time(),
    }
    _save_json(ckpt_path, ckpt_out)

    out = {
        "path": players_path,
        "phase": "Country",
        "cursor": new_cursor,
        "total": total,
        "processed": processed,
        "updated": updated,
        "cached": cached_hits,
        "errors": errors,
        "done": done,
        "cache_path": cache_path,
        "checkpoint_path": ckpt_path,
    }
    if details:
        out["range"] = [start, end]
        out["mode"] = mode

    return out
