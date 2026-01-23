# pms_enrich.py
from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_player_key(name: str) -> str:
    """
    Normalise un nom joueur pour matcher:
    - "Last, First" <-> "First Last"
    - accents, ponctuation, doubles espaces
    """
    raw = str(name or "").strip()
    if not raw:
        return ""

    s = _strip_accents(raw).lower()
    s = re.sub(r"[\.\'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # si "last, first" -> "first last"
    if "," in raw:
        parts = [p.strip() for p in raw.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s2 = f"{parts[1]} {parts[0]}"
            s2 = _strip_accents(s2).lower()
            s2 = re.sub(r"[\.\'\-]", " ", s2)
            s2 = re.sub(r"\s+", " ", s2).strip()
            if s2:
                s = s2

    return s


def _guess_name_col(players_db: pd.DataFrame) -> Optional[str]:
    for c in ["Player", "Joueur", "Name", "Nom", "Nom Joueur"]:
        if c in players_db.columns:
            return c
    return None


def enrich_level_from_players_db(
    df: pd.DataFrame,
    players_db: pd.DataFrame,
    *,
    name_col_df: str = "Joueur",
    name_col_db: str | None = None,
    level_col_db: str = "Level",
    expiry_col_db: str = "Expiry Year",
) -> pd.DataFrame:
    """
    Remplit df["Level"] (STD/ELC) et df["Expiry Year"] depuis players_db,
    en matchant sur le nom joueur normalisé.

    - Ne plante jamais si Expiry est NaN / vide
    - Ne touche pas aux valeurs déjà présentes si valides (Level STD/ELC)
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if name_col_df not in out.columns:
        # rien à enrichir
        return out

    # Colonnes cibles dans df
    if "Level" not in out.columns:
        out["Level"] = ""
    if "Expiry Year" not in out.columns:
        out["Expiry Year"] = ""

    # Trouver la colonne "nom" dans players_db si non fournie
    if name_col_db is None:
        name_col_db = _guess_name_col(players_db)

    if not name_col_db or name_col_db not in players_db.columns:
        return out

    db = players_db.copy()

    if level_col_db not in db.columns:
        db[level_col_db] = ""
    if expiry_col_db not in db.columns:
        db[expiry_col_db] = ""

    db["_k"] = db[name_col_db].astype(str).map(_norm_player_key)
    db["_level"] = db[level_col_db].astype(str).str.strip().str.upper()

    # Expiry: normaliser en "YYYY" string ou ""
    exp_raw = pd.to_numeric(db[expiry_col_db], errors="coerce")
    exp_clean: list[str] = []
    for v in exp_raw.values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            exp_clean.append("")
            continue
        try:
            exp_clean.append(str(int(float(v))))
        except Exception:
            exp_clean.append("")
    db["_exp"] = exp_clean

    # mapping key -> (level, exp)
    m_level: dict[str, str] = {}
    m_exp: dict[str, str] = {}
    for _, r in db.iterrows():
        k = str(r["_k"] or "").strip()
        if not k:
            continue
        lv = str(r["_level"] or "").strip().upper()
        ex = str(r["_exp"] or "").strip()

        if k not in m_level or (not m_level[k] and lv):
            m_level[k] = lv
        if k not in m_exp or (not m_exp[k] and ex):
            m_exp[k] = ex

    def _valid_level(x: str) -> bool:
        return str(x or "").strip().upper() in {"STD", "ELC"}

    out["_k"] = out[name_col_df].astype(str).map(_norm_player_key)

    # Remplir Level si vide/invalide
    lvl_now = out["Level"].astype(str)
    need_lvl = ~lvl_now.map(_valid_level)
    out.loc[need_lvl, "Level"] = out.loc[need_lvl, "_k"].map(lambda k: m_level.get(k, "")).fillna("")

    # Remplir Expiry Year si vide
    exp_now = out["Expiry Year"].astype(str).str.strip().str.lower()
    need_exp = exp_now.eq("") | exp_now.eq("nan")
    out.loc[need_exp, "Expiry Year"] = out.loc[need_exp, "_k"].map(lambda k: m_exp.get(k, "")).fillna("")

    out.drop(columns=["_k"], inplace=True, errors="ignore")
    return out
