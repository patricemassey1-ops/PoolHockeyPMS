# services/roster_common.py
from __future__ import annotations

import io
import os
import re
import csv
import unicodedata
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import streamlit as st
from pandas.errors import ParserError


# ============================================================
# Common roster IO + normalisation
#   - one single truth for Admin / Alignement / Joueurs
#   - robust CSV reading (Fantrax weird lines)
#   - normalise to: Propriétaire, Joueur, Pos, Equipe, Salaire, Level, Statut, Slot, IR Date
# ============================================================

REQUIRED = ["Propriétaire", "Joueur", "Pos", "Salaire", "Slot"]
CANON = ["Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"]

FANTRAX_ID_COLS = {"id", "playerid", "player_id", "fantraxid", "fantrax id", "fantrax_id"}


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_player_key(name: str) -> str:
    s = _strip_accents(str(name or "")).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-\.]", "", s)
    return s


def _guess_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        candidates = [",", ";", "\t", "|"]
        scores = {d: sample.count(d) for d in candidates}
        return max(scores, key=scores.get) if scores else ","


def read_csv_bytes_robust(raw: bytes, name: str = "csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    enc_used = None
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            enc_used = enc
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("latin-1", errors="replace")
        enc_used = "latin-1"

    sample = text[:5000]
    delim = _guess_delimiter(sample)

    report: Dict[str, Any] = {
        "file": name,
        "encoding": enc_used,
        "delimiter": delim if delim != "\t" else "\\t",
        "engine": "c",
        "warning": "",
    }

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="c")
        return df, report
    except ParserError as e:
        report["warning"] = f"C-engine ParserError: {e}"
    except Exception as e:
        report["warning"] = f"C-engine error: {e}"

    report["engine"] = "python"
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip")
        return df, report
    except Exception as e:
        last_err = e
        for d in [",", ";", "\t", "|"]:
            if d == delim:
                continue
            try:
                report["delimiter"] = d if d != "\t" else "\\t"
                df = pd.read_csv(io.BytesIO(raw), sep=d, engine="python", on_bad_lines="skip")
                report["warning"] = (report["warning"] + f" | fallback delimiter {report['delimiter']}").strip(" |")
                return df, report
            except Exception as ee:
                last_err = ee
        raise last_err


def read_csv_path_robust(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    with open(path, "rb") as f:
        raw = f.read()
    return read_csv_bytes_robust(raw, name=os.path.basename(path))


def _coerce_int_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("$", "", regex=False)
    s = s.str.replace(" ", "", regex=False).str.replace("\u00a0", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _derive_slot_from_status(status: str) -> str:
    v = str(status or "").strip().upper()
    if not v:
        return "Actif"
    if "IR" in v or "INJ" in v:
        return "IR"
    if "MIN" in v or "AHL" in v or "PROS" in v or "NA" in v:
        return "Mineur"
    if "BENCH" in v or "BN" in v:
        return "Banc"
    return "Actif"


@st.cache_data(show_spinner=False)
def _load_level_map(players_db_path: str) -> Dict[str, str]:
    if not players_db_path or not os.path.exists(players_db_path):
        return {}
    try:
        df = pd.read_csv(players_db_path)
    except Exception:
        try:
            df, _ = read_csv_path_robust(players_db_path)
        except Exception:
            return {}

    name_col = None
    for c in ["Joueur", "Player", "Name", "Full Name", "Player Name"]:
        if c in df.columns:
            name_col = c
            break
    if not name_col:
        return {}

    level_col = None
    for c in ["Level", "Contract Level", "level"]:
        if c in df.columns:
            level_col = c
            break
    if not level_col:
        return {}

    m: Dict[str, str] = {}
    for _, row in df[[name_col, level_col]].dropna().iterrows():
        k = norm_player_key(row[name_col])
        v = str(row[level_col] or "").strip().upper()
        if v in ("ELC", "STD"):
            m[k] = v
    return m


def apply_level_auto(df: pd.DataFrame, players_db_path: str) -> pd.DataFrame:
    if df is None or df.empty or "Joueur" not in df.columns:
        return df
    level_map = _load_level_map(players_db_path)
    if not level_map:
        return df
    if "Level" not in df.columns:
        df["Level"] = ""

    def needs(x) -> bool:
        s = str(x or "").strip()
        return s == "" or s in ("0", "0.0")

    mask = df["Level"].apply(needs)
    if mask.any():
        keys = df.loc[mask, "Joueur"].apply(norm_player_key)
        df.loc[mask, "Level"] = keys.map(level_map).fillna("STD")
    return df


def normalize_roster_df(df_in: pd.DataFrame, owner: Optional[str] = None, players_db_path: str = "") -> pd.DataFrame:
    df = df_in.copy()
    # drop Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    # remove duplicate columns like "Pos.1"
    seen = set()
    keep = []
    for c in df.columns:
        base = str(c).split(".")[0]
        if base not in seen:
            keep.append(c)
            seen.add(base)
    df = df[keep].copy()

    # rename obvious id column away from Joueur
    for c in list(df.columns):
        if str(c).strip().lower() in FANTRAX_ID_COLS and "FantraxID" not in df.columns:
            df.rename(columns={c: "FantraxID"}, inplace=True)
            break

    # JOUEUR priority
    if "Joueur" not in df.columns:
        if "Player" in df.columns:
            df.rename(columns={"Player": "Joueur"}, inplace=True)
        elif "Skaters" in df.columns:
            df.rename(columns={"Skaters": "Joueur"}, inplace=True)
        elif "Name" in df.columns:
            df.rename(columns={"Name": "Joueur"}, inplace=True)
        elif "Player Name" in df.columns:
            df.rename(columns={"Player Name": "Joueur"}, inplace=True)
        elif "Full Name" in df.columns:
            df.rename(columns={"Full Name": "Joueur"}, inplace=True)

    # other mappings
    for src, dst in [
        (["Pos", "Position"], "Pos"),
        (["Team", "NHL Team", "Equipe", "Équipe"], "Equipe"),
        (["Salary", "Cap Hit", "CapHit", "Salaire"], "Salaire"),
        (["Status", "Roster Status", "Statut"], "Statut"),
        (["IR Date", "IRDate", "Date IR"], "IR Date"),
        (["Level", "Contract Level"], "Level"),
    ]:
        if dst in df.columns:
            continue
        for c in src:
            if c in df.columns:
                df.rename(columns={c: dst}, inplace=True)
                break

    # ensure cols
    if "Joueur" not in df.columns:
        df["Joueur"] = ""
    if "Pos" not in df.columns:
        df["Pos"] = ""
    if "Salaire" not in df.columns:
        df["Salaire"] = 0

    # fix wrong Joueur that looks like IDs
    j = df["Joueur"].astype(str).str.strip()
    if len(j) > 0:
        looks_like_id = (j.str.len().fillna(0) <= 3).mean() > 0.6
        if looks_like_id:
            for alt in ["Player", "Skaters", "Player Name", "Full Name", "Name"]:
                if alt in df.columns:
                    df["Joueur"] = df[alt].astype(str).str.strip()
                    break

    df["Joueur"] = df["Joueur"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.strip()

    df["Salaire"] = _coerce_int_series(df["Salaire"])

    if "Slot" not in df.columns:
        if "Statut" in df.columns:
            df["Slot"] = df["Statut"].apply(_derive_slot_from_status)
        else:
            df["Slot"] = "Actif"
    else:
        df["Slot"] = df["Slot"].astype(str).str.strip().replace({"": "Actif"})

    if owner is not None and str(owner).strip():
        df["Propriétaire"] = str(owner).strip()
    elif "Propriétaire" not in df.columns:
        df["Propriétaire"] = ""

    # ensure canon optional cols exist
    for c in ["Equipe", "Level", "Statut", "IR Date"]:
        if c not in df.columns:
            df[c] = ""

    # apply Level auto
    if players_db_path:
        df = apply_level_auto(df, players_db_path)

    # final order
    cols = [c for c in CANON if c in df.columns] + [c for c in df.columns if c not in CANON]
    return df[cols].copy()


def equipes_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")


def players_db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "hockey.players.csv")


def save_roster(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


@st.cache_data(show_spinner=False)
def load_roster_cached(path: str, mtime: float) -> pd.DataFrame:
    # mtime is here only to bust cache
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df, _ = read_csv_path_robust(path)
        except Exception:
            return pd.DataFrame(columns=CANON)
    return normalize_roster_df(df, owner=None, players_db_path="")


def load_roster(data_dir: str, season: str) -> Tuple[pd.DataFrame, str]:
    path = equipes_path(data_dir, season)
    if not os.path.exists(path):
        return pd.DataFrame(columns=CANON), path
    mtime = os.path.getmtime(path)
    df = load_roster_cached(path, mtime)
    return df, path


def derive_scope(row: pd.Series) -> str:
    """
    GC vs CE inference:
      - if Statut contains "Grand" => GC
      - if Statut contains "École"/"Ecole"/"CE" => CE
      - else if Slot == Mineur => CE
      - else GC
    """
    s = str(row.get("Statut") or "").upper()
    slot = str(row.get("Slot") or "").upper()
    if "GRAND" in s or s.strip() == "GC":
        return "GC"
    if "ECOLE" in s or "ÉCOLE" in s or s.strip() == "CE":
        return "CE"
    if slot == "MINEUR":
        return "CE"
    return "GC"
