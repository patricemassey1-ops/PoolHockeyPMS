import re
import unicodedata
import pandas as pd
import streamlit as st
from .storage import path_players_db, safe_read_csv
@st.cache_data(show_spinner=False)
def load_players_df(players_db_path: str | None = None) -> pd.DataFrame:
    """Charge hockey.players.csv (ou autre DB joueurs) avec cache."""
    path = players_db_path or path_players_db()
    df = safe_read_csv(path)
    if df is None:
        return pd.DataFrame()
    return df.copy()

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def norm_player_key(name: str) -> str:
    s = _strip_accents(str(name or "")).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("matthew ", "matt ")
    return s

def country_to_flag_emoji(cc: str) -> str:
    cc = (cc or "").strip().upper()
    if len(cc) != 2 or not cc.isalpha():
        return ""
    return chr(127397 + ord(cc[0])) + chr(127397 + ord(cc[1]))

@st.cache_data(show_spinner=False)
def load_players_map(players_db_path: str | None = None) -> dict:
    path = players_db_path or path_players_db()
    df = safe_read_csv(path)
    if df is None or df.empty:
        return {}

    name_col = None
    for c in ["Joueur", "Player", "Name", "Nom"]:
        if c in df.columns:
            name_col = c
            break
    if not name_col:
        return {}

    country_col = "Country" if "Country" in df.columns else None

    out = {}
    for _, r in df.iterrows():
        k = norm_player_key(r.get(name_col))
        if not k:
            continue
        if k not in out:
            out[k] = {
                "country": str(r.get(country_col) or "").strip().upper() if country_col else "",
            }
    return out
