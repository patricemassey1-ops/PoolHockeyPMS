# tabs/joueurs.py
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st


# =====================================================
# Helpers
# =====================================================
def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip()


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    s = _strip_accents(s).lower()
    s = s.replace(".", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if "," in s:
        a, b = [p.strip() for p in s.split(",", 1)]
        s = f"{b} {a}"
    return s


def _first_existing(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _guess_col(df: pd.DataFrame, names) -> str:
    for n in names:
        if n in df.columns:
            return n
    return ""


# =====================================================
# Loaders
# =====================================================
@st.cache_data(show_spinner=False)
def load_players_db(data_dir: str) -> pd.DataFrame:
    path = _first_existing(
        os.path.join(data_dir, "hockey.players.csv"),
        os.path.join(data_dir, "Hockey.Players.csv"),
    )
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    name_col = _guess_col(df, ["Joueur", "Player", "Name", "name"])
    df = df.copy()
    df["_display_name"] = df[name_col].astype(str)
    df["_name_key"] = df[name_col].astype(str).map(_norm_name)

    # normalisation Level
    if "Level" not in df.columns:
        df["Level"] = ""

    # NHL ID
    nhl_col = _guess_col(df, ["NHL ID", "NHL_ID", "playerId", "nhl_id"])
    df["_nhl_id"] = df[nhl_col].astype(str) if nhl_col else ""

    df.attrs["__path__"] = path
    return df


@st.cache_data(show_spinner=False)
def load_contracts(data_dir: str) -> pd.DataFrame:
    path = _first_existing(os.path.join(data_dir, "puckpedia.contracts.csv"))
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    name_col = _guess_col(df, ["Player", "Joueur", "Name"])
    df = df.copy()
    df["_name_key"] = df[name_col].astype(str).map(_norm_name)
    df.attrs["__path__"] = path
    return df


@st.cache_data(show_spinner=False)
def load_points(data_dir: str, season: str) -> pd.DataFrame:
    path = _first_existing(os.path.join(data_dir, f"points_periods_{season}.csv"))
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    name_col = _guess_col(df, ["Joueur", "Player", "Name"])
    pts_col = _guess_col(df, ["Fantasy Points", "Points", "Pts"])

    out = df[[name_col, pts_col]].copy()
    out["_name_key"] = out[name_col].astype(str).map(_norm_name)
    out["_pts"] = pd.to_numeric(out[pts_col], errors="coerce").fillna(0)
    out = out.groupby("_name_key", as_index=False)["_pts"].sum()
    out = out.sort_values("_pts", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out.attrs["__path__"] = path
    return out


# =====================================================
# Level resolution (⭐ IMPORTANT)
# =====================================================
def resolve_level(player_row: pd.Series | None, contract_row: pd.Series | None) -> str:
    # 1) Players DB (source de vérité)
    if player_row is not None:
        lvl = str(player_row.get("Level") or "").upper().strip()
        if lvl in ("ELC", "STD"):
            return lvl

    # 2) Puckpedia (détection ELC)
    if contract_row is not None:
        t = str(contract_row.get("Type") or contract_row.get("Contract Type") or "").upper()
        if "ELC" in t or "ENTRY" in t:
            return "ELC"

    return "—"


# =====================================================
# UI rendering
# =====================================================
def render(ctx: dict) -> None:
    st.header("👤 Joueurs")
    st.caption("Recherche + fiche joueur + Level (STD / ELC) + comparatif")

    data_dir = _data_dir(ctx)
    season = _season(ctx)

    players = load_players_db(data_dir)
    contracts = load_contracts(data_dir)
    points = load_points(data_dir, season)

    if players.empty:
        st.error("hockey.players.csv introuvable ou vide")
        return

    # -------------------------
    # Recherche
    # -------------------------
    q = st.text_input("🔎 Rechercher un joueur")
    df_disp = players[["_display_name", "_name_key"]]

    if q.strip():
        k = _norm_name(q)
        df_disp = df_disp[
            df_disp["_display_name"].str.lower().str.contains(q.lower(), na=False)
            | df_disp["_name_key"].str.contains(k, na=False)
        ]

    names = df_disp["_display_name"].head(200).tolist()
    if not names:
        st.info("Aucun joueur trouvé")
        return

    sel = st.selectbox("Choisir un joueur", names)
    key = _norm_name(sel)

    prow = players.loc[players["_name_key"] == key].iloc[0]
    crow = contracts.loc[contracts["_name_key"] == key].iloc[0] if not contracts.empty and key in contracts["_name_key"].values else None

    level = resolve_level(prow, crow)

    # -------------------------
    # Fiche joueur
    # -------------------------
    st.divider()
    st.subheader("🧾 Fiche joueur")

    col1, col2 = st.columns([1, 2])
    with col1:
        nhl_id = str(prow.get("_nhl_id") or "").strip()
        if nhl_id:
            st.image(f"https://assets.nhle.com/mugs/nhl/{nhl_id}.png")
        else:
            st.info("Photo indisponible")

    with col2:
        st.markdown(f"""
**Nom** : {prow['_display_name']}  
**Position** : {prow.get('Pos','—')}  
**Équipe NHL** : {prow.get('Team','—')}  
**Pays** : {prow.get('Country','—')}  
**Level (Pool)** : **{level}**
""")

    # -------------------------
    # Classement
    # -------------------------
    if not points.empty and key in points["_name_key"].values:
        r = points.loc[points["_name_key"] == key].iloc[0]
        st.markdown(f"### 🏆 Classement\n**Points** : {r['_pts']}  \n**Rang** : {r['rank']} / {points.shape[0]}")

    # -------------------------
    # Comparatif
    # -------------------------
    st.divider()
    st.subheader("⚖️ Comparatif 2 joueurs")

    a, b = st.columns(2)
    with a:
        j1 = st.selectbox("Joueur A", players["_display_name"].head(2000), index=0)
    with b:
        j2 = st.selectbox("Joueur B", players["_display_name"].head(2000), index=1)

    for title, name in [("Joueur A", j1), ("Joueur B", j2)]:
        k = _norm_name(name)
        p = players.loc[players["_name_key"] == k].iloc[0]
        c = contracts.loc[contracts["_name_key"] == k].iloc[0] if not contracts.empty and k in contracts["_name_key"].values else None
        lvl = resolve_level(p, c)

        st.markdown(f"""
**{title}**  
Nom : {p['_display_name']}  
Position : {p.get('Pos','—')}  
Level : **{lvl}**
""")

    # -------------------------
    # Debug
    # -------------------------
    with st.expander("🧪 Debug (sources)", expanded=False):
        st.write("Players DB:", players.attrs.get("__path__"))
        st.write("Contracts:", contracts.attrs.get("__path__", "(absent)"))
        st.write("Points:", points.attrs.get("__path__", "(absent)"))
