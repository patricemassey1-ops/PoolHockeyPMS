
# tabs/admin.py — CLEAN, SAFE, CONTEXT-DICT COMPATIBLE
from __future__ import annotations

import os
import time
from typing import Dict, Any, List

import streamlit as st
import pandas as pd


# =====================================================
# Helpers
# =====================================================
def _get(ctx: Dict[str, Any], key: str, default=None):
    return ctx.get(key, default)


def _norm_player(s: str) -> str:
    s = str(s or "").strip()
    s = " ".join(s.split())
    return s


# =====================================================
# CORE SYNC — PUCKPEDIA → LEVEL
# =====================================================
def sync_level(players_path: str, puck_path: str) -> Dict[str, Any]:
    result = {"ok": False, "updated": 0}

    missing = []
    if not os.path.exists(players_path):
        missing.append(players_path)
    if not os.path.exists(puck_path):
        missing.append(puck_path)
    if missing:
        result["error"] = "Fichier introuvable"
        result["missing"] = missing
        return result

    pdb = pd.read_csv(players_path)
    pk = pd.read_csv(puck_path)

    # detect name columns
    name_pdb = None
    for c in ["Player", "Joueur", "Name"]:
        if c in pdb.columns:
            name_pdb = c
            break

    name_pk = None
    for c in ["Skaters", "Player", "Name"]:
        if c in pk.columns:
            name_pk = c
            break

    if not name_pdb or not name_pk:
        result["error"] = "Colonne joueur introuvable"
        result["players_cols"] = list(pdb.columns)
        result["puck_cols"] = list(pk.columns)
        return result

    if "Level" not in pk.columns or "Level" not in pdb.columns:
        result["error"] = "Colonne Level manquante"
        return result

    mp = {}
    for _, r in pk.iterrows():
        nm = _norm_player(r.get(name_pk))
        lv = str(r.get("Level") or "").upper()
        if nm and lv in ("ELC", "STD"):
            mp[nm] = lv

    updated = 0
    new_levels = []
    for _, r in pdb.iterrows():
        nm = _norm_player(r.get(name_pdb))
        new_lv = mp.get(nm)
        if new_lv:
            new_levels.append(new_lv)
            updated += 1
        else:
            new_levels.append(r.get("Level"))

    pdb["Level"] = new_levels
    pdb.to_csv(players_path, index=False)

    result["ok"] = True
    result["updated"] = updated
    return result


# =====================================================
# CORE SYNC — NHL_ID (STUB + PROGRESS)
# =====================================================
def sync_nhl_id(players_path: str, limit: int = 250):
    df = pd.read_csv(players_path)
    if "NHL_ID" not in df.columns:
        df["NHL_ID"] = ""

    missing = df[df["NHL_ID"].isna() | (df["NHL_ID"] == "")]
    total = min(len(missing), limit)

    bar = st.progress(0)
    txt = st.empty()

    for i, idx in enumerate(missing.index[:total]):
        time.sleep(0.01)  # simulate API
        df.at[idx, "NHL_ID"] = f"FAKE_{i+1}"
        bar.progress((i + 1) / total)
        txt.write(f"{i+1} / {total} NHL_ID assignés")

    df.to_csv(players_path, index=False)
    st.success(f"NHL_ID assignés: {total}")


# =====================================================
# UI
# =====================================================
def render(ctx: Dict[str, Any]):
    data_dir = _get(ctx, "DATA_DIR", "data")
    is_admin = bool(_get(ctx, "is_admin", False))

    st.title("🛠️ Gestion Admin")

    if not is_admin:
        st.warning("Accès admin requis.")
        return

    tab = st.radio("Section", ["Outils"], horizontal=True)

    if tab == "Outils":
        st.subheader("🔧 Outils")

        with st.expander("🧾 Sync PuckPedia → Level (STD/ELC)", expanded=False):
            puck = st.text_input(
                "Fichier PuckPedia",
                os.path.join(data_dir, "PuckPedia2025_26.csv"),
            )
            players = st.text_input(
                "Players DB",
                os.path.join(data_dir, "hockey.players.csv"),
            )

            if st.button("Synchroniser Level"):
                res = sync_level(players, puck)
                if res.get("ok"):
                    st.success(f"Levels mis à jour: {res.get('updated', 0)}")
                else:
                    st.error(res.get("error", "Erreur inconnue"))
                    if res.get("players_cols"):
                        st.write("Players DB colonnes:", res["players_cols"])
                    if res.get("puck_cols"):
                        st.write("PuckPedia colonnes:", res["puck_cols"])

        with st.expander("🆔 Sync NHL_ID manquants", expanded=False):
            players = st.text_input(
                "Players DB (NHL_ID)",
                os.path.join(data_dir, "hockey.players.csv"),
                key="nhl_players",
            )
            limit = st.number_input("Max par run", 1, 1000, 250)
            if st.button("Associer NHL_ID"):
                sync_nhl_id(players, limit)
