# tabs/joueurs.py
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from services.roster_common import load_roster, normalize_roster_df, players_db_path


def render(ctx: dict) -> None:
    st.subheader("👤 Joueurs")

    data_dir = str(ctx.get("DATA_DIR") or "data")
    season = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"

    df, roster_path = load_roster(data_dir, season)
    if df is None or df.empty:
        st.warning("Roster CSV manquant ou vide.")
        st.caption(f"Roster: {roster_path}")
        return

    # Normalise une dernière fois avec Level auto (si players DB dispo)
    pdb = players_db_path(data_dir)
    df = normalize_roster_df(df, owner=None, players_db_path=pdb)

    teams = sorted([t for t in df["Propriétaire"].dropna().astype(str).str.strip().unique().tolist() if t])
    if not teams:
        st.warning("Aucune équipe détectée dans le roster.")
        return

    team = st.selectbox("Équipe", teams, key="players_team")

    # Filtres avancés
    colA, colB, colC, colD = st.columns([2, 2, 2, 2])
    with colA:
        slot_filter = st.multiselect("Slot", ["Actif", "Banc", "IR", "Mineur"], default=["Actif","Banc","IR","Mineur"])
    with colB:
        level_vals = sorted([v for v in df["Level"].dropna().astype(str).str.strip().unique().tolist() if v])
        level_filter = st.multiselect("Level", level_vals, default=level_vals if level_vals else [])
    with colC:
        pos_vals = sorted([v for v in df["Pos"].dropna().astype(str).str.strip().unique().tolist() if v])
        pos_filter = st.multiselect("Pos", pos_vals, default=pos_vals if pos_vals else [])
    with colD:
        q = st.text_input("Recherche joueur", value="", placeholder="ex: Marner")

    view = df[df["Propriétaire"].astype(str).str.strip().eq(team)].copy()
    if slot_filter:
        view = view[view["Slot"].isin(slot_filter)]
    if level_filter:
        view = view[view["Level"].astype(str).isin(level_filter)]
    if pos_filter:
        view = view[view["Pos"].astype(str).isin(pos_filter)]
    if q.strip():
        view = view[view["Joueur"].astype(str).str.contains(q.strip(), case=False, na=False)]

    st.caption(f"{len(view)} joueurs — {team}")
    cols = [c for c in ["Joueur","Pos","Equipe","Salaire","Level","Slot","Statut","IR Date"] if c in view.columns]
    st.dataframe(view[cols].sort_values(["Slot","Pos","Joueur"]), use_container_width=True, hide_index=True)

    total = int(view["Salaire"].fillna(0).sum()) if "Salaire" in view.columns else 0
    st.metric("💰 Masse salariale (filtre)", f"{total:,} $".replace(",", " "))
