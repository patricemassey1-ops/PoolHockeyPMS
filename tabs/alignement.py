# tabs/alignement.py
from __future__ import annotations

import os
import streamlit as st
import pandas as pd

from services.roster_common import load_roster, normalize_roster_df, players_db_path, derive_scope

# -----------------------------------------------------
# Cache helpers (speed: avoid re-reading roster on every rerun)
# -----------------------------------------------------
def _file_sig(path: str) -> tuple[int, int]:
    try:
        st_ = os.stat(path)
        return (int(st_.st_mtime_ns), int(st_.st_size))
    except Exception:
        return (0, 0)

@st.cache_data(show_spinner=False)
def _load_roster_cached(data_dir: str, season: str, owner: str, roster_sig: tuple[int, int], players_sig: tuple[int, int]):
    # Keyed by owner + file signatures. load_roster reads st.session_state internally; owner arg keeps cache safe.
    return load_roster(data_dir, season)


def render(ctx: dict) -> None:
    st.subheader("🧾 Alignement")

    data_dir = str(ctx.get("DATA_DIR") or "data")
    season = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"

    owner = str(st.session_state.get('owner') or 'Canadiens')
    roster_guess = os.path.join(data_dir, f"{owner}.csv")
    players_path = str(players_db_path(data_dir))
    df, roster_path = _load_roster_cached(data_dir, season, owner, _file_sig(roster_guess), _file_sig(players_path))
    st.caption(f"Roster: {roster_path}")

    if df is None or df.empty:
        st.warning("Roster CSV manquant ou vide.")
        return

    pdb = players_db_path(data_dir)
    df = normalize_roster_df(df, owner=None, players_db_path=pdb)

    teams = sorted([t for t in df["Propriétaire"].dropna().astype(str).str.strip().unique().tolist() if t])
    if not teams:
        st.warning("Aucune équipe (Propriétaire) détectée.")
        return

    team = st.selectbox("Équipe", teams, key="al_team")

    # Filtres avancés
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        scope_filter = st.multiselect("Scope", ["GC", "CE"], default=["GC","CE"])
    with col2:
        level_vals = sorted([v for v in df["Level"].dropna().astype(str).str.strip().unique().tolist() if v])
        level_filter = st.multiselect("Level", level_vals, default=level_vals if level_vals else [])
    with col3:
        slot_filter = st.multiselect("Slot", ["Actif","Banc","IR","Mineur"], default=["Actif","Banc","IR","Mineur"])

    d = df[df["Propriétaire"].astype(str).str.strip().eq(team)].copy()
    d["Scope"] = d.apply(derive_scope, axis=1)

    if scope_filter:
        d = d[d["Scope"].isin(scope_filter)]
    if level_filter:
        d = d[d["Level"].astype(str).isin(level_filter)]
    if slot_filter:
        d = d[d["Slot"].isin(slot_filter)]

    # Grouping
    def show_block(title: str, sub: pd.DataFrame):
        st.markdown(f"### {title}")
        if sub.empty:
            st.caption("Aucun joueur.")
            return
        cols = [c for c in ["Joueur","Pos","Equipe","Salaire","Level","Scope","Slot","IR Date"] if c in sub.columns]
        st.dataframe(sub[cols].sort_values(["Scope","Pos","Joueur"]), use_container_width=True, hide_index=True)

    # split
    actifs = d[d["Slot"].eq("Actif")].copy()
    banc = d[d["Slot"].eq("Banc")].copy()
    mineur = d[d["Slot"].eq("Mineur")].copy()
    ir = d[d["Slot"].eq("IR")].copy()

    cA, cB, cC = st.columns([3, 2, 2])
    with cA:
        show_block("⭐ Actifs", actifs)
    with cB:
        show_block("🪑 Banc", banc)
        show_block("🩼 IR", ir)
    with cC:
        show_block("🧊 Mineur", mineur)

    # KPIs
    total = int(d["Salaire"].fillna(0).sum()) if "Salaire" in d.columns else 0
    gc = int(d.loc[d["Scope"].eq("GC"), "Salaire"].fillna(0).sum()) if "Salaire" in d.columns else 0
    ce = int(d.loc[d["Scope"].eq("CE"), "Salaire"].fillna(0).sum()) if "Salaire" in d.columns else 0
    st.divider()
    colx, coly, colz = st.columns(3)
    colx.metric("💰 Total", f"{total:,} $".replace(",", " "))
    coly.metric("🏒 GC", f"{gc:,} $".replace(",", " "))
    colz.metric("🧊 CE", f"{ce:,} $".replace(",", " "))
