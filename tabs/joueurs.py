# tabs/joueurs.py
from __future__ import annotations

import os
import pandas as pd
import streamlit as st


# ============================================================
# HELPERS
# ============================================================
def _safe_col(df: pd.DataFrame, col: str, default="") -> pd.Series:
    """
    Retourne une Series TOUJOURS safe.
    - si colonne existe → fillna
    - sinon → Series remplie avec default
    """
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def _safe_int_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return pd.Series([0] * len(df), index=df.index)


# ============================================================
# MAIN
# ============================================================
def render(ctx: dict) -> None:
    st.subheader("👤 Joueurs")

    DATA_DIR = str(ctx.get("DATA_DIR") or "data")
    season = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    equipes_path = os.path.join(DATA_DIR, f"equipes_joueurs_{season}.csv")

    if not os.path.exists(equipes_path):
        st.info("Aucun fichier équipes trouvé. Importe d’abord les équipes.")
        return

    try:
        df = pd.read_csv(equipes_path)
    except Exception as e:
        st.error("Impossible de lire le fichier équipes.")
        st.exception(e)
        return

    if df.empty:
        st.info("Aucun joueur dans le fichier équipes.")
        return

    # ========================================================
    # NORMALISATION SAFE (AUCUN .fillna SUR STRING)
    # ========================================================
    df["Propriétaire"] = _safe_col(df, "Propriétaire")
    df["Joueur"] = (
        _safe_col(df, "Joueur")
        if "Joueur" in df.columns
        else _safe_col(df, "Player")
    ).astype(str)

    df["Pos"] = _safe_col(df, "Pos")
    df["Equipe"] = _safe_col(df, "Equipe")
    df["Slot"] = _safe_col(df, "Slot", "Actif")
    df["Level"] = _safe_col(df, "Level", "")
    df["Salaire"] = _safe_int_col(df, "Salaire")

    # Colonnes optionnelles (ne doivent JAMAIS casser)
    df["Statut"] = _safe_col(df, "Statut")
    df["IR Date"] = _safe_col(df, "IR Date")

    # ========================================================
    # FILTRES UI
    # ========================================================
    teams = sorted(df["Propriétaire"].dropna().unique().tolist())
    team = st.selectbox("Équipe", teams)

    col1, col2, col3 = st.columns(3)
    with col1:
        slot_filter = st.multiselect(
            "Slot",
            ["Actif", "Banc", "Mineur", "IR"],
            default=["Actif", "Banc", "Mineur", "IR"],
        )
    with col2:
        level_filter = st.multiselect(
            "Level",
            sorted(df["Level"].dropna().unique().tolist()),
            default=sorted(df["Level"].dropna().unique().tolist()),
        )
    with col3:
        pos_filter = st.multiselect(
            "Position",
            sorted(df["Pos"].dropna().unique().tolist()),
            default=sorted(df["Pos"].dropna().unique().tolist()),
        )

    # ========================================================
    # APPLY FILTERS
    # ========================================================
    view = df[
        (df["Propriétaire"] == team)
        & (df["Slot"].isin(slot_filter))
        & (df["Level"].isin(level_filter))
        & (df["Pos"].isin(pos_filter))
    ].copy()

    # ========================================================
    # AFFICHAGE
    # ========================================================
    st.caption(f"{len(view)} joueurs")

    show_cols = [
        "Joueur",
        "Pos",
        "Equipe",
        "Salaire",
        "Level",
        "Slot",
        "Statut",
        "IR Date",
    ]
    show_cols = [c for c in show_cols if c in view.columns]

    st.dataframe(
        view[show_cols].sort_values(["Slot", "Pos", "Joueur"]),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # STATS RAPIDES
    # ========================================================
    total_salary = int(view["Salaire"].sum())
    st.metric("💰 Masse salariale", f"{total_salary:,} $".replace(",", " "))

