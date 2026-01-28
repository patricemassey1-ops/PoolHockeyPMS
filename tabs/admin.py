# tabs/admin.py
# =====================================================
# ADMIN — Import équipes (version stable clean)
# =====================================================

from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from typing import Dict, List

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data"
DEFAULT_CAP_GC = 88_000_000
DEFAULT_CAP_CE = 10_000_000

EQUIPES_FILE_TMPL = "equipes_joueurs_{season}.csv"

# Colonnes minimales attendues après normalisation
EQUIPES_COLUMNS = [
    "ID",
    "Player",
    "Pos",
    "Team",
    "Equipe",
    "Slot",
]

# =====================================================
# HELPERS
# =====================================================

def _equipes_path(season: str) -> str:
    season = season or "2025-2026"
    return os.path.join(DATA_DIR, EQUIPES_FILE_TMPL.format(season=season))


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "PlayerName": "Player",
        "Name": "Player",
        "Position": "Pos",
        "TeamAbbrev": "Team",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    return df


def _auto_slot_from_pos(pos: str) -> str:
    if not isinstance(pos, str):
        return "Actif"
    pos = pos.upper()
    if pos in ("G", "GK"):
        return "Actif"
    return "Actif"


def normalize_team_import_df(
    df: pd.DataFrame,
    equipe: str,
) -> pd.DataFrame:
    df = _normalize_columns(df)

    if "Player" not in df.columns:
        raise ValueError("Colonne 'Player' manquante")

    if "Pos" not in df.columns:
        df["Pos"] = ""

    if "Team" not in df.columns:
        df["Team"] = ""

    df["Equipe"] = equipe
    df["Slot"] = df["Pos"].apply(_auto_slot_from_pos)

    keep = [c for c in EQUIPES_COLUMNS if c in df.columns]
    return df[keep]


# =====================================================
# RENDER
# =====================================================

def render(ctx: Dict):
    st.subheader("🛠️ Gestion Admin")

    season = ctx.get("season") or ctx.get("season_lbl") or "2025-2026"

    # -------------------------------
    # Caps (paramètres uniquement)
    # -------------------------------
    with st.expander("☁️ Paramètres de cap (GC / CE)", expanded=False):
        st.session_state.setdefault("CAP_GC", DEFAULT_CAP_GC)
        st.session_state.setdefault("CAP_CE", DEFAULT_CAP_CE)

        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "Cap GC",
                min_value=0,
                step=500_000,
                key="CAP_GC",
            )
        with c2:
            st.number_input(
                "Cap CE",
                min_value=0,
                step=500_000,
                key="CAP_CE",
            )

    # -------------------------------
    # Import local (fallback)
    # -------------------------------
    with st.expander("📥 Import local — CSV équipes", expanded=True):
        equipes_path = _equipes_path(season)
        st.caption(f"Destination locale : `{equipes_path}`")

        use_existing = st.checkbox(
            "Utiliser les fichiers déjà présents dans /data (sans upload)",
            value=True,
        )

        uploaded_files = []
        if not use_existing:
            uploaded_files = st.file_uploader(
                "Uploader des CSV (1 par équipe)",
                type=["csv"],
                accept_multiple_files=True,
            )

        data_files: List[str] = []
        if use_existing:
            if os.path.isdir(DATA_DIR):
                data_files = [
                    f for f in os.listdir(DATA_DIR)
                    if f.lower().endswith(".csv")
                ]
        else:
            data_files = [f.name for f in uploaded_files]

        if not data_files:
            st.info("Aucun fichier CSV détecté.")
            return

        st.markdown("### Attribution des fichiers → équipe")

        assignments = {}
        for fname in data_files:
            equipe_guess = os.path.splitext(fname)[0]
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{fname}**")
            with col2:
                assignments[fname] = st.text_input(
                    f"Équipe pour {fname}",
                    value=equipe_guess,
                    key=f"assign_{fname}",
                )

        # -------------------------------
        # Préparation
        # -------------------------------
        if st.button("🧪 Dry-run (voir résumé)"):
            rows_total = 0
            for fname, equipe in assignments.items():
                if use_existing:
                    path = os.path.join(DATA_DIR, fname)
                    df = _safe_read_csv(path)
                else:
                    fobj = next(f for f in uploaded_files if f.name == fname)
                    df = pd.read_csv(fobj)

                try:
                    df_n = normalize_team_import_df(df, equipe)
                    rows_total += len(df_n)
                except Exception as e:
                    st.error(f"{fname} : {e}")
                    return

            st.success(f"Dry-run OK — {rows_total} lignes prêtes à importer")

        # -------------------------------
        # Import réel
        # -------------------------------
        if st.button("⬇️ Importer tous → Local + Reload"):
            all_rows = []

            for fname, equipe in assignments.items():
                if use_existing:
                    path = os.path.join(DATA_DIR, fname)
                    df = _safe_read_csv(path)
                else:
                    fobj = next(f for f in uploaded_files if f.name == fname)
                    df = pd.read_csv(fobj)

                df_n = normalize_team_import_df(df, equipe)
                all_rows.append(df_n)

            if not all_rows:
                st.warning("Aucune donnée à importer.")
                return

            df_out = pd.concat(all_rows, ignore_index=True)

            os.makedirs(DATA_DIR, exist_ok=True)
            df_out.to_csv(equipes_path, index=False)

            st.success(f"Import terminé — {len(df_out)} lignes écrites")
            st.experimental_rerun()
