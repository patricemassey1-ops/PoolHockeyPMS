
# tabs/admin.py
import os
import io
import re
import csv
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import streamlit as st
from pandas.errors import ParserError

# ============================================================
# STATE
# ============================================================
def _init_state():
    st.session_state.setdefault("admin_prepared", [])
    st.session_state.setdefault("admin_last_parse_report", [])


# ============================================================
# HELPERS
# ============================================================
def infer_team_from_filename(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0].strip()


def equipes_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")


def _backup_dir(data_dir: str, season: str) -> str:
    d = os.path.join(data_dir, "admin_backups", season)
    os.makedirs(d, exist_ok=True)
    return d


def backup_team(df_all: pd.DataFrame, data_dir: str, season: str, team: str):
    if df_all.empty or "Propriétaire" not in df_all.columns:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", team)
    path = os.path.join(_backup_dir(data_dir, season), f"{safe}_{ts}.csv")
    df_all[df_all["Propriétaire"] == team].to_csv(path, index=False)


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .fillna("0")
        .astype(int, errors="ignore")
    )


def _derive_slot_from_status(v: str) -> str:
    v = str(v).upper()
    if "IR" in v:
        return "IR"
    if "MIN" in v or "AHL" in v:
        return "Mineur"
    if "BN" in v or "BENCH" in v:
        return "Banc"
    return "Actif"


# ============================================================
# NORMALISATION (PRIORITÉ PLAYER)
# ============================================================
def normalize_equipes_df(df: pd.DataFrame, owner: str) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # JOUEUR (priorité stricte)
    if "Player" in df.columns:
        df.rename(columns={"Player": "Joueur"}, inplace=True)
    elif "Name" in df.columns:
        df.rename(columns={"Name": "Joueur"}, inplace=True)
    elif "Skaters" in df.columns:
        df.rename(columns={"Skaters": "Joueur"}, inplace=True)
    else:
        df["Joueur"] = ""

    # Autres mappings
    if "Pos" not in df.columns:
        for c in ["Position"]:
            if c in df.columns:
                df.rename(columns={c: "Pos"}, inplace=True)
    if "Salaire" not in df.columns:
        for c in ["Salary", "Cap Hit"]:
            if c in df.columns:
                df.rename(columns={c: "Salaire"}, inplace=True)

    if "Statut" not in df.columns:
        for c in ["Status"]:
            if c in df.columns:
                df.rename(columns={c: "Statut"}, inplace=True)

    if "Pos" not in df.columns:
        df["Pos"] = ""
    if "Salaire" not in df.columns:
        df["Salaire"] = 0

    df["Salaire"] = _coerce_int_series(df["Salaire"])

    if "Slot" not in df.columns:
        if "Statut" in df.columns:
            df["Slot"] = df["Statut"].apply(_derive_slot_from_status)
        else:
            df["Slot"] = "Actif"

    df["Propriétaire"] = owner

    order = ["Propriétaire", "Joueur", "Pos", "Salaire", "Slot"]
    rest = [c for c in df.columns if c not in order]
    return df[order + rest]


# ============================================================
# MAIN RENDER
# ============================================================
def render(ctx: dict):
    _init_state()

    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    DATA_DIR = ctx.get("DATA_DIR", "data")
    season = ctx.get("season", "2025-2026")
    os.makedirs(DATA_DIR, exist_ok=True)
    data_path = equipes_path(DATA_DIR, season)

    st.subheader("🛠️ Gestion Admin")

    # =====================================================
    # IMPORT DEPUIS /data
    # =====================================================
    with st.expander("📂 Import équipes depuis /data", expanded=True):
        files = sorted(
            f for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".csv")
            and not f.startswith("equipes_joueurs_")
            and f not in ("hockey.players.csv", "PuckPedia2025_26.csv")
        )

        if not files:
            st.info("Aucun CSV équipe trouvé dans /data.")
        else:
            selected = st.multiselect("Fichiers équipes", files, default=files)
            if st.button("🧼 Préparer depuis /data"):
                st.session_state["admin_prepared"] = []
                for fname in selected:
                    df_raw = pd.read_csv(os.path.join(DATA_DIR, fname))
                    team = infer_team_from_filename(fname)
                    df_norm = normalize_equipes_df(df_raw, team)
                    st.session_state["admin_prepared"].append({
                        "filename": fname,
                        "team": team,
                        "df": df_norm
                    })
                st.success("CSV préparés et normalisés.")

    # =====================================================
    # PREVIEW + IMPORT
    # =====================================================
    if st.session_state["admin_prepared"]:
        st.markdown("### Preview & import")
        for item in st.session_state["admin_prepared"]:
            st.markdown(f"**{item['team']}**")
            st.dataframe(item["df"].head(20), use_container_width=True)

        if st.button("⬇️ Importer (remplacer équipes + backup)"):
            if os.path.exists(data_path):
                df_all = pd.read_csv(data_path)
            else:
                df_all = pd.DataFrame()

            for item in st.session_state["admin_prepared"]:
                backup_team(df_all, DATA_DIR, season, item["team"])
                df_all = df_all[df_all["Propriétaire"] != item["team"]]
                df_all = pd.concat([df_all, item["df"]], ignore_index=True)

            df_all.to_csv(data_path, index=False)
            st.success("Import terminé.")
            st.session_state["admin_prepared"] = []
            st.rerun()
