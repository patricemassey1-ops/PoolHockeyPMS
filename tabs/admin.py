# =====================================================
# tabs/admin.py — Gestion Admin (LOCAL ONLY)
# =====================================================

import os
import shutil
import pandas as pd
import streamlit as st
from datetime import datetime

# =====================================================
# CONSTANTES & PATHS
# =====================================================

DATA_DIR = "data"
PLAYERS_DB_FILENAME = "hockey.players.csv"

DEFAULT_CAP_GC = 88_000_000
DEFAULT_CAP_CE = 15_000_000


def equipes_path(data_dir: str, season_lbl: str) -> str:
    season_lbl = str(season_lbl or "").strip() or "2025-2026"
    return os.path.join(data_dir, f"equipes_joueurs_{season_lbl}.csv")


def admin_log_path(data_dir: str, season_lbl: str) -> str:
    season_lbl = str(season_lbl or "").strip() or "2025-2026"
    return os.path.join(data_dir, f"admin_history_{season_lbl}.csv")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# =====================================================
# LOADERS
# =====================================================

@st.cache_data(show_spinner=False)
def load_players_db(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "Joueur" not in df.columns and "Player" in df.columns:
        df = df.rename(columns={"Player": "Joueur"})
    return df


@st.cache_data(show_spinner=False)
def load_equipes(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_admin_log(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "action", "details"])
    return pd.read_csv(path, low_memory=False)


def append_admin_log(path: str, action: str, details: str):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "details": details,
    }
    df = load_admin_log(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    st.cache_data.clear()


# =====================================================
# UI SECTIONS
# =====================================================

def section_test_drive_write():
    st.subheader("🧪 Test drive write (LOCAL)")
    ensure_data_dir()

    if st.button("Tester écriture locale"):
        test_path = os.path.join(DATA_DIR, "_test_write.txt")
        with open(test_path, "w") as f:
            f.write("OK")
        st.success(f"Écriture OK → {test_path}")


def section_historique_admin(log_path: str):
    st.subheader("📜 Historique admin")
    log = load_admin_log(log_path)
    if log.empty:
        st.info("Aucune action enregistrée.")
    else:
        st.dataframe(log, use_container_width=True)


def section_plafond_salarial():
    st.subheader("💰 Plafond salarial (LOCAL)")
    col1, col2 = st.columns(2)

    with col1:
        cap_gc = st.number_input(
            "Plafond Grand Club ($)",
            min_value=0,
            value=DEFAULT_CAP_GC,
            step=500_000,
        )
    with col2:
        cap_ce = st.number_input(
            "Plafond Club École ($)",
            min_value=0,
            value=DEFAULT_CAP_CE,
            step=500_000,
        )

    st.info("⚠️ Ces valeurs sont indicatives (pas encore persistées).")


def section_ajouter_joueur(players_db: pd.DataFrame, equipes_df: pd.DataFrame):
    st.subheader("➕ Ajouter joueur")

    if players_db.empty:
        st.warning("hockey.players.csv introuvable.")
        return

    joueur = st.selectbox("Joueur", sorted(players_db["Joueur"].dropna().unique()))
    equipe = st.text_input("Équipe")
    slot = st.selectbox("Slot", ["GC", "CE", "Banc", "IR"])

    if st.button("Ajouter"):
        row = players_db[players_db["Joueur"] == joueur].iloc[0].to_dict()
        row["Equipe"] = equipe
        row["Slot"] = slot

        equipes_df = pd.concat([equipes_df, pd.DataFrame([row])], ignore_index=True)
        st.success(f"{joueur} ajouté à {equipe} ({slot})")
        return equipes_df

    return equipes_df


def section_deplacer_gc_ce(equipes_df: pd.DataFrame):
    st.subheader("🔁 Déplacer GC ↔ CE")

    if equipes_df.empty:
        st.info("Aucun joueur chargé.")
        return equipes_df

    joueur = st.selectbox("Joueur", equipes_df["Joueur"].unique())
    nouveau_slot = st.selectbox("Nouveau slot", ["GC", "CE"])

    if st.button("Déplacer"):
        equipes_df.loc[equipes_df["Joueur"] == joueur, "Slot"] = nouveau_slot
        st.success(f"{joueur} déplacé vers {nouveau_slot}")

    return equipes_df


def section_retirer_joueur(equipes_df: pd.DataFrame):
    st.subheader("❌ Retirer joueur")

    if equipes_df.empty:
        st.info("Aucun joueur chargé.")
        return equipes_df

    joueur = st.selectbox("Joueur à retirer", equipes_df["Joueur"].unique())

    if st.button("Retirer"):
        equipes_df = equipes_df[equipes_df["Joueur"] != joueur]
        st.success(f"{joueur} retiré")

    return equipes_df


def section_preview_local(equipes_df: pd.DataFrame):
    st.subheader("👀 Preview Local")
    if equipes_df.empty:
        st.info("Aucune donnée.")
    else:
        st.dataframe(equipes_df, use_container_width=True)


def section_import_local(equipes_path_str: str):
    st.subheader("📥 Import Local CSV")

    uploaded = st.file_uploader("Importer equipes_joueurs_*.csv", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded, low_memory=False)
        df.to_csv(equipes_path_str, index=False)
        st.success(f"Import réussi → {equipes_path_str}")
        st.cache_data.clear()


# =====================================================
# RENDER
# =====================================================

def render(ctx: dict):
    """
    ctx attendu:
    - ctx["season_lbl"]
    """

    season_lbl = ctx.get("season_lbl", "2025-2026")

    ensure_data_dir()

    e_path = equipes_path(DATA_DIR, season_lbl)
    log_path = admin_log_path(DATA_DIR, season_lbl)

    players_db = load_players_db(os.path.join(DATA_DIR, PLAYERS_DB_FILENAME))
    equipes_df = load_equipes(e_path)

    st.header("🛠️ Gestion Admin")

    # ORDRE EXACT DEMANDÉ
    section_test_drive_write()
    section_historique_admin(log_path)
    section_plafond_salarial()

    equipes_df = section_ajouter_joueur(players_db, equipes_df)
    equipes_df = section_deplacer_gc_ce(equipes_df)
    equipes_df = section_retirer_joueur(equipes_df)

    section_preview_local(equipes_df)
    section_import_local(e_path)

    # Sauvegarde finale si modifié
    if isinstance(equipes_df, pd.DataFrame) and not equipes_df.empty:
        equipes_df.to_csv(e_path, index=False)
