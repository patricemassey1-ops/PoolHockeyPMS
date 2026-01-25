import os
import pandas as pd
import streamlit as st
from datetime import datetime

# ============================================================
# STATE INIT
# ============================================================
def _init_admin_state():
    if "admin_prepared" not in st.session_state:
        st.session_state["admin_prepared"] = []

# ============================================================
# HELPERS
# ============================================================
def infer_team_from_filename(filename: str) -> str:
    if not filename:
        return ""
    return os.path.splitext(os.path.basename(filename))[0].strip()


def equipes_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")


def backup_team(df_all: pd.DataFrame, data_dir: str, season: str, team: str):
    if df_all is None or df_all.empty or not team:
        return
    bdir = os.path.join(data_dir, "admin_backups", season)
    os.makedirs(bdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(bdir, f"{team}_{ts}.csv")
    df_all[df_all["Propriétaire"] == team].to_csv(path, index=False)


def list_backups(data_dir: str, season: str, team: str):
    bdir = os.path.join(data_dir, "admin_backups", season)
    if not os.path.isdir(bdir):
        return []
    return sorted(
        [os.path.join(bdir, f) for f in os.listdir(bdir) if f.startswith(team + "_")],
        reverse=True,
    )

# ============================================================
# MAIN RENDER
# ============================================================
def render(ctx: dict) -> None:
    _init_admin_state()

    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    DATA_DIR = str(ctx.get("DATA_DIR") or "Data")
    os.makedirs(DATA_DIR, exist_ok=True)

    season = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    data_path = equipes_path(DATA_DIR, season)

    st.subheader("🛠️ Gestion Admin")

    # =====================================================
    # 📥 IMPORT MULTI CSV
    # =====================================================
    with st.expander("📥 Import multi CSV équipes", expanded=True):

        files = st.file_uploader(
            "Uploader un ou plusieurs CSV (ex: Whalers.csv, Nordiques.csv)",
            type=["csv"],
            accept_multiple_files=True,
        )

        # ── ÉTAPE 1 : PRÉPARATION
        if files and st.button("🧼 Préparer les fichiers", use_container_width=True):
            prepared = []

            for f in files:
                try:
                    try:
                        df = pd.read_csv(f)
                    except Exception:
                        f.seek(0)
                        df = pd.read_csv(f, encoding="latin-1")

                    team = infer_team_from_filename(f.name)

                    prepared.append({
                        "filename": f.name,
                        "team": team,
                        "df": df,
                    })

                except Exception as e:
                    st.error(f"Erreur lecture {f.name}")
                    st.exception(e)

            if prepared:
                st.session_state["admin_prepared"] = prepared
                st.success("Fichiers préparés. Vérifie les équipes puis importe.")

        # ── ÉTAPE 2 : ATTRIBUTION + IMPORT
        if st.session_state["admin_prepared"]:
            st.markdown("### Attribution des équipes")

            for i, item in enumerate(st.session_state["admin_prepared"]):
                item["team"] = st.text_input(
                    f"Équipe pour {item['filename']}",
                    value=item["team"],
                    key=f"admin_team_{i}",
                )
                st.caption(f"{len(item['df'])} lignes détectées")

            if st.button("⬇️ Importer les équipes", use_container_width=True):
                if os.path.exists(data_path):
                    df_all = pd.read_csv(data_path)
                else:
                    df_all = pd.DataFrame()

                for item in st.session_state["admin_prepared"]:
                    team = item["team"]
                    df = item["df"].copy()
                    df["Propriétaire"] = team

                    # backup avant écrasement
                    backup_team(df_all, DATA_DIR, season, team)

                    # remove ancienne équipe
                    if "Propriétaire" in df_all.columns:
                        df_all = df_all[df_all["Propriétaire"] != team]

                    df_all = pd.concat([df_all, df], ignore_index=True)

                df_all.to_csv(data_path, index=False)
                st.session_state["admin_prepared"] = []
                st.success("Import terminé avec succès")
                st.rerun()

    # =====================================================
    # ↩️ ROLLBACK PAR ÉQUIPE
    # =====================================================
    with st.expander("↩️ Rollback par équipe", expanded=False):

        if not os.path.exists(data_path):
            st.info("Aucune équipe détectée.")
            return

        df_all = pd.read_csv(data_path)
        if "Propriétaire" not in df_all.columns:
            st.info("Colonne Propriétaire absente.")
            return

        teams = sorted(df_all["Propriétaire"].dropna().unique().tolist())

        if not teams:
            st.info("Aucune équipe détectée.")
            return

        team = st.selectbox("Équipe", teams)
        backups = list_backups(DATA_DIR, season, team)

        if not backups:
            st.info("Aucun backup pour cette équipe.")
            return

        backup = st.selectbox("Backup disponible", backups)

        if st.button("↩️ Restaurer ce backup", use_container_width=True):
            df_restore = pd.read_csv(backup)
            df_all = df_all[df_all["Propriétaire"] != team]
            df_all = pd.concat([df_all, df_restore], ignore_index=True)
            df_all.to_csv(data_path, index=False)
            st.success("Rollback effectué")
            st.rerun()
