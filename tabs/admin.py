
import os
import io
import re
from datetime import datetime
import pandas as pd
import streamlit as st

# ============================================================
# ADMIN TAB — STABLE + MULTI IMPORT + AUTO-ASSIGN + ROLLBACK
# ============================================================

# -----------------------------
# Helpers
# -----------------------------
def infer_owner_from_filename(filename: str) -> str:
    if not filename:
        return ""
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.strip()


def backup_team(df_all: pd.DataFrame, data_dir: str, season: str, owner: str) -> None:
    if df_all is None or df_all.empty or not owner:
        return
    bdir = os.path.join(data_dir, "backups_admin", season)
    os.makedirs(bdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(bdir, f"{owner}_{ts}.csv")
    df_all[df_all["Propriétaire"] == owner].to_csv(path, index=False)


def list_backups(data_dir: str, season: str, owner: str):
    bdir = os.path.join(data_dir, "backups_admin", season)
    if not os.path.isdir(bdir):
        return []
    return sorted(
        [os.path.join(bdir, f) for f in os.listdir(bdir) if f.startswith(owner + "_")],
        reverse=True,
    )


# -----------------------------
# MAIN RENDER
# -----------------------------
def render(ctx: dict) -> None:
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    DATA_DIR = str(ctx.get("DATA_DIR") or "Data")
    os.makedirs(DATA_DIR, exist_ok=True)

    season = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    data_path = os.path.join(DATA_DIR, f"equipes_joueurs_{season}.csv")

    st.subheader("🛠️ Gestion Admin")

    # =====================================================
    # 📥 Import multi CSV
    # =====================================================
    with st.expander("📥 Import multi CSV équipes", expanded=True):
        files = st.file_uploader(
            "Uploader un ou plusieurs CSV (ex: Whalers.csv, Nordiques.csv)",
            type=["csv"],
            accept_multiple_files=True,
        )

        if files:
            if st.button("🧼 Préparer"):
                prepared = []
                for f in files:
                    try:
                        try:
                            df = pd.read_csv(f)
                        except Exception:
                            f.seek(0)
                            df = pd.read_csv(f, encoding="latin-1")

                        owner = infer_owner_from_filename(f.name)
                        df["Propriétaire"] = owner
                        prepared.append((owner, df))

                        st.success(f"{f.name} → équipe '{owner}' ({len(df)} lignes)")
                        st.dataframe(df.head(10), use_container_width=True)

                    except Exception as e:
                        st.error(f"Erreur lecture {f.name}")
                        st.exception(e)

                if prepared and st.button("⬇️ Importer"):
                    if os.path.exists(data_path):
                        df_all = pd.read_csv(data_path)
                    else:
                        df_all = pd.DataFrame()

                    for owner, df in prepared:
                        backup_team(df_all, DATA_DIR, season, owner)
                        df_all = df_all[df_all.get("Propriétaire") != owner]
                        df_all = pd.concat([df_all, df], ignore_index=True)

                    df_all.to_csv(data_path, index=False)
                    st.success("Import terminé + backups créés")
                    st.rerun()

    # =====================================================
    # ↩️ Rollback
    # =====================================================
    with st.expander("↩️ Rollback par équipe", expanded=False):
        if os.path.exists(data_path):
            df_all = pd.read_csv(data_path)
            owners = sorted(df_all["Propriétaire"].dropna().unique().tolist())
        else:
            owners = []

        if owners:
            owner = st.selectbox("Équipe", owners)
            backups = list_backups(DATA_DIR, season, owner)

            if backups:
                pick = st.selectbox("Backup", backups)
                if st.button("↩️ Restaurer"):
                    df_all = df_all[df_all["Propriétaire"] != owner]
                    df_restore = pd.read_csv(pick)
                    df_all = pd.concat([df_all, df_restore], ignore_index=True)
                    df_all.to_csv(data_path, index=False)
                    st.success("Rollback effectué")
                    st.rerun()
            else:
                st.info("Aucun backup pour cette équipe.")
        else:
            st.info("Aucune équipe détectée.")
