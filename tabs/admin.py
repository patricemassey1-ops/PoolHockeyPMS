import os
import io
import pandas as pd
import streamlit as st

# ============================================================
# ADMIN TAB — STABLE BASE
# ============================================================

def render(ctx: dict) -> None:
    # 🔒 Sécurité admin
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    DATA_DIR = str(ctx.get("DATA_DIR") or "Data")
    os.makedirs(DATA_DIR, exist_ok=True)

    season_lbl = str(ctx.get("season") or "2025-2026").strip()
    if not season_lbl:
        season_lbl = "2025-2026"

    st.subheader("🛠️ Gestion Admin")

    # =====================================================
    # 🔐 OAuth Drive (placeholder safe)
    # =====================================================
    with st.expander("🔐 Connexion Google Drive (OAuth)", expanded=False):
        st.info("OAuth Drive désactivé temporairement (base stable).")

    # =====================================================
    # 📥 Import CSV équipes (simple, stable)
    # =====================================================
    with st.expander("📥 Import CSV équipes", expanded=True):
        up = st.file_uploader(
            "Uploader un fichier CSV équipes",
            type=["csv"],
            key="admin_upload_csv",
        )

        if up is not None:
            try:
                try:
                    df = pd.read_csv(up)
                except Exception:
                    up.seek(0)
                    df = pd.read_csv(up, encoding="latin-1")

                st.success(f"CSV chargé ({len(df)} lignes)")
                st.dataframe(df.head(50), use_container_width=True)

                if st.button("💾 Sauvegarder (test)"):
                    path = os.path.join(DATA_DIR, f"equipes_joueurs_{season_lbl}.csv")
                    df.to_csv(path, index=False)
                    st.success(f"Fichier sauvegardé : {path}")

            except Exception as e:
                st.error("Erreur lors de la lecture du CSV")
                st.exception(e)
