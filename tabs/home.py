# tabs/home.py
import streamlit as st
from services.storage import path_pool_logo, path_team_logo


def render(ctx: dict) -> None:
    st.header("🏠 Home")
    st.caption("Home reste clean — aucun bloc Admin ici.")

    # ----------------------------
    # Pool logo
    # ----------------------------
    pool_logo = path_pool_logo()
    if pool_logo:
        try:
            st.image(pool_logo, width=120)
        except Exception:
            pass

    # ----------------------------
    # Source de vérité: selected_owner
    # ----------------------------
    owners = ctx.get("owners")
    if not isinstance(owners, list) or not owners:
        # fallback safe
        owners = ["Canadiens", "Cracheurs", "Nordiques", "Prédateurs", "Red Wings", "Whalers"]

    if "selected_owner" not in st.session_state:
        st.session_state["selected_owner"] = owners[0]

    # si valeur invalide (ex: liste a changé)
    if st.session_state["selected_owner"] not in owners:
        st.session_state["selected_owner"] = owners[0]

    st.subheader("🏒 Sélection d'équipe")
    c1, c2 = st.columns([1.2, 2.2], vertical_alignment="center")

    with c1:
        owner = st.selectbox(
            "Équipe (propriétaire)",
            owners,
            key="selected_owner",
        )

    with c2:
        # Team logo (assets/previews puis data)
        fn_candidates = [
            f"{owner}_Logo.png",
            f"{owner}E_Logo.png",
            f"{owner}_logo.png",
            f"{owner}.png",
            f"{owner.replace(' ', '_')}_Logo.png",
            f"{owner.replace(' ', '_')}E_Logo.png",
        ]
        shown = False
        for fn in fn_candidates:
            p = path_team_logo(fn)
            if p:
                try:
                    st.image(p, width=130)
                    shown = True
                    break
                except Exception:
                    pass
        if not shown:
            st.caption("Logo équipe introuvable (ok).")

    st.success(f"✅ Équipe sélectionnée: **{owner}**")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    st.divider()

    # ----------------------------
    # Debug logos (optionnel)
    # ----------------------------
    with st.expander("🔎 Debug — chemins de logos (optionnel)", expanded=False):
        st.caption("Résolution: assets/previews puis data.")
        for fn in ["Whalers_Logo.png","Nordiques_Logo.png","Predateurs_Logo.png","Cracheurs_Logo.png","Red_Wings_Logo.png"]:
            p = path_team_logo(fn)
            if p:
                st.write(f"- {fn} → {p}")
            else:
                st.write(f"- {fn} → (introuvable)")
