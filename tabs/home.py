import streamlit as st
from services.storage import path_pool_logo, path_team_logo

def render(ctx: dict) -> None:
    st.header("🏠 Home")
    st.caption("Home reste clean — aucun bloc Admin ici.")

    # Example logo resolution (safe)
    pool_logo = path_pool_logo()
    if pool_logo:
        try:
            st.image(pool_logo, width=120)
        except Exception:
            pass

    st.caption("Exemples de logos (assets/previews puis data):")
    for fn in ["Whalers_Logo.png","Nordiques_Logo.png","Predateurs_Logo.png","Cracheurs_Logo.png","Red_Wings_Logo.png"]:
        p = path_team_logo(fn)
        if p:
            st.write(f"- {fn} → {p}")
