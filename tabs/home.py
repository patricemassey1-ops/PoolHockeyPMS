# tabs/home.py
from __future__ import annotations

import os
import streamlit as st
from services.storage import path_pool_logo, path_team_logo

POOL_TEAMS = ["Home", "GM", "Joueurs", "Alignement", "Transactions", "Historique", "Classement"]


def render(ctx: dict) -> None:
    # 🍎 Apple glass header + logo_pool (gros, centré)
    st.markdown("""<style>
    .pms-home-top{display:flex;justify-content:center;margin: 6px 0 18px 0;}
    .pms-home-top img{border-radius:18px;box-shadow:0 18px 40px rgba(0,0,0,0.22);}
    </style>""", unsafe_allow_html=True)
    try:
        lp = path_pool_logo('logo_pool.png') or os.path.join(data_dir, 'logo_pool.png')
        if lp and os.path.exists(lp):
            st.markdown("<div class='pms-home-top'>", unsafe_allow_html=True)
            st.image(lp, width=520)
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        pass

    # Home should NOT call st.set_page_config (only app.py)
    data_dir = str(ctx.get("DATA_DIR") or os.getenv("DATA_DIR") or "data")

    # Small top spacing so the banner/logo can sit above the title
    st.markdown("""<style>
    .block-container{padding-top:1.4rem !important;}
    .pms-home-logo img{border-radius:18px; box-shadow:0 18px 40px rgba(0,0,0,.28);}
    </style>""", unsafe_allow_html=True)

    # ✅ Pool logo — centered & bigger (pro)
    logo = path_pool_logo()
    if not logo:
        cand = os.path.join(data_dir, "logo_pool.png")
        logo = cand if os.path.exists(cand) else None

    if logo:
        c1, c2, c3 = st.columns([1, 3, 1], gap="small")
        with c2:
            st.markdown("<div class='pms-home-logo'>", unsafe_allow_html=True)
            st.image(str(logo), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.header("🏠 Home")
    st.caption("Choisis ton équipe — le reste de l'app suit automatiquement.")

    # Teams list comes from ctx if provided
    owners = ctx.get("owners")
    if not isinstance(owners, list) or not owners:
        owners = ["Canadiens", "Whalers", "Nordiques", "Red_Wings", "Predateurs", "Cracheurs"]

    # Session state
    if "owner" not in st.session_state or st.session_state.get("owner") not in owners:
        st.session_state["owner"] = owners[0]

    st.subheader("🪄 Sélection d'équipe")

    colA, colB = st.columns([4, 1], gap="medium")
    with colA:
        selected = st.selectbox(
            "Équipe (propriétaire)",
            options=owners,
            index=owners.index(st.session_state.get("owner")),
            key="owner_select_home",
        )
        st.session_state["owner"] = selected

    # Team logo (clean) — same block as selection
    with colB:
        # prefer assets/previews; fallback data
        # file naming: {Team}_Logo.png variants
        candidates = [
            f"{selected}_Logo.png",
            f"{selected}E_Logo.png",
            f"{selected}e_Logo.png",
            f"{selected}_Logo-2.png",
        ]
        p = None
        for fn in candidates:
            pp = path_team_logo(fn)
            if pp:
                p = pp
                break
        if not p:
            # fallback: try data/{team}.png or assets
            for fn in candidates:
                fp = os.path.join(data_dir, fn)
                if os.path.exists(fp):
                    p = fp
                    break
        if p:
            st.image(str(p), width=64)

    st.success(f"✅ Équipe sélectionnée: {st.session_state.get('owner')}")

