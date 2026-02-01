# tabs/home.py
import streamlit as st
from services.storage import path_pool_logo, path_team_logo
import json
import os


def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or os.getenv("DATA_DIR") or "data")


def _load_season_state(data_dir: str) -> dict:
    """Read data/season_state.json if present (dummy-proof)."""
    p = os.path.join(data_dir, "season_state.json")
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


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
        owners = ["Canadiens", "Cracheurs", "Nordiques", "Predateurs", "Red Wings", "Whalers"]

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
    # ----------------------------
    # 🚨 Alerte saison (Whalers seulement)
    # ----------------------------
    data_dir = _data_dir(ctx)
    ss = _load_season_state(data_dir)
    needs = bool(ss.get("needs_master_rebuild"))
    if owner == "Whalers" and needs:
        cur = str(ss.get("current_season") or st.session_state.get("season") or "").strip() or "nouvelle saison"
        st.warning(
            f"⚠️ Nouvelle saison détectée (**{cur}**) — tu dois reconstruire le master.",
            icon="⚠️",
        )
        st.markdown("👉 **Clique ici :** Admin → **4️⃣ Master + Audit** → bouton rouge **Construire Master + Audit**.")
        if st.button("🛠️ J'ai compris — je vais dans Admin (Étape 4)", use_container_width=True, key="home_go_admin_step4"):
            # On ne peut pas forcer la sélection d’un onglet Streamlit, mais on garde un flag pour que l'Admin affiche une bannière.
            st.session_state["admin_hint_step"] = 4
            st.success("✅ OK. Va maintenant dans l’onglet **Admin** puis clique **4️⃣ Master + Audit**.")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    st.divider()

    # ----------------------------
    # Debug logos (optionnel)
    # ----------------------------
    with st.expander("🔎 Debug — chemins de logos (optionnel)", expanded=False):
        st.caption("Résolution: assets/previews puis data.")
        for fn in ["Whalers_Logo.png","Nordiques_Logo.png","Predateurs_Logo.png","Cracheurs_Logo.png","Canadiens_Logo.png","Red_Wings_Logo.png"]:
            p = path_team_logo(fn)
            if p:
                st.write(f"- {fn} → {p}")
            else:
                st.write(f"- {fn} → (introuvable)")

