# services/auth.py
from __future__ import annotations

import time
import hmac
import streamlit as st


def require_password() -> None:
    """
    Gate simple + lockout léger.
    - st.session_state["auth_ok"] = True quand bon
    - après 5 essais, cooldown 30s
    """
    if st.session_state.get("auth_ok"):
        return

    pwd = str(st.secrets.get("app_password", "") or "")
    if not pwd:
        # si pas de mot de passe défini, on ne bloque pas
        st.session_state["auth_ok"] = True
        return

    st.title("🔒 Accès sécurisé")
    st.caption("Entre le mot de passe pour accéder à l’application.")

    tries = int(st.session_state.get("auth_tries", 0))
    locked_until = float(st.session_state.get("auth_locked_until", 0.0))
    now = time.time()

    if now < locked_until:
        wait = int(locked_until - now)
        st.error(f"Trop d’essais. Réessaie dans {wait}s.")
        st.stop()

    entered = st.text_input("Mot de passe", type="password", key="auth_pwd")

    col1, col2 = st.columns([1, 2])
    with col1:
        go = st.button("✅ Entrer", type="primary")
    with col2:
        st.caption("Astuce: ajoute `app_password` dans Secrets Streamlit Cloud.")

    if go:
        ok = hmac.compare_digest(str(entered or ""), pwd)
        if ok:
            st.session_state["auth_ok"] = True
            st.session_state["auth_pwd"] = ""
            st.success("Accès autorisé ✅")
            st.rerun()
        else:
            tries += 1
            st.session_state["auth_tries"] = tries
            st.session_state["auth_pwd"] = ""
            st.error("Mot de passe incorrect.")

            if tries >= 5:
                st.session_state["auth_locked_until"] = now + 30
                st.session_state["auth_tries"] = 0

    st.stop()
