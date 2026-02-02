import streamlit as st

def apply_theme() -> None:
    """
    Theme is driven by app.py (single injection). This file keeps only utility classes.
    """
    if st.session_state.get("_theme_utils_injected"):
        return

    st.markdown(
        r"""
        <style>
          .nowrap { white-space: nowrap; }
          .right { text-align: right; }
          .muted { color: rgba(180,190,210,0.75); font-size: 0.92rem; }
          .small { font-size: 0.92rem; }
          .card { padding: 10px 12px; border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; background: rgba(255,255,255,0.04); }
          div.stButton > button { padding: 0.40rem 0.70rem; border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_theme_utils_injected"] = True
