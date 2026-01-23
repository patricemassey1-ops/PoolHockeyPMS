import streamlit as st

THEME_CSS = r'''
<style>
.nowrap { white-space: nowrap; }
.right { text-align: right; }
.muted { color: rgba(120,120,120,0.95); font-size: 0.90rem; }
.small { font-size: 0.92rem; }
.card { padding: 10px 12px; border: 1px solid rgba(120,120,120,0.25); border-radius: 14px; }
div.stButton > button { padding: 0.35rem 0.6rem; border-radius: 10px; }
</style>
'''

def apply_theme() -> None:
    # One single injection per run
    if st.session_state.get("_theme_injected"):
        return
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.session_state["_theme_injected"] = True
