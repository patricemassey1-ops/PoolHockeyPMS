import streamlit as st

def render(ctx: dict) -> None:
    st.header("🧠 GM")
    st.caption("Tab module: tabs/gm.py")
    st.code(ctx)
