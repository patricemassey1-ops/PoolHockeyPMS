import streamlit as st

def render(ctx: dict) -> None:
    st.header("🕘 Historique")
    st.caption("Tab module: tabs/historique.py")
    st.code(ctx)
