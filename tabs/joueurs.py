import streamlit as st

def render(ctx: dict) -> None:
    st.header("👤 Joueurs")
    st.caption("Tab module: tabs/joueurs.py")
    st.code(ctx)
