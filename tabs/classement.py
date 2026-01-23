import streamlit as st

def render(ctx: dict) -> None:
    st.header("🏆 Classement")
    st.caption("Tab module: tabs/classement.py")
    st.code(ctx)
