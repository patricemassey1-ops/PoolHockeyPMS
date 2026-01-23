import streamlit as st

def render(ctx: dict) -> None:
    st.header("🏠 Home")
    st.caption("Tab module: tabs/home.py")
    st.code(ctx)
