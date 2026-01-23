import streamlit as st

def render(ctx: dict) -> None:
    st.header("⚖️ Transactions")
    st.caption("Tab module: tabs/transactions.py")
    st.code(ctx)
