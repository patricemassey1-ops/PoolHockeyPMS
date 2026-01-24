import os
import time
from datetime import datetime
import pandas as pd
import streamlit as st
from services.storage import path_transactions, safe_read_csv, safe_write_csv

TX_COLS = [
    "trade_id","timestamp","season",
    "owner_a","owner_b",
    "a_players","b_players",
    "a_picks","b_picks",
    "a_cash","b_cash",
    "status","notes"
]

def _anti_double_run_guard(tag: str, min_seconds: float = 0.8) -> bool:
    k = f"_last_run__{tag}"
    t = time.time()
    last = float(st.session_state.get(k, 0.0) or 0.0)
    if t - last < min_seconds:
        return False
    st.session_state[k] = t
    return True

def _make_trade_id() -> str:
    return "TR-" + datetime.now().strftime("%Y%m%d") + "-" + hex(int(time.time() * 1000))[-6:].upper()

def _tx_read(path: str) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=TX_COLS)
    for c in TX_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[TX_COLS].copy()

def _tx_write(path: str, df: pd.DataFrame) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        pass
    return safe_write_csv(path, df)

def render(ctx: dict) -> None:
    st.header("⚖️ Transactions")

    season = str(ctx.get("season") or "").strip() or "season"
    tx_path = path_transactions(season)
    st.caption(f"Fichier: {tx_path}")

    df_tx = _tx_read(tx_path)

    st.markdown("### ➕ Proposer une transaction")
    c1, c2 = st.columns(2)
    with c1:
        owner_a = st.text_input("Équipe A (propose)", key="tx_owner_a")
        a_players = st.text_area("Joueurs A (séparés par virgule)", key="tx_a_players", height=80)
        a_picks = st.text_input("Picks A (ex: 2026-1,2027-2)", key="tx_a_picks")
        a_cash = st.text_input("Cash A", key="tx_a_cash")
    with c2:
        owner_b = st.text_input("Équipe B", key="tx_owner_b")
        b_players = st.text_area("Joueurs B (séparés par virgule)", key="tx_b_players", height=80)
        b_picks = st.text_input("Picks B", key="tx_b_picks")
        b_cash = st.text_input("Cash B", key="tx_b_cash")

    notes = st.text_area("Notes", key="tx_notes", height=80)

    if st.button("✅ Enregistrer la proposition", type="primary"):
        if not _anti_double_run_guard("save_tx", 0.8):
            st.info("Patiente une seconde (anti double-click).")
        else:
            tid = _make_trade_id()
            new = {
                "trade_id": tid,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "season": season,
                "owner_a": owner_a,
                "owner_b": owner_b,
                "a_players": a_players,
                "b_players": b_players,
                "a_picks": a_picks,
                "b_picks": b_picks,
                "a_cash": a_cash,
                "b_cash": b_cash,
                "status": "PROPOSED",
                "notes": notes,
            }
            df_tx = pd.concat([df_tx, pd.DataFrame([new])], ignore_index=True)
            ok = _tx_write(tx_path, df_tx)
            if ok:
                st.success(f"Transaction enregistrée: {tid}")
                # clear inputs (soft)
                for k in ["tx_a_players","tx_b_players","tx_a_picks","tx_b_picks","tx_a_cash","tx_b_cash","tx_notes"]:
                    st.session_state[k] = ""
            else:
                st.error("Échec d’écriture CSV (permissions?)")

    st.divider()
    st.markdown("### 📋 Transactions enregistrées")
    if df_tx.empty:
        st.caption("Aucune transaction.")
        return

    # Display newest first
    try:
        show = df_tx.sort_values("timestamp", ascending=False)
    except Exception:
        show = df_tx

    st.dataframe(show, use_container_width=True)

    with st.expander("🧹 Admin rapide (local)", expanded=False):
        st.caption("Actions locales uniquement (pas Drive).")
        colA, colB = st.columns(2)
        with colA:
            if st.button("🗑️ Vider fichier transactions (local)"):
                empty = pd.DataFrame(columns=TX_COLS)
                if _tx_write(tx_path, empty):
                    st.success("Fichier transactions vidé.")
                else:
                    st.error("Échec.")
        with colB:
            st.caption("Tu peux aussi restaurer via Gestion Admin → Drive restore.")
