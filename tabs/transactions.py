# tabs/transactions.py
from __future__ import annotations
import os
import json
import uuid
import re
import unicodedata
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import streamlit as st

# --- HELPERS ---
def _data_dir(ctx: dict) -> str:
    d = str(ctx.get("DATA_DIR") or "data")
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return d

def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip()

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def _norm_player_key(name: str) -> str:
    s = str(name or "").strip()
    if not s: return ""
    s = _strip_accents(s).lower().replace(".", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _safe_write_csv(df: pd.DataFrame, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return True
    except Exception:
        return False

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _to_json_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(v) for v in x]
    s = str(x).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else [s]
    except:
        return [p.strip() for p in s.split("|") if p.strip()]

def _list_to_store(x: List[str]) -> str:
    return json.dumps(list(x or []), ensure_ascii=False)

def _money_float(x) -> float:
    try: return float(str(x).replace("$", "").replace(",", "").strip() or 0)
    except: return 0.0

def _pct_float(x) -> float:
    try: return float(str(x).replace("%", "").strip() or 0)
    except: return 0.0

def _find_team_logo(owner: str, data_dir: str) -> str:
    owner_key = _norm(owner)
    paths = [os.path.join("assets", "previews"), data_dir, "."]
    for d in paths:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if owner_key in _norm(fn) and fn.lower().endswith((".png", ".jpg")):
                    return os.path.join(d, fn)
    return ""

# --- PATHS & SCHEMA ---
TX_COLS = ["trade_id", "timestamp", "season", "owner_a", "owner_b", "a_players", "b_players", 
           "a_picks", "b_picks", "a_cash", "b_cash", "a_retained_pct", "b_retained_pct", 
           "status", "approved_a", "approved_b", "notes"]

def _equipes_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")

def _transactions_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"transactions_{season}.csv")

@st.cache_data(show_spinner=False)
def load_equipes_joueurs(data_dir: str, season: str) -> pd.DataFrame:
    df = _safe_read_csv(_equipes_path(data_dir, season))
    if df.empty: return df
    # Mapping colonnes
    df["_player"] = df.get("Joueur", df.columns[0]).astype(str)
    df["_owner"] = df.get("Proprietaire", df.get("Équipe", "")).astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_transactions(data_dir: str, season: str) -> pd.DataFrame:
    df = _safe_read_csv(_transactions_path(data_dir, season))
    if df.empty: return pd.DataFrame(columns=TX_COLS)
    return df

def save_transactions(data_dir: str, season: str, df: pd.DataFrame) -> bool:
    return _safe_write_csv(df[TX_COLS], _transactions_path(data_dir, season))

# --- UI COMPONENTS ---
def _status_badge(s: str) -> str:
    badges = {"accepted": "✅", "declined": "❌", "cancelled": "🚫", "proposed": "🟡"}
    return f"{badges.get(s.lower(), '⚪')} {s}"

def _trade_title(r: pd.Series) -> str:
    return f"{r.get('trade_id')} — {r.get('owner_a')} ⇄ {r.get('owner_b')}"

def _render_trade_detail(r: pd.Series, data_dir: str):
    colA, colB = st.columns(2)
    with colA:
        st.subheader(f"Donné par {r['owner_a']}")
        for p in _to_json_list(r['a_players']): st.write(f"- {p}")
        st.caption(f"Picks: {r['a_picks']} | Cash: {r['a_cash']}$")
    with colB:
        st.subheader(f"Donné par {r['owner_b']}")
        for p in _to_json_list(r['b_players']): st.write(f"- {p}")
        st.caption(f"Picks: {r['b_picks']} | Cash: {r['b_cash']}$")

# --- MAIN RENDER ---
def render(ctx: dict):
    st.header("📦 Transactions")
    data_dir = _data_dir(ctx)
    season = _season(ctx)
    
    eq = load_equipes_joueurs(data_dir, season)
    tx = load_transactions(data_dir, season)

    if eq.empty:
        st.warning(f"Fichier `equipes_joueurs_{season}.csv` manquant dans `{data_dir}/`.")
        return

    t1, t2, t3 = st.tabs(["➕ Nouvelle", "📋 Historique", "🛠️ Actions"])

    with t1:
        owners = sorted(eq["_owner"].unique().tolist())
        c1, c2 = st.columns(2)
        oa = c1.selectbox("Proposeur (A)", owners, key="new_tx_a")
        ob = c2.selectbox("Cible (B)", [o for o in owners if o != oa], key="new_tx_b")
        
        pa = eq[eq["_owner"] == oa]["_player"].tolist()
        pb = eq[eq["_owner"] == ob]["_player"].tolist()
        
        sel_a = st.multiselect(f"Joueurs de {oa}", pa)
        sel_b = st.multiselect(f"Joueurs de {ob}", pb)
        
        if st.button("Envoyer la proposition"):
            new_id = f"TR-{uuid.uuid4().hex[:5].upper()}"
            new_row = pd.DataFrame([{
                "trade_id": new_id, "timestamp": _now_iso(), "season": season,
                "owner_a": oa, "owner_b": ob, "a_players": _list_to_store(sel_a),
                "b_players": _list_to_store(sel_b), "a_picks": "", "b_picks": "",
                "a_cash": 0, "b_cash": 0, "a_retained_pct": 0, "b_retained_pct": 0,
                "status": "proposed", "approved_a": False, "approved_b": False, "notes": ""
            }])
            updated_tx = pd.concat([tx, new_row], ignore_index=True)
            if save_transactions(data_dir, season, updated_tx):
                st.success("Transaction proposée !")
                st.cache_data.clear()
                st.rerun()

    with t2:
        if tx.empty:
            st.info("Aucune transaction.")
        else:
            st.dataframe(tx[["trade_id", "owner_a", "owner_b", "status", "timestamp"]], use_container_width=True)

    with t3:
        pending = tx[tx["status"] == "proposed"]
        if pending.empty:
            st.info("Aucune action requise.")
        else:
            selected_tid = st.selectbox("Sélectionner un échange", pending["trade_id"].tolist())
            trade_row = pending[pending["trade_id"] == selected_tid].iloc[0]
            _render_trade_detail(trade_row, data_dir)
            
            if st.button("✅ Approuver (Simulé - Les deux GMs)"):
                tx.loc[tx["trade_id"] == selected_tid, "status"] = "accepted"
                save_transactions(data_dir, season, tx)
                st.cache_data.clear()
                st.rerun()
