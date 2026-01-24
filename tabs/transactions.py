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


# =====================================================
# Helpers
# =====================================================
def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_player_key(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    s = _strip_accents(s).lower()
    s = s.replace(".", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if "," in s:
        a, b = [p.strip() for p in s.split(",", 1)]
        if a and b:
            s = f"{b} {a}"
    return s


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _safe_write_csv(df: pd.DataFrame, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _to_json_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(v) for v in obj if str(v).strip()]
    except Exception:
        pass
    parts = [p.strip() for p in s.split("|")]
    return [p for p in parts if p]


def _list_to_store(x: List[str]) -> str:
    try:
        return json.dumps(list(x or []), ensure_ascii=False)
    except Exception:
        return "[]"


def _money_float(x) -> float:
    try:
        if x is None:
            return 0.0
        s = str(x).replace("$", "").replace(",", "").strip()
        if not s:
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _pct_float(x) -> float:
    try:
        s = str(x).replace("%", "").strip()
        if not s:
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _safe_owner_from_session() -> str:
    return str(st.session_state.get("selected_owner") or st.session_state.get("owner") or "").strip()


def _asset_dirs(data_dir: str) -> List[str]:
    return [
        os.path.join("assets", "previews"),
        data_dir,
        ".",
    ]


def _find_team_logo(owner: str, data_dir: str) -> str:
    """
    Cherche un logo d'équipe correspondant au GM/Owner.
    Match: filename contient owner normalisé
    """
    owner_key = _norm(owner)
    if not owner_key:
        return ""

    candidates: List[str] = []
    for d in _asset_dirs(data_dir):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            if fn.lower() in ["gm_logo.png", "logo_pool.png"]:
                continue
            fkey = _norm(fn.replace("_logo", "").replace("logo", ""))
            if owner_key in fkey:
                candidates.append(os.path.join(d, fn))

    if not candidates:
        # tentative pattern <Owner>_Logo.png
        for d in _asset_dirs(data_dir):
            p = os.path.join(d, f"{owner}_Logo.png")
            if os.path.exists(p):
                return p
        return ""

    candidates = sorted(candidates, key=lambda x: len(os.path.basename(x)))
    return candidates[0]


# =====================================================
# Paths / schema
# =====================================================
def _equipes_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")


def _transactions_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"transactions_{season}.csv")


TX_COLS = [
    "trade_id",
    "timestamp",
    "season",
    "owner_a",
    "owner_b",
    "a_players",
    "b_players",
    "a_picks",
    "b_picks",
    "a_cash",
    "b_cash",
    "a_retained_pct",
    "b_retained_pct",
    "status",        # proposed / accepted / declined / cancelled
    "approved_a",    # bool
    "approved_b",    # bool
    "notes",
]


# =====================================================
# Loaders
# =====================================================
@st.cache_data(show_spinner=False)
def load_equipes_joueurs(data_dir: str, season: str) -> pd.DataFrame:
    path = _equipes_path(data_dir, season)
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # detect columns
    player_col = ""
    for c in ["Joueur", "Player", "Name", "player", "name"]:
        if c in df.columns:
            player_col = c
            break
    if not player_col:
        player_col = df.columns[0]

    owner_col = ""
    for c in ["Proprietaire", "Owner", "Équipe", "Equipe", "Team", "GM"]:
        if c in df.columns:
            owner_col = c
            break

    slot_col = ""
    for c in ["Statut", "Status", "Slot", "Position Slot", "Roster Slot", "Type"]:
        if c in df.columns:
            slot_col = c
            break

    df["_player"] = df[player_col].astype(str)
    df["_name_key"] = df["_player"].astype(str).map(_norm_player_key)
    df["_owner"] = df[owner_col].astype(str) if owner_col else ""
    df["_slot"] = df[slot_col].astype(str) if slot_col else ""

    df.attrs["__path__"] = path
    return df


@st.cache_data(show_spinner=False)
def load_transactions(data_dir: str, season: str) -> pd.DataFrame:
    path = _transactions_path(data_dir, season)
    df = _safe_read_csv(path)
    if df is None or df.empty:
        df = pd.DataFrame(columns=TX_COLS)
        df.attrs["__path__"] = path
        return df

    df = df.copy()
    for c in TX_COLS:
        if c not in df.columns:
            df[c] = ""

    df.attrs["__path__"] = path
    return df


def save_transactions(data_dir: str, season: str, df: pd.DataFrame) -> bool:
    path = _transactions_path(data_dir, season)
    out = df.copy()
    for c in TX_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[TX_COLS]
    return _safe_write_csv(out, path)


# =====================================================
# Roster picker helpers
# =====================================================
def owners_list(eq: pd.DataFrame) -> List[str]:
    if eq is None or eq.empty:
        return []
    owners = eq["_owner"].astype(str).str.strip()
    owners = owners[owners.ne("") & owners.ne("nan")]
    return sorted(owners.unique().tolist())


def players_for_owner(eq: pd.DataFrame, owner: str) -> List[str]:
    if eq is None or eq.empty:
        return []
    sub = eq[eq["_owner"].astype(str).str.strip() == str(owner).strip()]
    if sub.empty:
        return []
    names = sub["_player"].astype(str).dropna()
    names = [n.strip() for n in names.tolist() if str(n).strip()]
    seen = set()
    out = []
    for n in names:
        k = _norm_player_key(n)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out


# =====================================================
# Transactions logic
# =====================================================
def new_trade(
    season: str,
    owner_a: str,
    owner_b: str,
    a_players: List[str],
    b_players: List[str],
    a_picks: str,
    b_picks: str,
    a_cash: float,
    b_cash: float,
    a_retained: float,
    b_retained: float,
    notes: str,
) -> Dict[str, Any]:
    tid = f"TR-{uuid.uuid4().hex[:8].upper()}"
    return {
        "trade_id": tid,
        "timestamp": _now_iso(),
        "season": season,
        "owner_a": owner_a,
        "owner_b": owner_b,
        "a_players": _list_to_store(a_players),
        "b_players": _list_to_store(b_players),
        "a_picks": str(a_picks or "").strip(),
        "b_picks": str(b_picks or "").strip(),
        "a_cash": float(a_cash or 0.0),
        "b_cash": float(b_cash or 0.0),
        "a_retained_pct": float(a_retained or 0.0),
        "b_retained_pct": float(b_retained or 0.0),
        "status": "proposed",
        "approved_a": False,
        "approved_b": False,
        "notes": str(notes or "").strip(),
    }


def _boolify(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ["1", "true", "yes", "y", "ok"]


def _status_badge(s: str) -> str:
    s = str(s or "").strip().lower()
    if s == "accepted":
        return "✅ accepted"
    if s == "declined":
        return "❌ declined"
    if s == "cancelled":
        return "🚫 cancelled"
    return "🟡 proposed"


def _trade_title(r: pd.Series) -> str:
    a = str(r.get("owner_a", "") or "").strip() or "A"
    b = str(r.get("owner_b", "") or "").strip() or "B"
    tid = str(r.get("trade_id", "") or "").strip()
    return f"{tid} — {a} ⇄ {b}"


def _render_trade_detail(r: pd.Series, data_dir: str) -> None:
    a = str(r.get("owner_a", "") or "").strip()
    b = str(r.get("owner_b", "") or "").strip()

    # logos
    la = _find_team_logo(a, data_dir)
    lb = _find_team_logo(b, data_dir)

    head1, head2, head3 = st.columns([1, 6, 1], gap="large")
    with head1:
        if la:
            try:
                st.image(la, width=70)
            except Exception:
                pass
    with head2:
        st.markdown(f"### {_trade_title(r)}")
        st.caption(f"Statut: **{_status_badge(r.get('status'))}**  |  Timestamp: {r.get('timestamp','')}")
    with head3:
        if lb:
            try:
                st.image(lb, width=70)
            except Exception:
                pass

    a_players = _to_json_list(r.get("a_players"))
    b_players = _to_json_list(r.get("b_players"))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader(a or "GM A")
        st.markdown("**Joueurs donnés**")
        if a_players:
            for p in a_players:
                st.write(f"• {p}")
        else:
            st.write("—")
        st.markdown("**Picks**")
        st.write(str(r.get("a_picks", "") or "—") or "—")
        st.markdown("**Cash**")
        st.write(f"{_money_float(r.get('a_cash')):,.0f} $".replace(",", " "))
        st.markdown("**Retention**")
        st.write(f"{_pct_float(r.get('a_retained_pct')):.0f}%")

    with c2:
        st.subheader(b or "GM B")
        st.markdown("**Joueurs donnés**")
        if b_players:
            for p in b_players:
                st.write(f"• {p}")
        else:
            st.write("—")
        st.markdown("**Picks**")
        st.write(str(r.get("b_picks", "") or "—") or "—")
        st.markdown("**Cash**")
        st.write(f"{_money_float(r.get('b_cash')):,.0f} $".replace(",", " "))
        st.markdown("**Retention**")
        st.write(f"{_pct_float(r.get('b_retained_pct')):.0f}%")

    notes = str(r.get("notes", "") or "").strip()
    if notes:
        st.markdown("**Notes**")
        st.write(notes)


def _render_trades_table(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("Aucune transaction enregistrée.")
        return

    view = df.copy()
    view["status_badge"] = view["status"].apply(_status_badge)
    view["title"] = view.apply(lambda r: _trade_title(r), axis=1)

    show = pd.DataFrame()
    show["Trade"] = view["title"]
    show["Statut"] = view["status_badge"]
    show["A approuvé"] = view["approved_a"].apply(_boolify)
    show["B approuvé"] = view["approved_b"].apply(_boolify)
    show["Date"] = view["timestamp"]
    st.dataframe(show, use_container_width=True, hide_index=True)


# =====================================================
# UI
# =====================================================
def render(ctx: dict) -> None:
    st.header("📦 Transactions")
    st.caption("Proposer, suivre et approuver des échanges (CSV local).")

    data_dir = _data_dir(ctx)
    season = _season(ctx)

    eq = load_equipes_joueurs(data_dir, season)
    tx = load_transactions(data_dir, season)

    owners = owners_list(eq)

    t1, t2, t3, t4 = st.tabs(["➕ Proposer", "📋 Toutes", "✅ Acceptées", "🛠️ Actions"])

    # -------------------------------------------------
    # Tab 1: proposer
    # -------------------------------------------------
    with t1:
        if not owners:
            st.error("Aucune équipe détectée dans equipes_joueurs. Importe d'abord `equipes_joueurs_<season>.csv`.")
            st.code(_equipes_path(data_dir, season))
            return

        default_owner = _safe_owner_from_session()
        default_idx = owners.index(default_owner) if default_owner in owners else 0

        colA, colB = st.columns(2, gap="large")
        with colA:
            owner_a = st.selectbox("GM A (proposeur)", owners, index=default_idx, key="tx_owner_a")
        with colB:
            owner_b_choices = [o for o in owners if o != owner_a] or owners
            owner_b = st.selectbox("GM B (cible)", owner_b_choices, index=0, key="tx_owner_b")

        st.divider()

        pa = players_for_owner(eq, owner_a)
        pb = players_for_owner(eq, owner_b)

        # max 5 strict: on construit options restantes selon sélection
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader(f"{owner_a} donne")
            prev_a = st.session_state.get("tx_a_players_pro", [])
            a_players = st.multiselect("Joueurs (max 5)", pa, default=prev_a, key="tx_a_players_pro")
            if len(a_players) > 5:
                a_players = a_players[:5]
                st.session_state["tx_a_players_pro"] = a_players
            st.caption(f"{len(a_players)} / 5")

        with col2:
            st.subheader(f"{owner_b} donne")
            prev_b = st.session_state.get("tx_b_players_pro", [])
            b_players = st.multiselect("Joueurs (max 5)", pb, default=prev_b, key="tx_b_players_pro")
            if len(b_players) > 5:
                b_players = b_players[:5]
                st.session_state["tx_b_players_pro"] = b_players
            st.caption(f"{len(b_players)} / 5")

        st.divider()

        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.subheader(f"Extras {owner_a}")
            a_picks = st.text_input("Picks (ex: 2026 R1 | 2027 R3)", value="", key="tx_a_picks")
            a_cash = st.number_input("Cash ($)", min_value=0.0, value=0.0, step=50000.0, key="tx_a_cash")
            a_ret = st.slider("Retention (%)", min_value=0, max_value=50, value=0, step=5, key="tx_a_ret")
        with c4:
            st.subheader(f"Extras {owner_b}")
            b_picks = st.text_input("Picks (ex: 2026 R2)", value="", key="tx_b_picks")
            b_cash = st.number_input("Cash ($)", min_value=0.0, value=0.0, step=50000.0, key="tx_b_cash")
            b_ret = st.slider("Retention (%)", min_value=0, max_value=50, value=0, step=5, key="tx_b_ret")

        notes = st.text_area("Notes (optionnel)", value="", height=80, key="tx_notes")

        st.divider()

        can_submit = (
            owner_a.strip()
            and owner_b.strip()
            and owner_a != owner_b
            and (
                len(a_players) + len(b_players) > 0
                or a_picks.strip()
                or b_picks.strip()
                or a_cash > 0
                or b_cash > 0
            )
        )

        if not can_submit:
            st.info("Sélectionne au moins 1 joueur (ou picks/cash) et 2 équipes différentes.")
        else:
            if st.button("📨 Envoyer la proposition", type="primary", use_container_width=True):
                row = new_trade(
                    season=season,
                    owner_a=owner_a,
                    owner_b=owner_b,
                    a_players=a_players,
                    b_players=b_players,
                    a_picks=a_picks,
                    b_picks=b_picks,
                    a_cash=float(a_cash),
                    b_cash=float(b_cash),
                    a_retained=float(a_ret),
                    b_retained=float(b_ret),
                    notes=notes,
                )
                tx2 = tx.copy()
                tx2 = pd.concat([tx2, pd.DataFrame([row])], ignore_index=True)
                ok = save_transactions(data_dir, season, tx2)
                if ok:
                    st.success(f"Proposition envoyée: {row['trade_id']}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Échec de sauvegarde transactions CSV.")

    # -------------------------------------------------
    # Tab 2: toutes
    # -------------------------------------------------
    with t2:
        if tx is None or tx.empty:
            st.info("Aucune transaction pour le moment.")
            return

        st.markdown("### 📋 Liste")
        status_filter = st.radio(
            "Filtrer",
            ["Toutes", "proposed", "accepted", "declined", "cancelled"],
            horizontal=True,
            key="tx_status_filter",
        )

        df = tx.copy()
        if status_filter != "Toutes":
            df = df[df["status"].astype(str).str.lower().eq(status_filter)]

        df = df.copy()
        df["_ts"] = df["timestamp"].astype(str)
        df = df.sort_values("_ts", ascending=False).drop(columns=["_ts"], errors="ignore")

        _render_trades_table(df)

        st.divider()
        st.markdown("### 🔎 Détail")
        titles = df.apply(lambda r: _trade_title(r), axis=1).tolist()
        if not titles:
            st.info("Aucun élément.")
            return
        pick = st.selectbox("Choisir une transaction", titles, key="tx_pick_detail_all")
        tid = pick.split("—", 1)[0].strip()
        rr = df[df["trade_id"].astype(str).str.strip() == tid]
        if rr.empty:
            st.warning("Introuvable.")
            return
        _render_trade_detail(rr.iloc[0], data_dir)

    # -------------------------------------------------
    # Tab 3: acceptées
    # -------------------------------------------------
    with t3:
        if tx is None or tx.empty:
            st.info("Aucune transaction.")
            return
        df = tx.copy()
        df["status"] = df["status"].astype(str)
        acc = df[df["status"].str.lower().eq("accepted")].copy()
        if acc.empty:
            st.info("Aucune transaction acceptée.")
            return

        acc["_ts"] = acc["timestamp"].astype(str)
        acc = acc.sort_values("_ts", ascending=False).drop(columns=["_ts"], errors="ignore")

        st.markdown("### ✅ Transactions acceptées")
        _render_trades_table(acc)

        st.divider()
        titles = acc.apply(lambda r: _trade_title(r), axis=1).tolist()
        pick = st.selectbox("Détail (acceptées)", titles, key="tx_pick_detail_acc")
        tid = pick.split("—", 1)[0].strip()
        rr = acc[acc["trade_id"].astype(str).str.strip() == tid]
        if rr.empty:
            st.warning("Introuvable.")
            return
        _render_trade_detail(rr.iloc[0], data_dir)

    # -------------------------------------------------
    # Tab 4: actions/approvals
    # -------------------------------------------------
    with t4:
        st.markdown("### 🛠️ Actions / Approvals")
        st.caption("Double approbation: A et B doivent approuver pour passer à accepted.")

        if tx is None or tx.empty:
            st.info("Aucune transaction.")
            return

        df = tx.copy()
        df["status"] = df["status"].astype(str)
        proposed = df[df["status"].str.lower().eq("proposed")].copy()
        if proposed.empty:
            st.info("Aucune transaction en attente.")
            return

        titles = proposed.apply(lambda r: _trade_title(r), axis=1).tolist()
        pick = st.selectbox("Transaction à traiter", titles, key="tx_pick_action")
        tid = pick.split("—", 1)[0].strip()
        row_idx = proposed.index[proposed["trade_id"].astype(str).str.strip() == tid]
        if len(row_idx) == 0:
            st.warning("Introuvable.")
            return
        idx = int(row_idx[0])

        r = df.loc[idx]
        _render_trade_detail(r, data_dir)

        st.divider()

        owners_all = sorted(list(set([str(x).strip() for x in df["owner_a"].tolist() + df["owner_b"].tolist()])))
        acting_default = _safe_owner_from_session()
        acting_default = acting_default if acting_default in owners_all else (owners_all[0] if owners_all else "")
        acting = st.selectbox(
            "Je suis (GM)",
            owners_all,
            index=owners_all.index(acting_default) if acting_default in owners_all else 0,
            key="tx_acting_owner",
        )

        owner_a = str(r.get("owner_a") or "").strip()
        owner_b = str(r.get("owner_b") or "").strip()

        approved_a = _boolify(r.get("approved_a"))
        approved_b = _boolify(r.get("approved_b"))

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                if acting == owner_a:
                    approved_a = True
                elif acting == owner_b:
                    approved_b = True
                else:
                    st.warning("Tu dois choisir GM A ou GM B pour approuver.")
                    st.stop()

                df.at[idx, "approved_a"] = bool(approved_a)
                df.at[idx, "approved_b"] = bool(approved_b)

                if approved_a and approved_b:
                    df.at[idx, "status"] = "accepted"
                    df.at[idx, "timestamp"] = _now_iso()

                ok = save_transactions(data_dir, season, df)
                if ok:
                    st.success("Mise à jour enregistrée.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Échec sauvegarde CSV.")

        with c2:
            if st.button("❌ Decline", use_container_width=True):
                if acting != owner_a and acting != owner_b:
                    st.warning("Tu dois choisir GM A ou GM B pour refuser.")
                    st.stop()
                df.at[idx, "status"] = "declined"
                df.at[idx, "timestamp"] = _now_iso()
                ok = save_transactions(data_dir, season, df)
                if ok:
                    st.success("Refus enregistré.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Échec sauvegarde CSV.")

        with c3:
            if st.button("🚫 Cancel (proposeur)", use_container_width=True):
                if acting != owner_a:
                    st.warning("Seul le proposeur (GM A) peut annuler.")
                    st.stop()
                df.at[idx, "status"] = "cancelled"
                df.at[idx, "timestamp"] = _now_iso()
                ok = save_transactions(data_dir, season, df)
                if ok:
                    st.success("Annulation enregistrée.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Échec sauvegarde CSV.")

        with st.expander("🧪 Debug (sources)", expanded=False):
            st.write("equipes_joueurs:", eq.attrs.get("__path__", ""))
            st.write("transactions:", tx.attrs.get("__path__", _transactions_path(data_dir, season)))
