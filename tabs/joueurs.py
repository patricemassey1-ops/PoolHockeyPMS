# tabs/joueurs.py
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st


# =====================================================
# Helpers
# =====================================================
def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip()


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_name(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    s = _strip_accents(s).lower()
    s = s.replace(".", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if "," in s:
        a, b = [p.strip() for p in s.split(",", 1)]
        s = f"{b} {a}"
    return s


def _first_existing(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


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


def _guess_col(df: pd.DataFrame, names) -> str:
    for n in names:
        if n in df.columns:
            return n
    return ""


def _as_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        s = str(x).replace(",", "").replace("$", "").strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _money(v) -> str:
    f = _as_float(v)
    if f is None:
        s = str(v or "").strip()
        return s if s else "—"
    return f"{int(round(f)):,}".replace(",", " ") + " $"


# =====================================================
# Loaders
# =====================================================
@st.cache_data(show_spinner=False)
def load_players_db(data_dir: str) -> pd.DataFrame:
    path = _first_existing(
        os.path.join(data_dir, "hockey.players.csv"),
        os.path.join(data_dir, "Hockey.Players.csv"),
    )
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    name_col = _guess_col(df, ["Joueur", "Player", "Name", "name"])
    if not name_col:
        name_col = df.columns[0]

    df = df.copy()
    df["_display_name"] = df[name_col].astype(str)
    df["_name_key"] = df[name_col].astype(str).map(_norm_name)

    if "Level" not in df.columns:
        df["Level"] = ""

    nhl_col = _guess_col(df, ["NHL ID", "NHL_ID", "playerId", "nhl_id", "player_id"])
    df["_nhl_id"] = df[nhl_col].astype(str) if nhl_col else ""

    df.attrs["__path__"] = path
    return df


@st.cache_data(show_spinner=False)
def load_contracts(data_dir: str) -> pd.DataFrame:
    path = _first_existing(os.path.join(data_dir, "puckpedia.contracts.csv"))
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    name_col = _guess_col(df, ["Player", "Joueur", "Name", "name"])
    if not name_col:
        name_col = df.columns[0]

    df = df.copy()
    df["_name_key"] = df[name_col].astype(str).map(_norm_name)
    df.attrs["__path__"] = path
    return df


@st.cache_data(show_spinner=False)
def load_points(data_dir: str, season: str) -> pd.DataFrame:
    path = _first_existing(os.path.join(data_dir, f"points_periods_{season}.csv"))
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    name_col = _guess_col(df, ["Joueur", "Player", "Name", "name"])
    pts_col = _guess_col(df, ["Fantasy Points", "FantasyPoints", "Points", "Pts", "Total", "FPTS"])
    if not name_col or not pts_col:
        return pd.DataFrame()

    out = df[[name_col, pts_col]].copy()
    out["_name_key"] = out[name_col].astype(str).map(_norm_name)
    out["_pts"] = pd.to_numeric(out[pts_col], errors="coerce").fillna(0)
    out = out.groupby("_name_key", as_index=False)["_pts"].sum()
    out = out.sort_values("_pts", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out.attrs["__path__"] = path
    return out


# =====================================================
# Level resolution + badge
# =====================================================
def resolve_level(player_row: Optional[pd.Series], contract_row: Optional[pd.Series]) -> str:
    if player_row is not None:
        lvl = str(player_row.get("Level") or "").upper().strip()
        if lvl in ("ELC", "STD"):
            return lvl

    if contract_row is not None:
        # détecte ELC via champs variés
        t = str(contract_row.get("Type") or contract_row.get("Contract Type") or contract_row.get("Level") or "").upper()
        if "ELC" in t or "ENTRY" in t:
            return "ELC"

    return "—"


def level_badge_html(level: str) -> str:
    lvl = (level or "—").upper().strip()
    if lvl == "ELC":
        bg, fg = "#133d1f", "#b7f7c6"
        txt = "ELC"
    elif lvl == "STD":
        bg, fg = "#11304a", "#bfe2ff"
        txt = "STD"
    else:
        bg, fg = "#2a2d33", "#d6d6d6"
        txt = "—"
    return f"""
    <span style="
        display:inline-block;
        padding:2px 10px;
        border-radius:999px;
        font-weight:700;
        font-size:12px;
        background:{bg};
        color:{fg};
        border:1px solid rgba(255,255,255,0.08);
        letter-spacing:0.5px;
    ">{txt}</span>
    """


# =====================================================
# Extractors
# =====================================================
def _row_by_key(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if df is None or df.empty or not key:
        return None
    m = df.loc[df["_name_key"] == key]
    return None if m.empty else m.iloc[0]


def _contract_by_key(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if df is None or df.empty or not key:
        return None
    m = df.loc[df["_name_key"] == key]
    return None if m.empty else m.iloc[0]


def _points_by_key(df: pd.DataFrame, key: str) -> Optional[Tuple[float, int, int]]:
    if df is None or df.empty or not key:
        return None
    m = df.loc[df["_name_key"] == key]
    if m.empty:
        return None
    r = m.iloc[0]
    return (float(r.get("_pts", 0) or 0), int(r.get("rank", 0) or 0), int(df.shape[0]))


def _get_contract_fields(crow: Optional[pd.Series]) -> Dict[str, str]:
    if crow is None:
        return {"aav": "—", "expiry": "—", "term": "—", "clause": "—"}
    aav = crow.get("AAV", "") or crow.get("Cap Hit", "") or crow.get("CapHit", "") or ""
    expiry = crow.get("Expiry", "") or crow.get("Expiry Year", "") or crow.get("End", "") or ""
    term = crow.get("Term", "") or ""
    clause = crow.get("Clause", "") or crow.get("Clauses", "") or ""
    return {
        "aav": _money(aav),
        "expiry": str(expiry).strip() if str(expiry).strip() else "—",
        "term": str(term).strip() if str(term).strip() else "—",
        "clause": str(clause).strip() if str(clause).strip() else "—",
    }


def _get_basic_fields(prow: pd.Series) -> Dict[str, str]:
    pos = str(prow.get("Pos", "") or "").strip() or "—"
    team = str(prow.get("Team", "") or prow.get("Equipe", "") or "").strip() or "—"
    country = str(prow.get("Country", "") or "").strip() or "—"
    salary = prow.get("Salary", "") or prow.get("Cap Hit", "") or prow.get("AAV", "") or ""
    salary = _money(salary)
    return {"pos": pos, "team": team, "country": country, "salary": salary}


def _headshot_url(nhl_id: str) -> str:
    nhl_id = str(nhl_id or "").strip()
    if not nhl_id or nhl_id.lower() == "nan":
        return ""
    return f"https://assets.nhle.com/mugs/nhl/{nhl_id}.png"


# =====================================================
# UI blocks
# =====================================================
def _profile_block(title: str, players: pd.DataFrame, contracts: pd.DataFrame, points: pd.DataFrame, key: str) -> None:
    prow = _row_by_key(players, key)
    if prow is None:
        st.warning("Joueur introuvable.")
        return
    crow = _contract_by_key(contracts, key)
    lvl = resolve_level(prow, crow)
    badge = level_badge_html(lvl)

    basic = _get_basic_fields(prow)
    cfields = _get_contract_fields(crow)
    prk = _points_by_key(points, key)

    colL, colR = st.columns([1, 2], gap="large")
    with colL:
        u = _headshot_url(str(prow.get("_nhl_id") or ""))
        if u:
            try:
                st.image(u, use_container_width=True)
            except Exception:
                st.info("Photo indisponible.")
        else:
            st.info("Photo indisponible (NHL ID manquant).")

    with colR:
        st.subheader(title)
        st.markdown(f"**Nom** : {prow.get('_display_name','—')}  \n**Level (Pool)** : {badge}", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Position")
            st.write(basic["pos"])
        with c2:
            st.caption("Équipe NHL")
            st.write(basic["team"])
        with c3:
            st.caption("Pays")
            st.write(basic["country"])

        st.markdown("##### Salaire / Contrat")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            st.caption("Salaire/Cap")
            st.write(basic["salary"])
        with c5:
            st.caption("AAV")
            st.write(cfields["aav"])
        with c6:
            st.caption("Expiry")
            st.write(cfields["expiry"])
        with c7:
            st.caption("Clause")
            st.write(cfields["clause"])

        if prk is not None:
            pts, rank, total = prk
            st.markdown("##### Classement")
            st.write(f"Points: **{pts:.1f}**  |  Rang: **{rank} / {total}**")


def _cmp_table(players: pd.DataFrame, contracts: pd.DataFrame, points: pd.DataFrame, a_key: str, b_key: str) -> pd.DataFrame:
    def row_for(key: str) -> Dict[str, Any]:
        prow = _row_by_key(players, key)
        crow = _contract_by_key(contracts, key)
        if prow is None:
            return {"Nom": "—"}
        lvl = resolve_level(prow, crow)
        basic = _get_basic_fields(prow)
        cfields = _get_contract_fields(crow)
        prk = _points_by_key(points, key)

        pts = "—"
        rk = "—"
        if prk is not None:
            pts = f"{prk[0]:.1f}"
            rk = f"{prk[1]} / {prk[2]}"

        return {
            "Nom": str(prow.get("_display_name") or "—"),
            "Level": lvl,
            "Pos": basic["pos"],
            "Team": basic["team"],
            "Country": basic["country"],
            "Salary/Cap": basic["salary"],
            "AAV": cfields["aav"],
            "Expiry": cfields["expiry"],
            "Points": pts,
            "Rank": rk,
        }

    a = row_for(a_key)
    b = row_for(b_key)
    df = pd.DataFrame([a, b], index=["Joueur A", "Joueur B"])
    return df


# =====================================================
# Main render
# =====================================================
def render(ctx: dict) -> None:
    st.header("👤 Joueurs")
    st.caption("Recherche hockey.players.csv + Level STD/ELC + classement + comparatif")

    data_dir = _data_dir(ctx)
    season = _season(ctx)

    players = load_players_db(data_dir)
    contracts = load_contracts(data_dir)
    points = load_points(data_dir, season)

    if players.empty:
        st.error("Players DB introuvable ou vide. Vérifie `data/hockey.players.csv`.")
        return

    # -------------------------
    # Filtre Level
    # -------------------------
    st.markdown("### 🎚️ Filtres")
    level_filter = st.radio("Level", ["Tous", "ELC seulement", "STD seulement"], horizontal=True, key="players_level_filter")

    # options candidates (limiter)
    df_opts = players[["_display_name", "_name_key", "Level"]].copy()

    # enrichir temporairement le Level via contracts pour filtrer correctement
    if not contracts.empty:
        # map key->detected level
        c = contracts[["_name_key"]].copy()
        # détection simple
        tcol = _guess_col(contracts, ["Type", "Contract Type", "Level"])
        if tcol:
            c["_lvl"] = contracts[tcol].astype(str).str.upper().apply(lambda x: "ELC" if ("ELC" in x or "ENTRY" in x) else "")
            lvl_map = dict(zip(c["_name_key"], c["_lvl"]))
            # fill only if empty
            df_opts["_lvl_tmp"] = df_opts["Level"].astype(str).str.upper().str.strip()
            df_opts.loc[df_opts["_lvl_tmp"].isin(["", "NAN"]), "_lvl_tmp"] = df_opts["_name_key"].map(lvl_map).fillna("")
        else:
            df_opts["_lvl_tmp"] = df_opts["Level"].astype(str).str.upper().str.strip()
    else:
        df_opts["_lvl_tmp"] = df_opts["Level"].astype(str).str.upper().str.strip()

    if level_filter == "ELC seulement":
        df_opts = df_opts[df_opts["_lvl_tmp"] == "ELC"]
    elif level_filter == "STD seulement":
        df_opts = df_opts[df_opts["_lvl_tmp"] == "STD"]

    # -------------------------
    # Recherche
    # -------------------------
    st.markdown("### 🔎 Recherche")
    q = st.text_input("Tape un nom (Marner, Draisaitl, Savoie...)")

    if q.strip():
        qs = _norm_name(q)
        mask = df_opts["_display_name"].astype(str).str.lower().str.contains(q.lower(), na=False)
        if qs:
            mask = mask | df_opts["_name_key"].astype(str).str.contains(qs, na=False)
        df_opts = df_opts.loc[mask]

    # limiter pour perf
    candidates = df_opts.head(300)
    opts = candidates["_display_name"].tolist()
    if not opts:
        st.info("Aucun joueur trouvé avec ces filtres.")
        return

    sel = st.selectbox("Choisir un joueur", opts, key="players_pick_main")
    key = _norm_name(sel)

    # -------------------------
    # Profil
    # -------------------------
    st.divider()
    st.markdown("### 🧾 Profil joueur")
    _profile_block("Profil", players, contracts, points, key)

    # -------------------------
    # Comparatif
    # -------------------------
    st.divider()
    st.markdown("### ⚖️ Comparatif (2 joueurs)")

    # listes filtrées pour cohérence
    pool = df_opts["_display_name"].head(2000).tolist()
    if len(pool) < 2:
        pool = players["_display_name"].head(2000).tolist()

    colA, colB = st.columns(2, gap="large")
    with colA:
        a_name = st.selectbox("Joueur A", pool, index=0, key="cmp_a")
    with colB:
        b_name = st.selectbox("Joueur B", pool, index=1, key="cmp_b")

    a_key = _norm_name(a_name)
    b_key = _norm_name(b_name)

    df_cmp = _cmp_table(players, contracts, points, a_key, b_key)

    # badges pour Level dans table
    df_show = df_cmp.copy()
    df_show["Level"] = df_show["Level"].apply(lambda x: "ELC 🟢" if x == "ELC" else ("STD 🔵" if x == "STD" else "—"))
    st.dataframe(df_show, use_container_width=True)

    # -------------------------
    # Debug sources (sans ctx brut)
    # -------------------------
    with st.expander("🧪 Debug (sources)", expanded=False):
        st.write("Players DB:", players.attrs.get("__path__", ""))
        st.write("Contracts:", contracts.attrs.get("__path__", "(absent)"))
        st.write("Points:", points.attrs.get("__path__", "(absent)"))
