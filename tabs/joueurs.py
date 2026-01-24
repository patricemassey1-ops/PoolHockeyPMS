# tabs/joueurs.py
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st


# =========================
# Helpers (safe)
# =========================
def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


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
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}".strip()
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
        # fallback permissif
        try:
            return pd.read_csv(path, engine="python", sep=",", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _guess_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def _to_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def _to_money(x) -> str:
    try:
        if pd.isna(x):
            return ""
        v = float(str(x).replace(",", "").replace("$", "").strip())
        if v >= 1000:
            # si c'est en "k" fantrax, on garde brut ailleurs. Ici on assume $
            pass
        return f"{int(round(v)):,}".replace(",", " ")
    except Exception:
        s = str(x or "").strip()
        return s


def _money_with_dollar(x) -> str:
    s = _to_money(x)
    return f"{s} $" if s else ""


# =========================
# Loaders (cached)
# =========================
@st.cache_data(show_spinner=False)
def load_players_db(data_dir: str) -> pd.DataFrame:
    # Source de vérité
    path = _first_existing(
        os.path.join(data_dir, "hockey.players.csv"),
        os.path.join(data_dir, "Hockey.Players.csv"),
        os.path.join(data_dir, "data", "hockey.players.csv"),
    )
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    # normalisation
    name_col = _guess_col(df, ["Joueur", "Player", "name", "Name"])
    if not name_col:
        name_col = df.columns[0]
    df = df.copy()
    df["_name_key"] = df[name_col].astype(str).map(_norm_name)
    df["_display_name"] = df[name_col].astype(str)

    # colonnes attendues (créées si absentes)
    for c in ["Country", "Level", "Pos", "Team", "Equipe", "Salary", "Cap Hit", "AAV"]:
        if c not in df.columns:
            df[c] = ""

    # détecter NHL ID
    nhl_id_col = _guess_col(df, ["NHL ID", "NHL_ID", "nhl_id", "playerId", "player_id", "PlayerID", "NHLID"])
    if nhl_id_col:
        df["_nhl_id"] = df[nhl_id_col].apply(lambda x: str(x).strip() if str(x).strip().lower() != "nan" else "")
    else:
        df["_nhl_id"] = ""

    # garder trace du chemin dans attrs
    df.attrs["__path__"] = path
    df.attrs["__name_col__"] = name_col
    return df


@st.cache_data(show_spinner=False)
def load_puckpedia_contracts(data_dir: str) -> pd.DataFrame:
    path = _first_existing(
        os.path.join(data_dir, "puckpedia.contracts.csv"),
        os.path.join(data_dir, "puckpedia.contracts.csv".lower()),
    )
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    # colonnes typiques puckpedia (varie selon export)
    name_col = _guess_col(df, ["Player", "Joueur", "Name", "player", "name"])
    if not name_col:
        name_col = df.columns[0]

    df = df.copy()
    df["_name_key"] = df[name_col].astype(str).map(_norm_name)

    # essayer d’unifier AAV/Cap Hit/Expiry/Term/Clause
    # on ne force pas, on lit si dispo
    df.attrs["__path__"] = path
    df.attrs["__name_col__"] = name_col
    return df


@st.cache_data(show_spinner=False)
def load_points(data_dir: str, season: str) -> pd.DataFrame:
    # si tu as un fichier points_periods_{season}.csv, on l’utilise pour ranking
    path = _first_existing(
        os.path.join(data_dir, f"points_periods_{season}.csv"),
        os.path.join(data_dir, f"points_periods_{season.replace('–','-')}.csv"),
    )
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    # normaliser clé joueur
    name_col = _guess_col(df, ["Joueur", "Player", "Name", "player", "name"])
    if not name_col:
        return pd.DataFrame()

    pts_col = _guess_col(df, ["Fantasy Points", "FantasyPoints", "Points", "Pts", "Total", "FPTS"])
    if not pts_col:
        # si pas de colonne points identifiable, on ne rank pas
        return pd.DataFrame()

    out = df[[name_col, pts_col]].copy()
    out["_name_key"] = out[name_col].astype(str).map(_norm_name)
    out["_pts"] = pd.to_numeric(out[pts_col], errors="coerce").fillna(0.0)
    # total par joueur
    agg = out.groupby("_name_key", as_index=False)["_pts"].sum()
    agg = agg.sort_values("_pts", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1
    agg.attrs["__path__"] = path
    return agg


# =========================
# NHL headshot (best-effort)
# =========================
def _headshot_urls(nhl_id: str) -> Tuple[str, str]:
    # patterns connus (peuvent changer, on fait best-effort)
    nhl_id = str(nhl_id or "").strip()
    if not nhl_id:
        return ("", "")
    # 1) pattern "mugs"
    u1 = f"https://assets.nhle.com/mugs/nhl/{nhl_id}.png"
    # 2) certains utilisent saison dans l’URL; on laisse un fallback plausible
    u2 = f"https://assets.nhle.com/mugs/nhl/20232024/{nhl_id}.png"
    return u1, u2


# =========================
# Merge / extraction profile
# =========================
def _get_player_row(players_db: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if players_db is None or players_db.empty or not key:
        return None
    m = players_db.loc[players_db["_name_key"] == key]
    if m.empty:
        return None
    return m.iloc[0]


def _get_contract_row(contracts: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if contracts is None or contracts.empty or not key:
        return None
    m = contracts.loc[contracts["_name_key"] == key]
    if m.empty:
        return None
    # si multiples, prendre le premier (souvent le plus récent selon export)
    return m.iloc[0]


def _extract_profile(players_row: Optional[pd.Series], contract_row: Optional[pd.Series], points_df: pd.DataFrame) -> Dict[str, Any]:
    prof: Dict[str, Any] = {}

    # base players db
    if players_row is not None:
        prof["name"] = str(players_row.get("_display_name", "") or "").strip()
        prof["country"] = str(players_row.get("Country", "") or "").strip()
        prof["level"] = str(players_row.get("Level", "") or "").strip()
        prof["pos"] = str(players_row.get("Pos", "") or "").strip()
        # team NHL (selon colonnes)
        prof["nhl_team"] = str(players_row.get("Team", "") or players_row.get("Equipe", "") or "").strip()
        # salary / cap
        prof["salary"] = str(players_row.get("Salary", "") or "").strip()
        prof["cap_hit"] = str(players_row.get("Cap Hit", "") or players_row.get("AAV", "") or "").strip()
        prof["nhl_id"] = str(players_row.get("_nhl_id", "") or "").strip()
    else:
        prof["name"] = ""
        prof["country"] = ""
        prof["level"] = ""
        prof["pos"] = ""
        prof["nhl_team"] = ""
        prof["salary"] = ""
        prof["cap_hit"] = ""
        prof["nhl_id"] = ""

    # contract row (puckpedia)
    if contract_row is not None:
        # colonnes possibles
        prof["contract_team"] = str(contract_row.get("Team", "") or contract_row.get("NHL Team", "") or "").strip()
        prof["aav"] = str(contract_row.get("AAV", "") or contract_row.get("Cap Hit", "") or contract_row.get("CapHit", "") or "").strip()
        prof["expiry"] = str(contract_row.get("Expiry", "") or contract_row.get("Expiry Year", "") or contract_row.get("End", "") or "").strip()
        prof["term"] = str(contract_row.get("Term", "") or "").strip()
        prof["type"] = str(contract_row.get("Type", "") or contract_row.get("Level", "") or "").strip()
        prof["clause"] = str(contract_row.get("Clause", "") or contract_row.get("Clauses", "") or "").strip()
    else:
        prof["contract_team"] = ""
        prof["aav"] = ""
        prof["expiry"] = ""
        prof["term"] = ""
        prof["type"] = ""
        prof["clause"] = ""

    # points / ranking
    prof["points"] = None
    prof["rank"] = None
    prof["players_ranked"] = None
    if points_df is not None and not points_df.empty:
        prof["players_ranked"] = int(points_df.shape[0])
        # on match via name_key si possible (le caller garde key)
        # rempli dans l’affichage via key
    return prof


def _format_field(label: str, value: str) -> str:
    v = str(value or "").strip()
    return v if v else "—"


# =========================
# UI
# =========================
def _render_profile_card(title: str, prof: Dict[str, Any], *, points_rank: Optional[Tuple[float, int, int]] = None) -> None:
    # photo
    nhl_id = str(prof.get("nhl_id") or "").strip()
    colL, colR = st.columns([1, 2], gap="large")

    with colL:
        if nhl_id:
            u1, u2 = _headshot_urls(nhl_id)
            shown = False
            for u in [u1, u2]:
                if not u:
                    continue
                try:
                    st.image(u, use_container_width=True)
                    shown = True
                    break
                except Exception:
                    pass
            if not shown:
                st.info("Photo indisponible (NHL ID présent mais URL non résolue).")
        else:
            st.info("Aucune photo (NHL ID manquant).")

    with colR:
        st.subheader(title)
        st.markdown(
            f"""
**Nom** : {_format_field("Nom", prof.get("name",""))}  
**Position** : {_format_field("Pos", prof.get("pos",""))}  
**Équipe (NHL)** : {_format_field("Team", prof.get("nhl_team",""))}  
**Pays** : {_format_field("Country", prof.get("country",""))}  
**Level (Pool)** : {_format_field("Level", prof.get("level",""))}  
"""
        )

        st.markdown("##### Contrat")
        st.markdown(
            f"""
**AAV / Cap Hit** : {_format_field("AAV", prof.get("aav","") or prof.get("cap_hit","") or prof.get("salary",""))}  
**Expiry** : {_format_field("Expiry", prof.get("expiry",""))}  
**Term** : {_format_field("Term", prof.get("term",""))}  
**Clause** : {_format_field("Clause", prof.get("clause",""))}  
"""
        )

        if points_rank is not None:
            pts, rank, total = points_rank
            st.markdown("##### Classement")
            st.markdown(f"**Points** : {pts:.1f}  \n**Rang** : {rank} / {total}")


def _resolve_points_rank(points_df: pd.DataFrame, key: str) -> Optional[Tuple[float, int, int]]:
    if points_df is None or points_df.empty or not key:
        return None
    m = points_df.loc[points_df["_name_key"] == key]
    if m.empty:
        return None
    r = m.iloc[0]
    pts = float(r.get("_pts", 0.0) or 0.0)
    rank = int(r.get("rank", 0) or 0)
    total = int(points_df.shape[0])
    return (pts, rank, total)


def render(ctx: dict) -> None:
    st.header("👤 Joueurs")
    st.caption("Recherche dans hockey.players.csv + contrat (puckpedia) + classement (points_periods) + comparatif.")

    data_dir = _data_dir(ctx)
    season = _season(ctx)

    players_db = load_players_db(data_dir)
    contracts = load_puckpedia_contracts(data_dir)
    points_df = load_points(data_dir, season)

    if players_db is None or players_db.empty:
        st.error("Players DB introuvable ou vide. Assure-toi d’avoir `data/hockey.players.csv`.")
        st.code(os.path.join(data_dir, "hockey.players.csv"))
        return

    # UI - search
    st.markdown("### 🔎 Recherche joueur")
    q = st.text_input("Nom du joueur (ex: Leon Draisaitl, Marner, Savoie)", value="", key="players_search_q")

    # suggestion list
    # filtre léger pour éviter de lagger
    df_disp = players_db[["_display_name", "_name_key", "_nhl_id"]].copy()
    if q.strip():
        qs = _norm_name(q)
        # match contenant
        mask = df_disp["_display_name"].astype(str).str.lower().str.contains(q.strip().lower(), na=False)
        # plus robuste sur clé
        if qs:
            mask = mask | df_disp["_name_key"].astype(str).str.contains(qs, na=False)
        cand = df_disp.loc[mask].head(200)
    else:
        cand = df_disp.head(50)

    opts = cand["_display_name"].tolist()
    if not opts:
        st.info("Aucun joueur trouvé avec ce filtre.")
        return

    sel_name = st.selectbox("Choisir un joueur", options=opts, index=0, key="players_search_pick")
    sel_key = _norm_name(sel_name)

    # profile joueur
    st.divider()
    st.markdown("### 🧾 Fiche joueur")

    prow = _get_player_row(players_db, sel_key)
    crow = _get_contract_row(contracts, sel_key)
    prof = _extract_profile(prow, crow, points_df)
    prk = _resolve_points_rank(points_df, sel_key)

    _render_profile_card("Profil", prof, points_rank=prk)

    # comparatif
    st.divider()
    st.markdown("### ⚖️ Comparatif (2 joueurs)")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        a_name = st.selectbox(
            "Joueur A",
            options=players_db["_display_name"].head(2000).tolist(),  # limite pour performance
            index=min(0, 1999),
            key="cmp_a",
        )
    with col2:
        b_name = st.selectbox(
            "Joueur B",
            options=players_db["_display_name"].head(2000).tolist(),
            index=min(1, 1999),
            key="cmp_b",
        )

    a_key = _norm_name(a_name)
    b_key = _norm_name(b_name)

    a_prow = _get_player_row(players_db, a_key)
    b_prow = _get_player_row(players_db, b_key)
    a_crow = _get_contract_row(contracts, a_key)
    b_crow = _get_contract_row(contracts, b_key)

    a_prof = _extract_profile(a_prow, a_crow, points_df)
    b_prof = _extract_profile(b_prow, b_crow, points_df)

    a_prk = _resolve_points_rank(points_df, a_key)
    b_prk = _resolve_points_rank(points_df, b_key)

    colA, colB = st.columns(2, gap="large")
    with colA:
        _render_profile_card("Joueur A", a_prof, points_rank=a_prk)
    with colB:
        _render_profile_card("Joueur B", b_prof, points_rank=b_prk)

    # debug minimal (pas de ctx brut)
    with st.expander("🧪 Debug (data sources)", expanded=False):
        st.write("Players DB:", players_db.attrs.get("__path__", ""))
        if not contracts.empty:
            st.write("Puckpedia:", contracts.attrs.get("__path__", ""))
        else:
            st.write("Puckpedia: (absent)")
        if points_df is not None and not points_df.empty:
            st.write("Points:", points_df.attrs.get("__path__", ""))
        else:
            st.write("Points: (absent)")
