import streamlit as st
import pandas as pd
from services.storage import path_roster, safe_read_csv, path_players_db
from services.players_db import load_players_map, norm_player_key, country_to_flag_emoji

ROSTER_COLS = {
    "owner":"Propriétaire",
    "player":"Joueur",
    "pos":"Pos",
    "team":"Equipe",
    "salary":"Salaire",
    "level":"Level",
    "status":"Statut",
    "slot":"Slot",
    "ir_date":"IR Date",
}

def _fmt_money(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x).strip().replace("$","").replace(" ","").replace(" ","")
        if s == "":
            return ""
        v = float(s.replace(",", ""))
        return f"{int(round(v)):,}".replace(",", " ") + " $"
    except Exception:
        return str(x)

def _slot_bucket(slot_val: str, statut_val: str = "") -> str:
    s = str(slot_val or "").strip().lower()
    t = str(statut_val or "").strip().lower()
    if "actif" in s:
        return "ACTIFS"
    if "banc" in s:
        return "BANC"
    if s == "ir" or "inj" in s or "bless" in s:
        return "IR"
    if "mineur" in s or "minor" in s or "ahl" in s or "farm" in s:
        return "MINEUR"
    if "ir" in t or "inj" in t or "bless" in t:
        return "IR"
    if "mineur" in t or "ahl" in t:
        return "MINEUR"
    if "banc" in t:
        return "BANC"
    return "ACTIFS"

def roster_click_list(df: pd.DataFrame, title: str, *, players_map: dict) -> None:
    st.markdown(f"### {title}")
    if df is None or df.empty:
        st.caption("Aucun joueur.")
        return

    h1, h2, h3 = st.columns([7, 2, 2])
    with h1:
        st.markdown('<div class="muted nowrap">Joueur</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="muted nowrap">Pos</div>', unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="muted nowrap right">Salaire</div>', unsafe_allow_html=True)

    for idx, row in df.iterrows():
        name = str(row.get(ROSTER_COLS["player"]) or "").strip()
        k = norm_player_key(name)
        cc = ""
        if k and k in players_map:
            cc = str(players_map[k].get("country") or "").strip().upper()
        flag = country_to_flag_emoji(cc)

        pos = str(row.get(ROSTER_COLS["pos"]) or "").strip()
        sal = row.get(ROSTER_COLS["salary"])

        c1, c2, c3 = st.columns([7, 2, 2])
        with c1:
            # For now: click just highlights (no dialog yet)
            if st.button(f"{flag}  {name}" if flag else name, key=f"{title}__p__{idx}"):
                st.session_state["align_selected_player"] = name
        with c2:
            st.markdown(f'<div class="nowrap small">{pos}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="nowrap right small">{_fmt_money(sal)}</div>', unsafe_allow_html=True)

def render(ctx: dict) -> None:
    st.header("🧾 Alignement")

    season = ctx.get("season")
    roster_path = path_roster(season)
    st.caption(f"Roster: {roster_path}")

    df = safe_read_csv(roster_path)
    if df.empty:
        st.warning("Roster CSV manquant ou vide.")
        return

    # Validate minimal columns
    needed = [ROSTER_COLS["owner"], ROSTER_COLS["player"], ROSTER_COLS["pos"], ROSTER_COLS["salary"], ROSTER_COLS["slot"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error("Colonnes manquantes dans equipes_joueurs: " + ", ".join(missing))
        st.caption("Colonnes détectées: " + ", ".join([str(c) for c in df.columns]))
        return

    players_map = load_players_map(path_players_db())

    owners = sorted([x for x in df[ROSTER_COLS["owner"]].dropna().astype(str).unique() if str(x).strip()])
    owner = st.selectbox("Équipe", owners, index=0) if owners else ""
    view = df[df[ROSTER_COLS["owner"]].astype(str).eq(owner)].copy() if owner else df.copy()

    statut_col = ROSTER_COLS["status"] if ROSTER_COLS["status"] in view.columns else ""
    view["_bucket"] = view.apply(lambda r: _slot_bucket(r.get(ROSTER_COLS["slot"]), r.get(statut_col, "")), axis=1)

    actifs = view[view["_bucket"].eq("ACTIFS")].copy()
    banc = view[view["_bucket"].eq("BANC")].copy()
    ir = view[view["_bucket"].eq("IR")].copy()
    mineur = view[view["_bucket"].eq("MINEUR")].copy()

    left, center, right = st.columns([1.1, 1.1, 1.1])
    with left:
        roster_click_list(actifs, "⭐ Actifs", players_map=players_map)
    with center:
        roster_click_list(banc, "🪑 Banc", players_map=players_map)
        st.divider()
        roster_click_list(ir, "🩹 IR", players_map=players_map)
    with right:
        roster_click_list(mineur, "🧊 Mineur", players_map=players_map)

    if st.session_state.get("align_selected_player"):
        st.info(f"Sélection: {st.session_state['align_selected_player']}")
