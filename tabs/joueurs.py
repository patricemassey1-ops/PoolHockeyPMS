# =====================================================
# tabs/joueurs.py — Onglet Joueurs (PRO)
#   ✅ Recherche typeahead (prénom OU nom)
#   ✅ Filtres: Équipe NHL, Position, Level (ELC/STD)
#   ✅ Photo headshot NHL via nhl_id (déjà dans hockey.players.csv)
#   ✅ Stats LIVE via NHL API (landing) en TABLE PRO (skater/goalie)
#   ✅ Utilise /data/hockey.players.csv (dans ton repo GitHub)
# =====================================================

import os
import re
import requests
import pandas as pd
import streamlit as st

DATA_DIR = "data"
PLAYERS_PATH = os.path.join(DATA_DIR, "hockey.players.csv")


# ----------------------------
# Load local players DB
# ----------------------------
@st.cache_data(show_spinner=False)
def load_players_db(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Colonnes attendues selon ton header (on garantit l'existence)
    for c in ["Player", "Team", "Position", "Level", "Age", "Flag", "Country", "Cap Hit", "Jersey#", "H(f)", "W(lbs)"]:
        if c not in df.columns:
            df[c] = ""

    # nhl_id: clé essentielle pour headshot + API live
    if "nhl_id" in df.columns:
        df["nhl_id"] = pd.to_numeric(df["nhl_id"], errors="coerce").astype("Int64")
    else:
        df["nhl_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # Normalise Level
    df["Level"] = df["Level"].astype(str).str.strip().str.upper().replace({"0": "", "NAN": ""})

    # Normalise Team/Position string
    df["Team"] = df["Team"].astype(str).str.strip()
    df["Position"] = df["Position"].astype(str).str.strip().str.upper()

    # Petit tri stable
    df["Player"] = df["Player"].astype(str).str.strip()

    return df


# ----------------------------
# Helpers (name / formatting)
# ----------------------------
def player_last_first_to_first_last(name: str) -> str:
    """'Zucker, Jason' -> 'Jason Zucker'."""
    s = str(name or "").strip()
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        if first:
            return f"{first} {last}".strip()
    return s


def safe_str(x) -> str:
    s = "" if x is None else str(x)
    return s if s.strip() else "—"


def _is_goalie(r: pd.Series) -> bool:
    # Selon tes données: Position = "G" pour gardien
    pos = str(r.get("Position", "") or "").strip().upper()
    sr_type = str(r.get("sr_position_type", "") or "").strip().lower() if "sr_position_type" in r.index else ""
    if pos == "G" or pos.startswith("G"):
        return True
    if "goal" in sr_type:
        return True
    return False


def _fmt_pct(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        s = str(v).strip()
        if not s:
            return "—"
        f = float(s)
        # 0.913 -> .913 style hockey ; 91.3 -> 91.3%
        if f > 1.0:
            return f"{f:.1f}%"
        if 0.0 <= f <= 1.0:
            return f"{f:.3f}".lstrip("0")
        return s
    except Exception:
        return str(v) if str(v).strip() else "—"


def _fmt_num(v, digits=0):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        s = str(v).strip()
        if not s:
            return "—"
        f = float(s)
        if digits == 0:
            return f"{int(round(f))}"
        return f"{f:.{digits}f}"
    except Exception:
        return str(v) if str(v).strip() else "—"


def headshot_url(player_id: int, size: int = 168) -> str:
    return f"https://cms.nhl.bamgrid.com/images/headshots/current/{size}x{size}/{int(player_id)}.jpg"


# ----------------------------
# NHL API (landing)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=600)  # 10 minutes
def nhl_player_landing(player_id: int) -> dict:
    url = f"https://api-web.nhle.com/v1/player/{int(player_id)}/landing"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return r.json()


# ----------------------------
# API -> PRO TABLE extraction (robust)
# ----------------------------
def _deep_find_lists(obj, key_hint=None, max_lists=20):
    """
    Parcourt un JSON dict/list et retourne des (path, list_value) pour toutes les listes de dict.
    Si key_hint est fourni, on préfère les chemins contenant ce hint.
    """
    found = []

    def rec(x, path=""):
        nonlocal found
        if len(found) >= max_lists:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}" if path else k
                if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
                    found.append((p, v))
                else:
                    rec(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x[:50]):
                rec(v, f"{path}[{i}]")

    rec(obj)

    if key_hint:
        hint = key_hint.lower()
        found.sort(key=lambda t: (0 if hint in t[0].lower() else 1, len(t[0])))
    else:
        found.sort(key=lambda t: len(t[0]))

    return found


def _pick_season_list(landing: dict):
    """
    Essaie de trouver la liste qui ressemble à des stats par saison.
    On teste d'abord quelques clés typiques, sinon on fait une recherche deep.
    """
    candidates = []

    # Clés fréquentes possibles
    for k in ["seasonTotals", "seasonTotal", "seasons", "statsBySeason", "playerStatsBySeason"]:
        v = landing.get(k)
        if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
            candidates.append((k, v))

    if candidates:
        # Priorise celle qui a le plus d'entrées
        candidates.sort(key=lambda t: len(t[1]), reverse=True)
        return candidates[0][1]

    # Sinon: search deep
    found = _deep_find_lists(landing, key_hint="season")
    # On préfère une liste où plusieurs éléments ont une clé 'season' ou 'seasonId' ou 'gameType'
    for _, lst in found:
        score = 0
        for it in lst[:5]:
            keys = set(it.keys())
            if "season" in keys or "seasonId" in keys:
                score += 2
            if "gameTypeId" in keys or "gameType" in keys or "gameTypeAbbrev" in keys:
                score += 1
            if "gamesPlayed" in keys or "gp" in keys:
                score += 1
        if score >= 6:
            return lst

    # fallback: première liste plausible
    return found[0][1] if found else []


def _to_number(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        # handle ".913"
        if s.startswith("."):
            s = "0" + s
        return float(s)
    except Exception:
        return None


def _map_skater_row(it: dict) -> dict:
    # Accepte variations: gp/gamesPlayed, pts/points
    season = it.get("season") or it.get("seasonId") or it.get("seasonYear") or ""
    team = it.get("teamAbbrev") or it.get("team") or it.get("teamName") or it.get("teamAbbreviation") or ""
    gt = it.get("gameTypeAbbrev") or it.get("gameType") or it.get("gameTypeId") or ""
    league = it.get("leagueAbbrev") or it.get("league") or ""

    gp = it.get("gamesPlayed", it.get("gp"))
    g = it.get("goals", it.get("g"))
    a = it.get("assists", it.get("a"))
    pts = it.get("points", it.get("pts"))
    pm = it.get("plusMinus", it.get("plus_minus", it.get("plusMinusValue")))
    pim = it.get("pim", it.get("penaltyMinutes"))
    ppg = it.get("powerPlayGoals", it.get("ppGoals"))
    ppa = it.get("powerPlayAssists", it.get("ppAssists"))
    ppp = it.get("powerPlayPoints", it.get("ppPoints"))
    shp = it.get("shortHandedPoints", it.get("shPoints"))
    sog = it.get("shots", it.get("sog", it.get("shotsOnGoal")))
    sh_pct = it.get("shootingPctg", it.get("shootingPercentage"))

    return {
        "Season": str(season),
        "Team": str(team),
        "Type": str(gt),
        "Lg": str(league),
        "GP": _to_number(gp),
        "G": _to_number(g),
        "A": _to_number(a),
        "PTS": _to_number(pts),
        "+/-": _to_number(pm),
        "PIM": _to_number(pim),
        "PPP": _to_number(ppp),
        "PPG": _to_number(ppg),
        "PPA": _to_number(ppa),
        "SHP": _to_number(shp),
        "SOG": _to_number(sog),
        "Sh%": _to_number(sh_pct),
    }


def _map_goalie_row(it: dict) -> dict:
    season = it.get("season") or it.get("seasonId") or it.get("seasonYear") or ""
    team = it.get("teamAbbrev") or it.get("team") or it.get("teamName") or it.get("teamAbbreviation") or ""
    gt = it.get("gameTypeAbbrev") or it.get("gameType") or it.get("gameTypeId") or ""
    league = it.get("leagueAbbrev") or it.get("league") or ""

    gp = it.get("gamesPlayed", it.get("gp"))
    w = it.get("wins", it.get("w"))
    l = it.get("losses", it.get("l"))
    ot = it.get("otLosses", it.get("otl", it.get("ties")))
    gaa = it.get("goalsAgainstAvg", it.get("goalsAgainstAverage", it.get("gaa")))
    svp = it.get("savePctg", it.get("savePercentage", it.get("svPct", it.get("sv_pct"))))
    so = it.get("shutouts", it.get("so"))
    sa = it.get("shotsAgainst", it.get("sa"))
    saves = it.get("saves", it.get("sv"))
    ga = it.get("goalsAgainst", it.get("ga"))

    return {
        "Season": str(season),
        "Team": str(team),
        "Type": str(gt),
        "Lg": str(league),
        "GP": _to_number(gp),
        "W": _to_number(w),
        "L": _to_number(l),
        "OTL": _to_number(ot),
        "SV%": _to_number(svp),
        "GAA": _to_number(gaa),
        "SO": _to_number(so),
        "Saves": _to_number(saves),
        "SA": _to_number(sa),
        "GA": _to_number(ga),
    }


def render_api_pro_tables(landing: dict, goalie: bool):
    """
    Affiche des tables pro en tentant:
    - une table "Regular Season" (game type RS si dispo)
    - une table "Playoffs" (PO si dispo)
    Sinon: affiche table unique.
    """
    season_list = _pick_season_list(landing)
    if not season_list:
        st.info("Aucune table de saisons trouvée dans la réponse API (landing).")
        st.json(landing, expanded=False)
        return

    # map -> DataFrame
    rows = []
    for it in season_list:
        try:
            rows.append(_map_goalie_row(it) if goalie else _map_skater_row(it))
        except Exception:
            continue

    if not rows:
        st.info("Impossible de mapper les stats API en table (structure différente).")
        st.json(landing, expanded=False)
        return

    df = pd.DataFrame(rows)

    # Nettoyage: drop lignes vides
    df = df.dropna(how="all")
    if df.empty:
        st.info("Table API vide après nettoyage.")
        st.json(landing, expanded=False)
        return

    # Formattage colonnes numériques
    num_cols = [c for c in df.columns if c not in ["Season", "Team", "Type", "Lg"]]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Heuristique de Type: si c'est numérique (gameTypeId), on le laisse mais on peut mieux
    # On essaie de normaliser: si "R" / "P" existent, sinon on garde tel quel.
    # Beaucoup de réponses utilisent "R" (regular) ou "P" (playoffs) ou "2"/"3" etc.
    def _type_norm(x):
        s = str(x or "").strip().upper()
        if s in ["R", "RS", "REG", "REGULAR", "2"]:
            return "RS"
        if s in ["P", "PO", "PLAYOFF", "PLAYOFFS", "3"]:
            return "PO"
        return s

    df["Type"] = df["Type"].apply(_type_norm)

    # Tri: saison desc
    df["_season_sort"] = pd.to_numeric(df["Season"].str.extract(r"(\d{4})")[0], errors="coerce")
    df = df.sort_values(by=["_season_sort", "Type", "Team"], ascending=[False, True, True]).drop(columns=["_season_sort"])

    # Split RS / PO si possible
    rs = df[df["Type"] == "RS"].copy()
    po = df[df["Type"] == "PO"].copy()

    if not rs.empty:
        st.markdown("#### Regular Season (API)")
        st.dataframe(_format_api_table(rs, goalie), use_container_width=True)

    if not po.empty:
        st.markdown("#### Playoffs (API)")
        st.dataframe(_format_api_table(po, goalie), use_container_width=True)

    if rs.empty and po.empty:
        st.markdown("#### Stats (API)")
        st.dataframe(_format_api_table(df, goalie), use_container_width=True)

    # FeaturedStats compact (si présent)
    featured = landing.get("featuredStats")
    if featured:
        with st.expander("Featured stats (API) — détail", expanded=False):
            st.json(featured, expanded=False)


def _format_api_table(df: pd.DataFrame, goalie: bool) -> pd.DataFrame:
    """
    Formatte un DF API en table lisible (arrondis + colonnes pertinentes).
    """
    df2 = df.copy()

    if goalie:
        keep = ["Season", "Team", "Type", "GP", "W", "L", "OTL", "SV%", "GAA", "SO", "Saves", "SA", "GA"]
        keep = [c for c in keep if c in df2.columns]
        df2 = df2[keep]
        if "SV%" in df2.columns:
            df2["SV%"] = df2["SV%"].apply(lambda x: _fmt_pct(x))
        if "GAA" in df2.columns:
            df2["GAA"] = df2["GAA"].apply(lambda x: _fmt_num(x, 2))
        for c in ["GP", "W", "L", "OTL", "SO", "Saves", "SA", "GA"]:
            if c in df2.columns:
                df2[c] = df2[c].apply(lambda x: _fmt_num(x, 0))
        return df2

    else:
        keep = ["Season", "Team", "Type", "GP", "G", "A", "PTS", "+/-", "PIM", "PPP", "PPG", "PPA", "SHP", "SOG", "Sh%"]
        keep = [c for c in keep if c in df2.columns]
        df2 = df2[keep]
        for c in ["GP", "G", "A", "PTS", "+/-", "PIM", "PPP", "PPG", "PPA", "SHP", "SOG"]:
            if c in df2.columns:
                df2[c] = df2[c].apply(lambda x: _fmt_num(x, 0))
        if "Sh%" in df2.columns:
            df2["Sh%"] = df2["Sh%"].apply(lambda x: _fmt_num(x, 1))
        return df2


# ----------------------------
# Local PRO stats (CSV) panels
# ----------------------------
def _get_first_present(r: pd.Series, keys: list[str], default=None):
    for k in keys:
        if k in r.index:
            v = r.get(k)
            if pd.notna(v) and str(v).strip() != "":
                return v
    return default


def render_local_pro_panel(row: pd.DataFrame):
    r = row.iloc[0]
    goalie = _is_goalie(r)

    gp = _get_first_present(r, ["nhl_gp", "NHL GP", "GP"], default=None)

    if goalie:
        w = _get_first_present(r, ["nhl_w"], default=None)
        l = _get_first_present(r, ["nhl_l"], default=None)
        ot = _get_first_present(r, ["nhl_otl", "NHL OT"], default=None)
        gaa = _get_first_present(r, ["nhl_gaa", "NHL GAA"], default=None)
        svp = _get_first_present(r, ["nhl_sv_pct", "NHL SV%"], default=None)
        so = _get_first_present(r, ["nhl_so"], default=None)
        saves = _get_first_present(r, ["nhl_saves"], default=None)
        sa = _get_first_present(r, ["nhl_sa"], default=None)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("GP", _fmt_num(gp, 0))
        c2.metric("W-L-OT", f"{_fmt_num(w,0)}-{_fmt_num(l,0)}-{_fmt_num(ot,0)}")
        c3.metric("SV%", _fmt_pct(svp))
        c4.metric("GAA", _fmt_num(gaa, 2))
        c5.metric("SO", _fmt_num(so, 0))
        c6.metric("Saves/SA", f"{_fmt_num(saves,0)}/{_fmt_num(sa,0)}")

    else:
        g = _get_first_present(r, ["nhl_g", "NHL G", "G"], default=None)
        a = _get_first_present(r, ["nhl_a", "NHL A", "A"], default=None)
        pts = _get_first_present(r, ["nhl_pts", "NHL P", "P"], default=None)

        pm = _get_first_present(r, ["nhl_plus_minus", "+/-"], default=None)
        pim = _get_first_present(r, ["nhl_pim", "PIM"], default=None)
        ppp = _get_first_present(r, ["nhl_ppp"], default=None)
        shp = _get_first_present(r, ["nhl_shp"], default=None)
        sog = _get_first_present(r, ["nhl_sog", "SOG"], default=None)
        sh = _get_first_present(r, ["nhl_sh_pct"], default=None)
        hits = _get_first_present(r, ["nhl_hits"], default=None)
        blk = _get_first_present(r, ["nhl_blk"], default=None)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("GP", _fmt_num(gp, 0))
        c2.metric("G", _fmt_num(g, 0))
        c3.metric("A", _fmt_num(a, 0))
        c4.metric("PTS", _fmt_num(pts, 0))
        c5.metric("SOG", _fmt_num(sog, 0))
        c6.metric("Sh%", _fmt_num(sh, 1) if sh is not None else "—")

        c7, c8, c9, c10, c11, c12 = st.columns(6)
        c7.metric("+/-", _fmt_num(pm, 0))
        c8.metric("PIM", _fmt_num(pim, 0))
        c9.metric("PPP", _fmt_num(ppp, 0))
        c10.metric("SHP", _fmt_num(shp, 0))
        c11.metric("HIT", _fmt_num(hits, 0))
        c12.metric("BLK", _fmt_num(blk, 0))


# ----------------------------
# Main render for tab
# ----------------------------
def render_tab_joueurs():
    st.header("🏒 Joueurs")

    df = load_players_db(PLAYERS_PATH)
    if df.empty:
        st.error("❌ data/hockey.players.csv introuvable ou vide.")
        st.info("Assure-toi que le fichier est bien dans /data/ (repo GitHub).")
        return

    # ---- Filtres
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        teams = sorted([t for t in df["Team"].dropna().astype(str).unique() if t.strip()])
        team_pick = st.selectbox("Équipe NHL", ["Toutes"] + teams, index=0)

    with c2:
        poss = sorted([p for p in df["Position"].dropna().astype(str).unique() if p.strip()])
        pos_pick = st.selectbox("Position", ["Toutes"] + poss, index=0)

    with c3:
        level_pick = st.selectbox("Level", ["Tous", "ELC", "STD"], index=0)

    filt = df.copy()
    if team_pick != "Toutes":
        filt = filt[filt["Team"].astype(str) == team_pick]
    if pos_pick != "Toutes":
        filt = filt[filt["Position"].astype(str).str.upper() == str(pos_pick).upper()]
    if level_pick != "Tous":
        filt = filt[filt["Level"].astype(str).str.upper() == level_pick]

    if filt.empty:
        st.warning("Aucun joueur ne correspond aux filtres.")
        return

    # ---- Recherche typeahead
    # selectbox est searchable -> tape prénom OU nom
    filt = filt.copy()
    filt["__display"] = filt["Player"].astype(str).str.strip()
    filt["__alt"] = filt["Player"].astype(str).apply(player_last_first_to_first_last)
    filt["__opt"] = filt["__display"] + "  |  " + filt["__alt"]
    options = filt.sort_values("__display")["__opt"].tolist()

    picked = st.selectbox(
        "Rechercher un joueur (tape quelques lettres du prénom OU du nom)",
        options,
        index=0
    )

    chosen_display = picked.split("|")[0].strip()
    row = filt[filt["__display"] == chosen_display].head(1)
    if row.empty:
        st.warning("Sélection invalide.")
        return

    r = row.iloc[0]
    goalie = _is_goalie(r)

    # ---- Fiche joueur (header)
    player_name = safe_str(r.get("Player"))
    player_name_alt = safe_str(player_last_first_to_first_last(player_name))
    st.subheader(player_name_alt if player_name_alt != "—" else player_name)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Équipe", safe_str(r.get("Team")))
    m2.metric("Position", safe_str(r.get("Position")))
    m3.metric("Level", safe_str(r.get("Level")))
    m4.metric("Âge", safe_str(r.get("Age")))
    m5.metric("Cap Hit", safe_str(r.get("Cap Hit")))

    # ---- Photo + drapeau + infos
    left, right = st.columns([1, 2])
    nhl_id = r.get("nhl_id")
    flag_url = str(r.get("Flag") or "").strip()

    with left:
        if pd.notna(nhl_id):
            st.image(headshot_url(int(nhl_id), size=168), use_container_width=True)
            st.caption(f"NHL ID: {int(nhl_id)}")
        else:
            st.info("Pas de nhl_id → pas de headshot / API.")

        if flag_url.startswith("http"):
            st.image(flag_url, caption=safe_str(r.get("Country")), width=90)

    with right:
        info_cols = st.columns(4)
        info_cols[0].metric("Jersey#", safe_str(r.get("Jersey#")))
        info_cols[1].metric("Taille", safe_str(r.get("H(f)")))
        info_cols[2].metric("Poids", safe_str(r.get("W(lbs)")))
        info_cols[3].metric("Statut", safe_str(r.get("Status")))

        st.markdown("### 📊 Stats (PRO — CSV)")
        render_local_pro_panel(row)

    # ---- API PRO TABLE
    st.markdown("### ⚡ Stats NHL LIVE (API — table pro)")
    if pd.notna(nhl_id):
        try:
            landing = nhl_player_landing(int(nhl_id))

            top = st.columns(4)
            top[0].metric("Prénom", safe_str(landing.get("firstName", {}).get("default")))
            top[1].metric("Nom", safe_str(landing.get("lastName", {}).get("default")))
            top[2].metric("Tire", safe_str(landing.get("shootsCatches")))
            top[3].metric("Numéro", safe_str(landing.get("sweaterNumber")))

            render_api_pro_tables(landing, goalie=goalie)

        except Exception as e:
            st.warning(f"API NHL indisponible (ID {int(nhl_id)}). Détail: {e}")
    else:
        st.info("Ajoute / corrige la colonne nhl_id pour activer la photo + stats live.")


# =====================================================
# Entry point attendu par ton app: joueurs.render(ctx)
# =====================================================
def render(ctx: dict):
    # ctx non requis ici, mais on garde la signature cohérente avec admin.render(ctx)
    render_tab_joueurs()
