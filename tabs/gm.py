# tabs/gm.py
from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------------------------------
# Cache helpers (speed: avoid re-reading CSV on every rerun)
# -----------------------------------------------------
def _file_sig(path: str) -> tuple[int, int]:
    try:
        st_ = os.stat(path)
        return (int(st_.st_mtime_ns), int(st_.st_size))
    except Exception:
        return (0, 0)

@st.cache_data(show_spinner=False, persist="disk", max_entries=32)
def _read_csv_cached(path: str, sig: tuple[int, int]) -> pd.DataFrame:
    """CSV loader cached by (path, mtime_ns, size).

    Prefer fast C engine; if parsing fails, retry with python engine.
    """
    try:
        # Fast path (C engine). low_memory is supported here.
        return pd.read_csv(path, engine="c", low_memory=False, on_bad_lines="skip")
    except Exception:
        # Fallback path (python engine). low_memory is NOT supported here.
        return pd.read_csv(path, engine="python", on_bad_lines="skip")



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


def _first_existing(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


def _safe_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV safely and fast.

    - Returns an empty DataFrame if path is empty or missing.
    - Uses cached reader (C engine first, python fallback).
    """
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return _read_csv_cached(path, _file_sig(path))
    except Exception:
        return pd.DataFrame()


def _guess_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
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


def _safe_str(v, default: str = "—") -> str:
    if v is None:
        return default
    if isinstance(v, float) and pd.isna(v):
        return default
    s = str(v).strip()
    return s if s else default


def _detect_bucket(label: str) -> str:
    s = str(label or "").strip().lower()
    if not s:
        return "Autre"
    if any(k in s for k in ["actif", "active", "starter", "lineup", "alignement", "a:"]):
        return "Actifs"
    if any(k in s for k in ["banc", "bench", "reserve", "res", "bn"]):
        return "Banc"
    if any(k in s for k in ["ir", "inj", "injury", "ltir"]):
        return "IR"
    if any(k in s for k in ["mineur", "minor", "ahl", "farm", "prospect"]):
        return "Mineur"
    return "Autre"


def _resolve_plafonds(ctx: dict) -> Dict[str, float]:
    pl = ctx.get("plafonds")
    if isinstance(pl, dict):
        out = {}
        for k, v in pl.items():
            f = _as_float(v)
            if f is not None:
                out[str(k)] = f
        return out

    if isinstance(pl, pd.DataFrame) and not pl.empty:
        owner_col = _guess_col(pl, ["Owner", "Proprietaire", "Équipe", "Equipe", "Team"])
        cap_col = _guess_col(pl, ["Plafond", "Cap", "Cap Limit", "Limit", "Salary Cap"])
        if owner_col and cap_col:
            out = {}
            for _, r in pl.iterrows():
                o = str(r.get(owner_col, "") or "").strip()
                f = _as_float(r.get(cap_col))
                if o and f is not None:
                    out[o] = f
            return out

    return {}


def _asset_dirs(data_dir: str) -> List[str]:
    return [
        os.path.join("assets", "previews"),
        data_dir,
        ".",
    ]


def _find_image(filename: str, data_dir: str) -> str:
    for d in _asset_dirs(data_dir):
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return ""


def _find_team_logo(owner: str, data_dir: str) -> str:
    """
    Cherche un logo d'équipe correspondant au GM/Owner.
    - Match: filename contient owner normalisé
    - Priorité: assets/previews puis data
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
            # évite gm_logo
            if fn.lower() in ["gm_logo.png", "logo_pool.png", "logo_pool.jpg", "logo_pool.webp"]:
                continue
            fkey = _norm(fn.replace("_logo", "").replace("logo", ""))
            if owner_key and owner_key in fkey:
                candidates.append(os.path.join(d, fn))

    # si rien: tenter pattern <Owner>_Logo.png
    if not candidates:
        for d in _asset_dirs(data_dir):
            p = os.path.join(d, f"{owner}_Logo.png")
            if os.path.exists(p):
                return p
        return ""

    # choisir le plus court (souvent le plus “exact”)
    candidates = sorted(candidates, key=lambda x: len(os.path.basename(x)))
    return candidates[0]


def _level_badge(level: str) -> str:
    lvl = (level or "—").upper().strip()
    if lvl == "ELC":
        return "ELC 🟢"
    if lvl == "STD":
        return "STD 🔵"
    return "—"


def _parse_year(s: str) -> Optional[int]:
    try:
        m = re.search(r"(19|20)\d{2}", str(s))
        if not m:
            return None
        return int(m.group(0))
    except Exception:
        return None


def _season_end_year(season: str) -> Optional[int]:
    # "2025-2026" -> 2026
    try:
        parts = re.split(r"[-–]", season)
        if len(parts) >= 2:
            return int(parts[1])
        return int(parts[0])
    except Exception:
        return None


# =====================================================
# Loaders
# =====================================================
@st.cache_data(show_spinner=False, persist="disk", max_entries=8)
def load_equipes_joueurs(data_dir: str, season: str) -> pd.DataFrame:
    path = _first_existing(
        os.path.join(data_dir, f"equipes_joueurs_{season}.csv"),
        os.path.join(data_dir, f"equipes_joueurs_{season.replace('–','-')}.csv"),
    )
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.attrs["__path__"] = path

    player_col = _guess_col(df, ["Joueur", "Player", "Name", "player", "name"])
    if not player_col:
        player_col = df.columns[0]
    df["_player"] = df[player_col].astype(str)
    df["_name_key"] = df["_player"].astype(str).map(_norm_player_key)

    owner_col = _guess_col(df, ["Proprietaire", "Owner", "Équipe", "Equipe", "Team", "GM"])
    df["_owner"] = df[owner_col].astype(str) if owner_col else ""

    slot_col = _guess_col(df, ["Statut", "Status", "Slot", "Position Slot", "Roster Slot", "Type"])
    df["_slot"] = df[slot_col].astype(str) if slot_col else ""

    club_col = _guess_col(df, ["Club", "Ligue", "League", "Groupe", "Roster"])
    if club_col and "_slot" in df.columns:
        df["_slot"] = df["_slot"].where(df["_slot"].astype(str).str.strip().ne(""), df[club_col].astype(str))

    return df


@st.cache_data(show_spinner=False, persist="disk", max_entries=8)
def load_players_db(data_dir: str) -> pd.DataFrame:
    path = _first_existing(
        os.path.join(data_dir, "hockey.players.csv"),
        os.path.join(data_dir, "Hockey.Players.csv"),
        os.path.join(data_dir, "data", "hockey.players.csv"),
    )
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.attrs["__path__"] = path

    name_col = _guess_col(df, ["Joueur", "Player", "Name", "name"])
    if not name_col:
        name_col = df.columns[0]
    df["_name_key"] = df[name_col].astype(str).map(_norm_player_key)
    df["_display_name"] = df[name_col].astype(str)

    if "Cap Hit" not in df.columns:
        df["Cap Hit"] = df.get("Salary", df.get("AAV", ""))
    if "Level" not in df.columns:
        df["Level"] = ""
    if "Pos" not in df.columns:
        df["Pos"] = df.get("Position", "")
    if "Team" not in df.columns:
        df["Team"] = df.get("Equipe", "")

    nhl_col = _guess_col(df, ["NHL ID", "NHL_ID", "nhl_id", "playerId", "player_id", "PlayerID", "NHLID"])
    df["_nhl_id"] = df[nhl_col].astype(str) if nhl_col else ""

    if "Country" not in df.columns:
        df["Country"] = ""

    return df


@st.cache_data(show_spinner=False, persist="disk", max_entries=8)
def load_contracts(data_dir: str) -> pd.DataFrame:
    path = _first_existing(
        os.path.join(data_dir, "puckpedia.contracts.csv"),
        os.path.join(data_dir, "PuckPedia2025_26.csv"),
        os.path.join(data_dir, "puckpedia2025_26.csv"),
        os.path.join(data_dir, "puckpedia2025_26_contracts.csv"),
        os.path.join(data_dir, "puckpedia_contracts.csv"),
    )
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.attrs["__path__"] = path
    name_col = _guess_col(df, ["Player", "Joueur", "Name", "name"])
    if not name_col:
        name_col = df.columns[0]
    df["_name_key"] = df[name_col].astype(str).map(_norm_player_key)
    return df


# =====================================================
# Merge + compute
# =====================================================
def _merge_roster(roster: pd.DataFrame, players: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    out = roster.copy()

    if players is not None and not players.empty:
        keep_cols = ["_name_key", "_display_name", "Pos", "Team", "Country", "Level", "Cap Hit", "_nhl_id"]
        keep_cols = [c for c in keep_cols if c in players.columns]
        p = players[keep_cols].drop_duplicates("_name_key", keep="first")
        out = out.merge(p, on="_name_key", how="left")

    if contracts is not None and not contracts.empty:
        c_keep = ["_name_key"]
        for c in ["AAV", "Cap Hit", "CapHit", "Expiry", "Expiry Year", "End", "Term", "Clause", "Clauses", "Type", "Contract Type"]:
            if c in contracts.columns and c not in c_keep:
                c_keep.append(c)
        cdf = contracts[c_keep].drop_duplicates("_name_key", keep="first")
        out = out.merge(cdf, on="_name_key", how="left", suffixes=("", "_pp"))

    if "_display_name" not in out.columns:
        out["_display_name"] = out["_player"].astype(str)

    cap_series = None
    if "Cap Hit" in out.columns:
        cap_series = out["Cap Hit"]
    elif "AAV" in out.columns:
        cap_series = out["AAV"]
    else:
        cap_series = pd.Series([""] * len(out))

    out["_cap_num"] = pd.to_numeric(
        cap_series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
        errors="coerce",
    ).fillna(0.0)

    exp = pd.Series([""] * len(out))
    for c in ["Expiry", "Expiry Year", "End"]:
        if c in out.columns:
            exp = out[c].astype(str)
            break
    out["_expiry"] = exp.replace({"nan": "", "None": ""})

    # Level robust
    lvl = out.get("Level", pd.Series([""] * len(out))).astype(str).str.upper().str.strip()
    if "Type" in out.columns:
        t = out["Type"].astype(str).str.upper()
        lvl = lvl.where(lvl.isin(["ELC", "STD"]), t.apply(lambda x: "ELC" if ("ELC" in x or "ENTRY" in x) else ""))
    elif "Contract Type" in out.columns:
        t = out["Contract Type"].astype(str).str.upper()
        lvl = lvl.where(lvl.isin(["ELC", "STD"]), t.apply(lambda x: "ELC" if ("ELC" in x or "ENTRY" in x) else ""))

    out["_level"] = lvl.where(lvl.astype(str).str.strip().ne(""), "—")
    out["_bucket"] = out["_slot"].apply(_detect_bucket)

    return out


def _build_owner_list(roster: pd.DataFrame) -> List[str]:
    if roster is None or roster.empty:
        return []
    owners = roster["_owner"].astype(str).str.strip()
    owners = owners[owners.ne("") & owners.ne("nan")]
    return sorted(owners.unique().tolist())


def _owner_summary(roster_all: pd.DataFrame, owner: str, plafonds: Dict[str, float]) -> Dict[str, Any]:
    sub = roster_all[roster_all["_owner"].astype(str).str.strip() == str(owner).strip()].copy()
    cap_total = float(sub["_cap_num"].sum()) if not sub.empty else 0.0
    counts = sub["_bucket"].value_counts().to_dict() if not sub.empty else {}

    cap_limit = plafonds.get(owner)
    return {
        "owner": owner,
        "cap_total": cap_total,
        "cap_limit": cap_limit,
        "n_total": int(len(sub)),
        "n_actifs": int(counts.get("Actifs", 0)),
        "n_banc": int(counts.get("Banc", 0)),
        "n_ir": int(counts.get("IR", 0)),
        "n_mineur": int(counts.get("Mineur", 0)),
        "df": sub,
    }


def _roster_table(df: pd.DataFrame) -> pd.DataFrame:
    show = pd.DataFrame()
    show["Joueur"] = df["_display_name"].astype(str)
    show["Pos"] = df.get("Pos", "—").astype(str)
    show["Team"] = df.get("Team", "—").astype(str)
    show["Country"] = df.get("Country", "—").astype(str)
    show["Level"] = df["_level"].astype(str).apply(_level_badge)
    show["Cap Hit"] = df["_cap_num"].apply(_money)
    show["Expiry"] = df["_expiry"].astype(str).replace({"nan": "—"}).replace({"": "—"})
    return show


def _top_cap_hits(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.sort_values("_cap_num", ascending=False).head(n).copy()
    return _roster_table(out)


def _expiring_contracts(df: pd.DataFrame, season: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    end_year = _season_end_year(season)
    if end_year is None:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["_exp_year"] = tmp["_expiry"].apply(_parse_year)
    tmp = tmp[tmp["_exp_year"].notna()].copy()
    tmp = tmp[tmp["_exp_year"].astype(int) <= int(end_year)].copy()
    tmp = tmp.sort_values(["_exp_year", "_cap_num"], ascending=[True, False])
    out = _roster_table(tmp)
    out.insert(0, "ExpiryYear", tmp["_exp_year"].astype(int).values)
    return out


# =====================================================
# UI
# =====================================================
def _render_header(owner: str, data_dir: str) -> None:
    gm_logo = _find_image("gm_logo.png", data_dir)
    team_logo = _find_team_logo(owner, data_dir)

    c1, c2, c3 = st.columns([1, 3, 1], gap="large")
    with c1:
        if gm_logo:
            try:
                st.image(gm_logo, width=90)
            except Exception:
                pass
    with c2:
        st.title("🧑‍💼 GM")
        st.caption("Vue pro : roster, cap hit, contrats, et comparatif.")
    with c3:
        if team_logo:
            try:
                st.image(team_logo, width=90)
            except Exception:
                pass


def _render_overview(summary: Dict[str, Any], season: str) -> None:
    cap_total = summary["cap_total"]
    cap_limit = summary.get("cap_limit")

    a, b, c, d = st.columns(4)
    with a:
        st.metric("Cap hit total", _money(cap_total))
    with b:
        st.metric("Joueurs", str(summary["n_total"]))
    with c:
        st.metric("Actifs", str(summary["n_actifs"]))
    with d:
        st.metric("Mineur", str(summary["n_mineur"]))

    if cap_limit is not None and cap_limit > 0:
        st.caption(f"Plafond: {_money(cap_limit)}")
        ratio = min(1.0, float(cap_total) / float(cap_limit))
        st.progress(ratio)
        st.caption(f"Écart plafond: {_money(float(cap_limit) - float(cap_total))}")
    else:
        st.caption("Plafond: — (non configuré)")

    st.caption(f"Saison: **{season}**")


def _render_rosters(df: pd.DataFrame) -> None:
    # affichage structuré
    for bucket in ["Actifs", "Banc", "IR", "Mineur", "Autre"]:
        sub = df[df["_bucket"] == bucket].copy()
        if sub.empty:
            continue
        st.subheader(bucket)
        st.dataframe(_roster_table(sub.sort_values("_cap_num", ascending=False)), use_container_width=True, hide_index=True)


def _render_contracts(df: pd.DataFrame, season: str) -> None:
    st.subheader("Top 10 Cap Hits")
    top = _top_cap_hits(df, 10)
    if top.empty:
        st.info("Aucune donnée cap hit.")
    else:
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Contrats expirants (≤ fin de saison)")
    exp = _expiring_contracts(df, season)
    if exp.empty:
        st.info("Aucun contrat expirant détecté (ou puckpedia.contracts.csv absent).")
    else:
        st.dataframe(exp, use_container_width=True, hide_index=True)


def _render_compare(owners: List[str], roster_all: pd.DataFrame, plafonds: Dict[str, float], season: str, data_dir: str) -> None:
    st.subheader("⚖️ Comparatif GM vs GM")

    if len(owners) < 2:
        st.info("Il faut au moins 2 équipes dans equipes_joueurs pour comparer.")
        return

    colA, colB = st.columns(2)
    with colA:
        o1 = st.selectbox("GM A", owners, index=0, key="gm_cmp_a")
    with colB:
        o2 = st.selectbox("GM B", owners, index=1, key="gm_cmp_b")

    s1 = _owner_summary(roster_all, o1, plafonds)
    s2 = _owner_summary(roster_all, o2, plafonds)

    df_sum = pd.DataFrame(
        [
            {
                "GM": s1["owner"],
                "Cap total": _money(s1["cap_total"]),
                "Plafond": _money(s1["cap_limit"]) if s1.get("cap_limit") is not None else "—",
                "Actifs": s1["n_actifs"],
                "Banc": s1["n_banc"],
                "IR": s1["n_ir"],
                "Mineur": s1["n_mineur"],
                "Total": s1["n_total"],
            },
            {
                "GM": s2["owner"],
                "Cap total": _money(s2["cap_total"]),
                "Plafond": _money(s2["cap_limit"]) if s2.get("cap_limit") is not None else "—",
                "Actifs": s2["n_actifs"],
                "Banc": s2["n_banc"],
                "IR": s2["n_ir"],
                "Mineur": s2["n_mineur"],
                "Total": s2["n_total"],
            },
        ]
    )

    st.dataframe(df_sum, use_container_width=True, hide_index=True)

    st.divider()
    c1, c2 = st.columns(2, gap="large")

    with c1:
        _render_header(s1["owner"], data_dir)
        _render_overview(s1, season)
        st.divider()
        st.subheader("Top 10 Cap Hits (A)")
        st.dataframe(_top_cap_hits(s1["df"], 10), use_container_width=True, hide_index=True)

    with c2:
        _render_header(s2["owner"], data_dir)
        _render_overview(s2, season)
        st.divider()
        st.subheader("Top 10 Cap Hits (B)")
        st.dataframe(_top_cap_hits(s2["df"], 10), use_container_width=True, hide_index=True)


# =====================================================
# Main render
# =====================================================
def render(ctx: dict) -> None:
    data_dir = _data_dir(ctx)
    season = _season(ctx)

    equipes = load_equipes_joueurs(data_dir, season)
    if equipes is None or equipes.empty:
        st.error("Aucune donnée d'équipes. Ajoute `data/equipes_joueurs_<season>.csv` via Gestion Admin → Restore.")
        st.code(os.path.join(data_dir, f"equipes_joueurs_{season}.csv"))
        return

    players = load_players_db(data_dir)
    contracts = load_contracts(data_dir)

    roster_all = _merge_roster(equipes, players, contracts)
    owners = _build_owner_list(roster_all)
    if not owners:
        st.error("Impossible de détecter les équipes/GM (colonne Owner/Proprietaire/Equipe manquante ?).")
        st.write("Colonnes détectées:", list(equipes.columns))
        return

    plafonds = _resolve_plafonds(ctx)

    # Sélection GM
    selected_owner = st.selectbox("Choisir une équipe", owners, key="gm_owner_pick")

    # Header pro + logos
    _render_header(selected_owner, data_dir)

    summary = _owner_summary(roster_all, selected_owner, plafonds)

    # Sous-onglets pro
    t_over, t_roster, t_contracts, t_compare = st.tabs(["📌 Vue d’ensemble", "📋 Roster", "📄 Contrats", "⚖️ Comparatif"])

    with t_over:
        _render_overview(summary, season)

        st.divider()
        st.subheader("Top 10 Cap Hits")
        top = _top_cap_hits(summary["df"], 10)
        if top.empty:
            st.info("Aucune donnée cap hit.")
        else:
            st.dataframe(top, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Résumé roster")
        st.write(
            f"Actifs: **{summary['n_actifs']}** | "
            f"Banc: **{summary['n_banc']}** | "
            f"IR: **{summary['n_ir']}** | "
            f"Mineur: **{summary['n_mineur']}** | "
            f"Total: **{summary['n_total']}**"
        )

    with t_roster:
        _render_rosters(summary["df"])

    with t_contracts:
        _render_contracts(summary["df"], season)

    with t_compare:
        _render_compare(owners, roster_all, plafonds, season, data_dir)

    # Debug minimal
    with st.expander("🧪 Debug (sources)", expanded=False):
        st.write("equipes_joueurs:", equipes.attrs.get("__path__", ""))
        st.write("players db:", players.attrs.get("__path__", "(absent)"))
        st.write("contracts:", contracts.attrs.get("__path__", "(absent)"))
        st.write("gm_logo:", _find_image("gm_logo.png", data_dir) or "(introuvable)")
        st.write("team_logo:", _find_team_logo(selected_owner, data_dir) or "(introuvable)")
