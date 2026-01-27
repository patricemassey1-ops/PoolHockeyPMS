# =====================================================
# tabs/joueurs.py — Onglet Joueurs (PRO)
#   ✅ Recherche typeahead (prénom OU nom)
#   ✅ Filtres: Équipe NHL, Position (Forward/Defense/Goalie), Level (ELC/STD), Pays
#   ✅ Retire combos (F,D / F,G / etc.) + retire NAN des choix
#   ✅ Logo équipe NHL à côté de l'équipe
#   ✅ Badge "déjà dans une de nos équipes" basé sur:
#       Whalers, Red_Wings, Predateurs, Nordiques, Cracheurs, Canadiens (CSV dans /data/)
#   ✅ Photo headshot NHL via nhl_id (si présent)
#   ✅ Stats LIVE via NHL API (landing) en TABLE PRO (skater/goalie) — seulement si nhl_id présent
#   ✅ Utilise /data/hockey.players.csv
# =====================================================

import os
import re
import ast
import difflib
import datetime
import urllib.parse
import requests
import pandas as pd
import streamlit as st

DATA_DIR = "data"
PLAYERS_PATH = os.path.join(DATA_DIR, "hockey.players.csv")

CURRENT_CTX: dict | None = None



EQUIPES_JOUEURS_FILES = [
    "equipes_joueurs_2025-2026.csv",
    "equipes_joueurs_2025_2026.csv",
    "equipes_joueurs_2025-2026.csv".replace("-","_"),
]


PMS_TEAM_NAMES = ["Whalers", "Red_Wings", "Predateurs", "Nordiques", "Cracheurs", "Canadiens"]

def _discover_pms_team_files(data_dir: str) -> list[str]:
    """
    Retourne la liste des fichiers CSV d'équipes du pool présents dans data_dir.
    Supporte la casse (Whalers.csv vs whalers.csv), underscores/espaces, etc.
    """
    if not data_dir:
        return []
    try:
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    except Exception:
        return []

    # index des fichiers par nom normalisé (sans extension)
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s or "").lower())
    want = {norm(n): n for n in PMS_TEAM_NAMES}

    picked = []
    for f in files:
        stem = os.path.splitext(f)[0]
        k = norm(stem)
        if k in want:
            picked.append(f)

    # fallback: si rien trouvé, on garde les noms "classiques" (au cas où)
    if not picked:
        for n in PMS_TEAM_NAMES:
            picked.append(f"{n}.csv")
            picked.append(f"{n.lower()}.csv")
    # unique en préservant l'ordre
    out = []
    seen = set()
    for f in picked:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


# ----------------------------
# Helpers (position / name / formatting)
# ----------------------------

def nhl_results_to_labels(results):
    """Convert NHL search results list into UI labels + mapping.
    Works with various NHL API payload shapes.
    Returns (labels, id_map) where id_map[label] -> nhl_id (int).
    """
    labels = []
    id_map = {}
    if not results:
        return labels, id_map

    for r in results:
        if not isinstance(r, dict):
            continue
        # NHL id
        rid = r.get("id") or r.get("playerId") or r.get("nhlId") or r.get("nhl_id")
        try:
            rid_int = int(rid) if rid is not None and str(rid).strip() != "" else None
        except Exception:
            rid_int = None

        # name
        name = (
            r.get("fullName")
            or r.get("name")
            or r.get("displayName")
            or (f"{r.get('firstName','')} {r.get('lastName','')}".strip() or None)
        ) or "Unknown"

        # team
        team = ""
        t = r.get("currentTeam") or r.get("team") or {}
        if isinstance(t, dict):
            team = t.get("abbrev") or t.get("triCode") or t.get("name") or ""
        elif isinstance(t, str):
            team = t
        team = str(team).strip()

        # position
        pos = ""
        p = r.get("primaryPosition") or r.get("position") or {}
        if isinstance(p, dict):
            pos = p.get("abbrev") or p.get("code") or p.get("name") or ""
        elif isinstance(p, str):
            pos = p
        pos = str(pos).strip()

        # birthdate / age hint
        b = r.get("birthDate") or r.get("birthdate") or ""
        b = str(b).strip()

        parts = [name]
        if team:
            parts.append(team)
        if pos:
            parts.append(pos)
        if b and b != "nan":
            parts.append(b)

        label = " — ".join(parts)
        if rid_int is None:
            # still allow selection, but mark unknown id
            label = f"{label} (id?)"
        labels.append(label)
        id_map[label] = rid_int

    return labels, id_map


def _pos_bucket(pos_raw: str) -> str:
    """
    Mappe Position brute vers: Forward / Defense / Goalie
    - retire NAN
    - retire combos (F,D / F,G / etc.) -> bucket simple
    """
    s = str(pos_raw or "").strip().upper()
    if not s or s in {"NAN", "NONE", "NULL"}:
        return ""
    if "G" in s:
        return "Goalie"
    if "D" in s:
        return "Defense"
    if "F" in s:
        return "Forward"
    return ""


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
    s = s.strip()
    if not s or s.upper() == "NAN":
        return "—"
    return s


def clean_player_name(name: str) -> str:
    """Nettoie un nom joueur provenant de labels UI.
    Ex: 'Michael Hage | Michael Hage|' -> 'Michael Hage'
    """
    s = str(name or '').strip()
    if not s:
        return ''
    # si le label contient des pipes, on garde le premier segment non vide
    if '|' in s:
        parts = [p.strip() for p in s.split('|') if p.strip()]
        if parts:
            s = parts[0]
    # enlève doublon simple 'X X'
    toks = s.split()
    if len(toks) >= 4 and toks[:2] == toks[2:4]:
        s = ' '.join(toks[:2])
    return s



def _norm_player_key(name: str) -> str:
    """
    Normalisation robuste pour matcher les joueurs entre fichiers:
    - 'Zucker, Jason' -> 'jason zucker'
    - enlève ponctuation
    """
    s = player_last_first_to_first_last(str(name or ""))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_goalie(r: pd.Series) -> bool:
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
        if not s or s.upper() == "NAN":
            return "—"
        f = float(s)
        if f > 1.0:
            return f"{f:.1f}%"
        if 0.0 <= f <= 1.0:
            return f"{f:.3f}".lstrip("0")
        return s
    except Exception:
        s = str(v).strip()
        return s if s and s.upper() != "NAN" else "—"


def _fmt_num(v, digits=0):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        s = str(v).strip()
        if not s or s.upper() == "NAN":
            return "—"
        f = float(s)
        if digits == 0:
            return f"{int(round(f))}"
        return f"{f:.{digits}f}"
    except Exception:
        s = str(v).strip()
        return s if s and s.upper() != "NAN" else "—"


def _get_first_present(r: pd.Series, keys: list[str], default=None):
    for k in keys:
        if k in r.index:
            v = r.get(k)
            if pd.notna(v) and str(v).strip() != "" and str(v).strip().upper() != "NAN":
                return v
    return default



# ----------------------------
# NHL Search (associer NHL_ID)
# ----------------------------
def nhl_search_players(query: str, limit: int = 25) -> list[dict]:
    """Recherche des joueurs dans l'index NHL (pour retrouver nhl_id)."""
    q = str(query or "").strip()
    if not q:
        return []
    try:
        url = (
            "https://search.d3.nhle.com/api/v1/search/player"
            f"?culture=en-us&limit={int(limit)}&q={urllib.parse.quote(q)}"
        )
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _norm_name(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"[^a-z\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _seq_ratio(a: str, b: str) -> float:
    a = _norm_name(a)
    b = _norm_name(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def _auto_pick_nhl_id(player_name: str, team_abbrev: str, pos_bucket: str, age, results: list[dict]):
    """Retourne (best_id, best_label, ranked_labels, id_map, confident_bool)."""
    team_abbrev = str(team_abbrev or "").strip().upper()
    pos_bucket = str(pos_bucket or "").strip()

    birth_year_target = None
    try:
        if pd.notna(age):
            birth_year_target = int(datetime.date.today().year) - int(float(age))
    except Exception:
        birth_year_target = None

    scored = []
    for it in (results or []):
        pid = it.get("playerId") or it.get("player_id") or it.get("id")
        if not pid:
            continue
        first = it.get("firstName", "") or ""
        last = it.get("lastName", "") or ""
        full = (f"{first} {last}").strip() or str(it.get("name") or "").strip()

        team = (it.get("teamAbbrev") or it.get("team") or "").strip().upper()
        pos = (it.get("position") or it.get("positionCode") or "").strip().upper()
        bd = (it.get("birthDate") or it.get("birthdate") or "").strip()

        name_score = _seq_ratio(full, player_name)
        team_score = 0.08 if (team_abbrev and team and team_abbrev == team) else 0.0

        # Position: map NHL position letters to bucket
        pos_score = 0.0
        if pos_bucket:
            if pos == "G" and pos_bucket == "Goalie":
                pos_score = 0.06
            elif pos in {"D"} and pos_bucket == "Defense":
                pos_score = 0.06
            elif pos in {"C","L","R","LW","RW","F"} and pos_bucket == "Forward":
                pos_score = 0.06

        birth_score = 0.0
        if birth_year_target and bd and len(bd) >= 4:
            try:
                by = int(bd[:4])
                if abs(by - birth_year_target) <= 1:
                    birth_score = 0.06
            except Exception:
                pass

        total = name_score + team_score + pos_score + birth_score

        lbl = f"{full} — {team} {pos} — {bd} — ID {pid}".strip()
        scored.append((total, name_score, lbl, int(pid)))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    id_map = {}
    labels = []
    for total, name_score, lbl, pid in scored:
        labels.append(lbl)
        id_map[lbl] = pid

    if not scored:
        return None, None, [], {}, False

    best_total, best_name, best_lbl, best_id = scored[0]
    second_total = scored[1][0] if len(scored) > 1 else 0.0

    # Heuristique confiance: nom très proche OU score total fort + marge
    confident = (best_name >= 0.86) or (best_total >= 0.92 and (best_total - second_total) >= 0.06)

    return best_id, best_lbl, labels, id_map, confident



def save_players_db(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde hockey.players.csv en retirant les colonnes calculées."""
    if df is None or df.empty:
        return

    out = df.copy()

    # Colonnes internes à ne pas écrire
    drop_cols = {"PosBucket", "__pkey", "__display", "__alt", "__opt", "__rowid"}
    for c in list(drop_cols):
        if c in out.columns:
            out = out.drop(columns=[c])

    # nhl_id -> entier nullable
    if "nhl_id" in out.columns:
        out["nhl_id"] = pd.to_numeric(out["nhl_id"], errors="coerce").astype("Int64")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False, na_rep="")


def _ensure_rowid(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne __rowid (volatile) si absente, pour pouvoir cibler la ligne sélectionnée."""
    if df is None or df.empty:
        return df
    if "__rowid" not in df.columns:
        df = df.copy()
        df["__rowid"] = range(len(df))
    return df


def _set_nhl_id_by_rowid(df: pd.DataFrame, rowid: int, nhl_id_val) -> pd.DataFrame:
    """Set nhl_id sur la ligne __rowid == rowid (en ajoutant __rowid si nécessaire)."""
    df = _ensure_rowid(df)
    if df is None or df.empty:
        return df
    try:
        nhl_int = int(nhl_id_val) if nhl_id_val not in (None, "") else None
    except Exception:
        nhl_int = None
    if nhl_int is None:
        return df
    df.loc[df["__rowid"] == int(rowid), "nhl_id"] = nhl_int
    return df

# ----------------------------
# NHL Images
# ----------------------------
def headshot_url(player_id: int, size: int = 168) -> str:
    return f"https://cms.nhl.bamgrid.com/images/headshots/current/{size}x{size}/{int(player_id)}.jpg"


def team_logo_svg_url(team_abbrev: str, variant: str = "light") -> str:
    ab = (team_abbrev or "").strip().upper()
    if not ab or ab == "—":
        return ""
    v = "light" if variant not in ("light", "dark") else variant
    return f"https://assets.nhle.com/logos/nhl/svg/{ab}_{v}.svg"


def render_team_with_logo(team_abbrev: str):
    team = safe_str(team_abbrev)
    if team == "—":
        st.write("—")
        return

    logo = team_logo_svg_url(team, "light")
    if not logo:
        st.write(team)
        return

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px;">
          <img src="{logo}" alt="{team}" style="height:30px; width:auto; display:block;" />
          <div style="font-size:28px; font-weight:700; line-height:1;">{team}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------
# Load local players DB
# ----------------------------
@st.cache_data(show_spinner=False)
def load_players_db(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    for c in ["Player", "Team", "Position", "Level", "Age", "Flag", "Country", "Cap Hit", "Jersey#", "H(f)", "W(lbs)", "Status"]:
        if c not in df.columns:
            df[c] = ""

    if "nhl_id" in df.columns:
        df["nhl_id"] = pd.to_numeric(df["nhl_id"], errors="coerce").astype("Int64")
    else:
        df["nhl_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df["Player"] = df["Player"].astype(str).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip().str.upper()
    df["Position"] = df["Position"].astype(str).str.strip().str.upper()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Level"] = df["Level"].astype(str).str.strip().str.upper().replace({"0": "", "NAN": ""})
    df["PosBucket"] = df["Position"].apply(_pos_bucket)

    for c in ["Team", "Country", "Flag", "Cap Hit", "Status"]:
        df[c] = df[c].replace({"nan": "", "NaN": "", "NAN": ""})

    df["__pkey"] = df["Player"].apply(clean_player_name).apply(_norm_player_key)
    # volatile row id to target updates (nhl_id association)
    if "__rowid" not in df.columns:
        df["__rowid"] = range(len(df))
    return df


# ----------------------------
# Owned index from 6 PMS team CSV files
# ----------------------------
def _team_name_from_filename(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    return base.replace("_", " ").strip()


def _detect_player_column(df: pd.DataFrame) -> str | None:
    """
    Détecte la colonne joueur dans un roster CSV.
    On essaie des noms fréquents; sinon, si une colonne contient des valeurs "Last, First" on la choisit.
    """
    cols = [c.strip() for c in df.columns]
    for c in ["Joueur", "Player", "Nom", "Name"]:
        if c in cols:
            return c

    # Heuristique: première colonne qui ressemble à des noms
    for c in cols[:5]:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        # si plusieurs valeurs contiennent une virgule (Last, First)
        sample = s.head(30).tolist()
        comma_hits = sum(1 for x in sample if "," in str(x))
        if comma_hits >= 5:
            return c

    # fallback: première colonne
    return cols[0] if cols else None


def _detect_scope_column(df: pd.DataFrame) -> str | None:
    for c in ["Scope", "Club", "GC/CE", "GC_CE", "Type", "Roster", "Slot", "Statut", "Status"]:
        if c in df.columns:
            return c
    return None



def _looks_like_nhl_abbrev_series(s: pd.Series) -> float:
    """Retourne la proportion de valeurs qui ressemblent à des abréviations NHL (2-3 lettres majuscules)."""
    if s is None:
        return 0.0
    vals = [str(x).strip() for x in s.dropna().astype(str).tolist()]
    vals = [v for v in vals if v and v.upper() != "NAN"]
    if not vals:
        return 0.0
    hits = 0
    for v in vals[:400]:
        vv = v.strip()
        if re.fullmatch(r"[A-Z]{2,3}", vv):
            hits += 1
    return hits / max(1, min(len(vals), 400))


def _pick_pool_team_column(dfu: pd.DataFrame) -> str | None:
    """Choisit la colonne qui correspond à l'équipe du pool (Whalers, etc.) dans le fichier unifié.

    On évite les colonnes ambiguës (ex: 'Team' = souvent NHL).
    Heuristiques:
      - Priorité aux colonnes explicitement pool: Owner/GM/EquipePMS/TeamPMS/Equipe Pool...
      - Si une colonne ressemble majoritairement à des abréviations NHL (MTL, BOS, etc.), on la rejette.
      - Si une colonne contient des valeurs qui matchent nos équipes PMS connues, on la favorise.
    """
    if dfu is None or dfu.empty:
        return None

    # Liste d'équipes pool connues (on compare en normalisant)
    known_pool = set([_team_name_from_filename(x + ".csv").lower() for x in [
        "Whalers","Red_Wings","Predateurs","Nordiques","Cracheurs","Canadiens"
    ]])
    known_pool |= set([k.replace("_"," ").lower() for k in known_pool])

    candidates = []
    preferred = [
        "EquipePMS","ÉquipePMS","TeamPMS","Team PMS","Equipe Pool","Équipe Pool",
        "EquipePool","ÉquipePool","TeamPool","Team Pool",
        "Owner","GM","PMS Team","PMS_Team",
    ]
    # 'Equipe' / 'Team' sont ambigus, on les garde mais en dernier
    ambiguous = ["Equipe","Équipe","Team"]

    for c in preferred + ambiguous:
        if c in dfu.columns:
            candidates.append(c)

    if not candidates:
        return None

    best = None
    best_score = -1e9

    for c in candidates:
        s = dfu[c]
        # score de base: préféré > ambigu
        base = 2.0 if c in preferred else 0.5
        nhl_like = _looks_like_nhl_abbrev_series(s)
        # pénalité si ça ressemble à des abréviations NHL
        penalty = 3.0 * nhl_like

        # bonus si on voit des noms d'équipes pool connues
        vals = [str(x).strip() for x in s.dropna().astype(str).tolist()][:400]
        vals_norm = [v.replace("_"," ").lower() for v in vals if v and v.upper() != "NAN"]
        pool_hits = sum(1 for v in vals_norm if v in known_pool)
        bonus = 1.5 * (pool_hits / max(1, len(vals_norm))) if vals_norm else 0.0

        score = base + bonus - penalty
        if score > best_score:
            best_score = score
            best = c

    # si meilleur candidat est trop "NHL-like", on refuse
    if best is None:
        return None
    if _looks_like_nhl_abbrev_series(dfu[best]) >= 0.6 and best in ambiguous:
        return None

    return best


@st.cache_data(show_spinner=False)
def load_owned_index(data_dir: str) -> dict:
    """Index des joueurs appartenant déjà au pool.

    Retour: pkey -> {"team_pms": "...", "scope": "...", "source": "..."}
    Sources supportées:
      1) fichier unifié equipes_joueurs_*.csv (si présent)
      2) fichiers par équipe (Whalers.csv, etc.)
    """
    idx: dict[str, dict] = {}

    # --- (1) fichier unifié equipes_joueurs_*.csv
    for uf in EQUIPES_JOUEURS_FILES:
        fp = os.path.join(data_dir, uf)
        if not os.path.exists(fp):
            continue
        try:
            dfu = pd.read_csv(fp, low_memory=False)
            dfu.columns = [c.strip() for c in dfu.columns]

            pcol = _detect_player_column(dfu)
            # colonne équipe du pool (heuristique robuste)
            tcol = _pick_pool_team_column(dfu)
            scol = _detect_scope_column(dfu)

            if not pcol or not tcol:
                continue

            for _, r in dfu.iterrows():
                pkey = _norm_player_key(r.get(pcol, ""))
                if not pkey or pkey in idx:
                    continue
                team_pms = str(r.get(tcol, "")).strip()
                scope_val = str(r.get(scol, "")).strip() if scol else ""
                if not team_pms or team_pms.upper() == "NAN":
                    continue
                idx[pkey] = {"team_pms": team_pms, "scope": scope_val, "source": uf}
        except Exception:
            pass

        if idx:
            return idx

    # --- (2) fallback fichiers par équipe
    for fname in _discover_pms_team_files(data_dir):
        fp = os.path.join(data_dir, fname)
        if not os.path.exists(fp):
            continue

        team_pms = _team_name_from_filename(fname)

        try:
            df = pd.read_csv(fp, low_memory=False)
            df.columns = [c.strip() for c in df.columns]

            pcol = _detect_player_column(df)
            if not pcol:
                continue

            scol = _detect_scope_column(df)

            for _, r in df.iterrows():
                pkey = _norm_player_key(r.get(pcol, ""))
                if not pkey or pkey in idx:
                    continue
                scope_val = str(r.get(scol, "")).strip() if scol else ""
                idx[pkey] = {"team_pms": team_pms, "scope": scope_val, "source": fname}
        except Exception:
            continue

    return idx


def render_owned_badge(pkey: str, owned_idx: dict):
    info = owned_idx.get(pkey)
    if not info:
        st.markdown("🟢 **Disponible**")
        return

    team_pms = info.get("team_pms") or "Équipe"
    scope = str(info.get("scope") or "").strip()
    src = info.get("source") or ""
    extra = f" — {scope}" if scope and scope.upper() != "NAN" else ""
    src_txt = f" (source: {src})" if src else ""
    st.markdown(f"🔴 **Non disponible** — dans **{team_pms}{extra}**{src_txt}")



# ----------------------------
# 🧪 Audit (sanity check)
#   Détecte les joueurs du pool qui pourraient encore apparaître "Disponible"
#   à cause d'un mismatch de normalisation ou d'une source de vérité divergente.
# ----------------------------
def _detect_pool_team_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes possibles indiquant l'équipe/owner du pool dans hockey.players.csv."""
    candidates = [
        "EquipePool", "ÉquipePool", "Equipe Pool", "Équipe Pool",
        "TeamPool", "Team Pool", "Owner", "GM", "EquipePMS", "ÉquipePMS",
        "PMS Team", "TeamPMS", "Team PMS", "PMS_Team",
    ]
    return [c for c in candidates if c in df.columns]


@st.cache_data(show_spinner=False)
def audit_pool_vs_availability(players_df: pd.DataFrame, data_dir: str) -> dict:
    """
    Retourne un dict avec:
      - dupes: joueurs présents dans >1 équipe (via fichiers équipes)
      - missing_in_players_db: joueurs présents dans fichiers équipes mais introuvables dans hockey.players.csv (pkey mismatch)
      - players_db_pool_but_available: joueurs marqués avec une équipe/owner dans hockey.players.csv mais non trouvés dans fichiers équipes (ils risquent d'apparaître 'Disponible')
    """
    # --- build raw pool map from team files with duplicates
    pool_map: dict[str, dict] = {}  # pkey -> {"teams": set, "samples": set}
    for fname in _discover_pms_team_files(data_dir):
        fp = os.path.join(data_dir, fname)
        if not os.path.exists(fp):
            continue
        team_pms = _team_name_from_filename(fname)
        try:
            tdf = pd.read_csv(fp, low_memory=False)
            tdf.columns = [c.strip() for c in tdf.columns]
            pcol = _detect_player_column(tdf)
            if not pcol:
                continue
            for _, rr in tdf.iterrows():
                raw_name = str(rr.get(pcol, '') or '').strip()
                pkey = _norm_player_key(raw_name)
                if not pkey:
                    continue
                ent = pool_map.setdefault(pkey, {"teams": set(), "samples": set(), "files": set()})
                ent["teams"].add(team_pms)
                ent["files"].add(fname)
                if raw_name:
                    ent["samples"].add(raw_name)
        except Exception:
            continue

    dupes = []
    for pkey, ent in pool_map.items():
        teams = sorted(ent["teams"])
        if len(teams) > 1:
            dupes.append({
                "PlayerKey": pkey,
                "Teams": ", ".join(teams),
                "Examples": ", ".join(sorted(list(ent["samples"]))[:2]),
            })

    # --- players db keys
    pdf = players_df.copy()
    if "__pkey" not in pdf.columns:
        # best effort: try Player column
        pcol = "Player" if "Player" in pdf.columns else None
        if pcol:
            pdf["__pkey"] = pdf[pcol].astype(str).apply(_norm_player_key)
        else:
            pdf["__pkey"] = ""

    pkeys_db = set([k for k in pdf["__pkey"].astype(str).tolist() if k])

    missing_in_players_db = []
    for pkey, ent in pool_map.items():
        if pkey not in pkeys_db:
            missing_in_players_db.append({
                "PlayerKey": pkey,
                "Team": ", ".join(sorted(ent["teams"])),
                "ExampleFromTeamCSV": ", ".join(sorted(list(ent["samples"]))[:2]),
            })

    # --- players db "pool team" columns vs team files
    pool_cols = _detect_pool_team_columns(pdf)
    players_db_pool_but_available = []
    if pool_cols:
        # any non-empty value in any pool col
        def _pool_team_for_row(row):
            for c in pool_cols:
                v = str(row.get(c, '') or '').strip()
                if v and v.upper() != "NAN":
                    return v
            return ""

        pdf["__pool_team_hint"] = pdf.apply(_pool_team_for_row, axis=1)
        hinted = pdf[pdf["__pool_team_hint"].astype(str).str.strip() != ""].copy()
        # availability in UI is based on team files index (owned_idx) -> so if not in pool_map, it's "Disponible"
        for _, rr in hinted.iterrows():
            pkey = str(rr.get("__pkey") or "")
            if not pkey:
                continue
            if pkey not in pool_map:
                players_db_pool_but_available.append({
                    "Player": safe_str(rr.get("Player")),
                    "PoolTeamInPlayersDB": rr.get("__pool_team_hint", ""),
                    "Note": "Présent dans hockey.players.csv mais absent des fichiers équipes (/data/*) — risque d'apparaitre 'Disponible'",
                })

    return {
        "dupes": dupes,
        "missing_in_players_db": missing_in_players_db,
        "players_db_pool_but_available": players_db_pool_but_available,
        "pool_cols_detected": pool_cols,
    }

# ----------------------------
# NHL API (landing)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def nhl_player_landing(player_id: int) -> dict:
    url = f"https://api-web.nhle.com/v1/player/{int(player_id)}/landing"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return r.json()


# ----------------------------
# API -> PRO TABLE extraction (robust)
# ----------------------------
def _deep_find_lists(obj, key_hint=None, max_lists=20):
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
    for k in ["seasonTotals", "seasonTotal", "seasons", "statsBySeason", "playerStatsBySeason"]:
        v = landing.get(k)
        if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
            return v

    found = _deep_find_lists(landing, key_hint="season")
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

    return found[0][1] if found else []


def _to_number(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s or s.upper() == "NAN":
            return None
        if s.startswith("."):
            s = "0" + s
        return float(s)
    except Exception:
        return None


def _map_skater_row(it: dict) -> dict:
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
        "Lg": str(league),
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
        "Lg": str(league),
    }


def _format_api_table(df: pd.DataFrame, goalie: bool) -> pd.DataFrame:
    df2 = df.copy()

    def _season_fmt(x):
        s = str(x or "").strip()
        # 20252026 -> 2025-2026
        if re.fullmatch(r"\d{8}", s):
            return f"{s[:4]}-{s[4:]}"
        # 2025-2026 / 2025_2026 -> normalize dash
        m = re.fullmatch(r"(\d{4})[^0-9]?(\d{4})", s)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        return s

    def _team_fmt(x):
        # dict -> take 'default' or first string-ish
        if isinstance(x, dict):
            if "default" in x and x["default"]:
                return str(x["default"])
            for v in x.values():
                if v:
                    return str(v)
            return ""
        s = str(x or "").strip()
        # string that looks like a dict: "{'default': 'Univ...'}"
        if s.startswith("{") and "default" in s:
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    return str(d.get("default") or next(iter(d.values()), "") or "")
            except Exception:
                pass
        return s

    if "Season" in df2.columns:
        df2["Season"] = df2["Season"].apply(_season_fmt)
    if "Team" in df2.columns:
        df2["Team"] = df2["Team"].apply(_team_fmt)

    def _type_norm(x):
        s = str(x or "").strip().upper()
        if s in ["R", "RS", "REG", "REGULAR", "2"]:
            return "RS"
        if s in ["P", "PO", "PLAYOFF", "PLAYOFFS", "3"]:
            return "PO"
        return s

    if "Type" in df2.columns:
        df2["Type"] = df2["Type"].apply(_type_norm)

    if goalie:
        keep = ["Season", "Team", "Type", "GP", "W", "L", "OTL", "SV%", "GAA", "SO", "Saves", "SA", "GA"]
        keep = [c for c in keep if c in df2.columns]
        df2 = df2[keep]
        if "SV%" in df2.columns:
            df2["SV%"] = df2["SV%"].apply(_fmt_pct)
        if "GAA" in df2.columns:
            df2["GAA"] = df2["GAA"].apply(lambda x: _fmt_num(x, 2))
        for c in ["GP", "W", "L", "OTL", "SO", "Saves", "SA", "GA"]:
            if c in df2.columns:
                df2[c] = df2[c].apply(lambda x: _fmt_num(x, 0))
        return df2

    keep = ["Season", "Team", "Type", "GP", "G", "A", "PTS", "+/-", "PIM", "PPP", "PPG", "PPA", "SHP", "SOG", "Sh%"]
    keep = [c for c in keep if c in df2.columns]
    df2 = df2[keep]
    for c in ["GP", "G", "A", "PTS", "+/-", "PIM", "PPP", "PPG", "PPA", "SHP", "SOG"]:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: _fmt_num(x, 0))
    if "Sh%" in df2.columns:
        df2["Sh%"] = df2["Sh%"].apply(lambda x: _fmt_num(x, 1))
    return df2


def render_api_pro_tables(landing: dict, goalie: bool):
    season_list = _pick_season_list(landing)
    if not season_list:
        st.info("Aucune table de saisons trouvée dans la réponse API (landing).")
        return

    rows = []
    for it in season_list:
        try:
            rows.append(_map_goalie_row(it) if goalie else _map_skater_row(it))
        except Exception:
            continue

    if not rows:
        st.info("Impossible de mapper les stats API en table (structure différente).")
        return

    df = pd.DataFrame(rows).dropna(how="all")
    if df.empty:
        st.info("Table API vide après nettoyage.")
        return

    df["_season_sort"] = pd.to_numeric(df["Season"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    df = df.sort_values(by=["_season_sort", "Type", "Team"], ascending=[False, True, True]).drop(columns=["_season_sort"])

    df_fmt = _format_api_table(df, goalie=goalie)
    rs = df_fmt[df_fmt["Type"] == "RS"] if "Type" in df_fmt.columns else pd.DataFrame()
    po = df_fmt[df_fmt["Type"] == "PO"] if "Type" in df_fmt.columns else pd.DataFrame()

    if not rs.empty:
        st.markdown("#### Regular Season (API)")
        st.dataframe(rs, use_container_width=True)
    if not po.empty:
        st.markdown("#### Playoffs (API)")
        st.dataframe(po, use_container_width=True)

    if rs.empty and po.empty:
        st.markdown("#### Stats (API)")
        st.dataframe(df_fmt, use_container_width=True)

    featured = landing.get("featuredStats")
    if featured:
        with st.expander("Featured stats (API) — détail", expanded=False):
            st.json(featured, expanded=False)


# ----------------------------
# Local PRO stats (CSV) panels
# ----------------------------
def render_local_pro_panel(row: pd.DataFrame):
    r = row.iloc[0]
    rowid = int(r.get("__rowid", 0))
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
def render_tab_joueurs(ctx: dict | None = None):
    st.header("🏒 Joueurs")

    # Outils admin (associer NHL_ID, audit) — idéalement dans Gestion Admin
    # On les affiche seulement si admin.
    _ctx = ctx or CURRENT_CTX or {}
    show_admin_tools = bool(_ctx.get("is_admin") or st.session_state.get("is_admin"))

    df = load_players_db(PLAYERS_PATH)
    if df.empty:
        st.error("❌ data/hockey.players.csv introuvable ou vide.")
        return

    owned_idx = load_owned_index(DATA_DIR)

    # ---- Filtres (4)
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    with c1:
        teams = sorted([t for t in df["Team"].dropna().astype(str).unique() if t.strip() and t.strip().upper() != "NAN"])
        team_pick = st.selectbox("Équipe NHL", ["Toutes"] + teams, index=0)

    with c2:
        pos_pick = st.selectbox("Position", ["Toutes", "Forward", "Defense", "Goalie"], index=0)

    with c3:
        level_pick = st.selectbox("Level", ["Tous", "ELC", "STD"], index=0)

    with c4:
        countries = sorted([c for c in df["Country"].dropna().astype(str).unique() if c.strip() and c.strip().upper() != "NAN"])
        country_pick = st.selectbox("Pays", ["Tous"] + countries, index=0)

    filt = df.copy()
    if team_pick != "Toutes":
        filt = filt[filt["Team"].astype(str) == team_pick]
    if pos_pick != "Toutes":
        filt = filt[filt["PosBucket"] == pos_pick]
    if level_pick != "Tous":
        filt = filt[filt["Level"].astype(str).str.upper() == level_pick]
    if country_pick != "Tous":
        filt = filt[filt["Country"].astype(str) == country_pick]

    if filt.empty:
        st.warning("Aucun joueur ne correspond aux filtres.")
        return

    # ---- Recherche typeahead
    filt = filt.copy()
    filt["__display"] = filt["Player"].astype(str).str.strip()
    filt["__alt"] = filt["Player"].astype(str).apply(player_last_first_to_first_last)
    filt["__opt"] = filt["__display"] + " | " + filt["__alt"]
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
    rowid = int(r.get("__rowid", 0))
    goalie = _is_goalie(r)

    # ---- Titre joueur
    raw_name = r.get("Player") or r.get("Joueur") or r.get("Nom") or r.get("Name") or ""
    player_name_clean = clean_player_name(raw_name) or str(raw_name or "").strip()
    player_name = safe_str(player_name_clean)
    player_name_alt = safe_str(player_last_first_to_first_last(player_name_clean))
    st.subheader(player_name_alt if player_name_alt != "—" else player_name)

    # ---- Owned badge
    render_owned_badge(str(r.get("__pkey", "")), owned_idx)

    # ---- Header metrics (avec logo NHL)
    pos_simple = safe_str(r.get("PosBucket")) if safe_str(r.get("PosBucket")) != "—" else safe_str(r.get("Position"))

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.caption("Équipe (NHL)")
        render_team_with_logo(r.get("Team"))
    with m2:
        st.metric("Position", pos_simple)
    with m3:
        st.metric("Level", safe_str(r.get("Level")))
    with m4:
        st.metric("Âge", safe_str(r.get("Age")))
    with m5:
        st.metric("Cap Hit", safe_str(r.get("Cap Hit")))

    # ---- Photo + drapeau + infos
    left, right = st.columns([1, 2])
    nhl_id = r.get("nhl_id")
    flag_url = str(r.get("Flag") or "").strip()
    missing_nhl = False


    with left:
        if pd.notna(nhl_id):
            pid = int(nhl_id)
            # Plusieurs sources possibles (certains prospects n'ont pas de mug)
            urls = [
                headshot_url(pid, size=168),
                f"https://nhl.bamcontent.com/images/headshots/current/168x168/{pid}.jpg",
                f"https://nhl.bamcontent.com/images/headshots/current/260x260/{pid}.jpg",
            ]
            shown = False
            for u in urls:
                try:
                    resp = requests.get(u, timeout=4)
                    if resp.status_code == 200 and resp.content:
                        st.image(u, use_container_width=True)
                        shown = True
                        break
                except Exception:
                    continue
            if not shown:
                st.warning("Photo NHL indisponible pour ce joueur (aucun headshot trouvé).")
            st.caption(f"NHL ID: {pid}")
        else:
            st.info("Photo NHL et stats live indisponibles (nhl_id manquant).")
            missing_nhl = True

    with right:
        info_cols = st.columns(4)
        info_cols[0].metric("Jersey#", safe_str(r.get("Jersey#")))
        info_cols[1].metric("Taille", safe_str(r.get("H(f)")))
        info_cols[2].metric("Poids", safe_str(r.get("W(lbs)")))
        info_cols[3].metric("Statut", safe_str(r.get("Status")))

        st.markdown("### 📊 Stats (PRO — CSV)")
        render_local_pro_panel(row)

    # ---- Associer NHL_ID (centré et plus étroit) si manquant
    if show_admin_tools and missing_nhl:
        cL, cM, cR = st.columns([1, 1.15, 1])
        with cM:
            with st.expander("🔗 Associer NHL_ID (activer photo + stats live)", expanded=False):
                default_q = player_last_first_to_first_last(player_name)
                q = st.text_input(
                    "Recherche NHL (nom / prénom)",
                    value=default_q if default_q != "—" else str(player_name or "").strip(),
                    key=f"nhl_q__{rowid}",
                )

                # --- Auto association (best effort)
                if st.button("🤖 Associer automatiquement", key=f"nhl_auto__{rowid}"):
                    auto_results = nhl_search_players(default_q, limit=25)
                    if not auto_results:
                        st.warning("Aucun résultat NHL trouvé (endpoint de recherche vide). Essaie 🔎 Chercher dans la NHL avec un nom complet.")
                    best_id, best_lbl, labels, id_map, confident = _auto_pick_nhl_id(
                        player_name=default_q,
                        team_abbrev=r.get("Team"),
                        pos_bucket=_pos_bucket(r.get("Position")),
                        age=r.get("Age"),
                        results=auto_results,
                    )
                    if best_id and confident:
                        df = _set_nhl_id_by_rowid(df, rowid, best_id)
                        save_players_db(df, PLAYERS_PATH)
                        load_players_db.clear()
                        st.success(f"✅ NHL_ID auto-associé: {best_id} ({best_lbl})")
                        st.rerun()
                    elif labels:
                        st.session_state[f"nhl_results__{rowid}"] = auto_results
                        st.warning("Match automatique incertain — sélectionne le bon joueur ci-dessous 👇")
                        st.session_state[f"nhl_pick__{rowid}"] = best_lbl
                    else:
                        st.warning("Aucun match trouvé automatiquement. Essaie la recherche manuelle.")

                if st.button("🔎 Chercher dans la NHL", key=f"nhl_search__{rowid}"):
                    st.session_state[f"nhl_results__{rowid}"] = nhl_search_players(q, limit=25)

                results = st.session_state.get(f"nhl_results__{rowid}", [])
                if results:
                    labels, id_map = nhl_results_to_labels(results)
                    pick = st.radio(
                        "Résultats",
                        options=labels,
                        index=labels.index(st.session_state.get(f"nhl_pick__{rowid}", labels[0])) if labels else 0,
                        key=f"nhl_pick__{rowid}",
                    )
                    if st.button("✅ Enregistrer NHL_ID", key=f"nhl_save__{rowid}"):
                        chosen_id = id_map.get(pick)
                        if chosen_id:
                            df = _set_nhl_id_by_rowid(df, rowid, chosen_id)
                            save_players_db(df, PLAYERS_PATH)
                            load_players_db.clear()
                            st.success(f"✅ NHL_ID enregistré: {chosen_id} ({pick})")
                            st.rerun()
                else:
                    st.caption("Astuce: tape le nom complet (ex: 'Michael Hage') puis lance la recherche.")



    # ---- 🧪 Audit automatique (sanity check) — même largeur que l’associateur NHL_ID
    if show_admin_tools:
        cAL, cAM, cAR = st.columns([1, 1.15, 1])
        with cAM:
            with st.expander("🧪 Audit automatique — sanity check (pool vs Disponible)", expanded=False):
                audit = audit_pool_vs_availability(df, DATA_DIR)
    
                dupes = audit.get("dupes", [])
                missing = audit.get("missing_in_players_db", [])
                risky = audit.get("players_db_pool_but_available", [])
    
                cols = st.columns(3)
                cols[0].metric("Doublons (2 équipes+)", len(dupes))
                cols[1].metric("Pool introuvable dans players DB", len(missing))
                cols[2].metric("Risque: marqué pool mais 'Disponible'", len(risky))
    
                if audit.get("pool_cols_detected"):
                    st.caption("Colonnes détectées dans hockey.players.csv pour l’équipe/owner pool: " + ", ".join(audit["pool_cols_detected"]))
    
                if dupes:
                    st.markdown("#### ⚠️ Joueurs présents dans plusieurs équipes (à corriger)")
                    st.dataframe(pd.DataFrame(dupes), use_container_width=True, hide_index=True)
    
                if missing:
                    st.markdown("#### ⚠️ Joueurs dans les fichiers équipes mais introuvables dans hockey.players.csv (mismatch nom / normalisation)")
                    st.dataframe(pd.DataFrame(missing), use_container_width=True, hide_index=True)
    
                if risky:
                    st.markdown("#### ⚠️ Joueurs marqués dans une équipe (hockey.players.csv) mais non détectés dans les fichiers équipes — ils peuvent apparaître 'Disponible'")
                    st.dataframe(pd.DataFrame(risky), use_container_width=True, hide_index=True)
    
                if (not dupes) and (not missing) and (not risky):
                    st.success("✅ Aucun problème détecté: tout ce qui est dans le pool est reconnu comme non-disponible.")
    
        # ---- API PRO TABLE (seulement si nhl_id présent)
        if pd.notna(nhl_id):
            st.markdown("### ⚡ Stats NHL LIVE (API — table pro)")
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
    
    
    # =====================================================
    # Entry point attendu: joueurs.render(ctx)
    # =====================================================
def render(ctx: dict):
    global CURRENT_CTX
    CURRENT_CTX = ctx or {}
    render_tab_joueurs(ctx)