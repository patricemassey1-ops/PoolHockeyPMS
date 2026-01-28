# tabs/admin.py
# =====================================================
# ADMIN — Import équipes (v7b — FIX: pas d'expander imbriqué)
# =====================================================

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data"
DEFAULT_CAP_GC = 88_000_000
DEFAULT_CAP_CE = 10_000_000

EQUIPES_FILE_TMPL = "equipes_joueurs_{season}.csv"

# Colonnes minimales attendues après normalisation
EQUIPES_COLUMNS = [
    "ID",
    "Player",
    "Pos",
    "Team",
    "Equipe",
    "Slot",
]


# =====================================================
# HELPERS — paths / i/o
# =====================================================

def _equipes_path(season: str) -> str:
    season = (season or "").strip() or "2025-2026"
    return os.path.join(DATA_DIR, EQUIPES_FILE_TMPL.format(season=season))


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not path or (not os.path.exists(path)):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_csv_header_cols(path: str) -> List[str]:
    """Read only header to detect columns."""
    try:
        df0 = pd.read_csv(path, nrows=0)
        return [str(c).strip() for c in df0.columns.tolist()]
    except Exception:
        return []


# =====================================================
# HELPERS — filtering which CSV are TEAM CSVs
# =====================================================

_SYSTEM_NAME_PATTERNS = [
    r"^equipes_joueurs_.*\.csv$",     # output file
    r"^players_master\.csv$",        # fusion output
    r"^players_master_.*\.csv$",     # variants
    r"^players_db.*\.csv$",          # db variants
    r"^hockey\.players\.csv$",       # players db
    r"^hockey\.players_.*\.csv$",
    r"^puckpedia.*\.csv$",           # contracts
    r"^backup_.*\.csv$",
    r"^transactions_.*\.csv$",
    r"^trade_market_.*\.csv$",
    r"^settings\.csv$",
]
_SYSTEM_NAME_RE = re.compile("|".join(_SYSTEM_NAME_PATTERNS), flags=re.IGNORECASE)

_TEAM_PLAYER_COLS = {"player", "joueur", "name", "nom", "playername"}


def _looks_like_team_csv_by_name(filename: str) -> bool:
    fn = (filename or "").strip()
    if not fn.lower().endswith(".csv"):
        return False
    if _SYSTEM_NAME_RE.match(fn):
        return False
    return True


def _looks_like_team_csv_by_header(path: str) -> bool:
    cols = _read_csv_header_cols(path)
    if not cols:
        return False
    low = {c.strip().lower() for c in cols}
    return bool(low.intersection(_TEAM_PLAYER_COLS))


def list_team_csv_files_from_data_dir() -> Tuple[List[str], List[str]]:
    """Returns (team_files, ignored_files) as filenames (not full paths)."""
    if not os.path.isdir(DATA_DIR):
        return ([], [])
    all_csv = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    team_files: List[str] = []
    ignored: List[str] = []

    for f in sorted(all_csv, key=lambda x: x.lower()):
        if not _looks_like_team_csv_by_name(f):
            ignored.append(f)
            continue
        p = os.path.join(DATA_DIR, f)
        if _looks_like_team_csv_by_header(p):
            team_files.append(f)
        else:
            ignored.append(f)

    return (team_files, ignored)


# =====================================================
# HELPERS — normalization
# =====================================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "PlayerName": "Player",
        "Name": "Player",
        "Nom": "Player",
        "Joueur": "Player",
        "Position": "Pos",
        "FantraxID": "ID",
        "Fantrax Id": "ID",
        "playerId": "ID",
        "PlayerId": "ID",
    }

    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    if "Player" not in df.columns:
        for c in df.columns:
            if str(c).strip().lower() in _TEAM_PLAYER_COLS:
                df["Player"] = df[c]
                break

    if "Pos" not in df.columns:
        df["Pos"] = ""
    if "Team" not in df.columns:
        df["Team"] = ""
    if "ID" not in df.columns:
        df["ID"] = ""

    return df


def _auto_slot_from_pos(pos: str) -> str:
    if not isinstance(pos, str):
        return "Actif"
    p = pos.upper().strip()
    if p in ("IR", "INJ", "INJURED"):
        return "IR"
    return "Actif"


def normalize_team_import_df(df: pd.DataFrame, equipe: str) -> pd.DataFrame:
    df = _normalize_columns(df)

    if "Player" not in df.columns:
        raise ValueError("Colonne 'Player' manquante")

    df["Equipe"] = equipe
    df["Slot"] = df["Pos"].apply(_auto_slot_from_pos)

    keep = [c for c in EQUIPES_COLUMNS if c in df.columns]
    return df[keep].copy()


# =====================================================
# RENDER
# =====================================================

def render(ctx: Dict):
    st.subheader("🛠️ Gestion Admin")

    season = (ctx.get("season") or ctx.get("season_lbl") or "2025-2026").strip()

    # -------------------------------
    # Caps (paramètres uniquement)
    # -------------------------------
    with st.expander("☁️ Paramètres de cap (GC / CE)", expanded=False):
        st.session_state.setdefault("CAP_GC", DEFAULT_CAP_GC)
        st.session_state.setdefault("CAP_CE", DEFAULT_CAP_CE)

        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Cap GC", min_value=0, step=500_000, key="CAP_GC")
        with c2:
            st.number_input("Cap CE", min_value=0, step=500_000, key="CAP_CE")

        st.caption("ℹ️ L'affichage des barres de cap est dans **Home** (pas dans Admin).")

    # -------------------------------
    # Import local (fallback)
    # -------------------------------
    with st.expander("📥 Import local — CSV équipes", expanded=True):
        equipes_path = _equipes_path(season)
        st.caption(f"Destination locale : `{equipes_path}`")

        use_existing = st.checkbox(
            "Utiliser les fichiers déjà présents dans /data (sans upload)",
            value=True,
        )

        uploaded_files = []
        data_files: List[str] = []
        ignored_files: List[str] = []

        if use_existing:
            data_files, ignored_files = list_team_csv_files_from_data_dir()

            # ✅ FIX Streamlit: pas d'expander imbriqué
            if ignored_files:
                show_ignored = st.checkbox("Afficher les fichiers ignorés (non-équipes)", value=False)
                if show_ignored:
                    st.caption("Détectés comme système/DB/output ou sans colonne Player/Joueur/Name.")
                    st.write(ignored_files)

        else:
            uploaded_files = st.file_uploader(
                "Uploader des CSV (1 par équipe)",
                type=["csv"],
                accept_multiple_files=True,
            ) or []
            data_files = [f.name for f in uploaded_files]

        if not data_files:
            st.info("Aucun fichier CSV d'équipe détecté.")
            return

        st.markdown("### Attribution des fichiers → équipe")

        assignments: Dict[str, str] = {}
        for fname in data_files:
            equipe_guess = os.path.splitext(fname)[0]
            col1, col2, col3 = st.columns([1.2, 1.8, 3.0])
            with col1:
                st.markdown(f"**{fname}**")
            with col2:
                assignments[fname] = st.text_input(
                    f"Équipe pour {fname}",
                    value=equipe_guess,
                    key=f"assign_{fname}",
                    label_visibility="collapsed",
                )
            with col3:
                if use_existing:
                    cols = _read_csv_header_cols(os.path.join(DATA_DIR, fname))
                else:
                    fobj = next((f for f in uploaded_files if f.name == fname), None)
                    cols = list(pd.read_csv(fobj, nrows=0).columns) if fobj is not None else []
                head = ", ".join([str(c) for c in cols[:10]]) + ("..." if len(cols) > 10 else "")
                st.caption("Colonnes: " + head)

        # -------------------------------
        # Dry-run
        # -------------------------------
        if st.button("🧪 Dry-run (voir résumé)"):
            rows_total = 0
            files_ok = 0
            for fname, equipe in assignments.items():
                if use_existing:
                    path = os.path.join(DATA_DIR, fname)
                    df = _safe_read_csv(path)
                else:
                    fobj = next((f for f in uploaded_files if f.name == fname), None)
                    if fobj is None:
                        st.error(f"Fichier manquant en upload: {fname}")
                        return
                    df = pd.read_csv(fobj)

                try:
                    df_n = normalize_team_import_df(df, equipe)
                    rows_total += len(df_n)
                    files_ok += 1
                except Exception as e:
                    st.error(f"{fname} : {e}")
                    return

            st.success(f"✅ Dry-run OK — {files_ok} fichiers / {rows_total} lignes prêtes à importer")

        # -------------------------------
        # Import réel
        # -------------------------------
        if st.button("⬇️ Importer tous → Local + Reload"):
            all_rows: List[pd.DataFrame] = []
            for fname, equipe in assignments.items():
                if use_existing:
                    path = os.path.join(DATA_DIR, fname)
                    df = _safe_read_csv(path)
                else:
                    fobj = next((f for f in uploaded_files if f.name == fname), None)
                    if fobj is None:
                        st.error(f"Fichier manquant en upload: {fname}")
                        return
                    df = pd.read_csv(fobj)

                try:
                    df_n = normalize_team_import_df(df, equipe)
                except Exception as e:
                    st.error(f"{fname} : {e}")
                    return
                all_rows.append(df_n)

            if not all_rows:
                st.warning("Aucune donnée à importer.")
                return

            df_out = pd.concat(all_rows, ignore_index=True)

            os.makedirs(DATA_DIR, exist_ok=True)
            df_out.to_csv(equipes_path, index=False)

            st.success(f"✅ Import terminé — {len(df_out)} lignes écrites → `{equipes_path}`")
            st.experimental_rerun()
