# tabs/admin.py
# ============================================================
# PMS Pool Hockey — Admin Tab (Streamlit)
# Compatible avec: admin.render(ctx) depuis app.py
# ============================================================
# ✅ Import équipes depuis Drive (OAuth) + Import local fallback
# ✅ Preview + validation colonnes attendues
# ✅ ➕ Ajouter joueurs (anti-triche cross-team)
# ✅ 🗑️ Retirer joueurs (UI + confirmation)
# ✅ 🔁 Déplacer GC ↔ CE (auto-slot / keep / force)
# ✅ 🧪 Barres visuelles cap GC/CE + dépassements
# ✅ 📋 Historique admin complet (ADD/REMOVE/MOVE/IMPORT)
# ✅ Auto-mapping Level via hockey.players.csv (+ heuristique salaire)
# ✅ Alertes IR mismatch + Salary/Level suspect + preview colorée
# ============================================================

from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ============================================================
# Defaults (Caps)
# ============================================================
# Valeurs par défaut si settings/csv non disponibles.
# NHL cap ~ 88M (2025-2026 approximatif). Ajuste au besoin.
DEFAULT_CAP_GC = 88_000_000
DEFAULT_CAP_CE = 10_000_000


ADMIN_VERSION = "ADMIN_PANEL_V5_NO_STATUS_2026-01-27"

# ============================================================
# OAuth UI (stub)
# - v6 inline-only: aucune dépendance Google/Drive chargée.
# - On conserve l'expander/section sans casser l'app.
# ============================================================
def _oauth_ui(*args, **kwargs):
    """UI OAuth (désactivée en v6). Retourne un dict compatible si utilisé."""
    try:
        import streamlit as st
        st.warning("Drive OAuth désactivé dans cette version (v6 inline-only). Utilise l'import local (fallback).")
    except Exception:
        pass
    return {"drive_ok": False, "folder_id": "", "reason": "oauth_disabled_v6"}

import pathlib
import time
import shutil


# ============================================================
# Paths — fichiers locaux (source de vérité / rosters)
# ============================================================
def equipes_path(data_dir: str, season_lbl: str) -> str:
    """Chemin du fichier fusionné des équipes (alignements) pour une saison."""
    season_lbl = str(season_lbl or "").strip() or "season"
    return os.path.join(str(data_dir), f"equipes_joueurs_{season_lbl}.csv")

# ============================================================
# Helpers — robust CSV / name normalization (required by Fusion)
# ============================================================
import unicodedata

def _strip_accents(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    except Exception:
        return str(s)

def _clean_str(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ").replace("\xa0", " ")
    return s.strip()

def _norm_player_key(name: str) -> str:
    """Stable key for matching across files (handles 'Last, First' and accents)."""
    n = _clean_str(name)
    if not n or n.lower() in {"nan","none","n/a","na"}:
        return ""
    # Convert 'Last, First' -> 'First Last'
    if "," in n:
        parts = [p.strip() for p in n.split(",") if p.strip()]
        if len(parts) >= 2:
            n = " ".join(parts[1:]) + " " + parts[0]
    n = _strip_accents(n).lower()
    # Remove non letters/numbers and collapse spaces
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def _first_existing_col(df, candidates):
    """Return first column name present in df from candidates (case-sensitive)."""
    if df is None or getattr(df, "empty", False):
        return None
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # fallback: case-insensitive
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        lc = str(c).lower()
        if lc in lower_map:
            return lower_map[lc]
    return None

def _extract_roster_from_fantrax(df, team_name: str = ""):
    """Extract pool roster rows from a team CSV (Fantrax export variations).

    Returns a DataFrame with at least:
      - player_raw, _pkey, team_pool, pos_pool, eligible_pool, salary
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["player_raw","_pkey","team_pool","pos_pool","eligible_pool","salary"])

    name_col = _first_existing_col(df, ["Player","player","Joueur","joueur","Name","name","Nom","nom"])
    if not name_col:
        # Some exports store 'First Name'/'Last Name'
        fn = _first_existing_col(df, ["First","First Name","Prenom","Prénom"])
        ln = _first_existing_col(df, ["Last","Last Name","Nom de famille"])
        if fn and ln:
            df = df.copy()
            df["_tmp_name"] = df[ln].astype(str).str.strip() + ", " + df[fn].astype(str).str.strip()
            name_col = "_tmp_name"
        else:
            return pd.DataFrame(columns=["player_raw","_pkey","team_pool","pos_pool","eligible_pool","salary"])

    pos_col = _first_existing_col(df, ["Pos","Position","pos","position"])
    elig_col = _first_existing_col(df, ["Eligible","Elig","eligible","elig"])
    salary_col = _first_existing_col(df, ["Salary","Cap Hit","CapHit","Cap Hit ($)","CapHit($)","CapHit ($)","Salaire","Cap","AAV","AAV($)"])

    out = pd.DataFrame()
    out["player_raw"] = df[name_col].astype(str)
    out["player_raw"] = out["player_raw"].apply(_clean_str)
    out = out[out["player_raw"].str.len() > 0].copy()

    out["_pkey"] = out["player_raw"].apply(_norm_player_key)
    out["team_pool"] = team_name or ""
    out["pos_pool"] = df[pos_col].astype(str).apply(_clean_str) if pos_col else ""
    out["eligible_pool"] = df[elig_col].astype(str).apply(_clean_str) if elig_col else ""
    if salary_col:
        # keep only digits/., convert to float
        s = df[salary_col].astype(str).str.replace(",", "").str.replace(" ", "")
        s = s.str.replace("$","", regex=False)
        out["salary"] = pd.to_numeric(s, errors="coerce")
    else:
        out["salary"] = pd.NA

    # Heuristic: drop header-like rows
    out = out[~out["player_raw"].str.lower().isin({"player","joueur","name","nom"})].copy()
    return out


# ---- Optional: Google Drive client (désactivé en v6 — aucun import externe)
build = None
MediaIoBaseDownload = None


# ---- Optional: project-specific OAuth helpers (si ton projet les a déjà)
# On essaie plusieurs noms possibles, sans casser si absent.
# ============================================================
# Google Drive (OAuth) — STUB v6
# ------------------------------------------------------------
# v6: aucune dépendance à des modules "services.*" ou Google API.
# L'import Drive est donc désactivé (utilise l'import local).
# ============================================================

def render_drive_oauth_connect_ui() -> None:
    st.info("Drive OAuth est désactivé dans cette version (v6). Utilise Import local (fallback).")

def _drive_service_from_existing_oauth():
    return None

def _drive_list_csv_files(svc, folder_id: str):
    return []

def _drive_download_file_bytes(svc, file_id: str) -> bytes:
    return b""

def _drive_upload_bytes(svc, folder_id: str, name: str, content: bytes, *, mime: str = "text/csv"):
    return None

def _read_csv_bytes(b: bytes, *, sep: str = "AUTO", on_bad_lines: str = "skip") -> pd.DataFrame:
    """Lecture CSV robuste (Fantrax / exports):
    - encodings: utf-8-sig -> utf-8 -> latin-1
    - sep: AUTO (sniff , ; 	 |) ou valeur explicite
    - fallback engine=python si le C-engine plante
    """
    import csv

    if not isinstance(b, (bytes, bytearray)) or not b:
        return pd.DataFrame()

    sample = b[:8192]

    # --- encoding detection
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    decoded = None
    used_enc = "utf-8-sig"
    for enc in encodings:
        try:
            decoded = sample.decode(enc)
            used_enc = enc
            break
        except Exception:
            continue
    if decoded is None:
        used_enc = "latin-1"

    # --- delimiter sniff
    used_sep = None
    if str(sep or "").upper() != "AUTO":
        used_sep = sep
    else:
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample.decode(used_enc, errors="ignore"), delimiters=[",", ";", "	", "|"])
            used_sep = dialect.delimiter
        except Exception:
            used_sep = None  # pandas will infer best effort

    # Try combos: (engine, sep)
    attempts = []
    if used_sep is None:
        attempts = [
            ("python", None),
            ("c", None),
        ]
    else:
        attempts = [
            ("c", used_sep),
            ("python", used_sep),
            ("python", None),
        ]

    last_err = None
    for engine, sep_try in attempts:
        try:
            return pd.read_csv(
                io.BytesIO(b),
                encoding=used_enc,
                sep=sep_try,
                engine=engine,
                on_bad_lines=on_bad_lines,
            )
        except TypeError:
            # pandas plus vieux: pas de on_bad_lines
            try:
                return pd.read_csv(
                    io.BytesIO(b),
                    encoding=used_enc,
                    sep=sep_try,
                    engine=engine,
                )
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e

    # dernier recours: latin-1 python engine, sans sniff
    try:
        return pd.read_csv(io.BytesIO(b), encoding="latin-1", engine="python")
    except Exception:
        if last_err:
            raise last_err
        return pd.DataFrame()


def _payload_to_bytes(payload: Any) -> Optional[bytes]:
    """Convertit un payload (bytes, UploadedFile, filepath) -> bytes."""
    try:
        if payload is None:
            return None
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        if isinstance(payload, str) and os.path.exists(payload):
            return Path(payload).read_bytes()
        if hasattr(payload, "getvalue"):
            return payload.getvalue()
        if hasattr(payload, "read"):
            return payload.read()
    except Exception:
        return None
    return None


def _drop_unnamed_and_dupes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    # Drop "Unnamed"
    drop_cols = [c for c in out.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        out.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Drop duplicated columns like "X.1"
    keep = []
    seen = set()
    for c in list(out.columns):
        base = str(c).split(".")[0]
        if base in seen:
            continue
        seen.add(base)
        keep.append(c)
    return out[keep].copy()


def normalize_team_import_df(df_raw: pd.DataFrame, owner_default: str, players_idx: dict) -> pd.DataFrame:
    """Normalise un export (Fantrax ou autre) vers le schéma EQUIPES_COLUMNS.
    Priorité absolue: colonne Player (nom complet) -> Joueur.
    """
    if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return pd.DataFrame(columns=EQUIPES_COLUMNS)

    df = _drop_unnamed_and_dupes(df_raw)

    # --- capture ID si présent (optionnel)
    id_col = _pick_col(df, ["ID", "Id", "PlayerId", "playerId", "FantraxID", "Fantrax Id"])
    if id_col and id_col != "FantraxID":
        df.rename(columns={id_col: "FantraxID"}, inplace=True)

    # --- Propriétaire
    owner_col = _pick_col(df, ["Propriétaire", "Proprietaire", "Owner", "Team Owner"])
    if owner_col and owner_col != "Propriétaire":
        df.rename(columns={owner_col: "Propriétaire"}, inplace=True)
    if "Propriétaire" not in df.columns:
        df["Propriétaire"] = str(owner_default or "").strip()

    # --- Joueur (PRIORITÉ: Player -> Joueur)
    if "Player" in df.columns:
        df.rename(columns={"Player": "Joueur"}, inplace=True)
    else:
        jcol = _pick_col(df, ["Joueur", "Skaters", "Name", "Player Name", "Full Name"])
        if jcol and jcol != "Joueur":
            df.rename(columns={jcol: "Joueur"}, inplace=True)

    if "Joueur" not in df.columns:
        df["Joueur"] = ""

    # Si Joueur ressemble à un ID (A/7/24...), on tente un autre champ nom
    j = df["Joueur"].astype(str).str.strip()
    try:
        looks_like_id = (j.str.len().fillna(0) <= 3).mean() > 0.6
    except Exception:
        looks_like_id = False
    if looks_like_id:
        for alt in ["Player", "Skaters", "Player Name", "Full Name", "Name"]:
            if alt in df.columns:
                df["Joueur"] = df[alt].astype(str).str.strip()
                break

    # --- Pos / Equipe / Salaire / Statut / IR Date / Level / Slot
    mappings = [
        (["Pos", "Position"], "Pos"),
        (["Team", "NHL Team", "Equipe", "Équipe"], "Equipe"),
        (["Salary", "Cap Hit", "CapHit", "Salaire"], "Salaire"),
        (["Status", "Roster Status", "Statut"], "Statut"),
        (["IR Date", "IRDate", "Date IR"], "IR Date"),
        (["Level", "Contract Level"], "Level"),
        (["Slot"], "Slot"),
    ]
    for src_list, dst in mappings:
        if dst in df.columns:
            continue
        scol = _pick_col(df, src_list)
        if scol and scol != dst:
            df.rename(columns={scol: dst}, inplace=True)

    if "Pos" not in df.columns:
        df["Pos"] = ""
    if "Equipe" not in df.columns:
        df["Equipe"] = ""
    if "Salaire" not in df.columns:
        df["Salaire"] = 0
    if "Level" not in df.columns:
        df["Level"] = ""
    if "Statut" not in df.columns:
        df["Statut"] = ""
    if "IR Date" not in df.columns:
        df["IR Date"] = ""
    if "Slot" not in df.columns:
        df["Slot"] = df["Statut"].apply(auto_slot_for_statut) if "Statut" in df.columns else "Actif"

    # Normalize salary int
    df["Salaire"] = pd.to_numeric(df["Salaire"], errors="coerce").fillna(0).astype(int)

    # Force owner if empty
    if owner_default:
        df["Propriétaire"] = df["Propriétaire"].astype(str).str.strip().replace({"": owner_default})
        df.loc[df["Propriétaire"].isna(), "Propriétaire"] = owner_default

    # Keep only expected columns (+ FantraxID if present)
    keep = [c for c in (EQUIPES_COLUMNS + ["FantraxID"]) if c in df.columns]
    df = df[keep].copy()
    return df


# ---- Optional NHL API enrichment (Pos/Equipe only; no salary from NHL)
try:
    import requests  # type: ignore
except Exception:
    requests = None


@st.cache_data(show_spinner=False)
def nhl_find_player_id(full_name: str) -> Optional[int]:
    if not requests:
        return None
    name = str(full_name or "").strip()
    if not name:
        return None
    try:
        url = "https://search.d3.nhle.com/api/v1/search/player"
        params = {"culture": "en-us", "limit": 5, "q": name}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json() or []
        if not data:
            return None
        pid = data[0].get("playerId") or data[0].get("id")
        return int(pid) if pid is not None else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def nhl_player_landing(player_id: int) -> dict:
    if not requests:
        return {}
    try:
        url = f"https://api-web.nhle.com/v1/player/{int(player_id)}/landing"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return {}
        return r.json() or {}
    except Exception:
        return {}


def enrich_df_from_nhl(df: pd.DataFrame) -> pd.DataFrame:
    """Complète Pos/Equipe si manquant. (Ne touche pas Salaire.)"""
    if df is None or df.empty or "Joueur" not in df.columns:
        return df

    out = df.copy()
    if "Pos" not in out.columns:
        out["Pos"] = ""
    if "Equipe" not in out.columns:
        out["Equipe"] = ""

    for idx, row in out.iterrows():
        name = str(row.get("Joueur") or "").strip()
        if not name:
            continue
        need_pos = str(row.get("Pos") or "").strip().lower() in ("", "nan", "none")
        need_team = str(row.get("Equipe") or "").strip().lower() in ("", "nan", "none")
        if not (need_pos or need_team):
            continue

        pid = nhl_find_player_id(name)
        if not pid:
            continue
        info = nhl_player_landing(pid)
        if not info:
            continue

        if need_pos:
            pos = info.get("position") or info.get("positionCode") or ""
            out.at[idx, "Pos"] = str(pos).strip()
        if need_team:
            team = info.get("currentTeamAbbrev") or (info.get("currentTeam") or {}).get("abbrev") or ""
            out.at[idx, "Equipe"] = str(team).strip()

    return out


# ============================================================
# MAIN RENDER
# ============================================================

# ============================================================

# =====================================================
# 🆔 BULK NHL_ID (AUTO) — par coup de 250 + checkpoint
#   - Safe: n'écrit que si match très confiant
#   - Sauvegarde atomique + cache bust
#   - Reprise via checkpoint (json)
# =====================================================

def _atomic_save_csv(df: pd.DataFrame, path: str) -> None:
    """Écriture atomique (évite fichiers partiels si crash)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _pms_norm_name(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if not s:
        return ""
    # 'A | A|' -> garder le 1er segment
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        if parts:
            s = parts[0]
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^A-Za-zÀ-ÿ'\- ]+", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _split_first_last(name: str) -> Tuple[str, str]:
    n = _pms_norm_name(name)
    if not n:
        return "", ""
    parts = n.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]


def _nhl_search_players(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Recherche joueurs via l'index public NHL (rapide)."""
    import requests  # local import (évite dépendance au top)

    q = (query or "").strip()
    if not q:
        return []
    url = "https://search.d3.nhle.com/api/v1/search/player"
    params = {"culture": "en-us", "limit": str(limit), "q": q}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
        data = r.json()
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _extract_nhl_id(item: Dict[str, Any]) -> Optional[int]:
    for k in ("playerId", "player_id", "id", "nhlId", "nhl_id"):
        if k in item and item[k] not in (None, ""):
            try:
                return int(str(item[k]).strip())
            except Exception:
                pass
    return None


def _extract_full_name(item: Dict[str, Any]) -> str:
    for k in ("name", "fullName", "full_name", "playerName"):
        if k in item and item[k]:
            return str(item[k])
    fn = item.get("firstName") or item.get("first_name") or ""
    ln = item.get("lastName") or item.get("last_name") or ""
    return f"{fn} {ln}".strip()


def _confidence_match(target_name: str, item: Dict[str, Any]) -> Tuple[bool, float]:
    """Match SAFE. ok=True seulement si score très élevé."""
    t = _pms_norm_name(target_name)
    if not t:
        return False, 0.0
    full = _pms_norm_name(_extract_full_name(item))
    if not full:
        return False, 0.0

    if full == t:
        return True, 1.0

    tf, tl = _split_first_last(t)
    ff, fl = _split_first_last(full)

    score = 0.0
    if tl and fl and tl == fl:
        score += 0.55
    if tf and ff and tf == ff:
        score += 0.35
    if tl and tl in full.split():
        score += 0.10

    return (score >= 0.90), score


def _ckpt_path(DATA_DIR: str, season_lbl: str) -> str:
    return os.path.join(DATA_DIR, f"_nhl_id_bulk_checkpoint_{season_lbl}.json")


def _load_ckpt(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        import json
        return json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return {}


def _save_ckpt(path: str, payload: Dict[str, Any]) -> None:
    try:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _clear_ckpt(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def render_bulk_nhl_id_admin(DATA_DIR: str, season_lbl: str, is_admin: bool) -> None:
    """Section Admin: associer automatiquement les NHL_ID manquants par batch (250)."""
    if not is_admin:
        st.warning("Accès admin requis.")
        return

    players_path = resolve_players_db_path(DATA_DIR)
    if not os.path.exists(players_path):
        st.error(f"Fichier introuvable: {players_path}")
        return

    df = load_players_db(players_path)
    if df is None or df.empty:
        st.warning("Players DB vide.")
        info = st.session_state.get('players_db_load_info') or {}
        if info.get('error'):
            st.error(f"Lecture CSV impossible: {info.get('error')}")
        # Toujours montrer un mini diagnostic
        st.caption(
            f"Chemin détecté: `{players_path}` | exists={info.get('exists')} | size={info.get('size')} | rows={info.get('rows')} | cols={len(info.get('cols') or [])}"
        )
        show_diag = st.checkbox("🔎 Afficher diagnostic hockey.players.csv", value=False, key="adm_diag_players_db")
        if show_diag:
            if info.get('raw_head'):
                st.code(info.get('raw_head'), language="text")
            st.write({k: info.get(k) for k in ['delimiter','rows','cols','error']})
        pm = os.path.join(DATA_DIR, "players_master.csv")
        if os.path.exists(pm):
            st.info("`players_master.csv` existe. Tu peux générer une Players DB minimale pour débloquer l’outil NHL_ID.")
            if st.button("🧱 Créer hockey.players.csv depuis players_master.csv", use_container_width=True, key="adm_bootstrap_players_db"):
                okb, msgb = bootstrap_players_db_from_master(DATA_DIR)
                if okb:
                    st.success(msgb)
                    st.rerun()
                else:
                    st.error(msgb)
        return

    if "__rowid" not in df.columns:
        df["__rowid"] = range(len(df))
    if "nhl_id" not in df.columns:
        df["nhl_id"] = ""

    missing = df[df["nhl_id"].astype(str).str.strip().eq("")]

    st.markdown("### 🆔 Bulk NHL_ID (AUTO) — par 250")
    st.caption("Safe: n’écrit un NHL_ID que si le match est très confiant. Sauvegarde + checkpoint pour reprise.")

    ckpt_file = _ckpt_path(DATA_DIR, season_lbl)
    ckpt = _load_ckpt(ckpt_file)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total joueurs", len(df))
    with c2:
        st.metric("NHL_ID manquants", int(missing.shape[0]))
    with c3:
        st.metric("Checkpoint", "Oui" if bool(ckpt) else "Non")

    if ckpt:
        with st.expander("Voir checkpoint", expanded=False):
            st.json(ckpt)

    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        batch_size = st.number_input("Batch (associations)", 50, 500, 250, step=50)
    with colB:
        max_requests = st.number_input("Max requêtes NHL/run", 50, 5000, 750, step=50)
    with colC:
        allow_exact_only = st.checkbox("Mode strict (exact seulement)", value=False)
    with colD:
        dry_run = st.checkbox("Dry run (pas de sauvegarde)", value=False)

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        do_run = st.button("🚀 Lancer / Reprendre", use_container_width=True, key="adm_bulk_nhl_run")
    with b2:
        do_reset = st.button("🧹 Reset checkpoint", use_container_width=True, key="adm_bulk_nhl_reset")
    with b3:
        st.write("")

    if do_reset:
        _clear_ckpt(ckpt_file)
        st.success("✅ Checkpoint supprimé.")
        st.rerun()

    if not do_run:
        st.info("Clique **Lancer / Reprendre** pour associer automatiquement (250 par run).")
        return

    # Resume
    work_ids = df[df["nhl_id"].astype(str).str.strip().eq("")]["__rowid"].tolist()
    next_rowid = ckpt.get("next_rowid") if ckpt else None
    if next_rowid in work_ids:
        work_ids = work_ids[work_ids.index(next_rowid):]

    if not work_ids:
        st.success("✅ Rien à faire.")
        _clear_ckpt(ckpt_file)
        return

    prog = st.progress(0)
    status = st.empty()

    updated = 0
    ambiguous = 0
    notfound = 0
    req_count = 0

    # Loop rows until we hit batch_size updates
    for i, rowid in enumerate(work_ids):
        if req_count >= int(max_requests):
            status.warning("⏸️ Arrêt (max requêtes atteint). Relance pour continuer.")
            break
        if updated >= int(batch_size):
            status.info("⏸️ Batch atteint. Sauvegarde + checkpoint.")
            break

        row = df.loc[df["__rowid"] == rowid].iloc[0]
        # name field variants
        name_raw = row.get("Player") or row.get("Joueur") or row.get("Nom") or row.get("name") or ""
        name = str(name_raw or "").strip()
        if not name:
            notfound += 1
            continue

        results = _nhl_search_players(name, limit=15)
        req_count += 1

        best_id = None
        best_score = 0.0

        if results:
            for it in results:
                pid = _extract_nhl_id(it)
                if pid is None:
                    continue
                full = _pms_norm_name(_extract_full_name(it))
                t = _pms_norm_name(name)
                if allow_exact_only:
                    ok = (full == t)
                    score = 1.0 if ok else 0.0
                else:
                    ok, score = _confidence_match(name, it)
                if ok and score >= best_score:
                    best_score = score
                    best_id = pid

        if best_id is None:
            if results:
                ambiguous += 1
            else:
                notfound += 1
        else:
            df.loc[df["__rowid"] == rowid, "nhl_id"] = int(best_id)
            updated += 1

        pct = int((i + 1) / max(1, len(work_ids)) * 100)
        prog.progress(min(100, pct))
        status.write(
            f"rowid={rowid} | req={req_count} | ✅ associés={updated} | ⚠️ ambigus={ambiguous} | ❌ introuvables={notfound}"
        )

    # Save
    if updated > 0 and not dry_run:
        _atomic_save_csv(df.drop(columns=["__pkey"], errors="ignore"), players_path)
        st.cache_data.clear()
        st.session_state["players_db_nonce"] = str(time.time())
        st.success(f"✅ Sauvegardé: {updated} NHL_ID ajoutés dans {PLAYERS_DB_FILENAME}")
    elif updated > 0 and dry_run:
        st.warning(f"Dry run: {updated} associations trouvées, aucune sauvegarde.")

    # Next checkpoint
    next_rowid_out = None
    if 'i' in locals() and (i + 1) < len(work_ids):
        next_rowid_out = work_ids[i + 1]

    if next_rowid_out is not None:
        _save_ckpt(ckpt_file, {
            "next_rowid": next_rowid_out,
            "ts": time.time(),
            "batch_size": int(batch_size),
            "max_requests": int(max_requests),
        })
        st.warning("Checkpoint écrit — relance pour continuer.")
    else:
        _clear_ckpt(ckpt_file)
        st.success("🎉 Terminé — il ne reste que des cas ambigus / introuvables.")

    st.markdown("#### Résumé")
    st.write({"associés": updated, "ambigus": ambiguous, "introuvables": notfound, "requêtes": req_count})

# =====================================================
# players_master — fusion vers un fichier unique par saison
# =====================================================

def players_master_path(data_dir: str, season_lbl: str) -> str:
    season_lbl = str(season_lbl or "").strip() or "season"
    return os.path.join(str(data_dir), f"players_master_{season_lbl}.csv")


def _norm_name_key(first: str, last: str) -> str:
    s = f"{last or ''}, {first or ''}".strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9, '\-]", "", s)
    return s.strip()


def _split_player_name(player: str) -> tuple[str, str]:
    p = str(player or "").strip()
    if not p:
        return ("", "")
    if "," in p:
        last, first = [x.strip() for x in p.split(",", 1)]
        return (first, last)
    parts = p.split()
    if len(parts) == 1:
        return ("", parts[0])
    first = " ".join(parts[:-1]).strip()
    last = parts[-1].strip()
    return (first, last)


def _safe_read_csv(path: str) -> pd.DataFrame:
    """Lecture CSV robuste (détecte ',' / ';' / '	' / sniff python)."""
    try:
        # 1) Essai standard
        return pd.read_csv(path, low_memory=False)
    except Exception:
        pass
    # 2) Sniff automatique (engine python)
    try:
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    except Exception:
        pass
    # 3) Separateurs fréquents
    for sep in [";", "	", "|"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python", low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def _discover_team_csvs(data_dir: str) -> list[str]:
    """Liste les CSV d'équipes (Whalers.csv, etc.) dans /data."""
    dd = pathlib.Path(str(data_dir))
    if not dd.exists():
        return []
    out: list[str] = []
    for fp in dd.glob("*.csv"):
        name = fp.name.lower()
        # fichiers non-équipe
        if "equipes_joueurs" in name:
            continue
        if "players_master" in name:
            continue
        if "puckpedia" in name:
            continue
        if "backup" in name:
            continue
        if "event_log" in name:
            continue
        if name in {"hockey_players.csv", "hockey.players.csv", "hockey.players.csv", "hockey.players.csv"}:
            continue
        out.append(str(fp))
    return sorted(out)


def _extract_roster_from_team_csv(path: str) -> pd.DataFrame:
    """Parse un CSV d'équipe type Fantrax (comme ta capture)."""
    raw = _safe_read_csv(path)
    if raw.empty:
        return raw

    cols = [c.strip() for c in raw.columns.astype(str).tolist()]
    if "Player" not in cols and "player" not in [c.lower() for c in cols]:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            header_i = None
            for i, line in enumerate(lines[:120]):
                if "Player" in line and "Team" in line and "Pos" in line:
                    header_i = i
                    break
            if header_i is not None:
                raw = pd.read_csv(path, skiprows=header_i, low_memory=False)
        except Exception:
            pass

    if raw.empty:
        return raw

    raw.columns = [str(c).strip() for c in raw.columns]
    colmap = {c.lower(): c for c in raw.columns}

    def _get(k: str) -> str:
        return colmap.get(k, "")

    c_player = _get("player")
    c_pos = _get("pos")
    c_team = _get("team")
    c_status = _get("status")
    c_age = _get("age")
    c_salary = _get("salary")

    if not c_player:
        return pd.DataFrame()

    df = raw.copy()
    df = df[df[c_player].notna()]
    df[c_player] = df[c_player].astype(str).str.strip()
    df = df[df[c_player].str.len() > 0]

    bad = {"skaters", "goalies", "player", "pos"}
    df = df[~df[c_player].str.lower().isin(bad)]

    pool_team = pathlib.Path(path).stem.replace("_", " ").strip()
    out = pd.DataFrame({
        "pool_team": pool_team,
        "player": df[c_player].astype(str),
        "pos_raw": df[c_pos].astype(str) if c_pos else "",
        "team_nhl": df[c_team].astype(str) if c_team else "",
        "slot": df[c_status].astype(str) if c_status else "",
        "age": df[c_age] if c_age else None,
        "salary": df[c_salary] if c_salary else None,
    })
    return out


def _assignments_from_sources(data_dir: str, season_lbl: str, prefer_equipes_fused: bool, read_team_csvs: bool) -> pd.DataFrame:
    assigns = []

    if prefer_equipes_fused:
        epath = equipes_path(data_dir, season_lbl)
        df_eq = _safe_read_csv(epath)
        if not df_eq.empty:
            cols = {c.lower(): c for c in df_eq.columns}
            c_j = cols.get("joueur") or cols.get("player") or cols.get("nom") or ""
            c_owner = cols.get("proprietaire") or cols.get("owner") or cols.get("équipe") or cols.get("equipe") or ""
            c_slot = cols.get("slot") or cols.get("statut") or cols.get("status") or ""
            if c_j and c_owner:
                tmp = df_eq.copy()
                tmp["_player_raw"] = tmp[c_j].astype(str)
                fl = tmp["_player_raw"].apply(lambda s: _split_player_name(s))
                tmp["_first"] = [a for a, b in fl]
                tmp["_last"] = [b for a, b in fl]
                tmp["key"] = tmp.apply(lambda r: _norm_name_key(r.get("_first", ""), r.get("_last", "")), axis=1)
                tmp["pool_team"] = tmp[c_owner].astype(str).str.strip()
                tmp["slot"] = tmp[c_slot].astype(str).str.strip() if c_slot else ""
                assigns.append(tmp[["key", "pool_team", "slot"]])

    if (not assigns) and read_team_csvs:
        for fp in _discover_team_csvs(data_dir):
            df_team = _extract_roster_from_team_csv(fp)
            if df_team is None or df_team.empty:
                continue
            fl = df_team["player"].apply(lambda s: _split_player_name(s))
            df_team["_first"] = [a for a, b in fl]
            df_team["_last"] = [b for a, b in fl]
            df_team["key"] = df_team.apply(lambda r: _norm_name_key(r.get("_first", ""), r.get("_last", "")), axis=1)
            assigns.append(df_team[["key", "pool_team", "slot"]])

    if not assigns:
        return pd.DataFrame(columns=["key", "pool_team", "slot"])

    out = pd.concat(assigns, ignore_index=True).dropna(subset=["key"])
    out["key"] = out["key"].astype(str)
    out = out[out["key"].str.len() > 0]
    out = out.drop_duplicates(subset=["key"], keep="first")
    return out


def build_players_master(
    data_dir: str,
    season_lbl: str,
    batch_size: int = 250,
    dry_run: bool = True,
    prefer_equipes_fused: bool = True,
    read_team_csvs: bool = True,
    progress_cb=None,
    **_ignored_kwargs,
) -> dict:
    """Construit data/players_master.csv (source de vérité) à partir de:
    - hockey.players.csv / Hockey.Players.csv (infos joueurs)
    - CSV Fantrax (1 par équipe): Whalers.csv, Canadiens.csv, etc. -> ownership + disponibilité
    - data/equipes_joueurs_YYYY-YYYY.csv (fallback ownership si rosters absents)

    IMPORTANT:
    - En mode dry_run=True, la fusion est considérée OK (ok=True) mais n'écrit rien.
    - Ne fait PAS de st.progress/st.empty ici: on utilise progress_cb si fourni (évite erreurs Streamlit).
    """
    import pathlib

    def _p(msg: str, i: int = 0, total: int = 0) -> None:
        if callable(progress_cb):
            try:
                progress_cb(str(msg), int(i), int(total))
            except Exception:
                pass

    dd = pathlib.Path(str(data_dir))
    dd.mkdir(parents=True, exist_ok=True)

    issues: list[str] = []

    # --- Trouver hockey.players.csv (source joueurs)
    hp_candidates = [
        dd / "hockey.players.csv",
        dd / "Hockey.Players.csv",
        dd / "hockey_players.csv",
        dd / "players.csv",
    ]
    hp_path = next((p for p in hp_candidates if p.exists()), None)
    if not hp_path:
        issues.append("hockey.players.csv introuvable — base joueurs vide (fallback).")
    else:
        _p(f"Lecture base joueurs: {hp_path.name}", 1, 10)

    # --- Charger base players (peut être vide si absent)
    hp = pd.DataFrame()
    if hp_path:
        hp = _safe_read_csv(str(hp_path))
        if hp is None:
            hp = pd.DataFrame()
        if hp.empty:
            issues.append(f"Base joueurs vide après lecture: {hp_path.name}")
    else:
        hp = pd.DataFrame()

    # Colonnes "player name" possibles
    name_col = _first_existing_col(hp, ["Player","player","Joueur","joueur","Name","name","Nom","nom"])
    first_col = _first_existing_col(hp, ["Prénom","Prenom","First","FirstName","first_name","firstname"])
    last_col  = _first_existing_col(hp, ["Nom","Last","LastName","last_name","lastname"])

    # Construire une clé joueur normalisée
    if hp is not None and not hp.empty:
        if name_col:
            hp["_player_raw"] = hp[name_col].astype(str)
        elif first_col and last_col:
            hp["_player_raw"] = (hp[last_col].astype(str) + ", " + hp[first_col].astype(str))
        else:
            hp["_player_raw"] = hp.iloc[:, 0].astype(str)

        hp["_pkey"] = hp["_player_raw"].apply(_norm_player_key)
        hp = hp[hp["_pkey"].astype(bool)].copy()
    else:
        hp = pd.DataFrame(columns=["_player_raw","_pkey"])

    # Préparer master (au minimum les joueurs de hp)
    if not hp.empty:
        master = pd.DataFrame({"pkey": hp["_pkey"].values}).drop_duplicates("pkey").reset_index(drop=True)
    else:
        master = pd.DataFrame(columns=["pkey"])

    # --- Lire rosters (Whalers.csv etc.)
    team_csvs = _discover_team_csvs(str(dd)) if read_team_csvs else []
    roster_frames: list[pd.DataFrame] = []

    _p(f"Découverte rosters: {len(team_csvs)} fichier(s)", 2, 10)

    for i, fp in enumerate(team_csvs, start=1):
        try:
            team_name = pathlib.Path(fp).stem.replace("_", " ").strip()
            df = _safe_read_csv(fp)
            r = _extract_roster_from_fantrax(df, team_name=team_name)
            if not r.empty:
                roster_frames.append(r)
        except Exception as e:
            issues.append(f"Roster non lisible: {pathlib.Path(fp).name} ({e})")
        _p(f"Lecture rosters ({i}/{max(len(team_csvs),1)})", 2 + i, 2 + max(len(team_csvs),1))

    rosters = pd.concat(roster_frames, ignore_index=True) if roster_frames else pd.DataFrame(columns=["pkey","pool_team","pool_status"])
    if not rosters.empty:
        rosters["pkey"] = rosters["pkey"].apply(_norm_player_key)
        rosters = rosters[rosters["pkey"].astype(bool)].copy()
        rosters = rosters.drop_duplicates(["pkey","pool_team"], keep="first")

    # Si hp vide mais rosters non vides: créer master à partir des rosters
    if master.empty and not rosters.empty:
        master = pd.DataFrame({"pkey": rosters["pkey"].unique()}).drop_duplicates("pkey").reset_index(drop=True)

    # --- Ownership principal depuis rosters
    owner_map: dict[str, str] = {}
    status_map: dict[str, str] = {}
    if not rosters.empty:
        for _, row in rosters.iterrows():
            pk = str(row.get("pkey", "")).strip()
            if pk and pk not in owner_map:
                owner_map[pk] = str(row.get("pool_team","")).strip()
                status_map[pk] = str(row.get("pool_status","")).strip()

    master["pool_team"] = master["pkey"].map(owner_map).fillna("")
    master["pool_status"] = master["pkey"].map(status_map).fillna("")
    master["is_available"] = master["pool_team"].eq("")

    # --- Fallback ownership depuis equipes_joueurs_*
    eq_candidates = sorted(dd.glob("equipes_joueurs_*.csv"))
    eq_path = eq_candidates[-1] if eq_candidates else None
    if eq_path and master["is_available"].any():
        try:
            eq = _safe_read_csv(str(eq_path))
            eq_key_col = _first_existing_col(eq, ["pkey","player_key","Joueur","Player","player","Name","Nom"])
            eq_team_col = _first_existing_col(eq, ["Proprietaire","Propriétaire","Owner","Equipe_Pool","Équipe Pool","Équipe","Equipe"])
            if eq_key_col and eq_team_col and not eq.empty:
                eq["_pkey"] = eq[eq_key_col].astype(str).apply(_norm_player_key)
                eq_map = dict(zip(eq["_pkey"], eq[eq_team_col].astype(str)))
                mask = master["pool_team"].eq("") & master["pkey"].isin(eq_map.keys())
                master.loc[mask, "pool_team"] = master.loc[mask, "pkey"].map(eq_map).fillna("")
                master["is_available"] = master["pool_team"].eq("")
        except Exception as e:
            issues.append(f"Fallback equipes_joueurs impossible: {eq_path.name} ({e})")

    # --- Enrichissements depuis hp (si dispo)
    if hp is not None and not hp.empty:
        nhl_id_col = _first_existing_col(hp, ["NHL_ID","nhl_id","NHL Id","NHLID","playerId","player_id"])
        if nhl_id_col:
            master["nhl_id"] = master["pkey"].map(dict(zip(hp["_pkey"], hp[nhl_id_col])))
        else:
            master["nhl_id"] = ""

        team_col = _first_existing_col(hp, ["Team","Équipe","Equipe","Team (NHL)","NHL Team","team_abbr","NHLTeam"])
        master["team"] = master["pkey"].map(dict(zip(hp["_pkey"], hp[team_col].astype(str)))).fillna("") if team_col else ""

        pos_col = _first_existing_col(hp, ["Position","Pos","pos","Position (NHL)","position"])
        master["position"] = master["pkey"].map(dict(zip(hp["_pkey"], hp[pos_col].astype(str)))).fillna("") if pos_col else ""

        cap_col = _first_existing_col(hp, ["Cap Hit","CapHit","cap_hit","Salary","Salaire","AAV","aav"])
        master["cap_hit"] = master["pkey"].map(dict(zip(hp["_pkey"], hp[cap_col]))).fillna("") if cap_col else ""

        level_col = _first_existing_col(hp, ["Level","level","Niveau","Contract Level"])
        master["level"] = master["pkey"].map(dict(zip(hp["_pkey"], hp[level_col].astype(str)))).fillna("") if level_col else ""

        # Nom affiché
        nm = dict(zip(hp["_pkey"], hp["_player_raw"].astype(str)))
        master["player"] = master["pkey"].map(nm).fillna("")
    else:
        for col in ["nhl_id","team","position","cap_hit","level"]:
            master[col] = ""
        master["player"] = ""

    # Fallback display name
    if "player" in master.columns:
        empty_name = master["player"].astype(str).str.strip().eq("")
        master.loc[empty_name, "player"] = master.loc[empty_name, "pkey"].apply(
            lambda k: " ".join([p.strip() for p in str(k).split(" ") if p.strip()]).title()
        )

    # --- Résumé
    out_path = dd / "players_master.csv"
    n = int(len(master))
    base_n = int(len(hp)) if hp is not None else 0
    rosters_n = int(len(rosters)) if rosters is not None else 0

    if n == 0:
        issues.append("Fusion: 0 joueur construit (base et rosters vides).")

    _p(f"Fusion prête: {n} joueurs (base={base_n} / rosters={rosters_n})", 9, 10)

    # --- Dry run
    if dry_run:
        _p("Dry run: aucune écriture effectuée.", 10, 10)
        return {
            "ok": n > 0,
            "dry_run": True,
            "out_path": str(out_path),
            "rows_out": n,
            "base_rows": base_n,
            "rosters_rows": rosters_n,
            "hp_path": str(hp_path) if hp_path else "",
            "team_csvs": team_csvs,
            "issues": issues,
        }

    # --- Écriture batch + swap atomique
    tmp = dd / "players_master.__tmp__.csv"
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    try:
        master.iloc[0:0].to_csv(tmp, index=False)
        written = 0
        step = max(1, int(batch_size))
        for i in range(0, n, step):
            chunk = master.iloc[i:i+step]
            chunk.to_csv(tmp, index=False, header=False, mode="a")
            written += len(chunk)
            _p(f"Écriture players_master.csv: {written}/{n}", min(10, 5 + int(5*written/max(n,1))), 10)

        if out_path.exists():
            out_path.unlink()
        tmp.rename(out_path)
        _p(f"✅ players_master.csv écrit: {n} lignes", 10, 10)

        return {
            "ok": True,
            "dry_run": False,
            "out_path": str(out_path),
            "rows_out": n,
            "base_rows": base_n,
            "rosters_rows": rosters_n,
            "hp_path": str(hp_path) if hp_path else "",
            "team_csvs": team_csvs,
            "issues": issues,
        }
    except Exception as e:
        issues.append(f"Écriture impossible: {e}")
        return {
            "ok": False,
            "dry_run": False,
            "out_path": str(out_path),
            "rows_out": n,
            "base_rows": base_n,
            "rosters_rows": rosters_n,
            "hp_path": str(hp_path) if hp_path else "",
            "team_csvs": team_csvs,
            "issues": issues,
        }
def render(ctx: dict) -> None:
    # --- Admin guard (source unique: ctx['is_admin'] défini dans app.py)
    is_admin = bool(ctx.get("is_admin"))
    if not is_admin:
        st.warning("Accès admin requis.")
        return

    DATA_DIR = str(ctx.get("DATA_DIR") or ("data" if os.path.isdir("data") else "Data"))
    os.makedirs(DATA_DIR, exist_ok=True)

    season_lbl = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    folder_id = str(ctx.get("drive_folder_id") or "").strip()

    e_path = equipes_path(DATA_DIR, season_lbl)
    log_path = admin_log_path(DATA_DIR, season_lbl)

    st.subheader("🛠️ Gestion Admin")

    # Panel (persistant) pour éviter de "retomber" sur Import après un rerun
    panel = st.radio(
        "Panel",
        ["Fusion", "Import", "Autres"],
        horizontal=True,
        label_visibility="collapsed",
        key="admin_panel",
    )


    # ---- OAuth UI (si ton projet l'avait déjà)
    with st.expander("🔐 Connexion Google Drive (OAuth)", expanded=False):
        if callable(_oauth_ui):
            try:
                _oauth_ui()
            except Exception:
                st.info("UI OAuth présente mais a échoué — vérifie tes secrets OAuth.")
        else:
            st.caption("Aucune UI OAuth détectée dans services/*. Tu peux quand même importer en local (fallback).")
            if callable(_oauth_enabled):
                try:
                    st.write("oauth_drive_enabled():", bool(_oauth_enabled()))
                except Exception:
                    pass

    # ---- caps inputs
    st.session_state.setdefault("CAP_GC", DEFAULT_CAP_GC)
    st.session_state.setdefault("CAP_CE", DEFAULT_CAP_CE)

    with st.expander("🧪 Vérification cap (live) + barres", expanded=(panel == "Autres")):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.session_state["CAP_GC"] = st.number_input("Cap GC", min_value=0, value=int(st.session_state["CAP_GC"]), step=500000)
        with c2:
            st.session_state["CAP_CE"] = st.number_input("Cap CE", min_value=0, value=int(st.session_state["CAP_CE"]), step=250000)
        with c3:
            st.caption("Caps utilisés ici pour affichage & alertes.")
        df_eq = load_equipes(e_path)
        if df_eq.empty:
            st.info("Aucun fichier équipes local trouvé (importe depuis Drive ou local).")
        else:
            _render_caps_bars(df_eq, int(st.session_state["CAP_GC"]), int(st.session_state["CAP_CE"]))

    # ---- Players DB index
    players_db = load_players_db(resolve_players_db_path(DATA_DIR))
    players_idx = build_players_index(players_db)
    if players_idx:
        st.success(f"Players DB détectée: {PLAYERS_DB_FILENAME} (Level auto + infos).")
    else:
        st.warning(f"{PLAYERS_DB_FILENAME} indisponible — fallback Level par Salaire.")
        st.caption(f"Chemin recherché: `{resolve_players_db_path(DATA_DIR)}` (exists={os.path.exists(resolve_players_db_path(DATA_DIR))})")

    # ---- Load équipes
    df = load_equipes(e_path)

    # =====================================================
    
    # =====================================================
    # 🧬 FUSION — construire un players_master.csv unique (et fiable)
    #   Objectif: arrêter de courir après 6 CSV + un CSV fusion.
    #   → crée/maj: data/players_master.csv
    #   - lit hockey.players.csv + equipes_joueurs_{season}.csv + (option) CSV équipes
    #   - unifie colonnes + positions + level + disponibilité
    #   - écriture en batch (250) + progression visible
    # =====================================================
    with st.expander("🧬 Fusion (master) — construire players_master.csv", expanded=(panel == "Fusion")):
        st.caption("Crée un fichier **unique** `players_master.csv` qui sert de source de vérité (disponibilité / équipes pool / NHL_ID / stats pro).")
        colA, colB, colC = st.columns([1,1,2])
        with colA:
            dry_run = st.toggle("Dry run", value=True, help="Ne rien écrire, seulement analyser.")
        with colB:
            bs = st.number_input("Batch", min_value=50, max_value=1000, value=250, step=50)
        with colC:
            st.markdown("<div class='muted'>Astuce: commence en <b>dry run</b>, puis décoche pour écrire.</div>", unsafe_allow_html=True)

        run = st.button("🧬 Lancer la fusion (players_master.csv)", use_container_width=True, key="run_fusion_master")
        if run:
            prog = st.progress(0.0)
            prog_lbl = st.empty()
            status_ph = st.empty()
            status_ph.info("Fusion en cours…")

            def _cb(msg: str, i: int = 0, total: int = 0):
                # Progress callback (no nested expanders / no st.status)
                if total and total > 0:
                    pct = min(1.0, max(0.0, float(i) / float(total)))
                    prog.progress(pct)
                    prog_lbl.caption(f"{msg} ({i}/{total})")
                else:
                    prog_lbl.caption(msg)

            try:
                res = build_players_master(
                    data_dir=DATA_DIR,
                    season_lbl=season_lbl,
                    dry_run=bool(dry_run),
                    prefer_equipes_fused=True,
                    read_team_csvs=True,
                    batch_size=int(bs),
                    progress_cb=_cb,
                )
                ok = bool(res.get("ok"))
                out_path = res.get("out_path") or ""
                n_rows = int(res.get("rows_out") or 0)
                issues = res.get("issues") or []
                if ok:
                    status_ph.success(f"✅ Fusion OK — {n_rows} lignes")
                    st.success(f"✅ Fusion OK — {n_rows} lignes → {out_path or 'players_master.csv'}")
                    if issues:
                        with st.expander("⚠️ Notes / issues détectés", expanded=False):
                            for it in issues[:200]:
                                st.write("•", it)
                else:
                    status_ph.error("❌ Fusion échouée")
                    st.error("❌ Fusion échouée. Voir détails ci-dessous.")
                    st.json(res)
            except Exception as e:
                status_ph.error("❌ Fusion — exception")
                st.exception(e)

    # =====================================================
    # 🔗 NHL_ID — bulk association (AUTO 250/batch) + manuel
    # =====================================================
    with st.expander("🔗 Associer NHL_ID manquants (AUTO — 250 par run)", expanded=(panel == "Fusion")):
        render_bulk_nhl_id_admin(DATA_DIR, season_lbl, is_admin)

# 🔄 IMPORT ÉQUIPES (Drive)
    # =====================================================
    with st.expander("🔄 Import équipes depuis Drive (OAuth)", expanded=(panel == "Import")):
        st.caption("Lister/télécharger les CSV dans ton folder_id. Si ça ne marche pas, utilise Import local (fallback).")
        st.write(f"folder_id (ctx): `{folder_id or ''}`")

        svc = _drive_service_from_existing_oauth()
        drive_ok = bool(svc) and bool(folder_id)

        if not drive_ok:
            st.warning("Drive OAuth non disponible (creds manquants ou service indisponible).")
            st.caption("Conseil: ouvre l’expander 'Connexion Google Drive (OAuth)' et connecte-toi.")
        else:
            files = _drive_list_csv_files(svc, folder_id)
            equipes_files = [f for f in files if "equipes_joueurs" in f["name"].lower()]

            if not equipes_files:
                st.info("Aucun fichier `equipes_joueurs...csv` trouvé sur Drive.")
            else:
                pick = st.selectbox("Choisir un CSV équipes (Drive)", equipes_files, format_func=lambda x: x["name"], key="adm_drive_pick")

                colA, colB, colC = st.columns([1, 1, 1])
                do_preview = colA.button("🧼 Preview", use_container_width=True, key="adm_drive_preview")
                do_validate = colB.button("🧪 Valider colonnes", use_container_width=True, key="adm_drive_validate")
                do_import = colC.button("⬇️ Importer → Local + QC + Reload", use_container_width=True, key="adm_drive_import")

                df_drive = None
                if do_preview or do_validate or do_import:
                    try:
                        b = _drive_download_bytes(svc, pick["id"])
                        df_drive = _read_csv_bytes(b)
                    except Exception as e:
                        st.error(f"Erreur téléchargement/lecture: {e}")

                if isinstance(df_drive, pd.DataFrame):
                    st.caption(f"Source: {pick['name']}")
                    st.dataframe(df_drive.head(80), use_container_width=True)

                    ok, missing, extras = validate_equipes_df(df_drive)
                    if do_validate:
                        if ok:
                            st.success("✅ Colonnes attendues OK.")
                            if extras:
                                st.info(f"Colonnes additionnelles: {extras}")
                        else:
                            st.error(f"❌ Colonnes manquantes: {missing}")
                            if extras:
                                st.info(f"Colonnes additionnelles: {extras}")

                    if do_import:
                        if not ok:
                            st.error(f"Import refusé: colonnes manquantes {missing}")
                        else:
                            df_imp = ensure_equipes_df(df_drive)
                            df_imp_qc, stats = apply_quality(df_imp, players_idx)
                            save_equipes(df_imp_qc, e_path)
                            st.session_state["equipes_df"] = df_imp_qc
                            append_admin_log(
                                log_path,
                                action="IMPORT",
                                owner="",
                                player="",
                                note=f"drive={pick['name']}; level_auto={stats.get('level_autofilled',0)}"
                            )
                            st.success(f"✅ Import OK → {os.path.basename(e_path)} | Level auto: {stats.get('level_autofilled',0)}")
                            st.rerun()

    # =====================================================
    # 📥 IMPORT LOCAL (fallback)
    # =====================================================

    # =====================================================
    # 📥 IMPORT LOCAL (fallback) — multi-upload CSV équipes
    #   Objectif: importer plusieurs fichiers (1 équipe par fichier)
    #   - Auto: si colonne "Propriétaire" contient 1 valeur unique -> assign auto
    #   - Sinon: tu choisis l’équipe dans un dropdown (et on force la colonne)
    #   - Merge: append dans equipes_joueurs_{season}.csv (option replace par équipe)
    # =====================================================
    
    # =====================================================
    # 📥 IMPORT LOCAL (fallback) — multi-upload CSV équipes (+ ZIP)
    #   - 1 fichier = 1 équipe
    #   - Auto-assign via colonne Propriétaire (si unique) ou via nom de fichier (ex: Whalers.csv)
    #   - Mode: append ou remplacer l’équipe
    #   - Backup local par équipe avant modification
    # =====================================================
    with st.expander("📥 Import local (fallback) — multi-upload CSV équipes", expanded=(panel == "Import")):
        st.caption("Upload plusieurs CSV (1 par équipe). Auto-assign via `Propriétaire` (si unique) ou via le nom du fichier (ex: `Whalers.csv`).")
        st.code(f"Destination locale (fusion): {e_path}", language="text")
        # Options lecture/normalisation
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            sep_choice = st.selectbox("Delimiter (AUTO par défaut)", ["AUTO", ",", ";", "\t", "|"], index=0, key="adm_sep_choice")
        with colB:
            ignore_bad = st.checkbox("Ignorer lignes brisées (on_bad_lines='skip')", value=True, key="adm_ignore_bad_lines")
        with colC:
            adm_nhl_enrich = st.checkbox("Compléter Pos/Equipe via NHL API (si manquant)", value=False, key="adm_nhl_enrich")
        adm_nhl_enrich = bool(st.session_state.get("adm_nhl_enrich", adm_nhl_enrich))

        mode = st.radio(
            "Mode de fusion",
            ["Ajouter (append)", "Remplacer l’équipe (delete puis insert)"],
            horizontal=True,
            key="adm_multi_mode",
        )

        use_from_data = st.checkbox(
            "📂 Utiliser les fichiers déjà présents dans /data (sans upload)",
            value=True,
            key="adm_use_data_files",
        )
        data_csvs = []
        try:
            data_csvs = sorted([
                f for f in os.listdir(DATA_DIR)
                if f.lower().endswith(".csv")
                and f.lower() not in {PLAYERS_DB_FILENAME.lower()}
                and not f.lower().startswith("equipes_joueurs_")
                and not f.lower().startswith("backup_")
                and not f.lower().startswith("admin_log_")
            ])
        except Exception:
            data_csvs = []

        selected_data_csvs = st.multiselect(
            "Fichiers CSV dans /data à importer (un fichier par équipe)",
            options=data_csvs,
            default=data_csvs,
            disabled=not use_from_data,
            key="adm_data_csvs_sel",
        )

        uploads = []
        zip_up = None
        if not use_from_data:
            uploads = st.file_uploader(
                "Uploader un ou plusieurs CSV (équipes)",
                type=["csv"],
                accept_multiple_files=True,
                key="adm_multi_uploads",
            )
            zip_up = st.file_uploader(
                "Ou uploader un ZIP contenant plusieurs CSV",
                type=["zip"],
                key="adm_multi_zip",
                help="Fallback: si tu préfères déposer un seul fichier .zip.",
            )
        else:
            st.caption("Les équipes sont auto-déduites du nom du fichier (ex: Whalers.csv → Whalers).")
        items: List[Tuple[str, Any]] = []
        if use_from_data and selected_data_csvs:
            for fn in selected_data_csvs:
                items.append((fn, os.path.join(DATA_DIR, fn)))
        if uploads:
            for f in uploads:
                items.append((getattr(f, "name", "upload.csv"), f))

        if zip_up is not None:
            try:
                z = zipfile.ZipFile(zip_up)
                for name in z.namelist():
                    if name.lower().endswith(".csv") and not name.endswith("/"):
                        items.append((name, z.read(name)))
            except Exception as e:
                st.error(f"ZIP invalide: {e}")

        if not items:
            st.info("Ajoute 1+ fichiers (multi) ou un ZIP de CSV.")
        else:
            prep = st.button("🧼 Préparer les fichiers (analyse + attribution)", use_container_width=True, key="adm_multi_prepare")
            st.caption("Astuce: l'analyse démarre seulement quand tu cliques, pour éviter les reruns qui cassent à la sélection.")
            if not prep:
                st.stop()
            parsed: List[Dict[str, Any]] = []
            errors: List[Tuple[str, str]] = []

            for file_name, payload in items:
                # read + normalize (robuste)
                try:
                    b = _payload_to_bytes(payload)
                    if b is None:
                        raise ValueError("payload vide / non supporté")
                    df_raw = _read_csv_bytes(b, sep=sep_choice, on_bad_lines=('skip' if ignore_bad else 'error'))
                except Exception as e:
                    errors.append((file_name, f"{type(e).__name__}: {e}"))
                    continue

                df_up = normalize_team_import_df(df_raw, owner_default="", players_idx=players_idx)

                owners_in_file = sorted([
                    x for x in df_up.get("Propriétaire", pd.Series(dtype=str)).astype(str).str.strip().unique()
                    if x and x.lower() != "nan"
                ])

                parsed.append({"file": file_name, "df": df_up, "owners_in_file": owners_in_file})

            if errors:
                st.error("Certains fichiers ont des erreurs et seront ignorés :")
                st.dataframe(pd.DataFrame(errors, columns=["Fichier", "Erreur"]), use_container_width=True)

            if not parsed:
                st.warning("Aucun fichier valide à importer.")
            else:
                df_current = load_equipes(e_path)
                owners_choices = sorted([
                    x for x in df_current.get("Propriétaire", pd.Series(dtype=str))
                    .dropna().astype(str).str.strip().unique() if x
                ])

                st.markdown("### Attribution des fichiers → équipe")
                assignments: List[Tuple[Dict[str, Any], str]] = []
                for i, p in enumerate(parsed):
                    owners_in_file = p["owners_in_file"]
                    preferred = owners_in_file[0] if len(owners_in_file) == 1 else ""
                    if not preferred:
                        preferred = infer_owner_from_filename(p["file"], owners_choices)

                    c1, c2, c3 = st.columns([2, 2, 3])
                    with c1:
                        st.write(f"**{p['file']}**")
                        st.caption(f"Lignes: {len(p['df'])} | Owners détectés: {', '.join(owners_in_file) if owners_in_file else '—'}")
                    with c2:
                        if owners_choices:
                            idx = owners_choices.index(preferred) if preferred in owners_choices else 0
                            chosen = st.selectbox("Équipe", owners_choices, index=idx, key=f"adm_multi_owner_{i}")
                        else:
                            chosen = st.text_input("Équipe", value=preferred, key=f"adm_multi_owner_txt_{i}").strip()
                    with c3:
                        st.caption("Preview")
                        st.dataframe(p["df"].head(10), use_container_width=True)

                    assignments.append((p, chosen))

                missing_choice = [p["file"] for p, chosen in assignments if not str(chosen or "").strip()]
                if missing_choice:
                    st.warning("Choisis une équipe pour: " + ", ".join(missing_choice))

                colA, colB = st.columns([1, 1])
                do_import = colA.button("⬇️ Importer tous → Local + QC + Reload", use_container_width=True, key="adm_multi_commit")
                do_dry = colB.button("🧪 Dry-run (voir résumé)", use_container_width=True, key="adm_multi_dry")

                if do_dry or do_import:
                    merged = load_equipes(e_path)
                    rows_before = len(merged)

                    replaced: Dict[str, int] = {}
                    imported = 0
                    backed_up: set = set()

                    for p, chosen in assignments:
                        owner = str(chosen or "").strip()
                        if not owner:
                            continue

                        if owner not in backed_up:
                            backup_team_rows(merged, DATA_DIR, season_lbl, owner, note=f"pre-import {mode}")
                            backed_up.add(owner)

                        df_up = p["df"].copy()
                        df_up["Propriétaire"] = owner
                        if st.session_state.get("adm_nhl_enrich"):
                            df_up = enrich_df_from_nhl(df_up)
                        df_up_qc, stats = apply_quality(df_up, players_idx)

                        if mode.startswith("Remplacer"):
                            before_owner = int((merged["Propriétaire"].astype(str).str.strip() == owner).sum()) if not merged.empty else 0
                            merged = merged[~merged["Propriétaire"].astype(str).str.strip().eq(owner)].copy()
                            replaced[owner] = before_owner

                        merged = pd.concat([merged, df_up_qc], ignore_index=True)
                        imported += len(df_up_qc)

                        append_admin_log(
                            log_path,
                            action="IMPORT_LOCAL_TEAM",
                            owner=owner,
                            player="",
                            note=f"file={p['file']}; rows={len(df_up_qc)}; level_auto={stats.get('level_autofilled',0)}; mode={mode}",
                        )

                    merged, stats_all = apply_quality(merged, players_idx)
                    rows_after = len(merged)

                    summary = {
                        "mode": mode,
                        "fichiers_valides": len(parsed),
                        "lignes_avant": rows_before,
                        "lignes_importees": imported,
                        "lignes_apres": rows_after,
                        "teams_replaced": replaced,
                        "qc_level_auto": stats_all.get("level_autofilled", 0),
                        "qc_ir_mismatch": stats_all.get("ir_mismatch", 0),
                        "qc_salary_level_suspect": stats_all.get("salary_level_suspect", 0),
                    }
                    st.markdown("### Résumé import")
                    st.json(summary)

                    if do_import:
                        save_equipes(merged, e_path)
                        st.session_state["equipes_df"] = merged
                        st.success("✅ Import multi terminé + QC + reload.")
                        st.rerun()

        st.divider()
        if st.button("🧱 Créer un fichier équipes vide (squelette)", use_container_width=True, key="adm_local_create_empty"):
            df_empty = pd.DataFrame(columns=EQUIPES_COLUMNS)
            save_equipes(df_empty, e_path)
            st.session_state["equipes_df"] = df_empty
            append_admin_log(log_path, action="INIT_EMPTY", owner="", player="", note="created empty equipes file")
            st.success("✅ Fichier équipes vide créé.")
            st.rerun()


    with st.expander("🧼 Preview local + alertes", expanded=False):
        df = load_equipes(e_path)
        if df.empty:
            st.info("Aucun fichier équipes local. Importe depuis Drive ou import local.")
        else:
            df_qc, stats = apply_quality(df, players_idx)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lignes", stats["rows"])
            c2.metric("Level auto", stats["level_autofilled"])
            c3.metric("⚠️ IR mismatch", stats["ir_mismatch"])
            c4.metric("⚠️ Salaire/Level", stats["salary_level_suspect"])
            try:
                st.dataframe(df_qc.head(140).style.apply(_preview_style_row, axis=1), use_container_width=True)
            except Exception:
                st.dataframe(df_qc.head(140), use_container_width=True)

            if st.button("💾 Appliquer QC + sauvegarder + reload", use_container_width=True, key="adm_apply_qc"):
                save_equipes(df_qc, e_path)
                st.session_state["equipes_df"] = df_qc
                st.success("✅ QC appliqué + sauvegarde + reload.")
                st.rerun()

    # refresh after potential import
    df = load_equipes(e_path)
    owners = sorted([x for x in df.get("Propriétaire", pd.Series(dtype=str)).dropna().astype(str).str.strip().unique() if x])
    if not owners:
        st.warning("Aucune équipe (Propriétaire) détectée. Importe d'abord le CSV équipes.")
        return

    # =====================================================
    # ➕ ADD (ANTI-TRICHE)
    # =====================================================
    with st.expander("➕ Ajouter joueur(s) (anti-triche)", expanded=False):
        owner = st.selectbox("Équipe", owners, key="adm_add_owner")
        assign = st.radio("Assignation", ["GC - Actif", "GC - Banc", "CE - Actif", "CE - Banc"], horizontal=True, key="adm_add_assign")
        statut = "Grand Club" if assign.startswith("GC") else "Club École"
        slot = "Actif" if assign.endswith("Actif") else "Banc"

        allow_override = st.checkbox("🛑 Autoriser override admin si joueur appartient déjà à une autre équipe", value=False, key="adm_add_override")

        if players_idx:
            all_names = sorted({v["Joueur"] for v in players_idx.values() if v.get("Joueur")})
            selected = st.multiselect("Joueurs", all_names, key="adm_add_players")
        else:
            raw = st.text_area("Saisir joueurs (1 par ligne)", height=120, key="adm_add_manual")
            selected = [x.strip() for x in raw.splitlines() if x.strip()]

        preview: List[Dict[str, Any]] = []
        blocked: List[Tuple[str, str]] = []

        for p in selected:
            info = players_idx.get(_norm_player(p), {}) if players_idx else {}
            name = info.get("Joueur", p)
            cur_owner = find_player_owner(df, name)
            if cur_owner and cur_owner != owner and not allow_override:
                blocked.append((name, cur_owner))
                continue

            preview.append({
                "Propriétaire": owner,
                "Joueur": name,
                "Pos": info.get("Pos", ""),
                "Equipe": info.get("Equipe", ""),
                "Salaire": int(info.get("Salaire", 0) or 0),
                "Level": info.get("Level", "0"),
                "Statut": statut,
                "Slot": slot,
                "IR Date": "",
            })

        if blocked and not allow_override:
            st.error("⛔ Anti-triche: ces joueurs appartiennent déjà à une autre équipe")
            st.dataframe(pd.DataFrame(blocked, columns=["Joueur", "Équipe actuelle"]), use_container_width=True)

        if preview:
            st.dataframe(pd.DataFrame(preview).head(80), use_container_width=True)

        if st.button("✅ Ajouter maintenant", use_container_width=True, key="adm_add_commit"):
            if not preview:
                st.warning("Rien à ajouter.")
                st.stop()

            existing = set(zip(df["Propriétaire"].astype(str).str.strip(), df["Joueur"].astype(str).str.strip()))
            new_rows = []
            skipped_dupe = 0
            skipped_block = 0

            for r in preview:
                k = (str(r["Propriétaire"]).strip(), str(r["Joueur"]).strip())
                if k in existing:
                    skipped_dupe += 1
                    continue
                cur_owner = find_player_owner(df, r["Joueur"])
                if cur_owner and cur_owner != owner and not allow_override:
                    skipped_block += 1
                    continue
                new_rows.append(r)

            if not new_rows:
                st.warning(f"Rien à ajouter (doublons: {skipped_dupe}, bloqués: {skipped_block}).")
                st.stop()

            backup_team_rows(df, DATA_DIR, season_lbl, owner, note="pre-add")
            df2 = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df2_qc, stats = apply_quality(df2, players_idx)
            save_equipes(df2_qc, e_path)
            st.session_state["equipes_df"] = df2_qc

            for r in new_rows:
                append_admin_log(
                    log_path,
                    action="ADD",
                    owner=r["Propriétaire"],
                    player=r["Joueur"],
                    to_statut=r["Statut"],
                    to_slot=r["Slot"],
                    note=f"assign={assign}; override={allow_override}"
                )

            st.success(f"✅ Ajout OK: {len(new_rows)} | doublons: {skipped_dupe} | bloqués: {skipped_block} | Level auto: {stats.get('level_autofilled',0)}")
            st.rerun()

    # =====================================================
    # 🗑️ REMOVE
    # =====================================================
    with st.expander("🗑️ Retirer joueur(s) (avec confirmation)", expanded=False):
        owner = st.selectbox("Équipe", owners, key="adm_rem_owner")
        team = df[df["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()

        if team.empty:
            st.info("Aucun joueur pour cette équipe.")
        else:
            team["__label__"] = team.apply(lambda r: f"{r['Joueur']}  —  {r.get('Pos','')}  —  {r.get('Statut','')} / {r.get('Slot','')}", axis=1)
            choices = team["__label__"].tolist()
            sel = st.multiselect("Sélectionner joueur(s) à retirer", choices, key="adm_rem_sel")

            confirm = st.checkbox("Je confirme la suppression", key="adm_rem_confirm")

            if st.button("🗑️ Retirer maintenant", use_container_width=True, key="adm_rem_commit"):
                if not sel:
                    st.warning("Sélectionne au moins 1 joueur.")
                    st.stop()
                if not confirm:
                    st.warning("Coche la confirmation.")
                    st.stop()

                sel_rows = team[team["__label__"].isin(sel)].copy()
                if sel_rows.empty:
                    st.warning("Sélection invalide.")
                    st.stop()

                keys = set(zip(
                    sel_rows["Propriétaire"].astype(str),
                    sel_rows["Joueur"].astype(str),
                    sel_rows["Statut"].astype(str),
                    sel_rows["Slot"].astype(str),
                ))

                def _keep_row(r):
                    k = (str(r["Propriétaire"]), str(r["Joueur"]), str(r["Statut"]), str(r["Slot"]))
                    return k not in keys

                backup_team_rows(df, DATA_DIR, season_lbl, owner, note="pre-remove")
                before = len(df)
                df2 = df[df.apply(_keep_row, axis=1)].copy()
                removed = before - len(df2)

                df2_qc, _ = apply_quality(df2, players_idx)
                save_equipes(df2_qc, e_path)
                st.session_state["equipes_df"] = df2_qc

                for _, r in sel_rows.iterrows():
                    append_admin_log(
                        log_path,
                        action="REMOVE",
                        owner=r["Propriétaire"],
                        player=r["Joueur"],
                        from_statut=r.get("Statut", ""),
                        from_slot=r.get("Slot", ""),
                        note="removed by admin"
                    )

                st.success(f"✅ Retrait OK: {removed} joueur(s).")
                st.rerun()

    # =====================================================
    # 🔁 MOVE GC ↔ CE
    # =====================================================
    with st.expander("🔁 Déplacer GC ↔ CE (auto-slot)", expanded=False):
        owner = st.selectbox("Équipe", owners, key="adm_move_owner")
        team = df[df["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()

        if team.empty:
            st.info("Aucun joueur pour cette équipe.")
        else:
            team["__label__"] = team.apply(lambda r: f"{r['Joueur']}  —  {r.get('Pos','')}  —  {r.get('Statut','')} / {r.get('Slot','')}", axis=1)
            choices = team["__label__"].tolist()
            sel = st.multiselect("Sélectionner joueur(s) à déplacer", choices, key="adm_move_sel")

            dest_statut = st.radio("Destination", ["Grand Club", "Club École"], horizontal=True, key="adm_move_dest")
            slot_mode = st.radio("Slot destination", ["Auto (selon Statut)", "Garder Slot actuel", "Forcer Actif", "Forcer Banc"], horizontal=True, key="adm_move_slot_mode")
            keep_ir = st.checkbox("Conserver IR si joueur déjà IR", value=True, key="adm_move_keep_ir")

            if st.button("🔁 Appliquer déplacement", use_container_width=True, key="adm_move_commit"):
                if not sel:
                    st.warning("Sélectionne au moins 1 joueur.")
                    st.stop()

                sel_rows = team[team["__label__"].isin(sel)].copy()
                if sel_rows.empty:
                    st.warning("Sélection invalide.")
                    st.stop()

                keyset = set(zip(
                    sel_rows["Propriétaire"].astype(str),
                    sel_rows["Joueur"].astype(str),
                    sel_rows["Statut"].astype(str),
                    sel_rows["Slot"].astype(str),
                ))

                backup_team_rows(df, DATA_DIR, season_lbl, owner, note="pre-move")
                df2 = df.copy()
                moved = 0
                for idx, r in df2.iterrows():
                    k = (str(r["Propriétaire"]), str(r["Joueur"]), str(r["Statut"]), str(r["Slot"]))
                    if k not in keyset:
                        continue

                    from_statut = str(r["Statut"])
                    from_slot = str(r["Slot"])

                    df2.at[idx, "Statut"] = dest_statut

                    if slot_mode.startswith("Auto"):
                        df2.at[idx, "Slot"] = auto_slot_for_statut(dest_statut, current_slot=from_slot, keep_ir=keep_ir)
                    elif slot_mode.startswith("Garder"):
                        df2.at[idx, "Slot"] = from_slot
                    elif slot_mode.endswith("Actif"):
                        df2.at[idx, "Slot"] = "Actif"
                    elif slot_mode.endswith("Banc"):
                        df2.at[idx, "Slot"] = "Banc"
                    else:
                        df2.at[idx, "Slot"] = auto_slot_for_statut(dest_statut, current_slot=from_slot, keep_ir=keep_ir)

                    moved += 1

                    append_admin_log(
                        log_path,
                        action="MOVE",
                        owner=r["Propriétaire"],
                        player=r["Joueur"],
                        from_statut=from_statut,
                        from_slot=from_slot,
                        to_statut=dest_statut,
                        to_slot=str(df2.at[idx, "Slot"]),
                        note=f"slot_mode={slot_mode}"
                    )

                df2_qc, stats = apply_quality(df2, players_idx)
                save_equipes(df2_qc, e_path)
                st.session_state["equipes_df"] = df2_qc

                st.success(f"✅ Move OK: {moved} joueur(s) | Level auto: {stats.get('level_autofilled',0)}")
                st.rerun()

    # =====================================================
    # 📋 HISTORIQUE ADMIN
    # =====================================================
    # =====================================================
    # 🆔 BULK NHL_ID (AUTO) — par 250 (checkpoint)
    # =====================================================
    with st.expander("🆔 Bulk NHL_ID (AUTO) — par 250", expanded=False):
        render_bulk_nhl_id_admin(DATA_DIR, season_lbl, is_admin)

    with st.expander("📋 Historique admin (ADD/REMOVE/MOVE/IMPORT)", expanded=False):
        if not os.path.exists(log_path):
            st.info("Aucun historique pour l’instant.")
        else:
            try:
                lg = pd.read_csv(log_path).sort_values("timestamp", ascending=False)

                f1, f2, f3 = st.columns(3)
                with f1:
                    act = st.multiselect("Action", sorted(lg["action"].dropna().unique()), default=[], key="adm_log_act")
                with f2:
                    own = st.multiselect("Équipe", sorted(lg["owner"].dropna().unique()), default=[], key="adm_log_own")
                with f3:
                    q = st.text_input("Recherche joueur", value="", key="adm_log_q").strip().lower()

                view = lg.copy()
                if act:
                    view = view[view["action"].isin(act)]
                if own:
                    view = view[view["owner"].isin(own)]
                if q:
                    view = view[view["player"].astype(str).str.lower().str.contains(q, na=False)]

                st.dataframe(view.head(400), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur log: {e}")

    # =====================================================

    # 🧩 FUSION — players_master_{season}.csv (source unique)

    # =====================================================

    with st.expander("🧩 Fusion — players_master (source unique)", expanded=False):

        st.caption("Crée/Met à jour un fichier unique **players_master_{season}.csv**. Tous les onglets pourront ensuite lire ce fichier pour la disponibilité.")

        season_lbl_master = st.text_input("Saison (format 2025-2026)", value=season_lbl, key="adm_master_season")

        master_path = players_master_path(DATA_DIR, season_lbl_master)


        colx, coly, colz = st.columns([1,1,2])

        dry_run = colx.checkbox("🧪 Dry run (ne rien écrire)", value=True, key="adm_master_dry")

        use_equipes_fused = coly.checkbox("Utiliser equipes_joueurs_{season}.csv (si présent)", value=True, key="adm_master_use_equipes_fused")

        use_team_csv = colz.checkbox("Sinon, lire aussi les CSV équipes dans /data (Whalers.csv, etc.)", value=True, key="adm_master_use_team_csv")


        st.write("Fichier master ciblé:", f"`{master_path}`")


        if st.button("🧩 Construire players_master (batch 250)", type="primary", use_container_width=True, key="adm_master_build"):

            try:

                report = build_players_master(

                    data_dir=DATA_DIR,

                    season_lbl=season_lbl_master,

                    dry_run=dry_run,

                    prefer_equipes_fused=use_equipes_fused,

                    read_team_csvs=use_team_csv,

                    batch_size=250,

                )

                if report.get("ok"):

                    st.success("✅ Fusion terminée.")

                else:

                    st.warning("⚠️ Fusion terminée avec avertissements.")

                st.json(report, expanded=False)

                if report.get("preview_path"):

                    try:

                        dfp = pd.read_csv(report["preview_path"])

                        st.dataframe(dfp.head(200), use_container_width=True)

                    except Exception:

                        pass

            except Exception as e:

                st.error(f"Erreur fusion master: {e}")

    st.caption("✅ Admin: OAuth Drive / Import local • Add/Remove/Move • Caps bars • Log • QC/Level auto")

# ============================================================
# Paths — admin log + players db (hockey.players.csv)
# ============================================================
def admin_log_path(data_dir: str, season_lbl: str) -> str:
    """Chemin du log admin pour une saison."""
    season_lbl = str(season_lbl or "").strip() or "season"
    return os.path.join(str(data_dir), f"admin_log_{season_lbl}.csv")


def resolve_players_db_path(data_dir: str) -> str:
    """Résout le chemin du fichier 'hockey.players.csv' (Players DB)."""
    dd = str(data_dir or "").strip() or "data"
    candidates = [
        os.path.join(dd, "hockey.players.csv"),
        os.path.join(dd, "Hockey.Players.csv"),
        os.path.join(dd, "Hockey.Players.CSV"),
        os.path.join(dd, "players_db.csv"),
        os.path.join(dd, "players.csv"),
        # fallbacks relatifs (repo)
        os.path.join("data", "hockey.players.csv"),
        os.path.join("Data", "hockey.players.csv"),
        os.path.join("data", "Hockey.Players.csv"),
        os.path.join("Data", "Hockey.Players.csv"),
    ]
    for p in candidates:
        try:
            if p and os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        except Exception:
            continue
    # dernier recours: même si absent (pour afficher un message clair côté UI)
    return candidates[0]


# ============================================================
# Equipes (rosters) — lecture / écriture
# ============================================================
def load_equipes(path: str) -> "pd.DataFrame":
    """Charge le fichier equipes_joueurs_*.csv. Retourne DF vide si absent/illisible.
    On reste permissif (delimiter auto + lignes brisées) pour éviter les crashes.
    """
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    # lecture robuste
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                on_bad_lines="skip",
                encoding=enc,
            )
            if isinstance(df, pd.DataFrame):
                return df
        except Exception as e:
            last_err = e
            continue

    # dernier recours
    try:
        import streamlit as st
        st.warning(f"⚠️ Lecture équipes impossible: {last_err}")
    except Exception:
        pass
    return pd.DataFrame()


# ============================================================
# UI — Cap bars (GC/CE)
# ============================================================
def _render_caps_bars(df_eq: "pd.DataFrame", cap_gc: int, cap_ce: int) -> None:
    """Affiche des barres de cap (GC/CE) par équipe si possible.
    Fonction permissive: si colonnes manquantes, affiche un message simple sans planter.
    """
    import streamlit as st

    if df_eq is None or (hasattr(df_eq, "empty") and df_eq.empty):
        st.info("Aucun fichier d'équipes chargé (equipes_joueurs_*.csv). Les barres cap seront disponibles après import.")
        return

    # Colonnes possibles
    owner_col = None
    for c in ["Proprietaire", "Propriétaire", "Owner", "Equipe", "Équipe", "Team"]:
        if c in df_eq.columns:
            owner_col = c
            break

    scope_col = None
    for c in ["Scope", "Club", "Type", "Roster", "Categorie", "Catégorie"]:
        if c in df_eq.columns:
            scope_col = c
            break

    sal_col = None
    for c in ["Salaire", "Salary", "Cap Hit", "CapHit", "CapHitAAV", "AAV", "cap_hit"]:
        if c in df_eq.columns:
            sal_col = c
            break

    if owner_col is None or sal_col is None:
        st.warning("Colonnes insuffisantes pour les barres cap (besoin d'une colonne équipe + salaire).")
        st.caption(f"Colonnes détectées: {list(df_eq.columns)[:30]}")
        return

    def _to_num(x):
        try:
            if pd.isna(x):
                return 0.0
        except Exception:
            pass
        s = str(x).strip().replace("$","").replace(" ", "").replace(",", "")
        try:
            return float(s)
        except Exception:
            return 0.0

    d = df_eq.copy()

    # Normalise Scope
    if scope_col is None:
        d["_scope"] = "GC"
    else:
        def _norm_scope(v):
            s = str(v or "").strip().upper()
            if "CE" in s or "ECO" in s or "ÉCO" in s or "ECOLE" in s:
                return "CE"
            if "GC" in s or "GRAND" in s:
                return "GC"
            # fallback: si vide -> GC
            return "GC"
        d["_scope"] = d[scope_col].map(_norm_scope)

    d["_sal"] = d[sal_col].map(_to_num)

    # Totaux par équipe
    owners = [o for o in d[owner_col].dropna().astype(str).unique().tolist() if str(o).strip()]
    owners = sorted(owners)

    st.markdown("#### 💰 Barres de cap (GC / CE) par équipe")
    for o in owners:
        dd = d[d[owner_col].astype(str) == str(o)]
        tot_gc = float(dd.loc[dd["_scope"]=="GC", "_sal"].sum())
        tot_ce = float(dd.loc[dd["_scope"]=="CE", "_sal"].sum())

        # ratios
        rgc = 0.0 if not cap_gc else min(1.0, tot_gc / float(cap_gc))
        rce = 0.0 if not cap_ce else min(1.0, tot_ce / float(cap_ce))

        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"**{o} — GC**  ({int(tot_gc):,}$ / {int(cap_gc):,}$)".replace(",", " "))
            st.progress(rgc)
        with c2:
            st.caption(f"**{o} — CE**  ({int(tot_ce):,}$ / {int(cap_ce):,}$)".replace(",", " "))
            st.progress(rce)

def save_equipes(df: "pd.DataFrame", path: str) -> bool:
    """Sauvegarde le fichier equipes_joueurs_*.csv (UTF-8-SIG)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        (df if df is not None else pd.DataFrame()).to_csv(path, index=False, encoding="utf-8-sig")
        return True
    except Exception:
        return False

