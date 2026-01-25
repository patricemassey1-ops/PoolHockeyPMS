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

# ---- Optional: Google Drive client (if installed)
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
except Exception:
    build = None
    MediaIoBaseDownload = None
    Credentials = None

# ---- Optional: project-specific OAuth helpers (si ton projet les a déjà)
# On essaie plusieurs noms possibles, sans casser si absent.
_oauth_ui = None
_oauth_enabled = None
_oauth_get_service = None

for _mod, _fn_ui, _fn_enabled, _fn_service in [
    ("services.gdrive_oauth", "render_oauth_connect_ui", "oauth_drive_enabled", "get_drive_service"),
    ("services.gdrive_oauth", "render_oauth_ui", "oauth_drive_enabled", "drive_get_service"),
    ("services.drive_oauth", "render_oauth_connect_ui", "oauth_drive_enabled", "get_drive_service"),
    ("services.drive_oauth", "render_oauth_ui", "oauth_drive_enabled", "drive_get_service"),
]:
    try:
        m = __import__(_mod, fromlist=[_fn_ui, _fn_enabled, _fn_service])
        _oauth_ui = getattr(m, _fn_ui, None) or _oauth_ui
        _oauth_enabled = getattr(m, _fn_enabled, None) or _oauth_enabled
        _oauth_get_service = getattr(m, _fn_service, None) or _oauth_get_service
    except Exception:
        pass



# ---- Optional: OAuth Flow fallback (self-contained) — nécessite google-auth-oauthlib
try:
    from google_auth_oauthlib.flow import Flow  # type: ignore
except Exception:
    Flow = None


def render_drive_oauth_connect_ui() -> None:
    """
    UI OAuth Google Drive autonome (fallback).
    Utilise st.secrets["gdrive_oauth"] (client_id / client_secret / redirect_uri).
    Stocke les creds dans st.session_state["drive_creds"].
    """
    if Flow is None:
        st.info("OAuth Drive: google-auth-oauthlib indisponible (fallback désactivé).")
        return

    cfg = st.secrets.get("gdrive_oauth", {})
    client_id = str(cfg.get("client_id", "") or "")
    client_secret = str(cfg.get("client_secret", "") or "")
    redirect_uri = str(cfg.get("redirect_uri", "") or "")
    if not (client_id and client_secret and redirect_uri):
        st.info("OAuth Drive: Secrets [gdrive_oauth] incomplets (client_id / client_secret / redirect_uri).")
        return

    # already connected?
    if st.session_state.get("drive_creds"):
        st.success("✅ Drive OAuth connecté.")
        if st.button("🔌 Déconnecter Drive OAuth", use_container_width=True, key="adm_drive_disconnect"):
            st.session_state.pop("drive_creds", None)
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()
        return

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": [redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    flow = Flow.from_client_config(client_config, scopes=scopes, redirect_uri=redirect_uri)

    # handle return ?code=
    code_param = ""
    try:
        qp = st.query_params if hasattr(st, "query_params") else {}
        c = qp.get("code", "")
        if isinstance(c, list):
            code_param = c[0] if c else ""
        else:
            code_param = str(c or "")
    except Exception:
        code_param = ""

    if code_param:
        try:
            flow.fetch_token(code=code_param)
            creds = flow.credentials
            st.session_state["drive_creds"] = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.success("✅ Drive OAuth connecté.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Échec OAuth: {e}")
            return

    auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
    st.link_button("🔐 Connecter Google Drive (OAuth)", auth_url, use_container_width=True)

# ============================================================
# CONFIG
# ============================================================
PLAYERS_DB_FILENAME = "hockey.players.csv"
EQUIPES_COLUMNS = [
    "Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"
]
DEFAULT_CAP_GC = 88_000_000
DEFAULT_CAP_CE = 12_000_000


# ============================================================
# SETTINGS (CAPS) — local + Drive
# ============================================================
SETTINGS_FILENAME = "settings.csv"

def _fmt_money(n: int) -> str:
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    return f"{n:,.0f}".replace(",", " ") + " $"

def _parse_money(s: str) -> int:
    s = str(s or "")
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        return int(digits) if digits else 0
    except Exception:
        return 0

def settings_path(data_dir: str) -> str:
    return os.path.join(str(data_dir), SETTINGS_FILENAME)

def _settings_df(cap_gc: int, cap_ce: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "key": "caps",
        "cap_gc": int(cap_gc or 0),
        "cap_ce": int(cap_ce or 0),
        "updated_ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }])

def load_settings_local(data_dir: str) -> dict:
    path = settings_path(data_dir)
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        return {
            "cap_gc": _safe_int(row.get("cap_gc"), 0),
            "cap_ce": _safe_int(row.get("cap_ce"), 0),
            "updated_ts": str(row.get("updated_ts") or ""),
        }
    except Exception:
        return {}

def save_settings_local(data_dir: str, cap_gc: int, cap_ce: int) -> bool:
    try:
        os.makedirs(data_dir, exist_ok=True)
        path = settings_path(data_dir)
        _settings_df(cap_gc, cap_ce).to_csv(path, index=False, encoding="utf-8")
        return True
    except Exception:
        return False

def _drive_find_file_id_by_name(svc: Any, folder_id: str, name: str) -> str:
    try:
        q = f"'{folder_id}' in parents and name = '{name}' and trashed=false"
        res = svc.files().list(
            q=q,
            fields="files(id,name,modifiedTime,createdTime)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=10,
        ).execute()
        files = (res or {}).get("files") or []
        return str(files[0].get("id") or "") if files else ""
    except Exception:
        return ""

def _drive_upload_bytes(svc: Any, folder_id: str, name: str, data: bytes, mime: str = "text/csv") -> str:
    """Create or update a file in Drive. Returns file_id or ''.
    Requires OAuth scopes that allow write. If not, it will fail silently (caller handles).
    """
    try:
        from googleapiclient.http import MediaIoBaseUpload  # type: ignore
    except Exception:
        MediaIoBaseUpload = None

    if not MediaIoBaseUpload:
        return ""

    file_id = _drive_find_file_id_by_name(svc, folder_id, name)
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)

    try:
        if file_id:
            svc.files().update(
                fileId=file_id,
                media_body=media,
                supportsAllDrives=True,
            ).execute()
            return file_id
        meta = {"name": name, "parents": [folder_id]}
        created = svc.files().create(
            body=meta,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return str((created or {}).get("id") or "")
    except Exception:
        return ""

def load_settings_drive_if_needed(data_dir: str, svc: Any, folder_id: str) -> dict:
    """If local settings missing, try download from Drive and cache locally."""
    try:
        local = load_settings_local(data_dir)
        if local:
            return local
        if not svc or not folder_id:
            return {}
        file_id = _drive_find_file_id_by_name(svc, folder_id, SETTINGS_FILENAME)
        if not file_id:
            return {}
        b = _drive_download_bytes(svc, file_id)
        if b:
            try:
                Path(settings_path(data_dir)).write_bytes(b)
            except Exception:
                pass
        return load_settings_local(data_dir)
    except Exception:
        return {}


# ============================================================
# UTILS
# ============================================================
def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _norm(x: Any) -> str:
    return str(x or "").strip()

def _norm_player(x: Any) -> str:
    return _norm(x).lower()

def _norm_level(v: Any) -> str:
    s = _norm(v).upper()
    return s if s in {"ELC", "STD"} else "0"

def _safe_int(v: Any, default: int = 0) -> int:
    try:
        n = pd.to_numeric(v, errors="coerce")
        if pd.isna(n):
            return default
        return int(n)
    except Exception:
        return default

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    if df is None or df.empty:
        return ""
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in low:
            return low[key]
    return ""



# ============================================================
# OWNER FROM FILENAME (🧠)
# ============================================================
def infer_owner_from_filename(filename: str, owners_choices: List[str]) -> str:
    """
    Auto-assign depuis nom de fichier.
    Ex: "Whalers.csv" -> "Whalers"
    Priorité: match d'un owner existant (owners_choices), sinon basename.
    """
    fn = str(filename or "").strip()
    if not fn:
        return ""
    base = os.path.splitext(os.path.basename(fn))[0].strip()
    low = fn.lower()
    base_low = base.lower()

    for o in owners_choices or []:
        ol = str(o or "").strip().lower()
        if not ol:
            continue
        if ol in low or ol in base_low:
            return str(o).strip()

    # token match
    tokens = re.split(r"[^a-z0-9]+", base_low)
    token_set = {t for t in tokens if t}
    for o in owners_choices or []:
        ol = str(o or "").strip().lower()
        if ol and ol in token_set:
            return str(o).strip()

    return base or ""


# ============================================================
# ROLLBACK (local) — par équipe
# ============================================================
def _backup_dir(data_dir: str, season_lbl: str) -> str:
    d = os.path.join(str(data_dir or "Data"), "backups_admin", str(season_lbl or "season"))
    os.makedirs(d, exist_ok=True)
    return d

def list_team_backups(data_dir: str, season_lbl: str, owner: str) -> List[str]:
    d = _backup_dir(data_dir, season_lbl)
    owner_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(owner or "").strip())
    files: List[str] = []
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.lower().endswith(".csv") and owner_key.lower() in fn.lower():
                files.append(os.path.join(d, fn))
    files.sort(reverse=True)
    return files

def backup_team_rows(df_all: pd.DataFrame, data_dir: str, season_lbl: str, owner: str, note: str = "") -> Optional[str]:
    if df_all is None or df_all.empty or not owner:
        return None
    d = _backup_dir(data_dir, season_lbl)
    owner_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(owner or "").strip())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(d, f"equipes_{season_lbl}__{owner_key}__{ts}.csv")
    sub = df_all[df_all["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()
    if sub.empty:
        return None
    sub = ensure_equipes_df(sub)
    sub.to_csv(path, index=False)
    if note:
        st.session_state[f"admin_last_backup_note__{season_lbl}__{owner_key}"] = note
    return path

def restore_team_from_backup(df_all: pd.DataFrame, backup_path: str, owner: str) -> pd.DataFrame:
    df_all = ensure_equipes_df(df_all)
    if not backup_path or not os.path.exists(backup_path) or not owner:
        return df_all
    try:
        sub = pd.read_csv(backup_path)
        sub = ensure_equipes_df(sub)
        sub["Propriétaire"] = str(owner).strip()
    except Exception:
        return df_all

    df_all = df_all[~df_all["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()
    df_all = pd.concat([df_all, sub], ignore_index=True)
    return ensure_equipes_df(df_all)

# ============================================================
# PATHS
# ============================================================
def equipes_path(data_dir: str, season_lbl: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season_lbl}.csv")

def admin_log_path(data_dir: str, season_lbl: str) -> str:
    return os.path.join(data_dir, f"admin_actions_{season_lbl}.csv")


# ============================================================
# LOADERS
# ============================================================
def ensure_equipes_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=EQUIPES_COLUMNS)
    out = df.copy()
    for c in EQUIPES_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    for c in ["Propriétaire", "Joueur", "Pos", "Equipe", "Level", "Statut", "Slot", "IR Date"]:
        out[c] = out[c].astype(str).fillna("").str.strip()
    out["Salaire"] = pd.to_numeric(out.get("Salaire", 0), errors="coerce").fillna(0).astype(int)
    out["Level"] = out["Level"].apply(_norm_level)
    return out[EQUIPES_COLUMNS + [c for c in out.columns if c not in EQUIPES_COLUMNS]]

def load_equipes(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=EQUIPES_COLUMNS)
    try:
        return ensure_equipes_df(pd.read_csv(path))
    except Exception:
        return pd.DataFrame(columns=EQUIPES_COLUMNS)

def save_equipes(df: pd.DataFrame, path: str) -> None:
    ensure_equipes_df(df).to_csv(path, index=False)

@st.cache_data(show_spinner=False)
def load_players_db(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, dtype=str, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

def build_players_index(players: pd.DataFrame) -> dict:
    if players is None or players.empty:
        return {}
    name_c = _pick_col(players, ["Joueur", "Player", "Name"])
    if not name_c:
        return {}
    pos_c  = _pick_col(players, ["Pos", "Position"])
    team_c = _pick_col(players, ["Equipe", "Équipe", "Team"])
    sal_c  = _pick_col(players, ["Salaire", "Cap Hit", "CapHit", "Cap", "Cap_Hit"])
    lvl_c  = _pick_col(players, ["Level"])

    idx: Dict[str, Dict[str, Any]] = {}
    for _, r in players.iterrows():
        name = _norm(r.get(name_c, ""))
        if not name:
            continue
        idx[_norm_player(name)] = {
            "Joueur": name,
            "Pos": _norm(r.get(pos_c, "")) if pos_c else "",
            "Equipe": _norm(r.get(team_c, "")) if team_c else "",
            "Salaire": _safe_int(r.get(sal_c, 0)) if sal_c else 0,
            "Level": _norm_level(r.get(lvl_c, "0")) if lvl_c else "0",
        }
    return idx


# ============================================================
# VALIDATION
# ============================================================
def validate_equipes_df(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    cols = list(df.columns)
    missing = [c for c in EQUIPES_COLUMNS if c not in cols]
    extras = [c for c in cols if c not in EQUIPES_COLUMNS]
    return (len(missing) == 0), missing, extras


# ============================================================
# QUALITY CHECKS + ANTI-CHEAT
# ============================================================
def _infer_level_from_salary(sal: int) -> str:
    return "ELC" if int(sal) <= 1_000_000 else "STD"

def find_player_owner(df: pd.DataFrame, player: str) -> Optional[str]:
    if df is None or df.empty or not player:
        return None
    k = _norm_player(player)
    m = df["Joueur"].astype(str).map(_norm_player).eq(k)
    if not m.any():
        return None
    return _norm(df.loc[m, "Propriétaire"].iloc[0])

def apply_quality(df: pd.DataFrame, players_idx: dict) -> Tuple[pd.DataFrame, Dict[str, int]]:
    out = ensure_equipes_df(df)
    filled = 0

    need = out["Level"].astype(str).str.strip().isin({"0", ""})
    if need.any():
        for i in out[need].index:
            key = _norm_player(out.at[i, "Joueur"])
            mapped = ""
            if key in players_idx:
                mapped = str(players_idx[key].get("Level", "")).strip().upper()
            if mapped in {"ELC", "STD"}:
                out.at[i, "Level"] = mapped
            else:
                out.at[i, "Level"] = _infer_level_from_salary(int(out.at[i, "Salaire"]))
            filled += 1

    out["⚠️ IR mismatch"] = (
        out["IR Date"].astype(str).str.strip().ne("") &
        out["IR Date"].astype(str).str.lower().ne("nan") &
        out["Slot"].astype(str).str.upper().ne("IR")
    )

    out["⚠️ Salary/Level suspect"] = (
        ((out["Level"] == "ELC") & (out["Salaire"] > 1_500_000)) |
        ((out["Level"] == "STD") & (out["Salaire"] <= 0))
    )

    stats = {
        "rows": int(len(out)),
        "level_autofilled": int(filled),
        "ir_mismatch": int(out["⚠️ IR mismatch"].sum()),
        "salary_level_suspect": int(out["⚠️ Salary/Level suspect"].sum()),
    }
    return out, stats

def _preview_style_row(row: pd.Series) -> List[str]:
    ir_mis = bool(row.get("⚠️ IR mismatch", False))
    sus = bool(row.get("⚠️ Salary/Level suspect", False))
    slot = str(row.get("Slot", "")).strip().upper()
    statut = str(row.get("Statut", "")).strip().lower()

    if ir_mis:
        return ["background-color: rgba(255, 0, 0, 0.18)"] * len(row)
    if sus:
        return ["background-color: rgba(255, 165, 0, 0.16)"] * len(row)
    if slot == "IR" or "ir" in statut:
        return ["background-color: rgba(160, 120, 255, 0.10)"] * len(row)
    if slot in {"MINEUR", "MIN", "AHL"} or "mineur" in statut:
        return ["background-color: rgba(120, 200, 255, 0.10)"] * len(row)
    return [""] * len(row)


# ============================================================
# AUTO SLOT
# ============================================================
def auto_slot_for_statut(dest_statut: str, *, current_slot: str = "", keep_ir: bool = True) -> str:
    cur = str(current_slot or "").strip().upper()
    if keep_ir and cur == "IR":
        return "IR"
    return "Actif"


# ============================================================
# CAPS
# ============================================================
def compute_caps(df: pd.DataFrame) -> pd.DataFrame:
    d = ensure_equipes_df(df)
    d["Salaire"] = pd.to_numeric(d["Salaire"], errors="coerce").fillna(0).astype(int)

    def _is_gc(x: str) -> bool:
        return str(x or "").strip().lower() == "grand club"

    def _is_ce(x: str) -> bool:
        s = str(x or "").strip().lower()
        return s in {"club école", "club ecole"}

    d["is_gc"] = d["Statut"].apply(_is_gc)
    d["is_ce"] = d["Statut"].apply(_is_ce)

    g = d.groupby("Propriétaire", dropna=False)
    out = pd.DataFrame({
        "GC $": g.apply(lambda x: int(x.loc[x["is_gc"], "Salaire"].sum())),
        "CE $": g.apply(lambda x: int(x.loc[x["is_ce"], "Salaire"].sum())),
        "Total $": g["Salaire"].sum().astype(int),
        "Nb joueurs": g.size().astype(int),
        "Nb GC": g.apply(lambda x: int(x["is_gc"].sum())),
        "Nb CE": g.apply(lambda x: int(x["is_ce"].sum())),
    }).reset_index()
    out["Propriétaire"] = out["Propriétaire"].astype(str).str.strip()
    return out.sort_values("Propriétaire")

def _render_caps_bars(df_eq: pd.DataFrame, cap_gc: int, cap_ce: int) -> None:
    caps = compute_caps(df_eq)
    if caps.empty:
        st.info("Aucune donnée équipes.")
        return

    for _, r in caps.iterrows():
        owner = str(r.get("Propriétaire", "")).strip()
        gc = int(r.get("GC $", 0))
        ce = int(r.get("CE $", 0))

        st.markdown(f"**{owner}**")
        c1, c2, c3 = st.columns([2, 2, 1])

        with c1:
            ratio = 0.0 if cap_gc <= 0 else min(1.0, gc / cap_gc)
            st.caption(f"GC: {gc:,} / {cap_gc:,}")
            st.progress(ratio)

        with c2:
            ratio = 0.0 if cap_ce <= 0 else min(1.0, ce / cap_ce)
            st.caption(f"CE: {ce:,} / {cap_ce:,}")
            st.progress(ratio)

        with c3:
            over = []
            if cap_gc > 0 and gc > cap_gc:
                over.append(f"⚠️ GC +{gc-cap_gc:,}")
            if cap_ce > 0 and ce > cap_ce:
                over.append(f"⚠️ CE +{ce-cap_ce:,}")
            st.write("\n".join(over) if over else "✅ OK")

        st.divider()


# ============================================================
# ADMIN LOG
# ============================================================
def append_admin_log(
    path: str,
    *,
    action: str,
    owner: str,
    player: str,
    from_statut: str = "",
    from_slot: str = "",
    to_statut: str = "",
    to_slot: str = "",
    note: str = "",
) -> None:
    row = {
        "timestamp": _now_ts(),
        "action": action,
        "owner": _norm(owner),
        "player": _norm(player),
        "from_statut": _norm(from_statut),
        "from_slot": _norm(from_slot),
        "to_statut": _norm(to_statut),
        "to_slot": _norm(to_slot),
        "note": _norm(note),
    }
    df = pd.DataFrame([row])
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, df], ignore_index=True)
            out.to_csv(path, index=False)
            return
        except Exception:
            pass
    df.to_csv(path, index=False)


# ============================================================
# DRIVE (OAuth) — get service
# ============================================================
def _drive_service_from_existing_oauth() -> Optional[Any]:
    """
    Essaie d'obtenir un service Drive "comme avant" :
    1) si ton projet a déjà un helper (services.*drive_oauth*), on l'utilise
    2) sinon, on tente st.session_state['drive_creds'] (dict OAuth) -> google.oauth2.credentials
    """
    # 1) helper projet
    if callable(_oauth_get_service):
        try:
            return _oauth_get_service()
        except TypeError:
            try:
                return _oauth_get_service(st.session_state)
            except Exception:
                pass
        except Exception:
            pass

    # 2) session_state creds dict
    if build is None or Credentials is None:
        return None
    creds_dict = st.session_state.get("drive_creds")
    if not creds_dict:
        return None
    try:
        creds = Credentials.from_authorized_user_info(creds_dict)
        return build("drive", "v3", credentials=creds)
    except Exception:
        return None

def _drive_list_csv_files(svc: Any, folder_id: str) -> List[Dict[str, str]]:
    if not svc or not folder_id:
        return []
    q = f"'{folder_id}' in parents and trashed=false and mimeType='text/csv'"
    res = svc.files().list(
        q=q,
        fields="files(id,name,modifiedTime)",
        pageSize=200,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", []) or []
    files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    return [{"id": f["id"], "name": f["name"]} for f in files if f.get("id") and f.get("name")]

def _drive_download_bytes(svc: Any, file_id: str) -> bytes:
    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

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
# PATH HELPERS (safe)
# =====================================================
def _first_existing_path(candidates: List[str]) -> str:
    """Retourne le premier chemin existant dans candidates, sinon '' (safe)."""
    for p in candidates or []:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return ""

def render(ctx: dict) -> None:
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    DATA_DIR = str(ctx.get("DATA_DIR") or "Data")
    # Resolve /data vs /Data robustly
    try:
        root = Path(__file__).resolve().parents[1]
        if not os.path.isabs(DATA_DIR):
            # if ctx gives relative, anchor to root
            DATA_DIR = str((root / DATA_DIR).resolve())
        if not os.path.isdir(DATA_DIR):
            for name in ("data","Data","DATA"):
                p = root / name
                if p.exists() and p.is_dir():
                    DATA_DIR = str(p.resolve())
                    break
    except Exception:
        pass
    os.makedirs(DATA_DIR, exist_ok=True)

    season_lbl = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    folder_id = str(ctx.get("drive_folder_id") or "").strip()

    e_path = equipes_path(DATA_DIR, season_lbl)
    log_path = admin_log_path(DATA_DIR, season_lbl)

    st.subheader("🛠️ Gestion Admin")

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

    with st.expander("💰 Plafonds salariaux (GC / CE)", expanded=False):
        st.caption("Définis ici les plafonds utilisés partout (affichage + alertes). Format: `1 000 000 $`.")
        # ---- load settings at first display (Drive -> local cache -> session)
        svc_for_settings = _drive_service_from_existing_oauth()
        settings = load_settings_drive_if_needed(DATA_DIR, svc_for_settings, folder_id) if folder_id else load_settings_local(DATA_DIR)

        if "CAP_GC" not in st.session_state:
            st.session_state["CAP_GC"] = int(settings.get("cap_gc") or DEFAULT_CAP_GC)
        if "CAP_CE" not in st.session_state:
            st.session_state["CAP_CE"] = int(settings.get("cap_ce") or DEFAULT_CAP_CE)

        col1, col2 = st.columns(2)
        with col1:
            cap_gc_str = st.text_input("Cap GC", value=_fmt_money(st.session_state.get("CAP_GC", DEFAULT_CAP_GC)), key="cap_gc_txt")
        with col2:
            cap_ce_str = st.text_input("Cap CE", value=_fmt_money(st.session_state.get("CAP_CE", DEFAULT_CAP_CE)), key="cap_ce_txt")

        # validation anti-absurde
        cap_gc_val = _parse_money(cap_gc_str)
        cap_ce_val = _parse_money(cap_ce_str)

        MIN_CAP = 1_000_000
        MAX_CAP = 200_000_000
        errs = []
        if cap_gc_val and (cap_gc_val < MIN_CAP or cap_gc_val > MAX_CAP):
            errs.append(f"Cap GC invalide ({_fmt_money(cap_gc_val)}) — doit être entre {_fmt_money(MIN_CAP)} et {_fmt_money(MAX_CAP)}.")
        if cap_ce_val and (cap_ce_val < MIN_CAP or cap_ce_val > MAX_CAP):
            errs.append(f"Cap CE invalide ({_fmt_money(cap_ce_val)}) — doit être entre {_fmt_money(MIN_CAP)} et {_fmt_money(MAX_CAP)}.")

        if errs:
            for e in errs:
                st.error(e)
            st.info("Corrige la valeur avant de sauvegarder.")
        else:
            st.session_state["CAP_GC"] = int(cap_gc_val or 0)
            st.session_state["CAP_CE"] = int(cap_ce_val or 0)

        b1, b2, b3 = st.columns([1,1,2])
        with b1:
            if st.button("💾 Sauvegarder (local + Drive)", use_container_width=True, key="adm_caps_save"):
                if errs:
                    st.warning("Impossible de sauvegarder — valeurs invalides.")
                else:
                    ok_local = save_settings_local(DATA_DIR, st.session_state["CAP_GC"], st.session_state["CAP_CE"])
                    ok_drive = False
                    if svc_for_settings and folder_id:
                        try:
                            payload = Path(settings_path(DATA_DIR)).read_bytes()
                            ok_drive = bool(_drive_upload_bytes(svc_for_settings, folder_id, SETTINGS_FILENAME, payload))
                        except Exception:
                            ok_drive = False
                    if ok_local:
                        st.success("✅ Settings sauvegardés en local.")
                    else:
                        st.error("❌ Échec sauvegarde locale.")
                    if folder_id:
                        if ok_drive:
                            st.success("✅ Settings uploadés sur Drive.")
                        else:
                            st.warning("⚠️ Drive: upload non effectué (OAuth write scopes manquants ou Drive indisponible).")
                    st.cache_data.clear()
        with b2:
            if st.button("🔄 Recharger (local/Drive)", use_container_width=True, key="adm_caps_reload"):
                st.cache_data.clear()
                settings = load_settings_drive_if_needed(DATA_DIR, svc_for_settings, folder_id) if folder_id else load_settings_local(DATA_DIR)
                st.session_state["CAP_GC"] = int(settings.get("cap_gc") or DEFAULT_CAP_GC)
                st.session_state["CAP_CE"] = int(settings.get("cap_ce") or DEFAULT_CAP_CE)
                st.success("✅ Rechargé.")
        with b3:
            st.caption(f"Actuel: **GC** {_fmt_money(int(st.session_state.get('CAP_GC',0)))} • **CE** {_fmt_money(int(st.session_state.get('CAP_CE',0)))}")
# ---- Players DB index
    players_db = load_players_db(_first_existing_path([
        os.path.join(DATA_DIR, "hockey.players.csv"),
        os.path.join(DATA_DIR, "Hockey.Players.csv"),
        os.path.join(DATA_DIR, "hockey.players.CSV"),
        os.path.join(DATA_DIR, "Hockey.Players.CSV"),
        os.path.join(DATA_DIR, PLAYERS_DB_FILENAME),
    ]))
    players_idx = build_players_index(players_db)
    if players_idx:
        st.success(f"Players DB détectée: {PLAYERS_DB_FILENAME} (Level auto + infos).")
    else:
        st.warning(f"{PLAYERS_DB_FILENAME} indisponible — fallback Level par Salaire.")

    # ---- Load équipes
    df = load_equipes(e_path)

    # =====================================================
    # 🔄 IMPORT ÉQUIPES (Drive)
    # =====================================================
    with st.expander("🔄 Import équipes depuis Drive (OAuth)", expanded=False):
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
    with st.expander("📥 Import local (fallback) — multi-upload CSV équipes", expanded=False):
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

    # Si le fichier équipes est vide / pas encore importé, on permet quand même l'admin tooling.
    DEFAULT_OWNERS = ["Canadiens","Cracheurs","Nordiques","Predateurs","Red_Wings","Whalers"]
    if not owners:
        owners = DEFAULT_OWNERS
        st.info("Aucune équipe détectée dans le fichier local pour l’instant — fonctions admin disponibles quand même (anti-duplication sur la base des lignes existantes).")

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
            all_names = sorted({v.get("Joueur","") for v in players_idx.values() if v.get("Joueur")})
            q = st.text_input("🔎 Recherche joueur (fuzzy)", value="", key="adm_add_search").strip()
            suggestions = all_names
            if q:
                # fuzzy léger sans dépendances
                import difflib
                suggestions = difflib.get_close_matches(q, all_names, n=25, cutoff=0.4)
                if not suggestions:
                    # fallback: contient
                    ql = q.lower()
                    suggestions = [n for n in all_names if ql in n.lower()][:25]

            st.caption("Sélection max: 5 joueurs.")
            selected = st.multiselect("Joueurs (suggestions)", suggestions, default=[], key="adm_add_players")
        else:
            raw = st.text_area("Saisir joueurs (1 par ligne)", height=120, key="adm_add_manual")
            selected = [x.strip() for x in raw.splitlines() if x.strip()]

        if len(selected) > 5:
            st.warning("Sélection limitée à 5 joueurs (les premiers ont été gardés).")
            selected = selected[:5]

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

    st.caption("✅ Admin: OAuth Drive / Import local • Add/Remove/Move • Caps bars • Log • QC/Level auto")
    