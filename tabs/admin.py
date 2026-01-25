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
import importlib.util
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
    except Excep

# ---- Fallback: si "services" n'est pas un package (pas de __init__.py), charger par PATH
def _load_module_from_path(mod_name: str, path: Path):
    try:
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
    except Exception:
        return None
    return None

def _try_scan_services_for_oauth():
    global _oauth_ui, _oauth_enabled, _oauth_get_service
    try:
        base = Path(__file__).resolve().parents[1]
        services_dir = base / "services"
        if not services_dir.exists():
            return
        # priorité aux fichiers qui ressemblent à du Drive OAuth
        candidates = []
        for p in services_dir.glob("*.py"):
            name = p.name.lower()
            if "oauth" in name and ("drive" in name or "gdrive" in name):
                candidates.append(p)
        # fallback: tout fichier contenant "oauth"
        if not candidates:
            candidates = [p for p in services_dir.glob("*.py") if "oauth" in p.name.lower()]

        fn_ui_names = ["render_oauth_connect_ui", "render_oauth_ui", "oauth_ui", "render_drive_oauth_connect_ui"]
        fn_enabled_names = ["oauth_drive_enabled", "drive_oauth_enabled", "is_drive_oauth_enabled"]
        fn_service_names = ["get_drive_service", "drive_get_service", "get_gdrive_service"]

        for p in candidates:
            mod = _load_module_from_path(f"services__{p.stem}", p)
            if not mod:
                continue
            for n in fn_ui_names:
                if callable(getattr(mod, n, None)):
                    _oauth_ui = getattr(mod, n) or _oauth_ui
                    break
            for n in fn_enabled_names:
                if callable(getattr(mod, n, None)):
                    _oauth_enabled = getattr(mod, n) or _oauth_enabled
                    break
            for n in fn_service_names:
                if callable(getattr(mod, n, None)):
                    _oauth_get_service = getattr(mod, n) or _oauth_get_service
                    break
            if _oauth_get_service and _oauth_ui:
                break
    except Exception:
        return

# Lance le scan seulement si l'import package a échoué
if _oauth_ui is None and _oauth_get_service is None:
    _try_scan_services_for_oauth()

# Si aucun UI projet trouvé, on utilise l'UI fallback intégrée
if _oauth_ui is None:
    _oauth_ui = render_drive_oauth_connect_ui

tion:
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
    scopes = ["https://www.googleapis.com/auth/drive"]
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


# =====================================================
# 📁 DATA_DIR — robuste (Linux case-sensitive)
# =====================================================
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_ROOT_DIR = _THIS_FILE.parents[1]  # .../poolhockeypms

def _resolve_data_dir(ctx: dict) -> str:
    """
    Resolve the /data directory robustly (data vs Data) and allow ctx override.
    Never uses ctx at import-time.
    """
    cands = []
    try:
        if isinstance(ctx, dict) and ctx.get("DATA_DIR"):
            cands.append(Path(str(ctx.get("DATA_DIR"))))
    except Exception:
        pass

    cands += [
        _ROOT_DIR / "data",
        _ROOT_DIR / "Data",
        _ROOT_DIR / "DATA",
        Path.cwd() / "data",
        Path.cwd() / "Data",
        Path.cwd() / "DATA",
    ]
    for d in cands:
        try:
            if d and d.exists() and d.is_dir():
                return str(d.resolve())
        except Exception:
            continue
    return str((_ROOT_DIR / "data").resolve())

def _first_existing_path(paths):
    for p in paths or []:
        try:
            if p and os.path.exists(str(p)):
                return str(p)
        except Exception:
            continue
    return ""

def _fmt_money(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        n = 0
    return f"{n:,}".replace(",", " ") + " $"

def _parse_money(s: str) -> int:
    s = str(s or "")
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        return int(digits) if digits else 0
    except Exception:
        return 0

def _default_owners():
    return ["Canadiens","Cracheurs","Nordiques","Predateurs","Red_Wings","Whalers"]

def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _safe_write_csv(df: pd.DataFrame, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        return True
    except Exception:
        return False

def _try_import_drive_helpers():
    """
    Returns dict of callables from services.drive/auth if present.
    All optional.
    """
    out = {
        "oauth_enabled": None,
        "render_oauth_ui": None,
        "get_drive_service": None,
        "download_file_by_name": None,
        "upload_bytes": None,
    }
    try:
        from services import drive as _drive
        out["oauth_enabled"] = getattr(_drive, "oauth_drive_enabled", None)
        out["render_oauth_ui"] = getattr(_drive, "render_drive_oauth_connect_ui", None) or getattr(_drive, "render_oauth_connect_ui", None)
        out["get_drive_service"] = getattr(_drive, "get_drive_service", None) or getattr(_drive, "get_drive_service_from_session", None)
        # optional helpers
        out["download_file_by_name"] = getattr(_drive, "drive_download_file_by_name", None) or getattr(_drive, "download_file_by_name", None)
        out["upload_bytes"] = getattr(_drive, "drive_upload_bytes", None) or getattr(_drive, "upload_bytes", None)
    except Exception:
        pass
    return out

def _drive_download_settings(svc, folder_id: str, filename: str) -> bytes | None:
    """
    Best-effort download: use services.drive helper if available, else None.
    """
    try:
        helpers = _try_import_drive_helpers()
        fn = helpers.get("download_file_by_name")
        if callable(fn):
            return fn(svc, folder_id, filename)
    except Exception:
        pass
    return None

def _drive_upload_settings(svc, folder_id: str, filename: str, content: bytes) -> bool:
    try:
        helpers = _try_import_drive_helpers()
        fn = helpers.get("upload_bytes")
        if callable(fn):
            return bool(fn(svc, folder_id, filename, content))
    except Exception:
        pass
    return False

def _settings_local_path(data_dir: str) -> str:
    return os.path.join(data_dir, "settings.csv")

def _load_settings_from_bytes(b: bytes) -> dict:
    try:
        import io
        df = pd.read_csv(io.BytesIO(b))
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        return {k: row.get(k) for k in row.keys()}
    except Exception:
        return {}

def _settings_to_csv_bytes(cap_gc: int, cap_ce: int) -> bytes:
    import io
    df = pd.DataFrame([{"CAP_GC": int(cap_gc), "CAP_CE": int(cap_ce)}])
    bio = io.BytesIO()
    df.to_csv(bio, index=False, encoding="utf-8")
    return bio.getvalue()

def _fuzzy_candidates(query: str, choices: list[str], limit: int = 20) -> list[str]:
    q = (query or "").strip().lower()
    if not q:
        return choices[:limit]
    # simple fuzzy: substring first, then difflib
    sub = [c for c in choices if q in c.lower()]
    if sub:
        return sub[:limit]
    try:
        import difflib
        return difflib.get_close_matches(query, choices, n=limit, cutoff=0.2)
    except Exception:
        return choices[:limit]

# =====================================================
# 🛠️ RENDER
# =====================================================
def render(ctx: dict) -> None:
    # ---- Guard
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    data_dir = _resolve_data_dir(ctx)
    os.makedirs(data_dir, exist_ok=True)

    season_lbl = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    folder_id = str(ctx.get("drive_folder_id") or "").strip()

    # Paths
    equipes_csv = os.path.join(data_dir, f"equipes_joueurs_{season_lbl}.csv")
    if os.path.exists(os.path.join(data_dir, "equipes_joueurs_2025-2026.csv")) and season_lbl == "2025-2026":
        equipes_csv = os.path.join(data_dir, "equipes_joueurs_2025-2026.csv")

    players_db_path = _first_existing_path([
        os.path.join(data_dir, "hockey.players.csv"),
        os.path.join(data_dir, "Hockey.Players.csv"),
        os.path.join(data_dir, "hockey.players.CSV"),
    ])

    st.subheader("🛠️ Gestion Admin")
    st.caption(f"DATA_DIR: {Path(data_dir).name}")

    # ---- Drive OAuth (UI only; never blocks)
    helpers = _try_import_drive_helpers()
    oauth_enabled = helpers.get("oauth_enabled")
    oauth_ui = helpers.get("render_oauth_ui")
    get_drive_service = helpers.get("get_drive_service")

    with st.expander("🔐 Connexion Google Drive (OAuth)", expanded=False):
        if callable(oauth_ui):
            try:
                oauth_ui()
            except Exception:
                st.warning("UI OAuth détectée mais a échoué. Vérifie secrets/scopes.")
        else:
            st.caption("Aucune UI OAuth détectée dans services/*. Tu peux quand même utiliser le fallback local.")

        drive_svc = None
        if callable(get_drive_service):
            try:
                drive_svc = get_drive_service()
            except Exception:
                drive_svc = None
        if drive_svc and folder_id:
            st.success("Drive OAuth: connecté.")
        else:
            st.info("Drive OAuth: non disponible (fallback local actif).")

    # ---- Caps settings (NO cap-live per team)
    DEFAULT_CAP_GC = 88_000_000
    DEFAULT_CAP_CE = 12_000_000

    # autoload settings once per session
    if "admin_settings_loaded" not in st.session_state:
        st.session_state["admin_settings_loaded"] = True
        # local first
        s_path = _settings_local_path(data_dir)
        if os.path.exists(s_path):
            df = _safe_read_csv(s_path)
            if not df.empty:
                try:
                    st.session_state["CAP_GC"] = int(df.iloc[0].get("CAP_GC", DEFAULT_CAP_GC))
                    st.session_state["CAP_CE"] = int(df.iloc[0].get("CAP_CE", DEFAULT_CAP_CE))
                except Exception:
                    pass
        else:
            # drive fallback
            try:
                if folder_id and drive_svc:
                    b = _drive_download_settings(drive_svc, folder_id, "settings.csv")
                    if b:
                        row = _load_settings_from_bytes(b)
                        if row:
                            st.session_state["CAP_GC"] = int(row.get("CAP_GC", DEFAULT_CAP_GC))
                            st.session_state["CAP_CE"] = int(row.get("CAP_CE", DEFAULT_CAP_CE))
                            # cache local
                            _safe_write_csv(pd.DataFrame([{"CAP_GC": st.session_state["CAP_GC"], "CAP_CE": st.session_state["CAP_CE"]}]), s_path)
            except Exception:
                pass

    with st.expander("💰 Plafonds salariaux (GC / CE)", expanded=False):
        cap_gc_txt = st.text_input("Cap GC", value=_fmt_money(st.session_state.get("CAP_GC", DEFAULT_CAP_GC)), key="cap_gc_txt")
        cap_ce_txt = st.text_input("Cap CE", value=_fmt_money(st.session_state.get("CAP_CE", DEFAULT_CAP_CE)), key="cap_ce_txt")

        cap_gc = _parse_money(cap_gc_txt) or int(st.session_state.get("CAP_GC", DEFAULT_CAP_GC))
        cap_ce = _parse_money(cap_ce_txt) or int(st.session_state.get("CAP_CE", DEFAULT_CAP_CE))

        # validate
        def _cap_ok(v): return 1_000_000 <= int(v) <= 200_000_000
        if not _cap_ok(cap_gc): st.error("Cap GC invalide (1M–200M).")
        if not _cap_ok(cap_ce): st.error("Cap CE invalide (1M–200M).")

        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            if st.button("💾 Sauvegarder (local + Drive)", use_container_width=True):
                if _cap_ok(cap_gc) and _cap_ok(cap_ce):
                    st.session_state["CAP_GC"] = int(cap_gc)
                    st.session_state["CAP_CE"] = int(cap_ce)
                    s_path = _settings_local_path(data_dir)
                    ok_local = _safe_write_csv(pd.DataFrame([{"CAP_GC": int(cap_gc), "CAP_CE": int(cap_ce)}]), s_path)
                    ok_drive = False
                    try:
                        if folder_id and drive_svc:
                            ok_drive = _drive_upload_settings(drive_svc, folder_id, "settings.csv", _settings_to_csv_bytes(cap_gc, cap_ce))
                    except Exception:
                        ok_drive = False
                    st.success(f"✅ Sauvé. Local={ok_local} Drive={ok_drive}")
        with col2:
            if st.button("🔄 Recharger (local/Drive)", use_container_width=True):
                # reset and reload next run
                st.session_state.pop("admin_settings_loaded", None)
                st.rerun()

        st.caption(f"Actuel: GC {_fmt_money(st.session_state.get('CAP_GC', cap_gc))} • CE {_fmt_money(st.session_state.get('CAP_CE', cap_ce))}")

    # ---- Players DB availability (should now resolve)
    if not players_db_path:
        st.warning("hockey.players.csv indisponible — fallback Level par Salaire.")
        players_db = pd.DataFrame()
    else:
        players_db = _safe_read_csv(players_db_path)

    # ---- Ensure equipes file exists
    if not os.path.exists(equipes_csv):
        base_cols = ["Proprietaire","Joueur","Slot","Salaire","Level","IR Date"]
        _safe_write_csv(pd.DataFrame(columns=base_cols), equipes_csv)

    equipes_df = _safe_read_csv(equipes_csv)
    if equipes_df.empty:
        owners = _default_owners()
    else:
        col_owner = "Proprietaire" if "Proprietaire" in equipes_df.columns else ("Propriétaire" if "Propriétaire" in equipes_df.columns else "")
        owners = sorted([str(x).strip() for x in equipes_df.get(col_owner, pd.Series([],dtype=str)).dropna().unique().tolist()]) if col_owner else []
        owners = owners or _default_owners()

    # ---- Add player block (always visible)
    with st.expander("➕ Ajouter joueur(s) (Admin)", expanded=False):
        if players_db.empty:
            st.info("Players DB vide/introuvable. Ajoute data/hockey.players.csv pour activer la recherche.")
        else:
            # determine player name column
            name_col = "Joueur" if "Joueur" in players_db.columns else ("Player" if "Player" in players_db.columns else players_db.columns[0])
            all_names = sorted([str(x).strip() for x in players_db[name_col].dropna().unique().tolist() if str(x).strip()])
            q = st.text_input("Recherche joueur", value="", key="addp_query")
            cand = _fuzzy_candidates(q, all_names, limit=50)
            pick = st.multiselect("Choisir jusqu'à 5 joueurs", options=cand, default=[], max_selections=5, key="addp_pick")

            c1,c2,c3 = st.columns([1,1,1])
            with c1:
                team = st.selectbox("Équipe", options=owners, index=0, key="addp_team")
            with c2:
                scope = st.selectbox("Groupe", options=["GC","CE"], index=0, key="addp_scope")
            with c3:
                slot = st.selectbox("Slot", options=["Actif","Banc","IR","Mineur"], index=0, key="addp_slot")

            if st.button("✅ Ajouter à l'équipe", use_container_width=True, disabled=(len(pick)==0)):
                # reload latest
                dfm = _safe_read_csv(equipes_csv)
                if dfm.empty:
                    dfm = pd.DataFrame(columns=["Proprietaire","Joueur","Slot","Salaire","Level","IR Date"])

                # normalize owner col
                if "Propriétaire" in dfm.columns and "Proprietaire" not in dfm.columns:
                    dfm = dfm.rename(columns={"Propriétaire":"Proprietaire"})

                # existing map for anti-dup
                existing = {}
                if "Joueur" in dfm.columns and "Proprietaire" in dfm.columns:
                    for _,r in dfm.iterrows():
                        j = str(r.get("Joueur","")).strip().lower()
                        o = str(r.get("Proprietaire","")).strip()
                        if j:
                            existing[j] = o

                added=0; blocked=[]
                for nm in pick:
                    key = str(nm).strip().lower()
                    if key in existing and existing[key] and existing[key] != team:
                        blocked.append(f"{nm} (déjà: {existing[key]})")
                        continue

                    # salary & level if exist
                    row = players_db.loc[players_db[name_col]==nm].head(1)
                    salaire = ""
                    level = ""
                    if not row.empty:
                        r0 = row.iloc[0].to_dict()
                        # try salary columns
                        for sc in ["Salaire","Salary","Cap Hit","CapHit","AAV"]:
                            if sc in r0 and str(r0.get(sc,"")).strip():
                                salaire = r0.get(sc,"")
                                break
                        if "Level" in r0 and str(r0.get("Level","")).strip():
                            level = str(r0.get("Level","")).strip()

                    dfm = pd.concat([dfm, pd.DataFrame([{
                        "Proprietaire": team,
                        "Joueur": nm,
                        "Slot": f"{scope} {slot}",
                        "Salaire": salaire,
                        "Level": level,
                        "IR Date": ""
                    }])], ignore_index=True)
                    existing[key]=team
                    added += 1

                ok = _safe_write_csv(dfm, equipes_csv)
                st.success(f"✅ Ajoutés: {added} • Sauvegarde: {ok}")
                if blocked:
                    st.warning("⛔ Bloqués (déjà dans une autre équipe):\n- " + "\n- ".join(blocked))
                st.cache_data.clear()
                st.rerun()
