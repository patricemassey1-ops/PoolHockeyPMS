# tabs/admin.py
# PMS Pool Hockey — Admin Tab (stable, no cap-live bars)

from __future__ import annotations

import os
import io
import re
import time
import difflib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import streamlit as st

# Optional Google OAuth libs (fallback OAuth UI)
try:
    from google_auth_oauthlib.flow import Flow  # type: ignore
except Exception:
    Flow = None  # type: ignore

try:
    from google.oauth2.credentials import Credentials  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaIoBaseUpload  # type: ignore
except Exception:
    Credentials = None  # type: ignore
    build = None  # type: ignore
    MediaIoBaseUpload = None  # type: ignore


# =============================================================================
# Paths (robust; Linux case-sensitive)
# =============================================================================
_THIS_FILE = Path(__file__).resolve()
_ROOT_DIR = _THIS_FILE.parents[1]  # .../poolhockeypms

def _resolve_data_dir(ctx: Optional[dict] = None) -> Path:
    cands: List[Path] = []
    if isinstance(ctx, dict):
        v = ctx.get("DATA_DIR") or ctx.get("data_dir")
        if v:
            cands.append(Path(str(v)))
    cands += [
        _ROOT_DIR / "data",
        _ROOT_DIR / "Data",
        _ROOT_DIR / "DATA",
        Path.cwd() / "data",
        Path.cwd() / "Data",
    ]
    for d in cands:
        try:
            if d.exists() and d.is_dir():
                return d.resolve()
        except Exception:
            pass
    return (_ROOT_DIR / "data").resolve()

def _first_existing_path(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p and p.exists():
                return p
        except Exception:
            pass
    return None


# =============================================================================
# Constants
# =============================================================================
SETTINGS_FILENAME = "settings.csv"
DEFAULT_CAP_GC = 88_000_000
DEFAULT_CAP_CE = 12_000_000
MIN_CAP = 1_000_000
MAX_CAP = 200_000_000

DEFAULT_TEAMS = ["Canadiens", "Cracheurs", "Nordiques", "Predateurs", "Red_Wings", "Whalers"]

EQUIPES_COLUMNS = ["Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"]


# =============================================================================
# CSV utils
# =============================================================================
@st.cache_data(show_spinner=False)
def _read_csv_safe(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p, low_memory=False, on_bad_lines="skip")
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python", low_memory=False, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def _norm_name(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _to_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    s = str(x or "")
    s = re.sub(r"[^\d]", "", s)
    try:
        return int(s) if s else 0
    except Exception:
        return 0

def _fmt_money(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        n = 0
    return f"{n:,}".replace(",", " ") + " $"

def _parse_money(s: str) -> int:
    s = str(s or "")
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else 0


# =============================================================================
# Settings (local + Drive)
# =============================================================================
def _settings_path(data_dir: Path) -> Path:
    return (data_dir / SETTINGS_FILENAME).resolve()

def load_settings_local(data_dir: Path) -> Dict[str, int]:
    p = _settings_path(data_dir)
    if not p.exists():
        return {}
    df = _read_csv_safe(str(p))
    if df.empty:
        return {}
    # accept either key/value or columns cap_gc/cap_ce
    out: Dict[str, int] = {}
    cols = [c.lower() for c in df.columns]
    if "key" in cols and "value" in cols:
        key_col = df.columns[cols.index("key")]
        val_col = df.columns[cols.index("value")]
        for _, r in df.iterrows():
            k = str(r.get(key_col, "")).strip().lower()
            v = _to_int(r.get(val_col, 0))
            if k:
                out[k] = v
    else:
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("cap_gc", "cap ce", "cap_ce"):
                out["cap_gc" if "gc" in cl else "cap_ce"] = _to_int(df.iloc[0][c])
    return out

def save_settings_local(data_dir: Path, cap_gc: int, cap_ce: int) -> bool:
    p = _settings_path(data_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [{"key": "cap_gc", "value": int(cap_gc)}, {"key": "cap_ce", "value": int(cap_ce)}]
        )
        df.to_csv(p, index=False, encoding="utf-8")
        return True
    except Exception:
        return False


# =============================================================================
# Drive helpers (uses existing services.drive if present, else fallback)
# =============================================================================
def _try_import_services_drive():
    try:
        from services import drive as drive_mod  # type: ignore
        return drive_mod
    except Exception:
        return None

def _drive_service_from_session() -> Optional[Any]:
    # allow app to inject drive service in session_state
    try:
        return st.session_state.get("drive_service")
    except Exception:
        return None

def _drive_service_from_oauth_creds() -> Optional[Any]:
    if build is None or Credentials is None:
        return None
    creds_dict = st.session_state.get("drive_creds")
    if not isinstance(creds_dict, dict):
        return None
    try:
        creds = Credentials(
            token=creds_dict.get("token"),
            refresh_token=creds_dict.get("refresh_token"),
            token_uri=creds_dict.get("token_uri"),
            client_id=creds_dict.get("client_id"),
            client_secret=creds_dict.get("client_secret"),
            scopes=creds_dict.get("scopes") or ["https://www.googleapis.com/auth/drive"],
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None

def _drive_upload_bytes(service, folder_id: str, filename: str, payload: bytes) -> bool:
    if not service or not folder_id or MediaIoBaseUpload is None:
        return False
    try:
        media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="text/csv", resumable=False)
        body = {"name": filename, "parents": [folder_id]}
        # try update if exists
        q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        res = service.files().list(q=q, fields="files(id,name)").execute()
        files = res.get("files", []) if isinstance(res, dict) else []
        if files:
            fid = files[0]["id"]
            service.files().update(fileId=fid, media_body=media).execute()
        else:
            service.files().create(body=body, media_body=media, fields="id").execute()
        return True
    except Exception:
        return False

def _drive_download_bytes(service, folder_id: str, filename: str) -> Optional[bytes]:
    if not service or not folder_id:
        return None
    try:
        q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        res = service.files().list(q=q, fields="files(id,name)").execute()
        files = res.get("files", []) if isinstance(res, dict) else []
        if not files:
            return None
        fid = files[0]["id"]
        req = service.files().get_media(fileId=fid)
        fh = io.BytesIO()
        downloader = None
        try:
            from googleapiclient.http import MediaIoBaseDownload  # type: ignore
            downloader = MediaIoBaseDownload(fh, req)
        except Exception:
            downloader = None
        if downloader is None:
            return None
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()
    except Exception:
        return None


# =============================================================================
# OAuth fallback UI (NO nested expander; safe)
# =============================================================================
def render_drive_oauth_connect_ui() -> None:
    if Flow is None:
        st.info("OAuth Drive: google-auth-oauthlib indisponible.")
        return

    cfg = st.secrets.get("gdrive_oauth", {})
    client_id = str(cfg.get("client_id", "") or "")
    client_secret = str(cfg.get("client_secret", "") or "")
    redirect_uri = str(cfg.get("redirect_uri", "") or "")

    if not (client_id and client_secret and redirect_uri):
        st.info("OAuth Drive: Secrets [gdrive_oauth] incomplets (client_id / client_secret / redirect_uri).")
        return

    if st.session_state.get("drive_creds"):
        st.success("✅ Drive OAuth connecté.")
        if st.button("🔌 Déconnecter Drive OAuth", use_container_width=True):
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


# =============================================================================
# Players DB + Fuzzy search
# =============================================================================
def _players_db_path(data_dir: Path) -> Optional[Path]:
    return _first_existing_path([
        data_dir / "hockey.players.csv",
        data_dir / "Hockey.Players.csv",
        data_dir / "Hockey.Players.CSV",
        data_dir / "hockey.players.CSV",
    ])

@st.cache_data(show_spinner=False)
def load_players_db(path: str) -> pd.DataFrame:
    return _read_csv_safe(path)

def _player_name_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in df.columns:
        if c.strip().lower() in ("joueur", "player", "name", "nom"):
            return c
    return df.columns[0] if len(df.columns) else None

def _fuzzy_suggestions(query: str, choices: List[str], k: int = 20) -> List[str]:
    q = _norm_name(query)
    if not q:
        return choices[:k]
    # first: contains
    contains = [c for c in choices if q in _norm_name(c)]
    if len(contains) >= k:
        return contains[:k]
    # then difflib
    rest = [c for c in choices if c not in contains]
    close = difflib.get_close_matches(query, rest, n=(k-len(contains)), cutoff=0.5)
    return (contains + close)[:k]


# =============================================================================
# equipes_joueurs master
# =============================================================================
def equipes_master_path(data_dir: Path, season_lbl: str) -> Path:
    # local cache always
    return (data_dir / f"equipes_joueurs_{season_lbl}.csv").resolve()

def load_master(data_dir: Path, season_lbl: str) -> pd.DataFrame:
    p = equipes_master_path(data_dir, season_lbl)
    df = _read_csv_safe(str(p)) if p.exists() else pd.DataFrame(columns=EQUIPES_COLUMNS)
    df = _ensure_cols(df, EQUIPES_COLUMNS)
    return df

def save_master(data_dir: Path, season_lbl: str, df: pd.DataFrame) -> bool:
    p = equipes_master_path(data_dir, season_lbl)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        _ensure_cols(df, EQUIPES_COLUMNS).to_csv(p, index=False, encoding="utf-8")
        return True
    except Exception:
        return False


# =============================================================================
# Main render
# =============================================================================
def render(ctx: dict) -> None:
    # Always render something (no black screen)
    st.subheader("🛠️ Gestion Admin")

    is_admin = bool(ctx.get("is_admin"))
    if not is_admin:
        st.warning("Accès admin requis.")
        return

    season_lbl = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    folder_id = str(ctx.get("drive_folder_id") or ctx.get("folder_id") or "").strip()

    data_dir = _resolve_data_dir(ctx)

    # ---- Drive service (best-effort)
    drive_mod = _try_import_services_drive()
    drive_svc = None
    if drive_mod is not None:
        # try common helpers
        for fn in ("get_drive_service", "get_drive_service_from_session", "drive_service_from_session"):
            if hasattr(drive_mod, fn):
                try:
                    drive_svc = getattr(drive_mod, fn)()
                    break
                except Exception:
                    drive_svc = None
    if drive_svc is None:
        drive_svc = _drive_service_from_session()
    if drive_svc is None:
        drive_svc = _drive_service_from_oauth_creds()

    # ---- OAuth expander (collapsed)
    with st.expander("🔐 Connexion Google Drive (OAuth)", expanded=False):
        # prefer project UI if exists, else fallback UI
        rendered = False
        if drive_mod is not None:
            for fn in ("render_oauth_connect_ui", "render_drive_oauth_connect_ui"):
                if hasattr(drive_mod, fn):
                    try:
                        getattr(drive_mod, fn)()
                        rendered = True
                        break
                    except Exception:
                        rendered = False
        if not rendered:
            render_drive_oauth_connect_ui()

        if folder_id:
            st.caption(f"folder_id (ctx): `{folder_id}`")
        if drive_svc is None:
            st.warning("Drive OAuth non disponible (creds manquants ou service indisponible).")
        else:
            st.success("Drive service prêt (si permissions OK).")

    # ---- Settings (caps)
    with st.expander("💰 Plafonds salariaux (GC / CE)", expanded=False):
        # auto-load once per session
        if "caps_loaded" not in st.session_state:
            st.session_state["caps_loaded"] = True
            # local first
            s = load_settings_local(data_dir)
            cap_gc = int(s.get("cap_gc", DEFAULT_CAP_GC))
            cap_ce = int(s.get("cap_ce", DEFAULT_CAP_CE))
            st.session_state["CAP_GC"] = cap_gc
            st.session_state["CAP_CE"] = cap_ce
            # if no local and drive available, try drive
            if (not s) and drive_svc and folder_id:
                payload = _drive_download_bytes(drive_svc, folder_id, SETTINGS_FILENAME)
                if payload:
                    try:
                        p = _settings_path(data_dir)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(payload)
                        s2 = load_settings_local(data_dir)
                        if s2:
                            st.session_state["CAP_GC"] = int(s2.get("cap_gc", DEFAULT_CAP_GC))
                            st.session_state["CAP_CE"] = int(s2.get("cap_ce", DEFAULT_CAP_CE))
                    except Exception:
                        pass

        col1, col2 = st.columns(2)
        with col1:
            cap_gc_str = st.text_input("Cap GC", value=_fmt_money(int(st.session_state.get("CAP_GC", DEFAULT_CAP_GC))), key="cap_gc_txt")
        with col2:
            cap_ce_str = st.text_input("Cap CE", value=_fmt_money(int(st.session_state.get("CAP_CE", DEFAULT_CAP_CE))), key="cap_ce_txt")

        cap_gc_val = _parse_money(cap_gc_str)
        cap_ce_val = _parse_money(cap_ce_str)

        errs = []
        if not (MIN_CAP <= cap_gc_val <= MAX_CAP):
            errs.append(f"Cap GC invalide ({_fmt_money(cap_gc_val)}) — doit être entre {_fmt_money(MIN_CAP)} et {_fmt_money(MAX_CAP)}.")
        if not (MIN_CAP <= cap_ce_val <= MAX_CAP):
            errs.append(f"Cap CE invalide ({_fmt_money(cap_ce_val)}) — doit être entre {_fmt_money(MIN_CAP)} et {_fmt_money(MAX_CAP)}.")

        if errs:
            for e in errs:
                st.error(e)
        else:
            st.session_state["CAP_GC"] = int(cap_gc_val)
            st.session_state["CAP_CE"] = int(cap_ce_val)

        b1, b2, b3 = st.columns([1,1,2])
        with b1:
            if st.button("💾 Sauvegarder (local + Drive)", use_container_width=True):
                if errs:
                    st.warning("Impossible de sauvegarder — valeurs invalides.")
                else:
                    ok_local = save_settings_local(data_dir, int(cap_gc_val), int(cap_ce_val))
                    ok_drive = False
                    if drive_svc and folder_id:
                        try:
                            payload = _settings_path(data_dir).read_bytes()
                            ok_drive = _drive_upload_bytes(drive_svc, folder_id, SETTINGS_FILENAME, payload)
                        except Exception:
                            ok_drive = False
                    if ok_local:
                        st.success("✅ Settings sauvegardés en local.")
                    else:
                        st.error("❌ Échec sauvegarde locale.")
                    if folder_id:
                        st.info("Drive: upload " + ("✅ OK" if ok_drive else "⚠️ non effectué"))
        with b2:
            if st.button("🔄 Recharger (local/Drive)", use_container_width=True):
                st.session_state.pop("caps_loaded", None)
                st.rerun()
        with b3:
            st.caption(f"Actuel: GC {st.session_state.get('CAP_GC')} • CE {st.session_state.get('CAP_CE')}")

    # ---- Players DB availability message
    pdb_path = _players_db_path(data_dir)
    if not pdb_path:
        st.warning("hockey.players.csv indisponible — fallback Level par Salaire.")
    else:
        st.caption(f"players_db: `{pdb_path.name}`")

    # ---- Add player (always visible)
    with st.expander("➕ Ajouter joueur(s) (Admin)", expanded=False):
        master = load_master(data_dir, season_lbl)
        owners = sorted([o for o in master.get("Propriétaire", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if o.strip()])
        if not owners:
            owners = DEFAULT_TEAMS

        st.caption("Recherche un joueur et attribue-le à une équipe/slot. Anti-duplication: empêche qu'un joueur soit déjà dans une autre équipe.")

        # Load players DB for search
        players_df = load_players_db(str(pdb_path)) if pdb_path else pd.DataFrame()
        name_col = _player_name_col(players_df) if not players_df.empty else None
        all_names = []
        if name_col:
            all_names = players_df[name_col].dropna().astype(str).tolist()

        q = st.text_input("Rechercher joueur", value="", key="adm_addplayer_q")
        sugg = _fuzzy_suggestions(q, all_names, k=25) if all_names else []
        pick = st.selectbox("Sélection", options=[""] + sugg, index=0, key="adm_addplayer_pick")

        colA, colB, colC = st.columns(3)
        with colA:
            team = st.selectbox("Équipe", options=owners, key="adm_addplayer_team")
        with colB:
            scope = st.selectbox("Groupe", options=["GC", "CE"], key="adm_addplayer_scope")
        with colC:
            slot = st.selectbox("Slot", options=["Actif", "Banc", "Mineur", "IR"], key="adm_addplayer_slot")

        max_n = st.number_input("Nombre max (batch)", min_value=1, max_value=5, value=1, step=1, key="adm_addplayer_max")

        # allow multi-add by comma list
        extra = st.text_area("Ou ajouter plusieurs noms (1 par ligne)", value="", height=90, key="adm_addplayer_bulk")

        def _already_owned(df: pd.DataFrame, player_name: str) -> Optional[str]:
            if df is None or df.empty:
                return None
            key = _norm_name(player_name)
            s = df["Joueur"].astype(str).map(_norm_name)
            hit = df.loc[s == key]
            if hit.empty:
                return None
            try:
                return str(hit.iloc[0].get("Propriétaire", "") or "")
            except Exception:
                return "?"

        if st.button("✅ Ajouter", use_container_width=True, key="adm_addplayer_go"):
            # build list
            names = []
            if pick:
                names.append(pick)
            if extra.strip():
                names += [x.strip() for x in extra.splitlines() if x.strip()]
            # dedupe, cap to 5
            dedup = []
            seen = set()
            for n in names:
                k = _norm_name(n)
                if k and k not in seen:
                    seen.add(k)
                    dedup.append(n)
            dedup = dedup[: int(max_n)]

            if not dedup:
                st.warning("Choisis au moins un joueur.")
                st.stop()

            added = 0
            blocked = []
            for nm in dedup:
                owner = _already_owned(master, nm)
                if owner and owner.strip() and owner.strip().lower() != str(team).strip().lower():
                    blocked.append((nm, owner))
                    continue

                row = {c: "" for c in EQUIPES_COLUMNS}
                row["Propriétaire"] = team
                row["Joueur"] = nm
                row["Statut"] = scope
                row["Slot"] = slot
                # keep salary/level blank; other tabs can enrich
                master = pd.concat([master, pd.DataFrame([row])], ignore_index=True)
                added += 1

            master = _ensure_cols(master, EQUIPES_COLUMNS)
            ok = save_master(data_dir, season_lbl, master)
            if ok:
                st.cache_data.clear()
                st.success(f"✅ Ajouté: {added}")
            else:
                st.error("❌ Échec sauvegarde du master local.")

            if blocked:
                st.warning("Bloqués (déjà assignés à une autre équipe):")
                for nm, ow in blocked:
                    st.write(f"- {nm} (déjà chez **{ow}**)")

            # optional upload master to Drive
            if drive_svc and folder_id and ok:
                try:
                    payload = equipes_master_path(data_dir, season_lbl).read_bytes()
                    up_ok = _drive_upload_bytes(drive_svc, folder_id, equipes_master_path(data_dir, season_lbl).name, payload)
                    st.info("Drive: upload master " + ("✅ OK" if up_ok else "⚠️ non effectué"))
                except Exception:
                    st.info("Drive: upload master ⚠️ non effectué")

    # ---- Import section placeholders (collapsed; keep app stable)
    with st.expander("📥 Import local (fallback) — multi-upload CSV équipes", expanded=False):
        st.caption("Import local/Drive des CSV équipes est géré ailleurs. Ici, Admin sert à config + ajout joueur.")
        st.write(f"DATA_DIR: `{data_dir}`")

