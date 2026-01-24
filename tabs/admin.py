# admin.py
# ============================================================
# PMS Pool Hockey — Admin Module (Streamlit)
# - Backups & Restore (Drive OAuth)
# - Restore local fallback
# - Équipes: Import / Preview / Validate / Reload / Rollback
# ============================================================

from __future__ import annotations

import os
import io
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---- Google OAuth Drive deps
# pip: google-auth, google-auth-oauthlib, google-api-python-client
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
    from google.oauth2.credentials import Credentials
except Exception:
    build = None  # type: ignore
    MediaIoBaseDownload = None  # type: ignore
    MediaIoBaseUpload = None  # type: ignore
    Credentials = None  # type: ignore


# ============================================================
# Config (adapte si besoin)
# ============================================================

DATA_DIR_DEFAULT = "Data"  # ton repo: /Data
DRIVE_FOLDER_ID_DEFAULT = ""  # tu le passes via param ou secrets

# Fichiers “critiques” qu’on backup/restore souvent
# Adapte au besoin
def critical_files_for_season(season_lbl: str) -> List[str]:
    season_lbl = str(season_lbl or "").strip() or "2025-2026"
    return [
        f"equipes_joueurs_{season_lbl}.csv",
        "hockey.players.csv",
        "puckpedia.contracts.csv",
        "backup_history.csv",
        f"transactions_{season_lbl}.csv",
        f"points_periods_{season_lbl}.csv",
        f"event_log_{season_lbl}.csv",
    ]


# ============================================================
# Drive OAuth helpers (minimal, robust)
# ============================================================

def oauth_drive_enabled() -> bool:
    """
    True si on a des creds OAuth valides dans st.session_state (ou secrets).
    Ici, on considère "drive_creds" en session_state.
    """
    return bool(st.session_state.get("drive_creds"))


def get_oauth_drive_service() -> Optional[Any]:
    """
    Retourne un service Drive (googleapiclient) si creds dispo.
    """
    if build is None or Credentials is None:
        return None
    creds_dict = st.session_state.get("drive_creds")
    if not creds_dict:
        return None

    try:
        creds = Credentials.from_authorized_user_info(creds_dict)
        svc = build("drive", "v3", credentials=creds)
        return svc
    except Exception:
        return None


def list_drive_csv_files(svc: Any, folder_id: str) -> List[Dict[str, str]]:
    """
    Liste les CSV dans un folder Drive. Retour: [{id,name}, ...]
    """
    if not svc or not folder_id:
        return []
    q = f"'{folder_id}' in parents and trashed=false and mimeType='text/csv'"
    res = svc.files().list(
        q=q,
        fields="files(id,name,createdTime,modifiedTime)",
        pageSize=200,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", []) or []
    # tri par modifiedTime desc si présent
    def _key(x):
        return x.get("modifiedTime") or x.get("createdTime") or ""
    files.sort(key=_key, reverse=True)
    return [{"id": f["id"], "name": f["name"]} for f in files if f.get("id") and f.get("name")]


def drive_download_file(svc: Any, file_id: str) -> bytes:
    """
    Download binaire d’un fichier Drive.
    """
    if not svc or not file_id:
        raise ValueError("drive_download_file: svc/file_id missing")

    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        # status peut être None
    return fh.getvalue()


def _find_drive_file_by_name(svc: Any, folder_id: str, name: str) -> Optional[Dict[str, str]]:
    """
    Cherche un fichier par name dans folder_id. Retour {id,name} ou None.
    """
    if not svc or not folder_id or not name:
        return None
    # échappement basique
    safe_name = name.replace("'", "\\'")
    q = f"'{folder_id}' in parents and trashed=false and name='{safe_name}'"
    res = svc.files().list(
        q=q,
        fields="files(id,name)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", []) or []
    if not files:
        return None
    f = files[0]
    return {"id": f["id"], "name": f["name"]}


def drive_upload_upsert_csv(
    svc: Any,
    folder_id: str,
    local_path: str,
    drive_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Upload/Upsert: si drive_name existe dans folder, update, sinon create.
    Retour {id,name}
    """
    if not svc or not folder_id:
        raise ValueError("drive_upload_upsert_csv: svc/folder_id missing")
    if not local_path or not os.path.exists(local_path):
        raise FileNotFoundError(local_path)

    drive_name = drive_name or os.path.basename(local_path)
    existing = _find_drive_file_by_name(svc, folder_id, drive_name)

    media = MediaIoBaseUpload(
        io.FileIO(local_path, "rb"),
        mimetype="text/csv",
        resumable=True,
    )

    if existing:
        file_id = existing["id"]
        updated = svc.files().update(
            fileId=file_id,
            media_body=media,
            fields="id,name",
            supportsAllDrives=True,
        ).execute()
        return {"id": updated["id"], "name": updated["name"]}
    else:
        metadata = {"name": drive_name, "parents": [folder_id]}
        created = svc.files().create(
            body=metadata,
            media_body=media,
            fields="id,name",
            supportsAllDrives=True,
        ).execute()
        return {"id": created["id"], "name": created["name"]}


# ============================================================
# Data helpers
# ============================================================

def ensure_data_dir(data_dir: str) -> str:
    data_dir = data_dir or DATA_DIR_DEFAULT
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def read_csv_bytes_to_df(csv_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(csv_bytes))
    except Exception:
        return pd.read_csv(io.BytesIO(csv_bytes), encoding="latin-1")


def validate_equipes_df(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Adapte expected à TES colonnes.
    """
    expected = [
        "Propriétaire",
        "Joueur",
        "Position",
        "Equipe",   # ou "Équipe" chez toi
        "Statut",
    ]
    cols = list(df.columns)
    missing = [c for c in expected if c not in cols]
    extras = [c for c in cols if c not in expected]
    ok = (len(missing) == 0)
    return ok, missing, extras


def reload_equipes_in_memory(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["Propriétaire", "Joueur", "Position", "Equipe", "Statut", "Équipe"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    st.session_state["equipes_df"] = df
    st.session_state["equipes_path"] = path
    st.session_state["equipes_last_loaded"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


# ============================================================
# UI blocks
# ============================================================

def ui_backups_restore_drive(
    *,
    svc: Any,
    folder_id: str,
    data_dir: str,
    season_lbl: str,
) -> None:
    """
    Section Drive: Restore selected CSV + Backup now (multi-select).
    """
    st.markdown("## ☁️ Backups & Restore (Drive)")
    st.caption("Dossier Drive: My Drive / PMS Pool Data / PoolHockeyData")
    st.code(f"folder_id = {folder_id}")

    # ---- Restore selected CSV (Drive)
    st.markdown("### ☁️ Drive — Restore selected CSV (OAuth)")
    drive_csvs = list_drive_csv_files(svc, folder_id)

    if not drive_csvs:
        st.info("Aucun CSV détecté dans le dossier Drive.")
    else:
        selected = st.selectbox(
            "Choisir un CSV à restaurer depuis Drive",
            drive_csvs,
            format_func=lambda x: x["name"],
            key="admin_restore_drive_select",
        )

        # Destination locale (dropdown)
        crit = critical_files_for_season(season_lbl)
        default_dest = f"equipes_joueurs_{season_lbl}.csv"
        dest_choice = st.selectbox(
            "Restaurer vers (local /Data)",
            crit,
            index=crit.index(default_dest) if default_dest in crit else 0,
            key="admin_restore_drive_dest",
        )
        target_path = os.path.join(data_dir, dest_choice)

        if st.button("⬇️ Restaurer depuis Drive", use_container_width=True, key="admin_restore_drive_btn"):
            try:
                content = drive_download_file(svc, selected["id"])
                with open(target_path, "wb") as f:
                    f.write(content)
                st.success(f"✅ Restauré → `{target_path}`")
                # reload auto si equipes
                if dest_choice.lower().startswith("equipes_joueurs_"):
                    try:
                        reload_equipes_in_memory(target_path)
                        st.info("🔄 Équipes rechargées en mémoire.")
                    except Exception as e:
                        st.warning(f"Équipes: reload échoué: {e}")
                st.rerun()
            except Exception as e:
                st.error(f"Restore Drive échoué: {e}")

    st.divider()

    # ---- Backup now (Drive) (upsert)
    st.markdown("### ☁️ Drive — Backup now (OAuth)")
    st.caption("Upload UPSERT: si le fichier existe déjà dans Drive, il est mis à jour.")

    crit = critical_files_for_season(season_lbl)
    default_sel = [x for x in crit if os.path.exists(os.path.join(data_dir, x))]
    selected_files = st.multiselect(
        "Choisir les fichiers à sauvegarder",
        crit,
        default=default_sel,
        key="admin_backup_drive_multiselect",
    )

    if st.button("⬆️ Backup maintenant", use_container_width=True, key="admin_backup_drive_btn"):
        ok_count, fail_count = 0, 0
        for name in selected_files:
            local_path = os.path.join(data_dir, name)
            try:
                if not os.path.exists(local_path):
                    fail_count += 1
                    st.warning(f"⛔ Introuvable local: {local_path}")
                    continue
                res = drive_upload_upsert_csv(svc, folder_id, local_path, drive_name=name)
                ok_count += 1
                st.success(f"✅ Drive upsert: {res['name']}")
            except Exception as e:
                fail_count += 1
                st.error(f"❌ Backup échoué ({name}): {e}")

        st.info(f"Backup terminé: ✅ {ok_count} | ❌ {fail_count}")


def ui_restore_local_fallback(*, data_dir: str, season_lbl: str) -> None:
    """
    Upload local -> write into /Data + optional reload
    """
    st.markdown("## 📦 Restore local (fallback)")
    st.caption("Si Drive OAuth n’est pas prêt, tu peux uploader un CSV et choisir la destination locale.")

    crit = critical_files_for_season(season_lbl)
    dest_choice = st.selectbox(
        "Restaurer vers (local)",
        crit,
        index=0,
        key="admin_restore_local_dest",
    )
    target_path = os.path.join(data_dir, dest_choice)

    uploaded = st.file_uploader("Uploader un CSV", type=["csv"], key="admin_restore_local_uploader")

    if st.button("💾 Restore local maintenant", use_container_width=True, key="admin_restore_local_btn"):
        if uploaded is None:
            st.warning("Upload un fichier CSV d’abord.")
            return
        try:
            with open(target_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"✅ Restauré → `{target_path}`")
            if dest_choice.lower().startswith("equipes_joueurs_"):
                try:
                    reload_equipes_in_memory(target_path)
                    st.info("🔄 Équipes rechargées en mémoire.")
                except Exception as e:
                    st.warning(f"Équipes: reload échoué: {e}")
            st.rerun()
        except Exception as e:
            st.error(f"Restore local échoué: {e}")


def ui_equipes_import_preview_validate_reload_rollback(
    *,
    svc: Optional[Any],
    drive_ok: bool,
    folder_id: str,
    data_dir: str,
    season_lbl: str,
) -> None:
    """
    Bloc unique:
    - Preview (Drive/Local)
    - Validate structure
    - Import + Reload
    - Rollback Drive -> Local + Reload
    """
    with st.expander("👥 Équipes — Import / Preview / Validate / Reload / Rollback", expanded=False):
        st.caption("Importer le CSV `equipes_joueurs_YYYY-YYYY.csv`, prévisualiser, valider, recharger en mémoire, et rollback depuis Drive.")

        target_path = os.path.join(data_dir, f"equipes_joueurs_{season_lbl}.csv")
        st.code(f"Destination locale: {target_path}")

        # session state preview
        st.session_state.setdefault("equipes_preview_df", None)
        st.session_state.setdefault("equipes_preview_src", "")

        colA, colB = st.columns(2)

        # ---- Drive source
        with colA:
            st.markdown("### ☁️ Drive (OAuth)")
            if drive_ok and svc and folder_id:
                drive_csvs = list_drive_csv_files(svc, folder_id)
                drive_equipes = [x for x in (drive_csvs or []) if "equipes_joueurs" in str(x.get("name", "")).lower()]

                if not drive_equipes:
                    st.info("Aucun CSV `equipes_joueurs...` trouvé sur Drive.")
                else:
                    selected_drive = st.selectbox(
                        "Choisir un CSV équipes sur Drive",
                        drive_equipes,
                        format_func=lambda x: x["name"],
                        key="equipes_drive_select",
                    )

                    if st.button("👁️ Preview (Drive)", use_container_width=True, key="equipes_preview_drive"):
                        try:
                            content = drive_download_file(svc, selected_drive["id"])
                            df_prev = read_csv_bytes_to_df(content)
                            st.session_state["equipes_preview_df"] = df_prev
                            st.session_state["equipes_preview_src"] = f"Drive: {selected_drive['name']}"
                            st.success("Preview chargé.")
                        except Exception as e:
                            st.error(f"Erreur preview Drive: {e}")

                    if st.button("⬇️ Importer (Drive) + Reload", use_container_width=True, key="equipes_import_drive"):
                        try:
                            content = drive_download_file(svc, selected_drive["id"])
                            df_check = read_csv_bytes_to_df(content)

                            ok, missing, extras = validate_equipes_df(df_check)
                            if not ok:
                                st.error(f"❌ Colonnes manquantes: {missing}")
                                st.stop()

                            with open(target_path, "wb") as f:
                                f.write(content)

                            reload_equipes_in_memory(target_path)
                            st.success("✅ Import Drive + reload OK.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur import Drive: {e}")
            else:
                st.warning("Drive OAuth non disponible.")

        # ---- Local source
        with colB:
            st.markdown("### 📦 Local (fallback)")
            uploaded = st.file_uploader("Uploader un CSV équipes", type=["csv"], key="equipes_local_uploader")

            if uploaded is not None:
                if st.button("👁️ Preview (Local)", use_container_width=True, key="equipes_preview_local"):
                    try:
                        df_prev = pd.read_csv(uploaded)
                        st.session_state["equipes_preview_df"] = df_prev
                        st.session_state["equipes_preview_src"] = f"Local: {uploaded.name}"
                        st.success("Preview chargé.")
                    except Exception as e:
                        st.error(f"Erreur preview Local: {e}")

                if st.button("💾 Importer (Local) + Reload", use_container_width=True, key="equipes_import_local"):
                    try:
                        df_check = pd.read_csv(uploaded)
                        ok, missing, extras = validate_equipes_df(df_check)
                        if not ok:
                            st.error(f"❌ Colonnes manquantes: {missing}")
                            st.stop()

                        uploaded.seek(0)
                        with open(target_path, "wb") as f:
                            f.write(uploaded.getbuffer())

                        reload_equipes_in_memory(target_path)
                        st.success("✅ Import Local + reload OK.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur import Local: {e}")

        st.divider()

        # ---- Preview + Validate + Reload current local
        st.markdown("### 🧼 Preview & Validation")
        df_prev = st.session_state.get("equipes_preview_df")

        if df_prev is None or not isinstance(df_prev, pd.DataFrame) or df_prev.empty:
            st.info("Aucun preview chargé. Clique sur **Preview (Drive)** ou **Preview (Local)**.")
        else:
            st.caption(f"Source: **{st.session_state.get('equipes_preview_src','')}**")
            st.dataframe(df_prev.head(50), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🧪 Valider structure (colonnes attendues)", use_container_width=True, key="equipes_validate_btn"):
                    ok, missing, extras = validate_equipes_df(df_prev)
                    if ok:
                        st.success("✅ Structure OK (colonnes attendues présentes).")
                        if extras:
                            st.info(f"Colonnes additionnelles (ok): {extras}")
                    else:
                        st.error(f"❌ Colonnes manquantes: {missing}")
                        if extras:
                            st.info(f"Colonnes additionnelles: {extras}")

            with c2:
                if st.button("🔄 Reload depuis fichier local actuel", use_container_width=True, key="equipes_reload_current_btn"):
                    if not os.path.exists(target_path):
                        st.error("Le fichier local cible n'existe pas encore.")
                    else:
                        try:
                            reload_equipes_in_memory(target_path)
                            st.success("✅ Reload effectué.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur reload: {e}")

        st.divider()

        # ---- Rollback from Drive
        st.markdown("### 🧯 Rollback rapide (Drive)")
        st.caption("Restaure un ancien `equipes_joueurs...` depuis Drive vers le fichier local, puis reload.")

        if drive_ok and svc and folder_id:
            try:
                drive_csvs = list_drive_csv_files(svc, folder_id)
                drive_equipes_all = [x for x in (drive_csvs or []) if "equipes_joueurs" in str(x.get("name", "")).lower()]

                if not drive_equipes_all:
                    st.info("Aucun CSV `equipes_joueurs...` disponible pour rollback sur Drive.")
                else:
                    rb = st.selectbox(
                        "Choisir un backup équipes (Drive)",
                        drive_equipes_all,
                        format_func=lambda x: x["name"],
                        key="equipes_rollback_select",
                    )

                    if st.button("🧯 Rollback (Drive) → Local + Reload", use_container_width=True, key="equipes_rollback_btn"):
                        content = drive_download_file(svc, rb["id"])
                        df_check = read_csv_bytes_to_df(content)
                        ok, missing, extras = validate_equipes_df(df_check)
                        if not ok:
                            st.error(f"Rollback refusé: colonnes manquantes {missing}")
                            st.stop()

                        with open(target_path, "wb") as f:
                            f.write(content)

                        reload_equipes_in_memory(target_path)
                        st.success(f"✅ Rollback effectué depuis `{rb['name']}` + reload OK.")
                        st.rerun()
            except Exception as e:
                st.error(f"Erreur rollback Drive: {e}")
        else:
            st.warning("Drive OAuth non disponible — rollback Drive impossible.")


# ============================================================
# Main entry: render admin tab
# ============================================================

def render_admin_tab(
    *,
    is_admin: bool,
    season_lbl: str,
    data_dir: str = DATA_DIR_DEFAULT,
    folder_id: str = DRIVE_FOLDER_ID_DEFAULT,
) -> None:
    """
    Call this from app.py when active_tab == "🛠️ Gestion Admin"
    """
    if not is_admin:
        st.warning("Accès admin requis.")
        st.stop()

    data_dir = ensure_data_dir(data_dir)
    season_lbl = str(season_lbl or "").strip() or "2025-2026"

    st.subheader("🛠️ Gestion Admin")

    # ---- Drive service
    svc = get_oauth_drive_service()
    drive_ok = bool(svc) and bool(folder_id)

    # ---- Equipes block (import/preview/validate/reload/rollback)
    ui_equipes_import_preview_validate_reload_rollback(
        svc=svc,
        drive_ok=drive_ok,
        folder_id=folder_id,
        data_dir=data_dir,
        season_lbl=season_lbl,
    )

    st.divider()

    # ---- Backups & Restore (Drive)
    if drive_ok and svc and folder_id:
        with st.expander("🧷 Backups & Restore (Drive)", expanded=False):
            ui_backups_restore_drive(
                svc=svc,
                folder_id=folder_id,
                data_dir=data_dir,
                season_lbl=season_lbl,
            )
    else:
        st.info("Drive OAuth indisponible. Les fonctions Drive sont désactivées (fallback local disponible).")

    st.divider()

    # ---- Restore local fallback
    with st.expander("📦 Restore local (fallback)", expanded=False):
        ui_restore_local_fallback(data_dir=data_dir, season_lbl=season_lbl)


# ============================================================
# Optional: quick self-test (run directly)
# ============================================================

def _demo_page():
    st.set_page_config(page_title="Admin Demo", layout="wide")
    st.title("Admin Demo")

    # Simule admin + season
    is_admin = True
    season_lbl = "2025-2026"

    # Mets ton folder_id ici ou en param depuis app.py
    folder_id = st.text_input("Drive folder_id", value=st.session_state.get("folder_id", DRIVE_FOLDER_ID_DEFAULT))
    st.session_state["folder_id"] = folder_id

    st.info("⚠️ Pour Drive OAuth: st.session_state['drive_creds'] doit contenir les creds OAuth (dict).")

    render_admin_tab(is_admin=is_admin, season_lbl=season_lbl, data_dir=DATA_DIR_DEFAULT, folder_id=folder_id)


if __name__ == "__main__":
    _demo_page()
