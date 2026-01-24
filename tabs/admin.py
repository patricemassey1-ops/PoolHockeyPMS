# tabs/admin.py
from __future__ import annotations

import os
import io
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from services.drive import (
    drive_ready,
    drive_list_files,
    drive_download_file,
    drive_upload_file,
)
from services.event_log import append_event, event_log_path


# =========================
# Helpers
# =========================
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


def _is_admin(ctx: dict) -> bool:
    return bool(ctx.get("is_admin") or st.session_state.get("is_admin") or False)


def _owners(ctx: dict) -> List[str]:
    owners = ctx.get("owners")
    if isinstance(owners, list) and owners:
        return owners
    # fallback safe
    return ["Canadiens", "Cracheurs", "Nordiques", "Prédateurs", "Red Wings", "Whalers"]


def _drive_folder_id(ctx: dict) -> str:
    # priorité: ctx -> secrets gdrive_folder_id -> compat old key
    return (
        str(ctx.get("drive_folder_id") or "").strip()
        or str(st.secrets.get("gdrive_folder_id", "") or "").strip()
        or str(st.secrets.get("drive_folder_id", "") or "").strip()
    )


def _critical_files(data_dir: str, season: str) -> List[Tuple[str, str]]:
    """
    (label, full_path)
    """
    return [
        (f"equipes_joueurs_{season}.csv", os.path.join(data_dir, f"equipes_joueurs_{season}.csv")),
        ("hockey.players.csv", os.path.join(data_dir, "hockey.players.csv")),
        ("puckpedia.contracts.csv", os.path.join(data_dir, "puckpedia.contracts.csv")),
        ("backup_history.csv", os.path.join(data_dir, "backup_history.csv")),
        (f"transactions_{season}.csv", os.path.join(data_dir, f"transactions_{season}.csv")),
        (f"points_periods_{season}.csv", os.path.join(data_dir, f"points_periods_{season}.csv")),
        (f"event_log_{season}.csv", os.path.join(data_dir, f"event_log_{season}.csv")),
    ]


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _append_backup_history(data_dir: str, row: Dict) -> None:
    """
    Append dans data/backup_history.csv (créé si absent).
    """
    path = os.path.join(data_dir, "backup_history.csv")
    _ensure_parent(path)
    base_cols = ["timestamp", "action", "mode", "file", "drive_name", "drive_id", "result", "details"]

    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=base_cols)

        for c in base_cols:
            if c not in df.columns:
                df[c] = ""

        out = {c: row.get(c, "") for c in base_cols}
        df = pd.concat([df, pd.DataFrame([out])], ignore_index=True)
        df.to_csv(path, index=False)
    except Exception:
        # jamais casser l'UI
        pass


def _human_bytes(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return ""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n} {unit}"
        n //= 1024
    return f"{n} TB"


# =========================
# Drive UI blocks
# =========================
def _ui_drive_restore(ctx: dict) -> None:
    data_dir = _data_dir(ctx)
    season = _season(ctx)
    folder_id = _drive_folder_id(ctx)

    st.subheader("☁️ Drive — Restore selected CSV (OAuth)")
    st.caption("Dossier Drive: My Drive / PMS Pool Data / PoolHockeyData")

    st.code(f"folder_id = {folder_id or '(missing)'}")

    if not folder_id:
        st.warning("folder_id manquant. Ajoute `gdrive_folder_id` dans Secrets Streamlit Cloud.")
        return

    if not drive_ready():
        st.warning("Drive OAuth non prêt. Ajoute `[gdrive_oauth]` + `gdrive_folder_id` dans Secrets.")
        return

    # Liste fichiers Drive (CSV seulement)
    files = drive_list_files(folder_id=folder_id, name_contains="", limit=400)
    csv_files = [f for f in files if str(f.get("name", "")).lower().endswith(".csv")]

    if not csv_files:
        st.info("Aucun CSV détecté dans le dossier Drive.")
        return

    # dropdown
    def _label(f):
        nm = f.get("name", "")
        mt = f.get("modifiedTime", "")
        sz = _human_bytes(f.get("size", 0))
        return f"{nm}  —  {mt}  —  {sz}"

    options = { _label(f): f for f in csv_files }
    pick_label = st.selectbox("Choisir un CSV à restaurer", list(options.keys()), key="admin_restore_drive_pick")
    picked = options.get(pick_label)

    # destination locale
    crit = _critical_files(data_dir, season)
    dest_names = [lab for lab, _ in crit] + ["(custom filename in data/)"]
    dest_pick = st.selectbox("Restaurer vers (local)", dest_names, key="admin_restore_drive_dest")

    custom = ""
    if dest_pick == "(custom filename in data/)":
        custom = st.text_input("Nom fichier destination (dans data/)", value="custom.csv", key="admin_restore_drive_custom")

    # resolve full dest path
    if dest_pick == "(custom filename in data/)":
        dest_path = os.path.join(data_dir, custom.strip() or "custom.csv")
        dest_name = os.path.basename(dest_path)
    else:
        dest_path = dict(crit).get(dest_pick)  # type: ignore
        dest_name = dest_pick

    c1, c2 = st.columns([1, 2])
    with c1:
        go = st.button("⬇️ Restore maintenant", type="primary", key="admin_restore_drive_go")
    with c2:
        st.caption("Le fichier local est remplacé par la version Drive choisie.")

    if go:
        if not picked:
            st.error("Aucun fichier Drive sélectionné.")
            return
        if not dest_path:
            st.error("Destination locale invalide.")
            return

        res = drive_download_file(picked.get("id", ""), dest_path)
        ts = _now_iso()
        if res.get("ok"):
            st.success(f"✅ Restored: `{dest_name}`")
            _append_backup_history(
                data_dir,
                {
                    "timestamp": ts,
                    "action": "restore",
                    "mode": "drive",
                    "file": dest_name,
                    "drive_name": picked.get("name", ""),
                    "drive_id": picked.get("id", ""),
                    "result": "ok",
                    "details": "",
                },
            )
            append_event(
                data_dir=data_dir,
                season=season,
                owner=str(st.session_state.get("selected_owner") or ""),
                event_type="restore",
                summary=f"Restore depuis Drive → {dest_name}",
                payload={"drive_file": picked.get("name", ""), "dest": dest_name},
            )
            st.rerun()
        else:
            st.error("❌ Restore échoué")
            st.code(res.get("error", "unknown error"))
            _append_backup_history(
                data_dir,
                {
                    "timestamp": ts,
                    "action": "restore",
                    "mode": "drive",
                    "file": dest_name,
                    "drive_name": picked.get("name", ""),
                    "drive_id": picked.get("id", ""),
                    "result": "fail",
                    "details": str(res.get("error", "")),
                },
            )


def _ui_drive_backup(ctx: dict) -> None:
    data_dir = _data_dir(ctx)
    season = _season(ctx)
    folder_id = _drive_folder_id(ctx)

    st.subheader("☁️ Drive — Backup now (OAuth)")
    st.caption("Upload UPSERT: si le fichier existe déjà dans Drive, il est mis à jour.")

    st.code(f"folder_id = {folder_id or '(missing)'}")

    if not folder_id:
        st.warning("folder_id manquant. Ajoute `gdrive_folder_id` dans Secrets Streamlit Cloud.")
        return

    if not drive_ready():
        st.warning("Drive OAuth non prêt. Ajoute `[gdrive_oauth]` + `gdrive_folder_id` dans Secrets.")
        return

    crit = _critical_files(data_dir, season)
    labels = [lab for lab, _ in crit]
    default_sel = labels  # tout par défaut

    selected = st.multiselect(
        "Choisir les fichiers à sauvegarder",
        options=labels,
        default=default_sel,
        key="admin_backup_files",
    )

    if st.button("⬆️ Backup maintenant", type="primary", key="admin_backup_go"):
        if not selected:
            st.warning("Aucun fichier sélectionné.")
            return

        ok_count = 0
        fail_count = 0
        results = []

        for lab, path in crit:
            if lab not in selected:
                continue
            ts = _now_iso()
            if not os.path.exists(path):
                fail_count += 1
                results.append((lab, "missing local file"))
                _append_backup_history(
                    data_dir,
                    {
                        "timestamp": ts,
                        "action": "backup",
                        "mode": "drive",
                        "file": lab,
                        "drive_name": lab,
                        "drive_id": "",
                        "result": "fail",
                        "details": "missing local file",
                    },
                )
                continue

            res = drive_upload_file(folder_id, path, drive_name=lab)
            if res.get("ok"):
                ok_count += 1
                results.append((lab, f"ok ({res.get('mode')})"))
                _append_backup_history(
                    data_dir,
                    {
                        "timestamp": ts,
                        "action": "backup",
                        "mode": "drive",
                        "file": lab,
                        "drive_name": lab,
                        "drive_id": res.get("id", ""),
                        "result": "ok",
                        "details": res.get("mode", ""),
                    },
                )
            else:
                fail_count += 1
                results.append((lab, f"fail: {res.get('error')}"))
                _append_backup_history(
                    data_dir,
                    {
                        "timestamp": ts,
                        "action": "backup",
                        "mode": "drive",
                        "file": lab,
                        "drive_name": lab,
                        "drive_id": "",
                        "result": "fail",
                        "details": str(res.get("error", "")),
                    },
                )

        append_event(
            data_dir=data_dir,
            season=season,
            owner=str(st.session_state.get("selected_owner") or ""),
            event_type="backup",
            summary=f"Backup Drive — ok:{ok_count} fail:{fail_count}",
            payload={"results": results},
        )

        if fail_count == 0:
            st.success(f"✅ Backup terminé — {ok_count} fichiers")
        else:
            st.warning(f"⚠️ Backup terminé — ok:{ok_count} fail:{fail_count}")
            st.dataframe(pd.DataFrame(results, columns=["file", "result"]), use_container_width=True, hide_index=True)


def _ui_local_restore(ctx: dict) -> None:
    data_dir = _data_dir(ctx)
    season = _season(ctx)

    st.subheader("📦 Restore local (fallback)")
    st.caption("Si Drive OAuth n’est pas prêt, tu peux uploader un CSV et choisir sa destination locale.")

    crit = _critical_files(data_dir, season)
    dest_names = [lab for lab, _ in crit] + ["(custom filename in data/)"]

    up = st.file_uploader("Uploader un CSV", type=["csv"], key="admin_local_restore_upload")

    dest_pick = st.selectbox("Restaurer vers (local)", dest_names, key="admin_local_restore_dest")
    custom = ""
    if dest_pick == "(custom filename in data/)":
        custom = st.text_input("Nom fichier destination (dans data/)", value="custom.csv", key="admin_local_restore_custom")

    if dest_pick == "(custom filename in data/)":
        dest_path = os.path.join(data_dir, custom.strip() or "custom.csv")
        dest_name = os.path.basename(dest_path)
    else:
        dest_path = dict(crit).get(dest_pick)  # type: ignore
        dest_name = dest_pick

    if st.button("💾 Restore local maintenant", type="primary", key="admin_local_restore_go"):
        if up is None:
            st.error("Uploader un fichier CSV d'abord.")
            return
        if not dest_path:
            st.error("Destination invalide.")
            return

        _ensure_parent(dest_path)
        try:
            content = up.getvalue()
            with open(dest_path, "wb") as f:
                f.write(content)

            ts = _now_iso()
            _append_backup_history(
                data_dir,
                {
                    "timestamp": ts,
                    "action": "restore",
                    "mode": "local",
                    "file": dest_name,
                    "drive_name": "",
                    "drive_id": "",
                    "result": "ok",
                    "details": "",
                },
            )
            append_event(
                data_dir=data_dir,
                season=season,
                owner=str(st.session_state.get("selected_owner") or ""),
                event_type="restore",
                summary=f"Restore local → {dest_name}",
                payload={"dest": dest_name},
            )
            st.success(f"✅ Restored local: `{dest_name}`")
            st.rerun()
        except Exception as e:
            st.error("❌ Restore local échoué")
            st.code(str(e))


# =========================
# Players DB Admin (hook)
# =========================
def _ui_players_db_admin(ctx: dict) -> None:
    st.subheader("🗂️ Players DB (Admin)")

    update_fn = ctx.get("update_players_db")  # fonction attendue
    if not callable(update_fn):
        st.warning("update_players_db introuvable. Assure-toi que `pms_enrich.py` expose `update_players_db` et que `app.py` le passe dans ctx.")
        st.caption("L’UI reste affichée, mais les boutons d’update/resume seront inactifs tant que la fonction n’est pas fournie.")
        return

    # Déléguer au module (si tu veux), sinon appeler direct
    # Ici: on appelle update_fn avec des flags st.session_state
    data_dir = _data_dir(ctx)
    season = _season(ctx)

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        roster_only = st.checkbox("⚡ Roster actif seulement", value=bool(st.session_state.get("pdb_roster_only", False)), key="pdb_roster_only")
    with colB:
        details = st.checkbox("Afficher détails", value=bool(st.session_state.get("pdb_details", False)), key="pdb_details")
    with colC:
        lock = st.checkbox("LOCK", value=bool(st.session_state.get("pdb_lock", False)), key="pdb_lock")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        reset_cache = st.button("🧹 Reset cache", key="pdb_reset_cache")
    with c2:
        reset_progress = st.button("🧹 Reset progress", key="pdb_reset_progress")
    with c3:
        reset_failed = st.button("🧽 Reset failed only", key="pdb_reset_failed")

    st.divider()

    colX, colY = st.columns(2)
    with colX:
        go_update = st.button("⬆️ Mettre à jour Players DB", type="primary", key="pdb_go_update")
    with colY:
        go_resume = st.button("▶️ Resume Country fill", key="pdb_go_resume")

    # Exécution
    if go_update or go_resume or reset_cache or reset_progress or reset_failed:
        try:
            res = update_fn(
                data_dir=data_dir,
                season=season,
                mode="resume" if go_resume else "update",
                roster_only=roster_only,
                details=details,
                lock=lock,
                reset_cache=reset_cache,
                reset_progress=reset_progress,
                reset_failed_only=reset_failed,
            )
            st.success("✅ Terminé.")
            st.json(res if isinstance(res, dict) else {"result": str(res)})

            append_event(
                data_dir=data_dir,
                season=season,
                owner=str(st.session_state.get("selected_owner") or ""),
                event_type="players_db",
                summary=f"Players DB: {'resume' if go_resume else 'update'}",
                payload=res if isinstance(res, dict) else {"result": str(res)},
            )
        except Exception as e:
            st.error("❌ Players DB error")
            st.code(str(e))


# =========================
# Main render
# =========================
def render(ctx: dict) -> None:
    st.header("🛠️ Gestion Admin")

    if not _is_admin(ctx):
        st.warning("Accès admin requis.")
        return

    data_dir = _data_dir(ctx)
    season = _season(ctx)
    os.makedirs(data_dir, exist_ok=True)

    # -------- Drive section
    with st.expander("☁️ Backups & Restore (Drive)", expanded=True):
        _ui_drive_restore(ctx)
        st.divider()
        _ui_drive_backup(ctx)
        st.divider()
        _ui_local_restore(ctx)

        with st.expander("🔎 Debug (paths)", expanded=False):
            st.write("Fichiers critiques attendus localement :")
            st.code("\n".join([p for _, p in _critical_files(data_dir, season)]))
            st.write("Event log:")
            st.code(event_log_path(data_dir, season))

    st.divider()

    # -------- Players DB
    with st.expander("🗂️ Players DB (Admin)", expanded=True):
        _ui_players_db_admin(ctx)
