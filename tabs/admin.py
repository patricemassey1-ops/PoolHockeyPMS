# tabs/admin.py — PoolHockeyPMS (Admin: Backups Drive + Outils Sync)
# ------------------------------------------------------------
# Compatible with app.py convention:
#   mod.render(ctx.as_dict()) OR mod.render(ctx)
# ------------------------------------------------------------

from __future__ import annotations

import os
import io
import glob
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

# ============================================================
# Drive helpers (services/drive.py)
# ============================================================
try:
    from services.drive import (
        drive_ready,
        get_drive_folder_id,
        drive_list_files,
        drive_upload_file,
        drive_download_file,
    )
except Exception:
    def drive_ready() -> bool:  # type: ignore
        return False

    def get_drive_folder_id() -> str:  # type: ignore
        return ""

    def drive_list_files(*a, **k):  # type: ignore
        return []

    def drive_upload_file(*a, **k):  # type: ignore
        return {"ok": False, "error": "services/drive.py introuvable"}

    def drive_download_file(*a, **k):  # type: ignore
        return {"ok": False, "error": "services/drive.py introuvable"}


# ============================================================
# ctx helpers (dict OR AppCtx object)
# ============================================================
def _ctx_s(ctx: Any, key: str, default: str = "") -> str:
    try:
        if isinstance(ctx, dict):
            return str(ctx.get(key, default) or default)
        return str(getattr(ctx, key, default) or default)
    except Exception:
        return default


def _ctx_b(ctx: Any, key: str, default: bool = False) -> bool:
    try:
        if isinstance(ctx, dict):
            return bool(ctx.get(key, default))
        return bool(getattr(ctx, key, default))
    except Exception:
        return default


# ============================================================
# ENTRYPOINT
# ============================================================
def render(ctx: Any = None) -> None:
    ctx = ctx or {}
    data_dir = _ctx_s(ctx, "DATA_DIR", _ctx_s(ctx, "data_dir", "data"))
    season = _ctx_s(ctx, "season_lbl", "2025-2026").strip() or "2025-2026"
    is_admin = _ctx_b(ctx, "is_admin", False)

    if not is_admin:
        st.warning("Accès admin requis.")
        st.stop()

    st.subheader("🛠️ Gestion Admin")

    tab = st.radio("", ["Backups", "Outils"], horizontal=True, index=1)
    if tab == "Backups":
        render_backups(data_dir, season)
    else:
        render_tools(data_dir, season)


# ============================================================
# BACKUPS (Drive)
# ============================================================
def render_backups(data_dir: str, season: str) -> None:
    st.markdown("### 📦 Backups complets (Google Drive)")

    folder_id = st.text_input("Folder ID Drive", value=(get_drive_folder_id() or "")).strip()
    drive_ok = bool(folder_id) and bool(drive_ready())

    if drive_ok:
        st.success("Drive prêt (OAuth OK).")
    else:
        st.warning("Drive non prêt (secrets manquants ou OAuth invalide).")

    files = collect_backup_files(data_dir, season)

    with st.expander("📁 Fichiers inclus", expanded=False):
        if not files:
            st.info("Aucun fichier trouvé pour ce backup.")
        for f in files:
            st.code(os.path.relpath(f, data_dir), language="text")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📦 Créer un backup complet", use_container_width=True):
            if not files:
                st.error("Aucun fichier à sauvegarder.")
            else:
                ts = dt.datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d_%H%M%S")
                zip_name = f"backup_{season}_{ts}.zip"
                zip_bytes = make_zip(files, data_dir)

                if drive_ok:
                    tmp = os.path.join(data_dir, f"__tmp__{zip_name}")
                    try:
                        os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
                        with open(tmp, "wb") as fh:
                            fh.write(zip_bytes)
                        res = drive_upload_file(folder_id, tmp, zip_name)
                        if res.get("ok"):
                            st.success(f"✅ Backup envoyé sur Drive : {zip_name}")
                        else:
                            st.error(f"❌ Upload Drive: {res.get('error')}")
                    finally:
                        try:
                            if os.path.exists(tmp):
                                os.remove(tmp)
                        except Exception:
                            pass
                else:
                    st.session_state["__last_zip_name__"] = zip_name
                    st.session_state["__last_zip_bytes__"] = zip_bytes
                    st.success("✅ Backup créé (local). Télécharge-le à droite.")

    with col2:
        zn = st.session_state.get("__last_zip_name__")
        zb = st.session_state.get("__last_zip_bytes__")
        if zn and zb:
            st.download_button("⬇️ Télécharger le ZIP", data=zb, file_name=zn, mime="application/zip", use_container_width=True)
        else:
            st.info("Aucun ZIP local prêt.")

    st.markdown("---")
    st.markdown("### ♻️ Restaurer depuis Drive")
    if not drive_ok:
        st.info("Connecte Drive (secrets) pour restaurer.")
        return

    backups = drive_list_files(folder_id, name_contains="backup_", limit=100) or []
    backups = [b for b in backups if str(b.get("name", "")).lower().endswith(".zip")]
    backups = sorted(backups, key=lambda x: str(x.get("modifiedTime", "")), reverse=True)

    if not backups:
        st.info("Aucun backup trouvé dans Drive.")
        return

    name_to_id = {b["name"]: b["id"] for b in backups if b.get("name") and b.get("id")}
    sel = st.selectbox("Choisir un backup", list(name_to_id.keys()))
    confirm = st.checkbox("Je confirme (écrase data/)", value=False)

    if st.button("♻️ Restaurer", disabled=not confirm, use_container_width=True):
        file_id = name_to_id.get(sel)
        if not file_id:
            st.error("Backup invalide.")
            return
        tmp = os.path.join(data_dir, "__restore__.zip")
        res = drive_download_file(file_id, tmp)
        if not res.get("ok"):
            st.error(f"❌ Download: {res.get('error')}")
            return
        try:
            restore_zip(tmp, data_dir)
            st.success("✅ Restore terminé.")
            st.rerun()
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass


def collect_backup_files(data_dir: str, season: str) -> List[str]:
    out: List[str] = []
    data_dir = str(data_dir or "data")

    # season-related files
    for f in glob.glob(os.path.join(data_dir, f"*{season}*")):
        if os.path.isfile(f):
            out.append(f)

    # core files
    for base in ["hockey.players.csv", "Hockey.Players.csv", "players_master.csv", "PuckPedia2025_26.csv", "puckpedia2025_26.csv", "settings.csv"]:
        p = os.path.join(data_dir, base)
        if os.path.exists(p) and os.path.isfile(p):
            out.append(p)

    # de-dupe
    return sorted(set(map(os.path.abspath, out)))


def make_zip(files: List[str], base_dir: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = os.path.relpath(f, base_dir) if base_dir else os.path.basename(f)
            z.write(f, arcname=arc)
    return mem.getvalue()


def restore_zip(zip_path: str, data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)


# ============================================================
# OUTILS
# ============================================================
def render_tools(data_dir: str, season: str) -> None:
    st.markdown("### 🧰 Outils")

    # ---------- Sync PuckPedia -> Level
    with st.expander("🧾 Sync PuckPedia → Level (STD/ELC)", expanded=False):
        puck = st.text_input("Fichier PuckPedia", os.path.join(data_dir, "PuckPedia2025_26.csv"))
        players = st.text_input("Players DB", os.path.join(data_dir, "hockey.players.csv"))

        colp1, colp2 = st.columns([1,1])
        with colp1:
            if st.button("👀 Voir colonnes PuckPedia", key="peek_pk_cols"):
                try:
                    pk0 = pd.read_csv(puck, nrows=0)
                    st.write(list(pk0.columns))
                except Exception as e:
                    st.error(f"Lecture PuckPedia échouée: {e}")
        with colp2:
            if st.button("👀 Voir colonnes Players DB", key="peek_pdb_cols"):
                try:
                    pdb0 = pd.read_csv(players, nrows=0)
                    st.write(list(pdb0.columns))
                except Exception as e:
                    st.error(f"Lecture Players DB échouée: {e}")

        if st.button("Synchroniser", use_container_width=False):
            res = sync_level(players, puck)
            if not isinstance(res, dict):
                st.error("Erreur interne: sync_level n'a pas retourné un résultat.")
                return

            if res.get("ok"):
                st.success(f"✅ Level synchronisé. Modifiés: {res.get('updated', 0)}")
            else:
                st.error(res.get("error", "Erreur."))
                if res.get("missing"):
                    st.caption("Chemins introuvables :")
                    for p in res["missing"]:
                        st.code(p, language="text")
                if res.get("players_cols") or res.get("puck_cols"):
                    st.markdown("**🔎 Diagnostic colonnes**")
                        if res.get("players_cols"):
                            st.write("Players DB colonnes:")
                            st.write(res["players_cols"])
                        if res.get("puck_cols"):
                            st.write("PuckPedia colonnes:")
                            st.write(res["puck_cols"])

    # ---------- Sync NHL_ID missing (progress)
    with st.expander("🪪 Sync NHL_ID manquants (progress)", expanded=False):
        players2 = st.text_input("Players DB", os.path.join(data_dir, "hockey.players.csv"), key="nhl_players_db")
        limit = st.number_input("Max par run", min_value=10, max_value=500, value=250, step=10)
        dry = st.checkbox("Dry-run (ne sauvegarde pas)", value=False)

        if st.button("Associer NHL_ID", use_container_width=False):
            res = fill_nhl_ids(players2, int(limit), dry_run=bool(dry), show_progress=True)
            if res.get("ok"):
                st.success(res.get("summary", "Terminé."))
            else:
                st.error(res.get("error", "Erreur."))


# ============================================================
# SYNC FUNCTIONS (safe returns)
# ============================================================


def _norm_player(s: str) -> str:
    s = str(s or "").strip()
    s = " ".join(s.split())
    return s
def sync_level(players_path: str, puck_path: str) -> Dict[str, Any]:
    players_path = str(players_path or "").strip()
    puck_path = str(puck_path or "").strip()

    missing: List[str] = []
    if not os.path.exists(players_path):
        missing.append(players_path)
    if not os.path.exists(puck_path):
        missing.append(puck_path)
    if missing:
        return {"ok": False, "error": "Fichier introuvable.", "missing": missing}

    try:
        pdb = pd.read_csv(players_path, low_memory=False)
        pk = pd.read_csv(puck_path, low_memory=False)
    except Exception as e:
        return {"ok": False, "error": f"Lecture CSV échouée: {e}"}

    # Detect name columns (players DB vs PuckPedia)
    name_pdb = None
    for c in ["Player", "Joueur", "Name", "Nom", "player_name"]:
        if c in pdb.columns:
            name_pdb = c
            break

    name_pk = None
    for c in ["Skaters", "Player", "Name", "Joueur", "player_name", "Nom"]:
        if c in pk.columns:
            name_pk = c
            break

    if not name_pdb or not name_pk:
        return {
            "ok": False,
            "error": "Colonne nom joueur introuvable dans un des fichiers.",
            "players_cols": list(pdb.columns),
            "puck_cols": list(pk.columns),
            "name_players_detected": name_pdb,
            "name_puck_detected": name_pk,
        }

    # Level column in puckpedia appears to be 'Level'
    level_pk = "Level" if "Level" in pk.columns else None
    if not level_pk:
        return {"ok": False, "error": "Colonne 'Level' introuvable dans PuckPedia.", "puck_cols": list(pk.columns)}

    if "Level" not in pdb.columns:
        pdb["Level"] = ""

    # build mapping Player -> Level (ELC/STD)
    mp: Dict[str, str] = {}
    for _, r in pk.iterrows():
        nm_raw = str(r.get(name_pk, "") or "").strip()
        nm = _norm_player(nm_raw)
        lv = str(r.get(level_pk, "") or "").strip().upper()
        if nm and lv in ("ELC", "STD"):
            mp[nm] = lv
            mp[nm_raw] = lv  # also keep raw

    if not mp:
        return {"ok": False, "error": "Aucun Level ELC/STD trouvé dans PuckPedia.", "puck_cols": list(pk.columns)}

    updated = 0
    for i, r in pdb.iterrows():
        nm_raw = str(r.get(name_pdb, "") or "").strip()
        if not nm_raw:
            continue
        nm = _norm_player(nm_raw)
        new_lv = mp.get(nm) or mp.get(nm_raw)
        if not new_lv:
            continue
        old_lv = str(r.get("Level", "") or "").strip().upper()
        if old_lv != new_lv:
            pdb.at[i, "Level"] = new_lv
            updated += 1

    try:
        pdb.to_csv(players_path, index=False)
    except Exception as e:
        return {"ok": False, "error": f"Écriture players DB échouée: {e}"}

    return {"ok": True, "updated": updated}


def fill_nhl_ids(players_path: str, limit: int, dry_run: bool = False, show_progress: bool = True) -> Dict[str, Any]:
    import requests

    players_path = str(players_path or "").strip()
    if not os.path.exists(players_path):
        return {"ok": False, "error": f"Players DB introuvable: {players_path}"}

    try:
        df = pd.read_csv(players_path, low_memory=False)
    except Exception as e:
        return {"ok": False, "error": f"Lecture players DB échouée: {e}"}

    # Your file uses 'Player' (per screenshot)
    if "Player" not in df.columns:
        return {"ok": False, "error": "Colonne 'Player' introuvable dans hockey.players.csv", "players_cols": list(df.columns)}

    if "NHL_ID" not in df.columns:
        df["NHL_ID"] = ""

    def _is_missing(v: Any) -> bool:
        s = str(v or "").strip()
        return (not s) or s.lower() == "nan" or s == "0"

    targets = df[df["NHL_ID"].apply(_is_missing)].head(int(limit))
    total = int(len(targets))
    if total == 0:
        return {"ok": True, "summary": "Aucun NHL_ID manquant à remplir.", "added": 0, "total": 0}

    prog = st.progress(0) if show_progress else None
    status = st.empty() if show_progress else None

    added = 0
    processed = 0

    for idx_num, (i, r) in enumerate(targets.iterrows(), start=1):
        name = str(r.get("Player", "") or "").strip()
        if not name:
            continue

        if show_progress and prog is not None and status is not None:
            prog.progress(min(1.0, idx_num / max(1, total)))
            status.caption(f"Traitement: {idx_num}/{total} — ajoutés: {added}")

        q = requests.utils.quote(name)
        url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={q}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json() or []
            if not data:
                continue
            pid = data[0].get("playerId")
            if pid:
                processed += 1
                added += 1
                if not dry_run:
                    df.at[i, "NHL_ID"] = int(pid)
        except Exception:
            continue

    if show_progress and prog is not None and status is not None:
        prog.progress(1.0)
        status.caption(f"Terminé ✅ — ajoutés: {added}/{total}")

    if not dry_run:
        try:
            df.to_csv(players_path, index=False)
        except Exception as e:
            return {"ok": False, "error": f"Écriture échouée: {e}"}

    return {
        "ok": True,
        "added": added,
        "processed": processed,
        "total": total,
        "summary": f"Terminé — ajoutés: {added}/{total}" + (" (dry-run)" if dry_run else ""),
    }
