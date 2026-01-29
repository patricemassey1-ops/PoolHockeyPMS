# tabs/admin.py
from __future__ import annotations

import os
import io
import glob
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo
from typing import Dict, Any, List

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
    def drive_ready(): return False
    def get_drive_folder_id(): return ""
    def drive_list_files(*a, **k): return []
    def drive_upload_file(*a, **k): return {"ok": False, "error": "drive service missing"}
    def drive_download_file(*a, **k): return {"ok": False, "error": "drive service missing"}


# ============================================================
# ENTRY POINT (called by app.py)
# ============================================================
def render(ctx: Dict[str, Any] | None = None) -> None:
    ctx = ctx or {}
    data_dir = str(ctx.get("DATA_DIR") or "data")
    season = str(ctx.get("season_lbl") or "2025-2026")
    is_admin = bool(ctx.get("is_admin"))

    if not is_admin:
        st.warning("Accès admin requis.")
        st.stop()

    st.subheader("🛠️ Gestion Admin")

    tab = st.radio("", ["Backups", "Outils"], horizontal=True)

    if tab == "Backups":
        render_backups(data_dir, season)
    else:
        render_tools(data_dir, season)


# ============================================================
# BACKUPS
# ============================================================
def render_backups(data_dir: str, season: str) -> None:
    st.markdown("### 📦 Backups complets (Google Drive)")

    folder_id = st.text_input(
        "Folder ID Drive",
        value=get_drive_folder_id() or "",
    ).strip()

    drive_ok = bool(folder_id) and drive_ready()

    if drive_ok:
        st.success("Drive prêt (OAuth OK).")
    else:
        st.warning("Drive non prêt (secrets manquants ou OAuth invalide).")

    files = collect_backup_files(data_dir, season)

    with st.expander("📁 Fichiers inclus", expanded=False):
        for f in files:
            st.code(os.path.relpath(f, data_dir))

    if st.button("📦 Créer un backup complet"):
        if not files:
            st.error("Aucun fichier à sauvegarder.")
        else:
            ts = dt.datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%d_%H%M%S")
            zip_name = f"backup_{season}_{ts}.zip"
            zip_bytes = make_zip(files, data_dir)

            if drive_ok:
                tmp = os.path.join(data_dir, zip_name)
                with open(tmp, "wb") as fh:
                    fh.write(zip_bytes)
                res = drive_upload_file(folder_id, tmp, zip_name)
                os.remove(tmp)
                if res.get("ok"):
                    st.success(f"Backup envoyé sur Drive : {zip_name}")
                else:
                    st.error(res.get("error"))
            else:
                st.download_button("Télécharger le ZIP", zip_bytes, zip_name)

    st.markdown("---")
    st.markdown("### ♻️ Restaurer depuis Drive")

    if drive_ok:
        backups = drive_list_files(folder_id, name_contains="backup_", limit=50)
        backups = [b for b in backups if b.get("name","").endswith(".zip")]
        names = [b["name"] for b in backups]

        if names:
            sel = st.selectbox("Choisir un backup", names)
            confirm = st.checkbox("Je confirme (écrase data/)")
            if st.button("♻️ Restaurer", disabled=not confirm):
                fid = next(b["id"] for b in backups if b["name"] == sel)
                tmp = os.path.join(data_dir, "__restore__.zip")
                res = drive_download_file(fid, tmp)
                if res.get("ok"):
                    restore_zip(tmp, data_dir)
                    os.remove(tmp)
                    st.success("Restore terminé.")
                    st.rerun()
                else:
                    st.error(res.get("error"))
        else:
            st.info("Aucun backup trouvé.")


def collect_backup_files(data_dir: str, season: str) -> List[str]:
    out = []
    for f in glob.glob(os.path.join(data_dir, f"*{season}*")):
        if os.path.isfile(f):
            out.append(f)
    for base in ["hockey.players.csv", "puckpedia2025_26.csv"]:
        p = os.path.join(data_dir, base)
        if os.path.exists(p):
            out.append(p)
    return sorted(set(out))


def make_zip(files: List[str], base_dir: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.relpath(f, base_dir))
    return mem.getvalue()


def restore_zip(zip_path: str, data_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)


# ============================================================
# OUTILS
# ============================================================
def render_tools(data_dir: str, season: str) -> None:
    st.markdown("### 🧰 Outils")

    with st.expander("🧾 Sync PuckPedia → Level (STD/ELC)", expanded=False):
        puck = st.text_input("Fichier PuckPedia", os.path.join(data_dir, "puckpedia2025_26.csv"))
        players = st.text_input("Players DB", os.path.join(data_dir, "hockey.players.csv"))
        if st.button("Synchroniser"):
            res = sync_level(players, puck)
            if res["ok"]:
                st.success(f"Modifiés: {res['updated']}")
            else:
                st.error(res["error"])

    with st.expander("🪪 Sync NHL_ID manquants", expanded=False):
        players = st.text_input("Players DB", os.path.join(data_dir, "hockey.players.csv"), key="nhl_players")
        limit = st.number_input("Max par run", 10, 500, 250)
        if st.button("Associer NHL_ID"):
            res = fill_nhl_ids(players, int(limit))
            if res["ok"]:
                st.success(f"Ajoutés: {res['added']}")
            else:
                st.error(res["error"])


# ============================================================
# SAFE HELPERS (no top-level loops)
# ============================================================
def detect_name_col(df: pd.DataFrame) -> str | None:
    for c in ["Joueur", "Player", "Name", "player_name", "Nom"]:
        if c in df.columns:
            return c
    return None


def sync_level(players_path: str, puck_path: str) -> Dict[str, Any]:
    if not os.path.exists(players_path) or not os.path.exists(puck_path):
        return {"ok": False, "error": "Fichier manquant."}

    pdb = pd.read_csv(players_path)
    pk = pd.read_csv(puck_path)

    name_pk = detect_name_col(pk)
    name_pdb = detect_name_col(pdb)

    if not name_pk or not name_pdb:
        return {"ok": False, "error": "Colonne nom joueur introuvable."}

    if "Level" not in pdb.columns:
        pdb["Level"] = ""

    mp = {}
    for _, r in pk.iterrows():
        nm = str(r[name_pk]).strip().lower()
        lv = str(r.get("Level","")).upper()
        if lv in ("ELC","STD"):
            mp[nm] = lv

    updated = 0
    for i, r in pdb.iterrows():
        nm = str(r[name_pdb]).strip().lower()
        if nm in mp and pdb.at[i,"Level"] != mp[nm]:
            pdb.at[i,"Level"] = mp[nm]
            updated += 1

    pdb.to_csv(players_path, index=False)
    return {"ok": True, "updated": updated}


def fill_nhl_ids(players_path: str, limit: int) -> Dict[str, Any]:
    import requests
    if not os.path.exists(players_path):
        return {"ok": False, "error": "Players DB introuvable."}

    df = pd.read_csv(players_path)
    if "NHL_ID" not in df.columns:
        df["NHL_ID"] = ""

    name_col = detect_name_col(df)
    if not name_col:
        return {"ok": False, "error": "Colonne nom introuvable."}

    added = 0
    for i, r in df[df["NHL_ID"].isna() | (df["NHL_ID"]=="")].head(limit).iterrows():
        q = requests.utils.quote(str(r[name_col]))
        url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={q}"
        try:
            res = requests.get(url, timeout=10).json()
            if res and "playerId" in res[0]:
                df.at[i,"NHL_ID"] = int(res[0]["playerId"])
                added += 1
        except Exception:
            pass

    df.to_csv(players_path, index=False)
    return {"ok": True, "added": added}
