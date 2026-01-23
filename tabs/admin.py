import os
import streamlit as st
from services.storage import path_players_db, path_roster, path_backup_history
from services.drive import drive_ready, drive_list_files, drive_download_file, drive_upload_file

def render(ctx: dict) -> None:
    st.header("🛠️ Gestion Admin")
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    folder_id = ctx.get("drive_folder_id", "")
    season = ctx.get("season")

    st.caption("Drive folder: My Drive / PMS Pool Data / PoolHockeyData")
    st.code(f"folder_id = {folder_id or '(missing)'}")

    targets = {
        "Players DB (data/hockey.players.csv)": path_players_db(),
        f"Roster (equipes_joueurs_{season}.csv)": path_roster(season),
        "Backup history (backup_history.csv)": path_backup_history(),
    }

    st.subheader("☁️ Drive — Restore selected CSV")
    if not drive_ready():
        st.warning("Drive OAuth non prêt. Ajoute [gdrive_oauth] + gdrive_folder_id dans Secrets.")
        return

    filter_text = st.text_input("Filtre (nom contient)", value=".csv", key="drive_filter")
    if st.button("🔄 Refresh Drive list"):
        st.session_state.pop("drive_files_cache", None)

    if "drive_files_cache" not in st.session_state:
        st.session_state["drive_files_cache"] = drive_list_files(folder_id, name_contains=filter_text.strip())

    files = st.session_state.get("drive_files_cache", []) or []
    if not files:
        st.info("Aucun fichier Drive trouvé.")
        return

    labels = []
    id_by_label = {}
    for f in files[:200]:
        name = f.get("name","")
        mid = f.get("modifiedTime","")
        size = f.get("size","")
        label = f"{name} — {mid} — {size}"
        labels.append(label)
        id_by_label[label] = f.get("id")

    c1, c2 = st.columns([1.2, 1.2])
    with c1:
        pick = st.selectbox("Drive file", [""] + labels, key="drive_pick")
    with c2:
        tgt = st.selectbox("Target", list(targets.keys()), key="drive_target")

    if st.button("⬇️ Restore → target", type="primary"):
        if not pick:
            st.warning("Choisis un fichier.")
            return
        fid = id_by_label.get(pick, "")
        dest = targets.get(tgt, "")
        res = drive_download_file(fid, dest)
        if res.get("ok"):
            st.success(f"Restore OK → {dest}")
        else:
            st.error(res.get("error") or "Restore failed")

    st.subheader("⬆️ Upload local file to Drive (optional)")
    local_path = st.text_input("Local path to upload", value="")
    if st.button("⬆️ Upload"):
        if not local_path:
            st.warning("Donne un path local.")
            return
        res = drive_upload_file(folder_id, local_path)
        if res.get("ok"):
            st.success(f"Upload OK: {res.get('name')}")
        else:
            st.error(res.get("error") or "Upload failed")
