# tabs/admin.py
from __future__ import annotations

import os
import io
import glob
import zipfile
import datetime as _dt
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd

# ------------------------------------------------------------
# Drive helpers (your project: services/drive.py)
# ------------------------------------------------------------
try:
    from services.drive import (
        drive_ready,
        get_drive_folder_id,
        drive_list_files,
        drive_upload_file,
        drive_download_file,
    )
except Exception:
    drive_ready = lambda: False  # type: ignore
    get_drive_folder_id = lambda: ""  # type: ignore
    drive_list_files = lambda *a, **k: []  # type: ignore
    drive_upload_file = lambda *a, **k: {"ok": False, "error": "services/drive.py not importable"}  # type: ignore
    drive_download_file = lambda *a, **k: {"ok": False, "error": "services/drive.py not importable"}  # type: ignore


# ============================================================
# Admin tab entrypoint expected by app.py
# ============================================================
def render(ctx: Dict[str, Any] | None = None) -> None:
    ctx = ctx or {}
    DATA_DIR = str(ctx.get("DATA_DIR") or ctx.get("data_dir") or "data")
    season_lbl = str(ctx.get("season_lbl") or ctx.get("season") or "2025-2026").strip() or "2025-2026"
    is_admin = bool(ctx.get("is_admin") or ctx.get("admin") or False)

    if not is_admin:
        st.warning("Accès admin requis.")
        st.stop()

    st.subheader("🛠️ Gestion Admin")

    # --- Mode selector
    mode = st.radio("", ["Backups", "Joueurs", "Outils"], horizontal=True, index=0)

    if mode == "Backups":
        _render_backups(DATA_DIR, season_lbl)
    elif mode == "Joueurs":
        _render_players_tools(DATA_DIR, season_lbl)
    else:
        _render_misc_tools(DATA_DIR, season_lbl)


# ============================================================
# Backups (Drive-first, local fallback)
# ============================================================
def _season_patterns(season_lbl: str) -> List[str]:
    s = season_lbl.replace("/", "-")
    return [
        f"*{s}*",
        f"*{s.replace('-', '_')}*",
    ]


def _collect_backup_files(data_dir: str, season_lbl: str) -> List[str]:
    data_dir = str(data_dir or "data")
    files: List[str] = []

    # Always include players DB if present
    for p in [
        os.path.join(data_dir, "hockey.players.csv"),
        os.path.join(data_dir, "Hockey.Players.csv"),
        os.path.join(data_dir, "players_master.csv"),
        os.path.join(data_dir, "puckpedia2025_26.csv"),
        os.path.join(data_dir, "settings.csv"),
    ]:
        if os.path.exists(p):
            files.append(p)

    # Include everything season-related
    pats = _season_patterns(season_lbl)
    for pat in pats:
        files.extend(glob.glob(os.path.join(data_dir, pat)))

    # De-dupe + only files
    out = []
    seen = set()
    for f in files:
        f = os.path.abspath(f)
        if f in seen:
            continue
        seen.add(f)
        if os.path.isfile(f):
            out.append(f)
    return sorted(out)


def _make_zip_bytes(file_paths: List[str], base_dir: str) -> bytes:
    base_dir = os.path.abspath(base_dir)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in file_paths:
            try:
                arc = os.path.relpath(fp, base_dir)
            except Exception:
                arc = os.path.basename(fp)
            z.write(fp, arcname=arc)
    return mem.getvalue()


def _render_backups(data_dir: str, season_lbl: str) -> None:
    st.markdown("### 📦 Backups complets (ZIP) — joueurs, alignements, transactions")

    # Folder ID (Drive)
    default_fid = ""
    try:
        default_fid = get_drive_folder_id() or ""
    except Exception:
        default_fid = ""

    fid = st.text_input("Folder ID Drive (backups)", value=default_fid, help="Dossier Google Drive où déposer les ZIP.")
    fid = (fid or "").strip()

    ok_drive = False
    try:
        ok_drive = bool(fid) and bool(drive_ready())
    except Exception:
        ok_drive = False

    if not ok_drive:
        st.warning("Drive OAuth non connecté (ou secrets manquants). Vérifie `st.secrets[gdrive_oauth]` + `gdrive_folder_id`.")
        st.caption("En attendant, tu peux quand même créer un ZIP local et le télécharger.")
    else:
        st.success("Drive prêt ✅ (refresh_token + scopes drive).")

    # Build list of candidate files
    files = _collect_backup_files(data_dir, season_lbl)
    with st.expander("📁 Fichiers inclus dans le backup", expanded=False):
        if not files:
            st.error("Aucun fichier trouvé à sauvegarder dans /data.")
        else:
            st.write(f"{len(files)} fichier(s) seront inclus:")
            for f in files:
                st.code(os.path.relpath(f, data_dir), language="text")

    # Create backup
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("📦 Créer un backup complet", use_container_width=True):
            if not files:
                st.error("Aucun fichier à zipper.")
            else:
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_name = f"backup_{season_lbl.replace('/','-')}_{ts}.zip"
                zip_bytes = _make_zip_bytes(files, data_dir)

                st.session_state["__last_backup_zip_name__"] = zip_name
                st.session_state["__last_backup_zip_bytes__"] = zip_bytes

                # Upload to drive if possible
                if ok_drive:
                    tmp_path = os.path.join(data_dir, f"__tmp__{zip_name}")
                    try:
                        os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)
                        with open(tmp_path, "wb") as f:
                            f.write(zip_bytes)
                        res = drive_upload_file(fid, tmp_path, drive_name=zip_name)
                        if res.get("ok"):
                            st.success(f"✅ Backup uploadé sur Drive: {zip_name}")
                        else:
                            st.error(f"❌ Upload Drive échoué: {res.get('error')}")
                    finally:
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                else:
                    st.success("✅ Backup ZIP créé (local en mémoire). Utilise le bouton de téléchargement à droite.")

    with colB:
        zip_name = st.session_state.get("__last_backup_zip_name__")
        zip_bytes = st.session_state.get("__last_backup_zip_bytes__")
        if zip_name and zip_bytes:
            st.download_button("⬇️ Télécharger le ZIP", data=zip_bytes, file_name=zip_name, mime="application/zip", use_container_width=True)
        else:
            st.info("Aucun backup créé dans cette session.")

    # List backups in Drive + restore
    st.markdown("---")
    st.markdown("### ♻️ Restaurer un backup (Drive)")

    if ok_drive:
        backups = drive_list_files(folder_id=fid, name_contains="backup_", limit=200)
        # Keep only .zip
        backups = [b for b in (backups or []) if str(b.get("name","")).lower().endswith(".zip")]
        backups = sorted(backups, key=lambda x: str(x.get("modifiedTime","")), reverse=True)

        if not backups:
            st.info("Aucun ZIP de backup trouvé dans ce dossier Drive.")
            return

        name_to_id = {b["name"]: b["id"] for b in backups if b.get("name") and b.get("id")}
        pick = st.selectbox("Choisir un backup à restaurer", options=list(name_to_id.keys()))
        confirm = st.checkbox("Je confirme le restore (écrase data/)")
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("⬇️ Télécharger ce ZIP (depuis Drive)", use_container_width=True):
                tmp = os.path.join(data_dir, "__tmp_download__.zip")
                res = drive_download_file(name_to_id[pick], tmp)
                if not res.get("ok"):
                    st.error(f"❌ Download échoué: {res.get('error')}")
                else:
                    try:
                        with open(tmp, "rb") as f:
                            b = f.read()
                        st.download_button("Téléchargement prêt", data=b, file_name=pick, mime="application/zip", use_container_width=True)
                    finally:
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass

        with col2:
            if st.button("♻️ Restaurer", use_container_width=True, disabled=not confirm):
                tmp = os.path.join(data_dir, "__tmp_restore__.zip")
                res = drive_download_file(name_to_id[pick], tmp)
                if not res.get("ok"):
                    st.error(f"❌ Download échoué: {res.get('error')}")
                else:
                    try:
                        _restore_zip_into_data(tmp, data_dir)
                        st.success("✅ Restore terminé. Redémarre l’app si nécessaire.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Restore échoué: {e}")
                    finally:
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
    else:
        st.info("Drive non prêt. Configure `gdrive_oauth` + folder id pour activer le restore Drive.")


def _restore_zip_into_data(zip_path: str, data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        # Extract into data_dir, preserving relative paths
        for member in z.infolist():
            # Prevent zip slip
            rel = member.filename.replace("\\", "/")
            rel = rel.lstrip("/")
            if ".." in rel.split("/"):
                continue
            dest = os.path.join(data_dir, rel)
            dest_dir = os.path.dirname(dest)
            os.makedirs(dest_dir, exist_ok=True)
            with z.open(member, "r") as src, open(dest, "wb") as out:
                out.write(src.read())


# ============================================================
# Players tools (manual-first)
# ============================================================
def _render_players_tools(data_dir: str, season_lbl: str) -> None:
    st.markdown("### 👤 Gestion manuelle des joueurs (sans import CSV)")

    st.info(
        "Ici, on travaille **sans import multi-CSV**. "
        "Tu ajoutes/retire/déplaces les joueurs directement dans tes fichiers saison."
    )

    # NOTE: This is a lightweight placeholder; your full project likely has richer logic.
    # We keep it simple and safe: edit `equipes_joueurs_{season}.csv` if present.

    teams_path = os.path.join(data_dir, f"equipes_joueurs_{season_lbl}.csv")
    if not os.path.exists(teams_path):
        st.warning(f"Fichier saison introuvable: {teams_path}")
        st.caption("Crée-le ou restaure un backup. Ensuite l’Admin pourra gérer les joueurs.")
        return

    df = _read_csv_safe(teams_path)
    if df.empty:
        st.warning("Le fichier équipes/joueurs est vide.")
    else:
        st.caption(f"Fichier: {teams_path} — {len(df)} lignes")

    with st.expander("➕ Ajouter joueur(s)", expanded=False):
        _add_players_ui(df, teams_path)

    with st.expander("🗑️ Retirer joueur(s)", expanded=False):
        _remove_players_ui(df, teams_path)

    with st.expander("🔁 Déplacer GC ↔ CE / Slot", expanded=False):
        _move_players_ui(df, teams_path)


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal columns used in many of your flows
    needed = ["Proprio", "Joueur", "Team", "Pos", "Slot", "Scope"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    return df


def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep=";", low_memory=False)
        except Exception:
            return pd.DataFrame()


def _write_csv_safe(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def _add_players_ui(df: pd.DataFrame, path: str) -> None:
    df = _ensure_cols(df.copy())
    teams = sorted([t for t in df["Proprio"].astype(str).unique().tolist() if t.strip()]) or [
        "Whalers","Red_Wings","Predateurs","Nordiques","Cracheurs","Canadiens"
    ]
    owner = st.selectbox("Équipe (Proprio)", teams)
    names = st.text_area("Noms des joueurs (1 par ligne)", height=120, placeholder="Connor McDavid\nSidney Crosby")
    slot = st.selectbox("Slot", ["Actif", "Banc", "Mineur", "IR"])
    scope = st.selectbox("Scope", ["GC", "CE"])

    if st.button("Ajouter", use_container_width=True):
        added = 0
        for raw in (names or "").splitlines():
            n = raw.strip()
            if not n:
                continue
            row = {"Proprio": owner, "Joueur": n, "Slot": slot, "Scope": scope}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            added += 1
        _write_csv_safe(df, path)
        st.success(f"✅ {added} joueur(s) ajouté(s).")
        st.rerun()


def _remove_players_ui(df: pd.DataFrame, path: str) -> None:
    df = _ensure_cols(df.copy())
    if df.empty:
        st.info("Aucun joueur à retirer.")
        return
    owner = st.selectbox("Équipe", sorted(df["Proprio"].astype(str).unique().tolist()))
    sub = df[df["Proprio"].astype(str) == str(owner)]
    picks = st.multiselect("Joueurs à retirer", sorted(sub["Joueur"].astype(str).unique().tolist()))
    confirm = st.checkbox("Je confirme la suppression")
    if st.button("Retirer", use_container_width=True, disabled=not confirm):
        before = len(df)
        df = df[~((df["Proprio"].astype(str) == str(owner)) & (df["Joueur"].astype(str).isin(picks)))]
        _write_csv_safe(df, path)
        st.success(f"✅ Retiré {before - len(df)} joueur(s).")
        st.rerun()


def _move_players_ui(df: pd.DataFrame, path: str) -> None:
    df = _ensure_cols(df.copy())
    if df.empty:
        st.info("Aucun joueur à déplacer.")
        return
    owner = st.selectbox("Équipe (move)", sorted(df["Proprio"].astype(str).unique().tolist()))
    sub = df[df["Proprio"].astype(str) == str(owner)]
    player = st.selectbox("Joueur", sorted(sub["Joueur"].astype(str).unique().tolist()))
    new_scope = st.selectbox("Nouveau Scope", ["GC", "CE"])
    new_slot = st.selectbox("Nouveau Slot", ["Actif", "Banc", "Mineur", "IR"])
    if st.button("Appliquer", use_container_width=True):
        mask = (df["Proprio"].astype(str) == str(owner)) & (df["Joueur"].astype(str) == str(player))
        df.loc[mask, "Scope"] = new_scope
        df.loc[mask, "Slot"] = new_slot
        _write_csv_safe(df, path)
        st.success("✅ Déplacement appliqué.")
        st.rerun()


# ============================================================
# Misc tools placeholder
# ============================================================
def _render_misc_tools(data_dir: str, season_lbl: str) -> None:
    st.markdown("### 🧰 Outils")
    st.caption("Garde cet espace pour tes outils avancés (fusion master, NHL_ID auto, etc.).")
    st.info("Si tu veux, je peux réintégrer tes outils existants ici **sans** replanter Streamlit (clés uniques, pas d’expander imbriqué).")
