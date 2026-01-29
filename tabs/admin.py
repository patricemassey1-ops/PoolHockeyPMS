# tabs/admin.py — PoolHockeyPMS Admin (Backups / Joueurs / Outils)
# ---------------------------------------------------------------
# ✅ Compatible with app.py that calls mod.render(ctx.as_dict()) first.
# ✅ No nested expanders (Streamlit restriction).
# ✅ Backups Google Drive (services/drive.py) + auto-backup 2x/day (00:00 / 12:00 America/Toronto)
# ✅ Joueurs: ajout/retrait/déplacement GC<->CE via un fichier roster saison (long format)
# ✅ Outils: PuckPedia -> Level (STD/ELC), NHL_ID sync (NHL free API) + audit + exports
# ---------------------------------------------------------------

from __future__ import annotations

import os
import io
import json
import time
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd

# ===== Teams (doit matcher app.py) =====
POOL_TEAMS = [
    "Whalers",
    "Red_Wings",
    "Predateurs",
    "Nordiques",
    "Cracheurs",
    "Canadiens",
]

TZ = ZoneInfo("America/Toronto")


# =====================================================
# Drive helpers (services/drive.py)
# =====================================================
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


# =====================================================
# ctx helpers (ctx is dict from ctx.as_dict())
# =====================================================
def _get(ctx: Dict[str, Any], key: str, default=None):
    try:
        return ctx.get(key, default)
    except Exception:
        return default


def _norm_player(s: str) -> str:
    s = str(s or "").strip()
    s = " ".join(s.split())
    return s


def _is_missing_id(v: Any) -> bool:
    s = str(v or "").strip()
    if not s or s.lower() == "nan":
        return True
    if s == "0":
        return True
    return False


def _data_path(data_dir: str, fname: str) -> str:
    return os.path.join(str(data_dir or "data"), fname)


# =====================================================
# Files
# =====================================================
def players_db_path(data_dir: str) -> str:
    return _data_path(data_dir, "hockey.players.csv")


def puckpedia_path(data_dir: str) -> str:
    return _data_path(data_dir, "PuckPedia2025_26.csv")


def roster_path(data_dir: str, season: str) -> str:
    season = str(season or "").strip() or "season"
    return _data_path(data_dir, f"roster_{season}.csv")


def autobackup_state_path(data_dir: str, season: str) -> str:
    season = str(season or "").strip() or "season"
    return _data_path(data_dir, f".autobackup_state_{season}.json")


# =====================================================
# CSV IO helpers
# =====================================================
def load_csv(path: str) -> Tuple[pd.DataFrame, str | None]:
    if not path or not os.path.exists(path):
        return pd.DataFrame(), f"Fichier introuvable: {path}"
    try:
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Lecture CSV échouée: {e}"


def save_csv(df: pd.DataFrame, path: str) -> str | None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        return None
    except Exception as e:
        return f"Écriture CSV échouée: {e}"


# =====================================================
# BACKUPS — build zip from key files
# =====================================================
def collect_backup_files(data_dir: str, season: str) -> List[str]:
    data_dir = str(data_dir or "data")
    season = str(season or "").strip() or "season"
    out: List[str] = []

    candidates = [
        players_db_path(data_dir),
        puckpedia_path(data_dir),
        roster_path(data_dir, season),
        _data_path(data_dir, f"equipes_joueurs_{season}.csv"),
        _data_path(data_dir, f"transactions_{season}.csv"),
        _data_path(data_dir, f"trade_market_{season}.csv"),
        _data_path(data_dir, f"backup_history_{season}.csv"),
        _data_path(data_dir, "backup_history.csv"),
        _data_path(data_dir, "settings.csv"),
    ]

    for p in candidates:
        if p and os.path.exists(p) and os.path.isfile(p):
            out.append(os.path.abspath(p))

    # include any season-tagged csv (safe)
    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            if season in fn and fn.lower().endswith(".csv"):
                p = os.path.abspath(os.path.join(data_dir, fn))
                if os.path.isfile(p):
                    out.append(p)

    return sorted(set(out))


def make_zip_bytes(files: List[str], base_dir: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arc = os.path.relpath(f, base_dir) if base_dir else os.path.basename(f)
            z.write(f, arcname=arc)
    return mem.getvalue()


def restore_zip_bytes(zip_bytes: bytes, data_dir: str) -> str | None:
    try:
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            z.extractall(data_dir)
        return None
    except Exception as e:
        return str(e)


# =====================================================
# AUTO BACKUP 2x/day (00:00 and 12:00 Toronto)
# Streamlit cannot run background jobs reliably; triggers when Admin is opened.
# =====================================================
def should_autobackup(now: dt.datetime, last_run_iso: str | None) -> bool:
    target_hours = {0, 12}
    if now.hour not in target_hours:
        return False
    if now.minute > 20:
        return False

    if not last_run_iso:
        return True

    try:
        last = dt.datetime.fromisoformat(last_run_iso)
        if last.date() == now.date() and last.hour == now.hour:
            return False
        return True
    except Exception:
        return True


def load_autobackup_state(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def save_autobackup_state(path: str, state: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def run_autobackup_if_due(data_dir: str, season: str, folder_id: str) -> Dict[str, Any]:
    now = dt.datetime.now(TZ)
    sp = autobackup_state_path(data_dir, season)
    state = load_autobackup_state(sp)
    last_run = state.get("last_run_iso")

    if not should_autobackup(now, last_run):
        return {"ok": True, "skipped": True, "reason": "Not in schedule window."}

    files = collect_backup_files(data_dir, season)
    if not files:
        return {"ok": False, "error": "Aucun fichier à sauvegarder."}

    zip_name = f"backup_{season}_{now.strftime('%Y%m%d_%H%M%S')}.zip"
    zip_bytes = make_zip_bytes(files, data_dir)

    tmp = _data_path(data_dir, f"__tmp__{zip_name}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(zip_bytes)
        res = drive_upload_file(folder_id, tmp, zip_name)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error", "Upload Drive échoué.")}
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    state["last_run_iso"] = now.isoformat()
    state["last_zip_name"] = zip_name
    save_autobackup_state(sp, state)
    return {"ok": True, "skipped": False, "zip_name": zip_name}


# =====================================================
# JOUEURS — roster file (long format)
# =====================================================
ROSTER_COLS = ["season", "owner", "bucket", "slot", "player"]


def ensure_roster_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ROSTER_COLS)
    for c in ROSTER_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def roster_load(data_dir: str, season: str) -> pd.DataFrame:
    p = roster_path(data_dir, season)
    if not os.path.exists(p):
        return pd.DataFrame(columns=ROSTER_COLS)
    df, _ = load_csv(p)
    return ensure_roster_schema(df)


def roster_save(df: pd.DataFrame, data_dir: str, season: str) -> str | None:
    p = roster_path(data_dir, season)
    return save_csv(df, p)


# =====================================================
# OUTILS — PuckPedia Level + NHL_ID
# =====================================================
def sync_level(players_path: str, puck_path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "updated": 0}

    missing = []
    if not os.path.exists(players_path):
        missing.append(players_path)
    if not os.path.exists(puck_path):
        missing.append(puck_path)
    if missing:
        result["error"] = "Fichier introuvable"
        result["missing"] = missing
        return result

    try:
        pdb = pd.read_csv(players_path, low_memory=False)
        pk = pd.read_csv(puck_path, low_memory=False)
    except Exception as e:
        result["error"] = f"Lecture CSV échouée: {e}"
        return result

    name_pdb = next((c for c in ["Player", "Joueur", "Name"] if c in pdb.columns), None)
    name_pk = next((c for c in ["Skaters", "Player", "Name"] if c in pk.columns), None)

    if not name_pdb or not name_pk:
        result["error"] = "Colonne joueur introuvable"
        result["players_cols"] = list(pdb.columns)
        result["puck_cols"] = list(pk.columns)
        return result

    if "Level" not in pk.columns:
        result["error"] = "Colonne Level manquante dans PuckPedia"
        result["puck_cols"] = list(pk.columns)
        return result

    if "Level" not in pdb.columns:
        pdb["Level"] = ""

    mp: Dict[str, str] = {}
    for _, r in pk.iterrows():
        nm = _norm_player(r.get(name_pk))
        lv = str(r.get("Level") or "").strip().upper()
        if nm and lv in ("ELC", "STD"):
            mp[nm] = lv

    if not mp:
        result["error"] = "Aucun Level ELC/STD trouvé dans PuckPedia"
        return result

    updated = 0
    new_col = []
    for _, r in pdb.iterrows():
        nm = _norm_player(r.get(name_pdb))
        new_lv = mp.get(nm)
        old_lv = str(r.get("Level") or "").strip().upper()
        if new_lv and old_lv != new_lv:
            updated += 1
            new_col.append(new_lv)
        else:
            new_col.append(r.get("Level"))

    pdb["Level"] = new_col
    err = save_csv(pdb, players_path)
    if err:
        result["error"] = err
        return result

    result["ok"] = True
    result["updated"] = updated
    return result


def audit_nhl_ids(df: pd.DataFrame) -> Dict[str, Any]:
    if "NHL_ID" not in df.columns:
        return {"total": len(df), "filled": 0, "missing": len(df), "missing_pct": 100.0, "duplicates": 0}

    missing_mask = df["NHL_ID"].apply(_is_missing_id)
    missing = int(missing_mask.sum())
    total = int(len(df))
    filled = total - missing
    pct = (missing / total * 100.0) if total else 0.0

    ids = df.loc[~missing_mask, "NHL_ID"].astype(str).str.strip()
    dup_count = int(ids.duplicated().sum())

    return {"total": total, "filled": filled, "missing": missing, "missing_pct": pct, "duplicates": dup_count}


def _nhl_search_player_id(name: str, session, timeout: int = 10) -> int | None:
    import requests
    q = requests.utils.quote(name)
    url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={q}"
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    data = r.json() or []
    if not data:
        return None
    pid = data[0].get("playerId")
    try:
        return int(pid)
    except Exception:
        return None


def sync_nhl_id(players_path: str, limit: int = 250, dry_run: bool = False) -> Dict[str, Any]:
    if not os.path.exists(players_path):
        return {"ok": False, "error": f"Players DB introuvable: {players_path}"}

    try:
        df = pd.read_csv(players_path, low_memory=False)
    except Exception as e:
        return {"ok": False, "error": f"Lecture players DB échouée: {e}"}

    if df.empty:
        return {"ok": False, "error": "Players DB vide."}

    if "Player" not in df.columns:
        return {"ok": False, "error": "Colonne 'Player' introuvable", "players_cols": list(df.columns)}

    if "NHL_ID" not in df.columns:
        df["NHL_ID"] = ""

    missing_mask = df["NHL_ID"].apply(_is_missing_id)
    targets = df[missing_mask].head(int(limit))
    total = int(len(targets))
    if total == 0:
        a = audit_nhl_ids(df)
        return {"ok": True, "added": 0, "total": 0, "audit": a, "summary": "✅ Aucun NHL_ID manquant."}

    bar = st.progress(0)
    txt = st.empty()

    import requests
    session = requests.Session()

    added = 0
    for n, (idx, row) in enumerate(targets.iterrows(), start=1):
        name = _norm_player(row.get("Player"))
        if not name:
            continue

        bar.progress(min(1.0, n / max(1, total)))
        txt.caption(f"Recherche NHL_ID: {n}/{total} — ajoutés: {added}")

        pid = None
        try:
            pid = _nhl_search_player_id(name, session=session, timeout=10)
        except Exception:
            pid = None

        if pid:
            added += 1
            if not dry_run:
                df.at[idx, "NHL_ID"] = pid

        time.sleep(0.05)

    bar.progress(1.0)
    txt.caption(f"Terminé ✅ — ajoutés: {added}/{total}" + (" (dry-run)" if dry_run else ""))

    if not dry_run:
        err = save_csv(df, players_path)
        if err:
            return {"ok": False, "error": err}

    a = audit_nhl_ids(df)
    return {"ok": True, "added": added, "total": total, "audit": a, "summary": f"✅ Terminé — ajoutés: {added}/{total}" + (" (dry-run)" if dry_run else "")}


# =====================================================
# UI — main
# =====================================================
def render(ctx: Dict[str, Any]) -> None:
    data_dir = str(_get(ctx, "DATA_DIR", "data"))
    season = str(_get(ctx, "season_lbl", "2025-2026")).strip() or "2025-2026"
    is_admin = bool(_get(ctx, "is_admin", False))

    st.title("🛠️ Gestion Admin")

    if not is_admin:
        st.warning("Accès admin requis.")
        return

    tab = st.radio("", ["Backups", "Joueurs", "Outils"], horizontal=True, index=2)

    if tab == "Backups":
        render_backups(data_dir, season)
    elif tab == "Joueurs":
        render_roster_admin(data_dir, season)
    else:
        render_tools(data_dir, season)


# =====================================================
# UI — Backups
# =====================================================
def render_backups(data_dir: str, season: str) -> None:
    st.subheader("📦 Backups complets (Google Drive)")

    folder_default = (get_drive_folder_id() or "").strip()
    folder_id = st.text_input("Folder ID Drive (backups)", value=folder_default).strip()

    drive_ok = bool(folder_id) and bool(drive_ready())
    if drive_ok:
        st.success("Drive OAuth prêt.")
    else:
        st.warning("Drive non prêt (secrets OAuth manquants / invalides).")

    st.markdown("### ⏱️ Auto-backup (00:00 et 12:00 — America/Toronto)")
    st.caption("Streamlit ne peut pas exécuter en arrière-plan: l’auto-backup se déclenche quand tu ouvres l’onglet Admin (dans la fenêtre horaire).")

    if drive_ok:
        auto = run_autobackup_if_due(data_dir, season, folder_id)
        if auto.get("ok") and not auto.get("skipped"):
            st.success(f"✅ Auto-backup envoyé: {auto.get('zip_name')}")
        elif auto.get("ok") and auto.get("skipped"):
            st.info("Auto-backup: rien à faire maintenant.")
        else:
            st.error(f"Auto-backup: {auto.get('error', 'Erreur')}")
    else:
        st.info("Auto-backup nécessite Drive (OAuth) + folder_id.")

    st.markdown("---")

    files = collect_backup_files(data_dir, season)
    st.markdown("### 🧾 Fichiers inclus")
    if not files:
        st.info("Aucun fichier trouvé pour ce backup.")
    else:
        st.write([os.path.relpath(f, data_dir) for f in files])

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("📦 Créer un backup complet", use_container_width=True):
            if not files:
                st.error("Aucun fichier à sauvegarder.")
            else:
                now = dt.datetime.now(TZ)
                zip_name = f"backup_{season}_{now.strftime('%Y%m%d_%H%M%S')}.zip"
                zip_bytes = make_zip_bytes(files, data_dir)

                if drive_ok:
                    tmp = _data_path(data_dir, f"__tmp__{zip_name}")
                    try:
                        with open(tmp, "wb") as fh:
                            fh.write(zip_bytes)
                        res = drive_upload_file(folder_id, tmp, zip_name)
                        if res.get("ok"):
                            st.success(f"✅ Backup envoyé sur Drive: {zip_name}")
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
        st.info("Connecte Drive (OAuth) pour restaurer.")
        return

    backups = drive_list_files(folder_id, name_contains="backup_", limit=200) or []
    backups = [b for b in backups if str(b.get("name", "")).lower().endswith(".zip")]
    backups = sorted(backups, key=lambda x: str(x.get("modifiedTime", "")), reverse=True)

    if not backups:
        st.info("Aucun backup trouvé dans Drive.")
        return

    name_to_id = {b["name"]: b["id"] for b in backups if b.get("name") and b.get("id")}
    sel = st.selectbox("Choisir un backup à restaurer", list(name_to_id.keys()))
    confirm = st.checkbox("Je confirme le restore (écrase data/)", value=False)

    if st.button("♻️ Restaurer", disabled=not confirm, use_container_width=True):
        file_id = name_to_id.get(sel)
        if not file_id:
            st.error("Backup invalide.")
            return

        tmp = _data_path(data_dir, "__restore__.zip")
        res = drive_download_file(file_id, tmp)
        if not res.get("ok"):
            st.error(f"❌ Download: {res.get('error')}")
            return

        try:
            with open(tmp, "rb") as fh:
                zip_bytes = fh.read()
            err = restore_zip_bytes(zip_bytes, data_dir)
            if err:
                st.error(f"❌ Restore échoué: {err}")
            else:
                st.success("✅ Restore terminé.")
                st.rerun()
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


# =====================================================
# UI — Joueurs (roster admin)
# =====================================================
def render_roster_admin(data_dir: str, season: str) -> None:
    st.subheader("👥 Joueurs — gestion roster (manuel)")

    st.caption(
        "Assigne des joueurs à une équipe + bucket/slot. "
        f"Fichier: {os.path.basename(roster_path(data_dir, season))}"
    )

    pdb_path = players_db_path(data_dir)
    pdb, err = load_csv(pdb_path)
    if err:
        st.error(err)
        return
    if "Player" not in pdb.columns:
        st.error("Colonne 'Player' introuvable dans hockey.players.csv")
        st.write(list(pdb.columns))
        return

    roster = roster_load(data_dir, season)

    st.markdown("### ➕ Ajouter joueur(s)")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    with c1:
        options = sorted({_norm_player(x) for x in pdb["Player"].dropna().astype(str).tolist() if str(x).strip()})
        picked = st.multiselect("Joueur(s)", options=options, max_selections=25)
    with c2:
        owner = st.selectbox("Équipe", options=POOL_TEAMS, index=0)
    with c3:
        bucket = st.selectbox("Bucket", options=["GC", "CE"], index=0)
    with c4:
        slot = st.selectbox("Slot", options=["Actif", "Banc", "IR", "Mineur"], index=0)

    if st.button("✅ Ajouter au roster"):
        if not picked:
            st.warning("Choisis au moins 1 joueur.")
        else:
            add_rows = []
            existing_keys = set(
                (str(r.get("season")), str(r.get("owner")), str(r.get("bucket")), str(r.get("slot")), _norm_player(r.get("player")))
                for _, r in roster.iterrows()
            )
            for p in picked:
                key = (season, owner, bucket, slot, _norm_player(p))
                if key in existing_keys:
                    continue
                add_rows.append({"season": season, "owner": owner, "bucket": bucket, "slot": slot, "player": _norm_player(p)})

            if add_rows:
                roster2 = pd.concat([roster, pd.DataFrame(add_rows)], ignore_index=True)
                err2 = roster_save(roster2, data_dir, season)
                if err2:
                    st.error(err2)
                else:
                    st.success(f"Ajoutés: {len(add_rows)}")
                    st.rerun()
            else:
                st.info("Aucun nouvel ajout (déjà présents).")

    st.markdown("---")

    st.markdown("### 🔁 Déplacer GC ↔ CE")
    c5, c6 = st.columns([2, 1])
    with c5:
        roster_owner = st.selectbox("Équipe (roster)", options=POOL_TEAMS, key="roster_owner_filter")
    with c6:
        players_for_owner = roster.loc[roster["owner"].astype(str) == roster_owner, "player"].astype(str).tolist()
        players_for_owner = sorted({_norm_player(x) for x in players_for_owner if str(x).strip()})
        move_player = st.selectbox("Joueur", options=[""] + players_for_owner, key="move_player")

    if st.button("Basculer GC↔CE"):
        if not move_player:
            st.warning("Choisis un joueur.")
        else:
            mask = (
                (roster["season"].astype(str) == season)
                & (roster["owner"].astype(str) == roster_owner)
                & (roster["player"].astype(str).apply(_norm_player) == _norm_player(move_player))
            )
            if int(mask.sum()) == 0:
                st.info("Joueur non trouvé dans le roster.")
            else:
                def _toggle(x):
                    x = str(x or "").strip().upper()
                    return "CE" if x == "GC" else "GC"
                roster.loc[mask, "bucket"] = roster.loc[mask, "bucket"].apply(_toggle)
                err3 = roster_save(roster, data_dir, season)
                if err3:
                    st.error(err3)
                else:
                    st.success("Déplacement effectué.")
                    st.rerun()

    st.markdown("---")

    st.markdown("### 🗑️ Retirer joueur(s)")
    roster_f = roster.copy()
    if not roster_f.empty:
        roster_f["player"] = roster_f["player"].astype(str).apply(_norm_player)

    filt_owner = st.selectbox("Filtrer par équipe", options=["(Toutes)"] + POOL_TEAMS, index=0, key="rm_owner")
    if filt_owner != "(Toutes)":
        roster_f = roster_f[roster_f["owner"].astype(str) == filt_owner]

    st.dataframe(roster_f.sort_values(["owner", "bucket", "slot", "player"]), use_container_width=True, height=360)

    to_remove = st.multiselect(
        "Sélectionne les lignes à retirer (format: index :: owner | bucket | slot | player)",
        options=[f"{i} :: {r['owner']} | {r['bucket']} | {r['slot']} | {r['player']}" for i, r in roster_f.iterrows()],
        max_selections=200,
    )
    confirm = st.checkbox("Je confirme la suppression", value=False)

    if st.button("🗑️ Supprimer", disabled=not (confirm and bool(to_remove))):
        idxs = []
        for s in to_remove:
            try:
                idxs.append(int(str(s).split("::")[0].strip()))
            except Exception:
                pass
        if not idxs:
            st.warning("Aucune ligne valide sélectionnée.")
        else:
            roster2 = roster.drop(index=idxs, errors="ignore").reset_index(drop=True)
            err4 = roster_save(roster2, data_dir, season)
            if err4:
                st.error(err4)
            else:
                st.success(f"Supprimés: {len(idxs)}")
                st.rerun()

    st.markdown("---")
    st.markdown("### 📤 Export roster")
    colx1, colx2 = st.columns(2)
    with colx1:
        st.download_button(
            "⬇️ Export roster (CSV)",
            data=roster.to_csv(index=False).encode("utf-8"),
            file_name=os.path.basename(roster_path(data_dir, season)),
            mime="text/csv",
            use_container_width=True,
        )
    with colx2:
        st.download_button(
            "⬇️ Export roster filtré (CSV)",
            data=roster_f.to_csv(index=False).encode("utf-8"),
            file_name=f"roster_filtered_{season}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =====================================================
# UI — Outils
# =====================================================
def render_tools(data_dir: str, season: str) -> None:
    st.subheader("🧰 Outils — synchros")

    with st.expander("🧾 Sync PuckPedia → Level (STD/ELC)", expanded=False):
        puck = st.text_input("Fichier PuckPedia", puckpedia_path(data_dir))
        players = st.text_input("Players DB", players_db_path(data_dir))

        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("👀 Voir colonnes PuckPedia"):
                try:
                    pk0 = pd.read_csv(puck, nrows=0)
                    st.write(list(pk0.columns))
                except Exception as e:
                    st.error(f"Lecture PuckPedia échouée: {e}")
        with colp2:
            if st.button("👀 Voir colonnes Players DB"):
                try:
                    pdb0 = pd.read_csv(players, nrows=0)
                    st.write(list(pdb0.columns))
                except Exception as e:
                    st.error(f"Lecture Players DB échouée: {e}")

        if st.button("Synchroniser Level"):
            res = sync_level(players, puck)
            if res.get("ok"):
                st.success(f"Levels mis à jour: {res.get('updated', 0)}")
            else:
                st.error(res.get("error", "Erreur inconnue"))
                if res.get("missing"):
                    st.write("Chemins manquants:", res["missing"])
                if res.get("players_cols"):
                    st.write("Players DB colonnes:", res["players_cols"])
                if res.get("puck_cols"):
                    st.write("PuckPedia colonnes:", res["puck_cols"])

    with st.expander("🆔 Sync NHL_ID manquants (avec progression + audit + exports)", expanded=False):
        players2 = st.text_input("Players DB (NHL_ID)", players_db_path(data_dir), key="nhl_players")
        limit = st.number_input("Max par run", 1, 2000, 1000, step=50)
        dry = st.checkbox("Dry-run (ne sauvegarde pas)", value=False)

        if st.button("🔎 Vérifier l'état des NHL_ID"):
            df, err = load_csv(players2)
            if err:
                st.error(err)
            else:
                if "NHL_ID" not in df.columns:
                    df["NHL_ID"] = ""

                a = audit_nhl_ids(df)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total joueurs", a["total"])
                c2.metric("Avec NHL_ID", a["filled"])
                c3.metric("Manquants", a["missing"])
                c4.metric("% manquants", f"{a['missing_pct']:.1f}%")

                if a.get("duplicates", 0):
                    st.warning(f"IDs dupliqués détectés: {a['duplicates']} (souvent normal si erreurs de match).")

                st.markdown("#### 📤 Export")
                ex1, ex2 = st.columns(2)
                with ex1:
                    st.download_button(
                        "⬇️ Exporter la liste complète (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="players_full.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with ex2:
                    missing_df = df[df["NHL_ID"].apply(_is_missing_id)].copy()
                    st.download_button(
                        "⬇️ Exporter la liste des manquants (CSV)",
                        data=missing_df.to_csv(index=False).encode("utf-8"),
                        file_name="players_missing_nhl_id.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                if a["missing"] > 0:
                    cols = ["Player"]
                    if "Team" in df.columns:
                        cols.append("Team")
                    if "Position" in df.columns:
                        cols.append("Position")
                    st.caption("Aperçu (max 200) des joueurs sans NHL_ID :")
                    st.dataframe(missing_df[cols].head(200), use_container_width=True)

        if st.button("Associer NHL_ID"):
            res = sync_nhl_id(players2, int(limit), dry_run=bool(dry))
            if res.get("ok"):
                st.success(res.get("summary", "Terminé."))
                a = res.get("audit") or {}
                if a:
                    st.caption(
                        f"État actuel — Total: {a.get('total')} | Avec NHL_ID: {a.get('filled')} | "
                        f"Manquants: {a.get('missing')} ({a.get('missing_pct', 0):.1f}%)"
                    )
            else:
                st.error(res.get("error", "Erreur inconnue"))
