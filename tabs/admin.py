# tabs/admin.py
# ============================================================
# PMS Pool Hockey — Admin Tab (Streamlit)
# Compatible avec: admin.render(ctx) depuis app.py
# ============================================================
# Features:
# ✅ Import équipes depuis Drive (OAuth creds dans st.session_state["drive_creds"])
# ✅ Preview + validation colonnes
# ✅ ➕ Ajouter joueur(s) (anti-triche cross-team) + override admin option
# ✅ 🗑️ Retirer joueur(s) (UI + confirmation)
# ✅ 🔁 Déplacer GC ↔ CE (auto-slot / keep / force)
# ✅ 🧪 Caps live: barres visuelles GC/CE par équipe + dépassements
# ✅ 📋 Historique admin (ADD/REMOVE/MOVE/IMPORT) filtrable (CSV par saison)
# ✅ Auto-mapping Level via hockey.players.csv + heuristique salaire
# ✅ Alertes IR mismatch + Salary/Level suspect + preview colorée
# ============================================================

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---- Google Drive OAuth deps (optional)
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
except Exception:
    build = None
    MediaIoBaseDownload = None
    Credentials = None


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
    # keep expected first
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
        return pd.read_csv(path)
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
    expected = EQUIPES_COLUMNS
    cols = list(df.columns)
    missing = [c for c in expected if c not in cols]
    extras = [c for c in cols if c not in expected]
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
# AUTO SLOT (🧠)
# ============================================================
def auto_slot_for_statut(dest_statut: str, *, current_slot: str = "", keep_ir: bool = True) -> str:
    cur = str(current_slot or "").strip().upper()
    if keep_ir and cur == "IR":
        return "IR"
    # simple: Actif par défaut
    return "Actif"


# ============================================================
# CAPS (🧪)
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


# ============================================================
# ADMIN LOG (📋)
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
# DRIVE IMPORT (🔄)
# ============================================================
def _get_drive_service_from_session() -> Optional[Any]:
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

def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    # utf-8 puis latin-1 fallback
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        return pd.read_csv(io.BytesIO(b), encoding="latin-1")


# ============================================================
# UI BLOCKS
# ============================================================
def _render_caps_bars(df_eq: pd.DataFrame, cap_gc: int, cap_ce: int) -> None:
    st.markdown("### 🧪 Caps — barres visuelles (GC / CE)")
    caps = compute_caps(df_eq)
    if caps.empty:
        st.info("Aucune donnée équipes.")
        return

    # display summary table + bars per owner
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
# MAIN RENDER
# ============================================================
def render(ctx: dict):
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    DATA_DIR = str(ctx.get("DATA_DIR") or "Data")
    season_lbl = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    folder_id = str(ctx.get("drive_folder_id") or "").strip()

    e_path = equipes_path(DATA_DIR, season_lbl)
    log_path = admin_log_path(DATA_DIR, season_lbl)

    st.subheader("🛠️ Gestion Admin")

    # ---- caps inputs (live)
    st.session_state.setdefault("CAP_GC", DEFAULT_CAP_GC)
    st.session_state.setdefault("CAP_CE", DEFAULT_CAP_CE)

    with st.expander("🧪 Vérification cap (live) + barres", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.session_state["CAP_GC"] = st.number_input("Cap GC", min_value=0, value=int(st.session_state["CAP_GC"]), step=500000)
        with c2:
            st.session_state["CAP_CE"] = st.number_input("Cap CE", min_value=0, value=int(st.session_state["CAP_CE"]), step=250000)
        with c3:
            st.caption("Caps utilisés ici pour affichage & alertes.")
        df_eq = load_equipes(e_path)
        if df_eq.empty:
            st.info("Aucun fichier équipes local trouvé (importe depuis Drive ou local).")
        else:
            _render_caps_bars(df_eq, int(st.session_state["CAP_GC"]), int(st.session_state["CAP_CE"]))

    # ---- Players DB index
    players_db = load_players_db(os.path.join(DATA_DIR, PLAYERS_DB_FILENAME))
    players_idx = build_players_index(players_db)
    if players_idx:
        st.info(f"✅ Players DB détectée: `{PLAYERS_DB_FILENAME}` (Level auto + infos).")
    else:
        st.warning(f"ℹ️ `{PLAYERS_DB_FILENAME}` indisponible — fallback Level par Salaire; sélection joueurs limitée.")

    # ---- Load équipes (local)
    df = load_equipes(e_path)

    # Owners list
    owners = sorted([x for x in df.get("Propriétaire", pd.Series(dtype=str)).dropna().astype(str).str.strip().unique() if x])

    # =====================================================
    # 🔄 IMPORT ÉQUIPES (Drive) + preview + validate
    # =====================================================
    with st.expander("🔄 Import équipes depuis Drive (OAuth)", expanded=False):
        st.caption("Nécessite OAuth: st.session_state['drive_creds'] et un folder_id valide.")

        svc = _get_drive_service_from_session()
        drive_ok = bool(svc) and bool(folder_id)

        if not drive_ok:
            st.warning("Drive OAuth non disponible (creds ou folder_id manquant).")
            st.write(f"folder_id (ctx): `{folder_id or ''}`")
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
                        ok, missing, extras = validate_equipes_df(df_drive)
                        if not ok:
                            st.error(f"Import refusé: colonnes manquantes {missing}")
                        else:
                            df_imp = ensure_equipes_df(df_drive)
                            df_imp_qc, stats = apply_quality(df_imp, players_idx)
                            save_equipes(df_imp_qc, e_path)
                            st.session_state["equipes_df"] = df_imp_qc  # reload in memory
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
    # 🧼 PREVIEW LOCAL + QC APPLY
    # =====================================================
    
    # =====================================================
    # 📥 IMPORT LOCAL (fallback) — upload CSV vers equipes_joueurs_{season}.csv
    # =====================================================
    with st.expander("📥 Import local (fallback) — uploader un CSV équipes", expanded=True):
        st.caption("Si Drive OAuth n’est pas prêt, uploade ici ton `equipes_joueurs_...csv` pour tester immédiatement.")
        st.code(f"Destination locale: {e_path}", language="text")
        up = st.file_uploader("Uploader un CSV (équipes)", type=["csv"], key="adm_local_upload")
        col1, col2, col3 = st.columns([1,1,2])
        do_validate_local = col1.button("🧪 Valider colonnes", use_container_width=True, key="adm_local_validate")
        do_preview_local = col2.button("🧼 Preview", use_container_width=True, key="adm_local_preview")
        col3.caption("Le fichier est lu, validé, puis sauvegardé dans /Data et chargé en mémoire.")

        if up is not None and (do_validate_local or do_preview_local or st.button("⬇️ Importer → Local + QC + Reload", use_container_width=True, key="adm_local_import")):
            try:
                df_up = pd.read_csv(up)
            except Exception:
                up.seek(0)
                df_up = pd.read_csv(up, encoding="latin-1")

            st.dataframe(df_up.head(80), use_container_width=True)

            ok, missing, extras = validate_equipes_df(df_up)
            if do_validate_local:
                if ok:
                    st.success("✅ Colonnes attendues OK.")
                    if extras:
                        st.info(f"Colonnes additionnelles: {extras}")
                else:
                    st.error(f"❌ Colonnes manquantes: {missing}")
                    if extras:
                        st.info(f"Colonnes additionnelles: {extras}")

            if ok:
                df_imp = ensure_equipes_df(df_up)
                df_imp_qc, stats = apply_quality(df_imp, players_idx)
                save_equipes(df_imp_qc, e_path)
                st.session_state["equipes_df"] = df_imp_qc
                append_admin_log(
                    log_path,
                    action="IMPORT_LOCAL",
                    owner="",
                    player="",
                    note=f"upload={up.name}; level_auto={stats.get('level_autofilled',0)}"
                )
                st.success(f"✅ Import local OK → {os.path.basename(e_path)} | Level auto: {stats.get('level_autofilled',0)}")
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
        if df.empty:
            st.info("Aucun fichier équipes local. Importe depuis Drive ou upload local via l'app.")
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

    # refresh from disk after potential import/qc
    df = load_equipes(e_path)
    owners = sorted([x for x in df.get("Propriétaire", pd.Series(dtype=str)).dropna().astype(str).str.strip().unique() if x])

    if not owners:
        st.warning("Aucune équipe (Propriétaire) détectée. Importe d'abord le CSV équipes.")
        return

    # =====================================================
    # ➕ ADD (ANTI-TRICHE)
    # =====================================================
    with st.expander("➕ Ajouter joueur(s) (anti-triche)", expanded=True):
        owner = st.selectbox("Équipe", owners, key="adm_add_owner")
        assign = st.radio("Assignation", ["GC - Actif","GC - Banc","CE - Actif","CE - Banc"], horizontal=True, key="adm_add_assign")
        statut = "Grand Club" if assign.startswith("GC") else "Club École"
        slot = "Actif" if assign.endswith("Actif") else "Banc"

        allow_override = st.checkbox("🛑 Autoriser override admin si joueur appartient déjà à une autre équipe", value=False, key="adm_add_override")

        if players_idx:
            all_names = sorted({v["Joueur"] for v in players_idx.values() if v.get("Joueur")})
            selected = st.multiselect("Joueurs", all_names, key="adm_add_players")
        else:
            raw = st.text_area("Saisir joueurs (1 par ligne)", height=120, key="adm_add_manual")
            selected = [x.strip() for x in raw.splitlines() if x.strip()]

        preview = []
        blocked = []
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
                "Pos": info.get("Pos",""),
                "Equipe": info.get("Equipe",""),
                "Salaire": int(info.get("Salaire", 0) or 0),
                "Level": info.get("Level","0"),
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

            # évite doublons same owner+player
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
    # 🗑️ REMOVE (UI + confirmation)
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
                        from_statut=r.get("Statut",""),
                        from_slot=r.get("Slot",""),
                        note="removed by admin"
                    )

                st.success(f"✅ Retrait OK: {removed} joueur(s).")
                st.rerun()

    # =====================================================
    # 🔁 MOVE GC ↔ CE (auto-slot)
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

    st.caption("✅ Admin: Import Drive • Add/Remove/Move • Caps bars • Log • QC/Level auto")
