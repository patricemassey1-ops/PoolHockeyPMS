# admin.py
# ============================================================
# PMS Pool Hockey — Admin Module (Streamlit)
# ============================================================
# ✔ Bonus: auto-mapping Level (ELC/STD) via hockey.players.csv
# ✔ Quality checks: IR mismatch + Salary/Level suspect
# ✔ Reload mémoire: st.session_state["equipes_df"]
# ✔ ➕ Ajouter joueur(s) à une équipe (GC/CE/Banc) + ANTI-TRICHE
# ✔ 🗑️ Retirer joueur(s) d’une équipe
# ✔ 🔁 Déplacer joueur(s) GC ↔ CE (+ auto-slot)
# ✔ 🧪 Vérification masse salariale live (GC/CE vs caps)
# ✔ 📋 Historique admin (CSV)
# ✔ 🧠 Auto-assign Slot selon Statut
# ============================================================

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
DATA_DIR_DEFAULT = "Data"
PLAYERS_DB_FILENAME = "hockey.players.csv"

DEFAULT_CAP_GC = 88_000_000
DEFAULT_CAP_CE = 12_000_000

EQUIPES_COLUMNS = [
    "Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"
]


def admin_log_path(data_dir: str, season_lbl: str) -> str:
    return os.path.join(data_dir, f"admin_actions_{season_lbl}.csv")


def equipes_path(data_dir: str, season_lbl: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season_lbl}.csv")


# ============================================================
# BASIC HELPERS
# ============================================================
def ensure_data_dir(path: str) -> str:
    path = path or DATA_DIR_DEFAULT
    os.makedirs(path, exist_ok=True)
    return path


def _norm_player_name(x: Any) -> str:
    return str(x or "").strip().lower()


def _norm_level(v: Any) -> str:
    s = str(v or "").strip().upper()
    return s if s in {"ELC", "STD"} else "0"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(pd.to_numeric(x, errors="coerce")) if x is not None else default
    except Exception:
        return default


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    if df is None or df.empty:
        return ""
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in low:
            return low[key]
    return ""


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# LOAD/SAVE EQUIPES
# ============================================================
def ensure_equipes_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=EQUIPES_COLUMNS)

    out = df.copy()
    for c in EQUIPES_COLUMNS:
        if c not in out.columns:
            out[c] = ""

    for c in ["Propriétaire", "Joueur", "Pos", "Equipe", "Level", "Statut", "Slot", "IR Date"]:
        out[c] = out[c].astype(str).fillna("").str.strip()

    out["Salaire"] = pd.to_numeric(out.get("Salaire", 0), errors="coerce").fillna(0).astype(int)
    out["Level"] = out["Level"].apply(_norm_level)

    # keep expected cols first
    return out[EQUIPES_COLUMNS + [c for c in out.columns if c not in EQUIPES_COLUMNS]]


def load_equipes_df(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=EQUIPES_COLUMNS)
    try:
        return ensure_equipes_df(pd.read_csv(path))
    except Exception:
        return pd.DataFrame(columns=EQUIPES_COLUMNS)


def save_equipes_df(df: pd.DataFrame, path: str) -> None:
    ensure_equipes_df(df).to_csv(path, index=False)


def reload_equipes_in_memory(path: str) -> pd.DataFrame:
    df = load_equipes_df(path)
    st.session_state["equipes_df"] = df
    st.session_state["equipes_path"] = path
    st.session_state["equipes_last_loaded"] = _now_ts()
    return df


# ============================================================
# PLAYERS DB (source de vérité Level + infos joueur)
# ============================================================
@st.cache_data(show_spinner=False)
def load_players_db(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_players_index(players_db: pd.DataFrame) -> dict:
    if not isinstance(players_db, pd.DataFrame) or players_db.empty:
        return {}

    c_name = _pick_col(players_db, ["Joueur", "Player", "Name"])
    if not c_name:
        return {}

    c_pos = _pick_col(players_db, ["Pos", "Position"])
    c_team = _pick_col(players_db, ["Equipe", "Équipe", "Team"])
    c_sal = _pick_col(players_db, ["Salaire", "Cap Hit", "CapHit", "Cap", "Cap_Hit"])
    c_lvl = _pick_col(players_db, ["Level"])

    idx: dict[str, dict] = {}
    for _, r in players_db.iterrows():
        name = str(r.get(c_name, "")).strip()
        if not name:
            continue
        idx[_norm_player_name(name)] = {
            "Joueur": name,
            "Pos": str(r.get(c_pos, "")).strip() if c_pos else "",
            "Equipe": str(r.get(c_team, "")).strip() if c_team else "",
            "Salaire": _safe_int(r.get(c_sal, 0)) if c_sal else 0,
            "Level": _norm_level(r.get(c_lvl, "0")) if c_lvl else "0",
        }
    return idx


# ============================================================
# QUALITY CHECKS + AUTO LEVEL
# ============================================================
def _infer_level_from_salary(sal: int) -> str:
    return "ELC" if int(sal) <= 1_000_000 else "STD"


def apply_quality_checks(df: pd.DataFrame, players_idx: dict) -> Tuple[pd.DataFrame, dict]:
    out = ensure_equipes_df(df)

    # Auto-map Level
    need = out["Level"].astype(str).str.strip().isin({"0", ""})
    filled = 0
    if need.any():
        for i in out[need].index:
            nm = out.at[i, "Joueur"]
            key = _norm_player_name(nm)
            mapped = ""
            if key in players_idx:
                mapped = str(players_idx[key].get("Level", "")).strip().upper()
            if mapped in {"ELC", "STD"}:
                out.at[i, "Level"] = mapped
                filled += 1
            else:
                out.at[i, "Level"] = _infer_level_from_salary(int(out.at[i, "Salaire"]))
                filled += 1

    # IR mismatch
    ir_present = out["IR Date"].astype(str).str.strip().ne("") & out["IR Date"].astype(str).str.lower().ne("nan")
    slot_is_ir = out["Slot"].astype(str).str.strip().str.upper().eq("IR")
    out["⚠️ IR mismatch"] = (ir_present & (~slot_is_ir))

    # Salary/Level suspect
    lvl = out["Level"].astype(str).str.upper()
    sal = out["Salaire"].astype(int)
    suspect_elc = lvl.eq("ELC") & (sal > 1_500_000)
    suspect_std = lvl.eq("STD") & (sal <= 0)
    out["⚠️ Salary/Level suspect"] = (suspect_elc | suspect_std)

    stats = {
        "rows": int(len(out)),
        "level_autofilled": int(filled),
        "ir_mismatch": int((out["⚠️ IR mismatch"] == True).sum()),
        "salary_level_suspect": int((out["⚠️ Salary/Level suspect"] == True).sum()),
    }
    return out, stats


def _preview_style_row(row: pd.Series) -> list[str]:
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
# ADMIN HISTORY (CSV)
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
        "owner": str(owner or "").strip(),
        "player": str(player or "").strip(),
        "from_statut": str(from_statut or "").strip(),
        "from_slot": str(from_slot or "").strip(),
        "to_statut": str(to_statut or "").strip(),
        "to_slot": str(to_slot or "").strip(),
        "note": str(note or "").strip(),
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
# AUTO SLOT LOGIC (🧠)
# ============================================================
def auto_slot_for_statut(statut: str, *, current_slot: str = "", keep_ir: bool = True) -> str:
    cur = str(current_slot or "").strip().upper()
    if keep_ir and cur == "IR":
        return "IR"
    # règle simple par défaut
    return "Actif"


# ============================================================
# SALARY CAP CHECK (🧪)
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
# ANTI-TRICHE (joueur déjà dans une autre équipe)
# ============================================================
def find_player_owner(df: pd.DataFrame, player_name: str) -> Optional[str]:
    """
    Retourne le propriétaire actuel d'un joueur, si le joueur existe déjà.
    Comparaison par nom normalisé.
    """
    if df is None or df.empty or not player_name:
        return None
    key = _norm_player_name(player_name)
    d = ensure_equipes_df(df)
    m = d["Joueur"].astype(str).map(_norm_player_name).eq(key)
    if not m.any():
        return None
    # si plusieurs entrées, on prend le premier propriétaire trouvé
    owner = d.loc[m, "Propriétaire"].astype(str).str.strip().iloc[0]
    return owner if owner else None


# ============================================================
# MAIN ADMIN UI
# ============================================================
def render_admin_tab(
    *,
    is_admin: bool,
    season_lbl: str,
    data_dir: str = DATA_DIR_DEFAULT,
    folder_id: str = "",  # réservé si tu veux ajouter Drive ici plus tard
) -> None:
    if not is_admin:
        st.warning("Accès admin requis.")
        st.stop()

    data_dir = ensure_data_dir(data_dir)
    season_lbl = str(season_lbl or "").strip() or "2025-2026"

    e_path = equipes_path(data_dir, season_lbl)
    log_path = admin_log_path(data_dir, season_lbl)

    st.subheader("🛠️ Gestion Admin")

    # caps live
    st.session_state.setdefault("CAP_GC", DEFAULT_CAP_GC)
    st.session_state.setdefault("CAP_CE", DEFAULT_CAP_CE)

    # Load players db
    players_db_path = os.path.join(data_dir, PLAYERS_DB_FILENAME)
    players_db = load_players_db(players_db_path)
    players_idx = build_players_index(players_db)

    if players_idx:
        st.info(f"✅ Bonus Level activé via `{PLAYERS_DB_FILENAME}` (mapping joueur → Level).")
    else:
        st.warning(f"ℹ️ `{PLAYERS_DB_FILENAME}` indisponible — fallback Level via Salaire.")

    # Load equipes
    df_eq = load_equipes_df(e_path)
    if df_eq.empty and not os.path.exists(e_path):
        # crée squelette si absent
        save_equipes_df(pd.DataFrame(columns=EQUIPES_COLUMNS), e_path)
        df_eq = load_equipes_df(e_path)

    owners = sorted([x for x in df_eq.get("Propriétaire", pd.Series(dtype=str)).dropna().astype(str).str.strip().unique() if x])

    # ---------------------------------------------------------
    # 🧪 Caps live
    # ---------------------------------------------------------
    with st.expander("🧪 Vérification masse salariale (live)", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.session_state["CAP_GC"] = st.number_input("Cap GC", min_value=0, value=int(st.session_state["CAP_GC"]), step=500000)
        with c2:
            st.session_state["CAP_CE"] = st.number_input("Cap CE", min_value=0, value=int(st.session_state["CAP_CE"]), step=250000)
        with c3:
            st.caption("Caps utilisés seulement pour affichage live ici.")

        if df_eq.empty:
            st.warning("Aucun fichier équipes chargé.")
        else:
            caps = compute_caps(df_eq)
            st.dataframe(caps, use_container_width=True)

    if df_eq.empty or not owners:
        st.warning("Importe d’abord `equipes_joueurs_...csv` (avec des Propriétaires).")
        return

    # ---------------------------------------------------------
    # 🧼 Preview + QC
    # ---------------------------------------------------------
    df_eq_qc, qc_stats = apply_quality_checks(df_eq, players_idx)
    with st.expander("🧼 Preview équipes + alertes", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes", qc_stats["rows"])
        c2.metric("Level auto", qc_stats["level_autofilled"])
        c3.metric("⚠️ IR mismatch", qc_stats["ir_mismatch"])
        c4.metric("⚠️ Salaire/Level", qc_stats["salary_level_suspect"])
        try:
            st.dataframe(df_eq_qc.head(120).style.apply(_preview_style_row, axis=1), use_container_width=True)
        except Exception:
            st.dataframe(df_eq_qc.head(120), use_container_width=True)

        if st.button("💾 Appliquer corrections + sauvegarder équipes", use_container_width=True, key="admin_save_qc"):
            save_equipes_df(df_eq_qc, e_path)
            reload_equipes_in_memory(e_path)
            st.success("✅ Corrections appliquées + sauvegarde + reload.")
            st.rerun()

    # ---------------------------------------------------------
    # ➕ AJOUT (avec ANTI-TRICHE)
    # ---------------------------------------------------------
    with st.expander("➕ Ajouter des joueurs à une équipe (anti-triche)", expanded=False):
        owner = st.selectbox("Équipe (Propriétaire)", owners, key="admin_add_owner")

        assign = st.radio(
            "Assignation",
            ["GC - Actif", "GC - Banc", "CE - Actif", "CE - Banc"],
            horizontal=True,
            key="admin_add_assign"
        )
        statut = "Grand Club" if assign.startswith("GC") else "Club École"
        slot = "Actif" if assign.endswith("Actif") else "Banc"

        # Option admin override (désactivé par défaut)
        allow_override = st.checkbox("🛑 Autoriser override (admin) si joueur appartient déjà à une autre équipe", value=False)

        st.markdown("### Sélection joueurs")
        if players_idx:
            all_names = sorted({v["Joueur"] for v in players_idx.values() if v.get("Joueur")})
            selected = st.multiselect("Joueurs", all_names, key="admin_add_players")
        else:
            raw = st.text_area("Saisir les joueurs (1 par ligne)", height=140, key="admin_add_players_manual")
            selected = [x.strip() for x in raw.splitlines() if x.strip()]

        df_current = load_equipes_df(e_path)

        # Build preview + anti-cheat check
        preview_rows = []
        blocked = []  # (player, current_owner)
        for p in selected:
            info = players_idx.get(_norm_player_name(p), {}) if players_idx else {}
            name = info.get("Joueur", p)

            current_owner = find_player_owner(df_current, name)
            if current_owner and str(current_owner).strip() != str(owner).strip():
                blocked.append((name, current_owner))
                if not allow_override:
                    continue

            preview_rows.append({
                "Propriétaire": owner,
                "Joueur": name,
                "Pos": info.get("Pos", ""),
                "Equipe": info.get("Equipe", ""),
                "Salaire": int(info.get("Salaire", 0) or 0),
                "Level": info.get("Level", "0"),
                "Statut": statut,
                "Slot": slot,
                "IR Date": "",
            })

        if blocked and not allow_override:
            st.error("⛔ Anti-triche: certains joueurs appartiennent déjà à une autre équipe.")
            st.dataframe(pd.DataFrame(blocked, columns=["Joueur", "Appartient déjà à"]), use_container_width=True)

        if preview_rows:
            st.markdown("### Preview ajouts")
            st.dataframe(pd.DataFrame(preview_rows).head(80), use_container_width=True)
        else:
            st.info("Aucun ajout possible (ou aucun joueur sélectionné).")

        if st.button("✅ Ajouter + sauvegarder + reload", use_container_width=True, key="admin_add_commit"):
            if not preview_rows:
                st.warning("Rien à ajouter.")
                st.stop()

            df = load_equipes_df(e_path)

            # anti-doublon interne (même owner + joueur)
            existing_same_owner = set(zip(
                df["Propriétaire"].astype(str).str.strip(),
                df["Joueur"].astype(str).str.strip(),
            ))

            new_rows = []
            skipped_dupe = 0
            skipped_blocked = 0

            for r in preview_rows:
                k = (str(r["Propriétaire"]).strip(), str(r["Joueur"]).strip())
                if k in existing_same_owner:
                    skipped_dupe += 1
                    continue

                # re-vérifie anti-cheat au commit
                current_owner = find_player_owner(df, r["Joueur"])
                if current_owner and str(current_owner).strip() != str(owner).strip() and not allow_override:
                    skipped_blocked += 1
                    continue

                new_rows.append(r)

            if not new_rows:
                st.warning(f"Rien à ajouter (doublons: {skipped_dupe}, bloqués: {skipped_blocked}).")
                st.stop()

            merged = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            merged_qc, stats = apply_quality_checks(merged, players_idx)

            save_equipes_df(merged_qc, e_path)
            reload_equipes_in_memory(e_path)

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

            st.success(
                f"✅ Ajout OK: {len(new_rows)} ajoutés | doublons: {skipped_dupe} | bloqués: {skipped_blocked} | Level auto: {stats['level_autofilled']}"
            )
            st.rerun()

    # ---------------------------------------------------------
    # 🗑️ RETIRER
    # ---------------------------------------------------------
    with st.expander("🗑️ Retirer des joueurs d’une équipe", expanded=False):
        owner = st.selectbox("Équipe (Propriétaire)", owners, key="admin_remove_owner")

        df = load_equipes_df(e_path)
        team_df = df[df["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()

        if team_df.empty:
            st.warning("Aucun joueur pour cette équipe.")
        else:
            team_df["__label__"] = team_df.apply(
                lambda r: f"{r['Joueur']}  —  {r.get('Pos','')}  —  {r.get('Statut','')} / {r.get('Slot','')}",
                axis=1
            )
            options = team_df["__label__"].tolist()
            to_remove = st.multiselect("Choisir joueur(s) à retirer", options, key="admin_remove_players")

            confirm = st.checkbox("Je confirme la suppression", key="admin_remove_confirm")
            if st.button("🗑️ Retirer + sauvegarder + reload", use_container_width=True, key="admin_remove_commit"):
                if not to_remove:
                    st.warning("Sélectionne au moins 1 joueur.")
                    st.stop()
                if not confirm:
                    st.warning("Coche la confirmation.")
                    st.stop()

                sel_rows = team_df[team_df["__label__"].isin(to_remove)].copy()
                if sel_rows.empty:
                    st.warning("Sélection invalide.")
                    st.stop()

                before = len(df)

                keys = set(zip(
                    sel_rows["Propriétaire"].astype(str),
                    sel_rows["Joueur"].astype(str),
                    sel_rows["Statut"].astype(str),
                    sel_rows["Slot"].astype(str),
                ))

                def _keep(r):
                    k = (str(r["Propriétaire"]), str(r["Joueur"]), str(r["Statut"]), str(r["Slot"]))
                    return k not in keys

                df2 = df[df.apply(_keep, axis=1)].copy()
                after = len(df2)

                df2_qc, _ = apply_quality_checks(df2, players_idx)
                save_equipes_df(df2_qc, e_path)
                reload_equipes_in_memory(e_path)

                for _, r in sel_rows.iterrows():
                    append_admin_log(
                        log_path,
                        action="REMOVE",
                        owner=r["Propriétaire"],
                        player=r["Joueur"],
                        from_statut=r.get("Statut", ""),
                        from_slot=r.get("Slot", ""),
                        note="removed by admin"
                    )

                st.success(f"✅ Suppression OK: {before - after} joueur(s) retiré(s).")
                st.rerun()

    # ---------------------------------------------------------
    # 🔁 MOVE GC ↔ CE
    # ---------------------------------------------------------
    with st.expander("🔁 Déplacer des joueurs GC ↔ CE", expanded=False):
        owner = st.selectbox("Équipe (Propriétaire)", owners, key="admin_move_owner")

        df = load_equipes_df(e_path)
        team_df = df[df["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()

        if team_df.empty:
            st.warning("Aucun joueur pour cette équipe.")
        else:
            team_df["__label__"] = team_df.apply(
                lambda r: f"{r['Joueur']}  —  {r.get('Pos','')}  —  {r.get('Statut','')} / {r.get('Slot','')}",
                axis=1
            )
            options = team_df["__label__"].tolist()
            sel = st.multiselect("Joueur(s) à déplacer", options, key="admin_move_players")

            dest_statut = st.radio(
                "Destination Statut",
                ["Grand Club", "Club École"],
                horizontal=True,
                key="admin_move_dest_statut"
            )

            mode_slot = st.radio(
                "Slot destination",
                ["Auto (selon Statut)", "Garder Slot actuel", "Forcer Actif", "Forcer Banc"],
                horizontal=True,
                key="admin_move_slot_mode"
            )

            keep_ir = st.checkbox("Conserver IR si joueur déjà IR", value=True, key="admin_move_keep_ir")

            if st.button("🔁 Déplacer + sauvegarder + reload", use_container_width=True, key="admin_move_commit"):
                if not sel:
                    st.warning("Sélectionne au moins 1 joueur.")
                    st.stop()

                sel_rows = team_df[team_df["__label__"].isin(sel)].copy()
                if sel_rows.empty:
                    st.warning("Sélection invalide.")
                    st.stop()

                df2 = df.copy()

                keyset = set(zip(
                    sel_rows["Propriétaire"].astype(str),
                    sel_rows["Joueur"].astype(str),
                    sel_rows["Statut"].astype(str),
                    sel_rows["Slot"].astype(str),
                ))

                moved = 0
                for idx, r in df2.iterrows():
                    k = (str(r["Propriétaire"]), str(r["Joueur"]), str(r["Statut"]), str(r["Slot"]))
                    if k not in keyset:
                        continue

                    from_statut = str(r["Statut"])
                    from_slot = str(r["Slot"])

                    df2.at[idx, "Statut"] = dest_statut

                    if mode_slot.startswith("Auto"):
                        df2.at[idx, "Slot"] = auto_slot_for_statut(dest_statut, current_slot=from_slot, keep_ir=keep_ir)
                    elif mode_slot.startswith("Garder"):
                        df2.at[idx, "Slot"] = from_slot
                    elif mode_slot.endswith("Actif"):
                        df2.at[idx, "Slot"] = "Actif"
                    elif mode_slot.endswith("Banc"):
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
                        note=f"slot_mode={mode_slot}"
                    )

                df2_qc, stats = apply_quality_checks(df2, players_idx)
                save_equipes_df(df2_qc, e_path)
                reload_equipes_in_memory(e_path)

                st.success(f"✅ Déplacement OK: {moved} joueur(s). | Level auto: {stats['level_autofilled']}")
                st.rerun()

    # ---------------------------------------------------------
    # 📋 LOG
    # ---------------------------------------------------------
    with st.expander("📋 Historique Admin (ajouts / retraits / moves)", expanded=False):
        if not os.path.exists(log_path):
            st.info("Aucun historique pour l’instant.")
        else:
            try:
                lg = pd.read_csv(log_path).sort_values("timestamp", ascending=False)

                f1, f2, f3 = st.columns(3)
                with f1:
                    action = st.multiselect("Action", sorted(lg["action"].dropna().unique()), default=[], key="admin_log_action")
                with f2:
                    owner = st.multiselect("Équipe", sorted(lg["owner"].dropna().unique()), default=[], key="admin_log_owner")
                with f3:
                    q = st.text_input("Recherche joueur", value="", key="admin_log_q").strip().lower()

                view = lg.copy()
                if action:
                    view = view[view["action"].isin(action)]
                if owner:
                    view = view[view["owner"].isin(owner)]
                if q:
                    view = view[view["player"].astype(str).str.lower().str.contains(q, na=False)]

                st.dataframe(view.head(300), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lecture log: {e}")

    st.caption("✅ Admin OK — Ajout / Retrait / Move / Caps / Log / Anti-triche.")
