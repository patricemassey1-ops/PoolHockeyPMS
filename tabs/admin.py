# tabs/admin.py
from __future__ import annotations

import os
import io
import re
import csv
import unicodedata
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import streamlit as st
from pandas.errors import ParserError

# ============================================================
# ADMIN TAB — IMPORT /data + NORMALISATION + ANTI-DOUBLON + LEVEL AUTO + ROLLBACK
#   ✅ Import depuis /data/*.csv (pas d'upload requis)
#   ✅ Lecture CSV robuste (auto delimiter + fallback python, skip lignes brisées)
#   ✅ Normalisation Fantrax/exports -> schéma PMS (Propriétaire/Joueur/Pos/Salaire/Slot)
#   ✅ Anti-doublon joueur inter-équipes (anti-triche)
#   ✅ Auto-Level ELC/STD quand Level vide/0 (via data/hockey.players.csv)
#   ✅ Résumé d’import (nb joueurs + masses GC/CE)
#   ✅ Backup automatique + rollback par équipe
# ============================================================


# -----------------------------
# State
# -----------------------------
def _init_state() -> None:
    st.session_state.setdefault("admin_prepared", [])          # list[dict(filename, team, df)]
    st.session_state.setdefault("admin_last_parse_report", []) # list[str]


# -----------------------------
# Paths
# -----------------------------
def equipes_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")


def admin_backup_dir(data_dir: str, season: str) -> str:
    d = os.path.join(data_dir, "admin_backups", season)
    os.makedirs(d, exist_ok=True)
    return d


def players_db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "hockey.players.csv")


# -----------------------------
# Utils
# -----------------------------
def infer_team_from_filename(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0].strip()


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_player_key(name: str) -> str:
    s = _strip_accents(str(name or "")).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 \-\.]", "", s)
    return s


def _guess_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        candidates = [",", ";", "\t", "|"]
        scores = {d: sample.count(d) for d in candidates}
        return max(scores, key=scores.get) if scores else ","


def read_csv_bytes_robust(raw: bytes, name: str = "csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Lecture CSV robuste:
      - detect encoding utf-8-sig / utf-8 / latin-1
      - detect delimiter
      - tente C-engine puis fallback python engine + on_bad_lines='skip'
    """
    enc_used = None
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            enc_used = enc
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode("latin-1", errors="replace")
        enc_used = "latin-1"

    sample = text[:5000]
    delim = _guess_delimiter(sample)

    report: Dict[str, Any] = {
        "file": name,
        "encoding": enc_used,
        "delimiter": delim if delim != "\t" else "\\t",
        "engine": "c",
        "warning": "",
    }

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="c")
        return df, report
    except ParserError as e:
        report["warning"] = f"C-engine ParserError: {e}"
    except Exception as e:
        report["warning"] = f"C-engine error: {e}"

    report["engine"] = "python"
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip")
        return df, report
    except Exception as e:
        last_err = e
        for d in [",", ";", "\t", "|"]:
            if d == delim:
                continue
            try:
                report["delimiter"] = d if d != "\t" else "\\t"
                df = pd.read_csv(io.BytesIO(raw), sep=d, engine="python", on_bad_lines="skip")
                report["warning"] = (report["warning"] + f" | fallback delimiter {report['delimiter']}").strip(" |")
                return df, report
            except Exception as ee:
                last_err = ee
                continue
        raise last_err


def read_csv_path_robust(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    with open(path, "rb") as f:
        raw = f.read()
    return read_csv_bytes_robust(raw, name=os.path.basename(path))


def _coerce_int_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("$", "", regex=False)
    s = s.str.replace(" ", "", regex=False).str.replace("\u00a0", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _derive_slot_from_status(status: str) -> str:
    v = str(status or "").strip().upper()
    if not v:
        return "Actif"
    if "IR" in v or "INJ" in v:
        return "IR"
    if "MIN" in v or "AHL" in v or "PROS" in v or "NA" in v:
        return "Mineur"
    if "BENCH" in v or "BN" in v:
        return "Banc"
    return "Actif"


# -----------------------------
# Level map (hockey.players.csv)
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_level_map(pdb_path: str) -> Dict[str, str]:
    if not pdb_path or (not os.path.exists(pdb_path)):
        return {}
    try:
        df = pd.read_csv(pdb_path)
    except Exception:
        try:
            df, _ = read_csv_path_robust(pdb_path)
        except Exception:
            return {}

    name_col = None
    for c in ["Joueur", "Player", "Name", "Full Name", "Player Name"]:
        if c in df.columns:
            name_col = c
            break
    if not name_col:
        return {}

    level_col = None
    for c in ["Level", "Contract Level", "level"]:
        if c in df.columns:
            level_col = c
            break
    if not level_col:
        return {}

    m: Dict[str, str] = {}
    for _, row in df[[name_col, level_col]].dropna().iterrows():
        k = norm_player_key(row[name_col])
        v = str(row[level_col] or "").strip().upper()
        if v in ("ELC", "STD"):
            m[k] = v
    return m


def apply_level_auto(df: pd.DataFrame, pdb_path: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Joueur" not in df.columns:
        return df

    level_map = _load_level_map(pdb_path)
    if not level_map:
        return df

    if "Level" not in df.columns:
        df["Level"] = ""

    def _needs_fill(x) -> bool:
        s = str(x or "").strip()
        return s == "" or s in ("0", "0.0")

    mask = df["Level"].apply(_needs_fill)
    if mask.any():
        keys = df.loc[mask, "Joueur"].apply(norm_player_key)
        df.loc[mask, "Level"] = keys.map(level_map).fillna("STD")
    return df


# -----------------------------
# Normalisation Fantrax/Exports
# -----------------------------
def normalize_equipes_df(df_in: pd.DataFrame, owner: str, pdb_path: str) -> pd.DataFrame:
    """
    Assure que le CSV final contient au minimum:
      Propriétaire, Joueur, Pos, Salaire, Slot
    + conserve le reste des colonnes si présentes.
    """
    df = df_in.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # JOUEUR — PRIORITÉ STRICTE AU NOM COMPLET (évite IDs Fantrax)
    if "Player" in df.columns:
        df.rename(columns={"Player": "Joueur"}, inplace=True)
    elif "Name" in df.columns:
        df.rename(columns={"Name": "Joueur"}, inplace=True)
    elif "Player Name" in df.columns:
        df.rename(columns={"Player Name": "Joueur"}, inplace=True)
    elif "Full Name" in df.columns:
        df.rename(columns={"Full Name": "Joueur"}, inplace=True)
    elif "Skaters" in df.columns:
        df.rename(columns={"Skaters": "Joueur"}, inplace=True)
    else:
        df["Joueur"] = ""

    for src, dst in [
        (["Pos", "Position"], "Pos"),
        (["Team", "NHL Team", "Equipe", "Équipe"], "Equipe"),
        (["Salary", "Cap Hit", "CapHit", "Salaire"], "Salaire"),
        (["Status", "Roster Status", "Statut"], "Statut"),
        (["IR Date", "IRDate", "Date IR"], "IR Date"),
        (["Level", "Contract Level"], "Level"),
    ]:
        for c in src:
            if c in df.columns and dst not in df.columns:
                df.rename(columns={c: dst}, inplace=True)
                break

    if "Pos" not in df.columns:
        df["Pos"] = ""
    if "Salaire" not in df.columns:
        df["Salaire"] = 0

    df["Salaire"] = _coerce_int_series(df["Salaire"])

    if "Slot" not in df.columns:
        if "Statut" in df.columns:
            df["Slot"] = df["Statut"].apply(_derive_slot_from_status)
        else:
            df["Slot"] = "Actif"
    else:
        df["Slot"] = df["Slot"].astype(str).str.strip().replace({"": "Actif"})

    df["Propriétaire"] = str(owner).strip()
    df["Joueur"] = df["Joueur"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.strip()

    df = apply_level_auto(df, pdb_path)

    preferred = ["Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].copy()


def validate_min_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = ["Propriétaire", "Joueur", "Pos", "Salaire", "Slot"]
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0), missing


# -----------------------------
# Anti-doublon inter-équipes
# -----------------------------
def find_cross_team_duplicates(df_all: pd.DataFrame, incoming: pd.DataFrame, replace_team: str) -> List[Tuple[str, str]]:
    if df_all is None or df_all.empty:
        return []
    if "Joueur" not in df_all.columns or "Propriétaire" not in df_all.columns:
        return []
    if incoming is None or incoming.empty or "Joueur" not in incoming.columns:
        return []

    existing = df_all.copy()
    existing_team = existing["Propriétaire"].astype(str).str.strip()
    existing = existing[~existing_team.eq(str(replace_team).strip())].copy()

    ex_map: Dict[str, str] = {}
    for _, row in existing[["Joueur", "Propriétaire"]].dropna().iterrows():
        k = norm_player_key(row["Joueur"])
        if k and k not in ex_map:
            ex_map[k] = str(row["Propriétaire"])

    dups: List[Tuple[str, str]] = []
    for _, row in incoming[["Joueur"]].dropna().iterrows():
        k = norm_player_key(row["Joueur"])
        if not k:
            continue
        other = ex_map.get(k)
        if other:
            dups.append((str(row["Joueur"]), other))
    return dups


# -----------------------------
# Summary (GC/CE)
# -----------------------------
def _compute_gc_ce(df_team: pd.DataFrame) -> Tuple[int, int, int]:
    if df_team is None or df_team.empty or "Salaire" not in df_team.columns:
        return 0, 0, 0

    total = int(df_team["Salaire"].fillna(0).sum())
    gc = 0
    ce = 0

    if "Statut" in df_team.columns:
        s = df_team["Statut"].astype(str).str.upper()
        gc_mask = s.str.contains("GRAND") | s.str.contains("GC")
        ce_mask = s.str.contains("ECOLE") | s.str.contains("CE") | s.str.contains("CLUB ECOLE")
        gc = int(df_team.loc[gc_mask, "Salaire"].sum())
        ce = int(df_team.loc[ce_mask, "Salaire"].sum())
        if gc == 0 and ce == 0 and "Slot" in df_team.columns:
            slot = df_team["Slot"].astype(str).str.upper()
            ce = int(df_team.loc[slot.eq("MINEUR"), "Salaire"].sum())
            gc = total - ce
    else:
        if "Slot" in df_team.columns:
            slot = df_team["Slot"].astype(str).str.upper()
            ce = int(df_team.loc[slot.eq("MINEUR"), "Salaire"].sum())
            gc = total - ce
        else:
            gc = total

    return total, gc, ce


# -----------------------------
# Backup / rollback
# -----------------------------
def backup_team(df_all: pd.DataFrame, data_dir: str, season: str, team: str) -> Optional[str]:
    if df_all is None or df_all.empty or not team:
        return None
    if "Propriétaire" not in df_all.columns:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", team.strip())
    path = os.path.join(admin_backup_dir(data_dir, season), f"{safe}_{ts}.csv")

    sub = df_all[df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()
    if sub.empty:
        return None

    sub.to_csv(path, index=False)
    return path


def list_backups(data_dir: str, season: str, team: str) -> List[str]:
    bdir = admin_backup_dir(data_dir, season)
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", team.strip())
    files = [os.path.join(bdir, f) for f in os.listdir(bdir) if f.startswith(safe + "_") and f.lower().endswith(".csv")]
    files.sort(reverse=True)
    return files


# -----------------------------
# MAIN RENDER
# -----------------------------
def render(ctx: dict) -> None:
    _init_state()

    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    DATA_DIR = str(ctx.get("DATA_DIR") or "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    season = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    data_path = equipes_path(DATA_DIR, season)
    pdb_path = players_db_path(DATA_DIR)

    st.subheader("🛠️ Gestion Admin")

    with st.expander("📂 Import équipes depuis /data", expanded=True):
        st.caption("Importe les CSV d'équipes déjà présents dans /data. Lecture robuste + normalisation.")

        allow_override_dups = st.checkbox("Autoriser doublons (override)", value=False, key="adm_override_dups")

        data_csvs = sorted([
            f for f in os.listdir(DATA_DIR)
            if f.lower().endswith(".csv")
            and not f.lower().startswith("equipes_joueurs_")
            and f not in ("hockey.players.csv", "PuckPedia2025_26.csv")
        ])

        if not data_csvs:
            st.info("Aucun CSV d’équipe trouvé dans /data.")
        else:
            selected = st.multiselect("Fichiers équipes", data_csvs, default=data_csvs, key="adm_data_csv_select")

            if st.button("🧼 Préparer depuis /data", use_container_width=True, key="adm_prepare_from_data"):
                st.session_state["admin_prepared"] = []
                st.session_state["admin_last_parse_report"] = []

                for fname in selected:
                    path = os.path.join(DATA_DIR, fname)
                    try:
                        df_raw, rep = read_csv_path_robust(path)  # ✅ fix ParserError from default pd.read_csv
                        team = infer_team_from_filename(fname)

                        df_norm = normalize_equipes_df(df_raw, team, pdb_path)
                        ok, missing = validate_min_schema(df_norm)
                        if not ok:
                            st.error(f"❌ {fname}: colonnes manquantes après normalisation: {', '.join(missing)}")
                            st.write("Colonnes détectées:", list(df_norm.columns))
                            st.stop()

                        st.session_state["admin_prepared"].append({"filename": fname, "team": team, "df": df_norm})

                        warn = rep.get("warning") or ""
                        st.session_state["admin_last_parse_report"].append(
                            f"{fname}: enc={rep.get('encoding')} sep={rep.get('delimiter')} engine={rep.get('engine')}"
                            + (f" | {warn}" if warn else "")
                        )
                    except Exception as e:
                        st.error(f"❌ Erreur lecture {fname}")
                        st.exception(e)

                if st.session_state["admin_last_parse_report"]:
                    st.info("Rapport lecture:")
                    for line in st.session_state["admin_last_parse_report"]:
                        st.write("•", line)

                if st.session_state["admin_prepared"]:
                    st.success("✅ CSV préparés + normalisés. Ajuste l’équipe si besoin puis importe plus bas.")

    # -----------------------------
    # Preview + Import
    # -----------------------------
    if st.session_state["admin_prepared"]:
        st.markdown("## Attribution des équipes (et preview normalisé)")

        # load existing
        if os.path.exists(data_path):
            try:
                df_all_existing = pd.read_csv(data_path)
            except Exception:
                try:
                    df_all_existing, _ = read_csv_path_robust(data_path)
                except Exception:
                    df_all_existing = pd.DataFrame()
        else:
            df_all_existing = pd.DataFrame()

        for i, item in enumerate(st.session_state["admin_prepared"]):
            c1, c2 = st.columns([2, 4])
            with c1:
                new_team = st.text_input(f"Équipe pour {item['filename']}", value=item.get("team", ""), key=f"adm_team_{i}").strip()
                if new_team and new_team != item.get("team"):
                    item["team"] = new_team
                    item["df"]["Propriétaire"] = new_team
                st.caption(f"Lignes: {len(item['df'])}")

                dups = find_cross_team_duplicates(df_all_existing, item["df"], replace_team=item["team"])
                if dups:
                    st.error(f"🚫 Doublons détectés ({len(dups)}) — anti-triche")
                    st.write("Exemples:", ", ".join([f"{p} (déjà chez {t})" for p, t in dups[:5]]))
                else:
                    st.success("✅ Aucun doublon inter-équipes")

                total, gc, ce = _compute_gc_ce(item["df"])
                st.metric("Joueurs", len(item["df"]))
                st.metric("Masse totale", f"{total:,}".replace(",", " "))
                st.metric("GC estimé", f"{gc:,}".replace(",", " "))
                st.metric("CE estimé", f"{ce:,}".replace(",", " "))

            with c2:
                st.dataframe(item["df"].head(25), use_container_width=True)

        st.markdown("---")
        do_import = st.button("⬇️ Importer (remplacer équipe + backup)", use_container_width=True, key="adm_do_import")

        if do_import:
            # reload existing
            if os.path.exists(data_path):
                try:
                    df_all = pd.read_csv(data_path)
                except Exception:
                    try:
                        df_all, _ = read_csv_path_robust(data_path)
                    except Exception:
                        df_all = pd.DataFrame()
            else:
                df_all = pd.DataFrame()

            if df_all.empty:
                df_all = pd.DataFrame(columns=["Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"])

            all_dups = []
            for item in st.session_state["admin_prepared"]:
                team = str(item["team"]).strip()
                for player, other_team in find_cross_team_duplicates(df_all, item["df"], replace_team=team):
                    all_dups.append((team, player, other_team))

            if all_dups and not st.session_state.get("adm_override_dups", False):
                st.error("🚫 Import bloqué: doublons inter-équipes détectés (anti-triche).")
                st.write("Exemples:", ", ".join([f"{pl} (import {tm}, déjà chez {ot})" for tm, pl, ot in all_dups[:10]]))
                st.stop()

            imported_teams = []
            for item in st.session_state["admin_prepared"]:
                team = str(item.get("team") or "").strip()
                if not team:
                    st.error(f"Équipe manquante pour {item['filename']}")
                    st.stop()

                backup_team(df_all, DATA_DIR, season, team)

                df_all = df_all[~df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()

                df_new = item["df"].copy()
                df_new["Propriétaire"] = team
                df_new = apply_level_auto(df_new, pdb_path)

                df_all = pd.concat([df_all, df_new], ignore_index=True)
                imported_teams.append(team)

            df_all.to_csv(data_path, index=False)
            st.session_state["admin_prepared"] = []

            st.success(f"✅ Import terminé → {data_path}")
            st.info("Équipes importées: " + ", ".join(imported_teams))
            st.rerun()

    # -----------------------------
    # Rollback
    # -----------------------------
    with st.expander("↩️ Rollback par équipe", expanded=False):
        if not os.path.exists(data_path):
            st.info("Aucune équipe détectée (importe d'abord).")
            return

        try:
            df_all = pd.read_csv(data_path)
        except Exception:
            try:
                df_all, _ = read_csv_path_robust(data_path)
            except Exception:
                st.error("Impossible de lire le fichier équipes local.")
                return

        if "Propriétaire" not in df_all.columns:
            st.info("Colonne 'Propriétaire' absente.")
            return

        teams = sorted([t for t in df_all["Propriétaire"].dropna().astype(str).str.strip().unique().tolist() if t])
        if not teams:
            st.info("Aucune équipe détectée.")
            return

        team = st.selectbox("Équipe", teams, key="adm_rb_team")
        backups = list_backups(DATA_DIR, season, team)

        if not backups:
            st.info("Aucun backup pour cette équipe.")
            return

        pick = st.selectbox("Backup", backups, format_func=lambda p: os.path.basename(p), key="adm_rb_pick")
        if st.button("↩️ Restaurer ce backup", use_container_width=True, key="adm_rb_restore"):
            try:
                df_restore = pd.read_csv(pick)
            except Exception:
                try:
                    df_restore, _ = read_csv_path_robust(pick)
                except Exception as e:
                    st.error("Backup illisible.")
                    st.exception(e)
                    return

            df_all = df_all[~df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()
            df_restore["Propriétaire"] = team
            df_all = pd.concat([df_all, df_restore], ignore_index=True)
            df_all.to_csv(data_path, index=False)
            st.success("✅ Rollback effectué.")
            st.rerun()
