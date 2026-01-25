import os
import io
import re
import csv
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import streamlit as st
from pandas.errors import ParserError

# ============================================================
# ADMIN TAB — STABLE + MULTI IMPORT + AUTO-ASSIGN + ROLLBACK
#   ✅ Préparer -> attribuer -> importer (state conservé)
#   ✅ Lecture CSV robuste (AUTO delimiter + fallback python engine)
#   ✅ NORMALISATION Fantrax/exports -> schéma PMS (Propriétaire/Joueur/Pos/Salaire/Slot)
# ============================================================


# -----------------------------
# State
# -----------------------------
def _init_state():
    if "admin_prepared" not in st.session_state:
        st.session_state["admin_prepared"] = []  # list[dict(filename, team, df_norm)]
    if "admin_last_parse_report" not in st.session_state:
        st.session_state["admin_last_parse_report"] = []  # list[str]


# -----------------------------
# Helpers
# -----------------------------
def infer_team_from_filename(filename: str) -> str:
    if not filename:
        return ""
    return os.path.splitext(os.path.basename(filename))[0].strip()


def _guess_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        candidates = [",", ";", "\t", "|"]
        scores = {d: sample.count(d) for d in candidates}
        return max(scores, key=scores.get) if scores else ","


def read_csv_robust(upload, *, force_delim: Optional[str] = None, skip_bad: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Lecture CSV robuste (supporte , ; tab | + encodings)."""
    name = getattr(upload, "name", "upload.csv")
    raw = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    try:
        upload.seek(0)
    except Exception:
        pass

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
    delim = force_delim or _guess_delimiter(sample)

    report: Dict[str, Any] = {
        "file": name,
        "encoding": enc_used,
        "delimiter": delim,
        "engine": "c",
        "skip_bad": bool(skip_bad),
        "warning": "",
    }

    # Try C engine (fast)
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="c")
        return df, report
    except ParserError as e:
        report["warning"] = f"C-engine ParserError: {e}"
    except Exception as e:
        report["warning"] = f"C-engine error: {e}"

    # Fallback python engine (more tolerant)
    report["engine"] = "python"
    try:
        if skip_bad:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip")
        else:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python")
        return df, report
    except Exception:
        # Try alternate delimiters
        alts = [",", ";", "\t", "|"]
        if delim in alts:
            alts.remove(delim)
        last_err = None
        for d in alts:
            try:
                report["delimiter"] = d
                if skip_bad:
                    df = pd.read_csv(io.BytesIO(raw), sep=d, engine="python", on_bad_lines="skip")
                else:
                    df = pd.read_csv(io.BytesIO(raw), sep=d, engine="python")
                report["warning"] = (report["warning"] + f" | fallback delimiter {d} worked").strip(" |")
                return df, report
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        raise


def equipes_path(data_dir: str, season: str) -> str:
    # IMPORTANT: ton app lit data/equipes_joueurs_2025-2026.csv (underscore)
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")


def _backup_dir(data_dir: str, season: str) -> str:
    d = os.path.join(data_dir, "admin_backups", season)
    os.makedirs(d, exist_ok=True)
    return d


def backup_team(df_all: pd.DataFrame, data_dir: str, season: str, team: str) -> Optional[str]:
    if df_all is None or df_all.empty or not team:
        return None
    if "Propriétaire" not in df_all.columns:
        return None
    bdir = _backup_dir(data_dir, season)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", team.strip())
    path = os.path.join(bdir, f"{safe}_{ts}.csv")
    sub = df_all[df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()
    if sub.empty:
        return None
    sub.to_csv(path, index=False)
    return path


def list_backups(data_dir: str, season: str, team: str) -> List[str]:
    bdir = _backup_dir(data_dir, season)
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", team.strip())
    files = [os.path.join(bdir, f) for f in os.listdir(bdir) if f.startswith(safe + "_") and f.lower().endswith(".csv")]
    files.sort(reverse=True)
    return files


# ============================================================
# NORMALISATION Fantrax/Exports -> schéma PMS Pool
# ============================================================
def _coerce_int_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("$", "", regex=False)
    s = s.str.replace(" ", "", regex=False).str.replace("\u00a0", "", regex=False)
    # 1,000,000 or 1 000 000 -> 1000000
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


def normalize_equipes_df(df_in: pd.DataFrame, owner: str) -> pd.DataFrame:
    """Assure que le CSV final contient au minimum:
    Propriétaire, Joueur, Pos, Salaire, Slot
    + conserve le reste des colonnes si présentes.
    """
    df = df_in.copy()

    # Drop Unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # -----------------------------------------------------
    # JOUEUR — PRIORITÉ STRICTE AU NOM COMPLET
    #   (évite que Fantrax ID devienne "Joueur")
    # -----------------------------------------------------
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

    # Other common mappings
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

    # Minimum required columns
    if "Pos" not in df.columns:
        df["Pos"] = ""
    if "Salaire" not in df.columns:
        df["Salaire"] = 0

    # Salary normalize
    df["Salaire"] = _coerce_int_series(df["Salaire"])

    # Slot logic: prefer explicit Slot column; else derive from Statut; else default Actif
    if "Slot" not in df.columns:
        if "Statut" in df.columns:
            df["Slot"] = df["Statut"].apply(_derive_slot_from_status)
        else:
            df["Slot"] = "Actif"
    else:
        df["Slot"] = df["Slot"].astype(str).str.strip().replace({"": "Actif"})

    # Owner
    df["Propriétaire"] = str(owner).strip()

    # Nice ordering (keep extras at end)
    preferred = ["Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols].copy()

    # Final cleanup
    df["Joueur"] = df["Joueur"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.strip()

    return df


def validate_min_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = ["Propriétaire", "Joueur", "Pos", "Salaire", "Slot"]
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0), missing


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

    st.subheader("🛠️ Gestion Admin")

    # =====================================================
    # 📥 Import multi CSV (NORMALISE)
    # =====================================================
    with st.expander("📥 Import multi CSV équipes (normalisé)", expanded=True):
        st.caption("Supporte Fantrax/exports. Normalise automatiquement vers: Propriétaire, Joueur, Pos, Salaire, Slot.")

        delim_choice = st.selectbox("Delimiter (auto par défaut)", ["AUTO", ";", ",", "\t", "|"], index=0, key="adm_delim_choice")
        skip_bad = st.checkbox("Ignorer les lignes brisées (on_bad_lines='skip')", value=True, key="adm_skip_bad")

        files = st.file_uploader(
            "Uploader un ou plusieurs CSV (ex: Whalers.csv, Nordiques.csv)",
            type=["csv"],
            accept_multiple_files=True,
            key="adm_multi_upload",
        )

        prep = st.button("🧼 Préparer (lecture + normalisation + attribution)", use_container_width=True, key="adm_prepare_btn")

        if prep:
            st.session_state["admin_prepared"] = []
            st.session_state["admin_last_parse_report"] = []

            if not files:
                st.warning("Aucun fichier sélectionné.")
                st.stop()

            for f in files:
                fname = getattr(f, "name", "upload.csv")
                try:
                    force = None if delim_choice == "AUTO" else (delim_choice if delim_choice != "\t" else "\t")
                    df_raw, rep = read_csv_robust(f, force_delim=force, skip_bad=skip_bad)
                    team = infer_team_from_filename(fname)

                    df_norm = normalize_equipes_df(df_raw, team)

                    ok, missing = validate_min_schema(df_norm)
                    if not ok:
                        st.error(f"❌ {fname}: colonnes manquantes après normalisation: {', '.join(missing)}")
                        st.write("Colonnes détectées:", list(df_norm.columns))
                        st.stop()

                    st.session_state["admin_prepared"].append({
                        "filename": fname,
                        "team": team,
                        "df": df_norm,
                    })

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
                st.success("✅ Fichiers préparés + normalisés. Ajuste l’équipe si besoin puis importe plus bas.")

        # Attribution + preview + import
        if st.session_state["admin_prepared"]:
            st.markdown("## Attribution des équipes (et preview normalisé)")

            for i, item in enumerate(st.session_state["admin_prepared"]):
                c1, c2 = st.columns([2, 4])
                with c1:
                    new_team = st.text_input(
                        f"Équipe pour {item['filename']}",
                        value=item.get("team", ""),
                        key=f"adm_team_{i}",
                    ).strip()
                    if new_team and new_team != item.get("team"):
                        item["team"] = new_team
                        item["df"]["Propriétaire"] = new_team
                    st.caption(f"Lignes: {len(item['df'])}")
                with c2:
                    st.dataframe(item["df"].head(25), use_container_width=True)

            do_import = st.button("⬇️ Importer (remplacer équipe + backup)", use_container_width=True, key="adm_do_import")
            if do_import:
                # load existing
                if os.path.exists(data_path):
                    try:
                        df_all = pd.read_csv(data_path)
                    except Exception:
                        df_all = pd.DataFrame()
                else:
                    df_all = pd.DataFrame()

                if df_all.empty:
                    df_all = pd.DataFrame(columns=["Propriétaire", "Joueur", "Pos", "Equipe", "Salaire", "Level", "Statut", "Slot", "IR Date"])

                for item in st.session_state["admin_prepared"]:
                    team = str(item.get("team") or "").strip()
                    if not team:
                        st.error(f"Équipe manquante pour {item['filename']}")
                        st.stop()

                    backup_team(df_all, DATA_DIR, season, team)

                    if "Propriétaire" in df_all.columns:
                        df_all = df_all[~df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()

                    df_new = item["df"].copy()
                    df_new["Propriétaire"] = team
                    df_all = pd.concat([df_all, df_new], ignore_index=True)

                df_all.to_csv(data_path, index=False)
                st.session_state["admin_prepared"] = []
                st.success(f"✅ Import terminé (normalisé) → {data_path}")
                st.rerun()

    # =====================================================
    # ↩️ Rollback
    # =====================================================
    with st.expander("↩️ Rollback par équipe", expanded=False):
        if not os.path.exists(data_path):
            st.info("Aucune équipe détectée (importe d'abord).")
            return

        try:
            df_all = pd.read_csv(data_path)
        except Exception:
            st.error("Impossible de lire le fichier équipes local (corrup?).")
            st.stop()

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
            except Exception as e:
                st.error("Backup illisible.")
                st.exception(e)
                st.stop()

            df_all = df_all[~df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()
            df_restore["Propriétaire"] = team
            df_all = pd.concat([df_all, df_restore], ignore_index=True)
            df_all.to_csv(data_path, index=False)
            st.success("✅ Rollback effectué.")
            st.rerun()
