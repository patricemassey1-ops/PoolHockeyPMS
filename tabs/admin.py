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
#   ✅ Ne kick plus (tout déclenché par boutons)
#   ✅ Lecture CSV robuste (détecte delimiter + fallback python engine)
# ============================================================

# -----------------------------
# State
# -----------------------------
def _init_state():
    if "admin_prepared" not in st.session_state:
        st.session_state["admin_prepared"] = []  # list[dict(filename, team, df)]
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
    # Try csv.Sniffer first; fallback to scoring common delimiters
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        candidates = [",",";","\t","|"]
        scores = {d: sample.count(d) for d in candidates}
        # choose highest count
        return max(scores, key=scores.get) if scores else ","


def read_csv_robust(upload, *, force_delim: Optional[str]=None, skip_bad: bool=False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Lit un CSV de façon robuste.
    - Détecte delimiter (souvent ; si les nombres ont des virgules)
    - Fallback python engine si C-engine plante
    - Option skip_bad: ignore les lignes brisées (on_bad_lines='skip')
    Retour: (df, report)
    """
    name = getattr(upload, "name", "upload.csv")
    raw = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    # reset pointer if possible
    try:
        upload.seek(0)
    except Exception:
        pass

    # decode sample
    enc_used = None
    for enc in ("utf-8-sig","utf-8","latin-1"):
        try:
            text = raw.decode(enc)
            enc_used = enc
            break
        except Exception:
            continue
    if enc_used is None:
        # last resort
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

    # Try fast C engine
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="c")
        return df, report
    except ParserError as e:
        report["warning"] = f"C-engine ParserError: {e}"
    except Exception as e:
        report["warning"] = f"C-engine error: {e}"

    # Fallback python engine
    report["engine"] = "python"
    try:
        if skip_bad:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python", on_bad_lines="skip")
        else:
            df = pd.read_csv(io.BytesIO(raw), sep=delim, engine="python")
        return df, report
    except Exception as e:
        # One more attempt: try alternate delimiter if we guessed wrong
        alts = [",",";","\t","|"]
        if delim in alts:
            alts.remove(delim)
        for d in alts:
            try:
                report["delimiter"] = d
                if skip_bad:
                    df = pd.read_csv(io.BytesIO(raw), sep=d, engine="python", on_bad_lines="skip")
                else:
                    df = pd.read_csv(io.BytesIO(raw), sep=d, engine="python")
                report["warning"] = (report["warning"] + f" | fallback delimiter {d} worked").strip(" |")
                return df, report
            except Exception:
                continue
        raise


def equipes_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"equipes_joueurs_{season}.csv")


def _backup_dir(data_dir: str, season: str) -> str:
    d = os.path.join(data_dir, "admin_backups", season)
    os.makedirs(d, exist_ok=True)
    return d


def backup_team(df_all: pd.DataFrame, data_dir: str, season: str, team: str) -> Optional[str]:
    if df_all is None or df_all.empty or not team:
        return None
    bdir = _backup_dir(data_dir, season)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(bdir, f"{team}_{ts}.csv")
    sub = df_all[df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()
    if sub.empty:
        return None
    sub.to_csv(path, index=False)
    return path


def list_backups(data_dir: str, season: str, team: str) -> List[str]:
    bdir = _backup_dir(data_dir, season)
    if not os.path.isdir(bdir):
        return []
    files = [os.path.join(bdir, f) for f in os.listdir(bdir) if f.startswith(team + "_") and f.lower().endswith(".csv")]
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

    DATA_DIR = str(ctx.get("DATA_DIR") or "Data")
    os.makedirs(DATA_DIR, exist_ok=True)

    season = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"
    data_path = equipes_path(DATA_DIR, season)

    st.subheader("🛠️ Gestion Admin")

    # =====================================================
    # 📥 Import multi CSV
    # =====================================================
    with st.expander("📥 Import multi CSV équipes", expanded=True):
        st.caption("Astuce: si tes nombres contiennent des virgules (1,000,000), le fichier est souvent séparé par ';'.")
        delim_choice = st.selectbox("Delimiter (auto par défaut)", ["AUTO", ";", ",", "\t", "|"], index=0, key="adm_delim_choice")
        skip_bad = st.checkbox("Ignorer les lignes brisées (on_bad_lines='skip')", value=True, key="adm_skip_bad")

        files = st.file_uploader(
            "Uploader un ou plusieurs CSV (ex: Whalers.csv, Nordiques.csv)",
            type=["csv"],
            accept_multiple_files=True,
            key="adm_multi_upload",
        )

        prep = st.button("🧼 Préparer (analyse + attribution)", use_container_width=True, key="adm_prepare_btn")

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
                    df, rep = read_csv_robust(f, force_delim=force, skip_bad=skip_bad)

                    team = infer_team_from_filename(fname)
                    # force owner column at import time; here only keep in prepared
                    st.session_state["admin_prepared"].append({
                        "filename": fname,
                        "team": team,
                        "df": df,
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
                st.success("✅ Fichiers préparés. Ajuste l’équipe si besoin puis importe plus bas.")

        # Attribution + preview + import
        if st.session_state["admin_prepared"]:
            st.markdown("### Attribution des équipes")
            for i, item in enumerate(st.session_state["admin_prepared"]):
                c1, c2 = st.columns([2, 3])
                with c1:
                    item["team"] = st.text_input(
                        f"Équipe pour {item['filename']}",
                        value=item.get("team",""),
                        key=f"adm_team_{i}",
                    ).strip()
                    st.caption(f"Lignes: {len(item['df'])}")
                with c2:
                    st.dataframe(item["df"].head(15), use_container_width=True)

            do_import = st.button("⬇️ Importer (remplacer équipe + backup)", use_container_width=True, key="adm_do_import")
            if do_import:
                # load existing
                if os.path.exists(data_path):
                    df_all = pd.read_csv(data_path)
                else:
                    df_all = pd.DataFrame(columns=["Propriétaire"])

                for item in st.session_state["admin_prepared"]:
                    team = str(item.get("team") or "").strip()
                    if not team:
                        st.error(f"Équipe manquante pour {item['filename']}")
                        st.stop()

                    # backup current team
                    backup_team(df_all, DATA_DIR, season, team)

                    # replace team rows
                    if "Propriétaire" in df_all.columns:
                        df_all = df_all[~df_all["Propriétaire"].astype(str).str.strip().eq(team)].copy()

                    df_new = item["df"].copy()
                    df_new["Propriétaire"] = team
                    df_all = pd.concat([df_all, df_new], ignore_index=True)

                df_all.to_csv(data_path, index=False)
                st.session_state["admin_prepared"] = []
                st.success(f"✅ Import terminé → {data_path}")
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
            st.error("Impossible de lire le fichier équipes local (corrup?)")
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
