# tabs/admin.py — NHL_ID Tools (SAFE) + Source Dropdown Fix + No "écran noir"
# -----------------------------------------------------------------------------
# Goals:
# 1) Never crash UI (global try/except in render)
# 2) Fix StreamlitDuplicateElementId (unique keys everywhere)
# 3) Source dropdown lists ALL CSVs in /data AND always includes nhl_search_players.csv + equipes_joueurs_*.csv
# 4) Associer NHL_ID works even if NHL_ID column is absent in target (it will be created)
# 5) Supports french column "Joueur" (and many others) as name column
# 6) SAFE MODE prevents catastrophic write (0 NHL_ID after operation)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import io
import json
import time
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

WKEY = "admin_nhlid_"  # widget key prefix (unique)


# =========================
# I/O
# =========================
def load_csv(path: str) -> Tuple[pd.DataFrame, str | None]:
    try:
        if not path:
            return pd.DataFrame(), "Chemin CSV vide."
        if not os.path.exists(path):
            return pd.DataFrame(), f"Fichier introuvable: {path}"
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Erreur lecture CSV: {type(e).__name__}: {e}"


def save_csv(df: pd.DataFrame, path: str, *, safe_mode: bool = True, allow_zero: bool = False) -> str | None:
    """SAFE MODE: refuse to write if NHL_ID exists but would be 0% filled (unless allow_zero)."""
    try:
        if safe_mode and not allow_zero:
            id_col = _resolve_nhl_id_col(df)
            if id_col and id_col in df.columns:
                s = pd.to_numeric(df[id_col], errors="coerce")
                if int(s.notna().sum()) == 0:
                    return "SAFE MODE: Refus d'écrire — NHL_ID serait 0/100% (colonne vide)."
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        return None
    except Exception as e:
        return f"Erreur écriture CSV: {type(e).__name__}: {e}"


# =========================
# Column resolution
# =========================

# =========================
# Master Builder helpers
# =========================
def _atomic_write_df(df: pd.DataFrame, out_path: str) -> Tuple[bool, str | None]:
    """Atomic CSV write to avoid partial files if crash."""
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8", newline="") as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _pick_name_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["player", "joueur", "skaters", "name", "full_name", "fullname"]:
        if key in cmap:
            return cmap[key]
    return None

def _make_row_key(df: pd.DataFrame) -> pd.Series:
    """Stable key for diff: prefer NHL_ID when present, else normalized player name."""
    if df is None or df.empty:
        return pd.Series([], dtype=str)
    name_col = _pick_name_col(df) or df.columns[0]
    names = df[name_col].astype(str).map(_normalize_player_name)

    if "NHL_ID" in df.columns:
        ids = df["NHL_ID"].astype(str).str.strip()
    else:
        ids = pd.Series([""] * len(df))

    # NHL_ID dominates when present
    key = np.where(ids.astype(str).str.strip() != "", "NHL:" + ids.astype(str).str.strip(), "NAME:" + names)
    return pd.Series(key, index=df.index, dtype=str)

def _safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})

def _build_diff_and_audit(before: pd.DataFrame, after: pd.DataFrame, max_rows: int = 50000, compare_cols: Optional[List[str]] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Returns (summary_dict, audit_df).
    audit_df columns: key, change_type, field, before, after
    """
    summary: Dict[str, Any] = {
        "before_rows": int(len(before)) if isinstance(before, pd.DataFrame) else 0,
        "after_rows": int(len(after)) if isinstance(after, pd.DataFrame) else 0,
        "added": 0,
        "removed": 0,
        "modified_rows": 0,
        "audit_truncated": False,
    }

    if not isinstance(after, pd.DataFrame) or after.empty:
        return summary, pd.DataFrame(columns=["key", "change_type", "field", "before", "after"])

    before = before if isinstance(before, pd.DataFrame) else pd.DataFrame()
    after = after.copy()

    bkey = _make_row_key(before) if not before.empty else pd.Series([], dtype=str)
    akey = _make_row_key(after)

    if not before.empty:
        b = before.copy()
        b["_row_key"] = bkey.values
    else:
        b = pd.DataFrame(columns=["_row_key"])

    a = after.copy()
    a["_row_key"] = akey.values

    # Key sets
    bset = set(b["_row_key"].dropna().astype(str).tolist()) if "_row_key" in b.columns else set()
    aset = set(a["_row_key"].dropna().astype(str).tolist()) if "_row_key" in a.columns else set()

    added_keys = sorted(list(aset - bset))
    removed_keys = sorted(list(bset - aset))
    common_keys = sorted(list(aset & bset))

    summary["added"] = len(added_keys)
    summary["removed"] = len(removed_keys)

    # Choose columns to compare (important ones first)
    default_cols_interest = [
        "Player", "Joueur", "Team", "Équipe", "Position", "Jersey#", "Country",
        "Level", "Cap Hit", "Length", "Start Year", "Signing Status", "Expiry Year", "Expiry Status",
        "Status",
    ]
    # If user selected specific columns in Admin, use them; else use defaults.
    cols_interest = list(compare_cols) if (compare_cols and len(compare_cols) > 0) else default_cols_interest
    # Keep only columns that exist in either frame (case-sensitive)
    cols_existing = []
    for c in cols_interest:
        if c in after.columns or (not before.empty and c in before.columns):
            cols_existing.append(c)

    # Also include NHL_ID if present
    if "NHL_ID" in after.columns or (not before.empty and "NHL_ID" in before.columns):
        cols_existing = ["NHL_ID"] + [c for c in cols_existing if c != "NHL_ID"]

    # Build index by key for before/after (dedupe by first occurrence)
    if not b.empty:
        b_idx = b.drop_duplicates("_row_key", keep="first").set_index("_row_key", drop=True)
    else:
        b_idx = pd.DataFrame().set_index(pd.Index([], name="_row_key"))

    a_idx = a.drop_duplicates("_row_key", keep="first").set_index("_row_key", drop=True)

    audit_rows: List[Dict[str, Any]] = []

    def _row_json(df_row: pd.Series) -> str:
        try:
            d = {k: ("" if pd.isna(v) else v) for k, v in df_row.to_dict().items()}
            # don't include huge blobs
            d.pop("_row_key", None)
            return json.dumps(d, ensure_ascii=False)
        except Exception:
            return ""

    # Added / Removed
    for k in added_keys:
        if len(audit_rows) >= max_rows:
            summary["audit_truncated"] = True
            break
        row = a_idx.loc[k] if k in a_idx.index else None
        audit_rows.append({"key": k, "change_type": "added", "field": "__row__", "before": "", "after": _row_json(row)})

    for k in removed_keys:
        if len(audit_rows) >= max_rows:
            summary["audit_truncated"] = True
            break
        row = b_idx.loc[k] if (not b.empty and k in b_idx.index) else None
        audit_rows.append({"key": k, "change_type": "removed", "field": "__row__", "before": _row_json(row), "after": ""})

    # Modified fields (common)
    modified_rows_set = set()
    for k in common_keys:
        if len(audit_rows) >= max_rows:
            summary["audit_truncated"] = True
            break
        brow = b_idx.loc[k] if (not b.empty and k in b_idx.index) else None
        arow = a_idx.loc[k] if k in a_idx.index else None
        if brow is None or arow is None:
            continue

        for col in cols_existing:
            if len(audit_rows) >= max_rows:
                summary["audit_truncated"] = True
                break
            bval = _to_str(brow.get(col, ""))
            aval = _to_str(arow.get(col, ""))
            # Normalize common empties
            if bval.lower() in ["nan", "none"]:
                bval = ""
            if aval.lower() in ["nan", "none"]:
                aval = ""
            if bval != aval:
                modified_rows_set.add(k)
                audit_rows.append({
                    "key": k,
                    "change_type": "modified",
                    "field": col,
                    "before": bval,
                    "after": aval,
                })

    summary["modified_rows"] = len(modified_rows_set)

    audit_df = pd.DataFrame(audit_rows, columns=["key", "change_type", "field", "before", "after"])
    return summary, audit_df


def _norm_col(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _resolve_nhl_id_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in [
        "nhl_id", "nhlid", "id_nhl", "player_id", "playerid", "nhlplayerid",
        "nhl_id_api", "nhl_id_nhl", "nhl_player_id",
    ]:
        if key in cmap:
            return cmap[key]
    # common variants with spaces
    for c in df.columns:
        if str(c).strip().lower() in ("nhl_id", "nhl id", "nhl-id"):
            return c
    return None


def _resolve_player_name_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    # include french 'joueur'
    for key in ["joueur", "player", "player_name", "nom", "name", "full_name", "playername"]:
        if key in cmap:
            return cmap[key]
    return None


def _resolve_team_col(df: pd.DataFrame) -> str | None:
    """Prefer nul_id as team column if present (per user requirement)."""
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["nul_id", "nulid"]:
        if key in cmap:
            return cmap[key]
    for key in ["team", "equipe", "club", "nhl_team", "team_abbrev", "owner", "proprietaire"]:
        if key in cmap:
            return cmap[key]
    return None


def _normalize_player_name(x: Any) -> str:
    x = str(x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    # handle "Last, First"
    if "," in x:
        a, b = [p.strip() for p in x.split(",", 1)]
        if a and b:
            x = f"{b} {a}"
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# =========================
# Source dropdown discovery
# =========================
def list_data_csvs(data_dir: str = "data") -> List[str]:
    """Return sorted list of CSV paths in data_dir, ensuring key files are present if they exist."""
    paths: List[str] = []
    try:
        if os.path.isdir(data_dir):
            for fn in os.listdir(data_dir):
                if fn.lower().endswith(".csv"):
                    paths.append(os.path.join(data_dir, fn))
    except Exception:
        pass

    # ensure important sources are present if exist
    must = [
        os.path.join(data_dir, "nhl_search_players.csv"),
        os.path.join(data_dir, "nhl_search_players_2025-2026.csv"),
        os.path.join(data_dir, "equipes_joueurs_2025-2026.csv"),
        os.path.join(data_dir, "hockey.players.csv"),
    ]
    for p in must:
        if os.path.exists(p) and p not in paths:
            paths.append(p)

    # stable sort: important first, then alpha
    def _rank(p: str) -> Tuple[int, str]:
        base = os.path.basename(p).lower()
        if base.startswith("nhl_search_players"):
            r = 0
        elif base.startswith("hockey.players"):
            r = 1
        elif base.startswith("equipes_joueurs"):
            r = 2
        else:
            r = 9
        return (r, base)

    paths = sorted(set(paths), key=_rank)
    return paths


# =========================
# Matching from source
# =========================
def recover_from_source(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    target_id_col: str,
    target_name_col: str,
    source_id_col: str,
    source_name_col: str,
    conf: float,
    source_tag: str,
    max_fill: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = target_df.copy()

    # ensure NHL_ID exists
    if target_id_col not in out.columns:
        out[target_id_col] = np.nan

    # tracking cols
    if "nhl_id_source" not in out.columns:
        out["nhl_id_source"] = ""
    if "nhl_id_confidence" not in out.columns:
        out["nhl_id_confidence"] = np.nan

    s = source_df.copy()
    s["_k_name"] = s[source_name_col].map(_normalize_player_name)
    s["_id"] = pd.to_numeric(s[source_id_col], errors="coerce")
    s = s.dropna(subset=["_id"]).drop_duplicates(subset=["_k_name"], keep="first")

    out["_k_name"] = out[target_name_col].map(_normalize_player_name)
    cur = pd.to_numeric(out[target_id_col], errors="coerce")

    miss_mask = cur.isna()
    idx_miss = out.index[miss_mask].tolist()
    if max_fill and len(idx_miss) > int(max_fill):
        idx_miss = idx_miss[: int(max_fill)]

    miss_slice = out.loc[idx_miss, ["_k_name"]].merge(s[["_k_name", "_id"]], on="_k_name", how="left")
    fill_mask = miss_slice["_id"].notna()

    filled = int(fill_mask.sum())
    if filled:
        out.loc[miss_slice.index[fill_mask], target_id_col] = miss_slice.loc[fill_mask, "_id"].values
        out.loc[miss_slice.index[fill_mask], "nhl_id_source"] = source_tag
        out.loc[miss_slice.index[fill_mask], "nhl_id_confidence"] = float(conf)

    out = out.drop(columns=["_k_name"], errors="ignore")
    return out, {"filled": filled}


def audit_nhl_ids(df: pd.DataFrame, id_col: str) -> Dict[str, Any]:
    s = pd.to_numeric(df[id_col], errors="coerce")
    total = int(len(df))
    with_id = int(s.notna().sum())
    missing = total - with_id
    dup_cnt = int(s.dropna().duplicated().sum())
    dup_pct = (dup_cnt / max(with_id, 1)) * 100.0
    miss_pct = (missing / max(total, 1)) * 100.0
    return {
        "total": total,
        "with_id": with_id,
        "missing": missing,
        "missing_pct": miss_pct,
        "dup_cnt": dup_cnt,
        "dup_pct": dup_pct,
    }


# =========================
# NHL Search API generator (optional)
# =========================
def _http_get_json(url: str, timeout: int = 20) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (PoolHockeyPMS)", "Accept": "application/json,text/plain,*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return json.loads(raw)


def _extract_items(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ["data", "players", "items", "results"]:
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def generate_nhl_search_source(
    out_path: str,
    *,
    active_only: bool = True,
    limit: int = 1000,
    timeout_s: int = 20,
    max_pages: int = 20,
    culture: str = "en-us",
    q: str = "*",
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
    base = "https://search.d3.nhle.com/api/v1/search/player"
    all_rows: List[dict] = []
    seen_ids: set[int] = set()
    pages = 0
    start = 0
    used_url = ""

    try:
        while pages < max_pages:
            params = {
                "culture": culture,
                "limit": str(int(limit)),
                "q": q,
                "active": "True" if active_only else "False",
                "start": str(int(start)),
            }
            url = f"{base}?{urllib.parse.urlencode(params)}"
            used_url = url

            payload = _http_get_json(url, timeout=timeout_s)
            items = _extract_items(payload)

            if pages == 0 and active_only and len(items) == 0:
                active_only = False
                continue

            if not items:
                break

            new_count = 0
            for it in items:
                nhl_id = it.get("playerId", it.get("id", it.get("player_id", it.get("NHL_ID"))))
                try:
                    nhl_id_int = int(nhl_id)
                except Exception:
                    continue
                if nhl_id_int in seen_ids:
                    continue
                seen_ids.add(nhl_id_int)
                new_count += 1

                all_rows.append(
                    {
                        "NHL_ID": nhl_id_int,
                        "Player": it.get("name", it.get("fullName", it.get("playerName", ""))),
                        "Team": it.get("teamAbbrev", it.get("team", "")),
                        "Position": it.get("positionCode", it.get("position", "")),
                        "Jersey#": it.get("sweaterNumber", it.get("jerseyNumber", "")),
                        "DOB": it.get("birthDate", it.get("dob", "")),
                        "_source": "nhl_search_api",
                    }
                )

            pages += 1
            if new_count == 0:
                break
            if len(items) < int(limit):
                break
            start += int(limit)
            time.sleep(0.05)

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df["NHL_ID"] = pd.to_numeric(df["NHL_ID"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["NHL_ID"]).drop_duplicates(subset=["NHL_ID"], keep="first")

        errw = save_csv(df, out_path, safe_mode=False, allow_zero=True)
        if errw:
            return pd.DataFrame(), {"url": used_url, "pages": pages, "rows": int(len(df))}, errw

        return df, {"url": used_url, "pages": pages, "rows_saved": int(len(df)), "out_path": out_path}, None

    except Exception as e:
        return pd.DataFrame(), {"url": used_url, "pages": pages}, f"Erreur NHL Search API: {type(e).__name__}: {e}"


# =========================
# UI (render)
# =========================
def render(*args, **kwargs):
    try:
        return _render_impl(*args, **kwargs)
    except Exception as e:
        st.error("Une erreur a été détectée (évite l’écran noir).")
        st.exception(e)
        return None


def _render_impl(ctx: Optional[Dict[str, Any]] = None):
    ctx = ctx or {}
    season = str(ctx.get("season") or "2025-2026")
    data_dir = str(ctx.get("data_dir") or "data")

    st.subheader("🛠️ Outils — synchros (NHL_ID)")

    # =====================================================
    # 🧱 Master Builder (hockey.players_master.csv)
    #   - Fusion hockey.players.csv + PuckPedia2025_26.csv + NHL API (optionnel)
    #   - Aperçu diff avant/après
    #   - Rapport audit CSV: data/master_build_report.csv
    # =====================================================
    with st.expander("🧱 Master Builder", expanded=False):
        master_path = os.path.join(data_dir, "hockey.players_master.csv")
        report_path = os.path.join(data_dir, "master_build_report.csv")

        st.caption("Fusionne **hockey.players.csv + PuckPedia2025_26.csv + NHL API** → **hockey.players_master.csv**.")
        st.caption("📄 Audit écrit dans **data/master_build_report.csv** (ajouts / suppressions / champs modifiés).")

        before_df = pd.DataFrame()
        if os.path.exists(master_path):
            before_df, err = load_csv(master_path)
            if err:
                st.warning(f"Avant: {err}")
            else:
                st.info(f"Avant: master existant ✅ ({len(before_df)} lignes)")
        else:
            st.info("Avant: aucun master trouvé (il sera créé).")

        # Colonnes à comparer (diff) — mode lisible
        default_compare = ["Level", "Cap Hit", "Expiry Year", "Expiry Status", "Team", "Position", "Jersey#", "Country", "Status", "NHL_ID"]
        opts_base = []
        try:
            if isinstance(before_df, pd.DataFrame) and (not before_df.empty):
                opts_base = list(before_df.columns)
        except Exception:
            opts_base = []
        opts = sorted(set(opts_base) | set(default_compare) | {"Player"})
        compare_cols = st.multiselect(
            "Colonnes à comparer dans le diff (laisser vide = colonnes clés par défaut)",
            options=opts,
            default=[c for c in default_compare if c in opts],
            key=WKEY + "mb_compare_cols",
        )

        colA, colB, colC = st.columns([1.1, 1.1, 1.2])
        with colA:
            enrich = st.checkbox("Enrichir via NHL API", value=True, key=WKEY + "mb_enrich")
        with colB:
            max_calls = st.number_input("Max appels NHL", min_value=0, max_value=5000, value=250, step=50, key=WKEY + "mb_max_calls")
        with colC:
            st.write("")
            st.write("")
            run_btn = st.button("🧱 Construire / Mettre à jour Master", type="primary", key=WKEY + "mb_build")

        if run_btn:
            # lazy import (évite crash si module absent)
            try:
                from services.master_builder import build_master, MasterBuildConfig
            except Exception as e:
                st.error("Impossible d'importer services.master_builder. Assure-toi que le fichier est dans /services/master_builder.py.")
                st.exception(e)
            else:
                cfg = MasterBuildConfig(
                    data_dir=data_dir,
                    enrich_from_nhl=bool(enrich),
                    max_nhl_calls=int(max_calls),
                )
                with st.spinner("Fusion + enrichissement…"):
                    after_df, rep = build_master(cfg)

                st.success("✅ Master généré: data/hockey.players_master.csv")
                st.json(rep)

                # Diff + audit
                summary, audit_df = _build_diff_and_audit(before_df, after_df, max_rows=50000, compare_cols=compare_cols)

                # Write audit report CSV
                ok, werr = _atomic_write_df(audit_df, report_path)
                if ok:
                    st.success(f"🧾 Audit écrit: {report_path} ({len(audit_df)} lignes)")
                    rep_bytes = _read_file_bytes(report_path)
                    if rep_bytes:
                        st.download_button(
                            "📥 Télécharger rapport CSV (audit fusion)",
                            data=rep_bytes,
                            file_name=os.path.basename(report_path),
                            mime="text/csv",
                            use_container_width=True,
                        )
                else:
                    st.error(f"❌ Échec écriture audit: {werr}")

                # Preview diff (avant/après)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lignes avant", summary.get("before_rows", 0))
                c2.metric("Lignes après", summary.get("after_rows", 0))
                c3.metric("Ajouts", summary.get("added", 0))
                c4.metric("Suppressions", summary.get("removed", 0))

                st.metric("Lignes modifiées (au moins 1 champ)", summary.get("modified_rows", 0))

                if summary.get("audit_truncated"):
                    st.warning("⚠️ Audit tronqué (trop de changements). Le fichier contient la première portion seulement.")
                # Détails (plus propre) — tout est dans un seul expander
                with st.expander("📄 Détails (diff + aperçus)", expanded=False):
                    if not audit_df.empty:
                        st.markdown("**Aperçu des changements (top 200)**")
                        st.dataframe(audit_df.head(200), use_container_width=True)
                    else:
                        st.info("Aucun changement détecté (ou master créé identique).")

                    st.markdown("**Aperçu du master (top 50)**")
                    st.dataframe(after_df.head(50), use_container_width=True)


    # --- Generator (optional)
    st.markdown("### 🌐 Générer source NHL_ID (NHL Search API)")
    out_src = os.path.join(data_dir, "nhl_search_players.csv")
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
    with c2:
        active_only = st.checkbox("Actifs seulement", value=True, key=WKEY + "gen_active")
    with c3:
        limit = st.number_input("Chunk limit", min_value=200, max_value=2000, value=1000, step=100, key=WKEY + "gen_limit")
    with c4:
        timeout_s = st.number_input("Timeout (s)", min_value=5, max_value=60, value=20, step=5, key=WKEY + "gen_timeout")

    if c1.button("🌐 Générer source NHL_ID", use_container_width=True, key=WKEY + "btn_gen"):
        with st.spinner("Génération en cours…"):
            df_out, dbg, err = generate_nhl_search_source(
                out_src, active_only=bool(active_only), limit=int(limit), timeout_s=int(timeout_s), max_pages=25
            )
        if err:
            st.error(err)
            if dbg.get("url"):
                st.caption(f"URL: {dbg.get('url')}")
        else:
            st.success(f"✅ Généré: {dbg.get('rows_saved', 0)} joueurs (pages={dbg.get('pages', 0)}).")
            st.caption(f"Sortie: {out_src}")
            st.caption(f"URL: {dbg.get('url')}")
            if not df_out.empty:
                st.dataframe(df_out.head(10), use_container_width=True)

    st.markdown("---")

    # --- Target file
    csvs = list_data_csvs(data_dir)
    default_target = os.path.join(data_dir, f"equipes_joueurs_{season}.csv")
    if not os.path.exists(default_target):
        default_target = os.path.join(data_dir, "equipes_joueurs_2025-2026.csv") if os.path.exists(os.path.join(data_dir, "equipes_joueurs_2025-2026.csv")) else (csvs[0] if csvs else "")

    target_path = st.selectbox(
        "Players DB (NHL_ID) — fichier cible",
        options=csvs if csvs else [default_target],
        index=(csvs.index(default_target) if csvs and default_target in csvs else 0),
        key=WKEY + "target",
    )

    max_per_run = st.number_input("Max par run", min_value=50, max_value=20000, value=1000, step=50, key=WKEY + "maxrun")
    dry_run = st.checkbox("Dry-run (ne sauvegarde pas)", value=False, key=WKEY + "dry")
    override_safe = st.checkbox("Override SAFE MODE (autoriser une baisse NHL_ID)", value=False, key=WKEY + "override_safe")

    st.caption(f"🔒 Prod lock: OFF (ENV/PMS_ENV).")

    # --- Source dropdown (optional)
    st.markdown("### Source de récupération (optionnel)")
    upload = st.file_uploader("Ou uploader un CSV source", type=["csv"], key=WKEY + "upload")
    uploaded_df = None
    uploaded_name = None
    if upload is not None:
        try:
            uploaded_df = pd.read_csv(upload, low_memory=False)
            uploaded_name = f"upload:{upload.name}"
            st.success(f"✅ Upload chargé: {upload.name} ({len(uploaded_df)} lignes)")
        except Exception as e:
            st.error(f"Upload invalide: {type(e).__name__}: {e}")

    # Build source options = all csvs EXCEPT target, plus (None), plus upload option label
    src_opts = ["(Aucune — API NHL uniquement)"] + [p for p in csvs if p != target_path]
    # ensure nhl_search_players is present when exists
    must_src = os.path.join(data_dir, "nhl_search_players.csv")
    if os.path.exists(must_src) and must_src != target_path and must_src not in src_opts:
        src_opts.insert(1, must_src)

    default_src = must_src if (os.path.exists(must_src) and must_src != target_path) else ("(Aucune — API NHL uniquement)")
    src_choice = st.selectbox(
        "Récupérer NHL_ID depuis…",
        options=src_opts,
        index=(src_opts.index(default_src) if default_src in src_opts else 0),
        key=WKEY + "source",
    )

    # Load target
    df_t, err_t = load_csv(target_path)
    if err_t:
        st.error(err_t)
        return

    # Resolve columns on target
    t_name_col = _resolve_player_name_col(df_t)
    if not t_name_col:
        st.error("Colonne nom joueur introuvable dans le fichier cible (ex: Joueur / Player / player_name).")
        st.caption(f"Colonnes détectées: {list(df_t.columns)}")
        return

    # Determine NHL_ID column name for target (create if missing)
    t_id_col = _resolve_nhl_id_col(df_t) or "NHL_ID"
    created_id = False
    if t_id_col not in df_t.columns:
        df_t[t_id_col] = np.nan
        created_id = True
        st.info("Colonne NHL_ID absente → créée (prête à remplir).")

    # Load source df
    source_df = None
    source_tag = ""
    if uploaded_df is not None:
        source_df = uploaded_df
        source_tag = uploaded_name or "upload"
    elif src_choice and src_choice != "(Aucune — API NHL uniquement)":
        source_df, err_s = load_csv(src_choice)
        if err_s:
            st.error(err_s)
            source_df = None
        else:
            source_tag = os.path.basename(src_choice)

    # If source==target (should not happen, but guard)
    if src_choice == target_path:
        st.warning("Source = fichier cible. Choisis une autre source (ex: nhl_search_players.csv).")
        source_df = None

    # Controls
    conf = st.slider("Score de confiance appliqué aux IDs récupérés", 0.50, 0.99, 0.85, 0.01, key=WKEY + "conf")
    dup_lock = st.slider("🔴 Seuil blocage duplication (%)", 0.5, 20.0, 5.0, 0.5, key=WKEY + "duplock")

    # Button
    if st.button("🔗 Associer NHL_ID", key=WKEY + "btn_assoc"):
        if source_df is None or source_df.empty:
            st.warning("Aucune source exploitable sélectionnée. Sélectionne nhl_search_players.csv ou upload un CSV avec NHL_ID.")
            return

        s_id_col = _resolve_nhl_id_col(source_df)
        s_name_col = _resolve_player_name_col(source_df)

        if not s_id_col or s_id_col not in source_df.columns:
            st.error("Source: colonne NHL_ID introuvable.")
            st.caption(f"Colonnes source: {list(source_df.columns)}")
            return
        if not s_name_col or s_name_col not in source_df.columns:
            st.error("Source: colonne nom joueur introuvable (Player/Joueur/Name).")
            st.caption(f"Colonnes source: {list(source_df.columns)}")
            return

        # --- Stats AVANT
        a0 = audit_nhl_ids(df_t, t_id_col)
        st.caption(f"Avant: total={a0['total']}, avec NHL_ID={a0['with_id']}, manquants={a0['missing']} ({a0['missing_pct']:.1f}%), doublons={a0['dup_cnt']} ({a0['dup_pct']:.1f}%).")

        df2, stats = recover_from_source(
            df_t,
            source_df,
            target_id_col=t_id_col,
            target_name_col=t_name_col,
            source_id_col=s_id_col,
            source_name_col=s_name_col,
            conf=float(conf),
            source_tag=source_tag or "source",
            max_fill=int(max_per_run) if int(max_per_run) > 0 else 0,
        )

        a = audit_nhl_ids(df2, t_id_col)
        st.success(f"✅ Récupération terminée: +{int(stats.get('filled', 0))} IDs remplis. Doublons: {a['dup_cnt']} (~{a['dup_pct']:.1f}%).")

        if a["dup_pct"] > float(dup_lock):
            st.error(f"🛑 Blocage duplication: {a['dup_pct']:.1f}% > seuil {dup_lock:.1f}%. Ajuste la source/stratégie avant write.")
            # do not write
            return

        if dry_run:
            st.info("Dry-run: aucune écriture.")
            return

        errw = save_csv(df2, target_path, safe_mode=(not override_safe), allow_zero=False)
        if errw:
            st.error(errw)
            return

        st.success(f"💾 Sauvegardé: {target_path}")

        # Download small audit report
        audit_df = pd.DataFrame({
            "player_name": df2[t_name_col].astype(str),
            "nhl_id": pd.to_numeric(df2[t_id_col], errors="coerce"),
            "missing": pd.to_numeric(df2[t_id_col], errors="coerce").isna(),
            "source": df2.get("nhl_id_source", ""),
            "confidence": df2.get("nhl_id_confidence", np.nan),
        })
        buf = io.StringIO()
        audit_df.to_csv(buf, index=False)
        st.download_button(
            "🧾 Télécharger audit NHL_ID (CSV)",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=WKEY + "dl_audit",
        )


# Backward-compat alias (if app expects _render_tools)
def _render_tools(*args, **kwargs):
    return render(*args, **kwargs)
