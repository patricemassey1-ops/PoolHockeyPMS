# tabs/admin.py — SAFE MINI (NHL_ID tools + prod lock + audit/heatmap)
# ✅ Fixes:
#   - Heatmap crash (matplotlib optional)
#   - Robust column detection (incl. nul_id -> team)
#   - Source scoring + auto “best default source”
#   - MATCHING amélioré: name, name+team, name+DOB, team+pos+jersey
#   - Mode “review collisions” (montre collisions avant write)
#   - Garde toutes les fonctions existantes (ajout uniquement, pas de suppression)

from __future__ import annotations

import os
import re
import io
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Small I/O helpers
# =========================
def load_csv(path: str) -> Tuple[pd.DataFrame, str | None]:
    try:
        if not os.path.exists(path):
            return pd.DataFrame(), f"Fichier introuvable: {path}"
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Erreur lecture CSV: {type(e).__name__}: {e}"


def save_csv(df: pd.DataFrame, path: str) -> Tuple[bool, str | None]:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return True, None
    except Exception as e:
        return False, f"Erreur sauvegarde CSV: {type(e).__name__}: {e}"


def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_name(s: Any) -> str:
    s = _norm(s).lower()
    s = re.sub(r"[^a-z0-9\s\-\']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_team(s: Any) -> str:
    s = _norm(s).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _norm_pos(s: Any) -> str:
    s = _norm(s).upper()
    s = re.sub(r"[^A-Z]", "", s)
    # normalize F/D/G variants to common:
    if s in {"LW", "RW", "C", "F", "FWD", "FORWARD"}:
        return "F"
    if s in {"D", "DEF", "DEFENSE", "DEFENCEMAN"}:
        return "D"
    if s in {"G", "GOL", "GOALIE", "GK"}:
        return "G"
    return s


def _norm_jersey(x: Any) -> str:
    s = _norm(x)
    s = re.sub(r"[^0-9]", "", s)
    return s


def _norm_dob(x: Any) -> str:
    """
    Retourne YYYY-MM-DD si possible, sinon ''.
    Accepte dates déjà string, timestamps, etc.
    """
    if x is None:
        return ""
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan] * len(df), index=df.index)


def _is_prod() -> bool:
    return str(os.environ.get("PMS_ENV", "")).strip().lower() in {"prod", "production"}


# =========================
# Column detection (robuste)
# =========================
def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(map(str, df.columns))
    for c in candidates:
        if c in cols:
            return c
    low_map = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str | None]:
    """
    Retourne (id_col, name_col, team_col) avec heuristiques.
    Important: si 'nul_id' existe, c'est un candidat fort pour 'team'.
    """
    id_candidates = [
        "NHL_ID", "nhl_id", "nhlid", "NHLID", "nhlId", "NHL Id", "NHL-ID", "player_id", "id_nhl"
    ]
    name_candidates = [
        "Player", "player", "player_name", "name", "Name", "Full Name", "full_name",
        "Nom", "nom", "Joueur", "joueur"
    ]
    team_candidates = [
        "Team", "team", "Equipe", "équipe", "team_name", "TeamName", "club",
        "Owner", "owner", "GM", "gm",
        "nul_id", "NUL_ID", "nulId"
    ]

    id_col = _first_existing(df, id_candidates)
    name_col = _first_existing(df, name_candidates)
    team_col = _first_existing(df, team_candidates)

    if id_col is None:
        id_col = "NHL_ID"
        if id_col not in df.columns:
            df[id_col] = np.nan

    if name_col is None:
        for c in df.columns:
            if str(c) == id_col or (team_col and str(c) == team_col):
                continue
            s = df[c]
            if s.dtype == object:
                name_col = str(c)
                break
        if name_col is None:
            name_col = "Player"
            if name_col not in df.columns:
                df[name_col] = ""

    nul_team = _first_existing(df, ["nul_id", "NUL_ID", "nulId"])
    if nul_team:
        team_col = nul_team

    return id_col, name_col, team_col


def detect_extra_match_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Détecte colonnes utiles pour matching avancé (DOB, Position, Jersey).
    """
    dob_candidates = ["DOB", "dob", "birth_date", "BirthDate", "Birth Date", "date_naissance", "Date de naissance"]
    pos_candidates = ["Position", "position", "Pos", "pos", "POS"]
    jersey_candidates = ["Jersey#", "Jersey", "jersey", "sweater_number", "SweaterNumber", "Number", "number", "No", "no"]

    dob_col = _first_existing(df, dob_candidates)
    pos_col = _first_existing(df, pos_candidates)
    jersey_col = _first_existing(df, jersey_candidates)

    return {"dob_col": dob_col, "pos_col": pos_col, "jersey_col": jersey_col}


def _count_present_ids(df: pd.DataFrame, id_col: str) -> int:
    s = _safe_col(df, id_col).astype(str).str.strip().str.lower()
    return int((~s.isna() & ~s.isin({"", "nan", "none"})).sum())


def score_source(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Score une source pour proposer la meilleure par défaut.
    """
    # do not mutate caller
    tmp = df.copy()
    id_col, name_col, team_col = detect_columns(tmp)
    extras = detect_extra_match_cols(tmp)
    present_ids = _count_present_ids(tmp, id_col)
    n = int(len(tmp))

    score = 0.0
    if n > 0:
        score += min(1.0, present_ids / max(1, n)) * 60.0  # principal: densité d'IDs
    # colonnes utiles pour matching
    score += 10.0 if team_col else 0.0
    score += 10.0 if extras.get("dob_col") else 0.0
    score += 10.0 if extras.get("pos_col") else 0.0
    score += 10.0 if extras.get("jersey_col") else 0.0

    return {
        "score": float(score),
        "id_col": id_col,
        "name_col": name_col,
        "team_col": team_col,
        "present_ids": present_ids,
        "rows": n,
        "extras": extras,
    }


# =========================
# Confidence + audit helpers
# =========================
def build_audit_report(
    players_df: pd.DataFrame,
    id_col: str,
    name_col: str,
    team_col: str | None = None,
) -> pd.DataFrame:
    df = players_df.copy()

    if team_col is None or team_col not in df.columns:
        _, _, auto_team = detect_columns(df)
        if auto_team in df.columns:
            team_col = auto_team
        else:
            team_col = None

    out = pd.DataFrame(
        {
            "player_name": _safe_col(df, name_col).astype(str).fillna(""),
            "team": (_safe_col(df, team_col).astype(str).fillna("(none)") if team_col else "(none)"),
            "nhl_id": _safe_col(df, id_col),
        }
    )

    out["missing"] = out["nhl_id"].isna() | (
        out["nhl_id"].astype(str).str.strip().str.lower().isin({"", "nan", "none"})
    )

    present = out.loc[~out["missing"], "nhl_id"].astype(str).str.strip()
    dup_mask = present.duplicated(keep=False)
    out["duplicate_id"] = False
    out.loc[~out["missing"], "duplicate_id"] = dup_mask.values

    out["confidence"] = np.where(out["missing"], 0.0, np.where(out["duplicate_id"], 0.2, 0.95))
    return out


def confidence_heatmap(df_audit: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.6, 0.75, 0.85, 1.01]
    labels = ["<0.60", "0.60-0.75", "0.75-0.85", ">=0.85"]
    conf = pd.to_numeric(df_audit.get("confidence", np.nan), errors="coerce")
    bucket = pd.cut(conf.fillna(0.0), bins=bins, labels=labels, right=False)

    teams = df_audit.get("team", pd.Series(["(none)"] * len(df_audit))).fillna("(none)").astype(str)

    piv = pd.pivot_table(
        pd.DataFrame({"team": teams, "bucket": bucket.astype(str)}),
        index="team",
        columns="bucket",
        values="bucket",
        aggfunc="count",
        fill_value=0,
    )

    for lab in labels:
        if lab not in piv.columns:
            piv[lab] = 0
    piv = piv[labels]
    piv = piv.sort_values(by=labels[::-1], ascending=False)
    return piv


# =========================
# Heatmap rendering (SAFE: no matplotlib required)
# =========================
def _hex_blend(c1: str, c2: str, t: float) -> str:
    c1 = c1.lstrip("#")
    c2 = c2.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _styler_heatmap_css(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    vals = df.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    vmin = float(np.nanmin(vals)) if finite.any() else 0.0
    vmax = float(np.nanmax(vals)) if finite.any() else 1.0
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0

    def color_for(x: Any) -> str:
        try:
            xv = float(x)
        except Exception:
            return "background-color:#111827;color:#9ca3af;text-align:center;"
        if not np.isfinite(xv):
            return "background-color:#111827;color:#9ca3af;text-align:center;"
        t = (xv - vmin) / denom
        t = max(0.0, min(1.0, float(t)))
        if t < 0.5:
            c = _hex_blend("#7f1d1d", "#92400e", t / 0.5)
        else:
            c = _hex_blend("#92400e", "#14532d", (t - 0.5) / 0.5)
        return f"background-color:{c};color:#f9fafb;font-weight:700;text-align:center;"

    def apply_fn(_: pd.DataFrame):
        return [[color_for(df.iat[i, j]) for j in range(df.shape[1])] for i in range(df.shape[0])]

    return df.style.apply(apply_fn, axis=None)


def render_confidence_heatmap(heat: pd.DataFrame) -> None:
    if heat is None or getattr(heat, "empty", True):
        st.info("Heatmap: aucune donnée à afficher.")
        return

    try:
        import matplotlib  # noqa: F401
        st.dataframe(heat.style.background_gradient(axis=None), use_container_width=True)
        return
    except Exception:
        pass

    try:
        st.dataframe(_styler_heatmap_css(heat), use_container_width=True)
        st.caption("ℹ️ Heatmap affichée en mode SAFE (matplotlib non disponible).")
    except Exception as e:
        st.warning(f"Heatmap: affichage stylé indisponible ({type(e).__name__}). Affichage simple.")
        st.dataframe(heat, use_container_width=True)


# =========================
# Recovery from source (MATCHING ADVANCED + REVIEW)
# =========================
def _prep_keys(
    df: pd.DataFrame,
    name_col: str,
    team_col: Optional[str],
    dob_col: Optional[str],
    pos_col: Optional[str],
    jersey_col: Optional[str],
) -> pd.DataFrame:
    out = df.copy()
    out["_k_name"] = _safe_col(out, name_col).map(_norm_name)

    if team_col and team_col in out.columns:
        out["_k_team"] = _safe_col(out, team_col).map(_norm_team)
    else:
        out["_k_team"] = ""

    if dob_col and dob_col in out.columns:
        out["_k_dob"] = _safe_col(out, dob_col).map(_norm_dob)
    else:
        out["_k_dob"] = ""

    if pos_col and pos_col in out.columns:
        out["_k_pos"] = _safe_col(out, pos_col).map(_norm_pos)
    else:
        out["_k_pos"] = ""

    if jersey_col and jersey_col in out.columns:
        out["_k_jersey"] = _safe_col(out, jersey_col).map(_norm_jersey)
    else:
        out["_k_jersey"] = ""

    # composite keys
    out["_k_name_team"] = out["_k_name"] + "||" + out["_k_team"]
    out["_k_name_dob"] = out["_k_name"] + "||" + out["_k_dob"]
    out["_k_team_pos_jersey"] = out["_k_team"] + "||" + out["_k_pos"] + "||" + out["_k_jersey"]

    return out


def _clean_id_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.where(~x.str.lower().isin({"", "nan", "none"}), other=np.nan)
    return x


def _make_join_map(source: pd.DataFrame, key_col: str, id_col: str) -> pd.DataFrame:
    """
    Retourne df[key, nhl_id, collision_count] où key est unique (si collision_count>1, collision).
    """
    tmp = source[[key_col, id_col]].copy()
    tmp[id_col] = _clean_id_series(tmp[id_col])
    tmp = tmp.dropna(subset=[id_col])
    tmp = tmp[tmp[key_col].astype(str).str.strip() != ""]
    if tmp.empty:
        return pd.DataFrame(columns=[key_col, "src_nhl_id", "collision_count"])

    g = tmp.groupby(key_col)[id_col].agg(["nunique"])
    collisions = g.rename(columns={"nunique": "collision_count"}).reset_index()

    # pick first id per key (deterministic)
    first_map = tmp.drop_duplicates(subset=[key_col], keep="first").rename(columns={id_col: "src_nhl_id"})
    out = first_map.merge(collisions, on=key_col, how="left")
    out["collision_count"] = out["collision_count"].fillna(1).astype(int)
    return out[[key_col, "src_nhl_id", "collision_count"]]


def recover_from_source(
    players_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    id_col: str,
    name_col: str,
    team_col: str | None,
    source_tag: str,
    conf: float,
    match_name: bool = True,
    match_name_team: bool = True,
    match_name_dob: bool = True,
    match_team_pos_jersey: bool = True,
    review_only: bool = False,
    max_preview: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remplit id_col dans players_df en utilisant source_df.
    Stratégies (ordre):
      1) name+dop (si dispo) — meilleur
      2) name+team (si dispo)
      3) team+pos+jersey (si dispo)
      4) name seul — fallback

    review_only=True => ne sauvegarde pas, mais retourne merged + debug + tables collisions.
    """
    p = players_df.copy()
    s = source_df.copy()

    if "nhl_id_source" not in p.columns:
        p["nhl_id_source"] = ""
    if "confidence" not in p.columns:
        p["confidence"] = np.nan

    # detect extras for target/source
    p_ex = detect_extra_match_cols(p)
    s_ex = detect_extra_match_cols(s)

    # prepare keys
    p_k = _prep_keys(p, name_col, team_col, p_ex.get("dob_col"), p_ex.get("pos_col"), p_ex.get("jersey_col"))
    s_id_col, s_name_col, s_team_col = detect_columns(s)
    s_k = _prep_keys(s, s_name_col, s_team_col, s_ex.get("dob_col"), s_ex.get("pos_col"), s_ex.get("jersey_col"))

    # ensure source id col exists
    if s_id_col not in s_k.columns:
        s_k[s_id_col] = np.nan

    # build join maps
    maps = []
    if match_name_dob:
        maps.append(("name_dob", "_k_name_dob", conf + 0.08))
    if match_name_team:
        maps.append(("name_team", "_k_name_team", conf + 0.03))
    if match_team_pos_jersey:
        maps.append(("team_pos_jersey", "_k_team_pos_jersey", conf + 0.02))
    if match_name:
        maps.append(("name", "_k_name", conf))

    # current missing
    cur = _clean_id_series(_safe_col(p_k, id_col))
    cur_missing = cur.isna()

    filled_total = 0
    collisions_rows = []

    # apply maps in order
    for tag, key_col, this_conf in maps:
        if not cur_missing.any():
            break

        join_map = _make_join_map(s_k, key_col, s_id_col)
        if join_map.empty:
            continue

        # collision report
        coll = join_map[join_map["collision_count"] > 1].copy()
        if not coll.empty:
            coll["match_type"] = tag
            collisions_rows.append(coll)

        # join
        tmp = p_k[[key_col]].copy()
        tmp["_idx"] = p_k.index
        tmp = tmp.merge(join_map, on=key_col, how="left")

        # only keys that map to a single source ID (collision_count==1)
        ok_map = tmp["src_nhl_id"].notna() & (tmp["collision_count"].fillna(0).astype(int) == 1)

        # fill only missing
        fillable = cur_missing & ok_map.set_axis(p_k.index)  # align on index
        nfill = int(fillable.sum())

        if nfill:
            p_k.loc[fillable, id_col] = tmp.set_index("_idx").loc[p_k.index, "src_nhl_id"]
            p_k.loc[fillable, "nhl_id_source"] = f"{source_tag}|{tag}"
            p_k.loc[fillable, "confidence"] = float(min(0.99, max(0.0, this_conf)))
            filled_total += nfill
            cur = _clean_id_series(_safe_col(p_k, id_col))
            cur_missing = cur.isna()

    # Build collisions preview (optional)
    collisions_df = pd.DataFrame()
    if collisions_rows:
        collisions_df = pd.concat(collisions_rows, ignore_index=True)
        collisions_df = collisions_df.rename(columns={"src_nhl_id": "example_src_nhl_id"})
        collisions_df = collisions_df.sort_values(["match_type", "collision_count"], ascending=[True, False])

    # clean temp columns
    p_k = p_k.drop(columns=[c for c in p_k.columns if c.startswith("_k_")], errors="ignore")

    dbg = {
        "filled": filled_total,
        "source_tag": source_tag,
        "conf": conf,
        "strategies": [m[0] for m in maps],
        "target_extras": p_ex,
        "source_extras": s_ex,
        "source_detected": {"id_col": s_id_col, "name_col": s_name_col, "team_col": s_team_col},
        "collisions": int(len(collisions_df)) if not collisions_df.empty else 0,
        "collisions_preview": collisions_df.head(max_preview) if not collisions_df.empty else pd.DataFrame(),
    }

    # review_only: do nothing special here; caller decides whether to write
    return p_k, dbg


# =========================
# Dup cleanup + write guard
# =========================
def duplicate_rate(df: pd.DataFrame, id_col: str) -> float:
    s = _clean_id_series(_safe_col(df, id_col))
    present = s.dropna()
    if present.empty:
        return 0.0
    dup = present.duplicated(keep=False).mean()
    return float(dup) * 100.0


def cleanup_duplicates_fallback(df: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    s = _clean_id_series(_safe_col(out, id_col))
    present = s.dropna()
    if present.empty:
        return out, 0

    dup_mask = present.duplicated(keep="first")
    dup_ids = present.loc[dup_mask].index
    n = int(len(dup_ids))
    if n:
        out.loc[dup_ids, id_col] = np.nan
        if "nhl_id_source" in out.columns:
            out.loc[dup_ids, "nhl_id_source"] = "fallback_cleanup"
        if "confidence" in out.columns:
            out.loc[dup_ids, "confidence"] = 0.0
    return out, n


# =========================
# Main render (call this from app.py)
# =========================
def render(ctx: Optional[Dict[str, Any]] = None) -> None:
    ctx = ctx or {}
    season = str(ctx.get("season") or "2025-2026")
    data_dir = str(ctx.get("data_dir") or "data")
    is_admin = bool(ctx.get("is_admin", True))
    prod_lock = bool(ctx.get("prod_lock", True))

    st.subheader("🛠️ Admin — NHL_ID Tools (SAFE)")

    if prod_lock and _is_prod() and not is_admin:
        st.error("🔒 Mode PROD: accès admin requis.")
        st.stop()

    # -----------------------------------
    # Source selection
    # -----------------------------------
    st.markdown("### Source de récupération (optionnel)")
    up = st.file_uploader("Ou uploader un CSV source", type=["csv"])

    src_paths = [
        os.path.join(data_dir, "equipes_joueurs_2025-2026.csv"),
        os.path.join(data_dir, "hockey.players.csv"),
    ]

    # Auto-pick best source by scoring (only for local file options)
    best_idx = 0
    scores: List[Dict[str, Any]] = []
    for i, pth in enumerate(src_paths):
        df_s, e = load_csv(pth)
        if e or df_s is None or df_s.empty:
            scores.append({"path": pth, "score": -1.0, "present_ids": 0, "rows": 0, "id_col": None})
            continue
        sc = score_source(df_s)
        sc["path"] = pth
        scores.append(sc)
    if scores:
        best_idx = int(np.argmax([s.get("score", -1.0) for s in scores]))

    pick = st.selectbox("Récupérer NHL_ID depuis…", options=src_paths, index=best_idx)

    with st.expander("🔎 Diagnostic sources (auto)", expanded=False):
        if scores:
            show = []
            for s in scores:
                show.append({
                    "path": os.path.basename(str(s.get("path", ""))),
                    "score": round(float(s.get("score", 0.0)), 1),
                    "rows": int(s.get("rows", 0)),
                    "present_ids": int(s.get("present_ids", 0)),
                    "id_col": str(s.get("id_col", "")),
                    "team_col": str(s.get("team_col", "")),
                    "dob_col": str((s.get("extras") or {}).get("dob_col", "")),
                    "pos_col": str((s.get("extras") or {}).get("pos_col", "")),
                    "jersey_col": str((s.get("extras") or {}).get("jersey_col", "")),
                })
            st.dataframe(pd.DataFrame(show), use_container_width=True)
            st.caption("Tip: la meilleure source est celle avec le plus de NHL_ID présents + colonnes utiles (DOB/pos/jersey).")

    # -----------------------------------
    # Load target
    # -----------------------------------
    target_path = os.path.join(data_dir, "equipes_joueurs_2025-2026.csv")
    df, err = load_csv(target_path)
    if err:
        st.error(err)
        st.stop()

    # robust detect for target
    id_col, name_col, team_col = detect_columns(df)
    df_ex = detect_extra_match_cols(df)

    # -----------------------------------
    # Verify state
    # -----------------------------------
    if st.button("🔎 Vérifier l'état des NHL_ID"):
        audit_df = build_audit_report(df, id_col, name_col, team_col)

        total = int(len(audit_df))
        with_id = int((~audit_df["missing"]).sum())
        missing = int(audit_df["missing"].sum())
        pct_missing = 0.0 if total == 0 else (missing / total) * 100.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total joueurs", f"{total}")
        c2.metric("Avec ID (NHL_ID)", f"{with_id}")
        c3.metric("Manquants", f"{missing}")
        c4.metric("% manquants", f"{pct_missing:.1f}%")

        dup_pct = duplicate_rate(df, id_col)
        st.info(f"IDs dupliqués détectés: {int((audit_df['duplicate_id']).sum())} (~{dup_pct:.1f}% des IDs présents).")

        st.markdown("#### 🧾 Audit (aperçu)")
        st.dataframe(audit_df.head(50), use_container_width=True)

        heat = confidence_heatmap(audit_df)
        st.markdown("#### 🧩 Heatmap de confiance (par équipe × bucket)")
        render_confidence_heatmap(heat)

        out = io.StringIO()
        audit_df.to_csv(out, index=False)
        st.download_button(
            "🧾 Télécharger rapport d'audit NHL_ID (CSV)",
            data=out.getvalue().encode("utf-8"),
            file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.divider()

    # -----------------------------------
    # Recovery action
    # -----------------------------------
    st.markdown("### 🔁 Associer / récupérer NHL_ID")

    if up is not None:
        src_df = pd.read_csv(up, low_memory=False)
        src_tag = "upload"
    else:
        src_df, src_err = load_csv(pick)
        if src_err:
            st.warning(src_err)
            src_df = pd.DataFrame()
        src_tag = os.path.basename(pick)

    # detect source + show warning if no IDs
    if isinstance(src_df, pd.DataFrame) and not src_df.empty:
        sc = score_source(src_df)
        st.caption(
            f"Source détectée: id_col='{sc.get('id_col')}', name_col='{sc.get('name_col')}', "
            f"team_col='{sc.get('team_col')}', NHL_ID présents={sc.get('present_ids')}/{sc.get('rows')}, score={sc.get('score'):.1f}"
        )
        if int(sc.get("present_ids", 0)) == 0:
            st.warning(
                "⚠️ Cette source ne contient AUCUN NHL_ID exploitable. "
                "Sélectionne hockey.players.csv (ou un CSV upload) qui a déjà des NHL_ID."
            )

    # Matching options
    st.markdown("#### 🧩 Stratégies de matching")
    cA, cB, cC, cD = st.columns(4)
    match_name_dob = cA.checkbox("Name + DOB", value=True, help="Plus fiable si DOB présent dans les 2 fichiers.")
    match_name_team = cB.checkbox("Name + Team", value=True, help="Fiable si les abbr d'équipe concordent (MTL, DAL, etc.)")
    match_team_pos_jersey = cC.checkbox("Team + Pos + Jersey", value=True, help="Utile si tes CSV ont position + numéro.")
    match_name = cD.checkbox("Name seul (fallback)", value=True, help="Dernier recours, risque collisions plus élevé.")

    conf = st.slider("Score de confiance appliqué aux IDs récupérés", min_value=0.50, max_value=0.99, value=0.85, step=0.01)
    max_dup_pct = st.slider("🛑 Bloquer toute écriture si duplication > X %", min_value=0, max_value=50, value=5, step=1)
    do_cleanup = st.checkbox("🧼 Nettoyage automatique des doublons (fallback) avant write", value=True)

    review_only = st.checkbox("🧪 Mode review: montrer collisions avant write", value=True)

    if st.button("🧩 Associer NHL_ID (depuis source)"):
        if src_df is None or src_df.empty:
            st.error("Source vide / indisponible.")
            st.stop()

        before_dup = duplicate_rate(df, id_col)

        merged, dbg = recover_from_source(
            df,
            src_df,
            id_col=id_col,
            name_col=name_col,
            team_col=team_col,
            source_tag=src_tag,
            conf=float(conf),
            match_name=match_name,
            match_name_team=match_name_team,
            match_name_dob=match_name_dob,
            match_team_pos_jersey=match_team_pos_jersey,
            review_only=review_only,
        )

        cleaned_n = 0
        if do_cleanup:
            merged, cleaned_n = cleanup_duplicates_fallback(merged, id_col)

        after_dup = duplicate_rate(merged, id_col)

        st.success(
            f"✅ Récupération terminée: {dbg.get('filled', 0)} IDs remplis. "
            f"Doublons avant: {before_dup:.1f}%, après: {after_dup:.1f}% (nettoyés: {cleaned_n})."
        )

        # Collisions preview
        if review_only:
            coll_prev = dbg.get("collisions_preview")
            if isinstance(coll_prev, pd.DataFrame) and not coll_prev.empty:
                st.warning(f"⚠️ Collisions détectées (aperçu) — {dbg.get('collisions', 0)} lignes. Résoudre avant write.")
                st.dataframe(coll_prev, use_container_width=True)
            else:
                st.info("✅ Aucune collision détectée par les stratégies sélectionnées.")

        # Always show audit + heatmap on the merged (preview)
        audit_df = build_audit_report(merged, id_col, name_col, team_col)
        st.markdown("#### 🧾 Audit après récupération (aperçu)")
        st.dataframe(audit_df.head(50), use_container_width=True)

        heat = confidence_heatmap(audit_df)
        st.markdown("#### 🧩 Heatmap de confiance (par équipe × bucket)")
        render_confidence_heatmap(heat)

        out = io.StringIO()
        audit_df.to_csv(out, index=False)
        st.download_button(
            "🧾 Télécharger rapport d'audit NHL_ID (CSV)",
            data=out.getvalue().encode("utf-8"),
            file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # If review mode AND collisions exist -> block write automatically
        coll_count = int(dbg.get("collisions", 0) or 0)
        if review_only and coll_count > 0:
            st.error("🔒 Write bloqué en mode review (collisions à résoudre).")
            st.stop()

        # block write if too many duplicates
        if after_dup > float(max_dup_pct):
            st.error(f"🔒 Écriture BLOQUÉE: duplication {after_dup:.1f}% > seuil {max_dup_pct}%.")
            st.stop()

        ok, save_err = save_csv(merged, target_path)
        if not ok:
            st.error(save_err or "Erreur inconnue.")
            st.stop()

        st.info(f"💾 Sauvegardé: {target_path}")


# Backward-compat alias (si ton app appelle _render_tools)
def _render_tools(*args: Any, **kwargs: Any) -> None:
    ctx = None
    if args and isinstance(args[0], dict):
        ctx = args[0]
    if kwargs and isinstance(kwargs.get("ctx"), dict):
        ctx = kwargs["ctx"]
    render(ctx or {})
