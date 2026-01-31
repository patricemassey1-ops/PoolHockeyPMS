# tabs/admin.py — SAFE MINI (NHL_ID tools + prod lock + audit/heatmap)
# NOTE: This file is intentionally self-contained to avoid NameError/indent issues.

from __future__ import annotations

import os
import re
import io
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

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


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan] * len(df), index=df.index)


def _is_prod() -> bool:
    # simple prod detector (env var)
    return str(os.environ.get("PMS_ENV", "")).strip().lower() in {"prod", "production"}


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
    if team_col and team_col not in df.columns:
        team_col = None

    out = pd.DataFrame(
        {
            "player_name": _safe_col(df, name_col).astype(str).fillna(""),
            "team": (_safe_col(df, team_col).astype(str).fillna("(none)") if team_col else "(none)"),
            "nhl_id": _safe_col(df, id_col),
        }
    )

    # basic missing
    out["missing"] = out["nhl_id"].isna() | (out["nhl_id"].astype(str).str.strip() == "") | (out["nhl_id"].astype(str).str.lower() == "nan")

    # duplicates among present IDs
    present = out.loc[~out["missing"], "nhl_id"].astype(str).str.strip()
    dup_mask = present.duplicated(keep=False)
    out["duplicate_id"] = False
    out.loc[~out["missing"], "duplicate_id"] = dup_mask.values

    # confidence heuristic:
    #  - missing -> 0.0
    #  - duplicate -> 0.2
    #  - otherwise -> 0.95
    out["confidence"] = np.where(out["missing"], 0.0, np.where(out["duplicate_id"], 0.2, 0.95))

    return out


def confidence_heatmap(df_audit: pd.DataFrame) -> pd.DataFrame:
    # bins
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
    # stable column order
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
    # CSS heatmap (red -> amber -> green), works without matplotlib
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
            c = _hex_blend("#7f1d1d", "#92400e", t / 0.5)  # red -> amber
        else:
            c = _hex_blend("#92400e", "#14532d", (t - 0.5) / 0.5)  # amber -> green
        return f"background-color:{c};color:#f9fafb;font-weight:700;text-align:center;"

    def apply_fn(_: pd.DataFrame):
        return [[color_for(df.iat[i, j]) for j in range(df.shape[1])] for i in range(df.shape[0])]

    return df.style.apply(apply_fn, axis=None)


def render_confidence_heatmap(heat: pd.DataFrame) -> None:
    """Render heatmap safely. Uses background_gradient if matplotlib exists, else CSS fallback."""
    if heat is None or getattr(heat, "empty", True):
        st.info("Heatmap: aucune donnée à afficher.")
        return

    # Prefer pandas background_gradient if matplotlib is available (nicer)
    try:
        import matplotlib  # noqa: F401
        st.dataframe(heat.style.background_gradient(axis=None), use_container_width=True)
        return
    except Exception:
        pass

    # Fallback: CSS-only styler (no matplotlib)
    try:
        st.dataframe(_styler_heatmap_css(heat), use_container_width=True)
        st.caption("ℹ️ Heatmap affichée en mode SAFE (matplotlib non disponible).")
    except Exception as e:
        st.warning(f"Heatmap: affichage stylé indisponible ({type(e).__name__}). Affichage simple.")
        st.dataframe(heat, use_container_width=True)


# =========================
# Recovery from source
# =========================
def recover_from_source(
    players_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    id_col: str,
    name_col: str,
    team_col: str | None,
    source_tag: str,
    conf: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = players_df.copy()
    s = source_df.copy()

    if "nhl_id_source" not in p.columns:
        p["nhl_id_source"] = ""

    # normalize
    p["_nname"] = p[name_col].map(_norm_name)
    s["_nname"] = s[name_col].map(_norm_name)

    s_id = _safe_col(s, id_col).astype(str).str.strip()
    s_id = s_id.where(~s_id.str.lower().isin({"", "nan", "none"}), other=np.nan)

    # name-only join
    join = s[["_nname"]].copy()
    join["src_nhl_id"] = s_id
    join = join.dropna(subset=["src_nhl_id"]).drop_duplicates(subset=["_nname"], keep="first")

    merged = p.merge(join, on="_nname", how="left")

    # fill only missing IDs
    cur = _safe_col(merged, id_col).astype(str).str.strip()
    cur_missing = cur.str.lower().isin({"", "nan", "none"}) | cur.isna()

    fillable = cur_missing & merged["src_nhl_id"].notna()
    filled_n = int(fillable.sum())

    merged.loc[fillable, id_col] = merged.loc[fillable, "src_nhl_id"]
    merged.loc[fillable, "nhl_id_source"] = f"{source_tag}|name"
    merged.loc[fillable, "confidence"] = conf

    merged = merged.drop(columns=["_nname", "src_nhl_id"], errors="ignore")

    dbg = {"filled": filled_n, "source_tag": source_tag, "conf": conf}
    return merged, dbg


# =========================
# Dup cleanup + write guard
# =========================
def duplicate_rate(df: pd.DataFrame, id_col: str) -> float:
    s = _safe_col(df, id_col).astype(str).str.strip()
    s = s.where(~s.str.lower().isin({"", "nan", "none"}), other=np.nan)
    present = s.dropna()
    if present.empty:
        return 0.0
    dup = present.duplicated(keep=False).mean()
    return float(dup) * 100.0


def cleanup_duplicates_fallback(df: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    s = _safe_col(out, id_col).astype(str).str.strip()
    s = s.where(~s.str.lower().isin({"", "nan", "none"}), other=np.nan)
    present = s.dropna()
    if present.empty:
        return out, 0

    # if dup, blank the later occurrences (fallback cleanup)
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
    """
    ctx expected keys (optional):
      - season: str
      - data_dir: str
      - is_admin: bool
      - prod_lock: bool
    """
    ctx = ctx or {}
    season = str(ctx.get("season") or "2025-2026")
    data_dir = str(ctx.get("data_dir") or "data")
    is_admin = bool(ctx.get("is_admin", True))
    prod_lock = bool(ctx.get("prod_lock", True))

    st.subheader("🛠️ Admin — NHL_ID Tools (SAFE)")

    if prod_lock and _is_prod() and not is_admin:
        st.error("🔒 Mode PROD: accès admin requis.")
        st.stop()

    # ---- source selection
    st.markdown("### Source de récupération (optionnel)")
    up = st.file_uploader("Ou uploader un CSV source", type=["csv"])
    src_paths = [
        os.path.join(data_dir, "equipes_joueurs_2025-2026.csv"),
        os.path.join(data_dir, "hockey.players.csv"),
    ]
    pick = st.selectbox("Récupérer NHL_ID depuis…", options=src_paths, index=0)

    # load target
    target_path = os.path.join(data_dir, "equipes_joueurs_2025-2026.csv")
    df, err = load_csv(target_path)
    if err:
        st.error(err)
        st.stop()

    # detect columns
    cols = list(df.columns)
    id_col = "NHL_ID" if "NHL_ID" in cols else ("nhl_id" if "nhl_id" in cols else cols[0])
    name_col = "Player" if "Player" in cols else ("player_name" if "player_name" in cols else cols[0])
    team_col = "Team" if "Team" in cols else ("team" if "team" in cols else None)

    # ---- verify
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

        heat = confidence_heatmap(audit_df)

        st.markdown("#### 🧩 Heatmap de confiance (par équipe × bucket)")
        render_confidence_heatmap(heat)

        # CSV audit download
        out = io.StringIO()
        audit_df.to_csv(out, index=False)
        st.download_button(
            "🧾 Télécharger rapport d'audit NHL_ID (CSV)",
            data=out.getvalue().encode("utf-8"),
            file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.divider()

    # ---- recovery action
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

    # pick source cols (best effort)
    src_cols = list(src_df.columns) if isinstance(src_df, pd.DataFrame) else []
    src_id_col = "NHL_ID" if "NHL_ID" in src_cols else ("nhl_id" if "nhl_id" in src_cols else (src_cols[0] if src_cols else id_col))
    src_name_col = "Player" if "Player" in src_cols else ("player_name" if "player_name" in src_cols else (src_cols[0] if src_cols else name_col))
    src_team_col = "Team" if "Team" in src_cols else ("team" if "team" in src_cols else None)

    # confidence slider
    conf = st.slider("Score de confiance appliqué aux IDs récupérés", min_value=0.50, max_value=0.99, value=0.85, step=0.01)

    # duplication guard
    max_dup_pct = st.slider("🛑 Bloquer toute écriture si duplication > X %", min_value=0, max_value=50, value=5, step=1)

    do_cleanup = st.checkbox("🧼 Nettoyage automatique des doublons (fallback) avant write", value=True)

    if st.button("🧩 Associer NHL_ID (depuis source)"):
        if src_df is None or src_df.empty:
            st.error("Source vide / indisponible.")
            st.stop()

        before_dup = duplicate_rate(df, id_col)

        # match by name only for now (safe mini)
        merged, dbg = recover_from_source(
            df,
            src_df,
            id_col=id_col,
            name_col=name_col,
            team_col=team_col,
            source_tag=src_tag,
            conf=float(conf),
        )

        # optional fallback cleanup
        cleaned_n = 0
        if do_cleanup:
            merged, cleaned_n = cleanup_duplicates_fallback(merged, id_col)

        after_dup = duplicate_rate(merged, id_col)

        st.success(f"✅ Récupération terminée: {dbg.get('filled', 0)} IDs remplis. Doublons avant: {before_dup:.1f}%, après: {after_dup:.1f}% (nettoyés: {cleaned_n}).")

        # block write if too many duplicates
        dup_pct = after_dup
        if dup_pct > float(max_dup_pct):
            st.error(f"🔒 Écriture BLOQUÉE: duplication {dup_pct:.1f}% > seuil {max_dup_pct}%.")
            st.stop()

        ok, save_err = save_csv(merged, target_path)
        if not ok:
            st.error(save_err or "Erreur inconnue.")
            st.stop()

        st.info(f"💾 Sauvegardé: {target_path}")

        # Show audit + downloads after write
        audit_df = build_audit_report(df, id_col, name_col, team_col)
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


# Backward-compat alias (si ton app appelle _render_tools)
def _render_tools(*args: Any, **kwargs: Any) -> None:
    ctx = None
    if args and isinstance(args[0], dict):
        ctx = args[0]
    if kwargs and isinstance(kwargs.get("ctx"), dict):
        ctx = kwargs["ctx"]
    render(ctx or {})
