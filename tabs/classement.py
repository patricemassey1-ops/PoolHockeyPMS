# tabs/classement.py
from __future__ import annotations

import os
import pandas as pd
import streamlit as st


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _points_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"points_periods_{season}.csv")


def _load_csv_safe(path: str) -> pd.DataFrame:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _fmt_pts(x) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.2f}"


def render(ctx: dict) -> None:
    st.header("🏆 Classement")
    season = _season(ctx)
    data_dir = _data_dir(ctx)

    st.caption(f"Saison: **{season}**")

    p_path = _points_path(data_dir, season)
    pts = _load_csv_safe(p_path)

    if pts.empty:
        st.info(f"Aucun fichier points trouvé. (attendu: `{p_path}`)")
        st.caption("Quand tu seras prêt, on peut aussi construire le classement à partir d’autres sources.")
        return

    # Colonnes attendues (tolérant, mais on veut au minimum owner + start_ts + points_start)
    for need in ["owner", "start_ts", "points_start"]:
        if need not in pts.columns:
            st.error(f"Colonne manquante dans `{os.path.basename(p_path)}`: `{need}`")
            st.caption("Colonnes détectées: " + ", ".join([str(c) for c in pts.columns]))
            return

    df = pts.copy()
    df["owner"] = df["owner"].astype(str).str.strip()
    df["start_ts"] = df["start_ts"].astype(str).str.strip()
    df["points_start"] = pd.to_numeric(df["points_start"], errors="coerce").fillna(0.0)

    # Ignore rows sans owner
    df = df[df["owner"].ne("")]

    # Convertit start_ts en datetime (pour trier), tout en gardant string pour pivot
    df["_ts"] = pd.to_datetime(df["start_ts"], errors="coerce")

    # Label période lisible
    # ex: 2026-01-19T08:57:02-05:00 -> 2026-01-19 08:57
    def _label(row) -> str:
        ts = row["_ts"]
        if pd.isna(ts):
            return str(row["start_ts"])
        return ts.strftime("%Y-%m-%d %H:%M")

    df["period"] = df.apply(_label, axis=1)

    # Agrégation: total équipe par période (somme points_start de tous les joueurs)
    agg = (
        df.groupby(["owner", "period"], as_index=False)["points_start"]
        .sum()
        .rename(columns={"points_start": "team_points"})
    )

    # Ordonner périodes chronologiquement (avec mapping)
    period_order = (
        df[["period", "_ts"]]
        .drop_duplicates()
        .sort_values("_ts", ascending=True)
    )
    periods = period_order["period"].tolist()

    # Pivot: rows=owner, cols=period, values=team_points
    pivot = agg.pivot_table(index="owner", columns="period", values="team_points", aggfunc="sum", fill_value=0.0)

    # Réordonner colonnes selon temps
    pivot = pivot.reindex(columns=periods)

    # Total = dernier snapshot (dernière colonne)
    if pivot.shape[1] >= 1:
        last_period = pivot.columns[-1]
        pivot["Total"] = pivot[last_period]
    else:
        pivot["Total"] = 0.0

    # Classement: tri par Total desc
    pivot_sorted = pivot.sort_values("Total", ascending=False).copy()

    # KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Équipes", int(pivot_sorted.shape[0]))
    with c2:
        st.metric("Périodes (snapshots)", int(len(periods)))
    with c3:
        st.metric("Dernière période", str(periods[-1]) if periods else "—")

    st.divider()

    # 1) Classement (dernier snapshot)
    st.subheader("📌 Classement — Dernier snapshot")
    rank_df = pivot_sorted[["Total"]].reset_index().rename(columns={"owner": "Équipe"})
    rank_df["Total"] = rank_df["Total"].apply(_fmt_pts)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    # Movers (si on a au moins 2 périodes)
    if len(periods) >= 2:
        st.subheader("📈 Variation (dernier - précédent)")
        prev_period = periods[-2]
        delta = (pivot_sorted[periods[-1]] - pivot_sorted[prev_period]).rename("Δ").reset_index()
        delta = delta.rename(columns={"owner": "Équipe"})
        delta["Δ"] = delta["Δ"].apply(_fmt_pts)
        st.dataframe(delta.sort_values("Δ", ascending=False), use_container_width=True, hide_index=True)

    st.divider()

    # 2) Table par périodes + total
    st.subheader("🧾 Points par période + Total")
    view = pivot_sorted.reset_index().rename(columns={"owner": "Équipe"}).copy()

    # Format
    for c in view.columns:
        if c != "Équipe":
            view[c] = view[c].apply(_fmt_pts)

    st.dataframe(view, use_container_width=True, hide_index=True)

    with st.expander("🔎 Debug fichier", expanded=False):
        st.caption(f"Source: `{p_path}`")
        st.write("Colonnes:", list(pts.columns))
        st.write("Périodes détectées:", periods)
