# tabs/classement.py
from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _points_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"points_periods_{season}.csv")


def _gm_points_path(data_dir: str) -> str:
    return os.path.join(data_dir, "gm_points.csv")


def _pick_history_path(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, "historique_admin.csv"),
        os.path.join(data_dir, "historique.csv"),
        os.path.join(data_dir, "history.csv"),
        os.path.join(data_dir, "transactions.csv"),
        os.path.join(data_dir, "backup_history.csv"),
        os.path.join(data_dir, "historique_transactions.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def _load_csv_safe(path: str) -> pd.DataFrame:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path, low_memory=False)
    except Exception:
        pass
    return pd.DataFrame()


def _read_file_bytes(path: str) -> bytes:
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    except Exception:
        pass
    return b""


def _atomic_write_df(df: pd.DataFrame, out_path: str) -> tuple[bool, str]:
    """Write CSV atomically in the same directory (avoid partial writes)."""
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        d = os.path.dirname(out_path) or "."
        with tempfile.NamedTemporaryFile("w", delete=False, dir=d, suffix=".tmp", encoding="utf-8", newline="") as tf:
            tmp_path = tf.name
            df.to_csv(tf, index=False)
        os.replace(tmp_path, out_path)
        return True, ""
    except Exception as e:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False, str(e)


def _fmt_pts(x) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.2f}"


def _normalize_gm_points_df(df: pd.DataFrame) -> pd.DataFrame:
    # Accept common column names
    cols = {c.lower().strip(): c for c in df.columns}
    gm_col = cols.get("gm") or cols.get("owner") or cols.get("equipe") or cols.get("équipe")
    pts_col = cols.get("points") or cols.get("pts") or cols.get("score")

    out = df.copy()
    if gm_col and gm_col != "GM":
        out = out.rename(columns={gm_col: "GM"})
    if pts_col and pts_col != "Points":
        out = out.rename(columns={pts_col: "Points"})

    if "GM" not in out.columns:
        out["GM"] = ""
    if "Points" not in out.columns:
        out["Points"] = 0

    out["GM"] = out["GM"].astype(str).str.strip()
    out["Points"] = pd.to_numeric(out["Points"], errors="coerce").fillna(0).astype(int)
    out = out[out["GM"].ne("")]
    return out[["GM", "Points"]]


def _load_or_create_gm_points(data_dir: str, owners: list[str]) -> tuple[pd.DataFrame, str, bool]:
    """Return (gm_points_df, path, created). Auto-create zeros if missing."""
    path = _gm_points_path(data_dir)
    created = False

    if os.path.exists(path):
        df = _load_csv_safe(path)
        if df.empty:
            df = pd.DataFrame({"GM": owners, "Points": [0] * len(owners)})
    else:
        df = pd.DataFrame({"GM": owners, "Points": [0] * len(owners)})
        ok, err = _atomic_write_df(df, path)
        # Even if write fails, we still return df in memory
        created = ok

    df = _normalize_gm_points_df(df)

    # Ensure every owner exists
    existing = set(df["GM"].tolist())
    missing = [o for o in owners if o not in existing]
    if missing:
        df = pd.concat([df, pd.DataFrame({"GM": missing, "Points": [0] * len(missing)})], ignore_index=True)
        df = _normalize_gm_points_df(df)
        _atomic_write_df(df, path)

    return df, path, created


def render(ctx: dict) -> None:
    st.header("🏆 Classement")
    season = _season(ctx)
    data_dir = _data_dir(ctx)

    st.caption(f"Saison: **{season}**")

    # -----------------------------
    # 1) Charger les points par période
    # -----------------------------
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

    # Ordonner périodes chronologiquement
    period_order = (
        df[["period", "_ts"]]
        .drop_duplicates()
        .sort_values("_ts", ascending=True)
    )
    periods = period_order["period"].tolist()

    # Pivot: rows=owner, cols=period, values=team_points
    pivot = agg.pivot_table(index="owner", columns="period", values="team_points", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(columns=periods)

    # Total = dernier snapshot (dernière colonne)
    if pivot.shape[1] >= 1:
        last_period = pivot.columns[-1]
        pivot["Total"] = pivot[last_period]
    else:
        pivot["Total"] = 0.0

    # -----------------------------
    # 2) Points GM (bonus/malus) depuis data/gm_points.csv
    # -----------------------------
    owners = [str(o) for o in pivot.index.tolist()]
    gm_df, gm_path, gm_created = _load_or_create_gm_points(data_dir, owners)
    gm_map = dict(zip(gm_df["GM"].tolist(), gm_df["Points"].tolist()))

    pivot["GM_Points"] = [int(gm_map.get(str(o), 0)) for o in pivot.index.tolist()]
    pivot["Total_Final"] = pivot["Total"] + pivot["GM_Points"]

    # Sort by Total_Final
    pivot_sorted = pivot.sort_values("Total_Final", ascending=False)

    # Info + download gm_points.csv (idiot-proof)
    with st.expander("🏆 Points GM (bonus) — info", expanded=False):
        if os.path.exists(gm_path):
            st.caption(f"Fichier: `{gm_path}`")
            if gm_created:
                st.success("✅ gm_points.csv a été créé automatiquement (points=0).")
            b = _read_file_bytes(gm_path)
            if b:
                st.download_button(
                    "📥 Télécharger gm_points.csv (mets-le dans ton repo /data/)",
                    data=b,
                    file_name="gm_points.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("✅ Option A: mets `data/gm_points.csv` dans ton repo (commit/push) pour ne pas le perdre après redémarrage.")
        st.dataframe(gm_df.sort_values(["Points","GM"], ascending=[False, True]), use_container_width=True, hide_index=True)

    # -----------------------------
    # 3) Affichages
    # -----------------------------
    st.subheader("🏅 Classement (Total + Bonus GM)")
    rank_df = pivot_sorted[["Total", "GM_Points", "Total_Final"]].reset_index().rename(columns={"owner": "Équipe"})
    rank_df["Total"] = rank_df["Total"].apply(_fmt_pts)
    rank_df["GM_Points"] = rank_df["GM_Points"].apply(_fmt_pts)
    rank_df["Total_Final"] = rank_df["Total_Final"].apply(_fmt_pts)
    rank_df = rank_df.rename(columns={"GM_Points": "Bonus GM", "Total_Final": "Total (final)"})
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    # Movers (si on a au moins 2 périodes) — variation sans bonus GM (plus logique)
    if len(periods) >= 2:
        st.subheader("📈 Variation (dernier - précédent) — sans bonus GM")
        prev_period = periods[-2]
        delta = (pivot_sorted[periods[-1]] - pivot_sorted[prev_period]).rename("Δ").reset_index()
        delta = delta.rename(columns={"owner": "Équipe"})
        delta["Δ"] = delta["Δ"].apply(_fmt_pts)
        st.dataframe(delta.sort_values("Δ", ascending=False), use_container_width=True, hide_index=True)

    st.divider()

    # Table par périodes + total
    st.subheader("🧾 Points par période + Total")
    view = pivot_sorted.reset_index().rename(columns={"owner": "Équipe"}).copy()

    # Reorder columns: periods, Total, Bonus, Total_Final
    cols = ["Équipe"] + periods + ["Total", "GM_Points", "Total_Final"]
    cols = [c for c in cols if c in view.columns]
    view = view[cols].copy()
    view = view.rename(columns={"GM_Points": "Bonus GM", "Total_Final": "Total (final)"})

    # Format
    for c in view.columns:
        if c != "Équipe":
            view[c] = view[c].apply(_fmt_pts)

    st.dataframe(view, use_container_width=True, hide_index=True)

    with st.expander("🔎 Debug fichier", expanded=False):

        # Historique (Option A)
        hist_path = _pick_history_path(data_dir)
        st.caption(f"Historique (auto): `{hist_path}`")
        hb = _read_file_bytes(hist_path)
        if hb:
            st.download_button(
                "📥 Télécharger historique (CSV) (mets-le dans ton repo /data/)",
                data=hb,
                file_name=os.path.basename(hist_path),
                mime="text/csv",
                use_container_width=True,
                key="dl_hist_rank",
            )
            st.caption("✅ Option A: mets ce fichier d’historique dans ton repo (commit/push) pour ne pas le perdre après redémarrage.")
        else:
            st.info("Aucun historique trouvé (il sera créé automatiquement quand tu ajoutes un joueur ou fais une action Admin).")

        st.caption(f"Source points: `{p_path}`")
        st.caption(f"Source GM points: `{gm_path}`")
        st.write("Colonnes points:", list(pts.columns))
        st.write("Périodes détectées:", periods)
