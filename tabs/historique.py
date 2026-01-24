# tabs/historique.py
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st


def _load_csv_safe(path: str) -> pd.DataFrame:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


def _owner(ctx: dict) -> str:
    return str(st.session_state.get("selected_owner") or ctx.get("selected_owner") or "").strip()


def _transactions_path(data_dir: str, season: str) -> str:
    return os.path.join(data_dir, f"transactions_{season}.csv")


def _backup_history_path(data_dir: str) -> str:
    return os.path.join(data_dir, "backup_history.csv")


def _coerce_ts(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.to_datetime(pd.Series([pd.NaT] * len(df)))
    return pd.to_datetime(df[col], errors="coerce")


def render(ctx: dict) -> None:
    st.header("🕘 Historique")
    data_dir = _data_dir(ctx)
    season = _season(ctx)
    owner = _owner(ctx)

    st.caption(f"Saison: **{season}**")
    if owner:
        st.caption(f"Équipe sélectionnée: **{owner}** (filtre optionnel ci-dessous)")

    # ==========================
    # Transactions (proposées / sauvées)
    # ==========================
    st.subheader("⚖️ Transactions")
    tx_path = _transactions_path(data_dir, season)
    tx = _load_csv_safe(tx_path)

    if tx.empty:
        st.info(f"Aucune transaction trouvée pour cette saison. (fichier attendu: `{tx_path}`)")
    else:
        # Normalisations colonnes (tolérant)
        for c in ["timestamp", "season", "owner_a", "owner_b", "status"]:
            if c not in tx.columns:
                tx[c] = ""

        tx["_ts"] = _coerce_ts(tx, "timestamp")
        tx = tx.sort_values("_ts", ascending=False)

        # Filtre
        with st.expander("🔎 Filtres Transactions", expanded=False):
            c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
            with c1:
                only_mine = st.checkbox("Seulement mon équipe", value=False, key="hist_tx_only_mine")
            with c2:
                status = st.selectbox(
                    "Statut",
                    ["(Tous)"] + sorted([s for s in tx["status"].dropna().astype(str).unique() if s.strip()]),
                    index=0,
                    key="hist_tx_status",
                )
            with c3:
                q = st.text_input("Recherche texte", value="", key="hist_tx_q", placeholder="joueur / équipe / pick...")

        view = tx.copy()

        if only_mine and owner:
            view = view[
                (view["owner_a"].astype(str).str.strip() == owner) | (view["owner_b"].astype(str).str.strip() == owner)
            ]

        if status and status != "(Tous)":
            view = view[view["status"].astype(str) == status]

        if q.strip():
            qq = q.strip().lower()
            mask = pd.Series([False] * len(view))
            for col in view.columns:
                if col.startswith("_"):
                    continue
                mask = mask | view[col].astype(str).str.lower().str.contains(qq, na=False)
            view = view[mask]

        # Affichage compact + détail
        cols_first = [c for c in ["timestamp", "owner_a", "owner_b", "status"] if c in view.columns]
        other_cols = [c for c in view.columns if c not in cols_first and not c.startswith("_")]
        show_cols = cols_first + other_cols[:8]  # on garde lisible
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

        with st.expander("📄 Détails (CSV complet)", expanded=False):
            st.caption(f"Source: `{tx_path}`")
            st.dataframe(tx.drop(columns=["_ts"], errors="ignore"), use_container_width=True, hide_index=True)

    st.divider()

    # ==========================
    # Backups (Drive/local history)
    # ==========================
    st.subheader("🧷 Backups & Restore — Historique")
    bh_path = _backup_history_path(data_dir)
    bh = _load_csv_safe(bh_path)

    if bh.empty:
        st.info(f"Aucun historique de backup trouvé. (fichier attendu: `{bh_path}`)")
        st.caption("Si tu utilises Drive Restore/Backup, ce fichier se remplit normalement.")
    else:
        # Essaye de repérer timestamp/createdTime
        ts_col = None
        for cand in ["createdTime", "timestamp", "time", "date"]:
            if cand in bh.columns:
                ts_col = cand
                break

        if ts_col:
            bh["_ts"] = _coerce_ts(bh, ts_col)
            bh = bh.sort_values("_ts", ascending=False)

        # Afficher
        st.dataframe(bh.drop(columns=["_ts"], errors="ignore"), use_container_width=True, hide_index=True)
        st.caption(f"Source: `{bh_path}`")

    st.divider()

    # ==========================
    # Logs / Info (utile en prod)
    # ==========================
    with st.expander("🧾 Info fichiers (debug)", expanded=False):
        st.write("Chemins attendus:")
        st.code(
            "\n".join(
                [
                    f"transactions: {tx_path}",
                    f"backup_history: {bh_path}",
                ]
            )
        )
