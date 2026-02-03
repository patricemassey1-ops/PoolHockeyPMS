# tabs/historique.py
from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st

from services.event_log import read_events, event_log_path

# -----------------------------------------------------
# Cache helpers (speed: avoid re-reading event log on every rerun)
# -----------------------------------------------------
def _file_sig(path: str) -> tuple[int, int]:
    try:
        st_ = os.stat(path)
        return (int(st_.st_mtime_ns), int(st_.st_size))
    except Exception:
        return (0, 0)

@st.cache_data(show_spinner=False)
def _read_events_cached(path: str, sig: tuple[int, int]):
    try:
        return _read_events_cached(path, _file_sig(path))
    except Exception:
        return []


def _data_dir(ctx: dict) -> str:
    return str(ctx.get("DATA_DIR") or "data")


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


def _owner(ctx: dict) -> str:
    return str(st.session_state.get("selected_owner") or ctx.get("selected_owner") or "").strip()


def render(ctx: dict) -> None:
    st.header("🕘 Historique")
    data_dir = _data_dir(ctx)
    season = _season(ctx)
    owner = _owner(ctx)

    path = event_log_path(data_dir, season)
    st.caption(f"Saison: **{season}**")
    st.caption(f"Journal: `{path}`")

    ev = read_events(data_dir, season)

    if ev.empty:
        st.info("Aucun événement pour l’instant. À partir de maintenant, toutes les actions (imports, transactions, moves) vont s’enregistrer ici.")
        with st.expander("✅ Créer un événement test", expanded=False):
            if st.button("➕ Ajouter 'App started'"):
                from services.event_log import append_event
                res = append_event(
                    data_dir=data_dir,
                    season=season,
                    owner=owner,
                    event_type="system",
                    summary="App started",
                    payload={"note": "first event"},
                )
                if res.get("ok"):
                    st.success("OK — refresh la page.")
                else:
                    st.error(res.get("error") or "error")
        return

    # nettoyage types
    for c in ["timestamp", "season", "owner", "type", "summary", "payload_json"]:
        if c not in ev.columns:
            ev[c] = ""

    ev["timestamp_dt"] = pd.to_datetime(ev["timestamp"], errors="coerce")
    ev = ev.sort_values("timestamp_dt", ascending=False)

    # filtres
    with st.expander("🔎 Filtres", expanded=False):
        c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
        with c1:
            only_mine = st.checkbox("Seulement mon équipe", value=False, key="hist_only_mine")
        with c2:
            types = ["(Tous)"] + sorted([t for t in ev["type"].dropna().astype(str).unique() if t.strip()])
            t_pick = st.selectbox("Type", types, index=0, key="hist_type_pick")
        with c3:
            q = st.text_input("Recherche", value="", key="hist_q", placeholder="joueur / action / équipe...")

    view = ev.copy()

    if only_mine and owner:
        view = view[view["owner"].astype(str).str.strip() == owner]

    if t_pick and t_pick != "(Tous)":
        view = view[view["type"].astype(str) == t_pick]

    if q.strip():
        qq = q.strip().lower()
        mask = (
            view["summary"].astype(str).str.lower().str.contains(qq, na=False)
            | view["owner"].astype(str).str.lower().str.contains(qq, na=False)
            | view["type"].astype(str).str.lower().str.contains(qq, na=False)
            | view["payload_json"].astype(str).str.lower().str.contains(qq, na=False)
        )
        view = view[mask]

    # timeline simple
    st.subheader("🧾 Timeline")
    for _, r in view.head(60).iterrows():
        ts = r.get("timestamp", "")
        typ = r.get("type", "")
        ow = r.get("owner", "")
        summ = r.get("summary", "")

        st.markdown(f"**{ts}** — `{typ}` — **{ow or '—'}**")
        st.write(summ or "")
        with st.expander("payload", expanded=False):
            try:
                st.json(json.loads(r.get("payload_json") or "{}"))
            except Exception:
                st.code(str(r.get("payload_json") or ""))

        st.divider()

    with st.expander("📄 Tableau complet", expanded=False):
        show = view.drop(columns=["timestamp_dt"], errors="ignore")
        st.dataframe(show, use_container_width=True, hide_index=True)
