# tabs/classement.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def _season(ctx: dict) -> str:
    return str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"


def _money(v: int | float) -> str:
    try:
        n = int(round(float(v)))
    except Exception:
        n = 0
    return f"{n:,}".replace(",", " ") + " $"


def render(ctx: dict) -> None:
    st.header("🏆 Classement")
    season = _season(ctx)
    st.caption(f"Saison: **{season}**")

    df = ctx.get("data")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Aucune donnée roster globale (`ctx['data']`) — va d’abord importer tes rosters.")
        return

    # Colonnes minimales
    if "Propriétaire" not in df.columns:
        st.error("Colonne `Propriétaire` manquante dans les données (roster).")
        return
    if "Salaire" not in df.columns:
        st.error("Colonne `Salaire` manquante dans les données (roster).")
        return

    work = df.copy()
    work["Propriétaire"] = work["Propriétaire"].astype(str).str.strip()
    work["Salaire"] = pd.to_numeric(work["Salaire"], errors="coerce").fillna(0).astype(int)

    # Paramètres caps (fallbacks)
    cap_gc = int(ctx.get("PLAFOND_GC") or st.session_state.get("PLAFOND_GC") or 95_500_000)
    cap_ce = int(ctx.get("PLAFOND_CE") or st.session_state.get("PLAFOND_CE") or 47_750_000)
    STATUT_GC = str(ctx.get("STATUT_GC") or st.session_state.get("STATUT_GC") or "Grand Club")
    STATUT_CE = str(ctx.get("STATUT_CE") or st.session_state.get("STATUT_CE") or "Club École")

    # Agrégation
    rows = []
    owners = sorted([o for o in work["Propriétaire"].dropna().unique() if str(o).strip()])

    for o in owners:
        sub = work[work["Propriétaire"] == o].copy()

        total = int(sub["Salaire"].sum())
        n_players = int(len(sub))

        if "Statut" in sub.columns:
            used_gc = int(pd.to_numeric(sub.loc[sub["Statut"] == STATUT_GC, "Salaire"], errors="coerce").fillna(0).sum())
            used_ce = int(pd.to_numeric(sub.loc[sub["Statut"] == STATUT_CE, "Salaire"], errors="coerce").fillna(0).sum())
        else:
            used_gc = total
            used_ce = 0

        rows.append(
            {
                "Équipe": o,
                "Joueurs": n_players,
                "Total GC": used_gc,
                "Reste GC": cap_gc - used_gc,
                "Total CE": used_ce,
                "Reste CE": cap_ce - used_ce,
                "Total": total,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        st.info("Aucune équipe détectée.")
        return

    # Tri: par dépassement GC d'abord, sinon par masse totale
    out["_over_gc"] = out["Reste GC"] < 0
    out = out.sort_values(by=["_over_gc", "Total"], ascending=[False, False])

    # KPIs rapides
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Équipes", len(out))
    with col2:
        st.metric("Cap GC", _money(cap_gc))
    with col3:
        st.metric("Cap CE", _money(cap_ce))

    st.divider()

    # Table affichage (format money)
    view = out.copy()
    for c in ["Total GC", "Reste GC", "Total CE", "Reste CE", "Total"]:
        view[c] = view[c].apply(_money)

    view = view.drop(columns=["_over_gc"], errors="ignore")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.caption("Tri: dépassement GC (en haut), puis masse totale.")
