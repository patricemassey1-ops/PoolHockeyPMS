# tabs/gm.py
from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional

import pandas as pd
import streamlit as st

from services.storage import path_players_db, path_contracts


# ----------------------------
# Helpers (safe, standalone)
# ----------------------------
def _norm_name(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def money(v: int | float) -> str:
    try:
        n = int(round(float(v)))
    except Exception:
        n = 0
    # format style "92 888 000 $"
    s = f"{n:,}".replace(",", " ")
    return f"{s} $"


def _load_csv_safe(path: str) -> pd.DataFrame:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _first_existing(paths: list[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _team_logo_path(owner: str, assets_dir: str, data_dir: str) -> Optional[str]:
    """
    Cherche un logo dans assets/previews d'abord, sinon data/.
    Supporte tes patterns: Whalers_Logo.png, WhalersE_Logo.png, etc.
    """
    base = str(owner or "").strip()
    if not base:
        return None

    candidates = [
        f"{base}_Logo.png",
        f"{base}E_Logo.png",
        f"{base}_logo.png",
        f"{base}E_logo.png",
        f"{base}.png",
    ]

    # petites normalisations (Red Wings -> Red_Wings)
    base2 = base.replace(" ", "_")
    if base2 != base:
        candidates = [
            f"{base2}_Logo.png",
            f"{base2}E_Logo.png",
            f"{base2}_logo.png",
            f"{base2}E_logo.png",
            f"{base2}.png",
        ] + candidates

    # aussi: retirer accents
    base3 = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base3 = base3.replace(" ", "_")
    if base3 and base3 not in (base, base2):
        candidates = [
            f"{base3}_Logo.png",
            f"{base3}E_Logo.png",
            f"{base3}.png",
        ] + candidates

    for root in [assets_dir, data_dir]:
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.exists(p):
                return p

    return None


def _safe_owner(ctx: dict) -> str:
    # le plus robuste possible selon tes versions
    for k in ["selected_owner", "selected_team", "owner", "team"]:
        v = ctx.get(k)
        if v:
            return str(v).strip()
    for k in ["selected_owner", "selected_team", "owner", "team"]:
        v = st.session_state.get(k)
        if v:
            return str(v).strip()
    return ""


# ----------------------------
# Main tab
# ----------------------------
def render(ctx: dict) -> None:
    owner = _safe_owner(ctx)
    if not owner:
        st.subheader("🧊 GM")
        st.info("Sélectionne une équipe (propriétaire) dans Home/Alignement.")
        return

    season = str(ctx.get("season") or st.session_state.get("season") or "2025-2026").strip() or "2025-2026"
    data_dir = str(ctx.get("DATA_DIR") or "data")
    assets_dir = str(ctx.get("ASSETS_DIR") or "assets/previews")

    df = ctx.get("data")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    # Filtrer équipe
    dprop = df.copy()
    if not dprop.empty and "Propriétaire" in dprop.columns:
        dprop = dprop[dprop["Propriétaire"].astype(str).str.strip().eq(owner)].copy()

    # ----------------------------
    # Header GM (logo + caps)
    # ----------------------------
    colL, colR = st.columns([1.2, 3], vertical_alignment="center")

    with colL:
        gm_logo = "gm_logo.png"
        if os.path.exists(gm_logo):
            st.image(gm_logo, width=110)
        else:
            logo = _team_logo_path(owner, assets_dir=assets_dir, data_dir=data_dir)
            if logo:
                st.image(logo, width=110)

        st.markdown(
            f"<div style='font-size:22px;font-weight:900;margin-top:6px;'>🧊 GM — {owner}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Saison: **{season}**")

    # Caps (fallbacks safe)
    cap_gc = int(ctx.get("PLAFOND_GC") or st.session_state.get("PLAFOND_GC") or 95_500_000)
    cap_ce = int(ctx.get("PLAFOND_CE") or st.session_state.get("PLAFOND_CE") or 47_750_000)

    # Statuts (fallbacks safe)
    STATUT_GC = str(ctx.get("STATUT_GC") or st.session_state.get("STATUT_GC") or "Grand Club")
    STATUT_CE = str(ctx.get("STATUT_CE") or st.session_state.get("STATUT_CE") or "Club École")

    with colR:
        used_gc = 0
        used_ce = 0
        if not dprop.empty and "Salaire" in dprop.columns:
            # si "Statut" existe -> split GC/CE ; sinon tout en GC
            if "Statut" in dprop.columns:
                used_gc = int(pd.to_numeric(dprop.loc[dprop["Statut"] == STATUT_GC, "Salaire"], errors="coerce").fillna(0).sum())
                used_ce = int(pd.to_numeric(dprop.loc[dprop["Statut"] == STATUT_CE, "Salaire"], errors="coerce").fillna(0).sum())
            else:
                used_gc = int(pd.to_numeric(dprop["Salaire"], errors="coerce").fillna(0).sum())
                used_ce = 0

        r_gc = cap_gc - used_gc
        r_ce = cap_ce - used_ce

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div style="padding:14px;border-radius:14px;background:rgba(255,255,255,.05)">
                  <div style="font-size:13px;opacity:.8">Masse Grand Club</div>
                  <div style="font-size:26px;font-weight:900;margin:4px 0">{money(used_gc)}</div>
                  <div style="font-size:13px;opacity:.75">Plafond {money(cap_gc)}</div>
                  <div style="font-size:14px;font-weight:800;color:{'#ef4444' if r_gc < 0 else '#22c55e'}">
                    Reste {money(r_gc)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div style="padding:14px;border-radius:14px;background:rgba(255,255,255,.05)">
                  <div style="font-size:13px;opacity:.8">Masse Club École</div>
                  <div style="font-size:26px;font-weight:900;margin:4px 0">{money(used_ce)}</div>
                  <div style="font-size:13px;opacity:.75">Plafond {money(cap_ce)}</div>
                  <div style="font-size:14px;font-weight:800;color:{'#ef4444' if r_ce < 0 else '#22c55e'}">
                    Reste {money(r_ce)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ----------------------------
    # Filtres Alignement (rapide) - compat avec ton legacy
    # ----------------------------
    with st.expander("🎛️ Filtres Alignement (rapide)", expanded=False):
        f1, f2, f3 = st.columns([2.2, 1, 1])
        with f1:
            st.text_input(
                "Recherche joueur",
                value=str(st.session_state.get("align_filter_q", "") or ""),
                key="align_filter_q",
                placeholder="ex: Suzuki",
            )
        with f2:
            st.checkbox("ELC", value=bool(st.session_state.get("align_filter_only_elc", False)), key="align_filter_only_elc")
        with f3:
            st.checkbox("STD", value=bool(st.session_state.get("align_filter_only_std", False)), key="align_filter_only_std")

    st.divider()

    # ----------------------------
    # Contrats (merge Players DB + PuckPedia)
    # ----------------------------
    st.subheader("📄 Contrats — équipe complète")
    st.caption("Source: data/hockey.players.csv + data/puckpedia.contracts.csv (si dispo).")

    if dprop.empty:
        st.info("Aucun joueur trouvé pour cette équipe (vérifie ton import roster).")
        return

    # Colonnes roster minimales
    show = dprop.copy()
    for col, default in {"Joueur": "", "Pos": "", "Equipe": "", "Salaire": 0, "Level": "", "Statut": "", "Slot": ""}.items():
        if col not in show.columns:
            show[col] = default

    # Load sources
    pdb_path = path_players_db()
    ctc_path = path_contracts()
    pdb = _load_csv_safe(pdb_path)
    ctc = _load_csv_safe(ctc_path)

    # Normalize keys
    show["_k"] = show["Joueur"].astype(str).map(_norm_name)

    # Players DB merge
    if not pdb.empty:
        # colonne nom dans hockey.players.csv : parfois "Player" ou "Joueur"
        name_col = "Player" if "Player" in pdb.columns else ("Joueur" if "Joueur" in pdb.columns else None)
        if name_col:
            p = pdb.copy()
            p["_k"] = p[name_col].astype(str).map(_norm_name)

            keep_cols = []
            for c in ["Country", "Flag", "FlagISO2", "Level", "Expiry Year", "nhl_id", "Cap Hit"]:
                if c in p.columns:
                    keep_cols.append(c)

            if keep_cols:
                p = p[["_k"] + keep_cols].drop_duplicates("_k")
                show = show.merge(p, on="_k", how="left", suffixes=("", "_pdb"))

                # Prefer Players DB level if roster empty
                if "Level_pdb" in show.columns:
                    show["Level"] = show["Level"].astype(str)
                    show["Level_pdb"] = show["Level_pdb"].astype(str)
                    show["Level"] = show["Level"].where(show["Level"].str.strip().ne(""), show["Level_pdb"])

                # Prefer Players DB Cap Hit if Salaire empty/0
                if "Cap Hit" in show.columns:
                    sal = pd.to_numeric(show["Salaire"], errors="coerce").fillna(0)
                    caphit = pd.to_numeric(show["Cap Hit"], errors="coerce").fillna(0)
                    show["Salaire"] = sal.where(sal > 0, caphit)

    # PuckPedia merge (si dispo)
    if not ctc.empty:
        # essaie de deviner la colonne nom
        c_name = None
        for cand in ["Player", "Name", "player", "name"]:
            if cand in ctc.columns:
                c_name = cand
                break
        if c_name:
            c = ctc.copy()
            c["_k"] = c[c_name].astype(str).map(_norm_name)

            keep_cols = []
            for cand in ["Expiry Year", "Expiry", "contract_end", "Cap Hit", "AAV", "Level", "contract_level"]:
                if cand in c.columns:
                    keep_cols.append(cand)

            if keep_cols:
                c = c[["_k"] + keep_cols].drop_duplicates("_k")
                show = show.merge(c, on="_k", how="left", suffixes=("", "_ctc"))

                # Expiry preference: roster -> players_db -> puckpedia
                if "Expiry Year" not in show.columns:
                    show["Expiry Year"] = ""

                for alt in ["Expiry Year_ctc", "Expiry_ctc", "contract_end_ctc"]:
                    if alt in show.columns:
                        show["Expiry Year"] = show["Expiry Year"].where(
                            show["Expiry Year"].astype(str).str.strip().ne(""),
                            show[alt],
                        )

                # Cap Hit preference if salary still 0
                for alt in ["Cap Hit_ctc", "AAV_ctc"]:
                    if alt in show.columns:
                        sal = pd.to_numeric(show["Salaire"], errors="coerce").fillna(0)
                        caphit = pd.to_numeric(show[alt], errors="coerce").fillna(0)
                        show["Salaire"] = sal.where(sal > 0, caphit)

                # Level preference
                for alt in ["Level_ctc", "contract_level_ctc"]:
                    if alt in show.columns:
                        show["Level"] = show["Level"].astype(str)
                        show[alt] = show[alt].astype(str)
                        show["Level"] = show["Level"].where(show["Level"].str.strip().ne(""), show[alt])

    # Nettoyage & affichage
    if "Salaire" in show.columns:
        show["Salaire"] = pd.to_numeric(show["Salaire"], errors="coerce").fillna(0).astype(int)

    cols_order = []
    for c in ["Joueur", "Pos", "Equipe", "Salaire", "Level", "Expiry Year", "Statut", "Slot", "Country", "Flag", "FlagISO2", "nhl_id"]:
        if c in show.columns:
            cols_order.append(c)

    view = show[cols_order].copy() if cols_order else show.copy()
    # tri par salaire desc
    if "Salaire" in view.columns:
        view = view.sort_values("Salaire", ascending=False)

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.caption(f"Players DB: `{pdb_path}` | Contracts: `{ctc_path}`")
