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
        if not path:
            return pd.DataFrame(), "Chemin CSV vide."
        if not os.path.exists(path):
            return pd.DataFrame(), f"Fichier introuvable: {path}"
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Erreur lecture CSV: {e}"


def save_csv(df: pd.DataFrame, path: str, *, safe_mode: bool = True) -> str | None:
    """SAFE MODE = protège contre écrasement catastrophique (ex: perte NHL_ID)."""
    try:
        if safe_mode:
            # minimal guard: refuse if NHL_ID col exists but is entirely empty
            id_col = _resolve_nhl_id_col(df)
            if id_col:
                s = pd.to_numeric(df[id_col], errors="coerce")
                if int(s.notna().sum()) == 0:
                    return "SAFE MODE: Refus d'écrire — NHL_ID serait 0/100% (colonne vide)."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return None
    except Exception as e:
        return f"Erreur écriture CSV: {e}"


# =========================
# Column resolution
# =========================
def _norm_col(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _resolve_nhl_id_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["nhl_id", "nhlid", "id_nhl", "player_id", "playerid", "nhlplayerid", "nhl_id_api"]:
        if key in cmap:
            return cmap[key]
    # tolerate exact
    for c in df.columns:
        if str(c).strip().lower() in ("nhl_id", "nhl id"):
            return c
    return None


def _resolve_player_name_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["player", "joueur", "name", "nom", "player_name", "full_name"]:
        if key in cmap:
            return cmap[key]
    return None


def _resolve_team_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["team", "equipe", "club", "nhl_team", "team_abbrev"]:
        if key in cmap:
            return cmap[key]
    return None


def _normalize_player_name(x: str) -> str:
    x = str(x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    # support "Last, First"
    if "," in x:
        parts = [p.strip() for p in x.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            x = f"{parts[1]} {parts[0]}"
    # remove accents-ish and punctuation
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# =========================
# Audit + confidence
# =========================
CONF_BY_SOURCE = {
    "existing": 0.95,
    "roster": 0.80,
    "equipes": 0.75,
    "manual": 0.70,
    "fallback": 0.55,
    "unknown": 0.50,
}


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
        "dup": dup_cnt,
        "dup_pct": dup_pct,
    }


def build_audit_report(df: pd.DataFrame, id_col: str, name_col: str | None, team_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["row"] = np.arange(len(df))
    if name_col and name_col in df.columns:
        out["player"] = df[name_col].astype(str)
    if team_col and team_col in df.columns:
        out["team"] = df[team_col].astype(str)
    out["nhl_id"] = pd.to_numeric(df[id_col], errors="coerce")
    out["missing"] = out["nhl_id"].isna()
    out["dup"] = out["nhl_id"].notna() & out["nhl_id"].duplicated(keep=False)

    if "nhl_id_source" in df.columns:
        out["source"] = df["nhl_id_source"].astype(str)
    else:
        out["source"] = ""

    if "nhl_id_confidence" in df.columns:
        out["confidence"] = pd.to_numeric(df["nhl_id_confidence"], errors="coerce")
    else:
        out["confidence"] = np.nan

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
    if "nhl_id_confidence" not in p.columns:
        p["nhl_id_confidence"] = np.nan

    s_id = _resolve_nhl_id_col(s)
    s_name = _resolve_player_name_col(s) or name_col
    s_team = _resolve_team_col(s) if team_col else None
    if not s_id:
        return p, {"filled": 0, "reason": "Aucune colonne NHL_ID détectée dans la source."}

    # mapping key -> id
    s_ids = pd.to_numeric(s[s_id], errors="coerce")
    s_names = s[s_name].astype(str).apply(_normalize_player_name)
    if s_team and s_team in s.columns:
        s_teams = s[s_team].astype(str).str.upper().str.strip()
    else:
        s_teams = pd.Series([""] * len(s), index=s.index)

    src_map: Dict[str, int] = {}
    for nm, tm, pid in zip(s_names.tolist(), s_teams.tolist(), s_ids.tolist()):
        if pd.isna(pid) or int(pid) <= 0:
            continue
        k = f"{tm}||{nm}"
        if k not in src_map:
            src_map[k] = int(pid)

    p_ids = pd.to_numeric(p[id_col], errors="coerce")
    missing_mask = p_ids.isna()

    p_names = p[name_col].astype(str).apply(_normalize_player_name)
    if team_col and team_col in p.columns:
        p_teams = p[team_col].astype(str).str.upper().str.strip()
    else:
        p_teams = pd.Series([""] * len(p), index=p.index)

    filled = 0
    for ix in p.index[missing_mask]:
        k = f"{p_teams.at[ix]}||{p_names.at[ix]}"
        if k in src_map:
            p.at[ix, id_col] = src_map[k]
            p.at[ix, "nhl_id_source"] = source_tag
            p.at[ix, "nhl_id_confidence"] = float(conf)
            filled += 1

    # mark existing
    p_ids2 = pd.to_numeric(p[id_col], errors="coerce")
    for ix in p.index[p_ids2.notna()]:
        if not str(p.at[ix, "nhl_id_source"] or "").strip():
            p.at[ix, "nhl_id_source"] = "existing"
            p.at[ix, "nhl_id_confidence"] = 0.95

    return p, {"filled": filled, "unique_keys": len(src_map)}


# =========================
# Prod lock
# =========================
def is_prod_env() -> bool:
    v = (os.getenv("PMS_ENV") or os.getenv("ENV") or "").strip().lower()
    if v in ("prod", "production", "cloud", "streamlit"):
        return True
    # secrets opt-in
    sv = str(st.secrets.get("ENV", "") if hasattr(st, "secrets") else "").strip().lower()
    return sv in ("prod", "production")


# =========================
# UI render
# =========================
def render(ctx: Dict[str, Any] | None = None) -> None:
    ctx = ctx or {}
    data_dir = str(ctx.get("data_dir") or "data")
    season = str(ctx.get("season") or ctx.get("season_lbl") or "2025-2026")

    st.title("🛠️ Gestion Admin")

    tab = st.radio("Sections", ["Backups", "Joueurs", "Outils"], horizontal=True, key="admin_section")
    if tab == "Backups":
        st.subheader("📦 Backups")
        st.info("Mini-version SAFE: la section Backups est volontairement légère ici.")
        st.caption("Tu peux réintégrer la version Drive complète ensuite — on sécurise d'abord NHL_ID.")
        return

    if tab == "Joueurs":
        st.subheader("👥 Joueurs")
        st.info("Mini-version SAFE: pas de gestion avancée ici (encore).")
        return

    # -------- Outils / NHL_ID
    st.subheader("🧰 Outils — synchros")

    players_path = os.path.join(data_dir, "hockey.players.csv")
    players2 = st.text_input("Players DB (NHL_ID)", value=players_path)

    limit = st.number_input("Max par run", min_value=100, max_value=5000, step=100, value=1000)
    dry = st.checkbox("Dry-run (ne sauvegarde pas)", value=False)

    # SAFE MODE: protect writes
    override_safe = st.checkbox("Override SAFE MODE (autoriser une baisse NHL_ID)", value=False)

    prod = is_prod_env()
    st.caption(f"🔒 Prod lock: {'ON' if prod else 'OFF'} (ENV/PMS_ENV).")

    # --- choose recovery source
    st.markdown("### Source de récupération (optionnel)")
    candidates = []
    # season roster exports you uploaded
    for fn in [f"roster_{season}.csv", f"roster_filtered_{season}.csv", f"equipes_joueurs_{season}.csv"]:
        pth = os.path.join(data_dir, fn)
        if os.path.exists(pth):
            candidates.append(pth)

    upload = st.file_uploader("Ou uploader un CSV source", type=["csv"], accept_multiple_files=False)
    source_df = None
    source_tag = "unknown"
    conf = 0.70

    if upload is not None:
        try:
            source_df = pd.read_csv(upload, low_memory=False)
            source_tag = "manual"
            conf = 0.70
            st.success("Source uploadée prête.")
        except Exception as e:
            st.error(f"Erreur lecture upload: {e}")

    default_choice = None
    if candidates:
        # best default: filtered roster > roster > equipes
        pref = [f"roster_filtered_{season}.csv", f"roster_{season}.csv", f"equipes_joueurs_{season}.csv"]
        for pf in pref:
            pp = os.path.join(data_dir, pf)
            if pp in candidates:
                default_choice = pp
                break
        default_choice = default_choice or candidates[0]

    if source_df is None:
        choice = st.selectbox(
            "Récupérer NHL_ID depuis…",
            options=["Aucune (API NHL uniquement)"] + candidates,
            index=(1 + candidates.index(default_choice)) if (default_choice and default_choice in candidates) else 0,
        )
        if choice != "Aucune (API NHL uniquement)":
            source_df, errS = load_csv(choice)
            if errS:
                st.error(errS)
                source_df = None
            else:
                # tag by filename
                bn = os.path.basename(choice).lower()
                if "roster" in bn:
                    source_tag = "roster"
                    conf = 0.80
                elif "equipes" in bn:
                    source_tag = "equipes"
                    conf = 0.75
                else:
                    source_tag = "fallback"
                    conf = 0.55

    # --- verify button (audit)
    if st.button("🔎 Vérifier l'état des NHL_ID"):
        df, err = load_csv(players2)
        if err:
            st.error(err)
        else:
            id_col = _resolve_nhl_id_col(df) or "NHL_ID"
            name_col = _resolve_player_name_col(df) or "Player"
            team_col = _resolve_team_col(df)
            # ✅ Vérifier doit être tolérant: on peut ne pas avoir encore NHL_ID.
            if name_col not in df.columns:
                st.error("Colonne joueur introuvable (ex: Joueur / Player / player_name).")
                st.stop()
            if id_col not in df.columns:
                df[id_col] = np.nan
                st.info(f"Colonne {id_col} absente → créée (tous manquants pour l’instant).")
            a = audit_nhl_ids(df, id_col)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total joueurs", a["total"])
            c2.metric("Avec ID (NHL_ID)", a["with_id"])
            c3.metric("Manquants", a["missing"])
            c4.metric("% manquants", f"{a['missing_pct']:.1f}%")
            st.warning(f"IDs dupliqués détectés: {a['dup']}  (≈{a['dup_pct']:.1f}% des IDs présents).")

            # audit report + heatmap
            name_col = _resolve_player_name_col(df)
            team_col = _resolve_team_col(df)
            audit_df = build_audit_report(df, id_col, name_col, team_col)

            # ensure confidence exists for display
            if "nhl_id_confidence" not in df.columns:
                audit_df["confidence"] = np.where(audit_df["missing"], 0.0, 0.95)

            heat = confidence_heatmap(audit_df)

            st.markdown("#### 🧩 Heatmap de confiance (par équipe × bucket)")
            st.dataframe(heat.style.background_gradient(axis=None), use_container_width=True)

            # CSV audit download
            out = io.StringIO()
            audit_df.to_csv(out, index=False)
            st.download_button(
                "🧾 Télécharger rapport d'audit NHL_ID (CSV)",
                data=out.getvalue().encode("utf-8"),
                file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # --- apply recovery + (optionally) API later
    lock_dup_pct = st.slider("🛑 Seuil blocage duplication (%)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    allow_prod_override = st.checkbox("Déverrouiller en prod (admin)", value=False, disabled=(not prod))

    if st.button("🔗 Associer NHL_ID"):
        df, err = load_csv(players2)
        if err:
            st.error(err)
            return

        id_col = _resolve_nhl_id_col(df) or "NHL_ID"
        name_col = _resolve_player_name_col(df) or "Player"
        team_col = _resolve_team_col(df)

        # ✅ Associer doit être tolérant: NHL_ID peut ne pas exister encore.
        if name_col not in df.columns:
            st.error("Colonne joueur introuvable (ex: Joueur / Player / player_name).")
            return
        if id_col not in df.columns:
            df[id_col] = np.nan
            st.info(f"Colonne {id_col} absente → créée (prête à remplir).")

        # init tracking cols
        if "nhl_id_source" not in df.columns:
            df["nhl_id_source"] = ""
        if "nhl_id_confidence" not in df.columns:
            df["nhl_id_confidence"] = np.nan

        # recover
        filled = 0
        if source_df is not None and not source_df.empty:
            df, stats = recover_from_source(
                df,
                source_df,
                id_col=id_col,
                name_col=name_col,
                team_col=team_col,
                source_tag=source_tag,
                conf=conf,
            )
            filled = int(stats.get("filled", 0))

        # recompute audit + locks
        a = audit_nhl_ids(df, id_col)
        dup_pct = float(a["dup_pct"])
        if prod and (dup_pct > lock_dup_pct) and not allow_prod_override:
            st.error(
                f"🔒 PROD LOCK: écriture bloquée — duplication {dup_pct:.1f}% > seuil {lock_dup_pct:.1f}%."
            )
            st.info("Corrige la source (mauvais match) ou coche 'Déverrouiller en prod (admin)'.")
            return

        # write (unless dry-run)
        if dry:
            st.info(f"Dry-run: +{filled} NHL_ID récupérés (aucune écriture).")
            return

        errw = save_csv(df, players2, safe_mode=(not override_safe))
        if errw:
            st.error(errw)
            return

        st.success(f"✅ Terminé — +{filled} NHL_ID récupérés. Duplication: {dup_pct:.1f}%.")

        # Show audit + downloads after write
        audit_df = build_audit_report(df, id_col, name_col, team_col)
        heat = confidence_heatmap(audit_df)
        st.markdown("#### 🧩 Heatmap de confiance (par équipe × bucket)")
        st.dataframe(heat.style.background_gradient(axis=None), use_container_width=True)

        out = io.StringIO()
        audit_df.to_csv(out, index=False)
        st.download_button(
            "🧾 Télécharger rapport d'audit NHL_ID (CSV)",
            data=out.getvalue().encode("utf-8"),
            file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# Backward-compat alias (si ton app appelle _render_tools)
def _render_tools(*args, **kwargs):
    return render(*args, **kwargs)
