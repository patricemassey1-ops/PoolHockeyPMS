# tabs/admin.py — SAFE MINI (NHL_ID tools + prod lock + audit/heatmap + NHL Search generator)
# -----------------------------------------------------------------------------
# ✅ Fix “écran noir après Générer”
#   - render() enveloppé dans try/except → affiche l’exception au lieu d’un écran noir
#   - boutons/checkbox/slider ont des keys uniques (évite StreamlitDuplicateElementId)
#   - Générateur NHL Search API en chunks (limit=1000) + garde-fous anti-boucle + timeout
# ✅ Fix matplotlib manquant
#   - plus aucun .style.background_gradient() sans fallback (CSS heatmap)
# ✅ Fix nul_id
#   - nul_id est prioritaire pour team_col si présent
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

WKEY = "admin_nhlid_"  # prefix unique pour tous les widgets


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
        return pd.DataFrame(), f"Erreur lecture CSV: {type(e).__name__}: {e}"


def save_csv(df: pd.DataFrame, path: str, *, safe_mode: bool = True) -> str | None:
    """SAFE MODE = protège contre écrasement catastrophique (ex: perte NHL_ID)."""
    try:
        if safe_mode:
            id_col = _resolve_nhl_id_col(df)
            if id_col:
                s = pd.to_numeric(df[id_col], errors="coerce")
                if int(s.notna().sum()) == 0:
                    return "SAFE MODE: Refus d'écrire — NHL_ID serait 0/100% (colonne vide)."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return None
    except Exception as e:
        return f"Erreur écriture CSV: {type(e).__name__}: {e}"


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
    for key in ["nhl_id", "nhlid", "id_nhl", "player_id", "playerid", "nhlplayerid", "nhl_id_api", "playerid_api"]:
        if key in cmap:
            return cmap[key]
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
    """
    ✅ nul_id doit être considéré comme la colonne team si présent
    """
    if df is None or df.empty:
        return None
    cmap = {_norm_col(c): c for c in df.columns}
    for key in ["nul_id", "nulid", "nul_id_"]:
        if key in cmap:
            return cmap[key]
    for key in ["team", "equipe", "club", "nhl_team", "team_abbrev", "teamabbrev", "owner", "gm"]:
        if key in cmap:
            return cmap[key]
    return None


def _normalize_player_name(x: str) -> str:
    x = str(x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    if "," in x:
        parts = [p.strip() for p in x.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            x = f"{parts[1]} {parts[0]}"
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
        "dup_cnt": dup_cnt,
        "dup_pct": dup_pct,
    }


def build_audit_report(df: pd.DataFrame, id_col: str, name_col: str, team_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame()
    out["player_name"] = df[name_col].astype(str) if name_col in df.columns else ""
    if team_col and team_col in df.columns:
        out["team"] = df[team_col].astype(str)
    else:
        out["team"] = "(none)"
    s = pd.to_numeric(df[id_col], errors="coerce") if id_col in df.columns else pd.Series([np.nan] * len(df))
    out["nhl_id"] = s

    out["missing"] = out["nhl_id"].isna()
    dupmask = out["nhl_id"].dropna().duplicated(keep=False)
    out["duplicate_id"] = False
    out.loc[out["nhl_id"].notna(), "duplicate_id"] = dupmask.values

    conf = np.where(out["missing"], 0.0, np.where(out["duplicate_id"], 0.2, 0.95))
    out["confidence"] = conf.astype(float)
    return out


def confidence_heatmap(audit_df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.6, 0.75, 0.85, 1.01]
    labels = ["<0.60", "0.60-0.75", "0.75-0.85", ">=0.85"]

    conf = pd.to_numeric(audit_df.get("confidence", 0.0), errors="coerce").fillna(0.0)
    bucket = pd.cut(conf, bins=bins, labels=labels, right=False).astype(str)
    team = audit_df.get("team", "(none)").fillna("(none)").astype(str)

    piv = pd.pivot_table(
        pd.DataFrame({"team": team, "bucket": bucket}),
        index="team",
        columns="bucket",
        values="bucket",
        aggfunc="count",
        fill_value=0,
    )
    for lab in labels:
        if lab not in piv.columns:
            piv[lab] = 0
    return piv[labels]


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


def render_heatmap_safe(heat: pd.DataFrame) -> None:
    if heat is None or getattr(heat, "empty", True):
        st.info("Heatmap: aucune donnée.")
        return
    # 1) try background_gradient (requires matplotlib)
    try:
        st.dataframe(heat.style.background_gradient(axis=None), use_container_width=True)
        return
    except Exception:
        pass
    # 2) CSS fallback
    st.dataframe(_styler_heatmap_css(heat), use_container_width=True)
    st.caption("ℹ️ Heatmap SAFE (matplotlib non disponible).")


# =========================
# Matching (simple)
# =========================
def recover_from_source(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    id_col: str,
    name_col: str,
    team_col: str | None,
    source_tag: str,
    conf: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    filled = 0

    if id_col not in out.columns:
        out[id_col] = np.nan

    # normalize source columns
    s_id_col = _resolve_nhl_id_col(source_df) or "NHL_ID"
    s_name_col = _resolve_player_name_col(source_df) or "Player"

    if s_id_col not in source_df.columns or s_name_col not in source_df.columns:
        return out, {"filled": 0, "reason": "Source missing columns"}

    s = source_df.copy()
    s["_k_name"] = s[s_name_col].map(_normalize_player_name)
    s["_id"] = pd.to_numeric(s[s_id_col], errors="coerce")
    s = s.dropna(subset=["_id"])
    s = s.drop_duplicates(subset=["_k_name"], keep="first")

    out["_k_name"] = out[name_col].map(_normalize_player_name)
    cur = pd.to_numeric(out[id_col], errors="coerce")

    miss_mask = cur.isna()
    join = out.loc[miss_mask, ["_k_name"]].merge(s[["_k_name", "_id"]], on="_k_name", how="left")
    to_fill = join["_id"].notna()
    idxs = join.index[to_fill].tolist()

    if len(idxs) > 0:
        out.loc[out.index[miss_mask][to_fill.values], id_col] = join.loc[to_fill, "_id"].values
        filled = int(to_fill.sum())

    if "nhl_id_source" not in out.columns:
        out["nhl_id_source"] = ""
    if "nhl_id_confidence" not in out.columns:
        out["nhl_id_confidence"] = np.nan

    out.loc[out[id_col].notna(), "nhl_id_source"] = source_tag
    out.loc[out[id_col].notna(), "nhl_id_confidence"] = float(conf)

    out = out.drop(columns=["_k_name"], errors="ignore")
    return out, {"filled": filled}


# =========================
# NHL Search API — generator (chunked)
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


def generate_nhl_search_source_chunked(
    out_path: str,
    *,
    active_only: bool = True,
    limit: int = 1000,
    timeout_s: int = 20,
    max_pages: int = 20,
    culture: str = "en-us",
    q: str = "*",
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
    """
    ⚠️ L'API NHL search n'est pas officiellement documentée ici.
    On tente une pagination via `start=` (si ignoré, on détecte boucle via NHL_ID).
    """
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

            # fallback: si actif renvoie vide au premier call, retente active=False
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

            # garde-fou anti-boucle si start ignoré: si aucun nouvel ID, stop
            if new_count == 0:
                break

            # si on reçoit moins que la taille de page, probablement fin
            if len(items) < int(limit):
                break

            start += int(limit)
            time.sleep(0.05)

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df["NHL_ID"] = pd.to_numeric(df["NHL_ID"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["NHL_ID"])
            df = df.drop_duplicates(subset=["NHL_ID"], keep="first")

        errw = save_csv(df, out_path, safe_mode=False)
        if errw:
            return pd.DataFrame(), {"url": used_url, "pages": pages, "rows": int(len(df))}, errw

        return df, {"url": used_url, "pages": pages, "rows_saved": int(len(df)), "out_path": out_path}, None

    except Exception as e:
        return pd.DataFrame(), {"url": used_url, "pages": pages}, f"Erreur NHL Search API: {type(e).__name__}: {e}"


# =========================
# UI — render (with global catch)
# =========================
def render(*args, **kwargs):
    """
    IMPORTANT: on catch TOUTE exception pour éviter un écran noir.
    """
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

    # paths
    nhl_src = os.path.join(data_dir, "nhl_search_players.csv")
    hockey_players = os.path.join(data_dir, "hockey.players.csv")
    equipes_joueurs = os.path.join(data_dir, f"equipes_joueurs_{season}.csv")
    if not os.path.exists(equipes_joueurs):
        equipes_joueurs = os.path.join(data_dir, "equipes_joueurs_2025-2026.csv")

    st.subheader("🛠️ Admin — NHL_ID (SAFE)")

    # =====================================================
    # 🌐 Générer source NHL_ID (NHL Search API)
    # =====================================================
    st.markdown("### 🌐 Générer source NHL_ID (NHL Search API)")
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
    with c2:
        active_only = st.checkbox("Actifs seulement", value=True, key=WKEY + "gen_active")
    with c3:
        limit = st.number_input("Chunk limit", min_value=200, max_value=2000, value=1000, step=100, key=WKEY + "gen_limit")
    with c4:
        timeout_s = st.number_input("Timeout (s)", min_value=5, max_value=60, value=20, step=5, key=WKEY + "gen_timeout")

    gen = c1.button("🌐 Générer source NHL_ID", use_container_width=True, key=WKEY + "btn_gen")
    st.caption(f"Sortie: {nhl_src}")

    if gen:
        with st.spinner("Génération en cours…"):
            df_out, dbg, err = generate_nhl_search_source_chunked(
                nhl_src,
                active_only=bool(active_only),
                limit=int(limit),
                timeout_s=int(timeout_s),
                max_pages=20,
                culture="en-us",
                q="*",
            )
        if err:
            st.error(err)
            if dbg.get("url"):
                st.caption(f"URL: {dbg.get('url')}")
        else:
            st.success(f"✅ Généré: {dbg.get('rows_saved', 0)} joueurs (pages={dbg.get('pages', 0)}).")
            st.caption(f"URL: {dbg.get('url')}")
            if isinstance(df_out, pd.DataFrame) and not df_out.empty:
                st.dataframe(df_out.head(10), use_container_width=True)

    st.markdown("---")

    # =====================================================
    # 🔎 Vérification et audit
    # =====================================================
    st.markdown("### 🔎 Vérifier l'état des NHL_ID")
    players2 = st.selectbox(
        "Fichier cible",
        options=[equipes_joueurs, hockey_players],
        index=0,
        key=WKEY + "target_file",
    )

    # optional source selection
    source_opts = []
    for p in [nhl_src, hockey_players, equipes_joueurs]:
        if os.path.exists(p):
            source_opts.append(p)

    source_path = st.selectbox(
        "Source de récupération (optionnel)",
        options=source_opts if source_opts else [hockey_players],
        index=0,
        key=WKEY + "source_file",
    )

    source_df, err_s = load_csv(source_path)
    source_tag = os.path.basename(source_path)

    # safe controls
    conf = st.slider("Score de confiance appliqué aux IDs récupérés", 0.50, 0.99, 0.85, 0.01, key=WKEY + "conf")
    dry = st.checkbox("Dry-run (aucune écriture)", value=True, key=WKEY + "dry")
    override_safe = st.checkbox("Désactiver SAFE MODE (danger)", value=False, key=WKEY + "override_safe")

    # buttons
    if st.button("🔎 Vérifier", key=WKEY + "btn_verify"):
        df, err = load_csv(players2)
        if err:
            st.error(err)
        else:
            id_col = _resolve_nhl_id_col(df) or "NHL_ID"
            name_col = _resolve_player_name_col(df) or "Player"
            team_col = _resolve_team_col(df)

            if id_col not in df.columns or name_col not in df.columns:
                st.error("Colonnes requises manquantes (NHL_ID et/ou Player).")
            else:
                a = audit_nhl_ids(df, id_col)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total joueurs", a["total"])
                m2.metric("Avec ID (NHL_ID)", a["with_id"])
                m3.metric("Manquants", a["missing"])
                m4.metric("% manquants", f"{a['missing_pct']:.1f}%")

                st.info(f"Doublons: {a['dup_cnt']} (~{a['dup_pct']:.1f}% des IDs présents).")

                audit_df = build_audit_report(df, id_col, name_col, team_col)
                heat = confidence_heatmap(audit_df)

                st.markdown("#### 🧩 Heatmap de confiance (par équipe × bucket)")
                render_heatmap_safe(heat)

                out = io.StringIO()
                audit_df.to_csv(out, index=False)
                st.download_button(
                    "🧾 Télécharger rapport d'audit NHL_ID (CSV)",
                    data=out.getvalue().encode("utf-8"),
                    file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=WKEY + "dl_audit",
                )

    st.markdown("---")

    # =====================================================
    # 🔗 Associer NHL_ID
    # =====================================================
    st.markdown("### 🔗 Associer / récupérer NHL_ID")

    lock_dup_pct = st.slider("🛑 Seuil blocage duplication (%)", min_value=0.5, max_value=20.0, value=5.0, step=0.5, key=WKEY + "dup_lock")
    prod = str(os.environ.get("PMS_ENV", "")).strip().lower() in {"prod", "production"}
    allow_prod_override = st.checkbox("Déverrouiller en prod (admin)", value=False, disabled=(not prod), key=WKEY + "prod_unlock")

    if st.button("🔗 Associer NHL_ID", key=WKEY + "btn_assoc"):
        df, err = load_csv(players2)
        if err:
            st.error(err)
            return

        id_col = _resolve_nhl_id_col(df) or "NHL_ID"
        name_col = _resolve_player_name_col(df) or "Player"
        team_col = _resolve_team_col(df)

        if id_col not in df.columns or name_col not in df.columns:
            st.error("Colonnes requises manquantes (NHL_ID et/ou Player).")
            return

        if source_df is None or source_df.empty or err_s:
            st.error("Source indisponible / vide.")
            return

        if "nhl_id_source" not in df.columns:
            df["nhl_id_source"] = ""
        if "nhl_id_confidence" not in df.columns:
            df["nhl_id_confidence"] = np.nan

        df2, stats = recover_from_source(
            df,
            source_df,
            id_col=id_col,
            name_col=name_col,
            team_col=team_col,
            source_tag=source_tag,
            conf=conf,
        )

        a = audit_nhl_ids(df2, id_col)
        dup_pct = float(a["dup_pct"])

        st.success(f"✅ Matching terminé — +{int(stats.get('filled', 0))} IDs récupérés. Duplication: {dup_pct:.1f}%.")

        if prod and (dup_pct > lock_dup_pct) and not allow_prod_override:
            st.error(f"🔒 PROD LOCK: écriture bloquée — duplication {dup_pct:.1f}% > seuil {lock_dup_pct:.1f}%.")
            st.info("Corrige la source (mauvais match) ou coche 'Déverrouiller en prod (admin)'.")
            return

        if dry:
            st.info("Dry-run: aucune écriture.")
            return

        errw = save_csv(df2, players2, safe_mode=(not override_safe))
        if errw:
            st.error(errw)
            return

        st.success("💾 Sauvegarde OK.")

        audit_df = build_audit_report(df2, id_col, name_col, team_col)
        heat = confidence_heatmap(audit_df)
        st.markdown("#### 🧩 Heatmap de confiance (par équipe × bucket)")
        render_heatmap_safe(heat)

        out = io.StringIO()
        audit_df.to_csv(out, index=False)
        st.download_button(
            "🧾 Télécharger rapport d'audit NHL_ID (CSV)",
            data=out.getvalue().encode("utf-8"),
            file_name=f"audit_nhl_id_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=WKEY + "dl_audit_after",
        )


# Backward-compat alias (si ton app appelle _render_tools)
def _render_tools(*args, **kwargs):
    return render(*args, **kwargs)
