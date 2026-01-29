# tabs/admin.py — ULTRA CLEAN (stable) — Tools: PuckPedia Level + NHL_ID real API + Audit
from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd


# =====================================================
# Helpers
# =====================================================
def _get(ctx: Dict[str, Any], key: str, default=None):
    return ctx.get(key, default)


def _norm_player(s: str) -> str:
    s = str(s or "").strip()
    s = " ".join(s.split())
    return s


def _is_missing_id(v: Any) -> bool:
    s = str(v or "").strip()
    if not s or s.lower() == "nan":
        return True
    if s == "0":
        return True
    return False


def audit_nhl_ids(df: pd.DataFrame) -> Dict[str, Any]:
    if "NHL_ID" not in df.columns:
        return {"total": len(df), "filled": 0, "missing": len(df), "missing_pct": 100.0}

    missing_mask = df["NHL_ID"].apply(_is_missing_id)
    missing = int(missing_mask.sum())
    total = int(len(df))
    filled = total - missing
    pct = (missing / total * 100.0) if total else 0.0

    # duplicates (optional signal)
    ids = df.loc[~missing_mask, "NHL_ID"].astype(str).str.strip()
    dup_count = int(ids.duplicated().sum())

    return {
        "total": total,
        "filled": filled,
        "missing": missing,
        "missing_pct": pct,
        "duplicates": dup_count,
    }


def load_players(players_path: str) -> Tuple[pd.DataFrame, str | None]:
    players_path = str(players_path or "").strip()
    if not os.path.exists(players_path):
        return pd.DataFrame(), f"Players DB introuvable: {players_path}"
    try:
        df = pd.read_csv(players_path, low_memory=False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Lecture players DB échouée: {e}"


def save_players(df: pd.DataFrame, players_path: str) -> str | None:
    try:
        df.to_csv(players_path, index=False)
        return None
    except Exception as e:
        return f"Écriture players DB échouée: {e}"


# =====================================================
# PUCKPEDIA → LEVEL
# =====================================================
def sync_level(players_path: str, puck_path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "updated": 0}

    missing = []
    if not os.path.exists(players_path):
        missing.append(players_path)
    if not os.path.exists(puck_path):
        missing.append(puck_path)
    if missing:
        result["error"] = "Fichier introuvable"
        result["missing"] = missing
        return result

    try:
        pdb = pd.read_csv(players_path, low_memory=False)
        pk = pd.read_csv(puck_path, low_memory=False)
    except Exception as e:
        result["error"] = f"Lecture CSV échouée: {e}"
        return result

    # detect name columns
    name_pdb = next((c for c in ["Player", "Joueur", "Name"] if c in pdb.columns), None)
    name_pk = next((c for c in ["Skaters", "Player", "Name"] if c in pk.columns), None)

    if not name_pdb or not name_pk:
        result["error"] = "Colonne joueur introuvable"
        result["players_cols"] = list(pdb.columns)
        result["puck_cols"] = list(pk.columns)
        return result

    if "Level" not in pk.columns:
        result["error"] = "Colonne Level manquante dans PuckPedia"
        result["puck_cols"] = list(pk.columns)
        return result

    if "Level" not in pdb.columns:
        pdb["Level"] = ""

    mp: Dict[str, str] = {}
    for _, r in pk.iterrows():
        nm = _norm_player(r.get(name_pk))
        lv = str(r.get("Level") or "").strip().upper()
        if nm and lv in ("ELC", "STD"):
            mp[nm] = lv

    if not mp:
        result["error"] = "Aucun Level ELC/STD trouvé dans PuckPedia"
        return result

    updated = 0
    new_col = []
    for _, r in pdb.iterrows():
        nm = _norm_player(r.get(name_pdb))
        new_lv = mp.get(nm)
        old_lv = str(r.get("Level") or "").strip().upper()
        if new_lv and old_lv != new_lv:
            updated += 1
            new_col.append(new_lv)
        else:
            new_col.append(r.get("Level"))

    pdb["Level"] = new_col
    err = save_players(pdb, players_path)
    if err:
        result["error"] = err
        return result

    result["ok"] = True
    result["updated"] = updated
    return result


# =====================================================
# NHL_ID — REAL NHL FREE API (search.d3.nhle.com)
# =====================================================
def _nhl_search_player_id(name: str, session, timeout: int = 10) -> int | None:
    # NHL search endpoint (no key)
    # returns list of matches; take first result
    import requests

    q = requests.utils.quote(name)
    url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={q}"
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    data = r.json() or []
    if not data:
        return None
    pid = data[0].get("playerId")
    try:
        return int(pid)
    except Exception:
        return None


def sync_nhl_id(players_path: str, limit: int = 250, dry_run: bool = False) -> Dict[str, Any]:
    df, err = load_players(players_path)
    if err:
        return {"ok": False, "error": err}
    if df.empty:
        return {"ok": False, "error": "Players DB vide."}

    if "Player" not in df.columns:
        return {"ok": False, "error": "Colonne 'Player' introuvable dans hockey.players.csv", "players_cols": list(df.columns)}

    if "NHL_ID" not in df.columns:
        df["NHL_ID"] = ""

    missing_mask = df["NHL_ID"].apply(_is_missing_id)
    targets = df[missing_mask].head(int(limit))
    total = int(len(targets))
    if total == 0:
        a = audit_nhl_ids(df)
        return {"ok": True, "added": 0, "total": 0, "audit": a, "summary": "✅ Aucun NHL_ID manquant."}

    bar = st.progress(0)
    txt = st.empty()

    import requests
    session = requests.Session()

    added = 0
    for n, (idx, row) in enumerate(targets.iterrows(), start=1):
        name = _norm_player(row.get("Player"))
        if not name:
            continue

        # progress UI
        bar.progress(min(1.0, n / max(1, total)))
        txt.caption(f"Recherche NHL_ID: {n}/{total} — ajoutés: {added}")

        pid = None
        try:
            pid = _nhl_search_player_id(name, session=session, timeout=10)
        except Exception:
            pid = None

        if pid:
            added += 1
            if not dry_run:
                df.at[idx, "NHL_ID"] = pid

        # tiny sleep to be polite
        time.sleep(0.05)

    # finish UI
    bar.progress(1.0)
    txt.caption(f"Terminé ✅ — ajoutés: {added}/{total}" + (" (dry-run)" if dry_run else ""))

    if not dry_run:
        err2 = save_players(df, players_path)
        if err2:
            return {"ok": False, "error": err2}

    a = audit_nhl_ids(df)
    return {
        "ok": True,
        "added": added,
        "total": total,
        "audit": a,
        "summary": f"✅ Terminé — ajoutés: {added}/{total}" + (" (dry-run)" if dry_run else ""),
    }


# =====================================================
# UI
# =====================================================
def render(ctx: Dict[str, Any]):
    data_dir = str(_get(ctx, "DATA_DIR", "data"))
    is_admin = bool(_get(ctx, "is_admin", False))

    st.title("🛠️ Gestion Admin")

    if not is_admin:
        st.warning("Accès admin requis.")
        return

    st.radio("Section", ["Outils"], horizontal=True)

    st.subheader("🔧 Outils")

    # ---------- Sync PuckPedia -> Level
    with st.expander("🧾 Sync PuckPedia → Level (STD/ELC)", expanded=False):
        puck = st.text_input("Fichier PuckPedia", os.path.join(data_dir, "PuckPedia2025_26.csv"))
        players = st.text_input("Players DB", os.path.join(data_dir, "hockey.players.csv"))

        if st.button("Synchroniser Level"):
            res = sync_level(players, puck)
            if res.get("ok"):
                st.success(f"Levels mis à jour: {res.get('updated', 0)}")
            else:
                st.error(res.get("error", "Erreur inconnue"))
                if res.get("missing"):
                    st.write("Chemins manquants:", res["missing"])
                if res.get("players_cols"):
                    st.write("Players DB colonnes:", res["players_cols"])
                if res.get("puck_cols"):
                    st.write("PuckPedia colonnes:", res["puck_cols"])

    # ---------- NHL_ID missing + audit
    with st.expander("🆔 Sync NHL_ID manquants (avec progression + audit)", expanded=False):
        players2 = st.text_input("Players DB (NHL_ID)", os.path.join(data_dir, "hockey.players.csv"), key="nhl_players")
        limit = st.number_input("Max par run", 1, 2000, 1000, step=50)
        dry = st.checkbox("Dry-run (ne sauvegarde pas)", value=False)

        # Audit button
        if st.button("🔎 Vérifier l'état des NHL_ID"):
            df, err = load_players(players2)
            if err:
                st.error(err)
            else:
                a = audit_nhl_ids(df)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total joueurs", a["total"])
                c2.metric("Avec NHL_ID", a["filled"])
                c3.metric("Manquants", a["missing"])
                c4.metric("% manquants", f"{a['missing_pct']:.1f}%")
                if a.get("duplicates", 0):
                    st.warning(f"IDs dupliqués détectés: {a['duplicates']} (souvent normal si erreurs de match).")

                if a["missing"] > 0:
                    miss = df[df["NHL_ID"].apply(_is_missing_id)][["Player", "Team", "Position"] if "Team" in df.columns and "Position" in df.columns else ["Player"]].head(200)
                    st.caption("Aperçu (max 200) des joueurs sans NHL_ID :")
                    st.dataframe(miss, use_container_width=True)

        # Sync button
        if st.button("Associer NHL_ID"):
            res = sync_nhl_id(players2, int(limit), dry_run=bool(dry))
            if res.get("ok"):
                st.success(res.get("summary", "Terminé."))
                a = res.get("audit") or {}
                if a:
                    st.caption(f"État actuel — Total: {a.get('total')} | Avec NHL_ID: {a.get('filled')} | Manquants: {a.get('missing')} ({a.get('missing_pct', 0):.1f}%)")
            else:
                st.error(res.get("error", "Erreur inconnue"))
