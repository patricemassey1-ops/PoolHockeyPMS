# NOTE: copied from your players_db.py (admin UI module)
from __future__ import annotations
import os, json
import pandas as pd
import streamlit as st


def nhl_cache_path_default(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "nhl_country_cache.json")

def reset_nhl_cache(cache_path: str) -> None:
    if os.path.exists(cache_path):
        os.remove(cache_path)

def reset_failed_only(cache_path: str) -> None:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f) or {}
    cache2 = {k:v for k,v in (cache or {}).items() if isinstance(v, dict) and v.get("ok") is True}
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache2, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)



def checkpoint_path_default(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "nhl_country_checkpoint.json")

def read_checkpoint(data_dir: str) -> dict:
    p = checkpoint_path_default(data_dir)
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def reset_progress(data_dir: str) -> None:
    p = checkpoint_path_default(data_dir)
    if os.path.exists(p):
        os.remove(p)
def lock_on():
    st.session_state["pdb_lock"] = True

def lock_off():
    st.session_state["pdb_lock"] = False

def is_locked() -> bool:
    return bool(st.session_state.get("pdb_lock", False))

def render_players_db_admin(*, pdb_path: str, data_dir: str, season_lbl=None, update_fn=None):
    """UI renderer for Players DB (standalone module)."""
    cache_path = nhl_cache_path_default(data_dir)

    # update function must be provided by app.py
    if update_fn is None:
        update_fn = globals().get('update_players_db')

    if not st.session_state.get("_pdb_css_injected"):
        st.session_state["_pdb_css_injected"]=True
        st.markdown("""<style>
    .pdb-sticky {position:sticky; top:0.5rem; z-index:999; padding:6px 10px; border-radius:999px;
                font-weight:700; font-size:0.85rem; margin-bottom:8px; display:inline-block;}
    .pdb-playerid{background:#1f4fd8;color:white;}
    .pdb-country{background:#1f7a5a;color:white;}
    .pdb-idle{background:#3b3f46;color:#eaeaea;}
    </style>""", unsafe_allow_html=True)

    # Sticky badge reads checkpoint first (persistent across reruns)
    ck = read_checkpoint(data_dir)
    last = st.session_state.get("pdb_last", {}) if isinstance(st.session_state.get("pdb_last", {}), dict) else {}
    phase = str(ck.get("phase") or last.get("phase") or "").strip()
    cursor = int(ck.get("cursor") or last.get("index") or 0)
    total = int(last.get("total", 0) or 0)
    cls = "pdb-idle"
    if phase == "playerId":
        cls = "pdb-playerid"
    elif phase == "Country":
        cls = "pdb-country"
    reste = max((total - cursor) if total else 0, 0)
    total_txt = str(total) if total else "…"
    reste_txt = f" (reste: {reste})" if total else ""
    st.markdown(
        f"<div class='pdb-sticky {cls}'>Dernier : {phase or '—'} {cursor}/{total_txt}{reste_txt}</div>",
        unsafe_allow_html=True
    )


    opt1, opt2, opt3, opt4 = st.columns([1.2, 1.1, 1.2, 1.5], vertical_alignment="center")
    with opt1:
        ck0 = read_checkpoint(data_dir)
        roster_only = st.checkbox("⚡ Roster actif seulement", value=bool(ck0.get("roster_only", False)), key="pdb_roster_only")
    with opt2:
        show_details = st.checkbox("Afficher détails", value=False, key="pdb_show_details")
    with opt3:
        st.caption("🔒 LOCK ON" if is_locked() else "🔓 LOCK OFF")
    with opt4:
        cR1, cR2, cR3 = st.columns([1,1,1])
        with cR1:
            if st.button("🧹 Reset cache", use_container_width=True):
                try:
                    reset_nhl_cache(cache_path)
                    st.success("Cache NHL supprimé.")
                except Exception as e:
                    st.error(str(e))
        with cR2:
            if st.button("🧹 Reset progress", use_container_width=True):
                try:
                    reset_progress(data_dir)
                    st.session_state["pdb_last"] = {"phase":"—","index":0,"total":0}
                    st.success("Progress reset (checkpoint supprimé).")
                except Exception as e:
                    st.error(str(e))

        with cR3:
            if st.button("🧽 Reset failed only", use_container_width=True):
                try:
                    reset_failed_only(cache_path)
                    st.success("Échecs supprimés du cache.")
                except Exception as e:
                    st.error(str(e))

    prog = st.progress(0.0)
    status = st.empty()

    def _cb(done: int, total: int, phase: str):
        total = max(int(total or 0), 1)
        done = int(done or 0)
        prog.progress(min(1.0, max(0.0, done/total)))
        status.caption(f"{phase}: {done}/{total}")
        st.session_state["pdb_last"] = {"phase": phase, "index": done, "total": total}

    if st.button("⬆️ Mettre à jour Players DB", use_container_width=True):
        try:
            lock_on()
            if not callable(update_fn):
                raise NameError("update_players_db is not provided (update_fn)")
            _, stats = update_fn(
                path=pdb_path,
                season_lbl=season_lbl,
                fill_country=True,
                resume_only=True,
                roster_only=roster_only,
                save_every=500,
                cache_path=cache_path,
                progress_cb=_cb,
            )
            st.success("✅ Terminé.")
            if show_details:
                st.json(stats)
        finally:
            lock_off()

    if st.button("▶️ Resume Country fill", use_container_width=True):
        try:
            lock_on()
            if not callable(update_fn):
                raise NameError("update_players_db is not provided (update_fn)")
            _, stats = update_fn(
                path=pdb_path,
                season_lbl=season_lbl,
                fill_country=True,
                resume_only=True,
                roster_only=roster_only,
                save_every=500,
                cache_path=cache_path,
                progress_cb=_cb,
            )
            st.success("✅ Terminé (resume).")
            if show_details:
                st.json(stats)
        finally:
            lock_off()
