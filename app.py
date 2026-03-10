from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# ============================================================
# PoolHockeyPMS — App (Pro Sidebar + Emoji PNG nav)
# ============================================================

APP_TITLE = "Pool GM"
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets")) / "previews"
CACHE_UI_DIR = DATA_DIR / "_ui_cache"
CACHE_UI_DIR.mkdir(parents=True, exist_ok=True)

POOL_TEAMS = [
    "Canadiens",
    "Whalers",
    "Nordiques",
    "Red_Wings",
    "Predateurs",
    "Cracheurs",
]

TEAM_LABEL = {
    "Canadiens": "Canadiens",
    "Whalers": "Whalers",
    "Nordiques": "Nordiques",
    "Red_Wings": "Red Wings",
    "Predateurs": "Prédateurs",
    "Cracheurs": "Cracheurs",
}

TEAM_LOGO_CANDIDATES = {
    "Canadiens": ["Canadiens_Logo.png", "CanadiensE_Logo.png"],
    "Whalers": ["Whalers_Logo.png", "WhalersE_Logo.png"],
    "Nordiques": ["Nordiques_Logo.png", "NordiquesE_Logo.png"],
    "Red_Wings": ["Red_Wings_Logo.png", "Red_WingsE_Logo.png"],
    "Predateurs": ["Predateurs_Logo.png", "PredateursE_Logo-2.png", "PredateursE_Logo.png"],
    "Cracheurs": ["Cracheurs_Logo.png", "CracheursE_Logo.png"],
}

EMOJI_FILES = {
    "home": "emoji_home.png",
    "gm": "emoji_gm.png",
    "joueurs": "emoji_joueur.png",
    "alignement": "emoji_alignement.png",
    "transactions": "emoji_transaction.png",
    "historique": "emoji_historique.png",
    "classement": "emoji_coupe.png",
}

@dataclass(frozen=True)
class NavItem:
    slug: str
    label: str

NAV_ORDER: list[NavItem] = [
    NavItem("home", "Home"),
    NavItem("gm", "GM"),
    NavItem("joueurs", "Joueurs"),
    NavItem("alignement", "Alignement"),
    NavItem("transactions", "Transactions"),
    NavItem("historique", "Historique"),
    NavItem("classement", "Classement"),
    NavItem("admin", "Admin"),
]

GM_LOGO = DATA_DIR / "gm_logo.png"
POOL_LOGO = DATA_DIR / "logo_pool.png"
RED_ACTIVE = "#ef4444"

def _transparent_copy(src: Path, thr: int = 245) -> Path:
    try:
        if not src.exists():
            return src
        dst = CACHE_UI_DIR / (src.stem + "_t.png")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst
        try:
            from PIL import Image
        except Exception:
            return src
        im = Image.open(src).convert("RGBA")
        px = im.getdata()
        new_px = []
        for r, g, b, a in px:
            if a == 0:
                new_px.append((r, g, b, 0))
                continue
            if r >= thr and g >= thr and b >= thr:
                new_px.append((r, g, b, 0))
            else:
                new_px.append((r, g, b, a))
        im.putdata(new_px)
        im.save(dst, "PNG")
        return dst
    except Exception:
        return src

@st.cache_data(show_spinner=False)
def _b64_png(path: Path) -> str:
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""

@st.cache_data(show_spinner=False, max_entries=256)
def _img_bytes_cached(p: str, mtime_size: tuple[float, int]) -> bytes:
    try:
        return Path(p).read_bytes()
    except Exception:
        return b""

def _img_bytes(path: Path) -> bytes:
    try:
        st_ = path.stat()
        sig = (st_.st_mtime, st_.st_size)
    except Exception:
        sig = (0.0, 0)
    return _img_bytes_cached(str(path), sig)

def _team_logo_path(team: str) -> Path:
    team = str(team or "").strip()
    cands = TEAM_LOGO_CANDIDATES.get(team, [])
    for fn in cands:
        p = ASSETS_DIR / fn
        if p.exists():
            return p
    p = ASSETS_DIR / f"{team}_Logo.png"
    return p if p.exists() else Path("")

def _emoji_path(slug: str) -> Optional[Path]:
    fn = EMOJI_FILES.get(slug)
    if not fn:
        return None
    p = ASSETS_DIR / fn
    return _transparent_copy(p) if p.exists() else None

def _apply_css():
    css = r"""
<style>
section[data-testid="stSidebar"] .stButton > button[kind="primary"]{
  background: linear-gradient(180deg, rgba(255,92,92,1) 0%, rgba(239,68,68,1) 55%, rgba(220,38,38,1) 100%) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  box-shadow: 0 18px 55px rgba(239,68,68,0.28), inset 0 1px 0 rgba(255,255,255,0.22), inset 0 -10px 22px rgba(0,0,0,0.18) !important;
}
/* ... le reste de votre CSS reste identique ... */
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

def _set_sidebar_mode(collapsed: bool):
    if collapsed:
        st.markdown("<style>:root { --pms-sb-w: 76px; --pms-emo: 44px; }</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>:root { --pms-sb-w: 320px; --pms-emo: 60px; }</style>", unsafe_allow_html=True)

def _sidebar_nav(owner_key: str, active_slug: str):
    collapsed = bool(st.session_state.get("pms_sidebar_collapsed", False))

    if not collapsed:
        st.sidebar.selectbox("Saison", options=["2025-2026", "2024-2025"], key="season")
        st.sidebar.selectbox("Équipe", options=POOL_TEAMS, key="owner")
        owner_key = str(st.session_state.get("owner", owner_key))

    _set_sidebar_mode(collapsed)

    gm_logo = DATA_DIR / "gm_logo.png"
    t_logo = _team_logo_path(owner_key)

    if not collapsed:
        c1, c2 = st.sidebar.columns(2)
        if gm_logo.exists(): c1.image(_img_bytes(gm_logo), width=110)
        if t_logo.exists(): c2.image(_img_bytes(t_logo), width=110)

    items = [it for it in NAV_ORDER if not (it.slug == "admin" and owner_key != "Whalers")]

    if collapsed:
        icon_map = {"home":"home","gm":"users","joueurs":"grid","alignement":"clipboard","transactions":"swap","historique":"clock","classement":"trophy","admin":"gear"}
        
        def _svg(name: str) -> str:
            if name == "home": return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7"/><path d="M9 22V12h6v10"/></svg>'
            if name == "users": return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>'
            if name == "grid": return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>'
            if name == "swap": return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 3h5v5"/><path d="M4 20l16-16"/></svg>'
            if name == "clipboard": return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="2" width="6" height="4"/><path d="M9 4H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V6"/></svg>'
            if name == "clock": return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>'
            if name == "trophy": return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 21h8"/><path d="M12 17v4"/><path d="M7 4h10v5a5 5 0 0 1-10 0V4z"/></svg>'
            if name == "gear":
                return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a7.8 7.8 0 0 0 .1-2l2-1.2-2-3.5-2.3.6a7.6 7.6 0 0 0-1.7-1l-.3-2.4h-4l-.3 2.4a7.6 7.6 0 0 0-1.7 1l-2.3-.6-2 3.5 2 1.2a7.8 7.8 0 0 0 0 2l-2 1.2 2 3.5 2.3-.6c.5.4 1.1.7 1.7 1l.3 2.4h4l.3-2.4c.6-.3 1.2-.6 1.7-1l2.3.6 2-3.5-2-1.2z"/></svg>'
            return '<svg viewBox="0 0 24 24"></svg>'

        html = ["<div class='pms-collapse-wrap'>"]
        for it in items:
            icon = icon_map.get(it.slug, "grid")
            cls = "pms-ic active" if it.slug == active_slug else "pms-ic"
            html.append(f"<a class='{cls}' href='?tab={it.slug}' title='{it.label}'>{_svg(icon)}</a>")
        html.append("</div>")
        st.sidebar.markdown("\n".join(html), unsafe_allow_html=True)
    else:
        for it in items:
            icon_p = _emoji_path(it.slug)
            is_active = (it.slug == active_slug)
            c1, c2 = st.sidebar.columns([1.1, 3.4], gap="small")
            with c1:
                if icon_p and icon_p.exists(): st.image(_img_bytes(icon_p), width=82)
            with c2:
                if st.button(it.label, key=f"nav_{it.slug}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state["active_tab"] = it.slug
                    st.rerun()

    if st.sidebar.button("◀" if not collapsed else "▶", key="pms_collapse_btn"):
        st.session_state["pms_sidebar_collapsed"] = not collapsed
        st.rerun()

# ... Reste des fonctions de rendu identiques ...

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏒", layout="wide")
    st.session_state.setdefault("owner", "Canadiens")
    st.session_state.setdefault("active_tab", "home")

    _apply_css()
    owner = st.session_state.get("owner")
    active = st.session_state.get("active_tab")
    
    _sidebar_nav(owner, active)

    if st.session_state.active_tab == "home":
        # Rendu simple de la home pour l'exemple
        st.title("Bienvenue")
    else:
        st.write(f"Page active : {st.session_state.active_tab}")

if __name__ == "__main__":
    main()
