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
# Notes:
# - st.set_page_config() MUST be called exactly once and first -> here.
# - No other module (tabs/*) should call set_page_config.
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

# Team logo filenames (your repo has multiple variants; we pick the "best" if it exists)
TEAM_LOGO_CANDIDATES = {
    "Canadiens": ["Canadiens_Logo.png", "CanadiensE_Logo.png"],
    "Whalers": ["Whalers_Logo.png", "WhalersE_Logo.png"],
    "Nordiques": ["Nordiques_Logo.png", "NordiquesE_Logo.png"],
    "Red_Wings": ["Red_Wings_Logo.png", "Red_WingsE_Logo.png"],
    "Predateurs": ["Predateurs_Logo.png", "PredateursE_Logo-2.png", "PredateursE_Logo.png"],
    "Cracheurs": ["Cracheurs_Logo.png", "CracheursE_Logo.png"],
}

# Emoji PNG in assets/previews (provided by you)
EMOJI_FILES = {
    "home": "emoji_home.png",
    "gm": "emoji_gm.png",
    "joueurs": "emoji_joueur.png",
    "alignement": "emoji_alignement.png",
    "transactions": "emoji_transaction.png",
    "historique": "emoji_historique.png",
    "classement": "emoji_coupe.png",
}

# =========================
# NAV order (single page)
# =========================
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

# Apple-like red (close to your screenshot)
RED_ACTIVE = "#ef4444"


# ----------------------------
# Transparent background helper
# ----------------------------
def _transparent_copy(src: Path, thr: int = 245) -> Path:
    """
    Creates a cached RGBA version removing near-white background.
    Returns original if Pillow is unavailable or file missing.
    """
    try:
        if not src.exists():
            return src
        dst = CACHE_UI_DIR / (src.stem + "_t.png")
        # Cache: regenerate only if missing or older
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst

        try:
            from PIL import Image  # type: ignore
        except Exception:
            return src

        im = Image.open(src).convert("RGBA")
        px = im.getdata()
        new_px = []
        for r, g, b, a in px:
            if a == 0:
                new_px.append((r, g, b, 0))
                continue
            # remove near-white / paper-white
            if r >= thr and g >= thr and b >= thr:
                new_px.append((r, g, b, 0))
            else:
                new_px.append((r, g, b, a))
        im.putdata(new_px)
        im.save(dst, "PNG")
        return dst
    except Exception:
        return src


def _transparent_copy_edge(src: Path, thr: int = 245) -> Path:
    """
    Creates a cached RGBA version removing a near-white *edge-connected* background.
    This preserves internal whites (e.g., logo highlights) while removing the white square behind it.
    Returns original if Pillow is unavailable or file missing.
    """
    try:
        if not src.exists():
            return src
        dst = CACHE_UI_DIR / (src.stem + "_edge_t.png")
        # Cache: regenerate only if missing or older
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            return dst

        try:
            from PIL import Image  # type: ignore
        except Exception:
            return src

        im = Image.open(src).convert("RGBA")
        w, h = im.size
        px = im.load()
        if px is None:
            return src

        # Helper: near-white pixel?
        def is_bg(x: int, y: int) -> bool:
            r, g, b, a = px[x, y]
            return a > 0 and r >= thr and g >= thr and b >= thr

        # BFS flood fill from edges for background pixels
        from collections import deque

        visited = bytearray(w * h)
        q = deque()

        def push(x: int, y: int):
            idx = y * w + x
            if visited[idx]:
                return
            if not is_bg(x, y):
                return
            visited[idx] = 1
            q.append((x, y))

        # seed with corner pixels (preserves internal/edge whites better)
        k = 6  # corner box size
        for y in range(min(k, h)):
            for x in range(min(k, w)):
                push(x, y)  # top-left
                push(w - 1 - x, y)  # top-right
                push(x, h - 1 - y)  # bottom-left
                push(w - 1 - x, h - 1 - y)  # bottom-right

        while q:
            x, y = q.popleft()
            # Make transparent
            r, g, b, a = px[x, y]
            px[x, y] = (r, g, b, 0)
            # neighbors
            if x > 0:
                push(x - 1, y)
            if x < w - 1:
                push(x + 1, y)
            if y > 0:
                push(x, y - 1)
            if y < h - 1:
                push(x, y + 1)

        im.save(dst, "PNG")
        return dst
    except Exception:
        return src


@st.cache_data(show_spinner=False)
def _file_sig(path: Path) -> tuple[int, int]:
    try:
        st_ = path.stat()
        return (int(st_.st_mtime_ns), int(st_.st_size))
    except Exception:
        return (0, 0)

@st.cache_data(show_spinner=False, max_entries=256)
def _b64_png_cached(p: str, sig: tuple[int, int]) -> str:
    try:
        b = Path(p).read_bytes()
        return base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

def _b64_png(path: Path) -> str:
    return _b64_png_cached

@st.cache_data(show_spinner=False, max_entries=256)
def _b64_image_resized_cached(p: str, sig: tuple[int, int], max_w: int) -> tuple[str, str]:
    """Return (mime, b64) for a resized WEBP (fast to transfer)."""
    try:
        from PIL import Image  # type: ignore
        src = Path(p)
        if not src.exists():
            return ("", "")
        im = Image.open(src).convert("RGBA")
        if max_w and im.width > max_w:
            h = max(1, int(im.height * (max_w / float(im.width))))
            im = im.resize((max_w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=82, method=6)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return ("image/webp", b64)
    except Exception:
        # Fallback to raw bytes
        try:
            b = Path(p).read_bytes()
            return ("image/png", base64.b64encode(b).decode("utf-8"))
        except Exception:
            return ("", "")

def _b64_image(path: Path, max_w: int = 0) -> tuple[str, str]:
    return _b64_image_resized_cached(str(path), _file_sig(path), int(max_w or 0))

(str(path), _file_sig(path))



def _pick_existing(base_dir: Path, candidates: list[str]) -> Optional[Path]:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    return None


def _team_logo_path(team: str) -> Path:
    team = str(team or "").strip()
    cands = TEAM_LOGO_CANDIDATES.get(team, [])
    for fn in cands:
        p = ASSETS_DIR / fn
        if p.exists():
            return _transparent_copy_edge(p)
    p = ASSETS_DIR / f"{team}_Logo.png"
    return _transparent_copy_edge(p) if p.exists() else Path("")
def _emoji_path(slug: str) -> Optional[Path]:
    fn = EMOJI_FILES.get(slug)
    if not fn:
        return None
    p = ASSETS_DIR / fn
    return _transparent_copy(p) if p.exists() else None


def _gm_logo_path() -> Path:
    p = DATA_DIR / "gm_logo.png"
    return p if p.exists() else Path("")

def _apply_css():
    """Inject global CSS (SAFE: all CSS stays inside strings)."""
    css = r"""
<style>

/* === Apple WOW++++++++++++ Active Pill (expanded sidebar) === */
section[data-testid="stSidebar"] .stButton > button[kind="primary"]{
  background: linear-gradient(180deg, rgba(255,92,92,1) 0%, rgba(239,68,68,1) 55%, rgba(220,38,38,1) 100%) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  box-shadow:
    0 18px 55px rgba(239,68,68,0.28),
    inset 0 1px 0 rgba(255,255,255,0.22),
    inset 0 -10px 22px rgba(0,0,0,0.18) !important;
  position: relative !important;
  overflow: hidden !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]::before{
  content:"";
  position:absolute;
  left: 10px;
  right: 10px;
  top: 7px;
  height: 55%;
  border-radius: 14px;
  background: radial-gradient(ellipse at top, rgba(255,255,255,0.26) 0%, rgba(255,255,255,0.10) 42%, rgba(255,255,255,0.00) 70%);
  pointer-events:none;
}
@keyframes pmsPillSheen {
  0% { transform: translateX(-120%) rotate(12deg); opacity: 0.0; }
  18% { opacity: 0.20; }
  100% { transform: translateX(220%) rotate(12deg); opacity: 0.0; }
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]::after{
  content:"";
  position:absolute;
  inset:-40% -60%;
  background: linear-gradient(90deg,
    rgba(255,255,255,0.00) 0%,
    rgba(255,255,255,0.10) 40%,
    rgba(255,255,255,0.22) 50%,
    rgba(255,255,255,0.10) 60%,
    rgba(255,255,255,0.00) 100%);
  transform: translateX(-140%) rotate(12deg);
  opacity: 0;
  pointer-events:none;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover::after{
  animation: pmsPillSheen 780ms ease-out;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:active{
  transform: translateY(0px) scale(0.985) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton > button[kind="primary"] span{
  color: rgba(255,255,255,0.98) !important;
  text-shadow: 0 1px 0 rgba(0,0,0,0.22);
}


/* === Apple WOW+++++++++++ Dock (collapsed) === */
.pms-nav.pms-collapsed { padding-top: 6px; }
.pms-dock-item{ position: relative; display:flex; justify-content:center; align-items:center; margin: 10px 0; }
.pms-dock-ico{
  width: 58px; height: 58px; border-radius: 18px; object-fit: contain;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 18px 55px rgba(0,0,0,0.28);
}
.pms-dock-item.active .pms-dock-ico{
  transform: scale(1.05);
  box-shadow: 0 18px 46px rgba(239,68,68,0.22), 0 0 0 2px rgba(239,68,68,0.30);
}
.pms-dock-item:hover .pms-dock-ico{ transform: translateY(-1px) scale(1.03); }

/* Invisible overlay button in dock */
.pms-dock-item .stButton{ position:absolute; inset:0; margin:0 !important; display:flex; justify-content:center; align-items:center; }
.pms-dock-item .stButton > button{
  width: 58px !important; height: 58px !important; border-radius: 18px !important;
  opacity: 0 !important; padding:0 !important; margin:0 !important;
}

/* Team logo next to dropdown (rounded card + transparent logo) */
.pms-team-hero{ display:flex; align-items:center; justify-content:center; min-height: 92px; margin-top: 18px; }
.pms-team-card{
  border-radius: 22px;
  padding: 14px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 18px 60px rgba(0,0,0,0.30);
  display: inline-block;
}
.pms-team-card img{
  width: 96px; height: 96px;
  border-radius: 16px;
  object-fit: contain;
  display: block;
}

/* Pool logo centered above Home */
.pms-pool-wrap{ display:flex; justify-content:center; margin: 6px 0 14px 0; }
.pms-pool-card{
  border-radius: 24px;
  padding: 16px 18px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 18px 60px rgba(0,0,0,0.28);
  display:inline-flex;
  align-items:center;
  justify-content:center;
}
.pms-pool-card img{ width: min(704px, 100%);  height: auto; display:block; object-fit: contain; }

/* Brand row (GM + selected team) */
.pms-brand-row{ display:flex; gap:12px; align-items:center; justify-content:flex-start; margin: 14px 0 6px 0; }
.pms-brand-row .pms-chip{ padding: 8px; }
.pms-brand-row img{ display:block; object-fit:contain; border-radius: 16px; }
.pms-brand-row img.pms-gm{ width: 96px; height: 96px; }
.pms-brand-row img.pms-team{ width: 96px; height: 96px; }

/* Center page icon (Home/GM/Joueurs/...) */
.pms-page-header{ display:flex; align-items:center; gap:10px; margin: 6px 0 10px 0; }
.pms-page-ico{
  width: clamp(260px, 20vw, 360px);
  height: clamp(260px, 20vw, 360px);
  border-radius: 0px;
  background: transparent;
  border: none;
  box-shadow: none;
  object-fit: contain;
  display:block;
}
.pms-page-title{ font-size: 3.0rem; font-weight: 800; line-height: 1.05; margin:0; padding:0; }

/* Hide tiny loader dots/containers sometimes rendered by st.image */
/* Hide Streamlit image skeleton/loader artifacts (keep real images visible) */
div[data-testid="stImage"] [data-testid="stSkeleton"] { display:none !important; }
div[data-testid="stImage"] [role="progressbar"] { display:none !important; }
div[data-testid="stImage"] svg { display:none !important; }
div[data-testid="stSpinner"], .stSpinner { display:none !important; }



/* === Apple WOW+++++++ === */

/* hide Streamlit image loaders (little circles) */
section[data-testid="stSidebar"] [role="progressbar"],
section.main [role="progressbar"] { display:none !important; }

/* global polish */
* { -webkit-font-smoothing: antialiased; }
section[data-testid="stSidebar"] > div {
  backdrop-filter: blur(18px) saturate(125%) !important;
  -webkit-backdrop-filter: blur(18px) saturate(125%) !important;
}

/* premium glass cards */
.pms-card, .pms-chip, .pms-item, section[data-testid="stSidebar"] .stButton > button {
  border: 1px solid rgba(255,255,255,.10) !important;
  box-shadow: 0 18px 55px rgba(0,0,0,.26) !important;
}

/* Expanded sidebar nav: bigger icons + perfect vertical alignment */
section[data-testid="stSidebar"] .pms-nav img.pms-emoji{
  width: 84px !important;
  height: 84px !important;
  border-radius: 18px !important;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  padding: 2px;
}
section[data-testid="stSidebar"] .pms-nav div[data-testid="stHorizontalBlock"]{
  align-items:center !important;
}
section[data-testid="stSidebar"] .pms-nav div[data-testid="stHorizontalBlock"] > div{
  display:flex;
  align-items:center;
}
section[data-testid="stSidebar"] .pms-nav div[data-testid="stHorizontalBlock"] > div:first-child{
  justify-content:center;
}
section[data-testid="stSidebar"] .pms-nav .stButton > button{
  min-height: 66px !important;
}


/* icon sizes */

/* Sidebar nav rows: align emoji icons with their buttons */
.pms-nav div[data-testid="stHorizontalBlock"] { align-items: center !important; }
.pms-nav div[data-testid="stHorizontalBlock"] > div { align-items: center !important; }
.pms-nav div[data-testid="stColumn"], .pms-nav div[data-testid="column"] { display:flex !important; align-items:center !important; }
.pms-nav div[data-testid="stColumn"] > div, .pms-nav div[data-testid="column"] > div { width:100% !important; }
section[data-testid="stSidebar"] .stButton > button { min-height: var(--pms-emo, 60px) !important; }

:root{
  --pms-emo: 72px;
  --pms-emo-c: 52px;
  --pms-gm: 120px;
}

/* keep ratio for all icons */
.pms-item img, .pms-brand img, .pms-team-badge img { object-fit: contain !important; }

/* hover + active */
section[data-testid="stSidebar"] .stButton > button:hover { transform: translateY(-1px); }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  box-shadow: 0 14px 38px rgba(239,68,68,.22) !important;
}

/* collapsed sidebar: hide any leftover labels */
.pms-collapsed label, .pms-collapsed .stSelectbox { display:none !important; }

/* Layout: custom sidebar widths */
section[data-testid="stSidebar"] {
  width: var(--pms-sb-w, 320px) !important;
  min-width: var(--pms-sb-w, 320px) !important;
  max-width: var(--pms-sb-w, 320px) !important;
}
/* Hide Streamlit menu footer spacing a bit */
section.main .block-container { padding-top: 2.2rem !important; }

/* Apple glass sidebar */
section[data-testid="stSidebar"] > div { backdrop-filter: blur(16px) !important; }

/* "Chip" glass frame for logos */
.pms-chip {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  box-shadow: 0 18px 40px rgba(0,0,0,0.18);
  padding: 8px;
}
.pms-chip img { display:block; border-radius: 14px; }

/* NAV */
.pms-nav { margin-top: 12px; display:flex; flex-direction:column; gap: 10px; }
.pms-item {
  position: relative;
  display:flex;
  align-items:center;
  gap: 12px;
  border-radius: 18px;
  padding: 12px 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.05);
  box-shadow: 0 10px 22px rgba(0,0,0,0.12);
  transition: transform 120ms ease, box-shadow 150ms ease, border-color 150ms ease, background 150ms ease;
}
.pms-item:hover {
  transform: translateY(-1px);
  border-color: rgba(239,68,68,0.55);
  background: rgba(255,255,255,0.08);
  box-shadow: 0 14px 28px rgba(0,0,0,0.18);
}
.pms-item.active {
  background: rgba(239,68,68,0.95);
  border-color: rgba(220,38,38,1);
  box-shadow: 0 16px 34px rgba(239,68,68,0.22);
}
.pms-item.active::before {
  content:"";
  position:absolute;
  left:-7px; top:10px; bottom:10px;
  width:3px; border-radius:6px;
  background: rgba(239,68,68,1);
  box-shadow: 0 10px 24px rgba(239,68,68,0.45);
}

/* Emojis/logos never stretch */
.pms-emoji, .pms-emoji-c, .pms-teamlogo, .pms-gmlogo {
  object-fit: contain !important;
  aspect-ratio: 1/1;
}

/* icon sizes */
.pms-item img.pms-emoji { width: var(--pms-emo, 60px); height: var(--pms-emo, 60px); border-radius: 16px; }
.pms-item img.pms-teamlogo { width: var(--pms-emo, 60px); height: var(--pms-emo, 60px); border-radius: 16px; }
.pms-brand img.pms-gmlogo { width: var(--pms-gm, 80px); height: var(--pms-gm, 80px); border-radius: 18px; }

/* Collapsed mode: show icons centered */
.pms-collapsed .pms-item { justify-content:center; padding: 10px 0; }
.pms-collapsed .pms-item .lbl { display:none; }
.pms-collapsed .pms-item img.pms-emoji,
.pms-collapsed .pms-item img.pms-teamlogo { width: var(--pms-emo-c,44px); height: var(--pms-emo-c,44px); border-radius: 14px; }

/* Primary buttons red (match selection) */
.stButton > button[kind="primary"] {
  background: rgba(239,68,68,1) !important;
  border-color: rgba(220,38,38,1) !important;
  box-shadow: 0 18px 40px rgba(239,68,68,0.18) !important;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.03); }

/* Micro press */
section[data-testid="stSidebar"] .pms-item:active,
section[data-testid="stSidebar"] div.stButton > button:active {
  transform: translateY(0px) scale(0.98) !important;
}

/* Snap page transition */
@keyframes pmsPageIn { from { opacity:0; transform:translateY(6px);} to { opacity:1; transform:translateY(0px);} }
section.main .block-container { animation: pmsPageIn 240ms ease-out; }

/* micro iOS press for all sidebar buttons */
section[data-testid="stSidebar"] .stButton > button{
  transition: transform 120ms ease, box-shadow 180ms ease, background 180ms ease;
}
section[data-testid="stSidebar"] .stButton > button:active{
  transform: scale(0.985);
}

</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def _set_sidebar_mode(collapsed: bool):
    # In Streamlit, easiest is to set CSS vars on :root
    if collapsed:
        st.markdown(
            """
<style>
:root { --pms-sb-w: 76px; --pms-ico: 54px; --pms-ico-c: 44px; --pms-gm: 52px; --pms-emo: 44px; --pms-emo-c: 40px; }
section[data-testid="stSidebar"] { padding-left: 4px !important; padding-right: 4px !important; }
</style>
""",
            unsafe_allow_html=True,
        )
        st.markdown("<style>section[data-testid='stSidebar']{}</style>", unsafe_allow_html=True)
        # Also add a class hook
        st.markdown("<style>body{}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
<style>
:root { --pms-sb-w: 320px; --pms-ico: 54px; --pms-ico-c: 44px; --pms-gm: 80px; --pms-emo: 60px; --pms-emo-c: 44px; }
</style>
""",
            unsafe_allow_html=True,
        )




def _sidebar_nav(owner_key: str, active_slug: str):
    collapsed = bool(st.session_state.get("pms_sidebar_collapsed", False))

    # Expanded: season + team controls
    if not collapsed:
        st.sidebar.selectbox(
            "Saison",
            options=[st.session_state.get("season", "2025-2026"), "2024-2025", "2023-2024"],
            key="season",
        )
        st.sidebar.selectbox(
            "Équipe",
            options=POOL_TEAMS,
            index=POOL_TEAMS.index(st.session_state.get("owner", owner_key)) if st.session_state.get("owner", owner_key) in POOL_TEAMS else 0,
            key="owner",
        )
        owner_key = str(st.session_state.get("owner", owner_key))

    _set_sidebar_mode(collapsed)

    # Brand: GM logo + selected team logo (side-by-side in expanded sidebar)
    gm_logo = _gm_logo_path()
    t_logo = _team_logo_path(owner_key)

    if (not collapsed) and gm_logo and gm_logo.exists():
        b64g = _b64_png(gm_logo)
        b64t = _b64_png(t_logo) if (t_logo and t_logo.exists() and t_logo.is_file()) else ""
        team_html = (
            f"<div class='pms-chip'><img class='pms-team' src='data:image/png;base64,{b64t}' alt='team'/></div>"
            if b64t else ""
        )
        st.sidebar.markdown(
            f"""<div class='pms-brand-row'>
  <div class='pms-chip'><img class='pms-gm' src='data:image/png;base64,{b64g}' alt='gm'/></div>
  {team_html}
</div>""",
            unsafe_allow_html=True,
        )
    else:
        # Fallback (collapsed mode or missing assets): stack like before
        if gm_logo and gm_logo.exists():
            st.sidebar.markdown("<div class='pms-brand pms-chip'>", unsafe_allow_html=True)
            st.sidebar.image(str(gm_logo), width=(56 if collapsed else 110))
            st.sidebar.markdown("</div>", unsafe_allow_html=True)
            # Intentionally no text labels under the brand images

        if t_logo and t_logo.exists() and t_logo.is_file():
            st.sidebar.markdown("<div class='pms-chip' style='margin-top:10px'>", unsafe_allow_html=True)
            st.sidebar.image(str(t_logo), width=(44 if collapsed else 56))
            st.sidebar.markdown("</div>", unsafe_allow_html=True)
            # Intentionally no text labels under the brand images


    # Items
    items = []
    for it in NAV_ORDER:
        if it.slug == "admin" and owner_key != "Whalers":
            continue
        items.append(it)

    st.sidebar.markdown("<div class='pms-nav %s'>" % ("pms-collapsed" if collapsed else ""), unsafe_allow_html=True)

    for it in items:
        icon_p = _emoji_path(it.slug)  # png emoji (no stretch)
        is_active = (it.slug == active_slug)
        # Collapsed: iOS dock item (image + invisible overlay button)
        if collapsed:
            cls = "pms-dock-item active" if is_active else "pms-dock-item"
            st.sidebar.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)

            if icon_p and icon_p.exists():
                b64 = _b64_png(icon_p)
                st.sidebar.markdown(
                    f"<img class='pms-dock-ico' src='data:image/png;base64,{b64}' alt='{it.label}'/>",
                    unsafe_allow_html=True,
                )

            # Invisible overlay button (captures click, no ugly box)
            if st.sidebar.button(" ", key=f"nav_{it.slug}", help=it.label, type="primary" if is_active else "secondary"):
                st.session_state["active_tab"] = it.slug
                st.session_state["_pms_qp_applied"] = True
                _set_query_tab(it.slug)
                st.rerun()

            st.sidebar.markdown("</div>", unsafe_allow_html=True)
        else:
            c1, c2 = st.sidebar.columns([1.6, 4.0], gap="small")
            with c1:
                if icon_p and icon_p.exists():
                    b64i = _b64_png(icon_p)
                    st.markdown(f"<img class='pms-emoji' src='data:image/png;base64,{b64i}' alt='{it.label}' />", unsafe_allow_html=True)
            with c2:
                if st.button(it.label, key=f"nav_{it.slug}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state["active_tab"] = it.slug
                    st.session_state["_pms_qp_applied"] = True
                    _set_query_tab(it.slug)
                    st.rerun()

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # Collapse toggle
    if st.sidebar.button("◀" if not collapsed else "▶", key="pms_collapse_btn"):
        st.session_state["pms_sidebar_collapsed"] = not collapsed

    # Theme toggle (sun above)
    st.sidebar.markdown("<div style='margin-top:10px'>☀️</div>", unsafe_allow_html=True)
    light = bool(st.session_state.get("pms_light_mode", False))
    st.sidebar.toggle("Clair", value=light, key="pms_light_toggle")
    st.session_state["pms_light_mode"] = bool(st.session_state.get("pms_light_toggle", False))

def _apply_theme_mode():
    # Light mode quick polish without breaking your global theme:
    # We switch background + default text a bit. Keeps your existing dark theme too.
    light = bool(st.session_state.get("pms_light_mode", False))
    if not light:
        return
    st.markdown(
        f"""
<style>
/* Light mode look */
html, body, [data-testid="stAppViewContainer"] {{
  background: #f6f7fb !important;
}}
section[data-testid="stSidebar"] {{
  background: #ffffff !important;
  border-right: 1px solid rgba(0,0,0,.06);
}}
/* Card backgrounds */
div[data-testid="stVerticalBlock"] > div {{
  background: transparent;
}}
/* Make our nav readable */
.pms-item {{
  background: rgba(0,0,0,.03) !important;
  border-color: rgba(0,0,0,.06) !important;
  color: rgba(0,0,0,.78) !important;
}}
.pms-item:hover {{
  background: rgba(0,0,0,.05) !important;
}}
.pms-item.active {{
  color: white !important;
}}
.pms-brand .t, .pms-team .tt {{
  color: rgba(0,0,0,.80) !important;
}}
/* Title color */
h1, h2, h3 {{
  color: rgba(0,0,0,.86) !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _get_query_tab() -> Optional[str]:
    # Supports both new & old Streamlit query param APIs
    try:
        q = st.query_params  # type: ignore
        if "tab" in q:
            v = q["tab"]
            if isinstance(v, list):
                return str(v[0])
            return str(v)
    except Exception:
        pass
    try:
        q = st.experimental_get_query_params()
        v = (q.get("tab") or [None])[0]
        return str(v) if v else None
    except Exception:
        return None


def _set_query_tab(slug: str) -> None:
    try:
        st.query_params.update({"tab": slug})  # type: ignore
    except Exception:
        try:
            st.experimental_set_query_params(tab=slug)
        except Exception:
            pass


# ----------------------------
# Page renders
# ----------------------------

def _nav_label(slug: str) -> str:
    for it in NAV_ORDER:
        if it.slug == slug:
            return it.label
    return slug.title()


def _render_page_header(slug: str, label: str, show_title: bool = True) -> None:
    """Renders the PNG emoji icon in the main area (optionally with a big title)."""
    icon_p = _emoji_path(slug)
    b64i = _b64_png(icon_p) if icon_p and icon_p.exists() else ""
    if not b64i:
        if show_title:
            st.title(label)
        return

    if show_title:
        st.markdown(
            f"""<div class='pms-page-header'>
  <img class='pms-page-ico' src='data:image/png;base64,{b64i}' alt='{label}' />
  <div><div class='pms-page-title'>{label}</div></div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class='pms-page-header'>
  <img class='pms-page-ico' src='data:image/png;base64,{b64i}' alt='{label}' />
</div>""",
            unsafe_allow_html=True,
        )

def _sync_owner_from_home():
    try:
        val = st.session_state.get("owner_select")
        if val:
            st.session_state["owner"] = val
    except Exception:
        pass

def _render_home(owner_key: str):
    # ✅ Pool logo — centered ABOVE title
    if POOL_LOGO.exists():
        p = _transparent_copy_edge(POOL_LOGO)
        mime_p, b64p = _b64_image(p if (p and p.exists()) else POOL_LOGO, max_w=680)
        if b64p:
            st.markdown(
                f"""<div class='pms-pool-wrap'><div class='pms-pool-card'>
<img src='data:{mime_p};base64,{b64p}' alt='pool' />
</div></div>""",
                unsafe_allow_html=True,
            )

    _render_page_header('home', 'Home', show_title=True)
    st.caption("Choisis ton équipe ci-dessous.")

    st.subheader("🏒 Sélection d'équipe")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    # Select team (full width) — team logo is shown in sidebar only
    idx = POOL_TEAMS.index(owner_key) if owner_key in POOL_TEAMS else 0
    st.selectbox(
        "Équipe (propriétaire)",
        POOL_TEAMS,
        index=idx,
        key="owner_select",
        on_change=_sync_owner_from_home,
    )
    new_owner = st.session_state.get("owner_select", owner_key)

    st.success(f"✅ Équipe sélectionnée: {TEAM_LABEL.get(new_owner, new_owner)}")

    # Optional banner (rendered as HTML to avoid loader dots)
    banner = DATA_DIR / "nhl_teams_header_banner.png"
    if active == 'home' and banner.exists():
        b = _transparent_copy_edge(banner)
        mime_b, b64b = _b64_image(b if (b and b.exists()) else banner, max_w=980)
        if b64b:
            st.markdown(
                f"""<div class='pms-pool-wrap'><div class='pms-pool-card'>
<img src='data:{mime_b};base64,{b64b}' alt='banner' />
</div></div>""",
                unsafe_allow_html=True,
            )
TAB_MODULES = {
    "home": None,
    "gm": "tabs.gm",
    "joueurs": "tabs.joueurs",
    "alignement": "tabs.alignement",
    "transactions": "tabs.transactions",
    "historique": "tabs.historique",
    "classement": "tabs.classement",
    "admin": "tabs.admin",
}

@st.cache_resource(show_spinner=False)
def _import_module_cached(modpath: str):
    import importlib
    return importlib.import_module(modpath)

def _safe_import_tab(slug: str):
    modpath = TAB_MODULES.get(slug)
    if not modpath:
        return None
    try:
        return _import_module_cached(modpath)
    except Exception as e:
        return e

def _safe_import_tabs() -> Dict[str, Any]:
    """Compatibility helper (avoid using this in main; it's slower)."""
    modules: Dict[str, Any] = {}
    for slug, modpath in TAB_MODULES.items():
        if not modpath:
            continue
        modules[slug] = _safe_import_tab(slug)
    return modules

def _render_module(mod: Any, ctx: Dict[str, Any]):
    if isinstance(mod, Exception):
        st.error(f"❌ Module erreur: {mod}")
        return
    # expected: module.render(ctx) else fallback
    if hasattr(mod, "render"):
        try:
            mod.render(ctx)  # type: ignore
            return
        except Exception as e:
            st.exception(e)
            return
    st.warning("Ce module n'a pas de fonction render(ctx).")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏒", layout="wide")

    # state defaults
    st.session_state.setdefault("season", "2025-2026")
    st.session_state.setdefault("owner", "Canadiens")
    st.session_state.setdefault("active_tab", "home")
    st.session_state.setdefault("_pms_qp_applied", False)
    st.session_state.setdefault("_pms_booted", False)
    st.session_state.setdefault("pms_sidebar_collapsed", False)
    st.session_state.setdefault("pms_light_mode", False)
    st.session_state.setdefault("_pms_prof", {})

    _t0 = time.perf_counter()
    _apply_css()
    _apply_theme_mode()
    st.session_state['_pms_prof']['css_theme_ms'] = int((time.perf_counter()-_t0)*1000)

    # Read query param tab
    qp = (_get_query_tab() or "").strip().lower()
    allowed = {it.slug for it in NAV_ORDER}
    # Apply URL query param only once (otherwise it can overwrite sidebar clicks and feel like "2 clicks").
    # Apply URL query param only at first boot (prevents overwrite + avoids the '2 clicks' feeling).
    if not bool(st.session_state.get("_pms_booted", False)):
        if qp in allowed:
            st.session_state["active_tab"] = qp
        st.session_state["_pms_booted"] = True
        st.session_state["_pms_qp_applied"] = True

    # Owner is needed for access control + sidebar rendering.
    owner = str(st.session_state.get("owner") or "Canadiens")
    active_pre = str(st.session_state.get("active_tab") or "home")
    # Sidebar: season + nav (handled inside _sidebar_nav)
    _t1 = time.perf_counter()
    _sidebar_nav(owner, active_pre)
    st.session_state['_pms_prof']['sidebar_ms'] = int((time.perf_counter()-_t1)*1000)
    # Re-read after sidebar: avoids "2 clicks" even if rerun is skipped
    active = str(st.session_state.get("active_tab") or active_pre or "home")

    # Prevent non-whalers from accessing Admin
    if active == "admin" and owner != "Whalers":
        st.session_state["active_tab"] = "home"
        active = "home"

    # Context passed to tabs
    ctx: Dict[str, Any] = {
        "season": st.session_state.get("season"),
        "owner": owner,
        "selected_owner": owner,
        "data_dir": str(DATA_DIR),
        "assets_dir": str(ASSETS_DIR),
    }

    # Render page
    if active == "home":
        _render_home(owner)
        return

    # Center icon for other pages (keeps tab modules intact; shows the same PNG icon as sidebar)
    _render_page_header(active, _nav_label(active), show_title=True)

    _t2 = time.perf_counter()
    mod = _safe_import_tab(active)
    st.session_state['_pms_prof']['import_ms'] = int((time.perf_counter()-_t2)*1000)
    if mod is None:
        st.warning("Onglet indisponible.")
        return

    _t3 = time.perf_counter()
    _render_module(mod, ctx)
    st.session_state['_pms_prof']['render_ms'] = int((time.perf_counter()-_t3)*1000)
    # Quick perf readout
    with st.sidebar.expander('⚡ Perf', expanded=False):
        st.json(st.session_state.get('_pms_prof', {}))


if __name__ == "__main__":
    main()
