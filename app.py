from __future__ import annotations

import base64
import os
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


@st.cache_data(show_spinner=False)
def _b64_png(path: Path) -> str:
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""


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
            return p
    p = ASSETS_DIR / f"{team}_Logo.png"
    return p if p.exists() else Path("")

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

    # Brand: GM logo only (smaller)
    gm_logo = _gm_logo_path()
    if gm_logo and gm_logo.exists():
        st.sidebar.markdown("<div class='pms-brand pms-chip'>", unsafe_allow_html=True)
        st.sidebar.image(str(gm_logo), width=80)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        st.sidebar.markdown("**Pool GM**")

    # Team badge (optional in expanded)
    if not collapsed:
        t_logo = _team_logo_path(owner_key)
        if t_logo and t_logo.exists():
            st.sidebar.markdown("<div class='pms-chip' style='margin-top:10px'>", unsafe_allow_html=True)
            st.sidebar.image(str(t_logo), width=48)
            st.sidebar.markdown("</div>", unsafe_allow_html=True)
            st.sidebar.markdown(f"**{TEAM_LABEL.get(owner_key, owner_key)}**")

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
        # Collapsed: icon-only button
        if collapsed:
            # show icon
            if icon_p and icon_p.exists():
                st.sidebar.image(str(icon_p), width=44)
            # clickable button (icon-only)
            if st.sidebar.button(" ", key=f"nav_{it.slug}", help=it.label, type="primary" if is_active else "secondary"):
                st.session_state["active_tab"] = it.slug
        else:
            c1, c2 = st.sidebar.columns([1.1, 3.4], gap="small")
            with c1:
                if icon_p and icon_p.exists():
                    st.image(str(icon_p), width=60)
            with c2:
                if st.button(it.label, key=f"nav_{it.slug}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state["active_tab"] = it.slug

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
def _sync_owner_from_home():
    try:
        val = st.session_state.get("owner_select")
        if val:
            st.session_state["owner"] = val
    except Exception:
        pass

def _render_home(owner_key: str):
    # ✅ Pool logo (double size) — centered ABOVE title
    if POOL_LOGO.exists():
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.markdown("<div class='pms-chip'>", unsafe_allow_html=True)
            st.image(str(POOL_LOGO), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.title("🏠 Home")
    st.caption("Choisis ton équipe ci-dessous.")

    st.subheader("🏒 Sélection d'équipe")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    colA, colB = st.columns([6, 1.4])
    with colA:
        idx = POOL_TEAMS.index(owner_key) if owner_key in POOL_TEAMS else 0
        st.selectbox(
            "Équipe (propriétaire)",
            POOL_TEAMS,
            index=idx,
            key="owner_select",
            on_change=_sync_owner_from_home,
        )
        new_owner = st.session_state.get("owner_select", owner_key)
    with colB:
        p = _team_logo_path(new_owner)
        if p and p.exists():
            st.markdown("<div class='pms-chip'>", unsafe_allow_html=True)
            st.image(str(p), width=72)
            st.markdown("</div>", unsafe_allow_html=True)

    st.success(f"✅ Équipe sélectionnée: {TEAM_LABEL.get(new_owner, new_owner)}")

    # Optional banner
    banner = DATA_DIR / "nhl_teams_header_banner.png"
    if banner.exists():
        st.image(str(banner), use_container_width=True)

def _safe_import_tabs() -> Dict[str, Any]:
    """
    IMPORTANT:
    - Avoid importing tabs/__init__.py that imports everything.
    - Your tabs/__init__.py should be EMPTY or minimal.
    """
    modules: Dict[str, Any] = {}
    try:
        import importlib
        for slug, mod in [
            ("joueurs", "tabs.joueurs"),
            ("alignement", "tabs.alignement"),
            ("transactions", "tabs.transactions"),
            ("gm", "tabs.gm"),
            ("historique", "tabs.historique"),
            ("classement", "tabs.classement"),
            ("admin", "tabs.admin"),
        ]:
            try:
                modules[slug] = importlib.import_module(mod)
            except Exception as e:
                modules[slug] = e
    except Exception:
        pass
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
    st.session_state.setdefault("pms_sidebar_collapsed", False)
    st.session_state.setdefault("pms_light_mode", False)

    _apply_css()
    _apply_theme_mode()

    # Read query param tab
    qp = (_get_query_tab() or "").strip().lower()
    allowed = {it.slug for it in NAV_ORDER}
    if qp in allowed:
        st.session_state["active_tab"] = qp

    owner = str(st.session_state.get("owner") or "Canadiens")
    active = str(st.session_state.get("active_tab") or "home")
    # Sidebar: season + nav (handled inside _sidebar_nav)
    _sidebar_nav(owner, active)

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

    mods = _safe_import_tabs()
    mod = mods.get(active)
    if mod is None:
        st.warning("Onglet indisponible.")
        return

    _render_module(mod, ctx)


if __name__ == "__main__":
    main()
