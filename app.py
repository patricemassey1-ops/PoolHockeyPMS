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
    """Return ORIGINAL team logo (keep white background)."""
    team = str(team or "").strip()
    cands = TEAM_LOGO_CANDIDATES.get(team, [])
    for fn in cands:
        p = ASSETS_DIR / fn
        if p.exists():
            return p
    p = ASSETS_DIR / f"{team}_Logo.png"
    return p if p.exists() else Path("")


def _emoji_path(slug: str) -> Path:
    slug = str(slug or "").strip()
    fn = EMOJI_ICON.get(slug, "")
    p = (ASSETS_DIR / fn) if fn else Path("")
    if not p.exists():
        return p
    # Only the cup icon tends to have a white box — remove it for that one only.
    if fn == "emoji_coupe.png":
        return _transparent_copy(p)
    return p


def _gm_logo_path() -> Path:
    """Return ORIGINAL GM logo (keep its background)."""
    p = DATA_DIR / "gm_logo.png"
    return p if p.exists() else Path("")


def _apply_css():
    """
    CSS global (Apple glass + navigation). IMPORTANT:
    - Tout le CSS est dans des strings (évite NameError / erreurs d'indent).
    - Pas de f-string (les accolades CSS restent normales).
    """
    css = """
<style>
/* Layout: custom sidebar widths */
section[data-testid="stSidebar"] {
  width: var(--pms-sb-w, 320px) !important;
  min-width: var(--pms-sb-w, 320px) !important;
  max-width: var(--pms-sb-w, 320px) !important;
}

/* Give content a little breathing room (also pushes titles down so they're not hidden) */
.block-container {
  padding-top: 2.1rem !important;
}

/* Hide Streamlit default nav + make our sidebar clean */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] .stRadio { display: none !important; }

/* Sidebar base (glass) */
section[data-testid="stSidebar"] > div {
  background: rgba(18,24,38,0.88) !important;
  backdrop-filter: blur(16px) !important;
  border-right: 1px solid rgba(255,255,255,0.10) !important;
}

/* Light mode sidebar background */
.pms-light section[data-testid="stSidebar"] > div {
  background: rgba(248,250,252,0.92) !important;
  border-right: 1px solid rgba(0,0,0,0.08) !important;
}

/* NAV container */
.pms-nav { padding: 10px 10px 6px 10px; }
.pms-nav a { text-decoration: none !important; }

/* NAV item (Apple glass pill) */
.pms-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 12px 14px;
  margin: 10px 0;
  border-radius: 18px;
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
  background: __RED_ACTIVE__;
  border-color: rgba(220,38,38,1);
  box-shadow: 0 16px 34px rgba(239,68,68,0.22);
}
.pms-item.active::before {
  content: "";
  position: absolute;
  left: -7px;
  top: 10px;
  bottom: 10px;
  width: 3px;
  border-radius: 6px;
  background: rgba(239,68,68,1);
  box-shadow: 0 10px 24px rgba(239,68,68,0.45);
}

/* Icon + label */
.pms-item img {
  width: var(--pms-ico, 54px);
  height: var(--pms-ico, 54px);
  border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.22);
}
.pms-item .txt {
  font-weight: 700;
  letter-spacing: .2px;
}

/* Collapsed mode (icons only) */
.pms-collapsed section[data-testid="stSidebar"] { width: var(--pms-sb-w, 76px) !important; }
.pms-collapsed .pms-item { justify-content: center; padding: 8px 0; }
.pms-collapsed .pms-item .txt { display: none; }
.pms-collapsed .pms-item img {
  width: var(--pms-ico-c, 44px);
  height: var(--pms-ico-c, 44px);
  border-radius: 14px;
}

/* Team chip (sidebar) */
.pms-team {
  display:flex;
  align-items:center;
  gap:10px;
  padding: 10px 12px 0 12px;
}
.pms-team img {
  width: var(--pms-ico-c, 44px);
  height: var(--pms-ico-c, 44px);
  border-radius: 14px;
}
.pms-team .tt { font-weight: 700; opacity: .9; }

/* Theme toggle bottom */
.pms-theme { padding: 10px 10px 10px 10px; }
.pms-theme .sun { font-size: 18px; opacity: .9; padding-left: 4px; padding-bottom: 2px; }

/* Custom tooltip for nav items (black bubble) */
.pms-item[title]:hover::after {
  content: attr(title);
  position: absolute;
  left: 74px;
  top: 50%;
  transform: translateY(-50%);
  background: rgba(0,0,0,0.84);
  color: #fff;
  padding: 8px 10px;
  border-radius: 10px;
  font-size: 12px;
  line-height: 12px;
  white-space: nowrap;
  z-index: 9999;
  box-shadow: 0 14px 30px rgba(0,0,0,0.35);
}
.pms-item[title]:hover::before {
  content: "";
  position: absolute;
  left: 66px;
  top: 50%;
  transform: translateY(-50%);
  width: 0; height: 0;
  border-top: 6px solid transparent;
  border-bottom: 6px solid transparent;
  border-right: 6px solid rgba(0,0,0,0.84);
  z-index: 9999;
}

/* Apple Snap transition (page/tab change) */
@keyframes pmsPageIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0px); }
}
section.main .block-container {
  animation: pmsPageIn 240ms ease-out;
}

/* Apple Micro Press (click) */
section[data-testid="stSidebar"] .pms-item:active {
  transform: translateY(0px) scale(0.98);
}

/* Apple Glass Shimmer */
@keyframes pmsShimmer {
  0% { transform: translateX(-120%) rotate(12deg); opacity: 0.0; }
  35% { opacity: 0.35; }
  100% { transform: translateX(220%) rotate(12deg); opacity: 0.0; }
}
.pms-item { overflow: hidden; }
.pms-item::after {
  content: "";
  position: absolute;
  inset: -40% -60%;
  background: linear-gradient(90deg,
    rgba(255,255,255,0.00) 0%,
    rgba(255,255,255,0.08) 40%,
    rgba(255,255,255,0.18) 50%,
    rgba(255,255,255,0.08) 60%,
    rgba(255,255,255,0.00) 100%);
  transform: translateX(-140%) rotate(12deg);
  opacity: 0;
  pointer-events: none;
}
.pms-item:hover::after { animation: pmsShimmer 650ms ease-out; }
</style>
"""
    css = css.replace("__RED_ACTIVE__", RED_ACTIVE)
    st.markdown(css, unsafe_allow_html=True)


def _set_sidebar_mode(collapsed: bool):
    # In Streamlit, easiest is to set CSS vars on :root
    if collapsed:
        st.markdown(
            """
<style>
:root { --pms-sb-w: 76px; --pms-ico: 54px; --pms-ico-c: 44px; --pms-gm: 120px; }
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
:root { --pms-sb-w: 320px; --pms-ico: 54px; --pms-ico-c: 44px; --pms-gm: 120px; }
</style>
""",
            unsafe_allow_html=True,
        )


def _sidebar_nav(owner_key: str, active_slug: str):
    collapsed = bool(st.session_state.get("pms_sidebar_collapsed", False))

    # pms_team_select_sidebar
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

    # Brand: GM logo only (no pool logo in sidebar)
    gm_logo = _gm_logo_path()
    gm_b64 = _b64_png(gm_logo) if gm_logo else ""
    brand_img = f"<img src='data:image/png;base64,{gm_b64}' />" if gm_b64 else "<div style='width:34px;height:34px'></div>"

    # Team logo chip
    t_logo = _team_logo_path(owner_key)
    t_b64 = _b64_png(t_logo) if t_logo else ""
    team_img = f"<img src='data:image/png;base64,{t_b64}' />" if t_b64 else "<div style='width:34px;height:34px'></div>"

    # Items (Admin only when Whalers)
    items = []
    for it in NAV_ORDER:
        if it.slug == "admin" and owner_key != "Whalers":
            continue
        items.append(it)

    # Build HTML
    sb_class = "pms-collapsed" if collapsed else ""
    html = [f"<div class='{sb_class}'>"]

    html.append(f"<div class='pms-brand'>{brand_img}<div class='t'>{APP_TITLE}</div></div>")
    html.append(f"<div class='pms-team'>{team_img}<div class='tt'>{TEAM_LABEL.get(owner_key, owner_key)}</div></div>")
    html.append("<div class='pms-nav'>")

    for it in items:
        ico_path = _emoji_path(it.emoji_slug) or _emoji_path("home")
        ico_b64 = _b64_png(ico_path) if ico_path else ""
        ico_html = f"<img src='data:image/png;base64,{ico_b64}' />" if ico_b64 else "<div style='width:34px;height:34px'></div>"
        active_cls = "active" if it.slug == active_slug else ""
        html.append(
            f"<a class='pms-item {active_cls}' href='?tab={it.slug}' title='{it.label}'>"
            f"{ico_html}<span class='lbl'>{it.label}</span>"
            f"</a>"
        )

    html.append("</div>")  # nav
    html.append("</div>")  # root

    st.sidebar.markdown("\n".join(html), unsafe_allow_html=True)

    # Footer: collapse toggle + theme toggle (below nav)
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("▶" if collapsed else "◀", key="pms_toggle_sidebar", help="Réduire / agrandir le menu"):
            st.session_state["pms_sidebar_collapsed"] = not collapsed
            # st.rerun()  # Streamlit rerun automatiquement
    with col2:
        # Spacer in expanded for symmetry
        st.write("")

    st.sidebar.markdown("<div class='pms-theme'><div class='sun'>☀️</div></div>", unsafe_allow_html=True)

    light = bool(st.session_state.get("pms_light_mode", False))
    new_light = st.sidebar.toggle("Clair", value=light, key="pms_light_mode_toggle", label_visibility="collapsed" if collapsed else "visible")
    if new_light != light:
        st.session_state["pms_light_mode"] = new_light
        # st.rerun()  # Streamlit rerun automatiquement


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
    # Callback: runs before script body on rerun (safe with Streamlit widgets)
    try:
        val = st.session_state.get("owner_select")
        if val:
            st.session_state["owner"] = val
    except Exception:
        pass

def _render_home(owner_key: str):
    # ✅ Pool logo MUST be above the page title
    if POOL_LOGO.exists():
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.image(str(POOL_LOGO), use_container_width=True)

    st.title("🏠 Home")
    st.caption("Choisis ton équipe ci-dessous.")

    st.subheader("🏒 Sélection d'équipe")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    colA, colB = st.columns([5, 1.2])
    with colA:
        idx = POOL_TEAMS.index(owner_key) if owner_key in POOL_TEAMS else 0
        new_owner = st.selectbox("Équipe (propriétaire)", POOL_TEAMS, index=idx, key="owner_select", on_change=_sync_owner_from_home)
    with colB:
        p = _team_logo_path(new_owner)
        if p and p.exists():
            st.image(str(p), width=72)

    st.success(f"✅ Équipe sélectionnée: {TEAM_LABEL.get(new_owner, new_owner)}")
    
    # Optional banner in center (kept if you want)
    banner = DATA_DIR / "nhl_teams_header_banner.png"
    if banner.exists():
        st.image(str(_transparent_copy(banner)), use_container_width=True)
    # owner is synced via callbackdef _safe_import_tabs() -> Dict[str, Any]:
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

    # Sidebar: season + nav
    with st.sidebar:
        _sidebar_nav(owner, active)

    # Prevent non-whalers from accessing Admin
    if active == "admin" and owner != "Whalers":
        st.session_state["active_tab"] = "home"
        _set_query_tab("home")
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
