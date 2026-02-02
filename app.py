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


def _team_logo_path(team_key: str) -> Optional[Path]:
    cand = TEAM_LOGO_CANDIDATES.get(team_key) or []
    p = _pick_existing(ASSETS_DIR, cand)
    if p:
        return _transparent_copy(p)
    # fallback: accept any file matching "{team}*.png"
    try:
        hits = sorted(ASSETS_DIR.glob(f"{team_key}*Logo*.png"))
        if hits:
            return _transparent_copy(hits[0])
    except Exception:
        pass
    return None


def _emoji_path(slug: str) -> Optional[Path]:
    fn = EMOJI_FILES.get(slug)
    if not fn:
        return None
    p = ASSETS_DIR / fn
    return _transparent_copy(p) if p.exists() else None


def _gm_logo_path() -> Optional[Path]:
    return _transparent_copy(GM_LOGO) if GM_LOGO.exists() else None


@dataclass
class NavItem:
    slug: str
    label: str
    emoji_slug: str  # key in EMOJI_FILES


# Navigation order requested
NAV_ORDER = [
    NavItem("home", "Home", "home"),
    NavItem("gm", "GM", "gm"),
    NavItem("joueurs", "Joueurs", "joueurs"),
    NavItem("alignement", "Alignement", "alignement"),
    NavItem("transactions", "Transactions", "transactions"),
    NavItem("historique", "Historique", "historique"),
    NavItem("classement", "Classement", "classement"),
    NavItem("admin", "Admin", "historique"),  # icon fallback (admin has no emoji file)
]


# ----------------------------
# CSS (Apple-like)
# ----------------------------
def _apply_css():
    st.markdown(
        f"""
<style>
/* Layout: custom sidebar widths */
section[data-testid="stSidebar"] {{
  width: var(--pms-sb-w, 320px) !important;
  min-width: var(--pms-sb-w, 320px) !important;
  max-width: var(--pms-sb-w, 320px) !important;
}}
/* Give content a little breathing room (also pushes titles down so they're not hidden) */
.block-container {{
  padding-top: 2.1rem !important;
}}

/* Hide Streamlit default nav + make our sidebar clean */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
  display:none !important;
}}
section[data-testid="stSidebar"] .stRadio, section[data-testid="stSidebar"] .stSelectbox {{
  margin-top: 0.25rem;
}}

/* Brand */
.pms-brand {{
  display:flex; align-items:center; gap:10px;
  padding: 10px 10px 12px 10px;
}}
.pms-brand img {{
  width: 34px; height: 34px; border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0,0,0,.20);
}}
.pms-brand .t {{
  font-weight: 700; font-size: 1.05rem;
  letter-spacing: .2px;
}}

/* Nav container */
.pms-nav {{
  display:flex; flex-direction: column; gap: 10px;
  padding: 6px 10px 8px 10px;
}}
.pms-item {{
  display:flex; align-items:center; gap: 12px;
  border-radius: 14px;
  padding: 8px 10px;
  text-decoration:none !important;
  color: rgba(255,255,255,.88);
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.05);
  transition: all .12s ease-in-out;
}}
.pms-item:hover {{
  transform: translateY(-1px);
  background: rgba(255,255,255,.06);
  border-color: rgba(255,255,255,.10);
}}
.pms-item.active {{
  background: {RED_ACTIVE};
  border-color: rgba(0,0,0,.18);
  box-shadow: 0 18px 36px rgba(239,68,68,.22);
  color: white;
}}
.pms-item img {{
  width: var(--pms-ico, 54px);
  height: var(--pms-ico, 54px);
  border-radius: 12px;
  object-fit: contain;
}}
.pms-item .lbl {{
  font-weight: 650;
  font-size: 1.00rem;
  line-height: 1;
}}

/* Collapsed mode: hide text, center icons, make pills */
.pms-collapsed .pms-brand .t {{ display:none; }}
.pms-collapsed .pms-item .lbl {{ display:none; }}
.pms-collapsed .pms-nav {{
  padding: 6px 8px;
}}
.pms-collapsed .pms-item {{
  justify-content: center;
  padding: 10px 0;
}}
.pms-collapsed .pms-item img {{
  width: var(--pms-ico-c, 36px);
  height: var(--pms-ico-c, 36px);
  border-radius: 12px;
}}
.pms-collapsed .pms-item {{
  border-radius: 16px;
}}

/* Team logo chip (collapsed + expanded) */
.pms-team {{
  display:flex; align-items:center; gap:10px;
  padding: 10px 10px 0 10px;
}}
.pms-team img {{
  width: 34px; height: 34px; border-radius: 12px;
  object-fit: contain;
  background: transparent;
}}
.pms-team .tt {{
  font-size: .92rem;
  font-weight: 650;
  opacity: .90;
}}
.pms-collapsed .pms-team .tt {{ display:none; }}
.pms-collapsed .pms-team {{
  justify-content:center;
  padding: 8px 8px 0 8px;
}}

/* Theme toggle bottom */
.pms-theme {{
  padding: 10px 10px 6px 10px;
}}
.pms-theme .sun {{
  font-size: 18px;
  opacity: .9;
  padding-left: 4px;
  padding-bottom: 2px;
}}

/* Active button (primary) uses same red as nav active */
.stButton > button[kind="primary"] {{
  background: {RED_ACTIVE} !important;
  border-color: rgba(0,0,0,.18) !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _set_sidebar_mode(collapsed: bool):
    # In Streamlit, easiest is to set CSS vars on :root
    if collapsed:
        st.markdown(
            """
<style>
:root { --pms-sb-w: 72px; --pms-ico: 54px; --pms-ico-c: 36px; }
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
:root { --pms-sb-w: 320px; --pms-ico: 54px; --pms-ico-c: 36px; }
</style>
""",
            unsafe_allow_html=True,
        )


def _sidebar_nav(owner_key: str, active_slug: str):
    collapsed = bool(st.session_state.get("pms_sidebar_collapsed", False))
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
            f"<a class='pms-item {active_cls}' href='?tab={it.slug}'>"
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
            st.rerun()
    with col2:
        # Spacer in expanded for symmetry
        st.write("")

    st.sidebar.markdown("<div class='pms-theme'><div class='sun'>☀️</div></div>", unsafe_allow_html=True)

    light = bool(st.session_state.get("pms_light_mode", False))
    new_light = st.sidebar.toggle("Clair", value=light, key="pms_light_mode_toggle", label_visibility="collapsed" if collapsed else "visible")
    if new_light != light:
        st.session_state["pms_light_mode"] = new_light
        st.rerun()


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
def _render_home(owner_key: str):
    # ✅ Pool logo MUST be above the page title
    if POOL_LOGO.exists():
        st.image(str(_transparent_copy(POOL_LOGO)), width=150)

    st.title("🏠 Home")
    st.caption("Home reste clean — aucun bloc Admin ici.")

    st.subheader("🏒 Sélection d'équipe")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    colA, colB = st.columns([5, 1.2])
    with colA:
        idx = POOL_TEAMS.index(owner_key) if owner_key in POOL_TEAMS else 0
        new_owner = st.selectbox("Équipe (propriétaire)", POOL_TEAMS, index=idx, key="owner_select")
    with colB:
        p = _team_logo_path(new_owner)
        if p and p.exists():
            st.image(str(p), width=54)

    st.success(f"✅ Équipe sélectionnée: {TEAM_LABEL.get(new_owner, new_owner)}")
    st.write(f"Saison: {st.session_state.get('season', '2025-2026')}")

    # Optional banner in center (kept if you want)
    banner = DATA_DIR / "nhl_teams_header_banner.png"
    if banner.exists():
        st.image(str(_transparent_copy(banner)), use_container_width=True)

    st.session_state["owner"] = new_owner


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

    # Sidebar: season + nav
    with st.sidebar:
        st.selectbox("Saison", ["2025-2026"], index=0, key="season_select", label_visibility="visible")

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
