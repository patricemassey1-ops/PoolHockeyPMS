# app.py — PoolHockeyPMS (routing + thème + contexte) — UI FINAL (sidebar rouge + banner top)
# ------------------------------------------------------------------------------------------------
# Objectifs:
# - Dark par défaut (navy, pas noir pur) + toggle mode clair (sidebar)
# - Accent rouge IDENTIQUE (style "pill" sélectionnée) dans la sidebar Navigation
# - Garder une seule navigation (radio) + une seule injection CSS par run
# - Home: logo_pool.png complètement en haut avec proportions stables
# - Pas de "session_state key modified after widget" (sync via callback)
# ------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import base64
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional

import streamlit as st

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Pool GM",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_SEASON = "2025-2026"

POOL_TEAMS = ["Whalers", "Red_Wings", "Predateurs", "Nordiques", "Cracheurs", "Canadiens"]

TEAM_LOGO = {t: os.path.join(DATA_DIR, f"{t}.png") for t in POOL_TEAMS}
APP_LOGO = os.path.join(DATA_DIR, "gm_logo.png")
BANNER = os.path.join(DATA_DIR, "logo_pool.png")

# =========================
# THEME (1 seule injection)
# =========================
ACCENT_RED = "#ef4444"   # identique (Tailwind red-500)
ACCENT_RED_D = "#dc2626" # hover

THEME_CSS_DARK = f"""
<style>
:root {{ color-scheme: dark; }}

/* Background (navy / gradient, pas noir pur) */
html, body, [data-testid="stAppViewContainer"]{{
  background:
    radial-gradient(1200px 800px at 18% -12%, rgba(239,68,68,.18), transparent 55%),
    radial-gradient(1200px 800px at 95% 10%, rgba(80,140,255,.12), transparent 55%),
    linear-gradient(180deg, #0b1220, #070b12 62%, #070b12) !important;
  color: #e7eef7 !important;
}}
.block-container {{ padding-top: .85rem; padding-bottom: 2.5rem; max-width: 1200px; }}

/* Sidebar */
section[data-testid="stSidebar"]{{
  background: linear-gradient(180deg,#0d1627,#0b1220) !important;
  border-right: 1px solid rgba(255,255,255,.06);
}}

/* Cards */
.pms-card{{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px;
  padding: 18px;
}}
.pms-muted{{ opacity:.72; font-size:13px; }}

/* Buttons */
.stButton>button{{ border-radius: 12px !important; }}
.stButton>button[kind="primary"]{{
  background: {ACCENT_RED} !important;
  border: 1px solid rgba(239,68,68,.55) !important;
}}
.stButton>button[kind="primary"]:hover{{ background: {ACCENT_RED_D} !important; }}

/* Selectbox / inputs rounded */
[data-testid="stSelectbox"] > div > div {{
  border-radius: 14px !important;
}}
[data-baseweb="select"] > div {{
  border-radius: 14px !important;
}}

/* ---- NAV RADIO: look rouge comme screenshot ---- */
section[data-testid="stSidebar"] div[role="radiogroup"] label{{
  width: 100%;
  margin: 0.15rem 0;
}}
/* Conteneur clickable */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div{{
  width: 100%;
  border-radius: 14px;
  padding: .70rem .85rem;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.03);
  transition: all .12s ease;
}}
/* Hover */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover > div{{
  background: rgba(255,255,255,.05);
  border-color: rgba(255,255,255,.16);
}}
/* Hide the little radio dot */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child{{
  display:none !important;
}}
/* Make text bigger like screenshot */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:last-child{{
  font-size: 16px;
  font-weight: 650;
}}
/* Selected = pill rouge */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) > div{{
  background: rgba(239,68,68,.95) !important;
  border-color: rgba(239,68,68,.95) !important;
  box-shadow: 0 14px 30px rgba(239,68,68,.14);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) > div > div:last-child{{
  color: white !important;
}}

/* Sidebar headings spacing */
.pms-sb-title{{ font-weight: 800; font-size: 20px; letter-spacing: -.02em; }}
.pms-sb-section{{ margin-top: .75rem; margin-bottom: .35rem; opacity:.85; font-weight: 700; }}

/* Banner card image */
.pms-banner-wrap img{{
  width: 100%;
  max-height: 260px;
  object-fit: cover;
  border-radius: 18px;
  display:block;
}}
</style>
"""

THEME_CSS_LIGHT = f"""
<style>
:root {{ color-scheme: light; }}

html, body, [data-testid="stAppViewContainer"]{{
  background:
    radial-gradient(1000px 700px at 18% -12%, rgba(239,68,68,.14), transparent 55%),
    radial-gradient(1000px 700px at 95% 10%, rgba(60,140,255,.12), transparent 55%),
    linear-gradient(180deg, #f7f8fc, #f3f5fb 60%, #f3f5fb) !important;
  color: #0b1020 !important;
}}
.block-container {{ padding-top: .85rem; padding-bottom: 2.5rem; max-width: 1200px; }}

section[data-testid="stSidebar"]{{
  background: #ffffff !important;
  border-right: 1px solid rgba(0,0,0,.06);
}}

.pms-card{{
  background: #ffffff;
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: 18px;
}}
.pms-muted{{ opacity:.72; font-size:13px; }}

/* Inputs rounded */
[data-testid="stSelectbox"] > div > div, [data-baseweb="select"] > div {{
  border-radius: 14px !important;
}}

/* Nav radio (même rouge) */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div{{
  width: 100%;
  border-radius: 14px;
  padding: .70rem .85rem;
  border: 1px solid rgba(0,0,0,.10);
  background: rgba(0,0,0,.02);
  transition: all .12s ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover > div{{
  background: rgba(0,0,0,.03);
  border-color: rgba(0,0,0,.14);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child{{ display:none !important; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:last-child{{ font-size: 16px; font-weight: 650; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) > div{{
  background: rgba(239,68,68,.95) !important;
  border-color: rgba(239,68,68,.95) !important;
  box-shadow: 0 14px 30px rgba(239,68,68,.12);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) > div > div:last-child{{ color: white !important; }}

.pms-sb-title{{ font-weight: 800; font-size: 20px; letter-spacing: -.02em; }}
.pms-sb-section{{ margin-top: .75rem; margin-bottom: .35rem; opacity:.85; font-weight: 700; }}

.pms-banner-wrap img{{
  width: 100%;
  max-height: 260px;
  object-fit: cover;
  border-radius: 18px;
  display:block;
}}
</style>
"""

def apply_theme() -> None:
    mode = st.session_state.get("ui_theme", "dark")
    st.markdown(THEME_CSS_LIGHT if mode == "light" else THEME_CSS_DARK, unsafe_allow_html=True)

# =========================
# CONTEXT
# =========================
@dataclass
class AppCtx:
    data_dir: str
    season_lbl: str
    owner: str
    is_admin: bool
    theme: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "DATA_DIR": self.data_dir,
            "season_lbl": self.season_lbl,
            "owner": self.owner,
            "is_admin": self.is_admin,
            "theme": self.theme,
        }

def _is_admin(owner: str) -> bool:
    return (owner or "").strip() == "Whalers"

def _b64(path: str) -> str:
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

def _safe_image(path: str, width: Optional[int] = None) -> None:
    try:
        if path and os.path.exists(path):
            st.image(path, width=width)
    except Exception:
        pass

# =========================
# NAV
# =========================
TABS = [
    ("🏠 Home", "home"),
    ("👥 Joueurs", "joueurs"),
    ("🧊 Alignement", "alignement"),
    ("🔁 Transactions", "transactions"),
    ("🧑‍💼 GM", "gm"),
    ("🕘 Historique", "historique"),
    ("🏆 Classement", "classement"),
    ("🛠️ Admin", "admin"),
]

def _import_tabs():
    try:
        from tabs import home, joueurs, alignement, transactions, gm, historique, classement, admin
        return {
            "home": home,
            "joueurs": joueurs,
            "alignement": alignement,
            "transactions": transactions,
            "gm": gm,
            "historique": historique,
            "classement": classement,
            "admin": admin,
        }
    except Exception:
        st.error("Impossible d'importer les modules dans /tabs/.")
        st.code(traceback.format_exc())
        st.stop()

# ---- Sync owner (SANS modifier la key d'un widget après instanciation)
def _on_home_owner_change():
    # callback: home_owner_select -> owner (source of truth)
    picked = st.session_state.get("home_owner_select")
    if picked:
        st.session_state["owner"] = picked

def _on_sidebar_owner_change():
    picked = st.session_state.get("owner_select")
    if picked:
        st.session_state["owner"] = picked

def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown('<div class="pms-sb-title">Pool GM</div>', unsafe_allow_html=True)

        cL, cR = st.columns([1, 4], vertical_alignment="center")
        with cL:
            _safe_image(APP_LOGO, width=44)
        with cR:
            st.caption("Gestion du pool (PMS)")

        st.markdown('<div class="pms-sb-section">Saison</div>', unsafe_allow_html=True)
        st.selectbox(
            "Saison",
            options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
            key="season_lbl",
            label_visibility="collapsed",
        )

        st.markdown('<div class="pms-sb-section">Navigation</div>', unsafe_allow_html=True)
        labels = [t[0] for t in TABS]
        cur = st.session_state.get("active_tab", "🏠 Home")
        idx = labels.index(cur) if cur in labels else 0

        active = st.radio(
            "Navigation",
            labels,
            index=idx,
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["active_tab"] = active

        st.markdown("---")

        st.markdown('<div class="pms-sb-section">Mon équipe</div>', unsafe_allow_html=True)
        # owner_select key in sidebar; sync via callback
        st.selectbox(
            "Mon équipe",
            options=POOL_TEAMS,
            key="owner_select",
            index=POOL_TEAMS.index(st.session_state.get("owner") or "Whalers") if (st.session_state.get("owner") or "Whalers") in POOL_TEAMS else 0,
            on_change=_on_sidebar_owner_change,
            label_visibility="collapsed",
        )
        # logo beside
        _safe_image(TEAM_LOGO.get(st.session_state.get("owner") or "", ""), width=42)

        st.markdown("")

        # Theme toggle (sidebar)
        is_light = st.toggle(
            "Mode clair",
            value=(st.session_state.get("ui_theme", "dark") == "light"),
            key="ui_theme_toggle_sidebar",
        )
        st.session_state["ui_theme"] = "light" if is_light else "dark"

    return active

# =========================
# HOME
# =========================
def _render_banner_top():
    if not (BANNER and os.path.exists(BANNER)):
        return
    b64 = _b64(BANNER)
    if not b64:
        # fallback
        st.image(BANNER, use_container_width=True)
        return
    st.markdown(
        f"""
        <div class="pms-banner-wrap" style="margin-bottom: 1rem;">
          <img src="data:image/png;base64,{b64}" alt="logo_pool"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_home(ctx: AppCtx):
    _render_banner_top()

    st.title("🏠 Home")
    st.markdown('<div class="pms-muted">Home reste clean — aucun bloc Admin ici.</div>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown('<div class="pms-card">', unsafe_allow_html=True)
    st.markdown("### 🏒 Sélection d'équipe")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    with c1:
        st.selectbox(
            "Équipe (propriétaire)",
            options=POOL_TEAMS,
            index=POOL_TEAMS.index(ctx.owner) if ctx.owner in POOL_TEAMS else 0,
            key="home_owner_select",
            on_change=_on_home_owner_change,
        )
        # ctx.owner is derived from session_state after callback; keep it in sync visually
        ctx.owner = st.session_state.get("owner") or ctx.owner

    with c2:
        _safe_image(TEAM_LOGO.get(ctx.owner, ""), width=80)

    st.success(f"✅ Équipe sélectionnée: **{ctx.owner}**")
    st.markdown(f"**Saison:** {ctx.season_lbl}")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MAIN
# =========================
def main() -> None:
    # session defaults (single source of truth)
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("owner", "Whalers")
    st.session_state.setdefault("season_lbl", DEFAULT_SEASON)
    st.session_state.setdefault("active_tab", "🏠 Home")

    # Apply theme once
    apply_theme()

    # Sidebar (sets active_tab, owner, theme)
    active_label = sidebar_nav()

    # Build ctx
    owner = st.session_state.get("owner") or "Whalers"
    season = st.session_state.get("season_lbl") or DEFAULT_SEASON
    ctx = AppCtx(
        data_dir=DATA_DIR,
        season_lbl=str(season),
        owner=str(owner),
        is_admin=_is_admin(owner),
        theme=st.session_state.get("ui_theme", "dark"),
    )

    # Import modules
    modules = _import_tabs()
    key = dict(TABS).get(active_label, "home")

    try:
        if key == "home":
            _render_home(ctx)
        elif key == "admin" and not ctx.is_admin:
            st.title("🛠️ Admin")
            st.warning("Accès admin requis.")
        else:
            mod = modules.get(key)
            if mod is None:
                st.error(f"Module introuvable: tabs/{key}.py")
            else:
                if hasattr(mod, "render"):
                    try:
                        mod.render(ctx.as_dict())
                    except TypeError:
                        mod.render(ctx)
                else:
                    st.error(f"tabs/{key}.py n'a pas de fonction render(ctx).")
    except Exception:
        st.error("Une erreur a été détectée (évite l'écran noir).")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
