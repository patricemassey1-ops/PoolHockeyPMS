# app.py — PoolHockeyPMS (routing + thème + contexte) — UI (mini sidebar icons + banner top)
# ------------------------------------------------------------
# - Dark par défaut (navy, pas noir) + accent rouge (#ef4444) + mode clair
# - Sidebar compacte (icônes seulement)
# - Bandeau logo_pool.png en haut (top) sur toutes les pages
# - Sélection équipe/season dans le header (main), avec logo d'équipe à côté du nom
# - 1 seule injection CSS (apply_theme appelé 1 fois)
# ------------------------------------------------------------

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import Dict, Any

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

TEAM_LABELS = {
    "Whalers": "Whalers",
    "Red_Wings": "Red Wings",
    "Predateurs": "Prédateurs",
    "Nordiques": "Nordiques",
    "Cracheurs": "Cracheurs",
    "Canadiens": "Canadiens",
}

TEAM_LOGO = {k: os.path.join(DATA_DIR, f"{k}.png") for k in POOL_TEAMS}
APP_LOGO = os.path.join(DATA_DIR, "gm_logo.png")
BANNER = os.path.join(DATA_DIR, "logo_pool.png")  # <- demandé: logo_pool en top

# =========================
# THEME (1 seule injection)
# =========================
THEME_CSS_DARK = """
<style>
:root { color-scheme: dark; }

/* Background (navy, pas noir pur) */
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 800px at 20% -10%, rgba(239,68,68,.18), transparent 55%),
    radial-gradient(1200px 800px at 95% 10%, rgba(80,140,255,.12), transparent 55%),
    linear-gradient(180deg, #0b1220, #070b12 65%, #070b12) !important;
  color: #e7eef7 !important;
}

.block-container { padding-top: .7rem; padding-bottom: 2.5rem; max-width: 1180px; }

/* -------- Sidebar compacte (icônes seulement) -------- */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg,#0d1627,#0b1220) !important;
  border-right: 1px solid rgba(255,255,255,.06);
  width: 86px !important;
  min-width: 86px !important;
}
section[data-testid="stSidebar"] *{
  font-size: 12px;
}

/* Cache les labels et “radio circles” */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{
  margin: .2rem 0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{
  display:none !important;
}

/* Style des boutons icônes */
.pms-sb{
  display:flex; flex-direction:column; align-items:center; gap:.45rem;
  padding: .35rem .25rem .75rem .25rem;
}
.pms-sb-top{
  display:flex; flex-direction:column; align-items:center; gap:.35rem;
  margin-bottom: .3rem;
}
.pms-sb-logo{
  width:44px;height:44px;border-radius:14px; overflow:hidden;
  border:1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.04);
  display:flex;align-items:center;justify-content:center;
}
.pms-sb-btn{
  width:52px; height:52px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.05);
  display:flex; align-items:center; justify-content:center;
  text-decoration:none;
  color:#cfd8e6;
  transition: transform .06s ease;
}
.pms-sb-btn:hover{ transform: translateY(-1px); border-color: rgba(255,255,255,.16); }
.pms-sb-btn.active{
  background: rgba(239,68,68,.90);
  border-color: rgba(239,68,68,.95);
  color: #0b1220;
  box-shadow: 0 0 0 4px rgba(239,68,68,.18);
}
.pms-sb-ico{ font-size: 20px; line-height:1; }

/* Cards */
.pms-card{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px;
  padding: 18px;
}
.smallmuted{ opacity:.72; font-size: 13px; }
hr{ border-color: rgba(255,255,255,.10); }

/* Inputs */
div[data-baseweb="select"] > div{
  border-radius: 12px !important;
}
div[data-baseweb="select"] > div:hover{
  border-color: rgba(255,255,255,.18) !important;
}

/* Primary buttons accent rouge */
.stButton>button[kind="primary"]{
  background: #ef4444 !important;
  border: 1px solid rgba(255,46,77,.55) !important;
  border-radius: 12px !important;
}

/* Banner */
.pms-banner{
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.03);
  box-shadow: 0 12px 34px rgba(0,0,0,.28);
}


/* Radio -> tiles (sidebar icons) */
section[data-testid="stSidebar"] div[role="radiogroup"]{
  display:flex; flex-direction:column; align-items:center; gap:.45rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"]{
  width:52px; height:52px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.05);
  display:flex; align-items:center; justify-content:center;
  margin: 0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"]:hover{
  border-color: rgba(255,255,255,.16);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"] div:nth-child(2){
  font-size: 20px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){
  background: rgba(239,68,68,.90);
  border-color: rgba(239,68,68,.95);
  box-shadow: 0 0 0 4px rgba(239,68,68,.18);
}

</style>
"""

THEME_CSS_LIGHT = """
<style>
:root { color-scheme: light; }

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 700px at 20% -10%, rgba(239,68,68,.14), transparent 55%),
    radial-gradient(1000px 700px at 95% 10%, rgba(60,140,255,.12), transparent 55%),
    linear-gradient(180deg, #f7f8fc, #f3f5fb 60%, #f3f5fb) !important;
  color: #0b1020 !important;
}
.block-container { padding-top: .7rem; padding-bottom: 2.5rem; max-width: 1180px; }

section[data-testid="stSidebar"]{
  background: #ffffff !important;
  border-right: 1px solid rgba(0,0,0,.06);
  width: 86px !important;
  min-width: 86px !important;
}

/* Buttons icônes */
.pms-sb-btn{
  border: 1px solid rgba(0,0,0,.08);
  background: rgba(0,0,0,.03);
  color:#2b3447;
}
.pms-sb-btn.active{
  background: rgba(239,68,68,.92);
  border-color: rgba(239,68,68,.95);
  color:#ffffff;
  box-shadow: 0 0 0 4px rgba(239,68,68,.16);
}

/* Cards */
.pms-card{
  background: #ffffff;
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: 18px;
}
.smallmuted{ opacity:.72; font-size: 13px; }
hr{ border-color: rgba(0,0,0,.10); }

.pms-banner{
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,.08);
  background: #ffffff;
  box-shadow: 0 12px 34px rgba(0,0,0,.10);
}

/* Radio -> tiles (sidebar icons) */
section[data-testid="stSidebar"] div[role="radiogroup"]{
  display:flex; flex-direction:column; align-items:center; gap:.45rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"]{
  width:52px; height:52px;
  border-radius: 16px;
  border: 1px solid rgba(0,0,0,.08);
  background: rgba(0,0,0,.03);
  display:flex; align-items:center; justify-content:center;
  margin: 0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"] div:nth-child(2){
  font-size: 20px !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){
  background: rgba(239,68,68,.92);
  border-color: rgba(239,68,68,.95);
  box-shadow: 0 0 0 4px rgba(239,68,68,.16);
}

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


def _safe_image(path: str, width: int | None = None) -> None:
    try:
        if path and os.path.exists(path):
            st.image(path, width=width)
    except Exception:
        pass


def _is_admin(owner: str) -> bool:
    return (owner or "").strip() == "Whalers"


# =========================
# NAV
# =========================
TABS = [
    ("🏠", "Home", "home"),
    ("👥", "Joueurs", "joueurs"),
    ("🧊", "Alignement", "alignement"),
    ("🔁", "Transactions", "transactions"),
    ("🧑‍💼", "GM", "gm"),
    ("🕘", "Historique", "historique"),
    ("🏆", "Classement", "classement"),
    ("🛠️", "Admin", "admin"),
]


def _sidebar_brand() -> None:
    st.markdown('<div class="pms-sb">', unsafe_allow_html=True)
    st.markdown('<div class="pms-sb-top">', unsafe_allow_html=True)
    if os.path.exists(APP_LOGO):
        st.markdown('<div class="pms-sb-logo">', unsafe_allow_html=True)
        st.image(APP_LOGO, width=44)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pms-sb-logo"></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def sidebar_nav() -> str:
    """Sidebar compacte: uniquement des icônes (radio stylé)."""
    with st.sidebar:
        _sidebar_brand()

        st.session_state.setdefault("active_tab_key", "home")

        # Radio (emoji only) — stable, sans JS
        opt_keys = [k for (_ico, _label, k) in TABS]
        def _fmt(k: str) -> str:
            for ico, _label, kk in TABS:
                if kk == k:
                    return ico
            return "•"

        active_key = st.radio(
            "Navigation",
            options=opt_keys,
            index=opt_keys.index(st.session_state.get("active_tab_key", "home")) if st.session_state.get("active_tab_key") in opt_keys else 0,
            format_func=_fmt,
            key="sb_nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["active_tab_key"] = active_key

        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

        is_light = st.toggle("Mode clair", value=(st.session_state.get("ui_theme", "dark") == "light"), key="ui_theme_toggle_sidebar")
        st.session_state["ui_theme"] = "light" if is_light else "dark"

        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state.get("active_tab_key", "home")


# =========================
# IMPORT TABS
# =========================
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


# =========================
# HEADER CONTROLS (main)
# =========================
def _render_banner_top() -> None:
    if os.path.exists(BANNER):
        st.markdown('<div class="pms-banner">', unsafe_allow_html=True)
        st.image(BANNER, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")


def _render_header_controls() -> None:
    """Saison + Équipe (avec logo) dans le main (pas dans sidebar)."""
    c1, c2, c3, c4 = st.columns([2.2, 2.6, 1.0, 0.6], vertical_alignment="center")
    with c1:
        st.selectbox(
            "Saison",
            options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
            key="season_lbl",
        )
    with c2:
        picked = st.selectbox(
            "Équipe",
            options=POOL_TEAMS,
            key="owner_main",
            format_func=lambda x: TEAM_LABELS.get(x, x),
        )
        st.session_state["owner"] = picked
    with c3:
        # Logo à côté du nom (comme screenshot)
        _safe_image(TEAM_LOGO.get(st.session_state.get("owner", "Whalers"), ""), width=46)
    with c4:
        # Indication “admin” possible, sans bouton (juste look)
        st.markdown("")


# =========================
# HOME
# =========================
def _render_home(ctx: AppCtx) -> None:
    st.title("🏠 Home")
    st.markdown('<div class="smallmuted">Home reste clean — aucun bloc Admin ici.</div>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown('<div class="pms-card">', unsafe_allow_html=True)
    st.markdown("### 🏒 Sélection d'équipe")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        picked = st.selectbox(
            "Équipe (propriétaire)",
            options=POOL_TEAMS,
            index=POOL_TEAMS.index(ctx.owner) if ctx.owner in POOL_TEAMS else 0,
            key="home_owner_select",
            format_func=lambda x: TEAM_LABELS.get(x, x),
        )
        if picked != ctx.owner:
            # Ici on ne touche PAS aux clés de widgets déjà instanciés ailleurs.
            st.session_state["owner"] = picked
            st.session_state["owner_main"] = picked  # synchronise le selectbox header (safe: même run)
            ctx.owner = picked
            st.rerun()

    with c2:
        _safe_image(TEAM_LOGO.get(ctx.owner, ""), width=88)

    st.success(f"✅ Équipe sélectionnée: **{TEAM_LABELS.get(ctx.owner, ctx.owner)}**")
    st.markdown(f"**Saison:** {ctx.season_lbl}")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# MAIN
# =========================
def main() -> None:
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("season_lbl", DEFAULT_SEASON)
    st.session_state.setdefault("owner", "Whalers")
    st.session_state.setdefault("owner_main", st.session_state.get("owner", "Whalers"))

    apply_theme()  # 1 seule injection

    active_key = sidebar_nav()

    # Top banner (logo_pool)
    _render_banner_top()

    # Header controls (season + owner + logo)
    _render_header_controls()
    st.markdown("")

    owner = st.session_state.get("owner") or "Whalers"
    season = st.session_state.get("season_lbl") or DEFAULT_SEASON
    ctx = AppCtx(
        data_dir=DATA_DIR,
        season_lbl=str(season),
        owner=str(owner),
        is_admin=_is_admin(owner),
        theme=st.session_state.get("ui_theme", "dark"),
    )

    modules = _import_tabs()

    try:
        if active_key == "home":
            _render_home(ctx)
        elif active_key == "admin" and not ctx.is_admin:
            st.title("🛠️ Admin")
            st.warning("Accès admin requis.")
        else:
            mod = modules.get(active_key)
            if mod is None:
                st.error(f"Module introuvable: tabs/{active_key}.py")
            else:
                if hasattr(mod, "render"):
                    try:
                        mod.render(ctx.as_dict())
                    except TypeError:
                        mod.render(ctx)
                else:
                    st.error(f"tabs/{active_key}.py n'a pas de fonction render(ctx).")
    except Exception:
        st.error("Une erreur a été détectée (évite l'écran noir).")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
