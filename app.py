# app.py — PoolHockeyPMS (routing + thème + contexte) — UI v2
# ------------------------------------------------------------
# Objectif: look "app" (accent rouge), dark par défaut, toggle clair,
# sidebar collapsée (hamburger), proportions d'images stables (logo_pool),
# + plus de warnings "label got an empty value".
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
    initial_sidebar_state="collapsed",  # hamburger visible par défaut (petites fenêtres)
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_SEASON = "2025-2026"

POOL_TEAMS = ["Whalers", "Red_Wings", "Predateurs", "Nordiques", "Cracheurs", "Canadiens"]

TEAM_LOGO = {
    "Whalers": os.path.join(DATA_DIR, "Whalers.png"),
    "Red_Wings": os.path.join(DATA_DIR, "Red_Wings.png"),
    "Predateurs": os.path.join(DATA_DIR, "Predateurs.png"),
    "Nordiques": os.path.join(DATA_DIR, "Nordiques.png"),
    "Cracheurs": os.path.join(DATA_DIR, "Cracheurs.png"),
    "Canadiens": os.path.join(DATA_DIR, "Canadiens.png"),
}

APP_LOGO = os.path.join(DATA_DIR, "gm_logo.png")
BANNER = os.path.join(DATA_DIR, "logo_pool.png")

# =========================
# THEME (1 seule injection)
# =========================

ACCENT = "#ff3b4d"  # rouge comme tes screenshots

THEME_CSS = f"""
<style>
/* --- Design tokens --- */
:root {{
  --accent: {ACCENT};
  --bg0: #070a12;
  --bg1: #0b1220;
  --bg2: #0f172a;
  --surface: rgba(255,255,255,.04);
  --surface2: rgba(255,255,255,.06);
  --border: rgba(255,255,255,.10);
  --text: #e7eef7;
  --muted: rgba(231,238,247,.72);
  --shadow: 0 18px 40px rgba(0,0,0,.35);
  --radius: 18px;
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 600px at 30% 10%, rgba(255,59,77,.12), transparent 60%),
              radial-gradient(900px 500px at 80% 0%, rgba(69,120,255,.10), transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1) 35%, #060812 100%) !important;
  color: var(--text) !important;
}}

.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 2.2rem;
  max-width: 1180px;
}}

h1,h2,h3 {{
  letter-spacing: -0.02em;
}}

.smallmuted {{ color: var(--muted); font-size: 13px; }}

/* --- Sidebar --- */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0b1220, #0a1020) !important;
  border-right: 1px solid rgba(255,255,255,.06);
}}
[data-testid="stSidebar"] .stSelectbox, 
[data-testid="stSidebar"] .stRadio, 
[data-testid="stSidebar"] .stToggle {{
  padding-top: 0.25rem;
}}
/* Radio styling */
[data-testid="stSidebar"] [role="radiogroup"] > label {{
  margin: 0.15rem 0;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label > div {{
  border-radius: 12px !important;
  padding: 0.55rem 0.75rem !important;
  border: 1px solid transparent;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) > div {{
  background: rgba(255,59,77,.18) !important;
  border-color: rgba(255,59,77,.35) !important;
}}
/* Make radio "dot" less visible */
[data-testid="stSidebar"] input[type="radio"] {{
  transform: scale(0.85);
}}

/* --- Hamburger (sidebar control) --- */
header {{
  background: transparent !important;
}}
/* Keep the collapse button visible and "app-like" */
button[kind="headerNoPadding"] {{
  border-radius: 12px !important;
}}
button[kind="headerNoPadding"]:hover {{
  background: rgba(255,255,255,.06) !important;
}}

/* --- Cards --- */
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.03));
  border: 1px solid rgba(255,255,255,.08);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px 18px;
}}

.hr {{
  height: 1px;
  background: rgba(255,255,255,.10);
  margin: 0.8rem 0;
}}

/* Success callout */
.callout {{
  background: rgba(34,197,94,.12);
  border: 1px solid rgba(34,197,94,.35);
  border-radius: 14px;
  padding: 12px 14px;
  display:flex;
  gap:10px;
  align-items:center;
}}
.callout .dot {{
  width: 12px; height: 12px; border-radius: 999px;
  background: rgba(34,197,94,.85);
  box-shadow: 0 0 0 6px rgba(34,197,94,.15);
}}
/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
  border-radius: 12px !important;
}}
</style>
"""

THEME_CSS_LIGHT = f"""
<style>
:root {{
  --accent: {ACCENT};
  --bg: #f5f7fb;
  --surface: #ffffff;
  --border: rgba(10, 20, 40, .10);
  --text: #0b1020;
  --muted: rgba(20,30,55,.70);
  --shadow: 0 14px 34px rgba(16,24,40,.10);
  --radius: 18px;
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(900px 500px at 25% 0%, rgba(255,59,77,.10), transparent 55%),
              linear-gradient(180deg, #f8fafc, #f3f4f6 100%) !important;
  color: var(--text) !important;
}}
.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 2.2rem;
  max-width: 1180px;
}}
.smallmuted {{ color: var(--muted); font-size: 13px; }}
[data-testid="stSidebar"] {{
  background: #ffffff !important;
  border-right: 1px solid rgba(10,20,40,.08);
}}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px;
}}
.hr {{
  height: 1px;
  background: rgba(10,20,40,.10);
  margin: 0.8rem 0;
}}
.callout {{
  background: rgba(34,197,94,.10);
  border: 1px solid rgba(34,197,94,.25);
  border-radius: 14px;
  padding: 12px 14px;
  display:flex;
  gap:10px;
  align-items:center;
}}
.callout .dot {{
  width: 12px; height: 12px; border-radius: 999px;
  background: rgba(34,197,94,.85);
  box-shadow: 0 0 0 6px rgba(34,197,94,.12);
}}
</style>
"""


def apply_theme() -> None:
    """Une seule injection CSS par run."""
    mode = st.session_state.get("ui_theme", "dark")
    st.markdown(THEME_CSS_LIGHT if mode == "light" else THEME_CSS, unsafe_allow_html=True)


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


def _safe_image(path: str, width: int | None = None, use_container_width: bool = False) -> None:
    try:
        if path and os.path.exists(path):
            st.image(path, width=width, use_container_width=use_container_width)
    except Exception:
        pass


def _is_admin(owner: str) -> bool:
    return (owner or "").strip() == "Whalers"


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


def sidebar_nav() -> str:
    with st.sidebar:
        # Header sidebar
        cols = st.columns([1, 5])
        with cols[0]:
            _safe_image(APP_LOGO, width=44)
        with cols[1]:
            st.markdown("### Pool GM")

        # Saison
        st.selectbox(
            "Saison",
            options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
            key="season_lbl",
        )

        st.markdown("")

        # IMPORTANT: label non-vide (plus de warning), mais caché
        labels = [t[0] for t in TABS]
        default_idx = labels.index(st.session_state.get("active_tab", "🏠 Home")) if st.session_state.get("active_tab") in labels else 0
        active = st.radio(
            "Navigation",
            labels,
            index=default_idx,
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["active_tab"] = active

        st.markdown("---")

        # Mon équipe (owner)
        st.selectbox("Mon équipe", options=POOL_TEAMS, key="owner_select")
        st.session_state["owner"] = st.session_state.get("owner_select") or st.session_state.get("owner") or "Whalers"
        st.session_state["home_owner_select"] = st.session_state["owner"]

        # Theme switch
        is_light = st.toggle(
            "Mode clair",
            value=(st.session_state.get("ui_theme", "dark") == "light"),
            key="ui_theme_toggle",
        )
        st.session_state["ui_theme"] = "light" if is_light else "dark"

    return active


# =========================
# RENDERERS IMPORT
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
# HOME
# =========================
def _render_home(ctx: AppCtx):
    # logo_pool complètement en haut (avant le titre)
    if os.path.exists(BANNER):
        # Stabilise les proportions: max-width + centré + pas gigantesque
        st.markdown(
            """
            <div style="width:100%; display:flex; justify-content:center; margin: 6px 0 14px 0;">
              <div style="max-width: 980px; width: 100%;">
            """,
            unsafe_allow_html=True,
        )
        _safe_image(BANNER, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("# 🏠 Home")
    st.markdown('<div class="smallmuted">Home reste clean — aucun bloc Admin ici.</div>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🏒 Sélection d'équipe")
        st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

        team = ctx.owner
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Widget selection local (home) = même valeur session_state "owner_select"
        # Label non vide (no warning)
        st.session_state.setdefault("home_owner_select", team)

        st.selectbox(
            "Équipe (propriétaire)",
            options=POOL_TEAMS,
            key="home_owner_select",
        )
        st.session_state["owner"] = st.session_state.get("home_owner_select") or team
        st.session_state["owner_select"] = st.session_state["owner"]

        st.markdown(
            f"""
            <div class="callout" style="margin-top:12px;">
              <div class="dot"></div>
              <div><b>Équipe sélectionnée:</b> {st.session_state['owner']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.write(f"**Saison:** {ctx.season_lbl}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        logo = TEAM_LOGO.get(st.session_state.get("owner") or ctx.owner, "")
        _safe_image(logo, width=140)


# =========================
# MAIN
# =========================
def main() -> None:
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("owner", "Whalers")
    st.session_state.setdefault("owner_select", "Whalers")
    st.session_state.setdefault("home_owner_select", st.session_state.get("owner_select") or st.session_state.get("owner") or "Whalers")
    st.session_state.setdefault("season_lbl", DEFAULT_SEASON)
    st.session_state.setdefault("active_tab", "🏠 Home")

    apply_theme()

    active_label = sidebar_nav()

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
