# app.py — PoolHockeyPMS (routing + thème + contexte)
# ------------------------------------------------------------
# Objectif: un app.py STABLE (pas d’expander nested, pas de rerun bizarre),
# avec routing propre vers /tabs/*.py, et switch sombre/clair.
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
    initial_sidebar_state="collapsed",  # ✅ hamburger / sidebar collapsée par défaut (meilleur mobile)
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_SEASON = "2025-2026"

# IMPORTANT: on garde une seule normalisation "owner" pour toute l'app
POOL_TEAMS = [
    "Whalers",
    "Red_Wings",
    "Predateurs",
    "Nordiques",
    "Cracheurs",
    "Canadiens",
]

# Mapping logos (optionnel)
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

ACCENT = "#ff3b4d"  # ✅ rouge comme tes screenshots


# =========================
# THEME (1 seule injection)
# =========================
THEME_CSS = f"""
<style>
/* ---------------------------------
   Design system (Dark/Light)
---------------------------------- */
:root {{
  --accent: {ACCENT};
  --radius: 18px;
  --border: rgba(15, 23, 42, .12);
  --shadow: 0 14px 40px rgba(2, 6, 23, .12);
  --shadow2: 0 10px 30px rgba(2, 6, 23, .10);

  --bg: #0b0f14;
  --panel: rgba(255,255,255,.04);
  --panel2: rgba(255,255,255,.06);
  --text: #e7eef7;
  --muted: rgba(231,238,247,.72);
  --line: rgba(255,255,255,.10);

  --side-bg: #0e131a;
  --side-line: rgba(255,255,255,.08);
  --side-item: transparent;
  --side-item-hover: rgba(255,255,255,.06);
  --side-active: color-mix(in srgb, var(--accent) 92%, #ffffff 8%);
  --side-active-text: #ffffff;
}}

:root[data-theme="light"] {{
  --bg: #f6f7fb;
  --panel: #ffffff;
  --panel2: rgba(2, 6, 23, .03);
  --text: #0b1020;
  --muted: rgba(11,16,32,.68);
  --line: rgba(2,6,23,.10);

  --side-bg: #ffffff;
  --side-line: rgba(2,6,23,.08);
  --side-item: transparent;
  --side-item-hover: rgba(2,6,23,.04);
  --side-active: color-mix(in srgb, var(--accent) 88%, #ffffff 12%);
  --side-active-text: #ffffff;
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: var(--bg) !important;
  color: var(--text) !important;
}}

.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 1.5rem;
  max-width: 1180px;
}}

h1,h2,h3 {{
  letter-spacing: -0.02em;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: var(--side-bg) !important;
  border-right: 1px solid var(--side-line);
}}
[data-testid="stSidebar"] .stSelectbox > div, 
[data-testid="stSidebar"] .stRadio > div {{
  background: transparent !important;
}}

/* ✅ Hamburger toujours visible et plus “3 lignes” */
[data-testid="stSidebarCollapsedControl"] {{
  position: fixed !important;
  top: 10px !important;
  left: 12px !important;
  z-index: 1000 !important;
  border-radius: 12px !important;
  background: color-mix(in srgb, var(--panel) 70%, transparent) !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--shadow2) !important;
}}
[data-testid="stSidebarCollapsedControl"] button {{
  width: 42px !important;
  height: 42px !important;
  padding: 0 !important;
}}
/* Le petit chevron Streamlit devient discret */
[data-testid="stSidebarCollapsedControl"] svg {{
  transform: scale(1.05);
}}

/* Cards */
.card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 18px 18px;
  box-shadow: var(--shadow);
}}
.card h3 {{
  margin-top: 0.1rem;
}}
.smallmuted {{
  color: var(--muted);
  font-size: 13px;
}}
hr {{
  border-color: var(--line);
}}

/* Boutons - accent */
.stButton > button {{
  border-radius: 14px !important;
}}
.stButton > button:focus {{
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 30%, transparent) !important;
}}
/* Radio nav: look “pill” */
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 8px 10px !important;
  border-radius: 14px !important;
  margin-bottom: 6px !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
  background: var(--side-item-hover) !important;
}}
/* Approx : Streamlit marque la sélection via input checked + sibling */
[data-testid="stSidebar"] [role="radiogroup"] input:checked + div {{
  background: var(--side-active) !important;
  border-radius: 14px !important;
}}
/* Text dans item actif */
[data-testid="stSidebar"] [role="radiogroup"] input:checked + div * {{
  color: var(--side-active-text) !important;
}}

/* Selects */
.stSelectbox > div > div {{
  border-radius: 14px !important;
}}
</style>
"""


def apply_theme() -> None:
    """Une seule injection CSS par run."""
    mode = st.session_state.get("ui_theme", "dark")
    # hack: on bascule variables via attribut data-theme sur root
    st.markdown(
        f"""
        <script>
        document.documentElement.setAttribute('data-theme', '{'light' if mode=='light' else 'dark'}');
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)


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
    # Règle actuelle: Whalers = admin
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
        # Brand row
        top = st.columns([1, 4])
        with top[0]:
            _safe_image(APP_LOGO, width=44)
        with top[1]:
            st.markdown("### Pool GM")

        # Saison
        st.caption("Saison")
        season = st.selectbox(
            "",
            options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
            key="season_lbl",
            label_visibility="collapsed",
        )

        st.markdown("")

        # Navigation
        st.caption("Navigation")
        labels = [t[0] for t in TABS]
        default_idx = labels.index(st.session_state.get("active_tab", "🏠 Home")) if st.session_state.get("active_tab") in labels else 0
        active = st.radio("", labels, index=default_idx, key="nav_radio", label_visibility="collapsed")
        st.session_state["active_tab"] = active

        st.markdown("---")

        # Mon équipe (owner)
        st.caption("Mon équipe")
        owner = st.selectbox("", options=POOL_TEAMS, key="owner_select", label_visibility="collapsed")
        st.session_state["owner"] = owner

        # Theme switch
        is_light = st.toggle("Mode clair", value=(st.session_state.get("ui_theme", "dark") == "light"), key="ui_theme_toggle")
        st.session_state["ui_theme"] = "light" if is_light else "dark"

    return active


# =========================
# RENDERERS IMPORT
# =========================
def _import_tabs():
    """Import des modules tabs/*.py."""
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


def _render_home(ctx: AppCtx):
    # ✅ logo_pool complètement en haut du contenu (avant tout)
    if os.path.exists(BANNER):
        _safe_image(BANNER, width=None)
        st.markdown("")

    st.markdown("# 🏠 Home")
    st.markdown('<div class="smallmuted">Home reste clean — aucun bloc Admin ici.</div>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🏒 Sélection d'équipe")
        st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")
        st.write(f"**Équipe sélectionnée:** {ctx.owner}")
        st.write(f"**Saison:** {ctx.season_lbl}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        logo = TEAM_LOGO.get(ctx.owner, "")
        _safe_image(logo, width=120)


def main() -> None:
    # init session defaults
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("owner", "Whalers")
    st.session_state.setdefault("season_lbl", DEFAULT_SEASON)
    st.session_state.setdefault("active_tab", "🏠 Home")

    # Theme injection (UNE fois)
    apply_theme()

    # Nav
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

    # Tabs modules
    modules = _import_tabs()

    # Route
    key = dict(TABS).get(active_label, "home")

    try:
        if key == "home":
            _render_home(ctx)
        elif key == "admin" and not ctx.is_admin:
            st.markdown("# 🛠️ Admin")
            st.warning("Accès admin requis.")
        else:
            mod = modules.get(key)
            if mod is None:
                st.error(f"Module introuvable: tabs/{key}.py")
            else:
                # Convention: render(ctx_dict) OR render(ctx)
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
