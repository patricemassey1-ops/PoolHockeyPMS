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
st.set_page_config(page_title="Pool GM", page_icon="🏒", layout="wide")

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


# =========================
# THEME (1 seule injection)
# =========================
THEME_CSS_DARK = """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stAppViewContainer"] {
  background: #0b0f14 !important;
  color: #e7eef7 !important;
}
.block-container { padding-top: 1.2rem; }
h1,h2,h3 { letter-spacing: -0.02em; }
[data-testid="stSidebar"] {
  background: #0e131a !important;
  border-right: 1px solid rgba(255,255,255,.06);
}
.card {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 18px;
}
.pill {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.04);
  font-size: 12px;
}
.navbtn button {
  width: 100% !important;
  text-align: left !important;
  border-radius: 12px !important;
}
.smallmuted { opacity:.72; font-size: 13px; }
hr { border-color: rgba(255,255,255,.10); }
</style>
"""

THEME_CSS_LIGHT = """
<style>
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"] {
  background: #f6f7fb !important;
  color: #0b1020 !important;
}
.block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid rgba(0,0,0,.06);
}
.card {
  background: #ffffff;
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 16px;
  padding: 18px;
}
.pill {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(0,0,0,.10);
  background: rgba(0,0,0,.03);
  font-size: 12px;
}
.navbtn button {
  width: 100% !important;
  text-align: left !important;
  border-radius: 12px !important;
}
.smallmuted { opacity:.72; font-size: 13px; }
hr { border-color: rgba(0,0,0,.10); }
</style>
"""


def apply_theme() -> None:
    """Une seule injection CSS par run."""
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
    # Règle actuelle: Whalers = admin (comme tu voulais)
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
        st.markdown("### Pool GM")

        _safe_image(APP_LOGO, width=56)

        # Saison
        season = st.selectbox(
            "Saison",
            options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
            key="season_lbl",
        )

        st.markdown("#### Navigation")

        # IMPORTANT: un seul widget de nav (radio)
        labels = [t[0] for t in TABS]
        default_idx = labels.index(st.session_state.get("active_tab", "🏠 Home")) if st.session_state.get("active_tab") in labels else 0
        active = st.radio("", labels, index=default_idx, key="nav_radio", label_visibility="collapsed")
        st.session_state["active_tab"] = active

        st.markdown("---")

        # Mon équipe (owner) — sert partout
        owner = st.selectbox("Mon équipe", options=POOL_TEAMS, key="owner_select")
        st.session_state["owner"] = owner

        # Theme switch
        is_light = st.toggle("Mode clair", value=(st.session_state.get("ui_theme", "dark") == "light"), key="ui_theme_toggle")
        st.session_state["ui_theme"] = "light" if is_light else "dark"

    return active


# =========================
# RENDERERS IMPORT
# =========================
def _import_tabs():
    """
    Import des modules tabs/*.py.
    Si un module manque, on affiche une erreur claire au lieu d'écran noir.
    """
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
    except Exception as e:
        st.error("Impossible d'importer les modules dans /tabs/.")
        st.code(traceback.format_exc())
        st.stop()


def _render_home(ctx: AppCtx):
    st.title("🏠 Home")
    st.markdown('<div class="smallmuted">Home reste clean — aucun bloc Admin ici.</div>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🏒 Sélection d'équipe")
        st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")
        st.write(f"**Équipe sélectionnée:** {ctx.owner}")
        st.write(f"**Saison:** {ctx.season_lbl}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        logo = TEAM_LOGO.get(ctx.owner, "")
        _safe_image(logo, width=140)

    if os.path.exists(BANNER):
        st.markdown("")
        _safe_image(BANNER, width=None)


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
            st.title("🛠️ Admin")
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
