# app.py — PoolHockeyPMS (routing + thème + contexte) — UI "like screenshot"
# -----------------------------------------------------------------------------
# Objectifs:
# - Look proche de tes screenshots: fond navy, accent rouge, sidebar propre (icônes + hover + actif rouge)
# - Dark par défaut + toggle Mode clair (sidebar)
# - Home: carte "Sélection d'équipe" + logo à droite + bannière logo_pool.png en dessous (proportions stables)
# - Routing stable vers tabs/*.py (render(ctx_dict) ou render(ctx))
# - Zéro expander nested ici (les tabs gèrent leur UI)
# - Zéro StreamlitDuplicateElementKey / et pas de modification de session_state d'un widget après instanciation
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
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
BANNER = os.path.join(DATA_DIR, "logo_pool.png")


# =========================
# THEME (1 seule injection)
# =========================
THEME_CSS_DARK = """
<style>
:root{ color-scheme: dark; }
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 760px at 18% -10%, rgba(239,68,68,.18), transparent 58%),
    radial-gradient(1050px 740px at 96% 10%, rgba(59,130,246,.14), transparent 58%),
    linear-gradient(180deg,#0b1220,#070b12 65%,#070b12) !important;
  color:#e7eef7 !important;
}
.block-container{ padding-top: 1.1rem; padding-bottom: 2.6rem; max-width: 1200px; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg,#0d1627,#0b1220) !important;
  border-right: 1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] .block-container{ padding-top: 1.1rem; }

/* Brand header in sidebar */
.pms-sidebrand{
  display:flex; align-items:center; gap:.65rem;
  margin-bottom: .85rem;
}
.pms-sidebrand img{ width: 44px; height: 44px; border-radius: 12px; }
.pms-sidebrand .t1{ font-weight: 800; font-size: 20px; letter-spacing:-.02em; }
.pms-sidebrand .t2{ opacity:.75; font-size: 12px; margin-top: -2px; }

/* ---- Cards ---- */
.card{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 12px 34px rgba(0,0,0,.22);
}
.smallmuted{ opacity:.72; font-size: 13px; }
hr{ border-color: rgba(255,255,255,.10); }

/* ---- Accent red (buttons) ---- */
.stButton>button{ border-radius: 12px !important; }
.stButton>button[kind="primary"]{
  background:#ef4444 !important;
  border:1px solid rgba(239,68,68,.55) !important;
}
.stButton>button[kind="primary"]:hover{ filter: brightness(1.02); }

/* ---- Sidebar Navigation RADIO styling (match screenshot) ---- */
section[data-testid="stSidebar"] div[role="radiogroup"]{ gap:.35rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label{
  padding: .65rem .75rem !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  background: rgba(255,255,255,.02) !important;
  margin: 0 !important;
  transition: all .14s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  border-color: rgba(255,255,255,.14) !important;
  background: rgba(255,255,255,.04) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p{
  color: #dbe5f5 !important;
  font-size: 15px !important;
  font-weight: 650 !important;
  margin: 0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] input{ display:none !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
  background: rgba(239,68,68,.18) !important;
  border-color: rgba(239,68,68,.55) !important;
  box-shadow: 0 0 0 3px rgba(239,68,68,.16) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{
  color: #ffffff !important;
}

/* Hide Streamlit's tiny "radio dot" container */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{
  display:none !important;
}

/* Inputs look */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]{
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] .stToggleSwitch{
  margin-top: .6rem;
}

/* Banner rounding */
.pms-banner img{ border-radius: 18px; box-shadow: 0 12px 34px rgba(0,0,0,.20); }
</style>
"""

THEME_CSS_LIGHT = """
<style>
:root{ color-scheme: light; }
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 720px at 18% -10%, rgba(239,68,68,.14), transparent 58%),
    radial-gradient(1000px 720px at 96% 10%, rgba(59,130,246,.14), transparent 58%),
    linear-gradient(180deg,#f7f8fc,#f2f5fb 60%,#f2f5fb) !important;
  color:#0b1020 !important;
}
.block-container{ padding-top: 1.1rem; padding-bottom: 2.6rem; max-width: 1200px; }

section[data-testid="stSidebar"]{
  background: #ffffff !important;
  border-right: 1px solid rgba(0,0,0,.06);
}
.pms-sidebrand .t2{ color: rgba(0,0,0,.55); }

.card{
  background: #ffffff;
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 12px 34px rgba(0,0,0,.08);
}
.smallmuted{ opacity:.72; font-size: 13px; }
hr{ border-color: rgba(0,0,0,.10); }

.stButton>button[kind="primary"]{
  background:#ef4444 !important;
  border:1px solid rgba(239,68,68,.35) !important;
}

/* Sidebar nav radio */
section[data-testid="stSidebar"] div[role="radiogroup"] label{
  padding: .65rem .75rem !important;
  border-radius: 12px !important;
  border: 1px solid rgba(0,0,0,.08) !important;
  background: rgba(0,0,0,.02) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  border-color: rgba(0,0,0,.12) !important;
  background: rgba(0,0,0,.03) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p{
  color: #1f2937 !important;
  font-size: 15px !important;
  font-weight: 650 !important;
  margin: 0 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] input{ display:none !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
  background: rgba(239,68,68,.12) !important;
  border-color: rgba(239,68,68,.45) !important;
  box-shadow: 0 0 0 3px rgba(239,68,68,.14) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{
  color: #111827 !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{
  display:none !important;
}

.pms-banner img{ border-radius: 18px; box-shadow: 0 12px 34px rgba(0,0,0,.08); }
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


def _safe_image(path: str, width: Optional[int] = None) -> None:
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
    ("🏠  Home", "home"),
    ("👥  Joueurs", "joueurs"),
    ("🧊  Alignement", "alignement"),
    ("🔁  Transactions", "transactions"),
    ("🧑‍💼  GM", "gm"),
    ("🕘  Historique", "historique"),
    ("🏆  Classement", "classement"),
    ("🛠️  Admin", "admin"),
]


def _queue_owner_sync(new_owner: str) -> None:
    """Home -> Sidebar sync sans toucher owner_select après instanciation: on planifie + rerun."""
    new_owner = str(new_owner or "").strip()
    if not new_owner:
        return
    st.session_state["owner"] = new_owner
    st.session_state["_pending_owner_select"] = new_owner
    st.rerun()


def _apply_pending_widget_values_before_widgets() -> None:
    """Applique les 'pending' AVANT la création des widgets (sinon StreamlitAPIException)."""
    if st.session_state.get("_pending_owner_select"):
        st.session_state["owner_select"] = st.session_state["_pending_owner_select"]
        st.session_state["home_owner_select"] = st.session_state["_pending_owner_select"]
        st.session_state.pop("_pending_owner_select", None)


def _sidebar_brand() -> None:
    with st.sidebar:
        c1, c2 = st.columns([1, 3], gap="small")
        with c1:
            if os.path.exists(APP_LOGO):
                st.image(APP_LOGO, width=46)
            else:
                st.markdown(
                    '<div style="width:46px;height:46px;border-radius:12px;background:rgba(239,68,68,.25)"></div>',
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown(
                """
                <div class="pms-sidebrand">
                  <div>
                    <div class="t1">Pool GM</div>
                    <div class="t2">Gestion du pool (PMS)</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def sidebar_nav() -> str:
    with st.sidebar:
        _sidebar_brand()

        # Saison
        st.selectbox(
            "Saison",
            options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
            key="season_lbl",
        )

        st.markdown("### Navigation")

        labels = [t[0] for t in TABS]
        cur = st.session_state.get("active_tab", labels[0])
        default_idx = labels.index(cur) if cur in labels else 0

        active = st.radio(
            "Navigation",
            labels,
            index=default_idx,
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["active_tab"] = active

        st.markdown("---")

        # Mon équipe (avec logo à droite)
        st.markdown("### Mon équipe")
        c1, c2 = st.columns([4, 1], gap="small")
        with c1:
            st.selectbox(
                "Mon équipe",
                options=POOL_TEAMS,
                key="owner_select",
                format_func=lambda x: TEAM_LABELS.get(x, x),
                label_visibility="collapsed",
            )
        with c2:
            _safe_image(TEAM_LOGO.get(st.session_state.get("owner_select", "Whalers"), ""), width=34)

        # Canonical owner value (no mutation of owner_select here)
        st.session_state["owner"] = st.session_state.get("owner_select", "Whalers")

        # Theme toggle (only here)
        is_light = st.toggle(
            "Mode clair",
            value=(st.session_state.get("ui_theme", "dark") == "light"),
            key="ui_theme_toggle_sidebar",
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
def _render_home(ctx: AppCtx) -> None:
    st.title("🏠 Home")
    st.markdown('<div class="smallmuted">Home reste clean — aucun bloc Admin ici.</div>', unsafe_allow_html=True)
    st.markdown("")

    # Card selection (same layout as screenshot)
    st.markdown('<div class="card">', unsafe_allow_html=True)
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
        # Home -> sidebar sync via pending + rerun (safe)
        if picked != ctx.owner:
            _queue_owner_sync(picked)

    with c2:
        logo = TEAM_LOGO.get(ctx.owner, "")
        if logo and os.path.exists(logo):
            st.image(logo, width=92)

    st.success(f"✅ Équipe sélectionnée: **{TEAM_LABELS.get(ctx.owner, ctx.owner)}**")
    st.markdown(f"**Saison:** {ctx.season_lbl}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # Banner below card (proportions)
    if os.path.exists(BANNER):
        st.markdown('<div class="pms-banner">', unsafe_allow_html=True)
        st.image(BANNER, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# MAIN
# =========================
def main() -> None:
    # Defaults
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("season_lbl", DEFAULT_SEASON)
    st.session_state.setdefault("active_tab", "🏠  Home")
    st.session_state.setdefault("owner", "Whalers")
    st.session_state.setdefault("owner_select", st.session_state.get("owner", "Whalers"))
    st.session_state.setdefault("home_owner_select", st.session_state.get("owner", "Whalers"))

    # Apply any pending widget sync BEFORE widgets are instantiated
    _apply_pending_widget_values_before_widgets()

    # Theme injection (once)
    apply_theme()

    # Sidebar
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

    # Route
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
