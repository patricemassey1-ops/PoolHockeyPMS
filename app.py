
# app.py — PoolHockeyPMS (routing + thème + contexte) — UI v4 (top bar + fix keys)
# ------------------------------------------------------------
# - Accent rouge + fond navy (dark default) + option light
# - Top bar (logo + titre + toggle + "menu" hint)
# - Fix StreamlitDuplicateElementKey: owner_select unique (sidebar) + home_owner_select (home)
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
    initial_sidebar_state="expanded",  # users can collapse via built-in control
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_SEASON = "2025-2026"

POOL_TEAMS = [
    "Whalers",
    "Red_Wings",
    "Predateurs",
    "Nordiques",
    "Cracheurs",
    "Canadiens",
]

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

/* --- Background (not pure black) --- */
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 800px at 20% -10%, rgba(255,0,64,.20), transparent 55%),
    radial-gradient(1200px 800px at 95% 10%, rgba(80,140,255,.12), transparent 55%),
    linear-gradient(180deg, #0b1220, #070b12 65%, #070b12) !important;
  color: #e7eef7 !important;
}

.block-container { padding-top: .8rem; padding-bottom: 4.2rem; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg,#0d1627,#0b1220) !important;
  border-right: 1px solid rgba(255,255,255,.06);
}

/* --- Top bar --- */
.pms-topbar{
  position: sticky;
  top: 0;
  z-index: 50;
  padding: .6rem .75rem;
  margin: -0.5rem -0.5rem 0.75rem -0.5rem;
  border-radius: 16px;
  backdrop-filter: blur(10px);
  background: rgba(10,16,28,.55);
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
.pms-brand{
  display:flex; align-items:center; gap:.6rem;
  font-weight: 800; letter-spacing:-.02em;
}
.pms-brand img{ width:34px; height:34px; border-radius:10px; }
.pms-right{
  display:flex; align-items:center; gap:.6rem; justify-content:flex-end;
}
.pms-chip{
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.35rem .55rem;
  border-radius: 999px;
  border:1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.05);
  font-size: 12px;
  opacity:.92;
}
.pms-hamburger{
  width:36px; height:36px;
  display:inline-flex; align-items:center; justify-content:center;
  border-radius: 12px;
  border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.05);
  font-size: 16px;
  line-height: 1;
}

/* Cards */
.card{
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px;
  padding: 18px;
}
.smallmuted{ opacity:.72; font-size: 13px; }
hr{ border-color: rgba(255,255,255,.10); }

/* Accent red for selected radio option */
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child{
  border-color: rgba(255,255,255,.18);
}
div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div{
  border-color: rgba(255,60,90,.95) !important;
  box-shadow: 0 0 0 3px rgba(255,60,90,.16);
}

/* Buttons */
.stButton>button{
  border-radius: 12px !important;
}
.stButton>button[kind="primary"]{
  background: #ef4444 !important;
  border: 1px solid rgba(255,46,77,.55) !important;
}
.stButton>button[kind="primary"]:hover{
  filter: brightness(1.03);
}

/* Bottom nav (mobile only) */
@media (max-width: 900px){
  .block-container{ padding-bottom: 5.4rem; }
  .pms-bottomnav{
    position: fixed; left: 0; right: 0; bottom: 0;
    z-index: 80;
    padding: .55rem .75rem;
    background: rgba(10,16,28,.70);
    backdrop-filter: blur(12px);
    border-top: 1px solid rgba(255,255,255,.10);
  }
  .pms-bottomnav a{
    text-decoration:none; color:#cfd8e6;
    font-size: 12px;
    display:flex; flex-direction:column; align-items:center; gap:.15rem;
  }
  .pms-bottomnav a.active{ color:#ef4444; }
  .pms-bottomnav .row{
    display:flex; justify-content:space-around; gap:.25rem;
  }
}
</style>
"""

THEME_CSS_LIGHT = """
<style>
:root { color-scheme: light; }

html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1000px 700px at 20% -10%, rgba(255,60,90,.14), transparent 55%),
    radial-gradient(1000px 700px at 95% 10%, rgba(60,140,255,.12), transparent 55%),
    linear-gradient(180deg, #f7f8fc, #f3f5fb 60%, #f3f5fb) !important;
  color: #0b1020 !important;
}

.block-container { padding-top: .8rem; padding-bottom: 4.2rem; }

section[data-testid="stSidebar"]{
  background: #ffffff !important;
  border-right: 1px solid rgba(0,0,0,.06);
}

.pms-topbar{
  position: sticky;
  top: 0;
  z-index: 50;
  padding: .6rem .75rem;
  margin: -0.5rem -0.5rem 0.75rem -0.5rem;
  border-radius: 16px;
  backdrop-filter: blur(10px);
  background: rgba(255,255,255,.72);
  border: 1px solid rgba(0,0,0,.06);
  box-shadow: 0 10px 30px rgba(0,0,0,.08);
}
.pms-brand{
  display:flex; align-items:center; gap:.6rem;
  font-weight: 800; letter-spacing:-.02em;
}
.pms-brand img{ width:34px; height:34px; border-radius:10px; }
.pms-right{
  display:flex; align-items:center; gap:.6rem; justify-content:flex-end;
}
.pms-chip{
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.35rem .55rem;
  border-radius: 999px;
  border:1px solid rgba(0,0,0,.08);
  background: rgba(0,0,0,.03);
  font-size: 12px;
  opacity:.85;
}
.pms-hamburger{
  width:36px; height:36px;
  display:inline-flex; align-items:center; justify-content:center;
  border-radius: 12px;
  border:1px solid rgba(0,0,0,.10);
  background: rgba(0,0,0,.03);
  font-size: 16px;
  line-height: 1;
}

.card{
  background: #ffffff;
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: 18px;
}
.smallmuted{ opacity:.72; font-size: 13px; }
hr{ border-color: rgba(0,0,0,.10); }

/* Accent red */
.stButton>button[kind="primary"]{
  background: #ef4444 !important;
  border: 1px solid rgba(255,46,77,.45) !important;
}

/* Bottom nav (mobile only) */
@media (max-width: 900px){
  .block-container{ padding-bottom: 5.4rem; }
  .pms-bottomnav{
    position: fixed; left: 0; right: 0; bottom: 0;
    z-index: 80;
    padding: .55rem .75rem;
    background: rgba(255,255,255,.80);
    backdrop-filter: blur(12px);
    border-top: 1px solid rgba(0,0,0,.08);
  }
  .pms-bottomnav a{
    text-decoration:none; color:#3a4252;
    font-size: 12px;
    display:flex; flex-direction:column; align-items:center; gap:.15rem;
  }
  .pms-bottomnav a.active{ color:#ef4444; }
  .pms-bottomnav .row{
    display:flex; justify-content:space-around; gap:.25rem;
  }
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
    ("🏠 Home", "home"),
    ("👥 Joueurs", "joueurs"),
    ("🧊 Alignement", "alignement"),
    ("🔁 Transactions", "transactions"),
    ("🧑‍💼 GM", "gm"),
    ("🕘 Historique", "historique"),
    ("🏆 Classement", "classement"),
    ("🛠️ Admin", "admin"),
]


def _render_topbar(active_label: str) -> None:
    # Pure HTML/CSS header; Streamlit widgets are placed just under via columns
    logo_ok = APP_LOGO if os.path.exists(APP_LOGO) else ""
    logo_html = f'<img src="data:image/png;base64,{_img_to_b64(logo_ok)}"/>' if logo_ok else '<div style="width:34px;height:34px;border-radius:10px;background:rgba(255,46,77,.25)"></div>'
    st.markdown(
        f"""
        <div class="pms-topbar">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;">
            <div class="pms-brand">
              {logo_html}
              <div>Pool GM</div>
              <span class="pms-chip">📍 {active_label}</span>
            </div>
            <div class="pms-right">
              <span class="pms-chip">Theme</span>
              <span class="pms-hamburger" title="Menu: utilise la flèche de la sidebar (en haut à gauche) sur petits écrans">☰</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _img_to_b64(path: str) -> str:
    import base64
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("### Pool GM")
        _safe_image(APP_LOGO, width=52)

        season = st.selectbox(
            "Saison",
            options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
            key="season_lbl",
        )

        st.markdown("#### Navigation")

        labels = [t[0] for t in TABS]
        default_idx = labels.index(st.session_state.get("active_tab", "🏠 Home")) if st.session_state.get("active_tab") in labels else 0
        active = st.radio("Navigation", labels, index=default_idx, key="nav_radio", label_visibility="collapsed")
        st.session_state["active_tab"] = active

        st.markdown("---")

        c_own, c_logo = st.columns([4, 1], gap="small")
        with c_own:
            owner = st.selectbox("Mon équipe", options=POOL_TEAMS, key="owner_select")
        with c_logo:
            _safe_image(TEAM_LOGO.get(owner, ""), width=36)
        st.session_state["owner"] = owner
        st.session_state["ui_theme"] = "light" if is_light else "dark"

    return active


def _render_bottom_nav(active_label: str) -> None:
    # Mobile helper; links are visual only (Streamlit can't navigate via anchor to session_state).
    # Still useful as a "dock" look; real nav remains in sidebar.
    items = [
        ("🏠", "Home"),
        ("👥", "Joueurs"),
        ("🧊", "Alignement"),
        ("🔁", "Transactions"),
        ("🧑‍💼", "GM"),
    ]
    active_simple = active_label.replace("🏠 ", "").replace("👥 ", "").replace("🧊 ", "").replace("🔁 ", "").replace("🧑‍💼 ", "").strip()
    links = []
    for icon, name in items:
        cls = "active" if name == active_simple else ""
        links.append(f'<a class="{cls}" href="#"><div style="font-size:16px">{icon}</div><div>{name}</div></a>')
    st.markdown(
        f"""
        <div class="pms-bottomnav">
          <div class="row">
            {''.join(links)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _sync_owner_to_sidebar(new_owner: str) -> None:
    """Synchronise le owner 'global' (sidebar) avec la sélection Home, sans DuplicateKey."""
    new_owner = str(new_owner or "").strip()
    if not new_owner:
        return
    st.session_state["owner"] = new_owner
    st.session_state["owner_select"] = new_owner


def _render_home(ctx: AppCtx):
    # banner top — fully at top (like your screenshot proportions)
    if os.path.exists(BANNER):
        st.markdown('<div class="card" style="padding:10px;">', unsafe_allow_html=True)
        st.image(BANNER, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

    st.title("🏠 Home")
    st.markdown('<div class="smallmuted">Home reste clean — aucun bloc Admin ici.</div>', unsafe_allow_html=True)
    st.markdown("")

    # Selection card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏒 Sélection d'équipe")
    st.caption("Cette sélection alimente Alignement / GM / Transactions (même clé session_state).")

    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        # UNIQUE KEY (no collision with sidebar)
        picked = st.selectbox(
            "Équipe (propriétaire)",
            options=POOL_TEAMS,
            index=POOL_TEAMS.index(ctx.owner) if ctx.owner in POOL_TEAMS else 0,
            key="home_owner_select",
        )
        if picked != ctx.owner:
            _sync_owner_to_sidebar(picked)
            ctx.owner = picked

    with c2:
        logo = TEAM_LOGO.get(ctx.owner, "")
        if logo and os.path.exists(logo):
            st.image(logo, width=88)

    st.success(f"✅ Équipe sélectionnée: **{ctx.owner}**")
    st.markdown(f"**Saison:** {ctx.season_lbl}")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_theme_toggle_inline() -> None:
    """Topbar theme toggle without empty labels."""
    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="center")
    with c1:
        # no-op spacer
        st.markdown("")
    with c2:
        is_light = st.toggle("Mode clair (top)", value=(st.session_state.get("ui_theme", "dark") == "light"),
                             key="ui_theme_toggle_top", label_visibility="collapsed")
        # keep single source of truth
        st.session_state["ui_theme"] = "light" if is_light else "dark"
    with c3:
        st.markdown('<div class="smallmuted">Astuce: sur petit écran, utilise la flèche de la sidebar ou ☰.</div>', unsafe_allow_html=True)


def main() -> None:
    st.session_state.setdefault("ui_theme", "dark")
    st.session_state.setdefault("owner", "Whalers")
    st.session_state.setdefault("season_lbl", DEFAULT_SEASON)
    st.session_state.setdefault("active_tab", "🏠 Home")
    st.session_state.setdefault("owner_select", st.session_state.get("owner", "Whalers"))

    apply_theme()

    active_label = sidebar_nav()

    # Top bar (visual) + inline toggle
    _render_topbar(active_label)
    _render_theme_toggle_inline()

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

    # bottom dock look (mobile)
    _render_bottom_nav(active_label)


if __name__ == "__main__":
    main()