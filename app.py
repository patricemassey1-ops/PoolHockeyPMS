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
try:
    from services.backup_drive import scheduled_backup_tick
except Exception:
    scheduled_backup_tick = None  # type: ignore
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


@dataclass
class AppCtx:
    data_dir: str
    season_lbl: str
    owner: str
    is_admin: bool
    theme: str = "dark"

    def as_dict(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "season_lbl": self.season_lbl,
            "owner": self.owner,
            "is_admin": self.is_admin,
            "theme": self.theme,
        }

POOL_TEAMS = ["Whalers", "Red_Wings", "Predateurs", "Nordiques", "Cracheurs", "Canadiens"]

TEAM_LABELS = {
    "Whalers": "Whalers",
    "Red_Wings": "Red Wings",
    "Predateurs": "Prédateurs",
    "Nordiques": "Nordiques",
    "Cracheurs": "Cracheurs",
    "Canadiens": "Canadiens",
}

ASSETS_PREV = os.path.join("assets", "previews")

# Emoji PNG (assets/previews) — icônes à côté des noms des onglets
EMOJI_ICON = {
    "home": os.path.join(ASSETS_PREV, "emoji_home.png"),
    "gm": os.path.join(ASSETS_PREV, "emoji_gm.png"),
    "joueurs": os.path.join(ASSETS_PREV, "emoji_joueur.png"),
    "alignement": os.path.join(ASSETS_PREV, "emoji_alignement.png"),
    "transactions": os.path.join(ASSETS_PREV, "emoji_transaction.png"),
    "historique": os.path.join(ASSETS_PREV, "emoji_historique.png"),
    "classement": os.path.join(ASSETS_PREV, "emoji_coupe.png"),
}

def _emoji_icon_path(slug: str) -> str:
    p = EMOJI_ICON.get(str(slug or "").strip(), "")
    return p if (p and os.path.exists(p)) else ""

def _clean_tab_label(label: str) -> str:
    # Labels actuels: "🏠  Home" → "Home"
    s = str(label or "").strip()
    if "  " in s:
        return s.split("  ", 1)[1].strip()
    # fallback: remove first token if it's a single emoji-like char
    parts = s.split()
    return (" ".join(parts[1:]) if len(parts) > 1 else s).strip()


def _team_logo_path(team: str) -> str:
    """
    Cherche le logo d'équipe dans:
    - assets/previews/<Team>_Logo.png (tes fichiers)
    - data/<Team>.png (fallback)
    """
    team = str(team or "").strip()
    if not team:
        return ""
    # assets/previews variants
    candidates = [
        os.path.join(ASSETS_PREV, f"{team}_Logo.png"),
        os.path.join(ASSETS_PREV, f"{team}_Logo-2.png"),
        os.path.join(ASSETS_PREV, f"{team}_Logo.jpg"),
        os.path.join(ASSETS_PREV, f"{team}_Logo-2.jpg"),
    ]
    # some duplicates you have (ex: Canadiens_Logo vs Canadiens_Logo)
    candidates += [
        os.path.join(ASSETS_PREV, f"{team}s_Logo.png"),
        os.path.join(ASSETS_PREV, f"{team}s_Logo.jpg"),
        os.path.join(ASSETS_PREV, f"{team}E_Logo.png"),
        os.path.join(ASSETS_PREV, f"{team}E_Logo.jpg"),
    ]
    # data fallback
    candidates += [
        os.path.join(DATA_DIR, f"{team}.png"),
        os.path.join(DATA_DIR, f"{team}.jpg"),
        os.path.join(DATA_DIR, f"{team}_logo.png"),
        os.path.join(DATA_DIR, f"{team}_logo.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

TEAM_LOGO = {k: _team_logo_path(k) for k in POOL_TEAMS}
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
    collapsed = bool(st.session_state.get("sidebar_collapsed", False))
    sb_w = "62px" if collapsed else "320px"

    base = THEME_CSS_LIGHT if mode == "light" else THEME_CSS_DARK

    red = "#ef4444"
    red_border = "#dc2626"

    page_bg = "radial-gradient(1200px 600px at 70% 10%, rgba(59,130,246,0.20), transparent 55%), radial-gradient(900px 500px at 15% 0%, rgba(239,68,68,0.16), transparent 55%)"
    sidebar_bg_dark = "linear-gradient(180deg, rgba(18,24,38,0.92), rgba(10,14,22,0.92))"
    sidebar_bg_light = "linear-gradient(180deg, rgba(248,250,252,0.98), rgba(241,245,249,0.98))"

    dyn = (
        "<style>"
        f"section.main {{ background:{page_bg}; }}"
        f"section[data-testid=\"stSidebar\"]{{ width:{sb_w} !important; min-width:{sb_w} !important; max-width:{sb_w} !important; }}"
        f"section[data-testid=\"stSidebar\"] > div{{ background: {(sidebar_bg_light if mode=='light' else sidebar_bg_dark)} !important; backdrop-filter: blur(14px) !important; }}"
        "section[data-testid=\"stSidebar\"]{ border-right: 1px solid rgba(255,255,255,0.10) !important; }"
        ".block-container{ padding-top:2.2rem !important; }"
        "h1,h2,h3{ margin-top:1.0rem !important; }"
        f".stButton > button[kind=\"primary\"]{{ background:{red} !important; border-color:{red_border} !important; }}"
        ".stButton > button[kind=\"primary\"]:hover{ filter:brightness(0.98) !important; transform: translateY(-1px) !important; }"
        "@keyframes pmsGlow { 0%{ box-shadow:0 0 0 rgba(239,68,68,0.0);} 100%{ box-shadow:0 14px 28px rgba(239,68,68,0.28);} }"
        "</style>"
    )

    if collapsed:
        dyn += (
            "<style>"
            "section[data-testid=\"stSidebar\"] div.stButton > button{"
            "  width:46px !important; height:46px !important; min-width:46px !important;"
            "  padding:0 !important; margin:7px auto !important;"
            "  border-radius:14px !important;"
            "  border: 1px solid rgba(255,255,255,0.12) !important;"
            "  background: rgba(255,255,255,0.05) !important;"
            "  transition: transform 120ms ease, box-shadow 150ms ease, border-color 150ms ease, background 150ms ease !important;"
            "  position: relative !important;"
            "}"
            "section[data-testid=\"stSidebar\"] div.stButton > button:hover{"
            "  border-color: rgba(239,68,68,0.55) !important;"
            "  background: rgba(255,255,255,0.07) !important;"
            "  transform: translateY(-1px) scale(1.03) !important;"
            "  box-shadow: 0 10px 22px rgba(0,0,0,0.28) !important;"
            "}"
            f"section[data-testid=\"stSidebar\"] div.stButton > button[kind=\"primary\"]{{"
            f"  box-shadow: 0 14px 28px rgba(239,68,68,0.28) !important;"
            f"  animation: pmsGlow 0.22s ease-out forwards;"
            f"}}"
            f"section[data-testid=\"stSidebar\"] div.stButton > button[kind=\"primary\"]::after{{"
            f"  content:''; position:absolute; inset:0; border-radius:14px;"
            f"  box-shadow: inset 0 1px 0 rgba(255,255,255,0.25);"
            f"}}"
            "section[data-testid=\"stSidebar\"] div.stButton:first-of-type > button{"
            "  width:42px !important; height:42px !important;"
            "  border-radius: 999px !important;"
            "  background: rgba(255,255,255,0.06) !important;"
            "  border-color: rgba(255,255,255,0.14) !important;"
            "}"
            "section[data-testid=\"stSidebar\"] div.stButton > button > div{"
            "  font-size: 20px !important;"
            "  line-height: 20px !important;"
            "}"
            "section[data-testid=\"stSidebar\"] [data-testid=\"stToggle\"]{ display:flex !important; justify-content:center !important; }"
            "</style>"
        )

        # pms_expanded_button_nav
    dyn += (
        "<style>"
        "section[data-testid='stSidebar'] div.stButton > button{transition: transform 120ms ease, box-shadow 150ms ease, border-color 150ms ease !important;}"
        "section[data-testid='stSidebar'] div.stButton > button:hover{transform: translateY(-1px) !important;}"
        "</style>"
    )

    st.markdown(base + dyn, unsafe_allow_html=True)


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
    ("🧑‍💼  GM", "gm"),
    ("🏒  Joueurs", "joueurs"),
    ("📋  Alignement", "alignement"),
    ("🔁  Transactions", "transactions"),
    ("🕘  Historique", "historique"),
    ("🏆  Classement", "classement"),
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
        collapsed = bool(st.session_state.get("sidebar_collapsed", False))

        tri = "▶" if collapsed else "◀"
        if st.button(tri, key="sb_toggle", help="Réduire / agrandir le menu", use_container_width=True):
            st.session_state["sidebar_collapsed"] = not collapsed
            st.rerun()

        icon_px = 30

        # GM logo only (sidebar)
        if os.path.exists(APP_LOGO):
            st.image(APP_LOGO, width=(icon_px if collapsed else 120))

        # Team logo only when collapsed (avoid duplicates)
        owner_now = str(st.session_state.get("owner_select") or st.session_state.get("owner") or "Canadiens")
        tlogo = _team_logo_path(owner_now)
        if collapsed and tlogo and os.path.exists(tlogo):
            st.image(tlogo, width=icon_px)

        if not collapsed:
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
        collapsed = bool(st.session_state.get("sidebar_collapsed", False))

        # Saison (hide in collapsed)
        if not collapsed:
            st.selectbox(
                "Saison",
                options=[DEFAULT_SEASON, "2024-2025", "2023-2024"],
                key="season_lbl",
            )

        owner_now = str(st.session_state.get("owner_select") or st.session_state.get("owner") or "Canadiens")

        # Tabs order (as requested) + Admin only if Whalers
        tabs = list(TABS)
        if _is_admin(owner_now):
            tabs.append(("⚙️  Admin", "admin"))

        labels = [t[0] for t in tabs]
        cur = str(st.session_state.get("active_tab", labels[0]) or labels[0])

        # NAV
        if collapsed:
            active = cur if cur in labels else labels[0]
            for lab in labels:
                icon = lab.split(" ", 1)[0]
                is_active = (lab == active)
                if st.button(
                    icon,
                    key=f"nav_btn_{lab}",
                    help=lab,
                    type=("primary" if is_active else "secondary"),
                    use_container_width=True,
                ):
                    st.session_state["active_tab"] = lab
                    st.rerun()
        else:
            # Mode agrandi (WOW): emoji PNG à gauche + bouton texte (même rouge/selection que le mode réduit)
            active = cur if cur in labels else labels[0]
            for lab, slug in tabs:
                txt = _clean_tab_label(lab)
                icon_path = _emoji_icon_path(slug)

                c1, c2 = st.columns([1, 8], gap="small", vertical_alignment="center")
                with c1:
                    if icon_path:
                        st.image(icon_path, width=22)
                    else:
                        # fallback: montre l'emoji unicode du label
                        st.markdown(f"<div style='text-align:center;font-size:18px;line-height:18px;'>{lab.split(' ',1)[0]}</div>", unsafe_allow_html=True)
                with c2:
                    if st.button(
                        txt,
                        key=f"nav_big_{slug}",
                        type=("primary" if lab == active else "secondary"),
                        use_container_width=True,
                        help=txt,
                    ):
                        st.session_state["active_tab"] = lab
                        st.rerun()

            st.session_state["active_tab"] = active

        # Mon équipe (expanded only)
        if not collapsed:
            st.markdown("---")
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
                _safe_image(_team_logo_path(st.session_state.get("owner_select", "Whalers")), width=34)

        st.session_state["owner"] = st.session_state.get("owner_select", owner_now)

        # Push light mode to bottom + sun above the switch in collapsed
        if collapsed:
            st.markdown("<div style='height: 44vh;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center; font-size:18px; opacity:0.95; margin: 0 0 6px 0;'>☀️</div>", unsafe_allow_html=True)
            is_light = st.toggle(
                "Mode clair",
                value=(st.session_state.get("ui_theme", "dark") == "light"),
                key="ui_theme_toggle_sidebar",
                label_visibility="collapsed",
                help="Mode clair/sombre (☀️/🌙)",
            )
        else:
            st.markdown("<div style='height: 2vh;'></div>", unsafe_allow_html=True)
            is_light = st.toggle(
                "☀️ Mode clair",
                value=(st.session_state.get("ui_theme", "dark") == "light"),
                key="ui_theme_toggle_sidebar",
                help="Mode clair/sombre (☀️/🌙)",
            )

        st.session_state["ui_theme"] = "light" if is_light else "dark"

    return st.session_state.get("active_tab", labels[0])  # type: ignore


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

    # -------------------------------------------------
    # 🔁 Backups AUTO (Drive) — midi & minuit (Whalers)
    # -------------------------------------------------
    try:
        if scheduled_backup_tick:
            did, msg = scheduled_backup_tick(DATA_DIR, str(season), str(owner), show_debug=False)
        else:
            did, msg = (False, "backup_drive module missing")
        if did:
            st.toast("✅ Backup Drive auto fait (midi/minuit).", icon="✅")
    except Exception:
        # Ne jamais bloquer l’app si Drive est down
        pass


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
