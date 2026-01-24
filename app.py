import os
import streamlit as st

# =========================================================
# MUST BE FIRST STREAMLIT COMMAND
# =========================================================
st.set_page_config(page_title="Pool Hockey", layout="wide")

from services.event_log import append_event
from services.storage import DATA_DIR, ensure_data_dir, season_default
from services.ui import apply_theme
from services.drive import resolve_drive_folder_id
from services.enrich import resolve_update_players_db
from tabs import home, joueurs, alignement, transactions, gm, historique, classement, admin


ensure_data_dir()
# -------------------------------------------------
# Event Log: "App started" (1x par session)
# -------------------------------------------------
DATA_DIR = str(globals().get("DATA_DIR") or "data")
os.makedirs(DATA_DIR, exist_ok=True)  # au lieu de ensure_data_dir(), ou en plus

season = str(st.session_state.get("season") or "2025-2026").strip() or "2025-2026"

boot_key = f"boot_logged__{season}"
if not st.session_state.get(boot_key, False):
    st.session_state[boot_key] = True
    try:
        append_event(
            data_dir=DATA_DIR,
            season=season,
            owner=str(st.session_state.get("selected_owner") or ""),
            event_type="system",
            summary="App started",
            payload={"page": "app.py"},
        )
    except Exception:
        pass


# =========================================================
# SINGLE CSS/THEME INJECTION (one time)
# =========================================================
apply_theme()

# =========================================================
# GLOBAL APP STATE
# =========================================================
season_lbl = st.session_state.get("season_lbl") or season_default()
season_lbl = st.sidebar.text_input("Saison", value=season_lbl, key="season_lbl")

# Admin gate (replace with your real logic)
def is_admin_user() -> bool:
    # Example: only Whalers
    return str(st.session_state.get("owner") or "").strip().lower() in {"whalers"}

is_admin = True


# Drive folder id (OAuth)
drive_folder_id = resolve_drive_folder_id(default="1hIJovsHid2L1cY_wKM_sY-wVZKXAwrh1")

# =========================================================
# NAV (pure logic, no CSS)
# =========================================================
NAV_TABS = [
    "🏠 Home",
    "👤 Joueurs",
    "🧾 Alignement",
    "⚖️ Transactions",
    "🧠 GM",
    "🕘 Historique",
    "🏆 Classement",
]
if is_admin:
    NAV_TABS.append("🛠️ Gestion Admin")

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = NAV_TABS[0]
if st.session_state["active_tab"] not in NAV_TABS:
    st.session_state["active_tab"] = NAV_TABS[0]

active_tab = st.sidebar.radio("Navigation", NAV_TABS, key="active_tab")
from services.storage import ASSETS_PREVIEWS_DIR
st.sidebar.caption(f"DATA_DIR: {DATA_DIR} | ASSETS: {ASSETS_PREVIEWS_DIR}")

# =========================================================
# CONTEXT (passed to every tab)
# =========================================================
ctx = {
    "DATA_DIR": DATA_DIR,
    "season": season_lbl,
    "is_admin": is_admin,
    "drive_folder_id": drive_folder_id,
    "update_players_db": resolve_update_players_db(),
}

# =========================================================
# ROUTING (one single chain)
# =========================================================
if active_tab == "🏠 Home":
    home.render(ctx)
elif active_tab == "👤 Joueurs":
    joueurs.render(ctx)
elif active_tab == "🧾 Alignement":
    alignement.render(ctx)
elif active_tab == "⚖️ Transactions":
    transactions.render(ctx)
elif active_tab == "🧠 GM":
    gm.render(ctx)
elif active_tab == "🕘 Historique":
    historique.render(ctx)
elif active_tab == "🏆 Classement":
    classement.render(ctx)
elif active_tab == "🛠️ Gestion Admin":
    admin.render(ctx)
else:
    st.error("Onglet inconnu.")
