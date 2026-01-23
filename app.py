import os
import streamlit as st

# =========================================================
# MUST BE FIRST STREAMLIT COMMAND
# =========================================================
st.set_page_config(page_title="Pool Hockey", layout="wide")

from services.storage import DATA_DIR, ensure_data_dir, season_default
from services.ui import apply_theme
from services.drive import resolve_drive_folder_id
from tabs import home, joueurs, alignement, transactions, gm, historique, classement, admin

ensure_data_dir()

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

is_admin = is_admin_user()

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
st.sidebar.caption(f"DATA_DIR: {DATA_DIR}")

# =========================================================
# CONTEXT (passed to every tab)
# =========================================================
ctx = {
    "DATA_DIR": DATA_DIR,
    "season": season_lbl,
    "is_admin": is_admin,
    "drive_folder_id": drive_folder_id,
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
