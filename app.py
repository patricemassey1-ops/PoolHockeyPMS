import streamlit as st
import os
import importlib.util

# Configuration de la page
st.set_page_config(page_title="GM Pool Manager", layout="wide")

# --- INITIALISATION DU SESSION STATE ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "home"
if "season" not in st.session_state:
    st.session_state.season = "2025-2026"
if "owner" not in st.session_state:
    st.session_state.owner = "Canadiens"

def _render_module(module, ctx):
    """Exécute la fonction render() du module s'il existe."""
    if hasattr(module, "render"):
        module.render(ctx)
    else:
        st.error(f"Le module {module.__name__} n'a pas de fonction render().")

def main():
    # --- SIDEBAR ---
    st.sidebar.title("Configuration")
    
    # Sélecteur de Saison
    season = st.sidebar.selectbox(
        "Saison", 
        ["2024-2025", "2025-2026", "2026-2027"], 
        index=1,
        key="season_selector"
    )
    st.session_state.season = season

    # Sélecteur d'Équipe (GM)
    owner = st.sidebar.selectbox(
        "Équipe", 
        ["Canadiens", "Bruins", "Maple Leafs", "Rangers"], # À adapter selon tes CSV
        key="owner_selector"
    )
    st.session_state.owner = owner

    st.sidebar.divider()

    # --- NAVIGATION ---
    tabs = {
        "Home": "home",
        "GM": "gm",
        "Joueurs": "joueurs",
        "Alignement": "alignement",
        "Transactions": "transactions"
    }

    for label, key in tabs.items():
        if st.sidebar.button(label, use_container_width=True, 
                            type="primary" if st.session_state.active_tab == key else "secondary"):
            st.session_state.active_tab = key
            st.rerun()

    # --- RENDU DE LA PAGE ---
    ctx = {
        "DATA_DIR": "data",
        "season": st.session_state.season,
        "owner": st.session_state.owner
    }

    active = st.session_state.active_tab

    if active == "home":
        st.title("🏠 Accueil")
        st.write(f"Bienvenue, GM des {owner} !")
    else:
        # Import dynamique du fichier dans le dossier /tabs
        try:
            module_name = f"tabs.{active}"
            # On s'assure que le module est rechargé si modifié
            spec = importlib.util.find_spec(module_name)
            if spec:
                module = importlib.import_module(module_name)
                _render_module(module, ctx)
            else:
                st.error(f"Fichier `tabs/{active}.py` introuvable.")
        except Exception as e:
            st.error(f"Erreur lors du chargement de la page : {e}")

if __name__ == "__main__":
    main()
