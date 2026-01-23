import streamlit as st
from services.storage import path_roster, safe_read_csv, path_players_db, path_contracts
from services.players_db import load_players_map, norm_player_key, country_to_flag_emoji

def render(ctx: dict) -> None:
    st.header("🧾 Alignement")
    season = ctx.get("season")
    roster_path = path_roster(season)
    st.caption(f"Roster: {roster_path}")
    st.caption(f"Contracts: {path_contracts()}")

    df = safe_read_csv(roster_path)
    if df.empty:
        st.warning("Roster CSV manquant ou vide.")
        return

    players_map = load_players_map(path_players_db())

    # simple table preview with flag
    name_col = "Joueur" if "Joueur" in df.columns else None
    if not name_col:
        st.dataframe(df, use_container_width=True)
        return

    def flag_for(name: str) -> str:
        k = norm_player_key(name)
        cc = (players_map.get(k, {}) or {}).get("country", "")
        return country_to_flag_emoji(cc)

    df2 = df.copy()
    df2.insert(0, "Flag", df2[name_col].astype(str).map(flag_for))
    st.dataframe(df2, use_container_width=True)
