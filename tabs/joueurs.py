# tabs/joueurs.py
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _norm_name(x: str) -> str:
    return (
        str(x or "")
        .lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )

def _safe(v, default="—"):
    if v is None:
        return default
    if isinstance(v, float) and pd.isna(v):
        return default
    if str(v).strip() == "":
        return default
    return v

def _photo_url(player_id: str | None) -> str | None:
    if not player_id:
        return None
    # NHL headshot standard
    return f"https://assets.nhle.com/mugs/nhl/20242025/{player_id}.png"


# -------------------------------------------------
# Main render
# -------------------------------------------------
def render(ctx: dict) -> None:
    st.header("👤 Joueurs")

    DATA_DIR = ctx.get("DATA_DIR", "data")
    season = ctx.get("season", "2025-2026")

    players_path = os.path.join(DATA_DIR, "hockey.players.csv")
    if not os.path.exists(players_path):
        st.error(f"Fichier introuvable: {players_path}")
        return

    # -------------------------------------------------
    # Load players DB
    # -------------------------------------------------
    try:
        df = pd.read_csv(players_path)
    except Exception as e:
        st.error(f"Erreur lecture hockey.players.csv: {e}")
        return

    if df.empty:
        st.warning("Base joueurs vide.")
        return

    # Normalisation minimale
    df["Joueur"] = df.get("Joueur", df.get("Player", "")).astype(str)
    df["Level"] = df.get("Level", "").fillna("")
    df["Pos"] = df.get("Pos", "").fillna("")
    df["Equipe"] = df.get("Equipe", "").fillna("")
    df["Cap Hit"] = df.get("Cap Hit", df.get("Salary", "")).fillna("")
    df["Contrat"] = df.get("Contract", "").fillna("")
    df["NHL GP"] = df.get("NHL GP", "").fillna("")
    df["Points"] = df.get("Points", "").fillna("")
    df["playerId"] = df.get("playerId", "").fillna("")

    df["_display_name"] = df["Joueur"].astype(str)
    df["_name_key"] = df["_display_name"].apply(_norm_name)

    # -------------------------------------------------
    # Filtres
    # -------------------------------------------------
    st.markdown("### 🎛️ Filtres")

    colf1, colf2 = st.columns(2)

    with colf1:
        level_filter = st.multiselect(
            "Level",
            options=sorted([x for x in df["Level"].unique() if x]),
            default=sorted([x for x in df["Level"].unique() if x]),
        )

    with colf2:
        pos_filter = st.multiselect(
            "Position",
            options=sorted([x for x in df["Pos"].unique() if x]),
            default=sorted([x for x in df["Pos"].unique() if x]),
        )

    df_opts = df.copy()
    if level_filter:
        df_opts = df_opts[df_opts["Level"].isin(level_filter)]
    if pos_filter:
        df_opts = df_opts[df_opts["Pos"].isin(pos_filter)]

    # -------------------------------------------------
    # Recherche (autocomplete performant)
    # -------------------------------------------------
    st.markdown("### 🔎 Recherche joueur")

    q = st.text_input("Tape un nom (ex: marner, draisaitl, savoie)")

    base = df_opts[["_display_name", "_name_key"]].dropna().copy()

    if q.strip():
        q_raw = q.strip()
        q_low = q_raw.lower()
        q_key = _norm_name(q_raw)

        sw = base[base["_display_name"].str.lower().str.startswith(q_low, na=False)]
        sw2 = base[base["_name_key"].str.startswith(q_key, na=False)]
        sw = pd.concat([sw, sw2], ignore_index=True).drop_duplicates()

        ct = base[base["_display_name"].str.lower().str.contains(q_low, na=False)]
        ct2 = base[base["_name_key"].str.contains(q_key, na=False)]
        ct = pd.concat([ct, ct2], ignore_index=True).drop_duplicates()

        res = pd.concat([sw, ct], ignore_index=True).drop_duplicates()
    else:
        res = base

    res = res.head(200)
    opts = res["_display_name"].tolist()

    if not opts:
        st.info("Aucun joueur trouvé.")
        return

    sel = st.selectbox("Choisir un joueur", opts)
    key = _norm_name(sel)

    p = df[df["_name_key"] == key].iloc[0]

    # -------------------------------------------------
    # Fiche joueur
    # -------------------------------------------------
    st.divider()
    st.markdown("### 🧾 Fiche joueur")

    colA, colB = st.columns([1, 2])

    with colA:
        photo = _photo_url(p.get("playerId"))
        if photo:
            st.image(photo, width=160)

    with colB:
        st.markdown(f"**{p['Joueur']}**")
        st.write(f"🏒 Équipe: {_safe(p['Equipe'])}")
        st.write(f"📌 Position: {_safe(p['Pos'])}")
        st.write(f"📄 Level: {_safe(p['Level'])}")
        st.write(f"💰 Salaire: {_safe(p['Cap Hit'])}")
        st.write(f"📃 Contrat: {_safe(p['Contrat'])}")
        st.write(f"🎯 Points: {_safe(p['Points'])}")
        st.write(f"🎮 NHL GP: {_safe(p['NHL GP'])}")

    # -------------------------------------------------
    # Comparatif
    # -------------------------------------------------
    st.divider()
    st.markdown("### ⚖️ Comparatif joueurs")

    sel2 = st.selectbox(
        "Comparer avec",
        [x for x in opts if x != sel],
        index=0,
        key="players_compare",
    )

    p2 = df[df["_name_key"] == _norm_name(sel2)].iloc[0]

    comp = pd.DataFrame(
        {
            sel: {
                "Équipe": p["Equipe"],
                "Pos": p["Pos"],
                "Level": p["Level"],
                "Salaire": p["Cap Hit"],
                "Points": p["Points"],
                "NHL GP": p["NHL GP"],
            },
            sel2: {
                "Équipe": p2["Equipe"],
                "Pos": p2["Pos"],
                "Level": p2["Level"],
                "Salaire": p2["Cap Hit"],
                "Points": p2["Points"],
                "NHL GP": p2["NHL GP"],
            },
        }
    )

    st.dataframe(comp, use_container_width=True)
