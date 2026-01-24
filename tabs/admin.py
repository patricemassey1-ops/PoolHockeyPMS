import os
import pandas as pd
import streamlit as st
from services.storage import path_players_db, path_roster, path_backup_history, path_contracts
from services.drive import drive_ready, drive_list_files, drive_download_file, drive_upload_file
from services.players_db_admin import render_players_db_admin

def render(ctx: dict) -> None:
    st.header("🛠️ Gestion Admin")
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    folder_id = ctx.get("drive_folder_id", "")
    season = ctx.get("season")
    update_fn = ctx.get("update_players_db")

    targets = {
        "Players DB (data/hockey.players.csv)": path_players_db(),
        "Contracts (data/puckpedia.contracts.csv)": path_contracts(),
        f"Roster (equipes_joueurs_{season}.csv)": path_roster(season),
        "Backup history (backup_history.csv)": path_backup_history(),
    }

    # =====================================================
    # 📥 Restore LOCAL (sans Drive) — pour tester tout de suite
    # =====================================================
    st.subheader("📥 Import local — Restore selected CSV (sans Drive)")

    import pandas as pd


    def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for c in df.columns:
            cc = str(c).strip().lower()
            if cc in ["owner", "propriétaire", "proprietaire", "team_owner"]:
                rename[c] = "Propriétaire"
            elif cc in ["player", "joueur", "name"]:
                rename[c] = "Joueur"
            elif cc in ["pos", "position"]:
                rename[c] = "Pos"
            elif cc in ["team", "equipe", "équipe", "nhlteam"]:
                 rename[c] = "Equipe"
            elif cc in ["salary", "salaire", "cap_hit", "caphit"]:
                 rename[c] = "Salaire"
            elif cc in ["status", "statut"]:
                rename[c] = "Statut"
            elif cc in ["contract"]:
                rename[c] = "Slot"   # ton fichier met Contract=Rachat, on le mappe dans Slot
        return df.rename(columns=rename)

    def _slot_from_status(status: str) -> str:
        s = str(status or "").strip().lower()
        if s in {"act", "active"}:
            return "Actifs"
        if s in {"min", "minor"}:
            return "Mineur"
        if s in {"res", "reserve", "ir"}:
            return "IR"
        return "Actifs"

    st.subheader("📥 Import roster Fantrax (par équipe)")
    st.caption("Upload un CSV Fantrax (comme Red_Wings.csv). On convertit et on remplace seulement l’équipe choisie dans equipes_joueurs_<saison>.csv")

    up = st.file_uploader("CSV Fantrax roster", type=["csv"], key="fantrax_roster_csv")

    if up is not None:
        imp = pd.read_csv(up)
        imp = _norm_cols(imp)

        # ton CSV n'a pas Propriétaire, donc on choisit via dropdown
        owner_pick = st.selectbox(
            "Équipe (Propriétaire) à assigner à ce CSV",
            ["Canadiens","Cracheurs","Nordiques","Prédateurs","Red Wings","Whalers"],
            key="fantrax_owner_pick"
        )

        # Colonnes attendues du fichier Fantrax
        required = ["Player","Pos","Team","Status","Salary"]
        missing = [c for c in required if c not in imp.columns and c not in imp.rename(columns=str).columns]
        # (on reste souple, mais ton fichier a exactement ces noms)

        # Normalisation minimale basée sur TON CSV exact
        if "Player" not in imp.columns:
            st.error("Colonne 'Player' introuvable.")
        else:
            # Build mapping vers le format du roster de ton app
            out = pd.DataFrame({
                "Propriétaire": owner_pick,
                "Joueur": imp["Player"].astype(str).str.strip(),
                "Pos": imp.get("Pos",""),
                "Equipe": imp.get("Team",""),
                "Salaire": imp.get("Salary",""),
                "Level": "",      # tu pourras le remplir ensuite via hockey.players.csv
                "Statut": imp.get("Status",""),
                "Slot": imp.get("Status","").astype(str).map(_slot_from_status),
                "IR Date": ""
            })

            # Charger le roster global et remplacer seulement cette équipe
            dest = path_roster(season)  # data/equipes_joueurs_<season>.csv
            try:
                cur = pd.read_csv(dest) if os.path.exists(dest) else pd.DataFrame()
            except Exception:
                cur = pd.DataFrame()

            if not cur.empty and "Propriétaire" in cur.columns:
                cur_other = cur[cur["Propriétaire"].astype(str) != str(owner_pick)]
            else:
                cur_other = pd.DataFrame()

            final = pd.concat([cur_other, out], ignore_index=True)

            if st.button("✅ Importer cette équipe (replace)", type="primary", key="do_fantrax_import"):
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                final.to_csv(dest, index=False)
                st.success(f"✅ Import OK: {owner_pick} remplacé dans {dest}")
                st.caption("Va dans Alignement pour valider.")
                st.dataframe(out.head(20), use_container_width=True)


    st.caption("Upload un CSV depuis ton ordi et on l’écrit directement dans le bon fichier sous /data.")
    tgt_local = st.selectbox("Target local", list(targets.keys()), key="local_target")
    up = st.file_uploader("Choisir un CSV", type=["csv"], key="local_csv")

    if st.button("⬇️ Restore (upload → target)", type="primary", key="local_restore"):
        if not up:
            st.warning("Choisis un fichier CSV.")
        else:
            dest = targets.get(tgt_local, "")
            try:
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(up.getbuffer())
                st.success(f"Restore local OK → {dest}")
                st.caption("Va dans Alignement / Transactions pour valider.")
            except Exception as e:
                st.error(f"Échec restore local: {e}")

    st.divider()

    # =====================================================
    # ☁️ Drive restore (OAuth)
    # =====================================================
    st.subheader("☁️ Drive — Restore selected CSV (OAuth)")
    st.caption("Dossier Drive: My Drive / PMS Pool Data / PoolHockeyData")
    st.code(f"folder_id = {str(folder_id or '').strip() or '(missing)'}")

    if not drive_ready():
        st.info("Drive OAuth non prêt. Ajoute [gdrive_oauth] + gdrive_folder_id dans Secrets si tu veux restaurer depuis Drive.")
    else:
        filter_text = st.text_input("Filtre (nom contient)", value=".csv", key="drive_filter")
        if st.button("🔄 Refresh Drive list"):
            st.session_state.pop("drive_files_cache", None)

        if "drive_files_cache" not in st.session_state:
            st.session_state["drive_files_cache"] = drive_list_files(folder_id, name_contains=filter_text.strip())

        files = st.session_state.get("drive_files_cache", []) or []
        if not files:
            st.info("Aucun fichier Drive trouvé.")
        else:
            labels = []
            id_by_label = {}
            for f in files[:200]:
                name = f.get("name","")
                mid = f.get("modifiedTime","")
                size = f.get("size","")
                label = f"{name} — {mid} — {size}"
                labels.append(label)
                id_by_label[label] = f.get("id")

            c1, c2 = st.columns([1.2, 1.2])
            with c1:
                pick = st.selectbox("Drive file", [""] + labels, key="drive_pick")
            with c2:
                tgt = st.selectbox("Target", list(targets.keys()), key="drive_target")

            if st.button("⬇️ Restore Drive → target", type="primary"):
                if not pick:
                    st.warning("Choisis un fichier.")
                else:
                    fid = id_by_label.get(pick, "")
                    dest = targets.get(tgt, "")
                    res = drive_download_file(fid, dest)
                    if res.get("ok"):
                        st.success(f"Restore OK → {dest}")
                        st.caption("Relance l’app si tu veux recharger les CSV/caches.")
                    else:
                        st.error(res.get("error") or "Restore failed")

        st.subheader("⬆️ Upload local file to Drive (optional)")
        local_path = st.text_input("Local path to upload", value="")
        if st.button("⬆️ Upload"):
            if not local_path:
                st.warning("Donne un path local.")
            else:
                res = drive_upload_file(folder_id, local_path)
                if res.get("ok"):
                    st.success(f"Upload OK: {res.get('name')}")
                else:
                    st.error(res.get("error") or "Upload failed")

    st.divider()

    # =====================================================
    # Players DB Admin UI
    # =====================================================
    st.subheader("🗃️ Players DB (Admin)")
    if update_fn is None:
        st.info("update_players_db non trouvé. Les boutons Update/Resume seront désactivés (UI ok).")

    render_players_db_admin(
        pdb_path=path_players_db(),
        data_dir=ctx.get("DATA_DIR", "data"),
        season_lbl=season,
        update_fn=update_fn,
    )
