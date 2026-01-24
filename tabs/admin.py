# tabs/admin.py
import os
import pandas as pd
import streamlit as st
from io import StringIO

from services.storage import (
    path_players_db,
    path_roster,
    path_backup_history,
    path_contracts,
)
from services.drive import (
    drive_ready,
    drive_list_files,
    drive_download_file,
    drive_upload_file,
)
from services.players_db_admin import render_players_db_admin


def render(ctx: dict) -> None:
    st.header("🛠️ Gestion Admin")
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return

    folder_id = ctx.get("drive_folder_id", "")
    season = ctx.get("season")
    update_fn = ctx.get("update_players_db")

    # Targets locaux (toujours sous /data grâce à services.storage)
    targets = {
        "Players DB (data/hockey.players.csv)": path_players_db(),
        "Contracts (data/puckpedia.contracts.csv)": path_contracts(),
        f"Roster (equipes_joueurs_{season}.csv)": path_roster(season),
        "Backup history (backup_history.csv)": path_backup_history(),
    }

    # =====================================================
    # 📥 Restore LOCAL (sans Drive) — upload direct vers target
    # =====================================================
    st.subheader("📥 Import local — Restore selected CSV (sans Drive)")
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
                st.success(f"✅ Restore local OK → {dest}")
                st.caption("Va dans Alignement / Transactions pour valider.")
            except Exception as e:
                st.error(f"Échec restore local: {e}")

    st.divider()

    # =====================================================
    # 📥 Import roster Fantrax par équipe (dropdown) — ROBUST
    #   - Upload d'un CSV Fantrax (ex: Red_Wings.csv)
    #   - Choix équipe/propriétaire dans dropdown
    #   - Parse robuste (Skaters + Goalies, lignes “cassées” ignorées)
    #   - Remplace seulement cette équipe dans equipes_joueurs_<saison>.csv
    # =====================================================
    st.subheader("📥 Import roster Fantrax (par équipe)")
    st.caption(
        "Upload un CSV Fantrax (comme Red_Wings.csv). On convertit et on remplace seulement l’équipe choisie "
        "dans data/equipes_joueurs_<saison>.csv. "
        "Parser robuste: sections Skaters/Goalies + lignes mal formées ignorées."
    )

    def _slot_from_status(status: str) -> str:
        s = str(status or "").strip().lower()
        if s in {"act", "active"}:
            return "Actifs"
        if s in {"min", "minor"}:
            return "Mineur"
        if s in {"res", "reserve", "ir"}:
            return "IR"
        return "Actifs"

    # ⚠️ Ajuste cette liste si tes noms d’équipes/propriétaires diffèrent
    OWNER_CHOICES = ["Canadiens", "Cracheurs", "Nordiques", "Prédateurs", "Red Wings", "Whalers"]

    def _read_fantrax_csv(uploaded_file) -> pd.DataFrame:
        """
        Parse un export Fantrax qui contient souvent:
        - une section Skaters, une section Goalies
        - des lignes titre/séparateurs
        - parfois des lignes qui ont +/− de colonnes (tokenizing error)
        On:
        - repère l'entête qui contient Player/Status/Salary
        - lit chaque bloc avec engine=python + on_bad_lines=skip
        - concatène
        """
        raw = uploaded_file.getvalue()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")

        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        # garde seulement les lignes non vides
        lines = [ln for ln in lines if ln.strip()]

        # entête typique attendue
        def is_header(ln: str) -> bool:
            l = ln.strip()
            return ("Player" in l and "Status" in l and "Salary" in l and "," in l)

        header_idxs = [i for i, ln in enumerate(lines) if is_header(ln)]
        if not header_idxs:
            # fallback: cherche ligne avec "ID,Pos,Player"
            for i, ln in enumerate(lines):
                if ln.strip().startswith("ID") and "Player" in ln and "," in ln:
                    header_idxs = [i]
                    break

        if not header_idxs:
            return pd.DataFrame()

        blocks = []
        for hi, hidx in enumerate(header_idxs):
            # le bloc se termine à la prochaine entête (ou fin)
            next_h = header_idxs[hi + 1] if hi + 1 < len(header_idxs) else len(lines)
            block_lines = lines[hidx:next_h]

            # sécurise: trop petit => skip
            if len(block_lines) < 2:
                continue

            buf = "\n".join(block_lines)
            try:
                df = pd.read_csv(
                    StringIO(buf),
                    engine="python",
                    sep=",",
                    on_bad_lines="skip",  # <-- évite "Expected X fields saw Y"
                )
                if df is not None and not df.empty:
                    blocks.append(df)
            except Exception:
                # si ce bloc est vraiment sale, on ignore
                pass

        if not blocks:
            return pd.DataFrame()

        out = pd.concat(blocks, ignore_index=True)

        # certains exports répètent l'entête dans les données; filtre les lignes où Player == "Player"
        if "Player" in out.columns:
            out = out[out["Player"].astype(str).str.strip().ne("Player")]

        # garde colonnes clés si elles existent
        return out

    fantrax_csv = st.file_uploader("CSV Fantrax roster", type=["csv"], key="fantrax_roster_csv")
    if fantrax_csv is not None:
        try:
            imp = _read_fantrax_csv(fantrax_csv)

            if imp.empty:
                st.error("Impossible de parser ce CSV Fantrax (aucune table détectée).")
            else:
                # Ton format attendu (après parse): Player, Pos, Team, Status, Salary
                required_cols = ["Player", "Pos", "Team", "Status", "Salary"]
                missing = [c for c in required_cols if c not in imp.columns]
                if missing:
                    st.error("Colonnes manquantes après parse: " + ", ".join(missing))
                    st.caption("Colonnes détectées: " + ", ".join([str(c) for c in imp.columns]))
                else:
                    owner_pick = st.selectbox(
                        "Équipe (Propriétaire) à assigner à ce CSV",
                        OWNER_CHOICES,
                        key="fantrax_owner_pick",
                    )

                    out = pd.DataFrame(
                        {
                            "Propriétaire": owner_pick,
                            "Joueur": imp["Player"].astype(str).str.strip(),
                            "Pos": imp["Pos"].astype(str).str.strip(),
                            "Equipe": imp["Team"].astype(str).str.strip(),
                            "Salaire": imp["Salary"],
                            "Level": "",
                            "Statut": imp["Status"].astype(str).str.strip(),
                            "Slot": imp["Status"].astype(str).map(_slot_from_status),
                            "IR Date": "",
                        }
                    )

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

                    colA, colB = st.columns([1, 1])
                    with colA:
                        st.caption("Aperçu importé (20 lignes)")
                        st.dataframe(out.head(20), use_container_width=True)
                    with colB:
                        st.caption("Résumé")
                        st.write(f"- Lignes importées: **{len(out)}**")
                        st.write(f"- Destination: `{dest}`")

                    if st.button("✅ Importer cette équipe (replace)", type="primary", key="do_fantrax_import"):
                        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                        final.to_csv(dest, index=False)
                        st.success(f"✅ Import OK: {owner_pick} remplacé dans {dest}")
                        st.caption("Va dans Alignement pour valider.")

        except Exception as e:
            st.error(f"Erreur lecture/parse CSV: {e}")

    st.divider()

    # =====================================================
    # ☁️ Drive restore (OAuth) — optionnel
    # =====================================================
    st.subheader("☁️ Drive — Restore selected CSV (OAuth)")
    st.caption("Dossier Drive: My Drive / PMS Pool Data / PoolHockeyData")
    st.code(f"folder_id = {str(folder_id or '').strip() or '(missing)'}")

    if not drive_ready():
        st.info(
            "Drive OAuth non prêt. Ajoute [gdrive_oauth] + gdrive_folder_id dans Secrets si tu veux restaurer depuis Drive."
        )
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
                name = f.get("name", "")
                mid = f.get("modifiedTime", "")
                size = f.get("size", "")
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
                        st.success(f"✅ Restore Drive OK → {dest}")
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
                    st.success(f"✅ Upload OK: {res.get('name')}")
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
