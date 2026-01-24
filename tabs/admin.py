# tabs/admin.py
import os
import re
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
    # 📥 Import roster Fantrax (BATCH) — multi-files
    # =====================================================
    st.subheader("📥 Import roster Fantrax — BATCH (multi-fichiers)")
    st.caption(
        "Upload 1 à 6+ fichiers Fantrax (même pattern). "
        "On parse Skaters+Goalies, puis on remplace chaque équipe dans data/equipes_joueurs_<saison>.csv."
    )

    # ⚠️ Ajuste si tes noms de propriétaires diffèrent
    OWNER_CHOICES = ["Canadiens", "Cracheurs", "Nordiques", "Prédateurs", "Red Wings", "Whalers"]

    # mapping “slug” -> owner officiel
    OWNER_ALIASES = {
        "canadiens": "Canadiens",
        "montreal": "Canadiens",
        "mtl": "Canadiens",
        "cracheurs": "Cracheurs",
        "nordiques": "Nordiques",
        "predateurs": "Prédateurs",
        "prédateurs": "Prédateurs",
        "predateurs_": "Prédateurs",
        "redwings": "Red Wings",
        "red_wings": "Red Wings",
        "red-wings": "Red Wings",
        "whalers": "Whalers",
    }

    def _slot_from_status(status: str) -> str:
        s = str(status or "").strip().lower()
        if s in {"act", "active"}:
            return "Actifs"
        if s in {"min", "minor"}:
            return "Mineur"
        if s in {"res", "reserve", "ir"}:
            return "IR"
        return "Actifs"

    def _guess_owner_from_filename(fname: str) -> str:
        base = os.path.splitext(os.path.basename(fname or ""))[0]
        s = base.strip().lower()
        s = s.replace(" ", "_")
        s = re.sub(r"[^a-z0-9_\-]+", "", s)

        # direct alias hits
        if s in OWNER_ALIASES:
            return OWNER_ALIASES[s]

        # try contains
        for k, v in OWNER_ALIASES.items():
            if k and k in s:
                return v

        # fallback: title-case-ish
        # ex: "red_wings" -> "Red Wings"
        pretty = s.replace("-", " ").replace("_", " ").strip().title()
        return pretty if pretty else ""

    def _read_fantrax_csv(uploaded_file) -> pd.DataFrame:
        """Parse un export Fantrax multi-sections + lignes brisées."""
        raw = uploaded_file.getvalue()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")

        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        lines = [ln for ln in lines if ln.strip()]

        def is_header(ln: str) -> bool:
            l = ln.strip()
            return ("Player" in l and "Status" in l and "Salary" in l and "," in l)

        header_idxs = [i for i, ln in enumerate(lines) if is_header(ln)]
        if not header_idxs:
            for i, ln in enumerate(lines):
                if ln.strip().startswith("ID") and "Player" in ln and "," in ln:
                    header_idxs = [i]
                    break

        if not header_idxs:
            return pd.DataFrame()

        blocks = []
        for hi, hidx in enumerate(header_idxs):
            next_h = header_idxs[hi + 1] if hi + 1 < len(header_idxs) else len(lines)
            block_lines = lines[hidx:next_h]
            if len(block_lines) < 2:
                continue

            buf = "\n".join(block_lines)
            try:
                df = pd.read_csv(
                    StringIO(buf),
                    engine="python",
                    sep=",",              # change to sep=None if you ever get ";" locale CSV
                    on_bad_lines="skip",  # 핵: ignore broken lines
                )
                if df is not None and not df.empty:
                    blocks.append(df)
            except Exception:
                pass

        if not blocks:
            return pd.DataFrame()

        out = pd.concat(blocks, ignore_index=True)

        if "Player" in out.columns:
            out = out[out["Player"].astype(str).str.strip().ne("Player")]

        return out

    def _fantrax_to_roster_df(fantrax_df: pd.DataFrame, owner: str) -> pd.DataFrame:
        required_cols = ["Player", "Pos", "Team", "Status", "Salary"]
        missing = [c for c in required_cols if c not in fantrax_df.columns]
        if missing:
            raise ValueError("Colonnes manquantes après parse: " + ", ".join(missing))

        df = fantrax_df.copy()

        # 🔥 FIX CRITIQUE
        # - enlève lignes NaN
        # - enlève "Skaters", "Goalies", titres, lignes vides
        df["Player"] = df["Player"].astype(str).str.strip()
        df = df[
            df["Player"].notna()
            & df["Player"].ne("")
            & (~df["Player"].str.lower().isin(["player", "skaters", "goalies", "nan"]))
        ]

        # sécurités supplémentaires
        df = df[df["Salary"].notna()]
        df = df[df["Pos"].notna()]

        return pd.DataFrame(
            {
                "Propriétaire": owner,
                "Joueur": df["Player"],
                "Pos": df["Pos"].astype(str).str.strip(),
                "Equipe": df["Team"].astype(str).str.strip(),
                "Salaire": pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int),
                "Level": "",
                "Statut": df["Status"].astype(str).str.strip(),
                "Slot": df["Status"].astype(str).map(_slot_from_status),
                "IR Date": "",
            }
        )


    def _load_current_roster(dest: str) -> pd.DataFrame:
        try:
            return pd.read_csv(dest) if os.path.exists(dest) else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _replace_owner_block(cur: pd.DataFrame, owner: str, block: pd.DataFrame) -> pd.DataFrame:
        if cur is not None and not cur.empty and "Propriétaire" in cur.columns:
            other = cur[cur["Propriétaire"].astype(str) != str(owner)]
        else:
            other = pd.DataFrame()
        return pd.concat([other, block], ignore_index=True)

    batch_files = st.file_uploader(
        "Upload CSV Fantrax (multi)",
        type=["csv"],
        accept_multiple_files=True,
        key="fantrax_batch_files",
    )

    dest_roster = path_roster(season)
    st.caption(f"Destination roster: `{dest_roster}`")

    if batch_files:
        st.markdown("#### 📦 Fichiers détectés")
        rows = []
        for i, f in enumerate(batch_files):
            fname = getattr(f, "name", f"file_{i}.csv")
            guess = _guess_owner_from_filename(fname)
            rows.append((fname, guess))
        st.dataframe(pd.DataFrame(rows, columns=["Fichier", "Équipe devinée"]), use_container_width=True)

        st.markdown("#### ✅ Assigner les équipes (au besoin)")
        assignments = {}
        for i, f in enumerate(batch_files):
            fname = getattr(f, "name", f"file_{i}.csv")
            guess = _guess_owner_from_filename(fname)
            default_idx = OWNER_CHOICES.index(guess) if guess in OWNER_CHOICES else 0
            pick = st.selectbox(
                f"Équipe pour {fname}",
                OWNER_CHOICES,
                index=default_idx,
                key=f"pick_owner__{i}",
            )
            assignments[fname] = pick

        dry = st.checkbox("Mode test (dry-run, n'écrit pas)", value=False, key="fantrax_dry")
        if st.button("🚀 Importer TOUT (replace par équipe)", type="primary", key="fantrax_batch_go"):
            cur = _load_current_roster(dest_roster)
            done, failed = [], []

            for i, f in enumerate(batch_files):
                fname = getattr(f, "name", f"file_{i}.csv")
                owner = assignments.get(fname) or OWNER_CHOICES[0]
                try:
                    df_f = _read_fantrax_csv(f)
                    if df_f.empty:
                        raise ValueError("Aucune table détectée (parse vide).")

                    block = _fantrax_to_roster_df(df_f, owner=owner)
                    cur = _replace_owner_block(cur, owner=owner, block=block)
                    done.append((fname, owner, len(block)))
                except Exception as e:
                    failed.append((fname, owner, str(e)))

            if dry:
                st.success(f"✅ Dry-run terminé. OK: {len(done)} | Échecs: {len(failed)}")
            else:
                try:
                    os.makedirs(os.path.dirname(dest_roster) or ".", exist_ok=True)
                    cur.to_csv(dest_roster, index=False)
                    st.success(f"✅ Import batch OK → {dest_roster}")
                except Exception as e:
                    st.error(f"Écriture roster impossible: {e}")
                    failed.append(("WRITE", "-", str(e)))

            if done:
                st.markdown("#### ✅ Imports OK")
                st.dataframe(pd.DataFrame(done, columns=["Fichier", "Équipe", "Lignes"]), use_container_width=True)

            if failed:
                st.markdown("#### ❌ Échecs")
                st.dataframe(pd.DataFrame(failed, columns=["Fichier", "Équipe", "Erreur"]), use_container_width=True)

            st.caption("Va dans Alignement pour valider.")

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
