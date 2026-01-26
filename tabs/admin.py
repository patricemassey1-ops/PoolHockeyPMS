# tabs/admin.py
# ============================================================
# PMS Pool Hockey — Admin Tab (Streamlit)
# Compatible avec: admin.render(ctx) depuis app.py
# ============================================================
# ✅ Import équipes (LOCAL) via CSV (multi-upload)
# ✅ Preview + validation colonnes attendues
# ✅ ➕ Ajouter joueurs (anti-triche cross-team)
# ✅ 🗑️ Retirer joueurs (UI + confirmation)
# ✅ 🔁 Déplacer GC ↔ CE (auto-slot / keep / force)
# ✅ 🧪 Barres visuelles cap GC/CE + dépassements
# ✅ 📋 Historique admin complet (ADD/REMOVE/MOVE/IMPORT)
# ✅ Auto-mapping Level via hockey.players.csv (+ heuristique salaire)
# ✅ Alertes IR mismatch + Salary/Level suspect + preview colorée
# ============================================================

from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import time
import streamlit as st

# ---- Paths (module defaults)
# DATA_DIR is resolved/overridden inside render(ctx); must exist at import time for helper functions.
DATA_DIR = os.path.join(os.getcwd(), "data") if os.path.isdir(os.path.join(os.getcwd(), "data")) else "data"


# ---- Google Drive retiré (fonctionnalités supprimées)

def _fmt_money(v: int) -> str:
    try:
        v = int(v)
    except Exception:
        v = 0
    return f"{v:,}".replace(",", " ") + " $"


def _parse_money(s: str) -> int:
    s = str(s or "")
    s = s.replace("$", "").replace(" ", "").replace(",", "")
    # allow dots in input like 1.000.000
    s = s.replace(".", "")
    try:
        return int(float(s))
    except Exception:
        return 0


def _validate_cap(v: int) -> Tuple[bool, str]:
    if v < 1_000_000:
        return False, "Valeur trop basse (< 1 000 000)."
    if v > 200_000_000:
        return False, "Valeur trop haute (> 200 000 000)."
    return True, ""


def _settings_local_path() -> str:
    return os.path.join(DATA_DIR, "settings.csv")


def _load_settings_local() -> Dict[str, Any]:
    p = _settings_local_path()
    if not os.path.exists(p):
        return {}
    try:
        df = pd.read_csv(p)
        out = {}
        for _, r in df.iterrows():
            k = str(r.get("key", "")).strip()
            if not k:
                continue
            out[k] = r.get("value")
        return out
    except Exception:
        return {}


def _save_settings_local(settings: Dict[str, Any]) -> bool:
    try:
        rows = [{"key": k, "value": settings[k]} for k in sorted(settings.keys())]
        pd.DataFrame(rows).to_csv(_settings_local_path(), index=False)
        return True
    except Exception:
        return False


# (Google Drive retiré)


def _zip_dir_bytes(dir_path: str) -> Optional[bytes]:
    """Zip un dossier en mémoire (pour download). Retourne bytes ou None."""
    if not dir_path or (not os.path.isdir(dir_path)):
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(dir_path):
            for f in files:
                p = os.path.join(root, f)
                arc = os.path.relpath(p, dir_path)
                z.write(p, arcname=arc)
    return buf.getvalue()


# ============================================================
# MAIN RENDER
# ============================================================
def render(ctx: dict) -> None:
    if not ctx.get("is_admin"):
        st.warning("Accès admin requis.")
        return
    global DATA_DIR
    DATA_DIR = str(ctx.get("DATA_DIR") or "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    season_lbl = str(ctx.get("season") or "2025-2026").strip() or "2025-2026"

    e_path = equipes_path(DATA_DIR, season_lbl)
    log_path = admin_log_path(DATA_DIR, season_lbl)

    st.subheader("🛠️ Gestion Admin")

    # ---- caps inputs (persistants via settings.csv local)
    st.session_state.setdefault("CAP_GC", DEFAULT_CAP_GC)
    st.session_state.setdefault("CAP_CE", DEFAULT_CAP_CE)

    # Auto-load settings local au démarrage (1 fois)
    if not st.session_state.get("_admin_settings_loaded", False):
        settings = _load_settings_local()
        if settings:
            if "CAP_GC" in settings:
                try:
                    st.session_state["CAP_GC"] = int(float(settings["CAP_GC"]))
                except Exception:
                    pass
            if "CAP_CE" in settings:
                try:
                    st.session_state["CAP_CE"] = int(float(settings["CAP_CE"]))
                except Exception:
                    pass
        st.session_state["_admin_settings_loaded"] = True

    # =====================================================
    # 🧪 Test drive write... (LOCAL) + 🗜️ Backup complet
    # =====================================================
    with st.expander("🧪 Test drive write... (local) + 🗜️ Backup complet", expanded=False):
        st.caption("Teste les droits d'écriture **dans /data** (Streamlit Cloud) + permet de télécharger un zip complet du dossier data.")
        test_name = f"_write_test_{int(time.time())}.txt"
        test_path = os.path.join(DATA_DIR, test_name)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧪 Tester écriture locale (/data)", use_container_width=True, key="adm_local_write_test"):
                try:
                    payload = f"LOCAL WRITE OK — {datetime.utcnow().isoformat()} UTC\n"
                    with open(test_path, "w", encoding="utf-8") as f:
                        f.write(payload)
                    ok = os.path.exists(test_path) and (os.path.getsize(test_path) > 0)
                    if ok:
                        st.success(f"OK: {test_name} écrit dans /data")
                    else:
                        st.error("Échec: le fichier test n'existe pas après écriture.")
                except Exception as e:
                    st.error(f"Échec écriture locale: {type(e).__name__}: {e}")

        with c2:
            if st.button("🗜️ Générer ZIP complet (/data)", use_container_width=True, key="adm_zip_data"):
                b = _zip_dir_bytes(DATA_DIR)
                if not b:
                    st.error("Impossible de zipper /data (dossier manquant ?).")
                else:
                    zip_name = f"backup_data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
                    st.download_button("⬇️ Télécharger le ZIP", data=b, file_name=zip_name, mime="application/zip", use_container_width=True)

        if os.path.exists(test_path):
            try:
                os.remove(test_path)
            except Exception:
                pass


    # =====================================================
    # 📋 Historique admin
    # =====================================================
    with st.expander("📋 Historique admin", expanded=False):
        if not os.path.exists(log_path):
            st.info("Aucun historique pour l’instant.")
        else:
            try:
                lg = pd.read_csv(log_path).sort_values("timestamp", ascending=False)

                f1, f2, f3 = st.columns(3)
                with f1:
                    act = st.multiselect("Action", sorted(lg["action"].dropna().unique()), default=[], key="adm_log_act")
                with f2:
                    own = st.multiselect("Équipe", sorted(lg["owner"].dropna().unique()), default=[], key="adm_log_own")
                with f3:
                    q = st.text_input("Recherche joueur", value="", key="adm_log_q").strip().lower()

                view = lg.copy()
                if act:
                    view = view[view["action"].isin(act)]
                if own:
                    view = view[view["owner"].isin(own)]
                if q:
                    view = view[view["player"].astype(str).str.lower().str.contains(q, na=False)]

                st.dataframe(view.head(400), use_container_width=True)
            except Exception as e:
                st.error(f"Erreur log: {e}")

    # =====================================================
    # 💰 Plafond salariale
    # =====================================================
    with st.expander("💰 Plafond salariale", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cap_gc_txt = st.text_input("Cap GC", value=_fmt_money(st.session_state.get("CAP_GC", DEFAULT_CAP_GC)), key="cap_gc_txt")
        with c2:
            cap_ce_txt = st.text_input("Cap CE", value=_fmt_money(st.session_state.get("CAP_CE", DEFAULT_CAP_CE)), key="cap_ce_txt")

        cap_gc = _parse_money(cap_gc_txt)
        cap_ce = _parse_money(cap_ce_txt)

        ok_gc, msg_gc = _validate_cap(cap_gc)
        ok_ce, msg_ce = _validate_cap(cap_ce)

        if not ok_gc:
            st.warning(f"GC: {msg_gc}")
        if not ok_ce:
            st.warning(f"CE: {msg_ce}")

        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("💾 Sauvegarder (local)", use_container_width=True, key="adm_caps_save"):
                if ok_gc and ok_ce:
                    st.session_state["CAP_GC"] = cap_gc
                    st.session_state["CAP_CE"] = cap_ce
                    settings = {"CAP_GC": cap_gc, "CAP_CE": cap_ce}
                    ok_local = _save_settings_local(settings)
                    if ok_local:
                        st.success("Sauvegarde OK (data/settings.csv).")
                    else:
                        st.error("Sauvegarde locale a échoué.")
                else:
                    st.error("Corrige les plafonds avant de sauvegarder.")
        with b2:
            if st.button("🔄 Recharger (local)", use_container_width=True, key="adm_caps_reload"):
                settings = _load_settings_local()
                if settings:
                    if "CAP_GC" in settings:
                        try:
                            st.session_state["CAP_GC"] = int(float(settings["CAP_GC"]))
                        except Exception:
                            pass
                    if "CAP_CE" in settings:
                        try:
                            st.session_state["CAP_CE"] = int(float(settings["CAP_CE"]))
                        except Exception:
                            pass
                    st.success("Plafonds rechargés.")
                else:
                    st.warning("Aucun settings.csv trouvé (local).")

        st.caption(f"Actuel: GC { _fmt_money(st.session_state.get('CAP_GC', DEFAULT_CAP_GC)) } • CE { _fmt_money(st.session_state.get('CAP_CE', DEFAULT_CAP_CE)) }")

    # ---- Players DB index
    players_db = load_players_db(os.path.join(DATA_DIR, PLAYERS_DB_FILENAME))
    players_idx = build_players_index(players_db)
    if players_idx:
        st.success(f"Players DB détectée: {PLAYERS_DB_FILENAME} (Level auto + infos).")
    else:
        st.warning(f"{PLAYERS_DB_FILENAME} indisponible — fallback Level par Salaire.")

    # ---- Load équipes
    df = load_equipes(e_path)

    # =====================================================
    # 📥 IMPORT ÉQUIPES (LOCAL)
    # =====================================================
    with st.expander("➕ Ajouter joueur(s) (anti-triche)", expanded=False):
        owner = st.selectbox("Équipe", owners, key="adm_add_owner")
        assign = st.radio("Assignation", ["GC - Actif", "GC - Banc", "CE - Actif", "CE - Banc"], horizontal=True, key="adm_add_assign")
        statut = "Grand Club" if assign.startswith("GC") else "Club École"
        slot = "Actif" if assign.endswith("Actif") else "Banc"

        allow_override = st.checkbox("🛑 Autoriser override admin si joueur appartient déjà à une autre équipe", value=False, key="adm_add_override")

        if players_idx:
            all_names = sorted({v["Joueur"] for v in players_idx.values() if v.get("Joueur")})
            selected = st.multiselect("Joueurs", all_names, key="adm_add_players")
        else:
            raw = st.text_area("Saisir joueurs (1 par ligne)", height=120, key="adm_add_manual")
            selected = [x.strip() for x in raw.splitlines() if x.strip()]

        preview: List[Dict[str, Any]] = []
        blocked: List[Tuple[str, str]] = []

        for p in selected:
            info = players_idx.get(_norm_player(p), {}) if players_idx else {}
            name = info.get("Joueur", p)
            cur_owner = find_player_owner(df, name)
            if cur_owner and cur_owner != owner and not allow_override:
                blocked.append((name, cur_owner))
                continue

            preview.append({
                "Propriétaire": owner,
                "Joueur": name,
                "Pos": info.get("Pos", ""),
                "Equipe": info.get("Equipe", ""),
                "Salaire": int(info.get("Salaire", 0) or 0),
                "Level": info.get("Level", "0"),
                "Statut": statut,
                "Slot": slot,
                "IR Date": "",
            })

        if blocked and not allow_override:
            st.error("⛔ Anti-triche: ces joueurs appartiennent déjà à une autre équipe")
            st.dataframe(pd.DataFrame(blocked, columns=["Joueur", "Équipe actuelle"]), use_container_width=True)

        if preview:
            st.dataframe(pd.DataFrame(preview).head(80), use_container_width=True)

        if st.button("✅ Ajouter maintenant", use_container_width=True, key="adm_add_commit"):
            if not preview:
                st.warning("Rien à ajouter.")
                st.stop()

            existing = set(zip(df["Propriétaire"].astype(str).str.strip(), df["Joueur"].astype(str).str.strip()))
            new_rows = []
            skipped_dupe = 0
            skipped_block = 0

            for r in preview:
                k = (str(r["Propriétaire"]).strip(), str(r["Joueur"]).strip())
                if k in existing:
                    skipped_dupe += 1
                    continue
                cur_owner = find_player_owner(df, r["Joueur"])
                if cur_owner and cur_owner != owner and not allow_override:
                    skipped_block += 1
                    continue
                new_rows.append(r)

            if not new_rows:
                st.warning(f"Rien à ajouter (doublons: {skipped_dupe}, bloqués: {skipped_block}).")
                st.stop()

            backup_team_rows(df, DATA_DIR, season_lbl, owner, note="pre-add")
            df2 = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df2_qc, stats = apply_quality(df2, players_idx)
            save_equipes(df2_qc, e_path)
            st.session_state["equipes_df"] = df2_qc

            for r in new_rows:
                append_admin_log(
                    log_path,
                    action="ADD",
                    owner=r["Propriétaire"],
                    player=r["Joueur"],
                    to_statut=r["Statut"],
                    to_slot=r["Slot"],
                    note=f"assign={assign}; override={allow_override}"
                )

            st.success(f"✅ Ajout OK: {len(new_rows)} | doublons: {skipped_dupe} | bloqués: {skipped_block} | Level auto: {stats.get('level_autofilled',0)}")
            st.rerun()

    # =====================================================
    # 🗑️ REMOVE
    # =====================================================
    with st.expander("🔁 Déplacer GC ↔ CE (auto-slot)", expanded=False):
        owner = st.selectbox("Équipe", owners, key="adm_move_owner")
        team = df[df["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()

        if team.empty:
            st.info("Aucun joueur pour cette équipe.")
        else:
            team["__label__"] = team.apply(lambda r: f"{r['Joueur']}  —  {r.get('Pos','')}  —  {r.get('Statut','')} / {r.get('Slot','')}", axis=1)
            choices = team["__label__"].tolist()
            sel = st.multiselect("Sélectionner joueur(s) à déplacer", choices, key="adm_move_sel")

            dest_statut = st.radio("Destination", ["Grand Club", "Club École"], horizontal=True, key="adm_move_dest")
            slot_mode = st.radio("Slot destination", ["Auto (selon Statut)", "Garder Slot actuel", "Forcer Actif", "Forcer Banc"], horizontal=True, key="adm_move_slot_mode")
            keep_ir = st.checkbox("Conserver IR si joueur déjà IR", value=True, key="adm_move_keep_ir")

            if st.button("🔁 Appliquer déplacement", use_container_width=True, key="adm_move_commit"):
                if not sel:
                    st.warning("Sélectionne au moins 1 joueur.")
                    st.stop()

                sel_rows = team[team["__label__"].isin(sel)].copy()
                if sel_rows.empty:
                    st.warning("Sélection invalide.")
                    st.stop()

                keyset = set(zip(
                    sel_rows["Propriétaire"].astype(str),
                    sel_rows["Joueur"].astype(str),
                    sel_rows["Statut"].astype(str),
                    sel_rows["Slot"].astype(str),
                ))

                backup_team_rows(df, DATA_DIR, season_lbl, owner, note="pre-move")
                df2 = df.copy()
                moved = 0
                for idx, r in df2.iterrows():
                    k = (str(r["Propriétaire"]), str(r["Joueur"]), str(r["Statut"]), str(r["Slot"]))
                    if k not in keyset:
                        continue

                    from_statut = str(r["Statut"])
                    from_slot = str(r["Slot"])

                    df2.at[idx, "Statut"] = dest_statut

                    if slot_mode.startswith("Auto"):
                        df2.at[idx, "Slot"] = auto_slot_for_statut(dest_statut, current_slot=from_slot, keep_ir=keep_ir)
                    elif slot_mode.startswith("Garder"):
                        df2.at[idx, "Slot"] = from_slot
                    elif slot_mode.endswith("Actif"):
                        df2.at[idx, "Slot"] = "Actif"
                    elif slot_mode.endswith("Banc"):
                        df2.at[idx, "Slot"] = "Banc"
                    else:
                        df2.at[idx, "Slot"] = auto_slot_for_statut(dest_statut, current_slot=from_slot, keep_ir=keep_ir)

                    moved += 1

                    append_admin_log(
                        log_path,
                        action="MOVE",
                        owner=r["Propriétaire"],
                        player=r["Joueur"],
                        from_statut=from_statut,
                        from_slot=from_slot,
                        to_statut=dest_statut,
                        to_slot=str(df2.at[idx, "Slot"]),
                        note=f"slot_mode={slot_mode}"
                    )

                df2_qc, stats = apply_quality(df2, players_idx)
                save_equipes(df2_qc, e_path)
                st.session_state["equipes_df"] = df2_qc

                st.success(f"✅ Move OK: {moved} joueur(s) | Level auto: {stats.get('level_autofilled',0)}")
                st.rerun()
    with st.expander("🗑️ Retirer joueur(s) (avec confirmation)", expanded=False):
        owner = st.selectbox("Équipe", owners, key="adm_rem_owner")
        team = df[df["Propriétaire"].astype(str).str.strip().eq(str(owner).strip())].copy()

        if team.empty:
            st.info("Aucun joueur pour cette équipe.")
        else:
            team["__label__"] = team.apply(lambda r: f"{r['Joueur']}  —  {r.get('Pos','')}  —  {r.get('Statut','')} / {r.get('Slot','')}", axis=1)
            choices = team["__label__"].tolist()
            sel = st.multiselect("Sélectionner joueur(s) à retirer", choices, key="adm_rem_sel")

            confirm = st.checkbox("Je confirme la suppression", key="adm_rem_confirm")

            if st.button("🗑️ Retirer maintenant", use_container_width=True, key="adm_rem_commit"):
                if not sel:
                    st.warning("Sélectionne au moins 1 joueur.")
                    st.stop()
                if not confirm:
                    st.warning("Coche la confirmation.")
                    st.stop()

                sel_rows = team[team["__label__"].isin(sel)].copy()
                if sel_rows.empty:
                    st.warning("Sélection invalide.")
                    st.stop()

                keys = set(zip(
                    sel_rows["Propriétaire"].astype(str),
                    sel_rows["Joueur"].astype(str),
                    sel_rows["Statut"].astype(str),
                    sel_rows["Slot"].astype(str),
                ))

                def _keep_row(r):
                    k = (str(r["Propriétaire"]), str(r["Joueur"]), str(r["Statut"]), str(r["Slot"]))
                    return k not in keys

                backup_team_rows(df, DATA_DIR, season_lbl, owner, note="pre-remove")
                before = len(df)
                df2 = df[df.apply(_keep_row, axis=1)].copy()
                removed = before - len(df2)

                df2_qc, _ = apply_quality(df2, players_idx)
                save_equipes(df2_qc, e_path)
                st.session_state["equipes_df"] = df2_qc

                for _, r in sel_rows.iterrows():
                    append_admin_log(
                        log_path,
                        action="REMOVE",
                        owner=r["Propriétaire"],
                        player=r["Joueur"],
                        from_statut=r.get("Statut", ""),
                        from_slot=r.get("Slot", ""),
                        note="removed by admin"
                    )

                st.success(f"✅ Retrait OK: {removed} joueur(s).")
                st.rerun()

    # =====================================================
    # 🔁 MOVE GC ↔ CE
    # =====================================================


    with st.expander("🧼 Preview local + alertes", expanded=False):
        df = load_equipes(e_path)
        if df.empty:
            st.info("Aucun fichier équipes local. Utilise Import Local ci-dessous.")
        else:
            df_qc, stats = apply_quality(df, players_idx)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lignes", stats["rows"])
            c2.metric("Level auto", stats["level_autofilled"])
            c3.metric("⚠️ IR mismatch", stats["ir_mismatch"])
            c4.metric("⚠️ Salaire/Level", stats["salary_level_suspect"])
            try:
                st.dataframe(df_qc.head(140).style.apply(_preview_style_row, axis=1), use_container_width=True)
            except Exception:
                st.dataframe(df_qc.head(140), use_container_width=True)

            if st.button("💾 Appliquer QC + sauvegarder + reload", use_container_width=True, key="adm_apply_qc"):
                save_equipes(df_qc, e_path)
                st.session_state["equipes_df"] = df_qc
                st.success("✅ QC appliqué + sauvegarde + reload.")
                st.rerun()

    # refresh after potential import
    df = load_equipes(e_path)
    owners = sorted([x for x in df.get("Propriétaire", pd.Series(dtype=str)).dropna().astype(str).str.strip().unique() if x])
    if not owners:
        st.warning("Aucune équipe (Propriétaire) détectée. Importe d'abord le CSV équipes.")
        return

    # =====================================================
    # ➕ ADD (ANTI-TRICHE)
    # =====================================================
    with st.expander("📥 Import local (fallback) — multi-upload CSV équipes", expanded=False):
        st.caption("Upload plusieurs CSV (1 par équipe). Auto-assign via `Propriétaire` (si unique) ou via le nom du fichier (ex: `Whalers.csv`).")
        st.code(f"Destination locale (fusion): {e_path}", language="text")

        mode = st.radio(
            "Mode de fusion",
            ["Ajouter (append)", "Remplacer l’équipe (delete puis insert)"],
            horizontal=True,
            key="adm_multi_mode",
        )

        uploads = st.file_uploader(
            "Uploader un ou plusieurs CSV (équipes)",
            type=["csv"],
            accept_multiple_files=True,
            key="adm_multi_uploads",
        )
        zip_up = st.file_uploader(
            "Ou uploader un ZIP contenant plusieurs CSV",
            type=["zip"],
            key="adm_multi_zip",
            help="Fallback si ton navigateur ne permet pas de multi-sélectionner.",
        )

        items: List[Tuple[str, Any]] = []
        if uploads:
            for f in uploads:
                items.append((getattr(f, "name", "upload.csv"), f))

        if zip_up is not None:
            try:
                z = zipfile.ZipFile(zip_up)
                for name in z.namelist():
                    if name.lower().endswith(".csv") and not name.endswith("/"):
                        items.append((name, z.read(name)))
            except Exception as e:
                st.error(f"ZIP invalide: {e}")

        if not items:
            st.info("Ajoute 1+ fichiers (multi) ou un ZIP de CSV.")
        else:
            prep = st.button("🧼 Préparer les fichiers (analyse + attribution)", use_container_width=True, key="adm_multi_prepare")
            st.caption("Astuce: l'analyse démarre seulement quand tu cliques, pour éviter les reruns qui cassent à la sélection.")
            if not prep:
                st.stop()
            parsed: List[Dict[str, Any]] = []
            errors: List[Tuple[str, str]] = []

            for file_name, payload in items:
                # read csv
                try:
                    if isinstance(payload, (bytes, bytearray)):
                        df_up = pd.read_csv(io.BytesIO(payload))
                    else:
                        df_up = pd.read_csv(payload)
                except Exception:
                    try:
                        if isinstance(payload, (bytes, bytearray)):
                            df_up = pd.read_csv(io.BytesIO(payload), encoding="latin-1")
                        else:
                            try:
                                payload.seek(0)
                            except Exception:
                                pass
                            df_up = pd.read_csv(payload, encoding="latin-1")
                    except Exception as e:
                        errors.append((file_name, f"Lecture CSV impossible: {e}"))
                        continue

                ok, missing, _extras = validate_equipes_df(df_up)
                if not ok:
                    errors.append((file_name, f"Colonnes manquantes: {missing}"))
                    continue

                df_up = ensure_equipes_df(df_up)
                owners_in_file = sorted([
                    x for x in df_up["Propriétaire"].astype(str).str.strip().unique()
                    if x and x.lower() != "nan"
                ])

                parsed.append({"file": file_name, "df": df_up, "owners_in_file": owners_in_file})

            if errors:
                st.error("Certains fichiers ont des erreurs et seront ignorés :")
                st.dataframe(pd.DataFrame(errors, columns=["Fichier", "Erreur"]), use_container_width=True)

            if not parsed:
                st.warning("Aucun fichier valide à importer.")
            else:
                df_current = load_equipes(e_path)
                owners_choices = sorted([
                    x for x in df_current.get("Propriétaire", pd.Series(dtype=str))
                    .dropna().astype(str).str.strip().unique() if x
                ])

                st.markdown("### Attribution des fichiers → équipe")
                assignments: List[Tuple[Dict[str, Any], str]] = []
                for i, p in enumerate(parsed):
                    owners_in_file = p["owners_in_file"]
                    preferred = owners_in_file[0] if len(owners_in_file) == 1 else ""
                    if not preferred:
                        preferred = infer_owner_from_filename(p["file"], owners_choices)

                    c1, c2, c3 = st.columns([2, 2, 3])
                    with c1:
                        st.write(f"**{p['file']}**")
                        st.caption(f"Lignes: {len(p['df'])} | Owners détectés: {', '.join(owners_in_file) if owners_in_file else '—'}")
                    with c2:
                        if owners_choices:
                            idx = owners_choices.index(preferred) if preferred in owners_choices else 0
                            chosen = st.selectbox("Équipe", owners_choices, index=idx, key=f"adm_multi_owner_{i}")
                        else:
                            chosen = st.text_input("Équipe", value=preferred, key=f"adm_multi_owner_txt_{i}").strip()
                    with c3:
                        st.caption("Preview")
                        st.dataframe(p["df"].head(10), use_container_width=True)

                    assignments.append((p, chosen))

                missing_choice = [p["file"] for p, chosen in assignments if not str(chosen or "").strip()]
                if missing_choice:
                    st.warning("Choisis une équipe pour: " + ", ".join(missing_choice))

                colA, colB = st.columns([1, 1])
                do_import = colA.button("⬇️ Importer tous → Local + QC + Reload", use_container_width=True, key="adm_multi_commit")
                do_dry = colB.button("🧪 Dry-run (voir résumé)", use_container_width=True, key="adm_multi_dry")

                if do_dry or do_import:
                    merged = load_equipes(e_path)
                    rows_before = len(merged)

                    replaced: Dict[str, int] = {}
                    imported = 0
                    backed_up: set = set()

                    for p, chosen in assignments:
                        owner = str(chosen or "").strip()
                        if not owner:
                            continue

                        if owner not in backed_up:
                            backup_team_rows(merged, DATA_DIR, season_lbl, owner, note=f"pre-import {mode}")
                            backed_up.add(owner)

                        df_up = p["df"].copy()
                        df_up["Propriétaire"] = owner
                        df_up_qc, stats = apply_quality(df_up, players_idx)

                        if mode.startswith("Remplacer"):
                            before_owner = int((merged["Propriétaire"].astype(str).str.strip() == owner).sum()) if not merged.empty else 0
                            merged = merged[~merged["Propriétaire"].astype(str).str.strip().eq(owner)].copy()
                            replaced[owner] = before_owner

                        merged = pd.concat([merged, df_up_qc], ignore_index=True)
                        imported += len(df_up_qc)

                        append_admin_log(
                            log_path,
                            action="IMPORT_LOCAL_TEAM",
                            owner=owner,
                            player="",
                            note=f"file={p['file']}; rows={len(df_up_qc)}; level_auto={stats.get('level_autofilled',0)}; mode={mode}",
                        )

                    merged, stats_all = apply_quality(merged, players_idx)
                    rows_after = len(merged)

                    summary = {
                        "mode": mode,
                        "fichiers_valides": len(parsed),
                        "lignes_avant": rows_before,
                        "lignes_importees": imported,
                        "lignes_apres": rows_after,
                        "teams_replaced": replaced,
                        "qc_level_auto": stats_all.get("level_autofilled", 0),
                        "qc_ir_mismatch": stats_all.get("ir_mismatch", 0),
                        "qc_salary_level_suspect": stats_all.get("salary_level_suspect", 0),
                    }
                    st.markdown("### Résumé import")
                    st.json(summary)

                    if do_import:
                        save_equipes(merged, e_path)
                        st.session_state["equipes_df"] = merged
                        st.success("✅ Import multi terminé + QC + reload.")
                        st.rerun()

        st.divider()
        if st.button("🧱 Créer un fichier équipes vide (squelette)", use_container_width=True, key="adm_local_create_empty"):
            df_empty = pd.DataFrame(columns=EQUIPES_COLUMNS)
            save_equipes(df_empty, e_path)
            st.session_state["equipes_df"] = df_empty
            append_admin_log(log_path, action="INIT_EMPTY", owner="", player="", note="created empty equipes file")
            st.success("✅ Fichier équipes vide créé.")
            st.rerun()

    # =====================================================
    # 📋 HISTORIQUE ADMIN
    # =====================================================
    st.caption("✅ Admin: Local only • Import local • Add/Remove/Move • Caps • Log • QC/Level auto")