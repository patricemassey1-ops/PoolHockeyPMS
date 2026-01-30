#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_nhl_id.py
-----------------
Objectif:
- Restaurer / récupérer la colonne nhl_id dans data/hockey.players.csv
- En utilisant n'importe quel fichier source existant dans /data (rosters, backups, exports, etc.)
- Sans jamais écraser des nhl_id existants avec des valeurs vides.

Usage:
  python recover_nhl_id.py --data-dir ./data --season 2025-2026

Sorties:
- hockey.players.RECOVERED.csv (dans data-dir)
- report_nhl_id_recovery.json (dans data-dir)

Notes:
- Le script cherche des colonnes candidates: nhl_id, NHL_ID, nhlId, player_id, playerId, id_nhl, etc.
- Matching par nom normalisé (supporte "Nom, Prénom" vs "Prénom Nom"), et optionnellement par équipe si présent.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# Normalisation helpers
# -------------------------
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_name(s: str) -> str:
    """Normalise un nom de joueur pour matching."""
    s = str(s or "").strip()
    if not s:
        return ""
    s = _strip_accents(s)
    s = s.replace(".", " ").replace("’", "'")
    s = re.sub(r"[^A-Za-z0-9,'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # "zucker, jason" -> "jason zucker"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) >= 2:
            s = f"{parts[1]} {parts[0]}".strip()
    return s

def norm_team(s: str) -> str:
    s = str(s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


# -------------------------
# Find sources
# -------------------------
ID_COL_CANDIDATES = [
    "nhl_id", "NHL_ID", "nhlId", "nhlID", "id_nhl", "idNHL",
    "player_id", "playerId", "nhl_player_id", "nhlplayerid", "nhlPlayerId",
]

NAME_COL_CANDIDATES = [
    "Player", "player", "Joueur", "joueur", "Name", "name", "Nom", "nom",
]

TEAM_COL_CANDIDATES = [
    "Team", "team", "Equipe", "équipe", "Equipe ", "Club", "club",
]

def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # fallback: partial contains
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() == cl:
                return c
    return None

def is_valid_id(x) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return False
    # often integer; keep as str
    return bool(re.fullmatch(r"\d+", s))

def build_source_maps(df: pd.DataFrame, name_col: str, id_col: str, team_col: Optional[str]) -> Tuple[Dict[str,str], Dict[Tuple[str,str],str]]:
    """Retourne:
      - map_name: norm_name -> nhl_id (dernier vu)
      - map_name_team: (norm_name, norm_team) -> nhl_id
    """
    map_name: Dict[str,str] = {}
    map_name_team: Dict[Tuple[str,str],str] = {}
    for _, row in df.iterrows():
        nm = norm_name(row.get(name_col))
        if not nm:
            continue
        pid = row.get(id_col)
        if not is_valid_id(pid):
            continue
        pid = str(pid).strip()
        map_name[nm] = pid
        if team_col:
            tm = norm_team(row.get(team_col))
            if tm:
                map_name_team[(nm, tm)] = pid
    return map_name, map_name_team

def scan_csv_files(data_dir: str, season: str) -> List[str]:
    pats = [
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "*", "*.csv"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    # prioriser quelques noms probables
    priority = []
    for f in files:
        base = os.path.basename(f).lower()
        if "backup" in base or "restore" in base:
            priority.append(f)
        if season in base or season.replace("-", "_") in base:
            priority.append(f)
        if "equipes" in base or "roster" in base or "alignment" in base or "alignement" in base:
            priority.append(f)
    # remove duplicates while preserving order: priority then others
    seen=set()
    ordered=[]
    for f in priority + files:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--players-file", default="hockey.players.csv")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    players_path = os.path.join(data_dir, args.players_file)

    if not os.path.exists(players_path):
        raise SystemExit(f"Players file introuvable: {players_path}")

    players = pd.read_csv(players_path)
    # ensure nhl_id column exists (but don't overwrite)
    if "nhl_id" not in players.columns:
        players["nhl_id"] = pd.NA

    before_nonnull = players["nhl_id"].apply(is_valid_id).sum()
    total = len(players)

    # Build target matching keys
    name_col_p = detect_col(players, ["Player","player","Joueur","joueur","Name","name","Nom","nom"])
    team_col_p = detect_col(players, ["Team","team","Equipe","équipe","Club","club"])
    if not name_col_p:
        raise SystemExit("Impossible de détecter la colonne nom (Player/Joueur/Name) dans hockey.players.csv")

    players["_k_name"] = players[name_col_p].map(norm_name)
    players["_k_team"] = players[team_col_p].map(norm_team) if team_col_p else ""

    # Scan sources
    files = scan_csv_files(data_dir, args.season)

    collected_sources = []
    map_name: Dict[str,str] = {}
    map_name_team: Dict[Tuple[str,str],str] = {}

    for f in files:
        if os.path.abspath(f) == os.path.abspath(players_path):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        id_col = detect_col(df, ID_COL_CANDIDATES)
        if not id_col:
            continue
        name_col = detect_col(df, NAME_COL_CANDIDATES)
        if not name_col:
            continue
        team_col = detect_col(df, TEAM_COL_CANDIDATES)

        m1, m2 = build_source_maps(df, name_col, id_col, team_col)
        if not m1 and not m2:
            continue

        # merge into global maps (last file wins)
        map_name.update(m1)
        map_name_team.update(m2)
        collected_sources.append({
            "file": os.path.relpath(f, data_dir),
            "rows": int(len(df)),
            "id_col": id_col,
            "name_col": name_col,
            "team_col": team_col,
            "valid_ids_by_name": int(len(m1)),
            "valid_ids_by_name_team": int(len(m2)),
        })

    # Apply recovery: prefer (name,team) then name only
    recovered = 0
    already = 0
    not_found = 0

    new_ids = []
    for i, row in players.iterrows():
        current = row.get("nhl_id")
        if is_valid_id(current):
            already += 1
            new_ids.append(str(current).strip())
            continue

        nm = row["_k_name"]
        tm = row["_k_team"] if team_col_p else ""
        pid = None
        if nm and tm and (nm, tm) in map_name_team:
            pid = map_name_team[(nm, tm)]
        elif nm and nm in map_name:
            pid = map_name[nm]

        if pid and is_valid_id(pid):
            recovered += 1
            new_ids.append(str(pid).strip())
        else:
            not_found += 1
            new_ids.append(pd.NA)

    players["nhl_id"] = new_ids
    after_nonnull = players["nhl_id"].apply(is_valid_id).sum()

    # Clean temp keys
    players.drop(columns=[c for c in ["_k_name","_k_team"] if c in players.columns], inplace=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_players = os.path.join(data_dir, f"hockey.players.RECOVERED_{stamp}.csv")
    players.to_csv(out_players, index=False)

    report = {
        "players_file": os.path.relpath(players_path, data_dir),
        "output_file": os.path.relpath(out_players, data_dir),
        "total_players": int(total),
        "nhl_id_nonnull_before": int(before_nonnull),
        "nhl_id_nonnull_after": int(after_nonnull),
        "recovered_now": int(recovered),
        "already_had_id": int(already),
        "still_missing": int(not_found),
        "sources_used": collected_sources,
        "notes": [
            "Si recovered_now = 0 et sources_used est vide: aucun fichier dans /data ne contient de NHL_ID exploitable.",
            "Dans ce cas: il faut soit RESTORE un ancien hockey.players.csv (Drive/local backup), soit refaire le mapping via API NHL."
        ],
    }
    out_report = os.path.join(data_dir, f"report_nhl_id_recovery_{stamp}.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("✅ Terminé.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
