#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_nhl_id.py
Petit audit lisible pour vérifier où sont les IDs, et détecter les colonnes candidates.

Usage:
  python audit_nhl_id.py --file ./data/hockey.players.csv
"""

from __future__ import annotations
import argparse, re
import pandas as pd

ID_COLS = ["nhl_id","NHL_ID","nhlId","nhlID","id_nhl","player_id","playerId","nhl_player_id","nhlPlayerId"]

def is_valid_id(x) -> bool:
    if x is None: return False
    s=str(x).strip()
    if s=="" or s.lower() in {"nan","none","null"}: return False
    return bool(re.fullmatch(r"\d+", s))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args=ap.parse_args()

    df=pd.read_csv(args.file)
    print(f"File: {args.file}")
    print(f"Rows: {len(df)} Cols: {len(df.columns)}")

    cand=[]
    for c in df.columns:
        cl=c.lower()
        if "nhl" in cl or cl in [x.lower() for x in ID_COLS] or re.search(r"\bid\b", cl):
            cand.append(c)
    print("\nCandidate columns (id-ish):")
    for c in cand:
        nonnull=df[c].apply(is_valid_id).sum() if c in df.columns else 0
        empties=(df[c].astype(str).str.strip().isin(["","nan","None","none","null"])).sum() if c in df.columns else 0
        print(f" - {c}: valid_ids={nonnull} / {len(df)} | emptyish={empties}")

    if "nhl_id" in df.columns:
        miss=len(df)-df["nhl_id"].apply(is_valid_id).sum()
        pct=miss/len(df)*100 if len(df) else 0
        print(f"\nnhl_id missing: {miss} ({pct:.1f}%)")
    else:
        print("\nNo nhl_id column found.")

if __name__=="__main__":
    main()
