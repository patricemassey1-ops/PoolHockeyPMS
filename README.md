# PoolHockey — Modular Starter (v2)

✅ Supports your repo layout:
- `assets/previews/` : team logos (preferred)
- `data/` : CSVs + (optional) logo copies
- Some CSVs may also exist at repo root (legacy) — we fallback safely.

## Expected files
- `data/hockey.players.csv`
- `data/puckpedia.contracts.csv`
- `data/equipes_joueurs_<season>.csv`
- (optional) `data/backup_history.csv`

## Secrets (Streamlit Cloud)
```toml
gdrive_folder_id = "1hIJovsHid2L1cY_wKM_sY-wVZKXAwrh1"

[gdrive_oauth]
client_id = "..."
client_secret = "..."
refresh_token = "..."
redirect_uri = "..."
```

## Admin tab
- Lists Drive files in your folder_id
- Restore selected CSV into local targets:
  - Players DB
  - Contracts
  - Roster
  - Backup history
