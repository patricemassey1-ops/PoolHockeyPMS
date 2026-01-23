# PoolHockey — Modular (v3)

This version keeps your repo layout:

- `assets/previews/` : team logos (preferred)
- `data/` : CSVs + (optional) logo copies
- Fallbacks to repo root for legacy CSV copies

## Files you mentioned (supported)
- `data/hockey.players.csv`
- `data/puckpedia.contracts.csv`
- `data/equipes_joueurs_<season>.csv`
- `data/backup_history.csv`

## Drive (OAuth)
Put in Streamlit Secrets:
```toml
gdrive_folder_id = "1hIJovsHid2L1cY_wKM_sY-wVZKXAwrh1"

[gdrive_oauth]
client_id = "..."
client_secret = "..."
refresh_token = "..."
redirect_uri = "..."
```

## Players DB Admin UI
The Players DB admin panel is in `services/players_db_admin.py` (copied from your `players_db.py`)
and rendered inside `tabs/admin.py`.

It requires `update_players_db` callable. By default we try to import it from `pms_enrich.py`.
If your function lives elsewhere, wire it in `services/enrich.py`.
