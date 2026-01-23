# PoolHockey — Modular Starter

## Files
- app.py : routing + theme + ctx
- tabs/* : one file per tab (render(ctx))
- services/* : shared logic (storage, players_db, drive, ui)
- data/ : CSV files (not included here)

## Secrets (Streamlit Cloud)
```toml
gdrive_folder_id = "1hIJovsHid2L1cY_wKM_sY-wVZKXAwrh1"

[gdrive_oauth]
client_id = "..."
client_secret = "..."
refresh_token = "..."
redirect_uri = "..."
```
