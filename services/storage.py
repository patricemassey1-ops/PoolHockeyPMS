import os
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------
# DATA + ASSETS (supports your repo layout)
#   - GitHub canonical: ./data/*
#   - Some files may also exist at repo root (legacy)
#   - Team logos may exist in ./assets/previews OR ./data
# ---------------------------------------------------------
DATA_DIR = "data"
ASSETS_PREVIEWS_DIR = os.path.join("assets", "previews")

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ASSETS_PREVIEWS_DIR, exist_ok=True)

def season_default() -> str:
    y = datetime.now().year
    m = datetime.now().month
    return f"{y}-{y+1}" if m >= 8 else f"{y-1}-{y}"

def _first_existing(paths: list[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else ""

# -------------------------
# CSV paths (with fallbacks)
# -------------------------
def path_players_db() -> str:
    # prefer data/hockey.players.csv then repo root
    return _first_existing([
        os.path.join(DATA_DIR, "hockey.players.csv"),
        "hockey.players.csv",
    ])

def path_contracts() -> str:
    # prefer data/puckpedia.contracts.csv then repo root
    return _first_existing([
        os.path.join(DATA_DIR, "puckpedia.contracts.csv"),
        "puckpedia.contracts.csv",
    ])

def path_backup_history() -> str:
    return _first_existing([
        os.path.join(DATA_DIR, "backup_history.csv"),
        "backup_history.csv",
    ])

def path_roster(season: str) -> str:
    season = str(season or "").strip() or season_default()
    return _first_existing([
        os.path.join(DATA_DIR, f"equipes_joueurs_{season}.csv"),
        f"equipes_joueurs_{season}.csv",
    ])

def path_transactions(season: str) -> str:
    season = str(season or "").strip() or season_default()
    return _first_existing([
        os.path.join(DATA_DIR, f"transactions_{season}.csv"),
        f"transactions_{season}.csv",
    ])

# -------------------------
# Logos (assets/previews + data)
# -------------------------
def path_team_logo(filename: str) -> str:
    # filename example: "Whalers_Logo.png" or "WhalersE_Logo.png"
    filename = str(filename or "").strip()
    if not filename:
        return ""
    return _first_existing([
        os.path.join(ASSETS_PREVIEWS_DIR, filename),
        os.path.join(DATA_DIR, filename),
        filename,
    ])

def path_pool_logo() -> str:
    return _first_existing([
        os.path.join(DATA_DIR, "Logo_Pool.png"),
        "logo_pool.png",
        "Logo_Pool.png",
    ])

def safe_read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def safe_write_csv(path: str, df: pd.DataFrame) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        return True
    except Exception:
        return False
