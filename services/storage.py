import os
import pandas as pd
from datetime import datetime

DATA_DIR = "data"

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def season_default() -> str:
    y = datetime.now().year
    m = datetime.now().month
    return f"{y}-{y+1}" if m >= 8 else f"{y-1}-{y}"

def path_roster(season: str) -> str:
    season = str(season or "").strip() or season_default()
    return os.path.join(DATA_DIR, f"equipes_joueurs_{season}.csv")

def path_players_db() -> str:
    return os.path.join(DATA_DIR, "hockey.players.csv")

def path_backup_history() -> str:
    return os.path.join(DATA_DIR, "backup_history.csv")

def path_transactions(season: str) -> str:
    season = str(season or "").strip() or season_default()
    return os.path.join(DATA_DIR, f"transactions_{season}.csv")

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
