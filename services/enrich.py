# Small resolver for update_players_db
from __future__ import annotations

def resolve_update_players_db():
    """Try to import update_players_db from pms_enrich.py if present."""
    try:
        from pms_enrich import update_players_db  # type: ignore
        return update_players_db
    except Exception:
        return None
