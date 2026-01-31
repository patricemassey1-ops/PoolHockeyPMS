from __future__ import annotations

def resolve_update_players_db():
    """Resolve a callable to update players DB, if available.

    Priority:
    1) pms_enrich.update_players_db (if you have it)
    2) players_db.update_players_db (legacy name)
    Otherwise returns None (UI still works; update buttons will be disabled).
    """
    # 1) pms_enrich.py at repo root
    try:
        from pms_enrich import update_players_db  # type: ignore
        return update_players_db
    except Exception:
        pass

    # 2) legacy module name
    try:
        from players_db import update_players_db  # type: ignore
        return update_players_db
    except Exception:
        return None
