from datetime import datetime


def alert_message(track_id: int, risk: float) -> str:
    return f"[{datetime.utcnow().isoformat()}] ALERT track={track_id} risk={risk:.1f}"
