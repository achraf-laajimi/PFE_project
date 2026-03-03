"""
TIER 4 — Recent Activity

Purpose : Return a chronological exercise timeline with effort / reward flags.
Use when: Query asks about recent work, today's session, or last activity.
"""

from mcp_server.helpers.database import get_histories, _safe_int
from mcp_server.helpers.logger import get_logger

logger = get_logger(__name__)


async def build_recent_activity(student_id: str, limit: int = 10) -> dict:
    """
    Return the *limit* most-recent exercise attempts, each flagged when
    time_seconds > 300 AND stars < 2 ("High effort, low reward").

    Example query: "شنوة خدم أحمد اليوم؟"
    """
    rows = get_histories(limit=limit)
    if not rows:
        logger.warning("[recent_activity] No history rows found")
        return {"error": "No recent activity"}

    activities = []
    for row in rows:
        time_sec = _safe_int(row.get("time_seconds")) or 0
        stars = _safe_int(row.get("stars")) or 0

        flag = "⚠ High effort, low reward" if time_sec > 300 and stars < 2 else None

        activities.append(
            {
                "timestamp": row.get("validated_at", ""),
                "time_seconds": time_sec,
                "stars": stars,
                "mistakes": _safe_int(row.get("nb_mistakes")) or 0,
                "flag": flag,
            }
        )

    return {"recent_activities": activities}
