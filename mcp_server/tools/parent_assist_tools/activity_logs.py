"""
Tool 3 — Daily Activity Logs (Timeline View)

Builds a chronological feed of recent exercise attempts from
student_score_histories.csv.
"""

from mcp_server.helpers.database import get_histories, _safe_int


async def build_daily_activity_logs(student_id: str, limit: int = 20) -> dict:
    """Return the *limit* most recent exercise attempts with flags."""
    rows = get_histories(limit=limit)
    if not rows:
        return {"error": "No activity history found"}

    activity_feed = []
    high_effort_low_reward_count = 0

    for row in rows:
        time_sec = _safe_int(row.get("time_seconds")) or 0
        stars = _safe_int(row.get("stars")) or 0
        mistakes = _safe_int(row.get("nb_mistakes")) or 0
        is_done = str(row.get("is_done", "false")).strip().lower() == "true"

        flag = ""
        if time_sec > 300 and stars < 2:
            flag = "⚠ High Effort / Low Reward"
            high_effort_low_reward_count += 1

        entry = {
            "validated_at": row.get("validated_at", ""),
            "time_seconds": time_sec,
            "stars": stars,
            "mistakes": mistakes,
            "is_done": is_done,
            "score": _safe_int(row.get("score")) or 0,
        }
        if flag:
            entry["flag"] = flag
        activity_feed.append(entry)

    return {
        "total_returned": len(activity_feed),
        "high_effort_low_reward_count": high_effort_low_reward_count,
        "activity_feed": activity_feed,
    }
