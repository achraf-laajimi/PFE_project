"""
TIER 4 — Study Habits

Purpose : Expose study-time patterns, focus score, and engagement trend.
Use when: Query asks about when / how the student studies.
"""

from mcp_server.helpers.database import get_student_analysis
from mcp_server.helpers.logger import get_logger

logger = get_logger(__name__)


async def build_study_habits(student_id: str) -> dict:
    """
    Return preferred study time, time distribution, focus score,
    consistency label, and weekly engagement trend.

    Example query: "وقتاش يقرا أحمد؟"
    """
    sa = get_student_analysis()
    if not sa:
        logger.warning("[study_habits] No analysis data found")
        return {"error": "No data"}

    return {
        "preferred_study_time": sa.get("preferred_study_time", ""),
        "study_time_distribution": sa.get("User_study_time", {}),
        "preferred_study_day": sa.get("Preferred_study_day", ""),
        "focus_score": sa.get("average_focus_score", 0),
        "consistency": sa.get("General_Consistency", ""),
        "weekly_trend": sa.get("Exercise_Weekly_Engagment", ""),
    }
