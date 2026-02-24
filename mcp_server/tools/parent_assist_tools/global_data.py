"""
Tool 1 — Global "Health Score" (Macro View)

Builds the student's overall performance snapshot from analysis.json.
"""

from mcp_server.helpers.database import get_student_analysis


async def build_student_global_data(student_id: str, timeframe: str = "total") -> dict:
    """Return global KPIs, usage summary, and weekly engagement trend."""
    sa = get_student_analysis()
    if not sa:
        return {"error": "No analysis data found"}

    # ── Timeframe-specific metrics ───────────────────────────
    if timeframe == "today":
        exercises = sa.get("Today_exercises", 0)
        exercise_time = sa.get("Today_exercise_time", 0)
        usage_time = sa.get("Today_usage_time", 0)
    elif timeframe == "yesterday":
        exercises = sa.get("Yesterday_exercises", 0)
        exercise_time = sa.get("Yesterday_exercise_time", 0)
        usage_time = sa.get("Yesterday_usage_time", 0)
    else:
        exercises = sa.get("total_exercises", 0)
        exercise_time = sa.get("Total_exercise_time", 0)
        usage_time = sa.get("Total_usage_time", 0)

    return {
        "student_name": sa.get("student_name", "Unknown"),
        "global_kpis": {
            "focus_score": sa.get("average_focus_score", 0),
            "excellence_rate": sa.get("average_success_rate_stars", 0),
            "average_stars": sa.get("average_stars", 0),
            "completion_rate": sa.get("completion_rate", 0),
            "average_mistake_percentage": sa.get("average_mistake_percentage", 0),
        },
        "usage_summary": {
            "total_exercises": exercises,
            "exercise_time_minutes": exercise_time,
            "usage_time_minutes": usage_time,
            "preferred_study_time": sa.get("preferred_study_time", ""),
            "study_time_distribution": sa.get("User_study_time", {}),
            "preferred_study_day": sa.get("Preferred_study_day", ""),
            "study_day_distribution": sa.get("User_study_days", {}),
        },
        "weekly_engagement_trend": sa.get("Exercise_Weekly_Engagment", ""),
        "general_analysis": sa.get("General_Analysis", ""),
        "consistency": sa.get("General_Consistency", ""),
    }
