"""
TIER 2 — Subject Performance Snapshot

Purpose : Complete subject performance in ONE focused call.
Use when: Query mentions a specific subject (math, French, Arabic, …).
"""

from mcp_server.helpers.database import get_subject_analysis, get_scores
from mcp_server.helpers.logger import get_logger
from mcp_server.tools.parent_agent_tools._helpers import build_chapter_summaries

logger = get_logger(__name__)


async def build_subject_performance(student_id: str, subject_id: int) -> dict:
    """
    Return KPIs, top-3 strongest chapters, top-3 weakest (with alert),
    and a diagnostic summary for the given subject.

    Example query: "كيفاش أحمد في الفرنسية؟" → subject_id=10
    """
    subj = get_subject_analysis(subject_id)
    if not subj:
        logger.warning(f"[subject_performance_snapshot] No data for subject_id={subject_id}")
        return {"error": f"No data for subject_id={subject_id}"}

    scores = get_scores(subject_id=subject_id)
    chapters = build_chapter_summaries(scores)

    # Sort by mastery descending
    chapters.sort(key=lambda c: c["mastery_percentage"], reverse=True)

    strongest_chapters = chapters[:3]
    # Weakest = bottom 5, restricted to those with an active alert
    weakest_chapters = [c for c in reversed(chapters[-5:]) if c["has_alert"]][:3]

    return {
        "subject_name": subj.get("subject_name"),
        "total_exercises": subj.get("total_exercises", 0),
        "kpis": {
            "average_stars": subj.get("average_stars", 0),
            "completion_rate": subj.get("completion_rate", 0),
            "focus_score": subj.get("average_focus_score", 0),
            "excellence_rate": subj.get("average_success_rate_stars", 0),
        },
        "strongest_chapters": strongest_chapters,
        "weakest_chapters": weakest_chapters,
        "diagnosis": {
            "strengths": subj.get("strengths", {}).get("messages", []),
            "difficulties": subj.get("difficulties", {}).get("messages", []),
        },
    }
