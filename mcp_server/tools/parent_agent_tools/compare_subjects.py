"""
TIER 3 — Compare Subjects

Purpose : Rank all subjects by completion rate (best → worst).
Use when: Query asks for an overall progress overview or cross-subject comparison.
"""

from mcp_server.helpers.database import get_student_analysis
from mcp_server.helpers.logger import get_logger

logger = get_logger(__name__)

# Canonical subject IDs present in analysis.json
_SUBJECT_IDS = ["3", "4", "10", "11", "15"]


async def build_compare_subjects(student_id: str) -> dict:
    """
    Return a ranked list of all subjects with key KPIs, plus the
    strongest and weakest subject for quick synthesis.

    Example query: "كيفاش أحمد في كل المواد؟"
    """
    sa = get_student_analysis()
    if not sa:
        logger.warning("[compare_subjects] No analysis data found")
        return {"error": "No analysis data"}

    subjects = []
    for key in _SUBJECT_IDS:
        subj = sa.get(key)
        if not subj or "subject_name" not in subj:
            continue
        subjects.append(
            {
                "subject_name": subj.get("subject_name"),
                "completion_rate": subj.get("completion_rate", 0),
                "average_stars": subj.get("average_stars", 0),
                "total_exercises": subj.get("total_exercises", 0),
                "has_difficulties": len(subj.get("difficulties", {}).get("messages", [])) > 0,
            }
        )

    if not subjects:
        return {"error": "No subject data available"}

    subjects.sort(key=lambda s: s["completion_rate"], reverse=True)

    return {
        "subjects_ranked": subjects,
        "strongest_subject": subjects[0],
        "weakest_subject": subjects[-1],
    }
