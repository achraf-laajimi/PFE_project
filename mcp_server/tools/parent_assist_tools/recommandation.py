"""
Tool 4 — Diagnostics & Recommendations (Actionable View)

Builds pedagogical diagnosis and remediation advice for a specific
subject from analysis.json per-subject keys.
"""

from mcp_server.helpers.database import get_subject_analysis


async def build_diagnostics_and_recommendations(
    student_id: str, subject_id: int
) -> dict:
    """Return strengths, difficulties, and the recommendation message."""
    subj = get_subject_analysis(subject_id)
    if not subj:
        return {"error": f"No analysis found for subject_id={subject_id}"}

    # If subject has too few exercises, return the special message
    if "recommandation_message" in subj and "strengths" not in subj:
        return {
            "subject_name": subj.get("subject_name", "Unknown"),
            "total_exercises": subj.get("total_exercises", 0),
            "recommendation": subj.get("recommandation_message", ""),
            "note": "Not enough exercises to generate full diagnostics",
        }

    return {
        "subject_name": subj.get("subject_name", "Unknown"),
        "total_exercises": subj.get("total_exercises", 0),
        "kpis": {
            "average_stars": subj.get("average_stars", 0),
            "completion_rate": subj.get("completion_rate", 0),
            "average_mistakes": subj.get("average_mistakes", 0),
            "success_rate_time": subj.get("average_success_rate_time", 0),
            "success_rate_stars": subj.get("average_success_rate_stars", 0),
            "focus_score": subj.get("average_focus_score", 0),
        },
        "strengths": subj.get("strengths", {}).get("messages", []),
        "difficulties": subj.get("difficulties", {}).get("messages", []),
        "recommendation": subj.get("recommandation_message", ""),
    }
