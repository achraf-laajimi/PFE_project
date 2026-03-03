"""
TIER 1 — Student Identity

Purpose : Verify the student exists and return basic profile info ONLY.
Use when: First call to validate student_id before any deeper query.
"""

from mcp_server.helpers.database import get_student_analysis
from mcp_server.helpers.logger import get_logger

logger = get_logger(__name__)


async def build_student_identity(student_id: str) -> dict:
    """
    Return minimal student profile: id, name, gender.

    Example query: "كيفاش أحمد؟" → validate identity first.
    """
    sa = get_student_analysis()
    if not sa:
        logger.warning(f"[student_identity] No analysis data found for student '{student_id}'")
        return {"error": "Student not found"}

    return {
        "student_id": sa.get("user_id"),
        "student_name": sa.get("student_name", "Unknown"),
        "gender": sa.get("student_gender", "Unknown"),
    }
