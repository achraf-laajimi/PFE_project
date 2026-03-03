"""
ClassQuiz MCP Server — query-specific, context-aware tools.

Tool business logic lives in mcp_server/tools/query_tools/.
This file only registers tools with FastMCP and delegates to those modules.

Start with:  python -m mcp_server.server
"""

from fastmcp import FastMCP, Context
import asyncio
from typing import Optional
from mcp_server.helpers.validation import is_valid_student_id

from mcp_server.helpers.logger import get_logger
from mcp_server.tools.parent_agent_tools import (
    build_student_identity,
    build_subject_performance,
    build_subject_roadmap,
    build_compare_subjects,
    build_study_habits,
    build_recent_activity,
)

logger = get_logger(__name__)

mcp = FastMCP("CQ_MCP")


# ── Security guard ────────────────────────────────────────────────────────────

def _check_student_id(student_id: str, tool_name: str) -> Optional[dict]:
    """Return an error dict if student_id is invalid, else None."""
    if not is_valid_student_id(student_id):
        logger.error(f"[Security] {tool_name}: unresolved alias '{student_id}'")
        return {"error": "Internal error: student identity could not be resolved."}
    return None


# ── TIER 1 — Identity ─────────────────────────────────────────────────────────

@mcp.tool()
async def get_student_identity(student_id: str) -> dict:
    """
    Verify the student exists and return basic profile info (name, gender).

    Use FIRST when the query references a student by name and you need to
    confirm their identity before calling any other tool.
    """
    if err := _check_student_id(student_id, "get_student_identity"):
        return err
    return await build_student_identity(student_id)


# ── TIER 2 — Subject-Specific ─────────────────────────────────────────────────

@mcp.tool()
async def get_subject_performance(student_id: str, subject_id: int) -> dict:
    """
    Complete subject performance in one focused call.

    Returns: KPIs (stars, completion, focus, excellence), top-3 strongest
    chapters, top-3 weakest chapters (with alert flag), and a diagnostic
    summary.

    Use when: query mentions a specific subject.
    subject_id: 3=Science, 4=Math, 10=Français, 11=Arabic, 15=English.

    Example: "كيفاش أحمد في الفرنسية؟" → subject_id=10
    """
    if err := _check_student_id(student_id, "get_subject_performance_snapshot"):
        return err
    return await build_subject_performance(student_id, subject_id)


@mcp.tool()
async def get_subject_roadmap(student_id: str, subject_id: int, ctx: Context) -> dict:
    """
    Prioritised action plan for improving a specific subject.

    Returns: ordered roadmap steps (HIGH / MEDIUM priority) with chapter
    names, exercises remaining, and time estimates.

    Use when: query asks for recommendations, a roadmap, or "what to work on".
    subject_id: 3=Science, 4=Math, 10=Français, 11=Arabic, 15=English.

    Example: "recommandili roadmap bch tethassen fel français" → subject_id=10
    """
    if err := _check_student_id(student_id, "get_subject_roadmap"):
        return err
    return await build_subject_roadmap(student_id, subject_id, ctx)


# ── TIER 3 — Comparative ──────────────────────────────────────────────────────

@mcp.tool()
async def compare_subjects(student_id: str) -> dict:
    """
    Rank all subjects by completion rate (best → worst).

    Returns: ranked subject list, strongest subject, weakest subject.

    Use when: query asks for an overall progress overview or
    cross-subject comparison.

    Example: "كيفاش أحمد في كل المواد؟"
    """
    if err := _check_student_id(student_id, "compare_subjects"):
        return err
    return await build_compare_subjects(student_id)


# ── TIER 4 — Behavioural / Temporal ──────────────────────────────────────────

@mcp.tool()
async def get_study_habits(student_id: str) -> dict:
    """
    Study-time patterns, focus score, consistency, and weekly engagement trend.

    Use when: query asks about when / how the student studies.

    Example: "وقتاش يقرا أحمد؟"
    """
    if err := _check_student_id(student_id, "get_study_habits"):
        return err
    return await build_study_habits(student_id)


@mcp.tool()
async def get_recent_activity(student_id: str, limit: int = 10) -> dict:
    """
    Chronological exercise timeline (most recent *limit* attempts).

    Each entry includes timestamp, time_seconds, stars, mistakes, and a
    '⚠ High effort, low reward' flag when time > 300 s AND stars < 2.

    Use when: query asks about recent work or today's session.

    Example: "شنوة خدم أحمد اليوم؟"
    """
    if err := _check_student_id(student_id, "get_recent_activity"):
        return err
    return await build_recent_activity(student_id, limit)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    logger.info("Starting CQ_MCP server on port 8000 …")
    await mcp.run_async(transport="http", port=8000)


if __name__ == "__main__":
    asyncio.run(main())