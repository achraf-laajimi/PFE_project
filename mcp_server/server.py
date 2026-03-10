"""
ClassQuiz MCP Server — query-specific, context-aware tools.

Tool business logic lives in mcp_server/tools/query_tools/.
This file only registers tools with FastMCP and delegates to those modules.

Start with:  python -m mcp_server.server
"""

import asyncio
import contextlib
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load mcp_server/.env before importing fastmcp so FastMCP settings are applied.
load_dotenv(Path(__file__).with_name(".env"))

from fastmcp import FastMCP, Context
from mcp_server.gateway.auth import get_jwt_verifier
from mcp_server.gateway.middleware import enforce_tool_access

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

mcp = FastMCP("CQ_MCP", auth=get_jwt_verifier())



# ── TIER 1 — Identity ─────────────────────────────────────────────────────────

@mcp.tool()
async def get_student_identity(student_id: str) -> dict:
    """
    Verify the student exists and return basic profile info (name, gender).

    Use FIRST when the query references a student by name and you need to
    confirm their identity before calling any other tool.
    """
    if sec := enforce_tool_access("get_student_identity", {"student_id": student_id}):
        return sec
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
    if sec := enforce_tool_access(
        "get_subject_performance",
        {"student_id": student_id, "subject_id": subject_id},
    ):
        return sec
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
    if sec := enforce_tool_access(
        "get_subject_roadmap",
        {"student_id": student_id, "subject_id": subject_id},
    ):
        return sec
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
    if sec := enforce_tool_access("compare_subjects", {"student_id": student_id}):
        return sec
    return await build_compare_subjects(student_id)


# ── TIER 4 — Behavioural / Temporal ──────────────────────────────────────────

@mcp.tool()
async def get_study_habits(student_id: str) -> dict:
    """
    Study-time patterns, focus score, consistency, and weekly engagement trend.

    Use when: query asks about when / how the student studies.

    Example: "وقتاش يقرا أحمد؟"
    """
    if sec := enforce_tool_access("get_study_habits", {"student_id": student_id}):
        return sec
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
    if sec := enforce_tool_access(
        "get_recent_activity",
        {"student_id": student_id, "limit": limit},
    ):
        return sec
    return await build_recent_activity(student_id, limit)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    logger.info("Starting CQ_MCP server on port 8000 …")
    await mcp.run_async(transport="http", port=8000)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt): asyncio.run(main())