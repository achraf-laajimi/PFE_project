"""
ClassQuiz MCP Server — real tools backed by analysis.json + CSV files.

Tool business logic lives in mcp_server/tools/parent_assist_tools/.
This file registers the tools with FastMCP and delegates to those modules.

Start with:  python -m mcp_server.server
"""

from fastmcp import FastMCP
import asyncio
from typing import Optional

from mcp_server.helpers.logger import get_logger
from mcp_server.tools.parent_assist_tools import (
    build_student_global_data,
    build_subject_curriculum_progress,
    build_daily_activity_logs,
    build_diagnostics_and_recommendations,
)

logger = get_logger(__name__)

mcp = FastMCP("CQ_MCP")


# ══════════════════════════════════════════════════════════════
# TOOL 1 — Global "Health Score" (Macro View)
# ══════════════════════════════════════════════════════════════


@mcp.tool()
async def get_student_global_data(student_id: str, timeframe: str = "total") -> dict:
    """
    Use this FIRST to gauge the student's overall performance, focus,
    perseverance, and study habits.
    Timeframe options: 'today', 'yesterday', 'total' (default).
    Returns global KPIs, study-time distribution, and weekly engagement trend.
    """
    return await build_student_global_data(student_id, timeframe)


# ══════════════════════════════════════════════════════════════
# TOOL 2 — Subject Curriculum Progress (Meso View)
# ══════════════════════════════════════════════════════════════


@mcp.tool()
async def get_subject_curriculum_progress(
    student_id: str, subject_id: Optional[int] = None
) -> dict:
    """
    Use this to drill down into a specific subject's chapters.
    Shows star counts, completion status, and difficulty per chapter.
    subject_id examples: 4 (Math), 11 (Arabic), 3 (Science), 10 (Français).
    Omit subject_id to get ALL subjects.
    """
    return await build_subject_curriculum_progress(student_id, subject_id)


# ══════════════════════════════════════════════════════════════
# TOOL 3 — Daily Activity Logs (Timeline View)
# ══════════════════════════════════════════════════════════════


@mcp.tool()
async def get_daily_activity_logs(student_id: str, limit: int = 20) -> dict:
    """
    Use this to analyze behavioral patterns chronologically.
    Returns the *limit* most recent exercise attempts with time spent,
    mistakes, and stars. Flags 'High Effort / Low Reward' entries
    (time > 300s AND stars < 2).
    """
    return await build_daily_activity_logs(student_id, limit)


# ══════════════════════════════════════════════════════════════
# TOOL 4 — Diagnostics & Recommendations (Actionable View)
# ══════════════════════════════════════════════════════════════


@mcp.tool()
async def get_diagnostics_and_recommendations(
    student_id: str, subject_id: int
) -> dict:
    """
    Use this to get the platform's pedagogical diagnosis and remediation
    advice for a specific subject. Returns strengths, difficulties,
    and the recommendation message.
    subject_id examples: 4 (Math), 11 (Arabic), 3 (Science), 10 (Français), 15 (English).
    """
    return await build_diagnostics_and_recommendations(student_id, subject_id)


# ══════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════


async def main():
    logger.info("Starting CQ_MCP server on port 8000 ...")
    await mcp.run_async(transport="http", port=8000)


if __name__ == "__main__":
    asyncio.run(main())