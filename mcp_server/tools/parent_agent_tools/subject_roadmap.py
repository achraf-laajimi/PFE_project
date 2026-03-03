"""
TIER 2 — Subject Roadmap

Purpose : Build a prioritised action plan for subject improvement.
Use when: Query asks for recommendations, roadmap, or "what to work on".

Sampling note
─────────────
This tool uses MCP Sampling: instead of calling an LLM SDK directly, it
delegates to the *client's* LLM via `ctx.sample()`.  FastMCP injects the
`Context` object automatically when the parameter is annotated `ctx: Context`.
"""

from fastmcp import Context
from mcp_server.helpers.database import get_scores, get_subject_analysis
from mcp_server.helpers.logger import get_logger
from mcp_server.tools.parent_agent_tools._helpers import build_chapter_summaries

logger = get_logger(__name__)


async def build_subject_roadmap(student_id: str, subject_id: int, ctx: Context) -> dict:
    # --- 1. GATHER RAW DATA ---
    scores = get_scores(subject_id=subject_id)
    subj = get_subject_analysis(subject_id)
    if not scores or not subj:
        return {"error": "Insufficient data"}

    chapters = build_chapter_summaries(scores)

    # --- 2. PREPARE EVIDENCE FOR THE AI ---
    evidence = {
        "subject": subj.get("subject_name"),
        "stuck_chapters": [c["chapter_name"] for c in chapters if c["has_alert"]],
        "low_completion": [c["chapter_name"] for c in chapters if c["completion_percentage"] < 50],
        "strengths": subj.get("strengths", {}).get("messages", [])[:2],
    }

    # --- 3. SAMPLING CALL ---
    # ctx.sample() sends a message to the CLIENT's LLM and returns the reply.
    # The server never touches an LLM SDK — it stays model-independent.
    prompt = (
        f"Based on this data for {evidence['subject']}: "
        f"Stuck on: {evidence['stuck_chapters']}. "
        f"Low completion: {evidence['low_completion']}. "
        f"Strengths: {evidence['strengths']}. "
        "Create a professional actionable roadmap for the user. "
        "Explain WHY each step matters pedagogically."
    )

    result = await ctx.sample(
        messages=prompt,
        system_prompt="You are a senior pedagogical expert. Keep advice brief and encouraging.",
    )

    return {
        "subject_name": evidence["subject"],
        "ai_generated_roadmap": result.text,
        "raw_stats": {
            "total_estimated_time": len(evidence["stuck_chapters"] + evidence["low_completion"]) * 15,
        },
    }
