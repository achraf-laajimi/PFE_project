"""
Shared helper functions for query_tools.
Not exposed as MCP tools — internal use only.
"""

from typing import Any, Dict, List, Optional
from mcp_server.helpers.database import _safe_int


def build_chapter_summaries(scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group a flat list of score rows (from student_scores.csv) by chapter and
    compute per-chapter mastery/completion statistics.

    Returns a list of chapter dicts, each containing:
        chapter_name, subject_name,
        stars_earned, stars_total, mastery_percentage,
        exercises_done, exercises_total, completion_percentage,
        difficulty_levels, has_alert
    """
    chapters_dict: Dict[tuple, Dict[str, Any]] = {}

    for row in scores:
        subj = row.get("subject_name", "Unknown")
        chap = row.get("chapter_name", "Unknown")
        key = (subj, chap)

        if key not in chapters_dict:
            chapters_dict[key] = {
                "subject_name": subj,
                "chapter_name": chap,
                "stars_earned": 0,
                "stars_total": 0,
                "exercises_done": 0,
                "exercises_total": 0,
                "difficulty_levels": set(),
                "has_alert": False,
            }

        entry = chapters_dict[key]
        stars = _safe_int(row.get("stars")) or 0
        entry["stars_earned"] += stars
        entry["stars_total"] += 3  # Max 3 stars per exercise
        entry["exercises_total"] += 1

        if str(row.get("is_done", "")).strip().lower() == "true":
            entry["exercises_done"] += 1

        diff = row.get("difficulty_level", "").strip().lower()
        if diff:
            entry["difficulty_levels"].add(diff)

        if diff == "hard" and stars < 2:
            entry["has_alert"] = True

    chapters: List[Dict[str, Any]] = []
    for entry in chapters_dict.values():
        mastery_pct = (
            (entry["stars_earned"] / entry["stars_total"] * 100)
            if entry["stars_total"]
            else 0
        )
        completion_pct = (
            (entry["exercises_done"] / entry["exercises_total"] * 100)
            if entry["exercises_total"]
            else 0
        )
        chapters.append(
            {
                "chapter_name": entry["chapter_name"],
                "subject_name": entry["subject_name"],
                "stars_earned": entry["stars_earned"],
                "stars_total": entry["stars_total"],
                "mastery_percentage": round(mastery_pct, 1),
                "exercises_done": entry["exercises_done"],
                "exercises_total": entry["exercises_total"],
                "completion_percentage": round(completion_pct, 1),
                "difficulty_levels": sorted(entry["difficulty_levels"]),
                "has_alert": entry["has_alert"],
            }
        )

    return chapters
