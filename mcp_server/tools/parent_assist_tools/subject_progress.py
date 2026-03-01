"""
Tool 2 — Subject Curriculum Progress (Meso View)

Builds per-chapter progress from student_scores.csv.
"""

from typing import Optional

from mcp_server.helpers.database import get_scores, _safe_int


async def build_subject_curriculum_progress(
    student_id: str, subject_id: Optional[int] = None
) -> dict:
    """Return star counts, completion status, and difficulty per chapter."""
    rows = get_scores(subject_id=subject_id)
    if not rows:
        return {"error": f"No scores found for subject_id={subject_id}"}

    # ── Group by (subject_name, chapter_name) ────────────────
    chapters: dict = {}
    for row in rows:
        subj = row.get("subject_name", "Unknown")
        chap = row.get("chapter_name", "Unknown")
        key = (subj, chap)

        if key not in chapters:
            chapters[key] = {
                "subject_name": subj,
                "subject_id": _safe_int(row.get("subject_id")),
                "chapter_name": chap,
                "chapter_id": _safe_int(row.get("chapter_id")),
                "stars_earned": 0,
                "exercises_total": 0,
                "exercises_done": 0,
                "difficulty_levels": set(),
                "has_hard_with_low_stars": False,
            }

        entry = chapters[key]
        stars = _safe_int(row.get("stars")) or 0
        entry["stars_earned"] += stars
        entry["exercises_total"] += 1
        if str(row.get("is_done", "")).strip().lower() == "true":
            entry["exercises_done"] += 1
        diff = row.get("difficulty_level", "").strip().lower()
        if diff:
            entry["difficulty_levels"].add(diff)
        if diff == "hard" and stars < 2:
            entry["has_hard_with_low_stars"] = True

    # ── Build output ─────────────────────────────────────────
    progress_list = []
    for entry in chapters.values():
        max_stars = entry["exercises_total"] * 3  # max 3 stars per exercise
        completion_pct = (
            round(entry["exercises_done"] / entry["exercises_total"] * 100, 1)
            if entry["exercises_total"]
            else 0
        )
        item = {
            "subject_name": entry["subject_name"],
            "chapter_name": entry["chapter_name"],
            "stars_earned": entry["stars_earned"],
            "stars_total": max_stars,
            "exercises_done": entry["exercises_done"],
            "exercises_total": entry["exercises_total"],
            "completion_percentage": completion_pct,
            "difficulty_levels": sorted(entry["difficulty_levels"]),
        }
        if entry["has_hard_with_low_stars"]:
            item["alert"] = "⚠ Hard exercises with low stars — student may be stuck"
        progress_list.append(item)

    return {"progress_list": progress_list}
