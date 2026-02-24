"""
Data loader — reads analysis.json + CSV files once at import time.

All functions return plain dicts/lists (no Pydantic) so the MCP tools
can reshape them into their own response schemas.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ANALYSIS_PATH = _DATA_DIR / "analysis.json"
_SCORES_PATH = _DATA_DIR / "student_scores.csv"
_HISTORIES_PATH = _DATA_DIR / "student_score_histories.csv"


# ── In-memory stores (loaded once) ───────────────────────────
_analysis: Dict[str, Any] = {}
_scores: List[Dict[str, Any]] = []
_histories: List[Dict[str, Any]] = []
_loaded = False


def _load_all() -> None:
    """Load every data file into memory. Called once on first access."""
    global _analysis, _scores, _histories, _loaded
    if _loaded:
        return

    # ── analysis.json ────────────────────────────────────────
    try:
        with open(_ANALYSIS_PATH, encoding="utf-8") as f:
            _analysis = json.load(f)
        logger.info(f"Loaded analysis.json ({len(_analysis)} top-level keys)")
    except FileNotFoundError:
        logger.error(f"analysis.json not found at {_ANALYSIS_PATH}")
    except json.JSONDecodeError as e:
        logger.error(f"analysis.json is invalid JSON: {e}")

    # ── student_scores.csv ───────────────────────────────────
    try:
        with open(_SCORES_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _scores = list(reader)
        logger.info(f"Loaded student_scores.csv ({len(_scores)} rows)")
    except FileNotFoundError:
        logger.error(f"student_scores.csv not found at {_SCORES_PATH}")

    # ── student_score_histories.csv ──────────────────────────
    try:
        with open(_HISTORIES_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _histories = list(reader)
        logger.info(f"Loaded student_score_histories.csv ({len(_histories)} rows)")
    except FileNotFoundError:
        logger.error(f"student_score_histories.csv not found at {_HISTORIES_PATH}")

    _loaded = True


# ── Public query helpers ─────────────────────────────────────


def get_student_analysis() -> Dict[str, Any]:
    """Return the full student_analysis block from analysis.json."""
    _load_all()
    return _analysis.get("student_analysis", {})


def get_subject_analysis(subject_id: int) -> Optional[Dict[str, Any]]:
    """
    Return the per-subject analysis dict from analysis.json.
    Keys in the JSON are string-ified subject_ids: "3", "4", "10", "11", "15".
    """
    _load_all()
    sa = _analysis.get("student_analysis", {})
    return sa.get(str(subject_id))


def get_scores(subject_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return rows from student_scores.csv, optionally filtered by subject_id.
    """
    _load_all()
    rows = _scores
    if subject_id is not None:
        rows = [r for r in rows if _safe_int(r.get("subject_id")) == subject_id]
    return rows


def get_histories(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return the most recent *limit* rows from student_score_histories.csv,
    sorted by validated_at descending (empty validated_at → treated as oldest).
    """
    _load_all()

    def _sort_key(row: Dict) -> str:
        v = row.get("validated_at", "") or ""
        return v if v else "0000-00-00"

    sorted_rows = sorted(_histories, key=_sort_key, reverse=True)
    return sorted_rows[:limit]


# ── Tiny helpers ─────────────────────────────────────────────


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int, return None on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val: Any) -> float:
    """Convert to float, return 0.0 on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0
