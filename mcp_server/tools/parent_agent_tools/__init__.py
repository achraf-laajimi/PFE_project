"""
Query-specific, context-aware MCP tools.

Each module exposes a single async build_* function.
The @mcp.tool() registration lives in mcp_server/server.py.
"""

from .student_identity import build_student_identity
from .subject_performance import build_subject_performance
from .subject_roadmap import build_subject_roadmap
from .compare_subjects import build_compare_subjects
from .study_habits import build_study_habits
from .recent_activity import build_recent_activity

__all__ = [
    "build_student_identity",
    "build_subject_performance",
    "build_subject_roadmap",
    "build_compare_subjects",
    "build_study_habits",
    "build_recent_activity",
]
