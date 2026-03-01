"""
Parent-assist tool logic modules.

Each module exposes a single async function containing the business logic
for one MCP tool.  The @mcp.tool() registration stays in mcp_server/server.py.
"""

from .global_data import build_student_global_data
from .subject_progress import build_subject_curriculum_progress
from .activity_logs import build_daily_activity_logs
from .recommandation import build_diagnostics_and_recommendations

__all__ = [
    "build_student_global_data",
    "build_subject_curriculum_progress",
    "build_daily_activity_logs",
    "build_diagnostics_and_recommendations",
]
