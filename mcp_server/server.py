from fastmcp import FastMCP
import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional

mcp = FastMCP("CQ_MCP")



# ==========================================
# PYDANTIC SCHEMAS (The "Intelligence" Params)
# ==========================================

class GlobalStats(BaseModel):
    excellence_rate: int = Field(description="Percentage of error-free completions")
    error_rate: int = Field(description="Percentage of incorrect answers")
    focus_score: int = Field(description="Active engagement metric (0-100)")
    perseverance_score: int = Field(description="Consistency and effort metric (0-100)")
    total_time_spent_minutes: int = Field(description="Total duration on platform in minutes")
    global_rank_delta: int = Field(description="How many spots they moved up/down globally")

class SubjectProgress(BaseModel):
    subject_name: str
    stars_earned: int
    stars_total: int
    exercises_completed: int
    completion_percentage: float
    last_accessed_unit_id: str = Field(description="The specific unit they are stuck on or viewing")

class DailyLog(BaseModel):
    date: str
    exercises_attempted: int
    correct_count: int
    wrong_count: int
    time_spent_minutes: int
    new_stars_earned: int = Field(description="Helps detect 'star farming' vs real progress")
    content_difficulty_avg: float = Field(description="Average difficulty from 1.0 to 5.0")
    is_exam_mode: bool = Field(description="True if they were doing T3 Exams")

class DiagnosticGap(BaseModel):
    unit_name: str
    pedagogical_note: str = Field(description="Internal DB flag like 'نتائج ضعيفة وبطاء واضح'")
    failing_exercises: List[str] = Field(description="Specific exercise IDs failed")
    recommended_action: str = Field(description="Link or ID of the exact remediation exercise")
    prerequisite_units_missed: List[str] = Field(description="Units they skipped that caused this failure")

# ==========================================
# MCP TOOLS (The Endpoints for the LLM)
# ==========================================

@mcp.tool()
async def get_student_global_data(student_id: str, timeframe: str = "all_time") -> GlobalStats:
    """
    Use this FIRST to gauge the student's overall performance, focus, and perseverance.
    Timeframe options: 'today', 'yesterday', 'current_week', 'all_time'.
    """
    # TODO: Replace with your actual SQL Database execution
    # query = f"SELECT excellence, error, focus, perseverance FROM global_kpi WHERE student_id = '{student_id}'"
    # data = await db.execute(query)
    
    # Mock return based on your screenshots
    return GlobalStats(
        excellence_rate=55,
        error_rate=3,
        focus_score=53,
        perseverance_score=100,
        total_time_spent_minutes=1240,
        global_rank_delta=-2
    )

@mcp.tool()
async def get_subject_curriculum_progress(student_id: str, subject_id: Optional[str] = None, trimester: Optional[int] = None) -> List[SubjectProgress]:
    """
    Use this to drill down into a specific subject (e.g., 'math', 'arabic') to see star ratios and completion limits.
    """
    # TODO: Execute SQL to get subject specific rows
    
    return [
        SubjectProgress(
            subject_name="Math (رياضيّات)",
            stars_earned=146,
            stars_total=403,
            exercises_completed=146,
            completion_percentage=36.2,
            last_accessed_unit_id="mental_math_t1"
        )
    ]

@mcp.tool()
async def get_daily_activity_logs(student_id: str, start_date: str, end_date: str) -> List[DailyLog]:
    """
    Use this to analyze behavioral patterns chronologically. Essential for checking 'Slowness' or 'Rushing'.
    Dates must be in ISO format (YYYY-MM-DD).
    """
    # TODO: Execute SQL on the activity_logs table
    
    return [
        DailyLog(
            date="2026-02-22",
            exercises_attempted=16,
            correct_count=2,
            wrong_count=14,
            time_spent_minutes=40,
            new_stars_earned=0,
            content_difficulty_avg=3.5,
            is_exam_mode=False
        )
    ]

@mcp.tool()
async def get_diagnostics_and_recommendations(student_id: str, subject_id: Optional[str] = None) -> List[DiagnosticGap]:
    """
    Use this to retrieve the platform's internal pedagogical flags (e.g., Slowness, Concept Gap) and get specific exercise recommendations.
    """
    # TODO: Execute SQL on the remediation/diagnostics table
    
    return [
        DiagnosticGap(
            unit_name="Unit 1: The Two Friends",
            pedagogical_note="نتائج ضعيفة وبطاء واضح",
            failing_exercises=["arabic_read_01", "arabic_listen_04"],
            recommended_action="exercise_mental_speed_01",
            prerequisite_units_missed=["arabic_basics_letters"]
        )
    ]


async def main():
    await mcp.run_async(transport="http", port=8000)

if __name__ == "__main__":
    asyncio.run(main())