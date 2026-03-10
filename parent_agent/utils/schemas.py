"""Pydantic schemas for agent reasoning"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class IntentClassification(BaseModel):
    """Intent detection result"""
    intent: Literal["chitchat", "platform_info", "tool_required", "pii_request"]
    reasoning: str
    subject: Optional[str] = None
    entities: Dict[str, List[str]] = Field(
        default_factory=lambda: {"student": [], "kinship": [], "subject": []}
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    revised_query: Optional[str] = None  # Expanded query for anaphora resolution / planner input


class TaskNode(BaseModel):
    """Single node in the execution DAG"""
    id: str
    tool: str
    params: Dict[str, Any]
    depends_on: Optional[str] = None
    output_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps fields from this task's result into named variables "
            "that downstream tasks can reference via $taskId.result.field. "
            "Example: {'weakest_subject': 'data.weaknesses[0].subject'}"
        ),
    )


class DAGPlan(BaseModel):
    """Directed Acyclic Graph of tool calls"""
    reasoning: str
    tasks: List[TaskNode] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Result from tool execution"""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None