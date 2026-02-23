"""DAG-based tool planning (single LLM call → execution graph)"""

from agent.utils.schemas import DAGPlan
from agent.utils.llm_service import LLMService
from functools import lru_cache
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    """Load planning prompt template (cached)"""
    path = Path(__file__).parent.parent / "prompts" / "planning.txt"
    return path.read_text(encoding="utf-8")


def create_dag_plan(
    query: str,
    entities: list,
    tools_schema: dict,
    llm: LLMService,
    student_id: str = None,
    history: str = "",
) -> DAGPlan:
    """
    Generate a DAG execution plan in a single LLM call.

    Returns:
        DAGPlan with TaskNode list (independent tasks run in parallel,
        dependent tasks declare their parent via depends_on).
    """
    logger.info(f"Building DAG plan | entities={entities}")

    prompt_template = _load_prompt()
    prompt = prompt_template.format(
        tools_schema=json.dumps(tools_schema, indent=2, ensure_ascii=False),
        query=query,
        entities=json.dumps(entities, ensure_ascii=False),
        student_id=student_id or "UNKNOWN",
        history=history or "(no previous conversation)",
    )

    try:
        response = llm.generate(prompt=prompt, temperature=0.2)
        result_dict = llm.extract_json(response)
        plan = DAGPlan(**result_dict)
        logger.info(f"DAG plan ready: {len(plan.tasks)} task(s)")
        return plan
    except Exception as e:
        logger.warning(f"DAG planning failed: {e}")
        return DAGPlan(reasoning=f"Planning failed: {e}", tasks=[])