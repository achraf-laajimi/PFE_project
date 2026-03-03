"""DAG-based tool planning (single LLM call → execution graph)"""

from agent.utils.schemas import DAGPlan
from agent.utils.llm_service import LLMService
from agent.utils.logger import get_logger
from datetime import date
from functools import lru_cache
from pathlib import Path
import json

logger = get_logger(__name__)


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
        # Shadow-ID: the LLM only ever sees the alias; the real ID is
        # injected back into params after the LLM call (see below).
        student_id="CURRENT_STUDENT",
        history=history or "(no previous conversation)",
        current_date=date.today().isoformat(),
    )

    try:
        response = llm.generate(prompt=prompt, temperature=0.2, max_tokens=2000)
        result_dict = llm.extract_json(response)
        plan = DAGPlan(**result_dict)

        # ── Alias resolution ────────────────────────────────────────────
        # Replace every "CURRENT_STUDENT" placeholder the LLM wrote into
        # task params with the real student_id before DAG execution.
        # The real ID is never passed to the LLM; it travels out-of-band.
        if student_id and student_id != "CURRENT_STUDENT":
            for task in plan.tasks:
                task.params = {
                    k: (student_id if v == "CURRENT_STUDENT" else v)
                    for k, v in task.params.items()
                }

        logger.info(f"DAG plan ready: {len(plan.tasks)} task(s)")
        return plan
    except Exception as e:
        logger.warning(f"DAG planning failed: {e}")
        return DAGPlan(reasoning=f"Planning failed: {e}", tasks=[])