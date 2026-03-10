"""Core agent orchestrator — DAG-based parallel execution with memory"""

import os
import re
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from rapidfuzz import fuzz
from unidecode import unidecode

from dotenv import load_dotenv

from parent_agent.execution.dag_executor import DAGExecutor
from parent_agent.reasoning.intent_classification import classify_intent
from parent_agent.reasoning.planning import create_dag_plan
from parent_agent.reasoning.synthesis import synthesize_response, generate_chitchat_response
from parent_agent.utils.llm_service import LLMService
from parent_agent.utils.logger import get_logger
from parent_agent.utils.memory import MemoryManager

load_dotenv()
logger = get_logger(__name__)




def _normalize_name(value: str) -> str:
    """Normalize and keep letters/spaces for multilingual names."""
    if not value:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s\u0600-\u06ff]", "", text)
    return re.sub(r"\s+", " ", text)


def _latinize_name(value: str) -> str:
    """Transliterate to latin and normalize for fuzzy comparison."""
    latin = unidecode(value or "")
    latin = latin.lower().strip()
    latin = re.sub(r"[^a-z0-9\s]", "", latin)
    return re.sub(r"\s+", " ", latin)


def _name_key(value: str) -> str:
    """Stable key for dedup across scripts/spellings."""
    return _latinize_name(_normalize_name(value)).replace(" ", "")


def is_same_name(name_a: str, name_b: str) -> bool:
    """Return True if two names likely refer to the same student."""
    a = _normalize_name(name_a)
    b = _normalize_name(name_b)
    if not a or not b:
        return False
    if a == b:
        return True

    a_lat = _latinize_name(a)
    b_lat = _latinize_name(b)
    if a_lat == b_lat:
        return True

    if fuzz.ratio(a_lat, b_lat) >= 86:
        return True

    a_alt = a_lat.replace("y", "i").replace("ou", "w").replace("ch", "sh")
    b_alt = b_lat.replace("y", "i").replace("ou", "w").replace("ch", "sh")
    if a_alt == b_alt:
        return True

    return fuzz.ratio(a_alt, b_alt) >= 90


def _extract_name_mentions(entities_dict: Dict[str, list]) -> list[str]:
    """Read pre-categorized student names from the intent payload."""
    students = (entities_dict or {}).get("student") or []
    seen: set[str] = set()
    output: list[str] = []
    for name in students:
        candidate = str(name or "").strip()
        norm = _name_key(candidate)
        if candidate and norm and norm not in seen:
            seen.add(norm)
            output.append(candidate)
    return output


def _flatten_entities(entities_dict: Dict[str, list]) -> list[str]:
    """Flatten categorized entities into unique text list for planner/history."""
    ordered = []
    for key in ("student", "kinship", "subject"):
        ordered.extend((entities_dict or {}).get(key) or [])
    seen: set[str] = set()
    unique: list[str] = []
    for value in ordered:
        text = str(value or "").strip()
        if not text:
            continue
        norm = _normalize_name(text)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(text)
    return unique


class ClassQuizAgent:
    """
    Agent orchestrator (max 3 LLM calls per query).

    Flow:
      1. Intent classification    → chitchat | tool_required
      2. DAG planning             → single graph with dependency + output_mapping
      3. DAG execution (parallel) → DAGExecutor handles retries, timeouts, validation
      4. Synthesis                → natural-language response from results
    """

    def __init__(self, mcp_client, llm_model: str = "gpt-5-mini", llm: LLMService = None):
        self.mcp_client = mcp_client
        self.llm = llm if llm is not None else LLMService(model=llm_model)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.memory = MemoryManager(redis_url=redis_url)
        self.tools_schema: Dict = {}
        self._executor: DAGExecutor | None = None
        # Per-session pipeline traces for /debug/trace/<id>
        self._traces: Dict[str, Dict[str, Any]] = {}
        logger.info("ClassQuizAgent initialized")

    # ── Startup ───────────────────────────────────────────────

    async def initialize(self):
        """Connect to Redis, fetch MCP tool definitions and build the executor."""
        try:
            await self.memory.connect()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        try:
            tools_list = await self.mcp_client.list_tools()
            self.tools_schema = self._build_schema(tools_list)
            self._executor = DAGExecutor(self.mcp_client, self.tools_schema)
            logger.info(f"Loaded {len(self.tools_schema)} tools from MCP server")
        except Exception as e:
            logger.error(f"Failed to load tools schema: {e}")
            self.tools_schema = {}
            self._executor = DAGExecutor(self.mcp_client, {})

    async def _get_cached_student_name(self, session_id: str, student_id: str) -> Optional[str]:
        """Resolve student name once via tool and cache it in session state."""
        state = await self.memory.get_state(session_id)
        cached_id = str(state.get("active_student") or "").strip()
        cached_name = str(state.get("active_student_name") or "").strip()
        if cached_id == str(student_id) and cached_name:
            return cached_name

        try:
            result = await self.mcp_client.call_tool("get_student_identity", {"student_id": student_id})
            raw = result.content[0].text if getattr(result, "content", None) else "{}"
            payload = json.loads(raw) if raw else {}
            student_name = str(payload.get("student_name") or "").strip()
            if student_name:
                await self.memory.update_state(
                    session_id,
                    {
                        "active_student": str(student_id),
                        "active_student_name": student_name,
                    },
                )
                return student_name
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{session_id}] student identity cache failed: {e}")
        return None

    @staticmethod
    def _build_schema(tools_list) -> Dict:
        schema = {}
        for tool in tools_list:
            schema[tool.name] = {
                "description": tool.description,
                "params": tool.inputSchema.get("properties", {}),
                "required": tool.inputSchema.get("required", []),
            }
        return schema

    # ── Main entry point ──────────────────────────────────────

    async def process_query(
        self, query: str, context: Dict = None, session_id: str = "default"
    ) -> Dict:
        """
        Process a user query end-to-end.

        Args:
            query:      User message
            context:    Optional dict (student_id, …)
            session_id: Session key for conversation memory

        Returns:
            {"response", "intent", "tools_used", "success"}
        """
        logger.info(f"[{session_id}] Processing: {query}")

        # ─ trace scaffold (filled in as we go) ───────────────────────────
        trace: Dict[str, Any] = {
            "session_id": session_id,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "history_text": None,
            "intent": None,
            "plan": None,
            "tool_results": None,
            "raw_response": None,
            "final_response": None,
            "error": None,
        }

        try:
            # Gather recent context from memory
            history_text = await self.memory.format_for_prompt(session_id)
            recent_entities = await self.memory.get_recent_entities(session_id)
            trace["history_text"] = history_text

            # ① Intent gate (LLM #1) ─── short-circuit chitchat / platform_info ──
            intent = classify_intent(query, self.llm, conversation_context=history_text)
            intent_entities = intent.entities or {"student": [], "kinship": [], "subject": []}
            entity_texts = _flatten_entities(intent_entities)
            trace["intent"] = {
                "intent": intent.intent,
                "confidence": intent.confidence,
                "entities": intent_entities,
                "revised_query": intent.revised_query,
                "reasoning": intent.reasoning,
            }

            if intent.intent == "pii_request":
                response = (
                    "عذراً، لا يمكنني مشاركة المعرّفات الشخصية أو بيانات الحساب. "
                    "إذا كنت بحاجة إلى دعم فني يرجى التواصل مع الإدارة."
                )
                trace["raw_response"] = response
                trace["final_response"] = response
                self._traces[session_id] = trace
                await self.memory.add_turn(
                    session_id, query, response,
                    entities=[], intent="pii_request",
                )
                return self._ok(response, "pii_request", [])

            if intent.intent in ("chitchat", "platform_info"):
                response = generate_chitchat_response(query, self.llm, history=history_text)
                trace["raw_response"] = response
                trace["final_response"] = response
                self._traces[session_id] = trace
                await self.memory.add_turn(
                    session_id, query, response,
                    entities=entity_texts, intent=intent.intent,
                )
                return self._ok(response, intent.intent, [])

            # Guard: tools unavailable — skip planning
            if not self.tools_schema:
                logger.warning("No tools loaded — cannot fulfil tool_required intent")
                response = "\u0639\u0630\u0631\u0627\u064b\u060c \u0627\u0644\u062e\u062f\u0645\u0629 \u063a\u064a\u0631 \u0645\u062a\u0648\u0641\u0631\u0629 \u062d\u0627\u0644\u064a\u0627\u064b. \u064a\u0631\u062c\u0649 \u0627\u0644\u0645\u062d\u0627\u0648\u0644\u0629 \u0644\u0627\u062d\u0642\u0627\u064b."
                await self.memory.add_turn(
                    session_id, query, response,
                    entities=entity_texts, intent="error",
                )
                return self._ok(response, "error", [])

            # Source of truth: trust student_id from request context when provided.
            student_id = context.get("student_id") if context else None
            if not student_id:
                student_id = await self.memory.get_student_id(session_id)

            # Name guard: only when query explicitly mentions a non-kinship name.
            if student_id:
                active_student_name = await self._get_cached_student_name(session_id, student_id)
                mentioned_names = _extract_name_mentions(intent_entities)
                if active_student_name and mentioned_names:
                    mismatched = [
                        name for name in mentioned_names
                        if not is_same_name(name, active_student_name)
                    ]
                    if mismatched:
                        response = (
                            f"عذراً، لا يمكنني الوصول إلا إلى بيانات أبنائك المسجلين فقط. "
                            f"الاسم (**{mismatched}**) غير متاح لي حالياً أو قد يكون مسجلاً ببيانات مختلفة."
                        )
                        trace["raw_response"] = response
                        trace["final_response"] = response
                        self._traces[session_id] = trace
                        await self.memory.add_turn(
                            session_id,
                            query,
                            response,
                            entities=entity_texts,
                            intent="identity_mismatch",
                        )
                        return self._ok(response, "identity_mismatch", [])

            # Avoid stale-name contamination: if current query has explicit student
            # names, do not merge prior-turn entities.
            if _extract_name_mentions(intent_entities):
                all_entities = list(dict.fromkeys(entity_texts))
            else:
                all_entities = list(dict.fromkeys(entity_texts + recent_entities))

            # Use revised_query from classifier (anaphora-resolved, self-contained)
            planning_query = intent.revised_query or query

            # ② DAG planning (LLM #2) ────────────────────────────
            # student_id already resolved above for identity guard
            plan = create_dag_plan(
                query=planning_query,
                entities=all_entities,
                tools_schema=self.tools_schema,
                llm=self.llm,
                student_id=student_id,
                history=history_text,
            )
            trace["plan"] = {
                "reasoning": plan.reasoning,
                "tasks": [
                    {"id": t.id, "tool": t.tool, "params": t.params,
                     "depends_on": t.depends_on}
                    for t in plan.tasks
                ],
            }

            if not plan.tasks:
                response = generate_chitchat_response(query, self.llm, history=history_text)
                trace["raw_response"] = response
                trace["final_response"] = response
                self._traces[session_id] = trace
                await self.memory.add_turn(
                    session_id, query, response,
                    entities=entity_texts, intent="tool_required",
                )
                return self._ok(response, "tool_required", [])

            # ③ DAG execution (parallel, 0 LLM calls) ────────────
            tool_results = await self._executor.execute(plan.tasks)
            trace["tool_results"] = [
                {
                    "tool": r.tool_name,
                    "success": r.success,
                    "result": r.data,
                    "error": r.error,
                }
                for r in tool_results
            ]

            # ④ Synthesis (LLM #3) ───────────────────────────────
            raw_response = synthesize_response(query, tool_results, self.llm, history=history_text)
            response = raw_response
            trace["raw_response"] = raw_response
            trace["final_response"] = response
            tools_used = [t.tool for t in plan.tasks]
            self._traces[session_id] = trace
            await self.memory.add_turn(
                session_id, query, response,
                entities=entity_texts, intent="tool_required",
                tools_used=tools_used,
            )
            return self._ok(response, "tool_required", tools_used)

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            trace["error"] = str(e)
            self._traces[session_id] = trace
            return {
                "response": "عذراً، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى.",
                "intent": "error",
                "tools_used": [],
                "success": False,
            }

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _ok(response: str, intent: str, tools_used: list) -> Dict:
        return {
            "response": response,
            "intent": intent,
            "tools_used": tools_used,
            "success": True,
        }