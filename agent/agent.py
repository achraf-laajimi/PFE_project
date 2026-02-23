"""Core agent orchestrator — DAG-based parallel execution with memory"""

import logging
from typing import Dict

from agent.execution.dag_executor import DAGExecutor
from agent.reasoning.intent_classification import classify_intent
from agent.reasoning.planning import create_dag_plan
from agent.reasoning.synthesis import synthesize_response, generate_chitchat_response
from agent.utils.llm_service import LLMService
from agent.utils.memory import MemoryManager

logger = logging.getLogger(__name__)


class ClassQuizAgent:
    """
    Agent orchestrator (max 3 LLM calls per query).

    Flow:
      1. Intent classification    → chitchat | tool_required
      2. DAG planning             → single graph with dependency + output_mapping
      3. DAG execution (parallel) → DAGExecutor handles retries, timeouts, validation
      4. Synthesis                → natural-language response from results
    """

    def __init__(self, mcp_client, llm_model: str = "gpt-4o-mini"):
        self.mcp_client = mcp_client
        self.llm = LLMService(model=llm_model)
        self.memory = MemoryManager()
        self.tools_schema: Dict = {}
        self._executor: DAGExecutor | None = None
        logger.info("ClassQuizAgent initialized")

    # ── Startup ───────────────────────────────────────────────

    async def initialize(self):
        """Fetch MCP tool definitions and build the executor."""
        try:
            tools_list = await self.mcp_client.list_tools()
            self.tools_schema = self._build_schema(tools_list)
            self._executor = DAGExecutor(self.mcp_client, self.tools_schema)
            logger.info(f"Loaded {len(self.tools_schema)} tools from MCP server")
        except Exception as e:
            logger.error(f"Failed to load tools schema: {e}")
            self.tools_schema = {}
            self._executor = DAGExecutor(self.mcp_client, {})

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

        try:
            # Gather recent context from memory
            history_text = self.memory.format_for_prompt(session_id)
            recent_entities = self.memory.get_recent_entities(session_id)

            # ① Intent gate (LLM #1) ─── short-circuit chitchat ──
            intent = classify_intent(query, self.llm, conversation_context=history_text)

            if intent.intent == "chitchat":
                response = generate_chitchat_response(query, self.llm)
                self.memory.add_turn(
                    session_id, query, response,
                    entities=intent.entities, intent="chitchat",
                )
                return self._ok(response, "chitchat", [])

            # Merge entities: fresh from intent + recent from memory
            all_entities = list(dict.fromkeys(intent.entities + recent_entities))

            # ② DAG planning (LLM #2) ────────────────────────────
            student_id = context.get("student_id") if context else None
            plan = create_dag_plan(
                query=query,
                entities=all_entities,
                tools_schema=self.tools_schema,
                llm=self.llm,
                student_id=student_id,
                history=history_text,
            )

            if not plan.tasks:
                response = generate_chitchat_response(query, self.llm)
                self.memory.add_turn(
                    session_id, query, response,
                    entities=intent.entities, intent="tool_required",
                )
                return self._ok(response, "tool_required", [])

            # ③ DAG execution (parallel, 0 LLM calls) ────────────
            if self._executor is None:
                self._executor = DAGExecutor(self.mcp_client, self.tools_schema)
            tool_results = await self._executor.execute(plan.tasks)

            # ④ Synthesis (LLM #3) ───────────────────────────────
            response = synthesize_response(query, tool_results, self.llm)

            tools_used = [t.tool for t in plan.tasks]
            self.memory.add_turn(
                session_id, query, response,
                entities=intent.entities, intent="tool_required",
                tools_used=tools_used,
            )
            return self._ok(response, "tool_required", tools_used)

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
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