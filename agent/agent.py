"""Core agent orchestrator — DAG-based parallel execution with memory"""

import os
from typing import Dict

from dotenv import load_dotenv

from agent.execution.dag_executor import DAGExecutor
from agent.reasoning.intent_classification import classify_intent
from agent.reasoning.planning import create_dag_plan
from agent.reasoning.synthesis import synthesize_response, generate_chitchat_response
from agent.utils.llm_service import LLMService
from agent.utils.logger import get_logger
from agent.utils.memory import MemoryManager

load_dotenv()
logger = get_logger(__name__)


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
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.memory = MemoryManager(redis_url=redis_url)
        self.tools_schema: Dict = {}
        self._executor: DAGExecutor | None = None
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
            history_text = await self.memory.format_for_prompt(session_id)
            recent_entities = await self.memory.get_recent_entities(session_id)

            # ① Intent gate (LLM #1) ─── short-circuit chitchat / platform_info ──
            intent = classify_intent(query, self.llm, conversation_context=history_text)

            if intent.intent in ("chitchat", "platform_info"):
                response = generate_chitchat_response(query, self.llm, history=history_text)
                await self.memory.add_turn(
                    session_id, query, response,
                    entities=intent.entities, intent=intent.intent,
                )
                return self._ok(response, intent.intent, [])

            # Guard: tools unavailable — skip planning
            if not self.tools_schema:
                logger.warning("No tools loaded — cannot fulfil tool_required intent")
                response = "\u0639\u0630\u0631\u0627\u064b\u060c \u0627\u0644\u062e\u062f\u0645\u0629 \u063a\u064a\u0631 \u0645\u062a\u0648\u0641\u0631\u0629 \u062d\u0627\u0644\u064a\u0627\u064b. \u064a\u0631\u062c\u0649 \u0627\u0644\u0645\u062d\u0627\u0648\u0644\u0629 \u0644\u0627\u062d\u0642\u0627\u064b."
                await self.memory.add_turn(
                    session_id, query, response,
                    entities=intent.entities, intent="error",
                )
                return self._ok(response, "error", [])

            # Merge entities: fresh from intent + recent from memory
            all_entities = list(dict.fromkeys(intent.entities + recent_entities))

            # Use revised_query from classifier (anaphora-resolved, self-contained)
            planning_query = intent.revised_query or query

            # ② DAG planning (LLM #2) ────────────────────────────
            student_id = context.get("student_id") if context else None
            if not student_id:
                student_id = await self.memory.get_student_id(session_id)
            plan = create_dag_plan(
                query=planning_query,
                entities=all_entities,
                tools_schema=self.tools_schema,
                llm=self.llm,
                student_id=student_id,
                history=history_text,
            )

            if not plan.tasks:
                response = generate_chitchat_response(query, self.llm, history=history_text)
                await self.memory.add_turn(
                    session_id, query, response,
                    entities=intent.entities, intent="tool_required",
                )
                return self._ok(response, "tool_required", [])

            # ③ DAG execution (parallel, 0 LLM calls) ────────────
            if self._executor is None:
                self._executor = DAGExecutor(self.mcp_client, self.tools_schema)
            tool_results = await self._executor.execute(plan.tasks)

            # ④ Synthesis (LLM #3) ───────────────────────────────
            response = synthesize_response(query, tool_results, self.llm, history=history_text)

            tools_used = [t.tool for t in plan.tasks]
            await self.memory.add_turn(
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