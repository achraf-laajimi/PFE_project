"""DAG executor — runs TaskNodes in parallel with dependency resolution"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

from agent.reasoning.validation import validate_tool_params
from agent.utils.schemas import TaskNode, ToolResult

logger = logging.getLogger(__name__)

# Per-task timeout (seconds). Prevents a single stuck MCP call from blocking everything.
_TASK_TIMEOUT = 30

# Max retries for a failed tool call before giving up.
_MAX_RETRIES = 2

# Base delay for exponential backoff (seconds). Retry 1 = 1s, Retry 2 = 2s.
_BACKOFF_BASE = 1

# Errors that should NOT be retried (permanent failures).
_PERMANENT_ERRORS = (
    "not found",
    "validation",
    "missing required",
    "invalid parameter",
    "unauthorized",
    "forbidden",
)


class DAGExecutor:
    """
    Executes a list of TaskNodes as a DAG.

    - Independent tasks (depends_on=null) run in parallel via asyncio.gather.
    - Dependent tasks wait on an asyncio.Event set by their parent.
    - $-references in params are resolved from earlier task outputs.
    - Each task is validated against tools_schema before calling MCP.
    - Failed tasks are retried up to _MAX_RETRIES times.
    """

    def __init__(self, mcp_client, tools_schema: Dict):
        self.mcp_client = mcp_client
        self.tools_schema = tools_schema

    async def execute(self, tasks: List[TaskNode]) -> List[ToolResult]:
        """Run the full DAG and return results."""
        # task_id → {"raw": str, "json": dict|None, "mapped": dict}
        results: Dict[str, Dict[str, Any]] = {}
        events: Dict[str, asyncio.Event] = {t.id: asyncio.Event() for t in tasks}
        tool_results: List[ToolResult] = []

        async def _run(task: TaskNode):
            # ── Wait for dependency ──────────────────────────
            if task.depends_on:
                dep_event = events.get(task.depends_on)
                if dep_event:
                    try:
                        await asyncio.wait_for(dep_event.wait(), timeout=_TASK_TIMEOUT)
                    except asyncio.TimeoutError:
                        _fail(task, f"Timeout waiting for dependency '{task.depends_on}'")
                        events[task.id].set()
                        return

                    # If the dependency failed, skip this task
                    if results.get(task.depends_on) is None:
                        _fail(task, f"Dependency '{task.depends_on}' failed — skipping")
                        events[task.id].set()
                        return

            # ── Resolve $-references ─────────────────────────
            params = self._resolve_params(task.params, results)

            # ── Validate params against schema ───────────────
            is_valid, error_msg = validate_tool_params(
                task.tool, params, self.tools_schema
            )
            if not is_valid:
                _fail(task, f"Validation: {error_msg}")
                events[task.id].set()
                return

            # ── Execute with retry ───────────────────────────
            last_error = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    logger.info(f"[DAG] {task.id} → {task.tool}({params})"
                                + (f" (retry {attempt})" if attempt else ""))

                    result = await asyncio.wait_for(
                        self.mcp_client.call_tool(task.tool, params),
                        timeout=_TASK_TIMEOUT,
                    )
                    raw_text = result.content[0].text if result.content else "{}"
                    parsed_json = self._try_parse_json(raw_text)
                    mapped = self._apply_output_mapping(parsed_json, task.output_mapping)

                    results[task.id] = {"raw": raw_text, "json": parsed_json, "mapped": mapped}
                    events[task.id].set()

                    tool_results.append(ToolResult(
                        tool_name=task.tool,
                        success=True,
                        data={
                            "raw": raw_text,
                            "json": parsed_json,
                            "mapped": mapped,
                        },
                    ))
                    return  # success — exit retry loop

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {_TASK_TIMEOUT}s"
                except Exception as e:
                    last_error = str(e)

                logger.warning(f"[DAG] {task.id} attempt {attempt} failed: {last_error}")

                # ── Intelligent retry logic ───────────────
                # Don't retry permanent errors (bad params, not found, etc.)
                if last_error and self._is_permanent_error(last_error):
                    logger.info(f"[DAG] {task.id} permanent error — skipping retries")
                    break

                # Exponential backoff before next attempt
                if attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE * (2 ** attempt)  # 1s, 2s, 4s...
                    logger.info(f"[DAG] {task.id} backing off {delay}s before retry")
                    await asyncio.sleep(delay)

            # All retries exhausted
            _fail(task, last_error)
            results[task.id] = None
            events[task.id].set()

        def _fail(task: TaskNode, error: str):
            logger.error(f"[DAG] {task.id} FAILED: {error}")
            tool_results.append(ToolResult(
                tool_name=task.tool,
                success=False,
                data={},
                error=error,
            ))

        await asyncio.gather(*[_run(t) for t in tasks])
        return tool_results

    # ── Param resolver ($taskX.result.field) ──────────────────

    @staticmethod
    def _is_permanent_error(error: str) -> bool:
        """Check if an error is permanent (should not be retried)."""
        error_lower = error.lower()
        return any(pe in error_lower for pe in _PERMANENT_ERRORS)

    @staticmethod
    def _resolve_params(
        params: Dict[str, Any], results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Replace $-references with actual values from earlier task results."""
        resolved = {}
        for key, value in params.items():
            resolved[key] = DAGExecutor._resolve_value(value, results)
        return resolved

    @staticmethod
    def _resolve_value(value: Any, results: Dict[str, Dict[str, Any]]) -> Any:
        """
        Recursively resolve $-references in any value type.

        Handles:
          - Strings:  "$t1.output.x" → resolved value
          - Lists:    ["$t1.output.x", "fixed"] → [resolved, "fixed"]
          - Dicts:    {"a": "$t1.output.x"} → {"a": resolved}
          - Other:    returned as-is (int, float, bool, None)
        """
        if isinstance(value, str) and value.startswith("$"):
            return DAGExecutor._deref(value, results)
        if isinstance(value, list):
            return [DAGExecutor._resolve_value(v, results) for v in value]
        if isinstance(value, dict):
            return {k: DAGExecutor._resolve_value(v, results) for k, v in value.items()}
        return value

    @staticmethod
    def _deref(ref: str, results: Dict[str, Dict[str, Any]]) -> Any:
        """
        Resolve a $-reference.

        Supported patterns:
          $t1.result           → full raw text of task t1
          $t1.result.field     → parsed JSON field from task t1
          $t1.result.a.b       → nested field lookup
          $t1.output.<name>    → value from output_mapping
        """
        # Output mapping path
        mapping_match = re.match(r"^\$(\w+)\.output(?:\.(.+))?$", ref)
        if mapping_match:
            task_id = mapping_match.group(1)
            key = mapping_match.group(2)
            payload = results.get(task_id)
            if payload and key:
                return payload.get("mapped", {}).get(key, ref)
            return ref

        # Result path
        match = re.match(r"^\$(\w+)\.result(?:\.(.+))?$", ref)
        if not match:
            return ref

        task_id = match.group(1)
        field_path = match.group(2)  # e.g. "student.name" or None

        payload = results.get(task_id)
        if payload is None:
            return ref

        # If no field requested, return raw string
        if not field_path:
            return payload.get("raw", ref)

        parsed = payload.get("json")
        if parsed is None:
            return ref

        value, ok = DAGExecutor._extract_path(parsed, field_path)
        return value if ok else ref

    # ── JSON helpers ─────────────────────────────────────────

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _apply_output_mapping(parsed: Any, mapping: Dict[str, str]) -> Dict[str, Any]:
        if not parsed or not mapping:
            return {}
        out: Dict[str, Any] = {}
        for name, path in mapping.items():
            value, ok = DAGExecutor._extract_path(parsed, path)
            if ok:
                out[name] = value
        return out

    @staticmethod
    def _extract_path(parsed: Any, path: str) -> tuple[Any, bool]:
        """Navigate dot/bracket paths like data.weaknesses[0].subject"""
        current = parsed
        tokens = re.split(r"\.(?![^\[]*\])", path)  # split on dots not inside []
        for tok in tokens:
            if tok.endswith("]") and "[" in tok:
                # e.g. weaknesses[0]
                m = re.match(r"^(.*)\[(\d+)\]$", tok)
                if not m:
                    return None, False
                key, idx = m.groups()
                try:
                    current = current[key][int(idx)]
                except Exception:
                    return None, False
            else:
                try:
                    current = current[tok]
                except Exception:
                    return None, False
        return current, True
