"""Session-based conversation memory (in-memory, per student)"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Max turns kept per session (oldest dropped first)
_MAX_HISTORY = 10


class MemoryManager:
    """
    Lightweight conversation memory keyed by session_id.

    Stores the last N turns (query + response + entities) so the agent
    can resolve anaphora ("his", "ses", "lui") by looking at recent context.

    For production, swap the internal dict with Redis or PostgreSQL.
    """

    def __init__(self, max_history: int = _MAX_HISTORY):
        self._max = max_history
        # session_id → list of Turn dicts
        self._store: Dict[str, List[Dict]] = defaultdict(list)

    # ── Public API ─────────────────────────────────────────────

    def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        entities: List[str] = None,
        intent: str = None,
        tools_used: List[str] = None,
    ) -> None:
        """Record a completed turn."""
        turn = {
            "query": query,
            "response": response,
            "entities": entities or [],
            "intent": intent,
            "tools_used": tools_used or [],
        }
        history = self._store[session_id]
        history.append(turn)
        # Evict oldest when over limit
        if len(history) > self._max:
            self._store[session_id] = history[-self._max :]
        logger.debug(f"[Memory] {session_id}: {len(self._store[session_id])} turns stored")

    def get_history(self, session_id: str) -> List[Dict]:
        """Return the full turn history for a session."""
        return list(self._store.get(session_id, []))

    def get_recent_entities(self, session_id: str, n: int = 3) -> List[str]:
        """
        Collect unique entities from the last *n* turns (most-recent-first).

        Useful for resolving pronouns: if the user says "give me his exercises"
        and the last turn mentioned "Ahmed", the planner can use "Ahmed".
        """
        history = self._store.get(session_id, [])
        seen = []
        for turn in reversed(history[-n:]):
            for entity in turn.get("entities", []):
                if entity not in seen:
                    seen.append(entity)
        return seen

    def get_last_student_id(self, session_id: str) -> Optional[str]:
        """
        Walk backwards through history and return the most recent
        student_id that was used in a tool call (stored in context).
        """
        for turn in reversed(self._store.get(session_id, [])):
            # student_id might be stored as an entity or in context
            for entity in turn.get("entities", []):
                # if it looks like a student ID, return it
                if entity.startswith("ST_"):
                    return entity
        return None

    def format_for_prompt(self, session_id: str, max_turns: int = 3) -> str:
        """
        Format recent history as plain text to inject into LLM prompts.

        Returns something like:
            User: كيفاش أحمد في الرياضيات؟
            Assistant: أحمد عنده 85% في الهندسة...
        """
        history = self._store.get(session_id, [])
        if not history:
            return "(no previous conversation)"

        lines = []
        for turn in history[-max_turns:]:
            lines.append(f"User: {turn['query']}")
            lines.append(f"Assistant: {turn['response']}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        """Wipe history for a session."""
        self._store.pop(session_id, None)
        logger.info(f"[Memory] Cleared session {session_id}")
