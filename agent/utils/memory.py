"""
Redis-based sliding-window conversation memory.

Architecture: Sliding Window + Structured Session State (Redis)

────────────────────────────────────────────────────────────

1) Sliding Window (Short-Term Conversation)
   - Redis List per session
   - LPUSH + LTRIM to maintain fixed window size
   - Stores raw turns (query/response/entities/intent/tools_used)

2) Structured Session State (Deterministic Runtime Context)
   - Redis Hash per session
   - Stores active_student, active_subject, last_subject, topic, last_intent
   - Used for deterministic context resolution (no LLM guessing)

3) TTL
   - Entire session expires automatically after inactivity
   - No background cleanup needed

No disk persistence.
Redis is the single source of truth for session memory.
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis.asyncio as redis

from agent.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_WINDOW_SIZE = 7
_SESSION_TTL_SECONDS = 60 * 60 * 24 * 3


class MemoryManager:
    """
    Redis-backed short-term session memory with async support.

    Public API
    ──────────
    add_turn(session_id, ...)          → Store conversation turn
    get_history(session_id)            → Get sliding window (oldest→newest)
    update_state(session_id, data)     → Update session state
    get_state(session_id)              → Get session state dict
    get_student_id(session_id)         → Get active student ID
    format_for_prompt(session_id)      → Format conversation for LLM
    clear(session_id)                  → Clear session memory
    close()                            → Close Redis connection
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        window_size: int = _WINDOW_SIZE,
    ):
        self._window_size = window_size
        self._redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

    # ────────────────────────────────────────────────────────────────────────
    # Connection Management
    # ────────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish async Redis connection."""
        if self.redis is None:
            self.redis = await redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis.ping()
            logger.info(f"[Memory] Connected to Redis at {self._redis_url}")

    async def close(self) -> None:
        """Close Redis connection gracefully."""
        if self.redis:
            await self.redis.close()
            logger.info("[Memory] Redis connection closed")

    # ────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _history_key(self, session_id: str) -> str:
        return f"session:{session_id}:history"

    def _state_key(self, session_id: str) -> str:
        return f"session:{session_id}:state"

    def _meta_key(self, session_id: str) -> str:
        return f"session:{session_id}:meta"

    def _student_sessions_key(self, student_id: str) -> str:
        """Sorted set of session_ids for a student (score = unix timestamp)."""
        return f"student:{student_id}:sessions"

    async def _touch_ttl(self, session_id: str) -> None:
        """
        Reset TTL on history, state, and meta.
        Keeps session alive while active.
        """
        if not self.redis:
            return
        await self.redis.expire(self._history_key(session_id), _SESSION_TTL_SECONDS)
        await self.redis.expire(self._state_key(session_id), _SESSION_TTL_SECONDS)
        await self.redis.expire(self._meta_key(session_id), _SESSION_TTL_SECONDS)

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    async def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        entities: Optional[List[str]] = None,
        intent: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
    ) -> None:
        """
        Append a completed turn to Redis sliding window.
        Uses LPUSH + LTRIM for O(1) window maintenance.
        
        Note: LPUSH stores newest first, so get_history() reverses
        the list to show oldest→newest (chronological order).
        """
        if not self.redis:
            logger.warning("[Memory] Redis not connected, skipping add_turn")
            return

        turn = {
            "query": query,
            "response": response,
            "entities": entities or [],
            "intent": intent,
            "tools_used": tools_used or [],
        }

        key = self._history_key(session_id)

        # Push newest at head (LPUSH = prepend)
        await self.redis.lpush(key, json.dumps(turn, ensure_ascii=False))

        # Trim to window size (keep indices 0 to window_size-1)
        await self.redis.ltrim(key, 0, self._window_size - 1)

        await self._touch_ttl(session_id)

        # Set the session title lazily from the first query
        meta_key = self._meta_key(session_id)
        current_title = await self.redis.hget(meta_key, "title")
        if current_title == "":
            await self.redis.hset(meta_key, "title", query[:60])

        logger.debug(
            f"[Memory] {session_id}: added turn (max {self._window_size})"
        )

    # ────────────────────────────────────────────────────────────────────────

    async def get_history(self, session_id: str) -> List[Dict]:
        """
        Return sliding window in chronological order (oldest → newest).
        
        Redis stores newest first (LPUSH), so we reverse the list
        to match natural conversation flow.
        
        Returns:
            List[Dict]: Turns in chronological order
        """
        if not self.redis:
            return []

        key = self._history_key(session_id)
        raw_turns = await self.redis.lrange(key, 0, -1)

        await self._touch_ttl(session_id)

        # Redis list is newest first (LPUSH), reverse for chronological order
        turns = [json.loads(t) for t in raw_turns]
        turns.reverse()  # Now oldest → newest
        
        return turns

    # ────────────────────────────────────────────────────────────────────────

    async def update_state(self, session_id: str, data: Dict) -> None:
        """
        Merge structured runtime state into Redis Hash.
        
        Handles multi-subject transitions automatically:
        - When active_subject changes, saves previous to last_subject
        
        Example:
            await memory.update_state("123", {
                "active_student": "42",
                "active_subject": "maths",
                "topic": "performance_review"
            })
            
            # Later, when switching subjects:
            await memory.update_state("123", {
                "active_subject": "physics"  # maths → last_subject
            })
        """
        if not self.redis:
            logger.warning("[Memory] Redis not connected, skipping update_state")
            return

        key = self._state_key(session_id)

        # Handle multi-subject transitions
        if "active_subject" in data:
            current_state = await self.redis.hgetall(key)
            current_subject = current_state.get("active_subject")
            new_subject = data["active_subject"]
            
            # If subject is changing, save old one as last_subject
            if current_subject and current_subject != new_subject:
                data["last_subject"] = current_subject
                logger.debug(
                    f"[Memory] {session_id}: subject transition "
                    f"{current_subject} → {new_subject}"
                )

        if data:
            await self.redis.hset(key, mapping=data)

        # Register session in the student's sorted set (first time active_student is set)
        if "active_student" in data:
            student_id = data["active_student"]
            registry_key = self._student_sessions_key(student_id)
            await self.redis.zadd(registry_key, {session_id: int(time.time())}, nx=True)
            await self.redis.expire(registry_key, _SESSION_TTL_SECONDS * 30)  # keep registry 30× longer
            # Init meta hash only if it doesn't exist yet
            meta_key = self._meta_key(session_id)
            if not await self.redis.exists(meta_key):
                await self.redis.hset(meta_key, mapping={
                    "student_id": student_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "title": "",
                })
                await self.redis.expire(meta_key, _SESSION_TTL_SECONDS)

        await self._touch_ttl(session_id)

        logger.debug(
            f"[Memory] {session_id}: updated state keys {list(data.keys())}"
        )

    # ────────────────────────────────────────────────────────────────────────

    async def get_state(self, session_id: str) -> Dict:
        """
        Retrieve structured runtime state.
        
        Returns:
            Dict with keys like:
            - active_student: Current student ID
            - active_subject: Current subject being discussed
            - last_subject: Previous subject (for transitions)
            - topic: Current conversation topic
            - last_intent: Previous intent classification
        """
        if not self.redis:
            return {}

        key = self._state_key(session_id)
        state = await self.redis.hgetall(key)

        await self._touch_ttl(session_id)

        return state or {}

    # ────────────────────────────────────────────────────────────────────────

    async def format_for_prompt(
        self,
        session_id: str,
        max_turns: int = 3,
        include_state: bool = True
    ) -> str:
        """
        Format conversation history and state for LLM context injection.
        
        Returns a formatted string like:
        
        [Session State]
        Student: 42 (Ahmed)
        Subject: Math → Physics (switched)
        Topic: performance_review
        
        [Conversation History]
        User: كيفاش أحمد في الرياضيات؟
        Assistant: أحمد عنده تقدم ممتاز...
        User: وشنوة في الفيزياء؟
        Assistant: الفيزياء فيها...
        
        Args:
            session_id: Session identifier
            max_turns: Max conversation turns to include
            include_state: Whether to include session state section
            
        Returns:
            Formatted string for LLM prompt injection
        """
        sections = []

        # ── Session State ────────────────────────────────────
        if include_state:
            state = await self.get_state(session_id)
            if state:
                state_lines = ["[Session State]"]

                student_id = state.get("active_student")
                if student_id:
                    # The planner resolves CURRENT_STUDENT → real ID after the
                    # LLM call, so it never enters the model's context window.
                    state_lines.append("Student: CURRENT_STUDENT")

                subject = state.get("active_subject")
                last_subject = state.get("last_subject")
                if subject:
                    if last_subject and last_subject != subject:
                        state_lines.append(
                            f"Subject: {last_subject} → {subject} (switched)"
                        )
                    else:
                        state_lines.append(f"Subject: {subject}")
                
                topic = state.get("topic")
                if topic:
                    state_lines.append(f"Topic: {topic}")
                
                if len(state_lines) > 1:  # Has content beyond header
                    sections.append("\n".join(state_lines))

        # ── Conversation History ─────────────────────────────
        history = await self.get_history(session_id)
        
        if not history:
            sections.append(
                "[Conversation History]\n(no previous conversation)"
            )
        else:
            lines = ["[Conversation History]"]
            
            # Get last N turns (already in chronological order)
            recent_turns = history[-max_turns:]
            
            for turn in recent_turns:
                lines.append(f"User: {turn['query']}")
                lines.append(f"Assistant: {turn['response']}")
            
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # ────────────────────────────────────────────────────────────────────────

    async def get_student_id(self, session_id: str) -> Optional[str]:
        """Return the real student_id from session state (used by planner, never by LLM)."""
        state = await self.get_state(session_id)
        student_id = state.get("active_student")
        if student_id:
            return student_id
        # Fallback: scan recent turns for numeric entities
        history = await self.get_history(session_id)
        for turn in reversed(history):
            for entity in turn.get("entities", []):
                if str(entity).isdigit():
                    return str(entity)
        return None

    # ────────────────────────────────────────────────────────────────────────

    async def get_recent_entities(
        self,
        session_id: str,
        n: int = 3
    ) -> List[str]:
        """
        Get unique entities from last N turns (for anaphora resolution).
        
        Example:
            User: "How is Ahmed doing?"  → entities: ["Ahmed"]
            User: "Show me his exercises" → resolve "his" = Ahmed
        
        Args:
            session_id: Session identifier
            n: Number of recent turns to check
            
        Returns:
            List of unique entities (most recent first)
        """
        history = await self.get_history(session_id)
        
        seen = []
        for turn in reversed(history[-n:]):  # Check newest first
            for entity in turn.get("entities", []):
                if entity not in seen:
                    seen.append(entity)
        
        return seen

    # ────────────────────────────────────────────────────────────────────────

    async def list_student_sessions(self, student_id: str) -> List[Dict]:
        """
        Return metadata for every session belonging to *student_id*,
        ordered newest first.

        Returns:
            List of dicts: {session_id, title, created_at, turn_count}
        """
        if not self.redis:
            return []
        registry_key = self._student_sessions_key(student_id)
        # zrevrange → newest scores first
        session_ids = await self.redis.zrevrange(registry_key, 0, -1)
        sessions = []
        for sid in session_ids:
            meta = await self.redis.hgetall(self._meta_key(sid))
            if not meta:
                continue
            turn_count = await self.redis.llen(self._history_key(sid))
            sessions.append({
                "session_id": sid,
                "title": meta.get("title") or "\u0645\u062d\u0627\u062f\u062b\u0629 \u062c\u062f\u064a\u062f\u0629",
                "created_at": meta.get("created_at", ""),
                "turn_count": turn_count,
            })
        return sessions

    # ────────────────────────────────────────────────────────────────────────

    async def clear(self, session_id: str) -> None:
        """
        Completely remove session memory (history + state + meta + registry).
        """
        if not self.redis:
            return

        # Fetch student_id before deleting state
        state = await self.get_state(session_id)
        student_id = state.get("active_student")

        await self.redis.delete(
            self._history_key(session_id),
            self._state_key(session_id),
            self._meta_key(session_id),
        )

        if student_id:
            await self.redis.zrem(self._student_sessions_key(student_id), session_id)

        logger.info(f"[Memory] Cleared session {session_id}")