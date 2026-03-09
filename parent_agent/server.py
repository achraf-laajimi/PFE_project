"""FastAPI server connecting frontend to agent"""

from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from parent_agent.agent import ClassQuizAgent
from parent_agent.utils.mcp_client import MCPClient
from parent_agent.utils.llm_service import LLMService
from parent_agent.utils.logger import get_logger
from parent_agent.reasoning.intent_classification import _load_prompt as _ic_load_prompt
from parent_agent.reasoning.planning import _load_prompt as _plan_load_prompt
from parent_agent.reasoning.synthesis import _load_synthesis_prompt, _load_chitchat_prompt

logger = get_logger(__name__)

# Global state
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage agent and MCP client lifecycle"""
    global agent_instance
    
    try:
        # 1. Shared LLM service (reused by agent + MCP sampling handler)
        llm = LLMService()

        # 2. Initialize HTTP MCP client with mandatory X-API-KEY.
        #    llm= wires the sampling handler so ctx.sample() calls in MCP tools
        #    are forwarded back through the agent's OpenAI connection.
        client = MCPClient(llm=llm)
        await client.connect()  # Open persistent MCP session

        # 3. Initialize Agent (reuse the same LLM instance)
        agent_instance = ClassQuizAgent(mcp_client=client, llm=llm)
        await agent_instance.initialize()
        
        logger.info("✓ ClassQuiz API Ready: Agent connected to MCP over HTTP")
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    
    yield

    # Cleanup: flush any pending tasks, then close MCP session
    try:
        await agent_instance.memory.close()
    except Exception:
        pass
    try:
        await client.disconnect()
    except Exception:
        pass

app = FastAPI(
    title="ClassQuiz ParentAgent API",
    lifespan=lifespan
)

# CORS: Allow your frontend (likely running on a different port/file)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---

class SessionMeta(BaseModel):
    session_id: str
    title: str
    created_at: str
    turn_count: int


class ChatRequest(BaseModel):
    message: str
    student_id: Optional[str] = "239645"  # Default to Chayma for demo if JS doesn't send it
    session_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    response: str
    intent: Optional[str] = "unknown"
    tools_used: List[str] = []
    timestamp: str
    success: bool

# --- Endpoints ---

@app.get("/sessions", response_model=List[SessionMeta])
async def list_sessions(student_id: str):
    """Return metadata for all sessions belonging to a student, newest first."""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    rows = await agent_instance.memory.list_student_sessions(student_id)
    return [
        SessionMeta(
            session_id=r["session_id"],
            title=r["title"],
            created_at=r["created_at"],
            turn_count=r["turn_count"],
        )
        for r in rows
    ]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a session from Redis."""
    global agent_instance
    if agent_instance:
        await agent_instance.memory.clear(session_id)
    return {"deleted": session_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global agent_instance
    
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing...")
    
    try:
        logger.info(f"📩 Message: {request.message} | Student: {request.student_id}")
        
        # Context building
        context = {}
        if request.student_id:
            context["student_id"] = request.student_id
            # Pre-register session state so the agent knows which student it is
            await agent_instance.memory.update_state(
                request.session_id, {"active_student": request.student_id}
            )

        # Process Query
        result = await agent_instance.process_query(
            query=request.message,
            context=context,
            session_id=request.session_id,
        )
        
        return ChatResponse(
            response=result["response"],
            intent=result.get("intent", "unknown"),
            tools_used=result.get("tools_used", []),
            timestamp=datetime.now().strftime("%I:%M %p"),
            success=result.get("success", False)
        )
    
    except Exception as e:
        logger.error(f"Error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────
# DEBUG endpoints  (Postman-friendly, not exposed in production)
# ─────────────────────────────────────────────────────────────

@app.get("/debug/history/{session_id}", tags=["debug"])
async def debug_history(session_id: str):
    """
    Return the full short-term conversation history kept in RAM (+ reloaded
    from disk) for a session.

    Postman: GET http://localhost:5000/debug/history/session_1234567890
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    history = await agent_instance.memory.get_history(session_id)
    return {
        "session_id": session_id,
        "turn_count": len(history),
        "turns": history,
    }


@app.get("/debug/window/{session_id}", tags=["debug"])
async def debug_window(session_id: str):
    """
    Show the sliding-window stats and full turn list for a session.

    Postman: GET http://localhost:5000/debug/window/session_1234567890
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    mem = agent_instance.memory
    history = await mem.get_history(session_id)
    return {
        "session_id":   session_id,
        "window_size":  mem._window_size,
        "turn_count":   len(history),
        "state":        await mem.get_state(session_id),
        "turns":        history,
    }


@app.get("/debug/sessions/all", tags=["debug"])
async def debug_all_sessions():
    """
    List every session found in Redis (all students) via SCAN on meta keys.

    Postman: GET http://localhost:5000/debug/sessions/all
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    r = agent_instance.memory.redis
    if not r:
        return {"sessions": []}

    # Scan for all session meta hashes
    cursor = 0
    meta_keys = []
    while True:
        cursor, keys = await r.scan(cursor, match="session:*:meta", count=200)
        meta_keys.extend(keys)
        if cursor == 0:
            break

    rows = []
    for key in meta_keys:
        meta = await r.hgetall(key)
        # key format: session:<id>:meta
        sid = key.split(":")[1] if ":" in key else key
        turn_count = await r.llen(f"session:{sid}:history")
        rows.append({
            "session_id": sid,
            "student_id": meta.get("student_id"),
            "created_at": meta.get("created_at"),
            "title":      meta.get("title", ""),
            "turn_count": turn_count,
        })
    rows.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"total": len(rows), "sessions": rows}


@app.get("/debug/context/{session_id}", tags=["debug"])
async def debug_context(session_id: str):
    """
    Show the in-memory context dict for a session (student_id, prefs, etc.).

    Postman: GET http://localhost:5000/debug/context/session_1234567890
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    state = await agent_instance.memory.get_state(session_id)
    return {"session_id": session_id, "state": state}


@app.get("/debug/trace/{session_id}", tags=["debug"])
async def debug_trace(session_id: str):
    """
    Return the full pipeline trace captured during the last process_query call
    for the given session.  Includes intent, plan, tool_results, raw_response,
    and any error that occurred.

    Postman: GET http://localhost:5000/debug/trace/session_1234567890
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    trace = agent_instance._traces.get(session_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trace found for session '{session_id}'. "
                   "Run /chat first to generate one.",
        )
    return {"session_id": session_id, "trace": trace}


@app.post("/debug/call-tool/{tool_name}", tags=["debug"])
async def debug_call_tool(tool_name: str, body: dict = None):
    """
    Directly invoke a named MCP tool with the supplied params dict.
    Returns the raw tool result without going through the full agent pipeline.

    Postman:
      POST http://localhost:5000/debug/call-tool/get_student_global_data
      Body (raw JSON): {"student_id": "239645"}
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    params = (body or {})
    try:
        result = await agent_instance.mcp_client.call_tool(tool_name, params)
        return {"tool": tool_name, "params": params, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/debug/cache/clear", tags=["debug"])
async def debug_cache_clear():
    """
    Bust the @lru_cache on all three prompt-loader functions so the next
    request re-reads the .txt files from disk.  Use after editing any
    prompt file without restarting the server.

    Postman: POST http://localhost:5000/debug/cache/clear
    """
    _ic_load_prompt.cache_clear()
    _plan_load_prompt.cache_clear()
    _load_synthesis_prompt.cache_clear()
    _load_chitchat_prompt.cache_clear()
    return {
        "cleared": [
            "intent_classification._load_prompt",
            "planning._load_prompt",
            "synthesis._load_synthesis_prompt",
            "synthesis._load_chitchat_prompt",
        ]
    }


if __name__ == "__main__":
    import uvicorn
    # Runs on port 5000 to match your app.js
    uvicorn.run(app, host="0.0.0.0", port=5000)