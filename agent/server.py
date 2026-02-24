"""FastAPI server connecting frontend to agent"""

import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from agent.agent import ClassQuizAgent
from agent.client import MCPClient
from mcp_server.server import mcp  # Direct import of local MCP server logic

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
agent_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage agent and MCP client lifecycle"""
    global agent_instance
    
    try:
        # 1. Initialize Client with LOCAL server (no separate process needed)
        client = MCPClient(server_instance=mcp)
        await client.connect()  # Open persistent MCP session
        
        # 2. Initialize Agent
        agent_instance = ClassQuizAgent(mcp_client=client)
        await agent_instance.initialize()
        
        logger.info("✓ ClassQuiz API Ready: Agent connected to Local MCP Server")
        
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    
    yield

    # Cleanup: close MCP session
    try:
        await client.disconnect()
    except Exception:
        pass

app = FastAPI(
    title="ClassQuiz Agent API",
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
            # Pre-load memory for this session so agent knows who it is
            agent_instance.memory.update_context(request.session_id, {"student_id": request.student_id})

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

if __name__ == "__main__":
    import uvicorn
    # Runs on port 5000 to match your app.js
    uvicorn.run(app, host="0.0.0.0", port=5000)