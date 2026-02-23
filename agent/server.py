"""FastAPI server connecting frontend to agent"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
from agent.agent import ClassQuizAgent
from client import client
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
agent_instance = None
client_connected = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage agent and MCP client lifecycle"""
    global agent_instance, client_connected
    
    # Startup
    try:
        # Connect to MCP server
        await client.__aenter__()
        await client.ping()
        client_connected = True
        logger.info("✓ Connected to MCP server")
        
        # Initialize agent
        agent_instance = ClassQuizAgent(mcp_client=client)
        await agent_instance.initialize()
        logger.info("✓ Agent initialized")
        
        # Check LLM
        if agent_instance.llm.health_check():
            logger.info("✓ LLM (Ollama) is available")
        else:
            logger.warning("✗ LLM (Ollama) not available - check if running")
    
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
    
    yield
    
    # Shutdown
    if client_connected:
        await client.__aexit__(None, None, None)
        logger.info("✓ Disconnected from MCP server")

app = FastAPI(
    title="ClassQuiz Agent API",
    description="Educational AI assistant with MCP tool orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    student_id: str = None
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    intent: str
    tools_used: list
    timestamp: str
    success: bool

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat messages from frontend
    
    Flow:
    1. Receive user message
    2. Pass to agent for processing
    3. Agent uses LLM to classify, plan, execute, synthesize
    4. Return natural language response
    """
    
    if not client_connected or not agent_instance:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Check MCP server and Ollama."
        )
    
    try:
        logger.info(f"Received message: {request.message}")
        
        # Build context
        context = {}
        if request.student_id:
            context["student_id"] = request.student_id
        
        # Process via agent (ALL responses are LLM-generated)
        result = await agent_instance.process_query(
            query=request.message,
            context=context,
            session_id=request.session_id,
        )
        
        return ChatResponse(
            response=result["response"],
            intent=result["intent"],
            tools_used=result.get("tools_used", []),
            timestamp=datetime.now().strftime("%I:%M %p"),
            success=result["success"]
        )
    
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")