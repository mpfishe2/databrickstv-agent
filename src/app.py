"""
DatabricksTV Recommendation Agent -- FastAPI application entry point.

Serves:
  - ``/``           Static chat UI (``static/index.html``)
  - ``/api/chat``   POST endpoint for agent interactions
  - ``/api/health`` GET  health check
"""

from __future__ import annotations

import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import run_agent, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLflow experiment
# ---------------------------------------------------------------------------

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(
    os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/databrickstv-agent")
)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

CONVERSATIONS: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    mlflow.config.enable_async_logging(True)
    logger.info("DatabricksTV agent ready.")
    yield
    mlflow.flush_trace_async_logging()
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DatabricksTV Recommendation Agent",
    description="Conversational AI agent for content recommendations, "
    "brand safety checks, and data exploration on DatabricksTV.",
    version="2.0.0",
    lifespan=lifespan,
)

# Static files
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message text")
    user_id: Optional[str] = Field(
        None,
        description="Optional DatabricksTV user ID for context (e.g. U0042)",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for multi-turn support",
    )


class ChatResponse(BaseModel):
    """Agent response returned to the frontend."""

    conversation_id: str = Field(..., description="Conversation ID for multi-turn")
    response: str = Field(..., description="Agent's text response")
    tool_calls: list = Field(default_factory=list, description="Tool call trace")
    error: Optional[str] = Field(None, description="Error message, if any")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Serve the chat UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "DatabricksTV Agent API is running. No frontend found."}


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Process a user message through the DatabricksTV agent."""
    cid = request.conversation_id or str(uuid.uuid4())

    # Initialize conversation with system prompt if new
    if cid not in CONVERSATIONS:
        CONVERSATIONS[cid] = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = CONVERSATIONS[cid]

    # Enrich the message with user context if available
    enriched_message = request.message
    if request.user_id:
        enriched_message = (
            f"[Current user context: {request.user_id}]\n\n{request.message}"
        )

    messages.append({"role": "user", "content": enriched_message})

    logger.info(
        "Chat request -- cid=%s, user_id=%s, message=%s",
        cid,
        request.user_id or "(none)",
        request.message[:120],
    )

    try:
        response_text, tool_trace = run_agent(messages)
        messages.append({"role": "assistant", "content": response_text})
        return ChatResponse(
            conversation_id=cid,
            response=response_text,
            tool_calls=tool_trace,
        )
    except Exception:
        tb = traceback.format_exc()
        logger.error("Agent error:\n%s", tb)
        return ChatResponse(
            conversation_id=cid,
            response="I encountered an error processing your request. Please try again.",
            tool_calls=[],
            error=str(tb),
        )


# ---------------------------------------------------------------------------
# Run with uvicorn when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.app:app", host="0.0.0.0", port=port, reload=True)
