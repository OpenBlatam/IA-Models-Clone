"""
Autonomous Long-Term Agent - Main Application
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.v1.routes import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info("🚀 Starting Autonomous Long-Term Agent API")
    logger.info(f"OpenRouter Model: {settings.openrouter_model}")
    logger.info(f"Learning Enabled: {settings.learning_enabled}")
    logger.info(f"Max Parallel Instances: {settings.agent_max_parallel_instances}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Autonomous Long-Term Agent API")
    from .infrastructure.openrouter.client import get_openrouter_client
    client = get_openrouter_client()
    await client.close()


app = FastAPI(
    title="Autonomous Long-Term Agent API",
    version="1.0.0",
    description="""
    Autonomous Long-Term Agent API
    
    Implements concepts from research papers on:
    - Long-horizon agents (WebResearcher)
    - Continual learning (SOLA)
    - Self-initiated learning and adaptation
    - Always-on autonomous operation
    
    Features:
    - Continuous operation until explicit stop
    - Parallel agent execution
    - Persistent knowledge base
    - Self-initiated learning
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Autonomous Long-Term Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "start_agent": "/api/v1/agents/start",
            "stop_agent": "/api/v1/agents/{agent_id}/stop",
            "list_agents": "/api/v1/agents",
            "agent_status": "/api/v1/agents/{agent_id}/status",
            "add_task": "/api/v1/agents/{agent_id}/tasks",
            "parallel_agents": "/api/v1/agents/parallel"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "autonomous_long_term_agent"
    }


if __name__ == "__main__":
    uvicorn.run(
        "autonomous_long_term_agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )




