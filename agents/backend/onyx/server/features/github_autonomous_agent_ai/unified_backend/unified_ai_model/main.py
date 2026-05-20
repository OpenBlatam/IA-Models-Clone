"""
Unified AI Model - Main Application
FastAPI application for AI chat and generation

Usage:
    python -m unified_ai_model.main
    
    Or with uvicorn:
    uvicorn unified_ai_model.main:app --host 0.0.0.0 --port 8050
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_config, UnifiedAIConfig
from .api.routes import router
from .core.llm_service import close_llm_service
from .core.chat_service import close_chat_service
from .core.llm_client import close_openrouter_client

# Version defined here to avoid circular import
__version__ = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    import asyncio
    from .core.continuous_agent import create_agent, stop_all_agents, get_agent
    
    # Startup
    config = get_config()
    logger.info(f"Starting Unified AI Model v{__version__}")
    logger.info(f"Default model: {config.default_model}")
    logger.info(f"API Key configured: {bool(config.openrouter.api_key)}")
    
    # Auto-start 24/7 Continuous Agent
    default_agent = None
    agent_task = None
    
    if config.openrouter.api_key or config.deepseek.api_key:
        logger.info("🤖 Starting 24/7 Continuous Agent...")
        default_agent = create_agent(
            name="DefaultAgent24x7",
            system_prompt="""You are an autonomous AI agent running 24/7. 
You process tasks submitted through the API and work on them until completion.
You maintain context, learn from interactions, and improve over time.
You are always available and ready to help.""",
            loop_interval=2.0,
            idle_interval=10.0,
            enable_parallel=True,
            max_concurrent_tasks=5,
            batch_size=10,
            num_workers=3
        )
        
        # Start agent in background task
        agent_task = asyncio.create_task(default_agent.start())
        logger.info(f"✅ Agent '{default_agent.name}' started (ID: {default_agent.agent_id})")
        logger.info("   Agent is now running 24/7 and ready to receive tasks")
    else:
        logger.warning("⚠️ No API key configured - 24/7 Agent NOT started")
        logger.warning("   Set OPENROUTER_API_KEY or DEEPSEEK_API_KEY to enable")
    
    # Store agent reference in app state for access from routes
    app.state.default_agent = default_agent
    app.state.agent_task = agent_task
    
    yield
    
    # Shutdown
    logger.info("Shutting down Unified AI Model...")
    
    # Stop all agents
    stop_all_agents()
    if agent_task:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
    
    await close_llm_service()
    await close_chat_service()
    await close_openrouter_client()
    logger.info("Shutdown complete")



def create_app(config: Optional[UnifiedAIConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Configured FastAPI application
    """
    if config:
        from .config import set_config
        set_config(config)
    
    app_config = get_config()
    
    app = FastAPI(
        title="Unified AI Model",
        description="""
## Unified AI Model API

A comprehensive API for AI chat and text generation powered by OpenRouter.

### Features
- **Chat**: Conversational AI with memory and context
- **Generate**: Single-turn text generation
- **Streaming**: Real-time streaming responses
- **Parallel**: Generate from multiple models simultaneously
- **Code Analysis**: Analyze code for bugs, performance, security

### Models
Supports all models available through OpenRouter including:
- DeepSeek (deepseek/deepseek-chat)
- OpenAI GPT-4 (openai/gpt-4o)
- Anthropic Claude (anthropic/claude-3.5-sonnet)
- Google Gemini (google/gemini-pro-1.5)
- And many more...

### Authentication
Configure your OpenRouter API key in the environment:
```
OPENROUTER_API_KEY=sk-or-v1-...
```
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # CORS middleware - configured for frontend access
    import os
    cors_origins_str = os.getenv("UNIFIED_AI_CORS_ORIGINS", "*")
    if cors_origins_str == "*":
        cors_origins = ["*"]
    else:
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info(f"CORS origins: {cors_origins}")
    
    # Include routes
    app.include_router(router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Unified AI Model",
            "version": __version__,
            "status": "running",
            "docs": "/docs",
            "api": "/api/v1"
        }
    
    # Error handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app


# Create default app instance
app = create_app()


def get_app() -> FastAPI:
    """Get the FastAPI application instance."""
    return app


def run():
    """Run the application with uvicorn."""
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "unified_ai_model.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug"
    )


if __name__ == "__main__":
    run()



