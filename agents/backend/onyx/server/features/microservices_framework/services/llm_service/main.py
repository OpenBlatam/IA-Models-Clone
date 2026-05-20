"""
LLM Service - Refactored with Modular Architecture
FastAPI service using all modular components.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from .api.endpoints import router
from .config import get_config
from shared.ml import EventBus, EventType, LoggingEventListener

# Configure structured logging
logger = structlog.get_logger()

# Load configuration
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    # Startup
    logger.info("llm_service_starting", config=config)
    
    # Setup event bus
    event_bus = EventBus()
    event_bus.subscribe_all(LoggingEventListener())
    event_bus.publish(EventType.MODEL_LOADED, {"service": "llm_service", "status": "starting"})
    
    yield
    
    # Shutdown
    logger.info("llm_service_shutting_down")
    event_bus.publish(EventType.MODEL_LOADED, {"service": "llm_service", "status": "stopping"})


# Create FastAPI app
app = FastAPI(
    title="LLM Service",
    description="Transformer Models & Language Model Inference Service (Refactored)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
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
    """Root endpoint."""
    return {
        "service": "llm_service",
        "version": "2.0.0",
        "status": "running",
        "architecture": "modular",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.get("host", "0.0.0.0"),
        port=config.get("port", 8001),
        reload=True,
        log_level="info",
    )
