"""
Diffusion Service - Refactored with Modular Architecture
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from .api.endpoints import router
from .config import get_config
from shared.ml import EventBus, EventType, LoggingEventListener

logger = structlog.get_logger()
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager."""
    logger.info("diffusion_service_starting", config=config)
    
    event_bus = EventBus()
    event_bus.subscribe_all(LoggingEventListener())
    event_bus.publish(EventType.MODEL_LOADED, {"service": "diffusion_service", "status": "starting"})
    
    yield
    
    logger.info("diffusion_service_shutting_down")
    event_bus.publish(EventType.MODEL_LOADED, {"service": "diffusion_service", "status": "stopping"})


app = FastAPI(
    title="Diffusion Service",
    description="Stable Diffusion & Image Generation Service (Refactored)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "diffusion_service",
        "version": "2.0.0",
        "status": "running",
        "architecture": "modular",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.get("host", "0.0.0.0"),
        port=config.get("port", 8002),
        reload=True,
        log_level="info",
    )
