"""
Clothing Changer API
====================

FastAPI endpoints for clothing change operations.

Refactored to use specialized routers and proper lifecycle management.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

from .routers import (
    clothing_router,
    tensor_router,
    model_router,
    health_router,
    image_router,
)
from .dependencies import set_service_instance, get_service_instance
from .middleware import ErrorHandlerMiddleware, RequestLoggerMiddleware
from ..core.clothing_changer_service import ClothingChangerService
from ..config.clothing_changer_config import ClothingChangerConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Clothing Changer API...")
    
    try:
        # Initialize configuration
        config = ClothingChangerConfig.from_env()
        
        # Create service
        service = ClothingChangerService(config=config)
        set_service(service)
        
        # Initialize model (with automatic DeepSeek fallback)
        logger.info("Attempting to initialize model on startup...")
        try:
            service.initialize_model()
            model_info = service.get_model_info()
            if model_info.get("fallback_mode"):
                logger.info("✅ DeepSeek model initialized as fallback (Flux2 unavailable)")
            else:
                logger.info("✅ Flux2 model initialized successfully")
        except Exception as e:
            logger.warning(f"Model initialization deferred: {e}")
            # Model will be initialized on first use
        
        logger.info("Startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Clothing Changer API...")
    try:
        from .dependencies import _service_instance
        if _service_instance:
            _service_instance.close()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    logger.info("Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Character Clothing Changer AI API",
    description="API for changing character clothing and generating ComfyUI-compatible safe tensors",
    version="1.0.0",
    lifespan=lifespan,
)

# Add request logging middleware (first)
app.add_middleware(RequestLoggerMiddleware)

# Add error handling middleware
app.add_middleware(ErrorHandlerMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(clothing_router)
app.include_router(tensor_router)
app.include_router(model_router)
app.include_router(health_router)
app.include_router(image_router)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Static files mounted from: {static_dir}")

# Serve HTML interface
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    """Serve the HTML interface."""
    html_path = Path(__file__).parent.parent / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
    <html>
        <head><title>Character Clothing Changer AI</title></head>
        <body>
            <h1>Character Clothing Changer AI</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
            <p>HTML interface not found. Please ensure index.html exists in the project root.</p>
        </body>
    </html>
    """)

# Root endpoint (API info)
@app.get("/api")
async def root():
    """API info endpoint."""
    return {
        "service": "Character Clothing Changer AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
