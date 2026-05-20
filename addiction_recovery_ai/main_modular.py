"""
Main Application - Modular Version
Uses module system for ultra-modular architecture
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from core.module_loader import get_loader
from core.module_registry import get_registry
from config.app_config import get_config
from utils.logging_config import setup_logging

# Get configuration
config = get_config()

# Setup logging
setup_logging(
    level=config.log_level,
    format_string=config.log_format
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Addiction Recovery AI",
    description="Sistema de IA para ayudar a dejar adicciones",
    version="3.3.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and initialize modules
logger.info("Loading modules...")
loader = get_loader()

# Load modules (from environment or default)
try:
    import os
    if os.getenv("ENABLED_MODULES"):
        loader.load_from_environment()
    else:
        loader.load_default_modules()
except Exception as e:
    logger.warning(f"Failed to load modules from environment: {str(e)}")
    loader.load_default_modules()

# Initialize all modules
logger.info("Initializing modules...")
loader.initialize_all()

# Include API routers from modules
registry = get_registry()
for module_name in ["recovery_api", "api"]:
    module = registry.get_module(module_name)
    if module and hasattr(module, "get_router"):
        router = module.get_router()
        app.include_router(router)
        logger.info(f"Included router from module: {module_name}")

# Add middleware (if modules are available)
try:
    from middleware.error_handler import ErrorHandlerMiddleware
    from middleware.rate_limit import RateLimitMiddleware
    from middleware.performance import PerformanceMonitoringMiddleware
    from middleware.logging_middleware import LoggingMiddleware
    
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.rate_limit_per_minute,
        requests_per_hour=config.rate_limit_per_hour
    )
    app.add_middleware(PerformanceMonitoringMiddleware)
    app.add_middleware(LoggingMiddleware)
except ImportError as e:
    logger.warning(f"Some middleware not available: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    registry = get_registry()
    modules = registry.list_modules()
    
    return {
        "service": "Addiction Recovery AI",
        "version": "3.3.0",
        "status": "running",
        "modules": modules,
        "architecture": "ultra-modular"
    }

# Health check
@app.get("/health")
async def health():
    """Health check"""
    registry = get_registry()
    modules_status = {}
    
    for module_name in registry.list_modules():
        module = registry.get_module(module_name)
        modules_status[module_name] = {
            "initialized": module.is_initialized() if module else False,
            "version": module.version if module else "unknown"
        }
    
    return {
        "status": "healthy",
        "modules": modules_status
    }

# Shutdown handler
@app.on_event("shutdown")
async def shutdown():
    """Shutdown all modules"""
    logger.info("Shutting down modules...")
    loader = get_loader()
    loader.shutdown_all()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)















