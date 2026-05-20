"""
Dog Training Coaching AI - Main Application
===========================================
Servidor principal para el sistema de coaching de adiestramiento de perros.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config.app_config import get_config
from .api.routes import router
from .api.routes.health import router as health_router
from .api.routes.metrics import router as metrics_router
from .api.routes.stats import router as stats_router
from .api.routes.config import router as config_router
from .api.routes.utils import router as utils_router
from .api.routes.helpers import router as helpers_router
from .api.routes.monitoring import router as monitoring_router
from .api.routes.streaming import router as streaming_router
from .api.routes.websocket import router as websocket_router
from .api.routes.batch import router as batch_router
from .api.routes.advanced import router as advanced_router
from .api.routes.events import router as events_router
from .api.routes.observability import router as observability_router
from .api.routes.idempotency import router as idempotency_router
from .api.routes.dev_tools import router as dev_tools_router
from .api.routes.data_analysis import router as data_analysis_router
from .api.routes.backup import router as backup_router
from .middleware import LoggingMiddleware, ErrorMiddleware
from .middleware.request_id_middleware import RequestIDMiddleware
from .utils.logger import setup_logging, get_logger
from .utils.rate_limiter import limiter
from slowapi.errors import RateLimitExceeded, _rate_limit_exceeded_handler

logger = get_logger(__name__)

config = get_config()

# Setup logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events."""
    from .utils.graceful_shutdown import get_graceful_shutdown
    
    shutdown_manager = get_graceful_shutdown()
    shutdown_manager.setup_signal_handlers()
    
    # Startup
    logger.info("Application starting up")
    yield
    
    # Shutdown
    logger.info("Application shutting down")
    await shutdown_manager.shutdown()


from .api.docs import API_DESCRIPTION

app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware (order matters)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(ErrorMiddleware)
app.add_middleware(LoggingMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
routers = [
    router,
    health_router,
    metrics_router,
    stats_router,
    config_router,
    utils_router,
    helpers_router,
    monitoring_router,
    streaming_router,
    websocket_router,
    batch_router,
    advanced_router,
    events_router,
    observability_router,
    idempotency_router,
    dev_tools_router,
    data_analysis_router,
    backup_router
]

for r in routers:
    app.include_router(r)


@app.get("/")
async def root():
    """Root endpoint."""
    # Generate endpoints dynamically from registered routes
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "path") and route.path != "/":
            key = route.path.replace("/api/v1/", "").replace("/", "-") or "root"
            endpoints[key] = route.path
    
    endpoints["docs"] = "/docs"
    
    return {
        "service": config.app_name,
        "version": config.app_version,
        "status": "running",
        "endpoints": endpoints
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)

