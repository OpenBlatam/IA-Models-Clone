from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.registry import CollectorRegistry
import psutil
from onyx.server.features.key_messages.models import (
from onyx.server.features.key_messages.service import (
from onyx.server.features.key_messages.config import get_settings
from typing import Any, List, Dict, Optional
import logging
"""
Optimized API for Key Messages feature with modern FastAPI practices and functional programming.
"""


    KeyMessageRequest,
    KeyMessageResponse,
    BatchKeyMessageRequest,
    BatchKeyMessageResponse,
    MessageType,
    MessageTone
)
    ServiceConfig, startup_service, shutdown_service, check_service_health,
    generate_response, analyze_message, generate_batch, clear_cache, get_cache_stats,
    validate_service_health, validate_message_request, validate_batch_request
)

# Configure structured logging
logger = structlog.get_logger(__name__)

# Prometheus metrics
API_REQUEST_COUNTER = Counter('key_messages_api_requests_total', 'API requests', ['endpoint', 'method', 'status'])
API_REQUEST_DURATION = Histogram('key_messages_api_duration_seconds', 'API request duration', ['endpoint'])
API_ERROR_COUNTER = Counter('key_messages_api_errors_total', 'API errors', ['endpoint', 'error_type'])

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global service configuration
service_config: Optional[ServiceConfig] = None

# Endpoint configuration for modular handling
ENDPOINT_CONFIGS = {
    "generate": {
        "path": "/generate",
        "method": "POST",
        "rate_limit": "50/minute",
        "response_model": KeyMessageResponse,
        "validators": [validate_message_request, validate_service_health],
        "service_function": generate_response,
        "background_task": lambda req: log_analytics("generate", req),
        "error_type": "generation"
    },
    "analyze": {
        "path": "/analyze", 
        "method": "POST",
        "rate_limit": "30/minute",
        "response_model": KeyMessageResponse,
        "validators": [validate_message_request, validate_service_health],
        "service_function": analyze_message,
        "background_task": lambda req: log_analytics("analyze", req),
        "error_type": "analysis"
    },
    "batch": {
        "path": "/batch",
        "method": "POST", 
        "rate_limit": "10/minute",
        "response_model": BatchKeyMessageResponse,
        "validators": [validate_batch_request, validate_service_health],
        "service_function": generate_batch,
        "background_task": lambda req: log_batch_analytics(req),
        "error_type": "batch_generation"
    },
    "clear_cache": {
        "path": "/cache",
        "method": "DELETE",
        "rate_limit": "5/minute",
        "response_model": None,
        "validators": [validate_service_health],
        "service_function": clear_cache,
        "background_task": None,
        "error_type": "cache_clear",
        "custom_response": lambda: {"success": True, "message": "Cache cleared successfully"}
    },
    "cache_stats": {
        "path": "/cache/stats",
        "method": "GET",
        "rate_limit": "20/minute", 
        "response_model": None,
        "validators": [validate_service_health],
        "service_function": get_cache_stats,
        "background_task": None,
        "error_type": "cache_stats",
        "custom_response": lambda data: {"success": True, "data": data}
    }
}

# Modular endpoint handler factory
def create_endpoint_handler(endpoint_name: str, config: Dict[str, Any]):
    """Create modular endpoint handler."""
    
    # Guard clause: Check if endpoint_name is None or empty
    if not endpoint_name or not endpoint_name.strip():
        raise ValueError("Endpoint name cannot be empty")
    
    # Guard clause: Check if config is None
    if config is None:
        raise ValueError("Endpoint config cannot be None")
    
    # Guard clause: Check if required config keys exist
    required_keys = ["path", "method", "service_function"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    async def endpoint_handler(
        request: Request,
        key_message_request: Optional[Union[KeyMessageRequest, BatchKeyMessageRequest]] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ):
        """Generic endpoint handler with common patterns."""
        start_time = time.perf_counter()
        
        # Guard clause: Check if request is None
        if request is None:
            raise HTTPException(status_code=400, detail="Request cannot be None")
        
        # Run validators
        for validator in config.get("validators", []):
            # Guard clause: Check if validator is callable
            if not callable(validator):
                logger.warning(f"Invalid validator in {endpoint_name}: {validator}")
                continue
                
            try:
                if key_message_request is not None:
                    validator(key_message_request)
                else:
                    validator()
            except Exception as e:
                logger.error(f"Validation error in {endpoint_name}", error=str(e))
                raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
        
        try:
            # Add background task if specified
            if background_tasks and config.get("background_task") and key_message_request:
                # Guard clause: Check if background_task is callable
                if callable(config["background_task"]):
                    background_tasks.add_task(config["background_task"], key_message_request)
            
            # Call service function
            service_function = config["service_function"]
            # Guard clause: Check if service_function is callable
            if not callable(service_function):
                raise HTTPException(status_code=500, detail="Service function not available")
            
            if key_message_request is not None:
                response = await service_function(key_message_request)
            else:
                response = await service_function()
            
            processing_time = time.perf_counter() - start_time
            
            # Update metrics
            API_REQUEST_COUNTER.labels(
                endpoint=config["path"], 
                method=config["method"], 
                status="success"
            ).inc()
            API_REQUEST_DURATION.labels(endpoint=config["path"]).observe(processing_time)
            
            # Return response
            if config.get("custom_response"):
                custom_response_func = config["custom_response"]
                # Guard clause: Check if custom_response_func is callable
                if callable(custom_response_func):
                    return custom_response_func(response if response else None)
                else:
                    return {"success": True, "data": response}
            else:
                return response
                
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            
            # Update error metrics
            API_REQUEST_COUNTER.labels(
                endpoint=config["path"], 
                method=config["method"], 
                status="error"
            ).inc()
            API_ERROR_COUNTER.labels(
                endpoint=config["path"], 
                error_type=config.get("error_type", "unknown")
            ).inc()
            
            logger.error(f"Error in {endpoint_name}", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    return endpoint_handler

# Modular system info handlers
SYSTEM_INFO_HANDLERS = {
    "cpu_usage_percent": lambda: psutil.cpu_percent(interval=1),
    "memory_usage_percent": lambda: psutil.virtual_memory().percent,
    "disk_usage_percent": lambda: psutil.disk_usage('/').percent,
    "network_connections": lambda: len(psutil.net_connections()),
    "process_count": lambda: len(psutil.pids())
}

async def get_system_info_modular() -> Dict[str, Any]:
    """Get system information using modular handlers."""
    # Guard clause: Check if SYSTEM_INFO_HANDLERS is empty
    if not SYSTEM_INFO_HANDLERS:
        return {"error": "No system info handlers available"}
    
    system_info = {}
    for key, handler in SYSTEM_INFO_HANDLERS.items():
        try:
            # Guard clause: Check if handler is callable
            if callable(handler):
                system_info[key] = handler()
            else:
                system_info[key] = None
        except Exception as e:
            logger.error(f"Error getting system info for {key}", error=str(e))
            system_info[key] = None
    
    return system_info

# Modular enum data handlers
ENUM_DATA_HANDLERS = {
    "types": MessageType,
    "tones": MessageTone
}

def create_enum_data_handler(enum_class) -> Any:
    """Create handler for enum data."""
    # Guard clause: Check if enum_class is None
    if enum_class is None:
        return []
    
    # Guard clause: Check if enum_class has __iter__ method
    if not hasattr(enum_class, '__iter__'):
        return []
    
    try:
        return [
            {"value": item.value, "label": item.value.replace("_", " ").title()}
            for item in enum_class
        ]
    except Exception as e:
        logger.error(f"Error creating enum data handler", error=str(e))
        return []

# Create FastAPI app
app = FastAPI(
    title="Key Messages API",
    description="AI-powered key message generation and analysis",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Guard clause: Check if app is None
    if app is None:
        return
    
    try:
        # Startup
        global service_config
        settings = get_settings()
        
        # Guard clause: Check if settings is None
        if settings is None:
            logger.error("Failed to load settings")
            return
        
        service_config = ServiceConfig(
            model_name=settings.model_name,
            max_concurrent_requests=settings.max_concurrent_requests,
            cache_size=settings.cache_size,
            timeout_seconds=settings.timeout_seconds,
            enable_analytics=settings.enable_analytics,
            enable_caching=settings.enable_caching,
            enable_monitoring=settings.enable_monitoring
        )
        
        await startup_service(service_config)
        logger.info("Application started successfully")
        
        yield
        
        # Shutdown
        await shutdown_service()
        logger.info("Application shutdown successfully")
        
    except Exception as e:
        logger.error("Application lifespan error", error=str(e))
        raise

app.router.lifespan_context = lifespan

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    # Guard clause: Check if request is None
    if request is None:
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed", "errors": []}
        )
    
    # Guard clause: Check if exc is None
    if exc is None:
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed", "errors": []}
        )
    
    logger.warning("Request validation error", errors=exc.errors())
    
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    # Guard clause: Check if request is None
    if request is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Guard clause: Check if exc is None
    if exc is None:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.get("/health")
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    # Guard clause: Check if request is None
    if request is None:
        raise HTTPException(status_code=400, detail="Request cannot be None")
    
    try:
        # Guard clause: Check if service is initialized
        if service_config is None:
            return {
                "status": "unhealthy",
                "message": "Service not initialized",
                "timestamp": time.time()
            }
        
        health_data = await check_service_health()
        
        # Guard clause: Check if health check failed
        if health_data is None:
            return {
                "status": "unhealthy",
                "message": "Health check failed",
                "timestamp": time.time()
            }
        
        return {
            "status": "healthy" if health_data.get("is_healthy", False) else "unhealthy",
            "data": health_data,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Health check error", error=str(e))
        return {
            "status": "unhealthy",
            "message": str(e),
            "timestamp": time.time()
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        return generate_latest()
    except Exception as e:
        logger.error("Metrics generation error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate metrics")

@app.get("/system/info")
@limiter.limit("10/minute")
async def get_system_info(request: Request):
    """Get system information."""
    # Guard clause: Check if request is None
    if request is None:
        raise HTTPException(status_code=400, detail="Request cannot be None")
    
    try:
        system_info = await get_system_info_modular()
        
        # Guard clause: Check if system_info is None
        if system_info is None:
            return {"error": "Failed to get system information"}
        
        return {
            "success": True,
            "data": system_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("System info error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Create enum endpoints dynamically
for enum_name, enum_class in ENUM_DATA_HANDLERS.items():
    # Guard clause: Check if enum_name is None or empty
    if not enum_name or not enum_name.strip():
        continue
    
    # Guard clause: Check if enum_class is None
    if enum_class is None:
        continue
    
    async def create_enum_endpoint(enum_class=enum_class, enum_name=enum_name) -> Any:
        """Create endpoint for enum data."""
        try:
            enum_data = create_enum_data_handler(enum_class)
            
            # Guard clause: Check if enum_data is None
            if enum_data is None:
                return {"error": f"Failed to get {enum_name} data"}
            
            return {
                "success": True,
                "data": enum_data,
                "count": len(enum_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting {enum_name} data", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # Register the endpoint
    app.get(f"/enums/{enum_name}")(create_enum_endpoint)

async def log_analytics(operation: str, request: KeyMessageRequest):
    """Log analytics for operations."""
    # Guard clause: Check if operation is None or empty
    if not operation or not operation.strip():
        return
    
    # Guard clause: Check if request is None
    if request is None:
        return
    
    try:
        logger.info("Analytics logged", 
                   operation=operation,
                   message_type=request.message_type,
                   tone=request.tone,
                   message_length=len(request.message))
    except Exception as e:
        logger.error("Analytics logging error", error=str(e))

async def log_batch_analytics(batch_request: BatchKeyMessageRequest):
    """Log analytics for batch operations."""
    # Guard clause: Check if batch_request is None
    if batch_request is None:
        return
    
    # Guard clause: Check if messages list is empty
    if not batch_request.messages:
        return
    
    try:
        logger.info("Batch analytics logged",
                   message_count=len(batch_request.messages),
                   message_types=[msg.message_type for msg in batch_request.messages],
                   tones=[msg.tone for msg in batch_request.messages])
    except Exception as e:
        logger.error("Batch analytics logging error", error=str(e))

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Key Messages API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "generate": "/generate",
            "analyze": "/analyze",
            "batch": "/batch",
            "cache": "/cache",
            "system": "/system/info"
        }
    } 