from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import os
import sys
import signal
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvloop
import httptools
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
import prometheus_client
from prometheus_fastapi_instrumentator import Instrumentator
import orjson
from application.use_cases.analyze_url_use_case import AnalyzeURLUseCase
from application.use_cases.batch_analyze_urls_use_case import BatchAnalyzeURLsUseCase
from application.dto.analyze_url_request import AnalyzeURLRequest
from application.dto.batch_analyze_request import BatchAnalyzeRequest
from application.dto.analyze_url_response import AnalyzeURLResponse
from application.dto.batch_analyze_response import BatchAnalyzeResponse
from presentation.dependencies import get_analyze_url_use_case, get_batch_analyze_use_case
from shared.config.settings import settings
from shared.utils.performance import performance_monitor
        import psutil
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            import psutil
        import psutil
        import psutil
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized SEO Service Production Entry Point v8
Maximum performance production server with advanced features
"""


# Ultra-fast imports

# Application imports


# Configure uvloop for maximum performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format=settings.log_format,
    level=settings.log_level,
    colorize=True,
    backtrace=True,
    diagnose=True
)
logger.add(
    "logs/seo_service.log",
    rotation="200 MB",
    retention="7 days",
    compression="zstd",
    format=settings.log_format,
    level=settings.log_level,
    backtrace=True,
    diagnose=True
)

# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ultra-optimized application lifespan management"""
    # Startup
    logger.info("🚀 Starting Ultra-Optimized SEO Service v8")
    logger.info(f"📊 Configuration: {settings.app_name} v{settings.app_version}")
    logger.info(f"🔧 Debug mode: {settings.debug}")
    logger.info(f"🌐 Server: {settings.host}:{settings.port}")
    logger.info(f"👥 Workers: {settings.workers}")
    
    # Initialize metrics
    prometheus_client.REGISTRY.unregister(prometheus_client.GC_COLLECTOR)
    prometheus_client.REGISTRY.unregister(prometheus_client.PLATFORM_COLLECTOR)
    prometheus_client.REGISTRY.unregister(prometheus_client.PROCESS_COLLECTOR)
    
    # Start background tasks
    asyncio.create_task(health_monitor())
    asyncio.create_task(performance_monitor_task())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Ultra-Optimized SEO Service")
    shutdown_event.set()
    
    # Wait for background tasks to complete
    await asyncio.sleep(1)
    logger.info("✅ Shutdown complete")


# Create ultra-optimized FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Ultra-Optimized SEO Analysis Service with Maximum Performance",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Ultra-optimized middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom middleware for ultra-fast performance monitoring
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Ultra-fast performance monitoring middleware"""
    start_time = time.perf_counter()
    
    # Add request ID for tracing
    request_id = f"req_{int(start_time * 1000000)}"
    request.state.request_id = request_id
    
    # Add custom headers
    response = await call_next(request)
    
    # Calculate performance metrics
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    # Log performance
    logger.info(f"Request {request_id}: {request.method} {request.url.path} - {process_time:.4f}s")
    
    return response


# Ultra-optimized error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ultra-optimized global exception handler"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Request {request_id} failed: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "timestamp": time.time()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ultra-optimized HTTP exception handler"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(f"Request {request_id} HTTP error: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "request_id": request_id,
            "timestamp": time.time()
        }
    )


# Ultra-optimized API routes
@app.post("/api/v1/seo/analyze", response_model=AnalyzeURLResponse)
@performance_monitor
async def analyze_url(
    request: AnalyzeURLRequest,
    use_case: AnalyzeURLUseCase = Depends(get_analyze_url_use_case)
) -> AnalyzeURLResponse:
    """Analyze single URL with ultra-fast performance"""
    try:
        logger.info(f"Analyzing URL: {request.url}")
        result = await use_case.execute(request)
        logger.info(f"Analysis complete for {request.url} - Score: {result.overall_score}")
        return result
    except Exception as e:
        logger.error(f"URL analysis failed for {request.url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/seo/analyze/batch", response_model=BatchAnalyzeResponse)
@performance_monitor
async def batch_analyze_urls(
    request: BatchAnalyzeRequest,
    use_case: BatchAnalyzeURLsUseCase = Depends(get_batch_analyze_use_case)
) -> BatchAnalyzeResponse:
    """Analyze multiple URLs with maximum concurrency"""
    try:
        logger.info(f"Batch analyzing {len(request.urls)} URLs")
        result = await use_case.execute(request)
        logger.info(f"Batch analysis complete - Success rate: {result.success_rate:.2%}")
        return result
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Ultra-fast health check with comprehensive metrics"""
    try:
        uptime = time.time() - startup_time
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        health_data = {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.app_version,
            "uptime_seconds": uptime,
            "uptime_formatted": format_uptime(uptime),
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        }
        
        # Check if system is healthy
        if cpu_percent > 90 or memory.percent > 90:
            health_data["status"] = "degraded"
            health_data["warnings"] = ["High resource usage detected"]
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/api/v1/seo/stats")
async def get_stats():
    """Get comprehensive system statistics"""
    try:
        # Get stats from use cases
        analyze_use_case = get_analyze_url_use_case()
        
        stats = {
            "service": {
                "name": settings.app_name,
                "version": settings.app_version,
                "uptime_seconds": time.time() - startup_time,
                "requests_processed": 0,  # Would be tracked in production
                "cache_hit_rate": 0.0,   # Would be tracked in production
                "average_response_time": 0.0  # Would be tracked in production
            },
            "performance": {
                "memory_usage_mb": get_memory_usage(),
                "cpu_usage_percent": get_cpu_usage(),
                "active_connections": 0,  # Would be tracked in production
                "queue_size": 0  # Would be tracked in production
            },
            "cache": {
                "size": 0,  # Would be tracked in production
                "hit_rate": 0.0,
                "evictions": 0,
                "compression_ratio": 1.0
            },
            "parser": {
                "total_parses": 0,  # Would be tracked in production
                "average_parse_time_ms": 0.0,
                "preferred_parser": "selectolax",
                "compression_enabled": True
            },
            "http_client": {
                "active_connections": 0,  # Would be tracked in production
                "total_requests": 0,
                "average_response_time_ms": 0.0,
                "error_rate": 0.0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Ultra-fast root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "endpoints": {
            "analyze": "/api/v1/seo/analyze",
            "batch_analyze": "/api/v1/seo/analyze/batch",
            "health": "/health",
            "metrics": "/metrics",
            "stats": "/api/v1/seo/stats",
            "docs": "/docs" if settings.debug else "disabled"
        },
        "timestamp": time.time()
    }


# Background tasks
async def health_monitor():
    """Background health monitoring task"""
    while not shutdown_event.is_set():
        try:
            # Monitor system health
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 80 or memory.percent > 80:
                logger.warning(f"High resource usage - CPU: {cpu_percent}%, Memory: {memory.percent}%")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(60)


async def performance_monitor_task():
    """Background performance monitoring task"""
    while not shutdown_event.is_set():
        try:
            # Monitor performance metrics
            memory_usage = get_memory_usage()
            cpu_usage = get_cpu_usage()
            
            logger.info(f"Performance - Memory: {memory_usage:.1f}MB, CPU: {cpu_usage:.1f}%")
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Performance monitor error: {e}")
            await asyncio.sleep(120)


# Utility functions
def format_uptime(seconds: float) -> str:
    """Format uptime in human readable format"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def get_cpu_usage() -> float:
    """Get current CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0.0


# Signal handlers for graceful shutdown
def signal_handler(signum, frame) -> Any:
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Production server configuration
if __name__ == "__main__":
    # Ultra-optimized uvicorn configuration
    uvicorn.run(
        "main_production_v8:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        loop="uvloop",
        http="httptools",
        access_log=False,  # We handle logging ourselves
        log_level="error",  # Minimal uvicorn logging
        server_header=False,  # Security
        date_header=False,  # Performance
        forwarded_allow_ips="*",  # Trust all proxies in production
        proxy_headers=True,
        forwarded_headers=True,
        # Ultra-fast performance settings
        limit_concurrency=1000,
        limit_max_requests=10000,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30,
        # Security settings
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE"),
        ssl_ca_certs=os.getenv("SSL_CA_CERTS"),
        # Development settings
        reload=settings.debug,
        reload_dirs=["."] if settings.debug else None,
    ) 