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
import logging
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from .config import get_config, CopywritingConfig
from .models import (
from .service import get_copywriting_service, cleanup_service
from .monitoring import get_metrics_collector
from typing import Any, List, Dict, Optional
"""
FastAPI Application
==================

Production-ready FastAPI application for the copywriting service with
comprehensive endpoints, middleware, and monitoring.
"""



    CopywritingRequest, CopywritingResponse, BatchCopywritingRequest, BatchCopywritingResponse,
    HealthCheckResponse, MetricsResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


class RateLimitMiddleware:
    """Simple rate limiting middleware"""
    
    def __init__(self, requests_per_minute: int = 100):
        
    """__init__ function."""
self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
    
    async def __call__(self, request: Request, call_next):
        
    """__call__ function."""
client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": 60}
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Copywriting Service...")
    
    # Initialize service
    service = await get_copywriting_service()
    
    # Start monitoring
    metrics_collector = get_metrics_collector()
    metrics_collector.start_monitoring()
    
    logger.info("✅ Copywriting Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Copywriting Service...")
    
    # Cleanup service
    await cleanup_service()
    
    # Stop monitoring
    metrics_collector.stop_monitoring()
    
    logger.info("✅ Copywriting Service shutdown complete")


def create_app(config: Optional[CopywritingConfig] = None) -> FastAPI:
    """Create and configure FastAPI application"""
    
    if config is None:
        config = get_config()
    
    app = FastAPI(
        title="Copywriting Service",
        description="High-performance copywriting service with AI integration",
        version="2.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Rate limiting
    rate_limit_middleware = RateLimitMiddleware(
        requests_per_minute=config.security.rate_limit_per_minute
    )
    
    @app.middleware("http")
    async def add_rate_limiting(request: Request, call_next):
        
    """add_rate_limiting function."""
return await rate_limit_middleware(request, call_next)
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        
    """add_process_time_header function."""
start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Record metrics
        metrics_collector = get_metrics_collector()
        metrics_collector.record_request(
            method=request.method,
            endpoint=str(request.url.path),
            status_code=response.status_code,
            duration=process_time
        )
        
        return response
    
    # Authentication dependency
    async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        """Verify API key if configured"""
        if not config.security.valid_api_keys:
            return True  # No authentication required
        
        if not credentials:
            raise HTTPException(status_code=401, detail="API key required")
        
        if credentials.credentials not in config.security.valid_api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return True
    
    # Routes
    @app.get("/", response_class=JSONResponse)
    async def root():
        """Root endpoint"""
        return {
            "service": "Copywriting Service",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs" if config.debug else "disabled"
        }
    
    @app.post("/generate", response_model=CopywritingResponse)
    async def generate_copy(
        request: CopywritingRequest,
        background_tasks: BackgroundTasks,
        authenticated: bool = Depends(verify_api_key)
    ):
        """Generate copywriting content"""
        try:
            service = await get_copywriting_service()
            response = await service.generate_copy(request)
            
            # Background task for analytics
            background_tasks.add_task(
                log_generation_analytics,
                request.use_case.value,
                request.language.value,
                request.tone.value,
                response.metrics.generation_time
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/generate/batch", response_model=BatchCopywritingResponse)
    async def generate_batch_copy(
        request: BatchCopywritingRequest,
        authenticated: bool = Depends(verify_api_key)
    ):
        """Generate multiple copywriting pieces"""
        try:
            service = await get_copywriting_service()
            response = await service.generate_batch(request)
            return response
            
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health", response_model=HealthCheckResponse)
    async def health_check():
        """Comprehensive health check"""
        try:
            service = await get_copywriting_service()
            health_data = await service.health_check()
            
            return HealthCheckResponse(
                status=health_data["status"],
                version=health_data["version"],
                uptime=health_data["uptime"],
                database_status="healthy",  # Placeholder
                redis_status=health_data["cache_status"]["redis_cache"]["status"],
                ai_providers_status=health_data["ai_providers_status"],
                memory_usage={
                    "rss_mb": health_data["performance"]["system_info"].get("memory_total", 0) / 1024 / 1024,
                    "available_mb": health_data["performance"]["system_info"].get("memory_available", 0) / 1024 / 1024
                },
                cache_stats=health_data["cache_stats"]
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return HealthCheckResponse(
                status="unhealthy",
                version="2.0.0",
                uptime=0,
                database_status="unknown",
                redis_status="unknown",
                ai_providers_status={},
                memory_usage={},
                cache_stats={}
            )
    
    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics(authenticated: bool = Depends(verify_api_key)):
        """Get service metrics"""
        try:
            service = await get_copywriting_service()
            metrics_data = await service.get_metrics()
            
            service_metrics = metrics_data["service_metrics"]
            
            return MetricsResponse(
                total_requests=int(service_metrics.get("counters", {}).get("requests_total", 0)),
                successful_requests=int(service_metrics.get("counters", {}).get("requests_success", 0)),
                failed_requests=int(service_metrics.get("counters", {}).get("requests_error", 0)),
                average_response_time=service_metrics.get("average_response_time", 0),
                cache_hit_rate=service_metrics.get("cache_hit_rate", 0),
                optimization_score=metrics_data["optimization_metrics"]["optimization_profile"]["total_score"],
                ai_provider_usage=service_metrics.get("counters", {}),
                model_usage={},  # Placeholder
                language_distribution={},  # Placeholder
                tone_distribution={},  # Placeholder
                use_case_distribution={}  # Placeholder
            )
            
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    async def get_prometheus_metrics():
        """Get metrics in Prometheus format"""
        try:
            metrics_collector = get_metrics_collector()
            return metrics_collector.get_prometheus_metrics()
            
        except Exception as e:
            logger.error(f"Prometheus metrics error: {e}")
            return f"# Error generating metrics: {e}"
    
    @app.get("/config")
    async def get_config_info(authenticated: bool = Depends(verify_api_key)):
        """Get service configuration (non-sensitive)"""
        return {
            "supported_languages": config.supported_languages,
            "supported_tones": config.supported_tones,
            "supported_use_cases": config.supported_use_cases,
            "ai_providers": list(config.get_ai_provider_config(provider).keys() 
                               for provider in ["openrouter", "openai", "anthropic", "google"]
                               if config.get_ai_provider_config(provider)),
            "optimization_info": {
                "performance_multiplier": get_metrics_collector().optimization_manager.profile.performance_multiplier,
                "available_libraries": len([lib for lib in get_metrics_collector().optimization_manager.profile.libraries.values() if lib.available])
            }
        }
    
    @app.post("/cache/clear")
    async def clear_cache(authenticated: bool = Depends(verify_api_key)):
        """Clear service cache"""
        try:
            service = await get_copywriting_service()
            await service.cache_manager.clear_all()
            return {"message": "Cache cleared successfully"}
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/optimization/report")
    async def get_optimization_report(authenticated: bool = Depends(verify_api_key)):
        """Get detailed optimization report"""
        try:
            service = await get_copywriting_service()
            report = service.optimization_manager.get_optimization_report()
            return report
            
        except Exception as e:
            logger.error(f"Optimization report error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Error handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "path": str(request.url.path)
            }
        )
    
    return app


async def log_generation_analytics(use_case: str, language: str, tone: str, duration: float):
    """Background task for logging analytics"""
    try:
        metrics_collector = get_metrics_collector()
        metrics_collector.record_counter(f"generation_by_use_case_{use_case}")
        metrics_collector.record_counter(f"generation_by_language_{language}")
        metrics_collector.record_counter(f"generation_by_tone_{tone}")
        metrics_collector.record_histogram("generation_duration_by_use_case", duration, {"use_case": use_case})
    except Exception as e:
        logger.warning(f"Analytics logging error: {e}")


def run_production():
    """Run the application in production mode"""
    config = get_config()
    
    # Create app
    app = create_app(config)
    
    # Production server configuration
    uvicorn_config = {
        "app": app,
        "host": config.host,
        "port": config.port,
        "workers": 1,  # Single worker for now due to shared state
        "loop": "uvloop" if config.optimization.enable_jit else "asyncio",
        "http": "httptools",
        "access_log": config.debug,
        "log_level": config.monitoring.log_level.lower(),
    }
    
    # Add SSL if configured
    if config.is_production():
        # In production, SSL should be handled by reverse proxy
        pass
    
    logger.info(f"🚀 Starting production server on {config.host}:{config.port}")
    uvicorn.run(**uvicorn_config)


match __name__:
    case "__main__":
    run_production() 