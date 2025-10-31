"""
Instagram Captions API v10.0 - Refactored Architecture

Complete API solution consolidating v9.0 ultra-advanced capabilities
into a clean, maintainable, and deployable architecture.
"""

import time
import logging
import asyncio
from typing import Any, List, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from .core_v10 import (
    RefactoredConfig, RefactoredCaptionRequest, RefactoredCaptionResponse,
    BatchRefactoredRequest, RefactoredUtils, Metrics, RefactoredAIEngine
)
from .ai_service_v10 import refactored_ai_service
from .config import get_config, validate_config
from .utils import (
    setup_logging, get_logger, SecurityUtils, CacheManager, 
    RateLimiter, PerformanceMonitor, rate_limit_middleware,
    logging_middleware, security_middleware
)

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Get configuration
config = get_config()

# Setup logging
setup_logging(config.LOG_LEVEL, config.LOG_FORMAT)
logger = get_logger(__name__)

# Security
security = HTTPBearer()

# =============================================================================
# MIDDLEWARE & SECURITY
# =============================================================================

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Enhanced API key verification with rate limiting and security checks."""
    api_key = credentials.credentials
    
    # Enhanced security validation
    if not SecurityUtils.verify_api_key(api_key):
        logger.warning(f"Invalid API key attempt from {request.client.host if 'request' in locals() else 'unknown'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Check for API key abuse patterns
    if len(api_key) < 32:
        logger.warning(f"Short API key detected: {len(api_key)} characters")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key too short"
        )
    
    return api_key

# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service temporarily unavailable (circuit breaker open)"
                )
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }

# =============================================================================
# REFACTORED API APPLICATION
# =============================================================================

class RefactoredCaptionsAPI:
    """
    Consolidated API application that combines the best of v9.0 ultra-advanced features
    with the simplicity and maintainability of refactored architecture.
    """
    
    def __init__(self) -> None:
        self.config = config
        self.metrics = Metrics()
        self.ai_engine = RefactoredAIEngine(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize utilities
        self.cache_manager = CacheManager(
            max_size=self.config.CACHE_SIZE, 
            ttl=self.config.CACHE_TTL
        )
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.RATE_LIMIT_PER_MINUTE,
            burst_size=self.config.RATE_LIMIT_BURST
        )
        
        # Initialize circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        
        # Set performance thresholds
        self.performance_monitor.set_threshold("caption_generation", "max", 10.0)  # 10 seconds max
        self.performance_monitor.set_threshold("batch_generation", "max", 30.0)   # 30 seconds max
        
        self.app = self._create_app()
        self._setup_middleware()
        self._setup_routes()
        self._setup_app_state()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with optimized settings."""
        return FastAPI(
            title=self.config.API_NAME,
            description="Refactored ultra-advanced Instagram caption generation with essential AI capabilities",
            version=self.config.API_VERSION,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            debug=self.config.DEBUG
        )
    
    def _setup_middleware(self) -> None:
        """Setup essential middleware stack."""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware
        self.app.middleware("http")(logging_middleware)
        self.app.middleware("http")(security_middleware)
        self.app.middleware("http")(rate_limit_middleware)
    
    def _setup_app_state(self) -> None:
        """Setup application state for middleware access."""
        self.app.state.rate_limiter = self.rate_limiter
        self.app.state.cache_manager = self.cache_manager
        self.app.state.performance_monitor = self.performance_monitor
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/", tags=["Root"])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": self.config.API_NAME,
                "version": self.config.API_VERSION,
                "status": "running",
                "environment": self.config.ENVIRONMENT,
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", tags=["Health"])
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": self.config.API_VERSION,
                "environment": self.config.ENVIRONMENT,
                "ai_engine": "available" if self.ai_engine.pipeline else "fallback"
            }
        
                       @self.app.post("/generate", tags=["Captions"], response_model=RefactoredCaptionResponse)
               async def generate_caption(
                   request: RefactoredCaptionRequest,
                   api_key: str = Depends(verify_api_key)
               ):
                   """Generate a single Instagram caption with enhanced error handling."""
                   start_time = time.time()
                   
                   try:
                       # Sanitize input with enhanced security
                       request.text = SecurityUtils.sanitize_input(request.text, strict=True)
                       
                       # Validate input length
                       if len(request.text) > 1000:
                           raise HTTPException(
                               status_code=status.HTTP_400_BAD_REQUEST,
                               detail="Input text too long (max 1000 characters)"
                           )
                       
                       # Generate caption with circuit breaker protection
                       async def _generate_caption():
                           return await self.ai_engine.generate_caption(request)
                       
                       response = await asyncio.to_thread(
                           self.circuit_breaker.call,
                           _generate_caption
                       )
                       
                       # Record metrics with metadata
                       processing_time = time.time() - start_time
                       self.metrics.record_request(True, processing_time)
                       self.performance_monitor.record_metric(
                           "caption_generation", 
                           processing_time,
                           metadata={
                               "text_length": len(request.text),
                               "style": request.style,
                               "length": request.length,
                               "api_key_length": len(api_key)
                           }
                       )
                       
                       # Add performance headers
                       response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
                       response.headers["X-Circuit-Breaker-State"] = self.circuit_breaker.state
                       
                       return response
                       
                   except HTTPException:
                       # Re-raise HTTP exceptions
                       raise
                   except Exception as e:
                       logger.error(f"Error generating caption: {e}", exc_info=True)
                       processing_time = time.time() - start_time
                       self.metrics.record_request(False, processing_time)
                       
                       # Record error metric
                       self.performance_monitor.record_metric(
                           "caption_generation_errors",
                           processing_time,
                           metadata={"error_type": type(e).__name__, "error_message": str(e)}
                       )
                       
                       # Return user-friendly error
                       raise HTTPException(
                           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                           detail="Failed to generate caption. Please try again later."
                       )
        
        @self.app.post("/generate/batch", tags=["Captions"])
        async def generate_batch_captions(
            batch_request: BatchRefactoredRequest,
            api_key: str = Depends(verify_api_key)
        ):
            """Generate multiple Instagram captions in batch."""
            start_time = time.time()
            
            try:
                results = []
                for req in batch_request.requests:
                    # Sanitize input
                    req.text = SecurityUtils.sanitize_input(req.text)
                    
                    # Generate caption
                    response = await self.ai_engine.generate_caption(req)
                    results.append(response)
                
                # Record metrics
                processing_time = time.time() - start_time
                self.metrics.record_request(True, processing_time)
                self.performance_monitor.record_metric("batch_generation", processing_time)
                
                return {
                    "batch_id": batch_request.batch_id,
                    "total_requests": len(batch_request.requests),
                    "results": results,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                logger.error(f"Error generating batch captions: {e}")
                processing_time = time.time() - start_time
                self.metrics.record_request(False, processing_time)
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate batch captions: {str(e)}"
                )
        
        @self.app.get("/metrics", tags=["Monitoring"])
        async def get_metrics(api_key: str = Depends(verify_api_key)):
            """Get API metrics and statistics."""
            return {
                "api_metrics": self.metrics.get_stats(),
                "performance_metrics": self.performance_monitor.get_all_statistics(),
                "cache_stats": self.cache_manager.get_stats(),
                "rate_limiting_stats": {
                    "requests_per_minute": self.config.RATE_LIMIT_PER_MINUTE,
                    "burst_size": self.config.RATE_LIMIT_BURST
                }
            }
        
        @self.app.get("/config", tags=["Configuration"])
        async def get_config_info(api_key: str = Depends(verify_api_key)):
            """Get current configuration (non-sensitive)."""
            return {
                "api_version": self.config.API_VERSION,
                "environment": self.config.ENVIRONMENT,
                "ai_model": self.config.AI_MODEL_NAME,
                "max_tokens": self.config.MAX_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "cache_size": self.config.CACHE_SIZE,
                "rate_limit_per_minute": self.config.RATE_LIMIT_PER_MINUTE
            }
        
        @self.app.post("/ai-service/test", tags=["AI Service"])
        async def test_ai_service(api_key: str = Depends(verify_api_key)):
            """Test AI service functionality."""
            try:
                result = await refactored_ai_service.test_service()
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"AI service test failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"AI service test failed: {str(e)}"
                )
        
                       @self.app.get("/status", tags=["System"])
               async def get_system_status(api_key: str = Depends(verify_api_key)):
                   """Get comprehensive system status with enhanced monitoring."""
                   config_validation = validate_config(self.config)
                   
                   # Get performance trends
                   caption_trends = self.performance_monitor.get_performance_trends("caption_generation", 60)
                   batch_trends = self.performance_monitor.get_performance_trends("batch_generation", 60)
                   
                   # Get active alerts
                   active_alerts = self.performance_monitor.get_alerts()
                   
                   return {
                       "system_status": "operational",
                       "config_validation": config_validation,
                       "ai_engine_status": "available" if self.ai_engine.pipeline else "fallback",
                       "cache_status": "operational",
                       "rate_limiting_status": "operational",
                       "circuit_breaker_status": self.circuit_breaker.get_status(),
                       "performance_trends": {
                           "caption_generation": caption_trends,
                           "batch_generation": batch_trends
                       },
                       "active_alerts": active_alerts,
                       "performance_summary": self.performance_monitor.get_performance_summary(),
                       "timestamp": time.time()
                   }
               
               @self.app.get("/circuit-breaker/status", tags=["System"])
               async def get_circuit_breaker_status(api_key: str = Depends(verify_api_key)):
                   """Get circuit breaker status and statistics."""
                   return {
                       "circuit_breaker": self.circuit_breaker.get_status(),
                       "performance_alerts": self.performance_monitor.get_alerts(),
                       "timestamp": time.time()
                   }
               
               @self.app.post("/circuit-breaker/reset", tags=["System"])
               async def reset_circuit_breaker(api_key: str = Depends(verify_api_key)):
                   """Reset circuit breaker to closed state."""
                   self.circuit_breaker.state = "CLOSED"
                   self.circuit_breaker.failure_count = 0
                   self.circuit_breaker.last_failure_time = 0
                   
                   logger.info("Circuit breaker manually reset")
                   
                   return {
                       "message": "Circuit breaker reset successfully",
                       "status": "CLOSED",
                       "timestamp": time.time()
                   }

# =============================================================================
# API INSTANCE
# =============================================================================

# Create API instance
api_instance = RefactoredCaptionsAPI()
app = api_instance.app

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": time.time()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("🚀 Instagram Captions API v10.0 starting up...")
    logger.info(f"📊 Environment: {api_instance.config.ENVIRONMENT}")
    logger.info(f"🔧 AI Model: {api_instance.config.AI_MODEL_NAME}")
    logger.info(f"🌐 Server: {api_instance.config.HOST}:{api_instance.config.PORT}")
    
    # Validate configuration
    validation_result = validate_config(api_instance.config)
    if not validation_result["valid"]:
        logger.warning("⚠️ Configuration validation warnings:")
        for warning in validation_result["warnings"]:
            logger.warning(f"   • {warning}")
    
    if validation_result["errors"]:
        logger.error("❌ Configuration validation errors:")
        for error in validation_result["errors"]:
            logger.error(f"   • {error}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("🛑 Instagram Captions API v10.0 shutting down...")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_v10:app",
        host=api_instance.config.HOST,
        port=api_instance.config.PORT,
        reload=api_instance.config.ENVIRONMENT == "development",
        log_level="info"
    )
