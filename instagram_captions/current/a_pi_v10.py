from typing import Any, List, Dict, Optional
import time
import logging
import asyncio
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

"""
Instagram Captions API v10.0 - Refactored Architecture

Complete API solution consolidating v9.0 ultra-advanced capabilities
into a clean, maintainable, and deployable architecture.
"""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# =============================================================================
# MIDDLEWARE & SECURITY
# =============================================================================

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication."""
    api_key = credentials.credentials
    if not RefactoredUtils.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    # In production, implement proper rate limiting with Redis/database
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = "9999"
    return response

# =============================================================================
# REFACTORED API APPLICATION
# =============================================================================

class RefactoredCaptionsAPI:
    """
    Consolidated API application that combines the best of v9.0 ultra-advanced features
    with the simplicity and maintainability of refactored architecture.
    """
    
    def __init__(self) -> None:
        self.config = RefactoredConfig()
        self.metrics = Metrics()
        self.ai_engine = RefactoredAIEngine(self.config)
        self.app = self._create_app()
        self._setup_middleware()
        self._setup_routes()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with optimized settings."""
        return FastAPI(
            title="Instagram Captions API v10.0",
            description="Refactored ultra-advanced Instagram caption generation with essential AI capabilities",
            version="10.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
    
    def _setup_middleware(self) -> None:
        """Setup essential middleware stack."""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Rate limiting
        self.app.middleware("http")(rate_limit_middleware)
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/", tags=["Root"])
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "Instagram Captions API v10.0",
                "version": self.config.API_VERSION,
                "status": "running",
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
                "environment": self.config.ENVIRONMENT
            }
        
        @self.app.post("/generate", tags=["Captions"], response_model=RefactoredCaptionResponse)
        async def generate_caption(
            request: RefactoredCaptionRequest,
            api_key: str = Depends(verify_api_key)
        ):
            """Generate a single Instagram caption."""
            start_time = time.time()
            
            try:
                # Sanitize input
                request.text = RefactoredUtils.sanitize_text(request.text)
                
                # Generate caption
                response = await self.ai_engine.generate_caption(request)
                
                # Record metrics
                processing_time = time.time() - start_time
                self.metrics.record_request(True, processing_time)
                
                return response
                
            except Exception as e:
                logger.error(f"Error generating caption: {e}")
                processing_time = time.time() - start_time
                self.metrics.record_request(False, processing_time)
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate caption: {str(e)}"
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
                    req.text = RefactoredUtils.sanitize_text(req.text)
                    
                    # Generate caption
                    response = await self.ai_engine.generate_caption(req)
                    results.append(response)
                
                # Record metrics
                processing_time = time.time() - start_time
                self.metrics.record_request(True, processing_time)
                
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
            return self.metrics.get_stats()
        
        @self.app.get("/config", tags=["Configuration"])
        async def get_config(api_key: str = Depends(verify_api_key)):
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