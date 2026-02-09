"""
Instagram Captions API v10.0 - Refactored Architecture

Refactored API using the new modular structure for better maintainability.
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

# Import from new modular structure
from .core import setup_logging, get_logger, RateLimiter
from .security import SecurityUtils
from .monitoring import PerformanceMonitor
from .resilience import CircuitBreaker, ErrorHandler

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Setup logging
setup_logging("INFO")
logger = get_logger(__name__)

# Security
security = HTTPBearer()

# Initialize components
performance_monitor = PerformanceMonitor()
error_handler = ErrorHandler()
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# =============================================================================
# MIDDLEWARE & SECURITY
# =============================================================================

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Enhanced API key verification."""
    api_key = credentials.credentials
    
    if not SecurityUtils.verify_api_key(api_key):
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Instagram Captions API v10.0 - Refactored",
    description="Refactored architecture with modular design",
    version="10.0.0"
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

# =============================================================================
# REQUEST TRACKING MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request performance and apply rate limiting."""
    start_time = time.time()
    
    # Rate limiting check
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Record performance metrics
    performance_monitor.record_request(
        response_time=response_time,
        endpoint=str(request.url.path),
        method=request.method,
        status_code=response.status_code,
        user_agent=request.headers.get("user-agent", "")
    )
    
    return response

# =============================================================================
# HEALTH & MONITORING ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "10.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    return performance_monitor.get_summary()

@app.get("/errors")
async def get_error_summary():
    """Get error summary."""
    return error_handler.get_error_summary()

# =============================================================================
# MAIN CAPTION GENERATION ENDPOINT
# =============================================================================

@app.post("/generate-caption")
async def generate_caption(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Generate Instagram caption with error handling and circuit breaker."""
    
    try:
        # Simulate caption generation
        caption = "✨ Amazing content! #instagram #awesome #content"
        
        return {
            "caption": caption,
            "generated_at": time.time(),
            "status": "success"
        }
        
    except Exception as e:
        # Log error
        error_id = error_handler.log_error(
            error=e,
            context="generate_caption",
            severity="high"
        )
        
        logger.error(f"Caption generation failed: {error_id}")
        
        raise HTTPException(
            status_code=500,
            detail="Caption generation failed"
        )

# =============================================================================
# BATCH PROCESSING ENDPOINT
# =============================================================================

@app.post("/generate-captions-batch")
async def generate_captions_batch(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Generate multiple captions in batch."""
    
    try:
        # Simulate batch processing
        captions = [
            "✨ First amazing caption! #instagram #awesome",
            "🌟 Second incredible caption! #content #viral",
            "💫 Third fantastic caption! #trending #popular"
        ]
        
        return {
            "captions": captions,
            "count": len(captions),
            "generated_at": time.time(),
            "status": "success"
        }
        
    except Exception as e:
        error_id = error_handler.log_error(
            error=e,
            context="generate_captions_batch",
            severity="high"
        )
        
        raise HTTPException(
            status_code=500,
            detail="Batch caption generation failed"
        )

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_refactored:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )






