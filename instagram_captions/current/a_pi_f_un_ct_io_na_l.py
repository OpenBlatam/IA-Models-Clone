from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request, Depends, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from .security_functions import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Functional Instagram Captions API

Pure functions, declarative patterns, no classes.
Uses security_functions.py for all security operations.
"""


# Import functional security
    validate_api_key, enforce_rate_limit, sanitize_content,
    validate_request_data, log_security_event, log_authentication_attempt,
    create_security_middleware, add_security_headers, generate_request_id,
    is_suspicious_request, calculate_request_hash
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


# =============================================================================
# PYDANTIC MODELS (Functional approach)
# =============================================================================

class CaptionRequest(BaseModel):
    content_description: str = Field(..., min_length=10, max_length=1000)
    style: str = Field(default="casual")
    tone: str = Field(default="friendly")
    hashtag_count: int = Field(default=15, ge=0, le=30)
    language: str = Field(default="en")
    include_emoji: bool = Field(default=True)
    
    @validator('content_description')
    def validate_content(cls, v) -> bool:
        return sanitize_content(v)
    
    @validator('style')
    def validate_style(cls, v) -> bool:
        allowed_styles = ['casual', 'formal', 'creative', 'professional']
        if v not in allowed_styles:
            raise ValueError('Invalid style')
        return v


class CaptionResponse(BaseModel):
    caption: str
    hashtags: List[str]
    style: str
    tone: str
    processing_time: float
    confidence_score: float
    request_id: str


class BatchRequest(BaseModel):
    requests: List[CaptionRequest] = Field(..., min_length=1, max_length=50)
    batch_id: str = Field(default_factory=lambda: f"batch_{int(time.time())}")


class BatchResponse(BaseModel):
    results: List[CaptionResponse]
    batch_id: str
    total_processing_time: float
    successful_count: int
    failed_count: int


# =============================================================================
# PURE FUNCTIONS - AUTHENTICATION
# =============================================================================

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication."""
    api_key = credentials.credentials
    
    if not validate_api_key(api_key):
        log_authentication_attempt("unknown", False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    log_authentication_attempt(api_key.split('.')[0] if '.' in api_key else "unknown", True)
    return api_key


async def get_user_id_from_api_key(api_key: str) -> str:
    """Extract user ID from API key."""
    try:
        return api_key.split('.')[0]
    except (IndexError, AttributeError):
        return "unknown"


# =============================================================================
# PURE FUNCTIONS - REQUEST PROCESSING
# =============================================================================

async def process_caption_request(request_data: Dict[str, Any], user_id: str) -> CaptionResponse:
    """Process single caption request."""
    start_time = time.time()
    
    # Validate and sanitize input
    is_valid, cleaned_data, error_message = validate_request_data(request_data)
    if not is_valid:
        log_security_event("input_validation_error", user_id, {"error": error_message})
        raise HTTPException(status_code=400, detail=error_message)
    
    # Check for suspicious patterns
    is_suspicious, reason = is_suspicious_request(cleaned_data)
    if is_suspicious:
        log_security_event("suspicious_request", user_id, {"reason": reason})
        raise HTTPException(status_code=400, detail="Invalid request")
    
    # Generate caption (simplified for demo)
    caption = f"Generated caption for: {cleaned_data['content_description'][:50]}..."
    hashtags = [f"#{cleaned_data['style']}", "#instagram", "#caption"]
    
    processing_time = time.time() - start_time
    
    return CaptionResponse(
        caption=caption,
        hashtags=hashtags,
        style=cleaned_data.get('style', 'casual'),
        tone=cleaned_data.get('tone', 'friendly'),
        processing_time=processing_time,
        confidence_score=0.95,
        request_id=generate_request_id()
    )


async def process_batch_request(batch_data: Dict[str, Any], user_id: str) -> BatchResponse:
    """Process batch caption request."""
    start_time = time.time()
    
    # Validate batch size
    requests = batch_data.get('requests', [])
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 requests per batch")
    
    results = []
    successful_count = 0
    failed_count = 0
    
    for request_data in requests:
        try:
            result = process_caption_request(request_data, user_id)
            results.append(result)
            successful_count += 1
        except Exception as e:
            failed_count += 1
            # Add error result
            results.append(CaptionResponse(
                caption=f"Error: {str(e)}",
                hashtags=[],
                style=request_data.get('style', 'casual'),
                tone=request_data.get('tone', 'friendly'),
                processing_time=0.0,
                confidence_score=0.0,
                request_id=generate_request_id()
            ))
    
    total_time = time.time() - start_time
    
    return BatchResponse(
        results=results,
        batch_id=batch_data.get('batch_id', f"batch_{int(time.time())}"),
        total_processing_time=total_time,
        successful_count=successful_count,
        failed_count=failed_count
    )


# =============================================================================
# PURE FUNCTIONS - HEALTH & METRICS
# =============================================================================

def get_health_status() -> Dict[str, Any]:
    """Get health status."""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "functional_v1.0",
        "services": {
            "api": "operational",
            "security": "operational",
            "rate_limiting": "operational"
        }
    }


def get_metrics() -> Dict[str, Any]:
    """Get performance metrics."""
    return {
        "requests_processed": 1000,
        "average_response_time": 0.5,
        "error_rate": 0.02,
        "active_users": 50,
        "cache_hit_rate": 0.85,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# =============================================================================
# PURE FUNCTIONS - API ENDPOINTS
# =============================================================================

def create_generate_endpoint(app: FastAPI):
    """Create caption generation endpoint."""
    
    @app.post(
        "/api/functional/generate",
        response_model=CaptionResponse,
        status_code=status.HTTP_201_CREATED
    )
    async def generate_caption(
        request: CaptionRequest,
        api_key: str = Depends(verify_api_key)
    ) -> CaptionResponse:
        """Generate single caption with functional security."""
        
        user_id = get_user_id_from_api_key(api_key)
        
        # Enforce rate limiting
        enforce_rate_limit(user_id)
        
        # Process request
        result = process_caption_request(request.dict(), user_id)
        
        # Log success
        log_security_event("caption_generated", user_id, {
            "request_id": result.request_id,
            "style": result.style,
            "processing_time": result.processing_time
        })
        
        return result


def create_batch_endpoint(app: FastAPI):
    """Create batch processing endpoint."""
    
    @app.post(
        "/api/functional/batch",
        response_model=BatchResponse,
        status_code=status.HTTP_202_ACCEPTED
    )
    async def generate_batch_captions(
        batch_request: BatchRequest,
        api_key: str = Depends(verify_api_key)
    ) -> BatchResponse:
        """Generate batch captions with functional security."""
        
        user_id = get_user_id_from_api_key(api_key)
        
        # Enforce rate limiting (stricter for batch)
        enforce_rate_limit(user_id)
        
        # Process batch
        result = process_batch_request(batch_request.dict(), user_id)
        
        # Log batch completion
        log_security_event("batch_completed", user_id, {
            "batch_id": result.batch_id,
            "total_requests": len(result.results),
            "successful": result.successful_count,
            "failed": result.failed_count,
            "processing_time": result.total_processing_time
        })
        
        return result


def create_health_endpoint(app: FastAPI):
    """Create health check endpoint."""
    
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return get_health_status()


def create_metrics_endpoint(app: FastAPI):
    """Create metrics endpoint."""
    
    @app.get("/metrics")
    async def get_metrics_endpoint() -> Dict[str, Any]:
        """Get performance metrics."""
        return get_metrics()


def create_info_endpoint(app: FastAPI):
    """Create API info endpoint."""
    
    @app.get("/api/functional/info")
    async async def get_api_info() -> Dict[str, Any]:
        """Get API information."""
        return {
            "name": "Instagram Captions API - Functional",
            "version": "1.0.0",
            "description": "Functional programming approach with pure functions",
            "features": [
                "Pure functions for all operations",
                "Functional security implementation",
                "Declarative patterns",
                "No classes - functional approach",
                "Immutable data structures",
                "Side-effect isolation"
            ],
            "endpoints": [
                "/api/functional/generate",
                "/api/functional/batch",
                "/health",
                "/metrics",
                "/api/functional/info"
            ],
            "security": [
                "JWT token validation",
                "Rate limiting",
                "Input sanitization",
                "XSS prevention",
                "Security logging"
            ]
        }


# =============================================================================
# PURE FUNCTIONS - MIDDLEWARE
# =============================================================================

def setup_middleware(app: FastAPI) -> None:
    """Setup middleware stack."""
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Security middleware
    security_middleware = create_security_middleware()
    app.middleware("http")(security_middleware)


def add_response_headers(response: Response, request_id: str, processing_time: float) -> None:
    """Add custom response headers."""
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = str(round(processing_time, 3))
    response.headers["X-API-Version"] = "functional_v1.0"


# =============================================================================
# PURE FUNCTIONS - APP CREATION
# =============================================================================

def create_functional_app() -> FastAPI:
    """Create FastAPI application with functional approach."""
    
    app = FastAPI(
        title="Instagram Captions API - Functional",
        description="Pure functions, declarative patterns, no classes",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Create endpoints
    create_generate_endpoint(app)
    create_batch_endpoint(app)
    create_health_endpoint(app)
    create_metrics_endpoint(app)
    create_info_endpoint(app)
    
    return app


# =============================================================================
# PURE FUNCTIONS - UTILITY
# =============================================================================

async def calculate_request_metrics(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate request metrics."""
    return {
        "content_length": len(str(request_data.get('content_description', ''))),
        "hashtag_count": request_data.get('hashtag_count', 0),
        "request_hash": calculate_request_hash(request_data),
        "timestamp": time.time()
    }


async def validate_batch_request(batch_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate batch request."""
    requests = batch_data.get('requests', [])
    
    if not requests:
        return False, "No requests provided"
    
    if len(requests) > 50:
        return False, "Maximum 50 requests per batch"
    
    # Check for duplicate content
    descriptions = [req.get('content_description', '') for req in requests]
    if len(descriptions) != len(set(descriptions)):
        return False, "Duplicate content descriptions not allowed"
    
    return True, "Valid"


def format_error_response(error: str, request_id: str) -> Dict[str, Any]:
    """Format error response."""
    return {
        "error": error,
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# =============================================================================
# FUNCTIONAL COMPOSITION
# =============================================================================

def compose_security_checks(user_id: str, request_data: Dict[str, Any]) -> bool:
    """Compose multiple security checks."""
    # Rate limiting
    enforce_rate_limit(user_id)
    
    # Input validation
    is_valid, _, error_message = validate_request_data(request_data)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Suspicious request check
    is_suspicious, reason = is_suspicious_request(request_data)
    if is_suspicious:
        raise HTTPException(status_code=400, detail=f"Suspicious request: {reason}")
    
    return True


def compose_processing_pipeline(request_data: Dict[str, Any], user_id: str) -> CaptionResponse:
    """Compose processing pipeline."""
    # Security checks
    compose_security_checks(user_id, request_data)
    
    # Process request
    result = process_caption_request(request_data, user_id)
    
    # Log event
    log_security_event("caption_generated", user_id, {
        "request_id": result.request_id,
        "processing_time": result.processing_time
    })
    
    return result


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

__all__ = [
    # Models
    "CaptionRequest", "CaptionResponse", "BatchRequest", "BatchResponse",
    
    # Authentication
    "verify_api_key", "get_user_id_from_api_key",
    
    # Processing
    "process_caption_request", "process_batch_request",
    
    # Health & Metrics
    "get_health_status", "get_metrics",
    
    # Endpoints
    "create_generate_endpoint", "create_batch_endpoint", 
    "create_health_endpoint", "create_metrics_endpoint", "create_info_endpoint",
    
    # Middleware
    "setup_middleware", "add_response_headers",
    
    # App creation
    "create_functional_app",
    
    # Utilities
    "calculate_request_metrics", "validate_batch_request", "format_error_response",
    
    # Composition
    "compose_security_checks", "compose_processing_pipeline"
] 