"""
API Best Practices Implementation for AI History Comparison System
Mejores Prácticas de API para el Sistema de Comparación de Historial de IA
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
import uuid
import time
import logging
from enum import Enum

# =============================================================================
# 1. ESTRUCTURA Y CONVENCIONES DE API
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool = Field(..., description="Indicates if the request was successful")
    data: Optional[Any] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Human-readable message")
    errors: Optional[List[str]] = Field(None, description="List of errors if any")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = Field(default="v1", description="API version")

class PaginationParams(BaseModel):
    """Standard pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort: Optional[str] = Field(None, description="Sort field")
    order: Optional[str] = Field(default="asc", regex="^(asc|desc)$", description="Sort order")

class PaginatedResponse(APIResponse):
    """Paginated response format"""
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    
    @classmethod
    def create(
        cls,
        data: List[Any],
        total: int,
        page: int,
        size: int,
        message: str = "Success"
    ):
        """Create a paginated response"""
        total_pages = (total + size - 1) // size
        return cls(
            success=True,
            data=data,
            message=message,
            pagination={
                "page": page,
                "size": size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        )

# =============================================================================
# 2. VALIDACIÓN Y ESQUEMAS DE DATOS
# =============================================================================

class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis"""
    content: str = Field(..., min_length=1, max_length=10000, description="Content to analyze")
    model_version: Optional[str] = Field(None, description="AI model version")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")
    include_metadata: bool = Field(default=True, description="Include metadata in response")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = ['comprehensive', 'basic', 'sentiment', 'readability']
        if v not in allowed_types:
            raise ValueError(f'Analysis type must be one of: {allowed_types}')
        return v

class ComparisonRequest(BaseModel):
    """Request model for content comparison"""
    content1: str = Field(..., min_length=1, max_length=10000)
    content2: str = Field(..., min_length=1, max_length=10000)
    comparison_type: str = Field(default="similarity", description="Type of comparison")
    include_recommendations: bool = Field(default=True)
    
    @validator('content1', 'content2')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

# =============================================================================
# 3. MANEJO DE ERRORES Y CÓDIGOS DE ESTADO
# =============================================================================

class APIError(Exception):
    """Custom API error class"""
    def __init__(self, message: str, status_code: int = 400, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)

class ErrorCodes(Enum):
    """Standard error codes"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"

def create_error_response(
    message: str,
    status_code: int = 400,
    error_code: str = None,
    details: Dict[str, Any] = None
) -> JSONResponse:
    """Create standardized error response"""
    error_data = {
        "success": False,
        "message": message,
        "error_code": error_code,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4())
    }
    
    if details:
        error_data["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=error_data
    )

# =============================================================================
# 4. MIDDLEWARE Y SEGURIDAD
# =============================================================================

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}

def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage.get(client_ip, [])
        if current_time - timestamp < 60  # 1 minute window
    ]
    
    # Check rate limit (10 requests per minute)
    if len(rate_limit_storage.get(client_ip, [])) >= 10:
        return create_error_response(
            "Rate limit exceeded",
            status_code=429,
            error_code=ErrorCodes.RATE_LIMITED.value
        )
    
    # Add current request
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    rate_limit_storage[client_ip].append(current_time)
    
    response = await call_next(request)
    return response

def request_logging_middleware(request: Request, call_next):
    """Request logging middleware"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Log request
    logging.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logging.info(f"Response {request_id}: {response.status_code} - {process_time:.3f}s")
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# =============================================================================
# 5. AUTENTICACIÓN Y AUTORIZACIÓN
# =============================================================================

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    # In production, validate JWT token here
    # For now, return a mock user
    return {
        "user_id": "user_123",
        "email": "user@example.com",
        "permissions": ["read", "write"]
    }

async def require_permission(permission: str):
    """Require specific permission"""
    def permission_checker(user: dict = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise APIError(
                "Insufficient permissions",
                status_code=403,
                error_code=ErrorCodes.FORBIDDEN.value
            )
        return user
    return permission_checker

# =============================================================================
# 6. CACHING Y OPTIMIZACIÓN
# =============================================================================

from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_analysis(content_hash: str, analysis_type: str):
    """Cached content analysis (in production, use Redis)"""
    # This would be implemented with actual analysis logic
    return {
        "readability_score": 0.8,
        "sentiment_score": 0.6,
        "complexity_score": 0.7,
        "cached": True
    }

def get_content_hash(content: str) -> str:
    """Generate content hash for caching"""
    return hashlib.md5(content.encode()).hexdigest()

# =============================================================================
# 7. DOCUMENTACIÓN Y METADATOS
# =============================================================================

def create_api_tags():
    """Create API tags for documentation"""
    return [
        {
            "name": "Content Analysis",
            "description": "Analyze content quality, readability, and sentiment"
        },
        {
            "name": "Comparison",
            "description": "Compare content between different versions or models"
        },
        {
            "name": "Trends",
            "description": "Analyze trends and patterns over time"
        },
        {
            "name": "Reports",
            "description": "Generate comprehensive analysis reports"
        },
        {
            "name": "System",
            "description": "System health and monitoring endpoints"
        }
    ]

def create_api_responses():
    """Create standard API responses for documentation"""
    return {
        200: {
            "description": "Successful response",
            "model": APIResponse
        },
        400: {
            "description": "Bad request - validation error",
            "model": APIResponse
        },
        401: {
            "description": "Unauthorized - authentication required",
            "model": APIResponse
        },
        403: {
            "description": "Forbidden - insufficient permissions",
            "model": APIResponse
        },
        404: {
            "description": "Not found - resource does not exist",
            "model": APIResponse
        },
        429: {
            "description": "Too many requests - rate limit exceeded",
            "model": APIResponse
        },
        500: {
            "description": "Internal server error",
            "model": APIResponse
        }
    }

# =============================================================================
# 8. UTILIDADES Y HELPERS
# =============================================================================

def validate_pagination(page: int, size: int) -> tuple:
    """Validate and normalize pagination parameters"""
    page = max(1, page)
    size = max(1, min(100, size))  # Limit to 100 items per page
    return page, size

def create_success_response(
    data: Any = None,
    message: str = "Success",
    request_id: str = None
) -> APIResponse:
    """Create standardized success response"""
    return APIResponse(
        success=True,
        data=data,
        message=message,
        request_id=request_id or str(uuid.uuid4())
    )

def create_paginated_success_response(
    data: List[Any],
    total: int,
    page: int,
    size: int,
    message: str = "Success",
    request_id: str = None
) -> PaginatedResponse:
    """Create standardized paginated success response"""
    return PaginatedResponse.create(
        data=data,
        total=total,
        page=page,
        size=size,
        message=message
    )

# =============================================================================
# 9. CONFIGURACIÓN DE FASTAPI
# =============================================================================

def create_optimized_fastapi_app() -> FastAPI:
    """Create FastAPI app with best practices"""
    
    app = FastAPI(
        title="AI History Comparison API",
        description="""
        ## AI History Comparison System API
        
        A comprehensive API for analyzing, comparing, and tracking AI model outputs over time.
        
        ### Features
        - **Content Analysis**: Analyze content quality, readability, sentiment
        - **Historical Comparison**: Compare content across different time periods
        - **Trend Analysis**: Track performance trends and patterns
        - **Quality Reporting**: Generate comprehensive quality reports
        
        ### Authentication
        All endpoints require Bearer token authentication.
        
        ### Rate Limiting
        API is rate limited to 100 requests per minute per IP.
        """,
        version="1.0.0",
        contact={
            "name": "API Support",
            "email": "support@ai-history.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        tags=create_api_tags()
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    # Add custom middleware
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(request_logging_middleware)
    
    return app

# =============================================================================
# 10. EJEMPLOS DE USO
# =============================================================================

def example_endpoints(app: FastAPI):
    """Example endpoints demonstrating best practices"""
    
    @app.post(
        "/api/v1/analyze",
        response_model=APIResponse,
        responses=create_api_responses(),
        tags=["Content Analysis"],
        summary="Analyze content",
        description="Analyze content for quality, readability, and sentiment metrics"
    )
    async def analyze_content(
        request: ContentAnalysisRequest,
        current_user: dict = Depends(require_permission("read"))
    ):
        """Analyze content with comprehensive metrics"""
        try:
            # Check cache first
            content_hash = get_content_hash(request.content)
            cached_result = cached_analysis(content_hash, request.analysis_type)
            
            if cached_result.get("cached"):
                return create_success_response(
                    data=cached_result,
                    message="Analysis completed (cached)"
                )
            
            # Perform analysis (mock implementation)
            analysis_result = {
                "readability_score": 0.85,
                "sentiment_score": 0.72,
                "complexity_score": 0.68,
                "word_count": len(request.content.split()),
                "sentence_count": request.content.count('.') + 1,
                "analysis_type": request.analysis_type,
                "model_version": request.model_version or "default"
            }
            
            return create_success_response(
                data=analysis_result,
                message="Analysis completed successfully"
            )
            
        except Exception as e:
            logging.error(f"Analysis error: {str(e)}")
            raise APIError(
                "Analysis failed",
                status_code=500,
                error_code=ErrorCodes.INTERNAL_ERROR.value
            )
    
    @app.get(
        "/api/v1/health",
        response_model=APIResponse,
        tags=["System"],
        summary="Health check",
        description="Check system health and status"
    )
    async def health_check():
        """System health check endpoint"""
        return create_success_response(
            data={
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0"
            },
            message="System is healthy"
        )

if __name__ == "__main__":
    # Create and run the app
    app = create_optimized_fastapi_app()
    example_endpoints(app)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)







