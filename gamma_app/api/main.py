"""
Gamma App - Main API Application
FastAPI application for AI-powered content generation
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn

from .routes import content_router, collaboration_router, export_router, analytics_router
from .bul_routes import router as bul_router
from .metaverse_consciousness_routes import router as metaverse_consciousness_router
from .quantum_neural_routes import router as quantum_neural_router
from .space_time_routes import router as space_time_router
from .dimension_reality_routes import router as dimension_reality_router
from .consciousness_harmony_routes import router as consciousness_harmony_router
from .infinite_omnipotent_routes import router as infinite_omnipotent_router
from .transcendent_absolute_routes import router as transcendent_absolute_router
from .ultimate_cosmic_routes import router as ultimate_cosmic_router
from .eternal_omnipotent_routes import router as eternal_omnipotent_router
from .infinite_divine_routes import router as infinite_divine_router
from .models import ErrorResponse
from ..core.content_generator import ContentGenerator
from ..services.collaboration_service import CollaborationService
from ..services.analytics_service import AnalyticsService
from ..utils.config import get_settings
from ..utils.auth import verify_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
content_generator = None
collaboration_service = None
analytics_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global content_generator, collaboration_service, analytics_service
    
    # Startup
    logger.info("Starting Gamma App API...")
    
    # Initialize services
    settings = get_settings()
    
    content_generator = ContentGenerator({
        'openai_api_key': settings.openai_api_key,
        'anthropic_api_key': settings.anthropic_api_key,
        'openai_model': settings.openai_model,
        'anthropic_model': settings.anthropic_model
    })
    
    collaboration_service = CollaborationService()
    analytics_service = AnalyticsService()
    
    logger.info("Gamma App API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gamma App API...")

# Create FastAPI application
app = FastAPI(
    title="Gamma App API",
    description="AI-Powered Content Generation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(content_router, prefix="/api/v1/content", tags=["content"])
app.include_router(collaboration_router, prefix="/api/v1/collaboration", tags=["collaboration"])
app.include_router(export_router, prefix="/api/v1/export", tags=["export"])
app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(bul_router, prefix="/api/v1", tags=["bul"])
app.include_router(metaverse_consciousness_router, prefix="/api/v1", tags=["metaverse-consciousness"])
app.include_router(quantum_neural_router, prefix="/api/v1", tags=["quantum-neural"])
app.include_router(space_time_router, prefix="/api/v1", tags=["space-time"])
app.include_router(dimension_reality_router, prefix="/api/v1", tags=["dimension-reality"])
app.include_router(consciousness_harmony_router, prefix="/api/v1", tags=["consciousness-harmony"])
app.include_router(infinite_omnipotent_router, prefix="/api/v1", tags=["infinite-omnipotent"])
app.include_router(transcendent_absolute_router, prefix="/api/v1", tags=["transcendent-absolute"])
app.include_router(ultimate_cosmic_router, prefix="/api/v1", tags=["ultimate-cosmic"])
app.include_router(eternal_omnipotent_router, prefix="/api/v1", tags=["eternal-omnipotent"])
app.include_router(infinite_divine_router, prefix="/api/v1", tags=["infinite-divine"])

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Gamma App API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "content_generator": "healthy" if content_generator else "unavailable",
            "collaboration_service": "healthy" if collaboration_service else "unavailable",
            "analytics_service": "healthy" if analytics_service else "unavailable"
        }
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            timestamp="2024-01-01T00:00:00Z"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            timestamp="2024-01-01T00:00:00Z"
        ).dict()
    )

# Dependency to get content generator
async def get_content_generator() -> ContentGenerator:
    """Get content generator instance"""
    if not content_generator:
        raise HTTPException(
            status_code=503,
            detail="Content generator not available"
        )
    return content_generator

# Dependency to get collaboration service
async def get_collaboration_service() -> CollaborationService:
    """Get collaboration service instance"""
    if not collaboration_service:
        raise HTTPException(
            status_code=503,
            detail="Collaboration service not available"
        )
    return collaboration_service

# Dependency to get analytics service
async def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance"""
    if not analytics_service:
        raise HTTPException(
            status_code=503,
            detail="Analytics service not available"
        )
    return analytics_service

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        # Verify token
        user_data = verify_token(credentials.credentials)
        return user_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )






