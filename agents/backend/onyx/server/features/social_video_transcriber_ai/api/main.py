"""
Main FastAPI application for Social Video Transcriber AI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from ..config.settings import get_settings
from .routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
# Social Video Transcriber AI

Transcribe videos from TikTok, Instagram, and YouTube with AI-powered analysis.

## Features

- **Video Transcription**: Transcribe videos with optional timestamps
- **Framework Analysis**: Detect content structure and framework (Hook-Story-Offer, AIDA, etc.)
- **Variant Generation**: Create text variants maintaining context and structure
- **Quick Variants**: One-click variant generation
- **Multi-Platform Support**: TikTok, Instagram, YouTube

## Endpoints

### Transcription
- `POST /api/v1/transcribe` - Start transcription job
- `GET /api/v1/transcribe/{job_id}` - Get job status
- `GET /api/v1/transcribe/{job_id}/text` - Get transcription text

### Analysis
- `POST /api/v1/analyze` - Analyze text structure and framework

### Variants
- `POST /api/v1/variants` - Generate variants from text or job
- `POST /api/v1/variants/quick` - Quick variant generation (one-click)
- `POST /api/v1/variants/text` - Generate variants from raw text

### Utilities
- `GET /api/v1/video/info` - Get video metadata
- `GET /api/v1/platforms` - List supported platforms
- `GET /api/v1/jobs` - List all jobs
""",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1", tags=["transcription"])
    
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {settings.api_title} v{settings.api_version}")
        logger.info(f"Environment: {settings.environment}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Social Video Transcriber AI")
    
    @app.get("/")
    async def root():
        return {
            "service": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
    )












