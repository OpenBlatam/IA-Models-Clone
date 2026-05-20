"""
PDF Variantes API - Root Endpoint
API information and discovery endpoint
"""

from fastapi import APIRouter

router = APIRouter(tags=["Root"], prefix="")


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "PDF Variantes API",
        "version": "2.0.0",
        "description": "Advanced PDF processing with AI capabilities",
        "docs": "/docs",
        "health": "/health",
        "api_version": "v1",
        "status": "ready",
        "frontend_ready": True,
        "cors_enabled": True,
        "endpoints": {
            "pdf": "/api/v1/pdf",
            "variants": "/api/v1/variants",
            "topics": "/api/v1/topics",
            "brainstorm": "/api/v1/brainstorm",
            "collaboration": "/api/v1/collaboration",
            "export": "/api/v1/export",
            "analytics": "/api/v1/analytics",
            "search": "/api/v1/search",
            "batch": "/api/v1/batch",
            "health": "/api/v1/health",
        },
        "features": [
            "PDF Upload and Processing",
            "AI-Powered Variant Generation",
            "Topic Extraction and Analysis",
            "Brainstorming and Ideation",
            "Real-time Collaboration",
            "Advanced Export Options",
            "Analytics and Monitoring",
            "Enterprise Security"
        ]
    }

