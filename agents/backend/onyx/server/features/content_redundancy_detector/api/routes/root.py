"""
Root Router - Root and API information endpoints
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter
from fastapi.responses import JSONResponse

try:
    from config import settings
except ImportError:
    logging.warning("config module not available")
    class Settings:
        app_name = "Content Redundancy Detector"
        app_version = "1.0.0"
    settings = Settings()

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Root"])


@router.get("/", response_model=Dict[str, Any])
async def get_root() -> JSONResponse:
    """Root endpoint with API information for frontend"""
    logger.info("Root endpoint accessed")
    
    return JSONResponse(content={
        "success": True,
        "data": {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "Advanced Content Redundancy Detector API with AI/ML capabilities",
            "status": "ready",
            "frontend_ready": True,
            "cors_enabled": True,
            "docs": "/docs",
            "api_docs": "/redoc",
            "health": "/health",
            "api_health": "/api/v1/health",
            "api_version": "v1",
            "timestamp": time.time()
        },
        "error": None,
        "endpoints": {
            "base": "/api/v1",
            "health": "/api/v1/health",
            "analyze": "/api/v1/analyze",
            "similarity": "/api/v1/similarity",
            "quality": "/api/v1/quality",
            "stats": "/api/v1/stats",
            "metrics": "/api/v1/metrics",
            "ai_sentiment": "/api/v1/ai/sentiment",
            "ai_language": "/api/v1/ai/language",
            "ai_topics": "/api/v1/ai/topics",
            "ai_semantic_similarity": "/api/v1/ai/semantic-similarity",
            "ai_plagiarism": "/api/v1/ai/plagiarism",
            "ai_entities": "/api/v1/ai/entities",
            "ai_summary": "/api/v1/ai/summary",
            "ai_readability": "/api/v1/ai/readability",
            "ai_comprehensive": "/api/v1/ai/comprehensive",
            "ai_batch": "/api/v1/ai/batch",
            "batch_process": "/api/v1/batch/process",
            "realtime_start": "/api/v1/realtime/start",
            "multimodal_analyze": "/api/v1/multimodal/analyze"
        },
        "features": [
            "Content redundancy analysis",
            "Text similarity comparison",
            "Content quality assessment",
            "AI-powered sentiment analysis",
            "Language detection",
            "Topic extraction",
            "Semantic similarity",
            "Plagiarism detection",
            "Entity extraction",
            "Text summarization",
            "Advanced readability analysis",
            "Comprehensive AI analysis",
            "Batch processing",
            "Real-time analysis",
            "Multimodal analysis",
            "Custom model training",
            "Analytics dashboard",
            "Webhook support",
            "Export capabilities",
            "Cloud integration",
            "Security features",
            "Monitoring and alerts",
            "Automation workflows"
        ]
    })






