"""
Router Registration - Centralized router setup
Modular approach following microservices patterns
"""

from fastapi import FastAPI
from typing import List, Optional

from ..core.config import get_settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)


def register_routers(app: FastAPI) -> None:
    """
    Register all domain routers
    Modular approach - each domain has its own router
    """
    settings = get_settings()
    
    try:
        # Core analysis routers
        from .routes.analysis import router as analysis_router
        from .routes.similarity import router as similarity_router
        from .routes.quality import router as quality_router
        
        app.include_router(
            analysis_router,
            prefix=f"{settings.api_v1_prefix}/analyze",
            tags=["Analysis"]
        )
        
        app.include_router(
            similarity_router,
            prefix=f"{settings.api_v1_prefix}/similarity",
            tags=["Similarity"]
        )
        
        app.include_router(
            quality_router,
            prefix=f"{settings.api_v1_prefix}/quality",
            tags=["Quality"]
        )
        
        # AI/ML routers (feature flags)
        if settings.enable_sentiment_analysis:
            from .routes.ai_sentiment import router as sentiment_router
            app.include_router(
                sentiment_router,
                prefix=f"{settings.api_v1_prefix}/ai/sentiment",
                tags=["AI - Sentiment"]
            )
        
        if settings.enable_topic_modeling:
            from .routes.ai_topics import router as topics_router
            app.include_router(
                topics_router,
                prefix=f"{settings.api_v1_prefix}/ai/topics",
                tags=["AI - Topics"]
            )
        
        if settings.enable_semantic_analysis:
            from .routes.ai_semantic import router as semantic_router
            app.include_router(
                semantic_router,
                prefix=f"{settings.api_v1_prefix}/ai/semantic",
                tags=["AI - Semantic"]
            )
        
        if settings.enable_plagiarism_detection:
            from .routes.ai_plagiarism import router as plagiarism_router
            app.include_router(
                plagiarism_router,
                prefix=f"{settings.api_v1_prefix}/ai/plagiarism",
                tags=["AI - Plagiarism"]
            )
        
        # System routers
        from .routes.health import router as health_router
        from .routes.metrics import router as metrics_router
        
        app.include_router(
            health_router,
            prefix=f"{settings.api_v1_prefix}/health",
            tags=["Health"]
        )
        
        app.include_router(
            metrics_router,
            prefix=f"{settings.api_v1_prefix}/metrics",
            tags=["Metrics"]
        )
        
        # Optional feature routers
        if settings.enable_batch_processing:
            from .routes.batch import router as batch_router
            app.include_router(
                batch_router,
                prefix=f"{settings.api_v1_prefix}/batch",
                tags=["Batch Processing"]
            )
        
        if settings.enable_export:
            from .routes.export import router as export_router
            app.include_router(
                export_router,
                prefix=f"{settings.api_v1_prefix}/export",
                tags=["Export"]
            )
        
        if settings.enable_webhooks:
            from .routes.webhooks import router as webhooks_router
            app.include_router(
                webhooks_router,
                prefix=f"{settings.api_v1_prefix}/webhooks",
                tags=["Webhooks"]
            )
        
        logger.info("✅ All routers registered successfully")
        
    except ImportError as e:
        # Graceful degradation - continue if optional routers are missing
        logger.warning(f"Some routers not available: {e}")
        logger.info("Continuing with available routers...")


# Fallback router for backward compatibility
def register_legacy_routers(app: FastAPI) -> None:
    """Register legacy routers from old structure"""
    try:
        from routers import router as legacy_router
        app.include_router(legacy_router, prefix="/api/v1/legacy", tags=["Legacy"])
        logger.info("Legacy routers registered for backward compatibility")
    except ImportError:
        logger.debug("Legacy routers not available")






