"""
Unified API Router for AI History Comparison System

This module creates a single, unified API router that consolidates all
endpoints from the various API modules into a clean, organized structure.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from ..core.config import get_config, SystemConfig
from .endpoints import (
    analysis_endpoints,
    comparison_endpoints,
    trend_endpoints,
    content_endpoints,
    system_endpoints
)

logger = logging.getLogger(__name__)


def create_api_router(config: SystemConfig = None) -> APIRouter:
    """
    Create the unified API router with all endpoints
    
    Args:
        config: System configuration (optional, will use global config if not provided)
        
    Returns:
        APIRouter: Configured router with all endpoints
    """
    if config is None:
        config = get_config()
    
    # Create main router
    router = APIRouter(
        prefix="/api/v1",
        tags=["AI History Comparison"],
        responses={
            404: {"description": "Not found"},
            500: {"description": "Internal server error"}
        }
    )
    
    # Include all endpoint modules
    router.include_router(
        analysis_endpoints.router,
        prefix="/analysis",
        tags=["Analysis"]
    )
    
    router.include_router(
        comparison_endpoints.router,
        prefix="/comparison",
        tags=["Comparison"]
    )
    
    router.include_router(
        trend_endpoints.router,
        prefix="/trends",
        tags=["Trends"]
    )
    
    router.include_router(
        content_endpoints.router,
        prefix="/content",
        tags=["Content"]
    )
    
    router.include_router(
        system_endpoints.router,
        prefix="/system",
        tags=["System"]
    )
    
    # Add root endpoint
    @router.get("/")
    async def api_root():
        """API root endpoint with system information"""
        return {
            "name": "AI History Comparison API",
            "version": "1.0.0",
            "description": "Unified API for AI content analysis and comparison",
            "endpoints": {
                "analysis": "/api/v1/analysis",
                "comparison": "/api/v1/comparison",
                "trends": "/api/v1/trends",
                "content": "/api/v1/content",
                "system": "/api/v1/system"
            },
            "features": {
                "content_analysis": config.features.get("content_analysis", False),
                "trend_analysis": config.features.get("trend_analysis", False),
                "comparison_engine": config.features.get("comparison_engine", False),
                "real_time_streaming": config.features.get("real_time_streaming", False),
                "advanced_analytics": config.features.get("advanced_analytics", False),
                "ai_governance": config.features.get("ai_governance", False),
                "content_lifecycle": config.features.get("content_lifecycle", False),
                "security_monitoring": config.features.get("security_monitoring", False),
                "quantum_computing": config.features.get("quantum_computing", False),
                "federated_learning": config.features.get("federated_learning", False),
                "neural_architecture_search": config.features.get("neural_architecture_search", False)
            }
        }
    
    # Add health check endpoint
    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            return {
                "status": "healthy",
                "version": "1.0.0",
                "environment": config.environment.value,
                "features_enabled": len([f for f in config.features.values() if f]),
                "total_features": len(config.features)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unhealthy")
    
    # Add metrics endpoint
    @router.get("/metrics")
    async def get_metrics():
        """Get system metrics"""
        try:
            return {
                "system": {
                    "environment": config.environment.value,
                    "debug_mode": config.debug,
                    "features_enabled": len([f for f in config.features.values() if f])
                },
                "database": {
                    "url": config.database.url.split("://")[0] + "://***",  # Hide credentials
                    "pool_size": config.database.pool_size
                },
                "redis": {
                    "url": config.redis.url.split("://")[0] + "://***",  # Hide credentials
                    "max_connections": config.redis.max_connections
                },
                "ai": {
                    "default_model": config.ai.default_model,
                    "max_tokens": config.ai.max_tokens,
                    "temperature": config.ai.temperature
                }
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve metrics")
    
    logger.info("Unified API router created successfully")
    return router


def create_legacy_router() -> APIRouter:
    """
    Create a legacy router for backward compatibility
    
    This router provides endpoints that match the old API structure
    to ensure existing integrations continue to work.
    """
    router = APIRouter(
        prefix="/ai-history",
        tags=["Legacy API"],
        deprecated=True
    )
    
    # Add legacy endpoints here if needed
    @router.get("/")
    async def legacy_root():
        """Legacy root endpoint"""
        return {
            "message": "This is a legacy endpoint. Please use /api/v1/ for new integrations.",
            "new_api": "/api/v1/",
            "deprecated": True
        }
    
    return router





















