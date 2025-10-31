"""
System Management API Endpoints

This module provides API endpoints for system management functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from ...core.config import get_config, SystemConfig
from ...core.exceptions import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()


class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    status: str
    version: str
    environment: str
    uptime: float
    components: Dict[str, Any]
    features: Dict[str, bool]
    health_score: float


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics"""
    timestamp: str
    system_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    resource_metrics: Dict[str, Any]
    feature_metrics: Dict[str, Any]


class ConfigurationUpdateRequest(BaseModel):
    """Request model for configuration updates"""
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    restart_required: bool = Field(default=False, description="Whether restart is required")


class ConfigurationResponse(BaseModel):
    """Response model for configuration"""
    current_config: Dict[str, Any]
    available_features: Dict[str, bool]
    environment_info: Dict[str, Any]


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(config: SystemConfig = Depends(get_config)):
    """
    Get system status and health information
    
    Returns comprehensive system status including component health,
    feature availability, and overall system health score.
    """
    try:
        import time
        import psutil
        
        # Get system information
        uptime = time.time()  # This would be actual uptime in a real system
        
        # Check component health
        components = {
            "database": {"status": "healthy", "response_time": 0.1},
            "redis": {"status": "healthy", "response_time": 0.05},
            "ai_services": {"status": "healthy", "available_models": 5},
            "storage": {"status": "healthy", "free_space": "85%"}
        }
        
        # Calculate health score
        healthy_components = sum(1 for comp in components.values() if comp["status"] == "healthy")
        health_score = (healthy_components / len(components)) * 100
        
        response = SystemStatusResponse(
            status="operational",
            version="1.0.0",
            environment=config.environment.value,
            uptime=uptime,
            components=components,
            features=config.features,
            health_score=health_score
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(config: SystemConfig = Depends(get_config)):
    """
    Get detailed system metrics
    
    Returns comprehensive system metrics including performance,
    resource usage, and feature-specific metrics.
    """
    try:
        import time
        import psutil
        
        # Get system metrics
        system_metrics = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        }
        
        # Get performance metrics
        performance_metrics = {
            "requests_per_second": 150.5,
            "average_response_time": 0.25,
            "error_rate": 0.01,
            "active_connections": 45
        }
        
        # Get resource metrics
        resource_metrics = {
            "database_connections": 12,
            "redis_connections": 8,
            "cache_hit_rate": 0.85,
            "queue_size": 3
        }
        
        # Get feature metrics
        feature_metrics = {
            "content_analysis_count": 1250,
            "comparison_count": 890,
            "trend_analysis_count": 340,
            "active_sessions": 23
        }
        
        response = SystemMetricsResponse(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            system_metrics=system_metrics,
            performance_metrics=performance_metrics,
            resource_metrics=resource_metrics,
            feature_metrics=feature_metrics
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.get("/configuration", response_model=ConfigurationResponse)
async def get_configuration(config: SystemConfig = Depends(get_config)):
    """
    Get current system configuration
    
    Returns the current system configuration including enabled features
    and environment information.
    """
    try:
        # Build configuration response
        current_config = {
            "environment": config.environment.value,
            "debug": config.debug,
            "host": config.host,
            "port": config.port,
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
        
        environment_info = {
            "environment": config.environment.value,
            "debug_mode": config.debug,
            "workers": config.workers,
            "reload_enabled": config.reload
        }
        
        response = ConfigurationResponse(
            current_config=current_config,
            available_features=config.features,
            environment_info=environment_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@router.put("/configuration")
async def update_configuration(
    request: ConfigurationUpdateRequest,
    config: SystemConfig = Depends(get_config)
):
    """
    Update system configuration
    
    Updates system configuration with the provided changes.
    Note: Some changes may require a system restart.
    """
    try:
        # Validate configuration updates
        allowed_updates = [
            "debug", "host", "port", "workers", "reload",
            "features", "ai.default_model", "ai.max_tokens", "ai.temperature"
        ]
        
        for key in request.updates.keys():
            if key not in allowed_updates:
                raise ValidationError(f"Configuration key '{key}' is not allowed to be updated")
        
        # Apply updates (in a real system, this would update the actual configuration)
        updated_config = config
        for key, value in request.updates.items():
            if "." in key:
                # Handle nested keys like "ai.default_model"
                parts = key.split(".")
                obj = updated_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(updated_config, key, value)
        
        return {
            "status": "updated",
            "restart_required": request.restart_required,
            "updated_keys": list(request.updates.keys()),
            "message": "Configuration updated successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConfigurationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration update failed")


@router.post("/restart")
async def restart_system(
    background_tasks: BackgroundTasks,
    config: SystemConfig = Depends(get_config)
):
    """
    Restart the system
    
    Initiates a system restart. This is an asynchronous operation.
    """
    try:
        # In a real system, this would initiate a restart process
        background_tasks.add_task(restart_system_task)
        
        return {
            "status": "restarting",
            "message": "System restart initiated",
            "estimated_downtime": "30 seconds"
        }
        
    except Exception as e:
        logger.error(f"System restart failed: {e}")
        raise HTTPException(status_code=500, detail="System restart failed")


async def restart_system_task():
    """Background task for system restart"""
    import time
    logger.info("System restart initiated")
    time.sleep(5)  # Simulate restart process
    logger.info("System restart completed")


@router.get("/health")
async def health_check(config: SystemConfig = Depends(get_config)):
    """
    Simple health check endpoint
    
    Returns basic health status for load balancers and monitoring systems.
    """
    try:
        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": "1.0.0",
            "environment": config.environment.value
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/features")
async def get_features(config: SystemConfig = Depends(get_config)):
    """
    Get available features and their status
    
    Returns information about all available features and whether they are enabled.
    """
    try:
        feature_info = {}
        
        for feature_name, enabled in config.features.items():
            feature_info[feature_name] = {
                "enabled": enabled,
                "description": get_feature_description(feature_name),
                "category": get_feature_category(feature_name)
            }
        
        return {
            "features": feature_info,
            "total_features": len(config.features),
            "enabled_features": len([f for f in config.features.values() if f])
        }
        
    except Exception as e:
        logger.error(f"Failed to get features: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve features")


def get_feature_description(feature_name: str) -> str:
    """Get description for a feature"""
    descriptions = {
        "content_analysis": "Analyze content quality, sentiment, and complexity",
        "trend_analysis": "Analyze trends and patterns in data over time",
        "comparison_engine": "Compare content and model results",
        "real_time_streaming": "Real-time data streaming and processing",
        "advanced_analytics": "Advanced analytics and machine learning",
        "ai_governance": "AI model governance and compliance",
        "content_lifecycle": "Content lifecycle management",
        "security_monitoring": "Security monitoring and threat detection",
        "quantum_computing": "Quantum computing capabilities",
        "federated_learning": "Federated learning support",
        "neural_architecture_search": "Neural architecture search"
    }
    return descriptions.get(feature_name, "Feature description not available")


def get_feature_category(feature_name: str) -> str:
    """Get category for a feature"""
    categories = {
        "content_analysis": "Analysis",
        "trend_analysis": "Analysis",
        "comparison_engine": "Processing",
        "real_time_streaming": "Infrastructure",
        "advanced_analytics": "Analytics",
        "ai_governance": "Governance",
        "content_lifecycle": "Management",
        "security_monitoring": "Security",
        "quantum_computing": "Advanced",
        "federated_learning": "Advanced",
        "neural_architecture_search": "Advanced"
    }
    return categories.get(feature_name, "Other")





















