"""
Performance Routes - API endpoints for Performance Optimizer
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ..core.performance_optimizer import (
    performance_optimizer, 
    PerformanceMetrics, 
    OptimizationRule,
    OptimizationResult
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/performance", tags=["Performance Optimization"])


# Pydantic models
class OptimizationRuleRequest(BaseModel):
    """Request model for optimization rule"""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    condition: str = Field(..., description="Rule condition")
    action: str = Field(..., description="Rule action")
    threshold: float = Field(..., description="Rule threshold")
    enabled: bool = Field(True, description="Whether rule is enabled")
    priority: int = Field(1, description="Rule priority")
    cooldown_seconds: int = Field(60, description="Cooldown period in seconds")


class OptimizationRuleUpdate(BaseModel):
    """Update model for optimization rule"""
    name: Optional[str] = Field(None, description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    condition: Optional[str] = Field(None, description="Rule condition")
    action: Optional[str] = Field(None, description="Rule action")
    threshold: Optional[float] = Field(None, description="Rule threshold")
    enabled: Optional[bool] = Field(None, description="Whether rule is enabled")
    priority: Optional[int] = Field(None, description="Rule priority")
    cooldown_seconds: Optional[int] = Field(None, description="Cooldown period in seconds")


class SystemOptimizationRequest(BaseModel):
    """Request model for system optimization"""
    optimize_memory: bool = Field(True, description="Optimize memory usage")
    optimize_database: bool = Field(True, description="Optimize database")
    optimize_api: bool = Field(True, description="Optimize API responses")
    optimize_async: bool = Field(True, description="Optimize async operations")


class QueryOptimizationRequest(BaseModel):
    """Request model for query optimization"""
    queries: List[str] = Field(..., description="List of queries to optimize")


class ResponseOptimizationRequest(BaseModel):
    """Request model for response optimization"""
    response_data: Any = Field(..., description="Response data to optimize")
    endpoint: str = Field(..., description="API endpoint")


class BatchOptimizationRequest(BaseModel):
    """Request model for batch optimization"""
    requests: List[Dict[str, Any]] = Field(..., description="List of requests to optimize")


# Performance monitoring endpoints
@router.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics():
    """Get current performance metrics"""
    try:
        metrics = await performance_optimizer.get_performance_metrics()
        return {
            "success": True,
            "data": metrics,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/history", response_model=Dict[str, Any])
async def get_performance_history(
    limit: int = Query(100, description="Number of metrics to return")
):
    """Get performance metrics history"""
    try:
        # Get recent metrics (this would need to be implemented in the optimizer)
        return {
            "success": True,
            "data": {
                "message": "Performance history endpoint",
                "limit": limit,
                "note": "Implementation needed for metrics history storage"
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System optimization endpoints
@router.post("/optimize/system", response_model=Dict[str, Any])
async def optimize_system(
    request: SystemOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Optimize entire system"""
    try:
        # Run optimization in background
        background_tasks.add_task(performance_optimizer.optimize_system)
        
        return {
            "success": True,
            "message": "System optimization started in background",
            "optimization_options": {
                "memory": request.optimize_memory,
                "database": request.optimize_database,
                "api": request.optimize_api,
                "async": request.optimize_async
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/memory", response_model=Dict[str, Any])
async def optimize_memory():
    """Optimize memory usage"""
    try:
        result = await performance_optimizer.memory_optimizer.optimize_memory()
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/database", response_model=Dict[str, Any])
async def optimize_database():
    """Optimize database connections and queries"""
    try:
        result = await performance_optimizer.db_optimizer.optimize_connection_pool()
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/queries", response_model=Dict[str, Any])
async def optimize_queries(request: QueryOptimizationRequest):
    """Optimize database queries"""
    try:
        result = await performance_optimizer.db_optimizer.optimize_queries(request.queries)
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/api", response_model=Dict[str, Any])
async def optimize_api_responses(request: ResponseOptimizationRequest):
    """Optimize API responses"""
    try:
        result = await performance_optimizer.api_optimizer.optimize_response(
            request.response_data, 
            request.endpoint
        )
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing API responses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/batch", response_model=Dict[str, Any])
async def optimize_batch_requests(request: BatchOptimizationRequest):
    """Optimize batch API requests"""
    try:
        result = await performance_optimizer.api_optimizer.optimize_batch_requests(request.requests)
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing batch requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/async", response_model=Dict[str, Any])
async def optimize_async_operations():
    """Optimize async operations"""
    try:
        # This would need actual tasks to optimize
        result = await performance_optimizer.async_optimizer.optimize_io_operations([])
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing async operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache optimization endpoints
@router.post("/optimize/cache", response_model=Dict[str, Any])
async def optimize_cache_strategy():
    """Optimize cache strategy"""
    try:
        result = await performance_optimizer.optimize_cache_strategy()
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing cache strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/compression", response_model=Dict[str, Any])
async def optimize_response_compression():
    """Optimize response compression"""
    try:
        result = await performance_optimizer.optimize_response_compression()
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error optimizing response compression: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optimization rules management
@router.get("/rules", response_model=Dict[str, Any])
async def get_optimization_rules():
    """Get optimization rules"""
    try:
        result = await performance_optimizer.get_optimization_rules()
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting optimization rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rules", response_model=Dict[str, Any])
async def add_optimization_rule(request: OptimizationRuleRequest):
    """Add new optimization rule"""
    try:
        rule = OptimizationRule(
            rule_id=request.rule_id,
            name=request.name,
            description=request.description,
            condition=request.condition,
            action=request.action,
            threshold=request.threshold,
            enabled=request.enabled,
            priority=request.priority,
            cooldown_seconds=request.cooldown_seconds
        )
        
        result = await performance_optimizer.add_optimization_rule(rule)
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error adding optimization rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rules/{rule_id}", response_model=Dict[str, Any])
async def update_optimization_rule(
    rule_id: str,
    request: OptimizationRuleUpdate
):
    """Update optimization rule"""
    try:
        updates = request.dict(exclude_unset=True)
        result = await performance_optimizer.update_optimization_rule(rule_id, updates)
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error updating optimization rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/{rule_id}", response_model=Dict[str, Any])
async def delete_optimization_rule(rule_id: str):
    """Delete optimization rule"""
    try:
        result = await performance_optimizer.delete_optimization_rule(rule_id)
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error deleting optimization rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optimization history
@router.get("/history", response_model=Dict[str, Any])
async def get_optimization_history():
    """Get optimization history"""
    try:
        result = await performance_optimizer.get_optimization_history()
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance analysis endpoints
@router.get("/analysis", response_model=Dict[str, Any])
async def get_performance_analysis():
    """Get performance analysis"""
    try:
        # Get current metrics
        metrics = await performance_optimizer.get_performance_metrics()
        
        # Get optimization history
        history = await performance_optimizer.get_optimization_history()
        
        # Analyze performance trends
        analysis = {
            "current_performance": metrics,
            "optimization_history": history,
            "recommendations": []
        }
        
        # Add recommendations based on current metrics
        if "cpu_percent" in metrics and metrics["cpu_percent"] > 80:
            analysis["recommendations"].append("High CPU usage detected - consider scaling")
        
        if "memory_percent" in metrics and metrics["memory_percent"] > 80:
            analysis["recommendations"].append("High memory usage detected - consider memory optimization")
        
        return {
            "success": True,
            "data": analysis,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting performance analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@router.get("/health", response_model=Dict[str, Any])
async def performance_health_check():
    """Performance optimizer health check"""
    try:
        # Check if optimizer is running
        metrics = await performance_optimizer.get_performance_metrics()
        
        return {
            "status": "healthy",
            "service": "Performance Optimizer",
            "timestamp": datetime.now(),
            "metrics_available": bool(metrics),
            "optimization_rules_count": len(performance_optimizer.optimization_rules),
            "active_monitoring": True
        }
    except Exception as e:
        logger.error(f"Performance optimizer health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "Performance Optimizer",
            "timestamp": datetime.now(),
            "error": str(e)
        }


# Performance capabilities
@router.get("/capabilities", response_model=Dict[str, Any])
async def get_performance_capabilities():
    """Get performance optimization capabilities"""
    return {
        "success": True,
        "data": {
            "memory_optimization": {
                "garbage_collection": True,
                "weak_reference_cleanup": True,
                "memory_profiling": True,
                "memory_tracking": True
            },
            "database_optimization": {
                "query_optimization": True,
                "connection_pool_optimization": True,
                "query_caching": True,
                "connection_monitoring": True
            },
            "api_optimization": {
                "response_compression": True,
                "response_caching": True,
                "batch_processing": True,
                "pagination_optimization": True
            },
            "async_optimization": {
                "concurrency_optimization": True,
                "io_operation_optimization": True,
                "task_scheduling": True,
                "resource_management": True
            },
            "monitoring": {
                "real_time_metrics": True,
                "performance_tracking": True,
                "optimization_history": True,
                "rule_based_optimization": True
            }
        },
        "timestamp": datetime.now()
    }