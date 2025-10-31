"""
AI Workflow Optimization API Endpoints
======================================

REST API endpoints for AI-powered workflow optimization.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.ai_workflow_optimizer import (
    AIWorkflowOptimizer, OptimizationType, OptimizationStrategy,
    WorkflowMetrics, OptimizationSuggestion, WorkflowPattern, PerformancePrediction
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/optimization", tags=["AI Workflow Optimization"])

# Pydantic models
class OptimizationRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID to optimize")
    optimization_type: str = Field("comprehensive", description="Type of optimization")
    include_ml_predictions: bool = Field(True, description="Include ML predictions")

class PerformancePredictionRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for prediction")

class OptimizationApplyRequest(BaseModel):
    suggestion_id: str = Field(..., description="Optimization suggestion ID")
    workflow_id: str = Field(..., description="Workflow ID")

class WorkflowAnalysisRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID to analyze")
    include_patterns: bool = Field(True, description="Include pattern analysis")
    include_trends: bool = Field(True, description="Include trend analysis")

# Global optimizer instance
optimizer = None

def get_optimizer() -> AIWorkflowOptimizer:
    """Get global optimizer instance."""
    global optimizer
    if optimizer is None:
        optimizer = AIWorkflowOptimizer({"cache_enabled": True, "ml_enabled": True})
    return optimizer

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_optimizer(
    current_user: User = Depends(require_permission("optimization:manage"))
):
    """Initialize the AI workflow optimizer."""
    
    optimizer = get_optimizer()
    
    try:
        await optimizer.initialize()
        return {"message": "AI Workflow Optimizer initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize optimizer: {str(e)}")

@router.post("/workflow/optimize", response_model=List[Dict[str, Any]])
async def optimize_workflow(
    request: OptimizationRequest,
    current_user: User = Depends(require_permission("workflows:optimize"))
):
    """Optimize a specific workflow."""
    
    optimizer = get_optimizer()
    
    try:
        # Convert string to enum
        opt_type = OptimizationType(request.optimization_type)
        
        # Optimize workflow
        suggestions = await optimizer.optimize_workflow(request.workflow_id, opt_type)
        
        # Convert to dict format
        result = []
        for suggestion in suggestions:
            suggestion_dict = {
                "suggestion_id": suggestion.suggestion_id,
                "workflow_id": suggestion.workflow_id,
                "optimization_type": suggestion.optimization_type.value,
                "strategy": suggestion.strategy.value,
                "description": suggestion.description,
                "expected_improvement": suggestion.expected_improvement,
                "confidence_score": suggestion.confidence_score,
                "implementation_effort": suggestion.implementation_effort,
                "risk_level": suggestion.risk_level,
                "prerequisites": suggestion.prerequisites,
                "steps": suggestion.steps,
                "estimated_impact": suggestion.estimated_impact,
                "cost_benefit_analysis": suggestion.cost_benefit_analysis,
                "created_at": suggestion.created_at.isoformat()
            }
            result.append(suggestion_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize workflow: {str(e)}")

@router.post("/workflow/predict", response_model=Dict[str, Any])
async def predict_workflow_performance(
    request: PerformancePredictionRequest,
    current_user: User = Depends(require_permission("workflows:view"))
):
    """Predict workflow performance using ML models."""
    
    optimizer = get_optimizer()
    
    try:
        prediction = await optimizer.predict_workflow_performance(
            request.workflow_id, 
            request.input_data
        )
        
        return {
            "workflow_id": prediction.workflow_id,
            "predicted_execution_time": prediction.predicted_execution_time,
            "predicted_success_rate": prediction.predicted_success_rate,
            "predicted_resource_usage": prediction.predicted_resource_usage,
            "confidence_interval": prediction.confidence_interval,
            "factors": prediction.factors,
            "recommendations": prediction.recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict workflow performance: {str(e)}")

@router.post("/workflow/analyze", response_model=Dict[str, Any])
async def analyze_workflow(
    request: WorkflowAnalysisRequest,
    current_user: User = Depends(require_permission("workflows:view"))
):
    """Analyze workflow for patterns and trends."""
    
    optimizer = get_optimizer()
    
    try:
        # Get workflow execution history
        execution_history = await optimizer._get_workflow_execution_history(request.workflow_id)
        
        # Get workflow data
        from ..services.database_service import DatabaseService
        db_service = DatabaseService({"cache_enabled": True})
        workflow = await db_service.get_workflow_by_id(request.workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Analyze performance
        metrics = await optimizer._analyze_workflow_performance(workflow, execution_history)
        
        result = {
            "workflow_id": request.workflow_id,
            "metrics": {
                "execution_time": metrics.execution_time,
                "success_rate": metrics.success_rate,
                "resource_usage": metrics.resource_usage,
                "cost": metrics.cost,
                "quality_score": metrics.quality_score,
                "error_rate": metrics.error_rate,
                "throughput": metrics.throughput,
                "latency": metrics.latency,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "network_usage": metrics.network_usage,
                "user_satisfaction": metrics.user_satisfaction,
                "business_value": metrics.business_value
            },
            "execution_count": len(execution_history)
        }
        
        # Add pattern analysis if requested
        if request.include_patterns:
            patterns = await optimizer._identify_workflow_patterns(execution_history)
            result["patterns"] = [
                {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "success_rate": pattern.success_rate,
                    "avg_execution_time": pattern.avg_execution_time,
                    "common_issues": pattern.common_issues,
                    "optimization_opportunities": pattern.optimization_opportunities
                }
                for pattern in patterns
            ]
        
        # Add trend analysis if requested
        if request.include_trends:
            trends = await optimizer._analyze_performance_trends(execution_history)
            result["trends"] = trends
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze workflow: {str(e)}")

@router.get("/insights", response_model=Dict[str, Any])
async def get_optimization_insights(
    current_user: User = Depends(require_permission("optimization:view"))
):
    """Get overall optimization insights."""
    
    optimizer = get_optimizer()
    
    try:
        insights = await optimizer.get_optimization_insights()
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization insights: {str(e)}")

@router.post("/apply", response_model=Dict[str, Any])
async def apply_optimization(
    request: OptimizationApplyRequest,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("workflows:optimize"))
):
    """Apply an optimization suggestion to a workflow."""
    
    optimizer = get_optimizer()
    
    try:
        if background_tasks:
            background_tasks.add_task(
                optimizer.apply_optimization, 
                request.suggestion_id, 
                request.workflow_id
            )
            return {
                "message": "Optimization application scheduled",
                "suggestion_id": request.suggestion_id,
                "workflow_id": request.workflow_id
            }
        else:
            result = await optimizer.apply_optimization(
                request.suggestion_id, 
                request.workflow_id
            )
            return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply optimization: {str(e)}")

@router.get("/patterns", response_model=List[Dict[str, Any]])
async def get_workflow_patterns(
    current_user: User = Depends(require_permission("optimization:view"))
):
    """Get identified workflow patterns."""
    
    optimizer = get_optimizer()
    
    try:
        patterns = list(optimizer.pattern_cache.values())
        
        result = []
        for pattern in patterns:
            pattern_dict = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "success_rate": pattern.success_rate,
                "avg_execution_time": pattern.avg_execution_time,
                "common_issues": pattern.common_issues,
                "optimization_opportunities": pattern.optimization_opportunities,
                "similar_workflows": pattern.similar_workflows
            }
            result.append(pattern_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow patterns: {str(e)}")

@router.get("/models/status", response_model=Dict[str, Any])
async def get_ml_models_status(
    current_user: User = Depends(require_permission("optimization:view"))
):
    """Get ML models training status."""
    
    optimizer = get_optimizer()
    
    try:
        models_status = {
            "total_models": len(optimizer.ml_models),
            "trained_models": list(optimizer.ml_models.keys()),
            "model_configs": list(optimizer.model_configs.keys()),
            "training_data_size": len(await optimizer._get_historical_workflow_data())
        }
        
        return models_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ML models status: {str(e)}")

@router.post("/models/retrain", response_model=Dict[str, str])
async def retrain_ml_models(
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("optimization:manage"))
):
    """Retrain ML models with latest data."""
    
    optimizer = get_optimizer()
    
    try:
        if background_tasks:
            background_tasks.add_task(optimizer._load_ml_models)
            return {"message": "ML models retraining scheduled"}
        else:
            await optimizer._load_ml_models()
            return {"message": "ML models retrained successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrain ML models: {str(e)}")

@router.get("/recommendations", response_model=List[str])
async def get_global_recommendations(
    current_user: User = Depends(require_permission("optimization:view"))
):
    """Get global optimization recommendations."""
    
    optimizer = get_optimizer()
    
    try:
        recommendations = await optimizer._get_global_recommendations()
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get global recommendations: {str(e)}")

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_optimization_history(
    limit: int = Query(50, description="Maximum number of records to return"),
    current_user: User = Depends(require_permission("optimization:view"))
):
    """Get optimization history."""
    
    optimizer = get_optimizer()
    
    try:
        history = optimizer.optimization_history[-limit:] if limit else optimizer.optimization_history
        
        result = []
        for record in history:
            record_dict = dict(record)
            if "timestamp" in record_dict and hasattr(record_dict["timestamp"], "isoformat"):
                record_dict["timestamp"] = record_dict["timestamp"].isoformat()
            result.append(record_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization history: {str(e)}")

@router.get("/performance/trends", response_model=Dict[str, Any])
async def get_performance_trends(
    current_user: User = Depends(require_permission("optimization:view"))
):
    """Get performance trends analysis."""
    
    optimizer = get_optimizer()
    
    try:
        trends = optimizer.performance_cache
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance trends: {str(e)}")

@router.post("/batch/optimize", response_model=Dict[str, Any])
async def batch_optimize_workflows(
    workflow_ids: List[str],
    optimization_type: str = "comprehensive",
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("workflows:optimize"))
):
    """Optimize multiple workflows in batch."""
    
    optimizer = get_optimizer()
    
    try:
        if background_tasks:
            background_tasks.add_task(
                _batch_optimize_workflows, 
                optimizer, 
                workflow_ids, 
                optimization_type
            )
            return {
                "message": f"Batch optimization scheduled for {len(workflow_ids)} workflows",
                "workflow_ids": workflow_ids,
                "optimization_type": optimization_type
            }
        else:
            results = await _batch_optimize_workflows(optimizer, workflow_ids, optimization_type)
            return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to batch optimize workflows: {str(e)}")

async def _batch_optimize_workflows(optimizer: AIWorkflowOptimizer, workflow_ids: List[str], optimization_type: str) -> Dict[str, Any]:
    """Helper function for batch optimization."""
    results = {
        "total_workflows": len(workflow_ids),
        "successful": 0,
        "failed": 0,
        "results": []
    }
    
    opt_type = OptimizationType(optimization_type)
    
    for workflow_id in workflow_ids:
        try:
            suggestions = await optimizer.optimize_workflow(workflow_id, opt_type)
            results["results"].append({
                "workflow_id": workflow_id,
                "status": "success",
                "suggestions_count": len(suggestions)
            })
            results["successful"] += 1
        except Exception as e:
            results["results"].append({
                "workflow_id": workflow_id,
                "status": "error",
                "error": str(e)
            })
            results["failed"] += 1
    
    return results

@router.get("/health", response_model=Dict[str, Any])
async def optimization_health_check():
    """Optimization service health check."""
    
    optimizer = get_optimizer()
    
    try:
        # Check if optimizer is initialized
        initialized = hasattr(optimizer, '_initialized') and optimizer._initialized
        
        # Check ML models
        ml_models_ready = len(optimizer.ml_models) > 0
        
        # Check pattern cache
        patterns_available = len(optimizer.pattern_cache) > 0
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "ml_models_ready": ml_models_ready,
            "patterns_available": patterns_available,
            "total_optimizations": len(optimizer.optimization_history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }




























