from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.gradient_accumulation import (
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService
from onyx.server.features.ads.multi_gpu_training import GPUMonitor, GPUConfig
from onyx.server.features.ads.optimized_config import settings
from typing import Any, List, Dict, Optional
import logging
"""
Gradient Accumulation API for Onyx Ads Backend

This module provides REST API endpoints for:
- Gradient accumulation configuration and management
- Large batch size training optimization
- Memory-efficient training with accumulation
- Performance monitoring and statistics
- Automatic batch size calculation
- Integration with multi-GPU training
"""

    GradientAccumulationConfig,
    GradientAccumulator,
    AdaptiveGradientAccumulator,
    GradientAccumulationTrainer,
    calculate_effective_batch_size,
    calculate_accumulation_steps,
    adjust_learning_rate
)

logger = setup_logger()

router = APIRouter(prefix="/gradient-accumulation", tags=["Gradient Accumulation"])

# Pydantic models for API requests/responses
class GradientAccumulationRequest(BaseModel):
    """Gradient accumulation configuration request model."""
    accumulation_steps: int = 4
    target_effective_batch_size: Optional[int] = None
    target_batch_size: Optional[int] = None
    max_memory_usage: float = 0.9
    memory_safety_margin: float = 0.1
    auto_adjust_batch_size: bool = True
    sync_gradients: bool = True
    gradient_scaling: bool = True
    mixed_precision: bool = True
    log_accumulation: bool = True
    log_memory_usage: bool = True
    gradient_clipping: Optional[float] = 1.0
    warmup_steps: int = 0

class GradientAccumulationResponse(BaseModel):
    """Gradient accumulation configuration response model."""
    success: bool
    training_id: str
    config: Dict[str, Any]
    effective_batch_size: int
    accumulation_steps: int
    learning_rate_scale: float

class TrainingRequest(BaseModel):
    """Training request with gradient accumulation."""
    model_name: str
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    user_id: int
    target_effective_batch_size: int = 32
    accumulation_steps: Optional[int] = None
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    gradient_accumulation_config: Optional[GradientAccumulationRequest] = None

class TrainingResponse(BaseModel):
    """Training response with accumulation statistics."""
    success: bool
    training_id: str
    model_path: str
    training_time: float
    best_loss: float
    effective_batch_size: int
    actual_batch_size_per_gpu: int
    accumulation_steps: int
    adjusted_learning_rate: float
    accumulation_stats: Dict[str, Any]
    gpu_stats: Dict[str, Any]
    training_history: List[Dict[str, Any]]

class BatchSizeCalculationRequest(BaseModel):
    """Batch size calculation request model."""
    model_name: str
    target_effective_batch_size: int
    gpu_ids: Optional[List[int]] = None
    max_memory_usage: float = 0.9

class BatchSizeCalculationResponse(BaseModel):
    """Batch size calculation response model."""
    success: bool
    target_effective_batch_size: int
    actual_batch_size_per_gpu: int
    total_gpus: int
    accumulation_steps: int
    effective_batch_size: int
    gpu_info: Dict[str, Any]

class AccumulationStats(BaseModel):
    """Accumulation statistics response model."""
    training_id: str
    accumulation_steps: int
    current_step: int
    total_loss: float
    total_samples: int
    avg_gradient_norm: float
    avg_memory_usage_gb: float
    avg_accumulation_time: float
    gradient_norms: List[float]
    memory_usage: List[float]
    accumulation_times: List[float]

class OptimizationRequest(BaseModel):
    """Optimization request model."""
    action: str = Field(..., description="Action: calculate, optimize, monitor")
    model_name: Optional[str] = None
    target_batch_size: Optional[int] = None
    gpu_ids: Optional[List[int]] = None

class OptimizationResponse(BaseModel):
    """Optimization response model."""
    success: bool
    action: str
    details: Dict[str, Any]

# Dependency for fine-tuning service
async def get_finetuning_service() -> OptimizedFineTuningService:
    """Get fine-tuning service instance."""
    return OptimizedFineTuningService()

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for gradient accumulation system."""
    try:
        # Check PyTorch availability
        pytorch_available = torch.cuda.is_available() if torch.cuda.is_available() else True
        
        # Get GPU info
        gpu_monitor = GPUMonitor(GPUConfig())
        gpu_info = gpu_monitor.get_gpu_info()
        available_gpus = gpu_monitor.get_available_gpus()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pytorch_available": pytorch_available,
            "gpu_count": len(gpu_info),
            "available_gpus": available_gpus,
            "gpu_info": gpu_info
        }
    except Exception as e:
        logger.error(f"Gradient accumulation health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Configuration endpoints
@router.post("/config", response_model=GradientAccumulationResponse)
async def configure_gradient_accumulation(
    request: GradientAccumulationRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Configure gradient accumulation settings."""
    try:
        # Setup gradient accumulation
        accumulation_setup = await finetuning_service.setup_gradient_accumulation(
            target_effective_batch_size=request.target_effective_batch_size,
            accumulation_steps=request.accumulation_steps
        )
        
        # Calculate effective batch size
        effective_batch_size = calculate_effective_batch_size(
            request.target_batch_size or 8,
            request.accumulation_steps
        )
        
        # Calculate learning rate scale
        learning_rate_scale = 1.0 / request.accumulation_steps
        
        return GradientAccumulationResponse(
            success=True,
            training_id=accumulation_setup["training_id"],
            config=accumulation_setup["config"],
            effective_batch_size=effective_batch_size,
            accumulation_steps=request.accumulation_steps,
            learning_rate_scale=learning_rate_scale
        )
    except Exception as e:
        logger.error(f"Gradient accumulation configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/config/{training_id}")
async def get_gradient_accumulation_config(
    training_id: str,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Get gradient accumulation configuration for a training session."""
    try:
        # Get accumulation stats
        stats_response = await finetuning_service.get_accumulation_stats(training_id)
        
        if not stats_response["success"]:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        return {
            "training_id": training_id,
            "config": stats_response["stats"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get accumulation config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

# Training endpoints
@router.post("/training/with-accumulation", response_model=TrainingResponse)
async def train_with_accumulation(
    request: TrainingRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Train model with gradient accumulation for large batch sizes."""
    try:
        # Prepare dataset
        dataset = await finetuning_service.prepare_dataset(
            texts=request.dataset_config.get("texts", []),
            model_name=request.model_name,
            max_length=request.dataset_config.get("max_length", 512),
            batch_size=request.dataset_config.get("batch_size", 32)
        )
        
        # Train model with accumulation
        result = await finetuning_service.finetune_model_with_accumulation(
            model_name=request.model_name,
            dataset=dataset,
            training_config=request.training_config,
            user_id=request.user_id,
            target_effective_batch_size=request.target_effective_batch_size,
            accumulation_steps=request.accumulation_steps,
            distributed=request.distributed,
            world_size=request.world_size,
            rank=request.rank
        )
        
        return TrainingResponse(
            success=True,
            training_id=f"accumulation_{int(datetime.now().timestamp())}",
            model_path=result["model_path"],
            training_time=result["training_time"],
            best_loss=result["best_loss"],
            effective_batch_size=result["effective_batch_size"],
            actual_batch_size_per_gpu=result["actual_batch_size_per_gpu"],
            accumulation_steps=result["accumulation_steps"],
            adjusted_learning_rate=result["adjusted_learning_rate"],
            accumulation_stats=result["accumulation_stats"],
            gpu_stats=result["gpu_stats"],
            training_history=result["training_history"]
        )
    except Exception as e:
        logger.error(f"Training with accumulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/training/large-batch", response_model=TrainingResponse)
async def train_large_batch(
    request: TrainingRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Train model with automatic large batch size optimization."""
    try:
        # Prepare dataset
        dataset = await finetuning_service.prepare_dataset(
            texts=request.dataset_config.get("texts", []),
            model_name=request.model_name,
            max_length=request.dataset_config.get("max_length", 512),
            batch_size=request.dataset_config.get("batch_size", 32)
        )
        
        # Train model with large batch optimization
        result = await finetuning_service.finetune_model_large_batch(
            model_name=request.model_name,
            dataset=dataset,
            training_config=request.training_config,
            user_id=request.user_id,
            target_batch_size=request.target_effective_batch_size,
            max_memory_usage=0.9
        )
        
        return TrainingResponse(
            success=True,
            training_id=f"large_batch_{int(datetime.now().timestamp())}",
            model_path=result["model_path"],
            training_time=result["training_time"],
            best_loss=result["best_loss"],
            effective_batch_size=result["effective_batch_size"],
            actual_batch_size_per_gpu=result["actual_batch_size_per_gpu"],
            accumulation_steps=result["accumulation_steps"],
            adjusted_learning_rate=result["adjusted_learning_rate"],
            accumulation_stats=result["accumulation_stats"],
            gpu_stats=result["gpu_stats"],
            training_history=result["training_history"]
        )
    except Exception as e:
        logger.error(f"Large batch training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Batch size calculation endpoints
@router.post("/calculate-batch-size", response_model=BatchSizeCalculationResponse)
async def calculate_optimal_batch_size(
    request: BatchSizeCalculationRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Calculate optimal batch size and accumulation steps."""
    try:
        result = await finetuning_service.calculate_optimal_batch_size(
            model_name=request.model_name,
            target_effective_batch_size=request.target_effective_batch_size,
            gpu_ids=request.gpu_ids
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return BatchSizeCalculationResponse(
            success=True,
            target_effective_batch_size=result["target_effective_batch_size"],
            actual_batch_size_per_gpu=result["actual_batch_size_per_gpu"],
            total_gpus=result["total_gpus"],
            accumulation_steps=result["accumulation_steps"],
            effective_batch_size=result["effective_batch_size"],
            gpu_info=result["gpu_info"]
        )
    except Exception as e:
        logger.error(f"Batch size calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

@router.get("/calculate-batch-size")
async def calculate_batch_size_simple(
    model_name: str = Query(..., description="Model name"),
    target_batch_size: int = Query(32, description="Target effective batch size"),
    gpu_ids: Optional[List[int]] = Query(None, description="GPU IDs to use")
):
    """Simple batch size calculation endpoint."""
    try:
        finetuning_service = OptimizedFineTuningService()
        result = await finetuning_service.calculate_optimal_batch_size(
            model_name=model_name,
            target_effective_batch_size=target_batch_size,
            gpu_ids=gpu_ids
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "model_name": model_name,
            "target_batch_size": target_batch_size,
            "recommendations": {
                "batch_size_per_gpu": result["actual_batch_size_per_gpu"],
                "accumulation_steps": result["accumulation_steps"],
                "effective_batch_size": result["effective_batch_size"],
                "total_gpus": result["total_gpus"]
            },
            "gpu_info": result["gpu_info"]
        }
    except Exception as e:
        logger.error(f"Simple batch size calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

# Statistics and monitoring endpoints
@router.get("/stats/{training_id}", response_model=AccumulationStats)
async def get_accumulation_stats(
    training_id: str,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Get gradient accumulation statistics for a training session."""
    try:
        result = await finetuning_service.get_accumulation_stats(training_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        stats = result["stats"]
        
        return AccumulationStats(
            training_id=training_id,
            accumulation_steps=stats["accumulation_steps"],
            current_step=stats["current_step"],
            total_loss=stats["total_loss"],
            total_samples=stats["total_samples"],
            avg_gradient_norm=stats["avg_gradient_norm"],
            avg_memory_usage_gb=stats["avg_memory_usage_gb"],
            avg_accumulation_time=stats["avg_accumulation_time"],
            gradient_norms=stats["gradient_norms"],
            memory_usage=stats["memory_usage"],
            accumulation_times=stats["accumulation_times"]
        )
    except Exception as e:
        logger.error(f"Failed to get accumulation stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/stats")
async def get_all_accumulation_stats(
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Get all gradient accumulation statistics."""
    try:
        # This would typically query a database for all training sessions
        # For now, return mock data
        stats = [
            {
                "training_id": f"accumulation_{i}",
                "model_name": "gpt2",
                "accumulation_steps": 4,
                "effective_batch_size": 32,
                "total_loss": 0.1 + i * 0.01,
                "avg_gradient_norm": 1.5 + i * 0.1,
                "avg_memory_usage_gb": 8.5 + i * 0.2,
                "timestamp": datetime.now().isoformat()
            }
            for i in range(5)
        ]
        
        return {
            "total_sessions": len(stats),
            "accumulation_stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get all accumulation stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Configuration management endpoints
@router.put("/config/{training_id}")
async def update_accumulation_config(
    training_id: str,
    config: GradientAccumulationRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Update gradient accumulation configuration."""
    try:
        result = await finetuning_service.update_accumulation_config(
            training_id=training_id,
            config=config.dict()
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "training_id": training_id,
            "message": "Configuration updated successfully",
            "config": config.dict()
        }
    except Exception as e:
        logger.error(f"Failed to update accumulation config: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

# Optimization endpoints
@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_accumulation(
    request: OptimizationRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Optimize gradient accumulation settings."""
    try:
        if request.action == "calculate":
            if not request.model_name or not request.target_batch_size:
                raise HTTPException(status_code=400, detail="Model name and target batch size required")
            
            result = await finetuning_service.calculate_optimal_batch_size(
                model_name=request.model_name,
                target_effective_batch_size=request.target_batch_size,
                gpu_ids=request.gpu_ids
            )
            
            details = {
                "recommendations": result,
                "optimization_type": "batch_size_calculation"
            }
            
        elif request.action == "optimize":
            # Perform optimization based on current GPU state
            gpu_monitor = GPUMonitor(GPUConfig())
            gpu_info = gpu_monitor.get_gpu_info()
            available_gpus = gpu_monitor.get_available_gpus()
            
            details = {
                "available_gpus": available_gpus,
                "gpu_info": gpu_info,
                "optimization_type": "gpu_state_optimization"
            }
            
        elif request.action == "monitor":
            # Monitor current accumulation performance
            details = {
                "monitoring_active": True,
                "timestamp": datetime.now().isoformat(),
                "optimization_type": "performance_monitoring"
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        return OptimizationResponse(
            success=True,
            action=request.action,
            details=details
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

# Utility endpoints
@router.post("/calculate-effective-batch-size")
async def calculate_effective_batch_size_endpoint(
    actual_batch_size: int = Query(..., description="Actual batch size"),
    accumulation_steps: int = Query(..., description="Accumulation steps")
):
    """Calculate effective batch size."""
    try:
        effective_batch_size = calculate_effective_batch_size(actual_batch_size, accumulation_steps)
        
        return {
            "actual_batch_size": actual_batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "calculation": f"{actual_batch_size} * {accumulation_steps} = {effective_batch_size}"
        }
    except Exception as e:
        logger.error(f"Effective batch size calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

@router.post("/calculate-accumulation-steps")
async def calculate_accumulation_steps_endpoint(
    target_batch_size: int = Query(..., description="Target batch size"),
    actual_batch_size: int = Query(..., description="Actual batch size")
):
    """Calculate required accumulation steps."""
    try:
        accumulation_steps = calculate_accumulation_steps(target_batch_size, actual_batch_size)
        
        return {
            "target_batch_size": target_batch_size,
            "actual_batch_size": actual_batch_size,
            "accumulation_steps": accumulation_steps,
            "calculation": f"ceil({target_batch_size} / {actual_batch_size}) = {accumulation_steps}"
        }
    except Exception as e:
        logger.error(f"Accumulation steps calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

@router.post("/adjust-learning-rate")
async def adjust_learning_rate_endpoint(
    base_lr: float = Query(..., description="Base learning rate"),
    accumulation_steps: int = Query(..., description="Accumulation steps")
):
    """Adjust learning rate for gradient accumulation."""
    try:
        adjusted_lr = adjust_learning_rate(base_lr, accumulation_steps)
        
        return {
            "base_learning_rate": base_lr,
            "accumulation_steps": accumulation_steps,
            "adjusted_learning_rate": adjusted_lr,
            "calculation": f"{base_lr} / {accumulation_steps} = {adjusted_lr}"
        }
    except Exception as e:
        logger.error(f"Learning rate adjustment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Adjustment failed: {str(e)}")

# Performance monitoring endpoints
@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get gradient accumulation performance metrics."""
    try:
        # Get GPU information
        gpu_monitor = GPUMonitor(GPUConfig())
        gpu_info = gpu_monitor.get_gpu_info()
        available_gpus = gpu_monitor.get_available_gpus()
        
        # Calculate performance metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "gpu_count": len(gpu_info),
            "available_gpus": len(available_gpus),
            "gpu_utilization": {},
            "memory_utilization": {},
            "accumulation_efficiency": {}
        }
        
        for gpu_id, stats in gpu_info.items():
            metrics["gpu_utilization"][gpu_id] = stats.get("gpu_utilization", 0)
            metrics["memory_utilization"][gpu_id] = stats.get("memory_utilization", 0)
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/performance/recommendations")
async def get_performance_recommendations(
    model_size: str = Query("medium", description="Model size: small, medium, large, xlarge"),
    target_batch_size: int = Query(32, description="Target effective batch size")
):
    """Get performance recommendations for gradient accumulation."""
    try:
        # Performance recommendations based on model size and target batch size
        recommendations = {
            "small": {
                "recommended_accumulation_steps": 2,
                "max_batch_size_per_gpu": 8,
                "memory_efficiency": "high"
            },
            "medium": {
                "recommended_accumulation_steps": 4,
                "max_batch_size_per_gpu": 4,
                "memory_efficiency": "medium"
            },
            "large": {
                "recommended_accumulation_steps": 8,
                "max_batch_size_per_gpu": 2,
                "memory_efficiency": "low"
            },
            "xlarge": {
                "recommended_accumulation_steps": 16,
                "max_batch_size_per_gpu": 1,
                "memory_efficiency": "very_low"
            }
        }
        
        rec = recommendations.get(model_size, recommendations["medium"])
        
        # Calculate optimal settings
        optimal_batch_size = rec["max_batch_size_per_gpu"]
        required_steps = calculate_accumulation_steps(target_batch_size, optimal_batch_size)
        effective_batch_size = calculate_effective_batch_size(optimal_batch_size, required_steps)
        
        return {
            "model_size": model_size,
            "target_batch_size": target_batch_size,
            "recommendations": rec,
            "optimal_settings": {
                "batch_size_per_gpu": optimal_batch_size,
                "accumulation_steps": required_steps,
                "effective_batch_size": effective_batch_size,
                "learning_rate_scale": 1.0 / required_steps
            }
        }
    except Exception as e:
        logger.error(f"Failed to get performance recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

# Cleanup endpoints
@router.delete("/cleanup/{training_id}")
async def cleanup_accumulation(
    training_id: str,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Cleanup gradient accumulation resources."""
    try:
        finetuning_service.gradient_accumulation_api.cleanup(training_id)
        
        return {
            "success": True,
            "training_id": training_id,
            "message": "Accumulation resources cleaned up successfully"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup accumulation: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.delete("/cleanup")
async def cleanup_all_accumulation(
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Cleanup all gradient accumulation resources."""
    try:
        # Cleanup all accumulators
        for training_id in list(finetuning_service.gradient_accumulation_api.accumulators.keys()):
            finetuning_service.gradient_accumulation_api.cleanup(training_id)
        
        return {
            "success": True,
            "message": "All accumulation resources cleaned up successfully",
            "cleaned_sessions": len(finetuning_service.gradient_accumulation_api.accumulators)
        }
    except Exception as e:
        logger.error(f"Failed to cleanup all accumulation: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}") 