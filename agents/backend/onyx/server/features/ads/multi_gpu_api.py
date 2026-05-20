from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.multi_gpu_training import (
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService
from onyx.server.features.ads.optimized_config import settings
            import torch
from typing import Any, List, Dict, Optional
import logging
"""
Multi-GPU Training API for Onyx Ads Backend

This module provides REST API endpoints for:
- Multi-GPU training management
- GPU configuration and monitoring
- DataParallel and DistributedDataParallel training
- GPU resource management
- Training performance monitoring
"""

    MultiGPUTrainingManager,
    GPUConfig,
    DataParallelTrainer,
    DistributedDataParallelTrainer,
    GPUMonitor
)

logger = setup_logger()

router = APIRouter(prefix="/multigpu", tags=["Multi-GPU Training"])

# Pydantic models for API requests/responses
class GPUConfigRequest(BaseModel):
    """GPU configuration request model."""
    use_multi_gpu: bool = True
    gpu_ids: List[int] = []
    distributed_training: bool = False
    backend: str = "nccl"
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 1
    sync_batch_norm: bool = True
    memory_fraction: float = 0.9
    mixed_precision: bool = True
    log_gpu_memory: bool = True
    log_gpu_utilization: bool = True

class GPUConfigResponse(BaseModel):
    """GPU configuration response model."""
    success: bool
    config: Dict[str, Any]
    available_gpus: List[int]
    gpu_info: Dict[str, Any]

class TrainingConfigRequest(BaseModel):
    """Training configuration request model."""
    model_name: str
    training_config: Dict[str, Any]
    user_id: int
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    gpu_config: Optional[GPUConfigRequest] = None

class TrainingConfigResponse(BaseModel):
    """Training configuration response model."""
    success: bool
    training_id: str
    config: Dict[str, Any]
    gpu_stats: Dict[str, Any]

class TrainingRequest(BaseModel):
    """Training request model."""
    model_name: str
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    user_id: int
    training_type: str = Field(..., description="Type: dataparallel, distributed, or auto")
    gpu_config: Optional[GPUConfigRequest] = None

class TrainingResponse(BaseModel):
    """Training response model."""
    success: bool
    training_id: str
    model_path: str
    training_time: float
    best_loss: float
    gpu_stats: Dict[str, Any]
    training_history: List[Dict[str, Any]]

class GPUStats(BaseModel):
    """GPU statistics response model."""
    gpu_info: Dict[str, Any]
    available_gpus: List[int]
    total_gpus: int
    gpu_utilization: Dict[str, float]
    memory_utilization: Dict[str, float]
    temperature: Dict[str, float]

class ResourceManagementRequest(BaseModel):
    """Resource management request model."""
    action: str = Field(..., description="Action: cleanup, monitor, optimize")
    gpu_ids: Optional[List[int]] = None

class ResourceManagementResponse(BaseModel):
    """Resource management response model."""
    success: bool
    action: str
    details: Dict[str, Any]

# Dependency for multi-GPU training manager
async def get_multi_gpu_manager() -> MultiGPUTrainingManager:
    """Get multi-GPU training manager instance."""
    return MultiGPUTrainingManager()

# Dependency for fine-tuning service
async def get_finetuning_service() -> OptimizedFineTuningService:
    """Get fine-tuning service instance."""
    return OptimizedFineTuningService()

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for multi-GPU training system."""
    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        # Get GPU info
        gpu_monitor = GPUMonitor(GPUConfig())
        gpu_info = gpu_monitor.get_gpu_info()
        available_gpus = gpu_monitor.get_available_gpus()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "available_gpus": available_gpus,
            "gpu_info": gpu_info
        }
    except Exception as e:
        logger.error(f"Multi-GPU health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# GPU configuration endpoints
@router.post("/config", response_model=GPUConfigResponse)
async def configure_gpu(
    request: GPUConfigRequest,
    manager: MultiGPUTrainingManager = Depends(get_multi_gpu_manager)
):
    """Configure GPU settings for multi-GPU training."""
    try:
        # Create GPU config
        gpu_config = GPUConfig(
            use_multi_gpu=request.use_multi_gpu,
            gpu_ids=request.gpu_ids,
            distributed_training=request.distributed_training,
            backend=request.backend,
            batch_size_per_gpu=request.batch_size_per_gpu,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            sync_batch_norm=request.sync_batch_norm,
            memory_fraction=request.memory_fraction,
            mixed_precision=request.mixed_precision,
            log_gpu_memory=request.log_gpu_memory,
            log_gpu_utilization=request.log_gpu_utilization
        )
        
        # Detect GPU configuration
        detected_config = manager.detect_gpu_configuration()
        
        # Get GPU info
        gpu_monitor = GPUMonitor(gpu_config)
        gpu_info = gpu_monitor.get_gpu_info()
        available_gpus = gpu_monitor.get_available_gpus()
        
        return GPUConfigResponse(
            success=True,
            config=gpu_config.__dict__,
            available_gpus=available_gpus,
            gpu_info=gpu_info
        )
    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"GPU configuration failed: {str(e)}")

@router.get("/config")
async def get_gpu_config(
    manager: MultiGPUTrainingManager = Depends(get_multi_gpu_manager)
):
    """Get current GPU configuration."""
    try:
        config = manager.detect_gpu_configuration()
        gpu_stats = manager.get_gpu_stats()
        
        return {
            "config": config.__dict__,
            "gpu_stats": gpu_stats
        }
    except Exception as e:
        logger.error(f"Failed to get GPU config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU config: {str(e)}")

# GPU statistics endpoints
@router.get("/stats", response_model=GPUStats)
async def get_gpu_stats(
    manager: MultiGPUTrainingManager = Depends(get_multi_gpu_manager)
):
    """Get comprehensive GPU statistics."""
    try:
        gpu_stats = manager.get_gpu_stats()
        gpu_monitor = GPUMonitor(GPUConfig())
        
        gpu_info = gpu_monitor.get_gpu_info()
        available_gpus = gpu_monitor.get_available_gpus()
        
        # Extract utilization metrics
        gpu_utilization = {}
        memory_utilization = {}
        temperature = {}
        
        for gpu_id, stats in gpu_info.items():
            gpu_utilization[gpu_id] = stats['gpu_utilization']
            memory_utilization[gpu_id] = stats['memory_utilization']
            temperature[gpu_id] = stats['temperature']
        
        return GPUStats(
            gpu_info=gpu_info,
            available_gpus=available_gpus,
            total_gpus=len(gpu_info),
            gpu_utilization=gpu_utilization,
            memory_utilization=memory_utilization,
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"Failed to get GPU stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU stats: {str(e)}")

@router.get("/gpu/{gpu_id}/stats")
async def get_specific_gpu_stats(
    gpu_id: int,
    manager: MultiGPUTrainingManager = Depends(get_multi_gpu_manager)
):
    """Get statistics for a specific GPU."""
    try:
        gpu_monitor = GPUMonitor(GPUConfig())
        gpu_stats = gpu_monitor.monitor_gpu_usage(gpu_id)
        
        if not gpu_stats:
            raise HTTPException(status_code=404, detail=f"GPU {gpu_id} not found")
        
        return {
            "gpu_id": gpu_id,
            "stats": gpu_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get GPU {gpu_id} stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU stats: {str(e)}")

# Training configuration endpoints
@router.post("/training/config", response_model=TrainingConfigResponse)
async def configure_training(
    request: TrainingConfigRequest,
    manager: MultiGPUTrainingManager = Depends(get_multi_gpu_manager)
):
    """Configure training for multi-GPU."""
    try:
        # Setup multi-GPU training
        if request.gpu_config:
            gpu_config = GPUConfig(**request.gpu_config.dict())
            manager = MultiGPUTrainingManager(gpu_config)
        
        # Setup trainer
        trainer = manager.setup_trainer(
            distributed=request.distributed,
            world_size=request.world_size,
            rank=request.rank
        )
        
        # Generate training ID
        training_id = f"training_{int(datetime.now().timestamp())}"
        
        # Get GPU stats
        gpu_stats = manager.get_gpu_stats()
        
        return TrainingConfigResponse(
            success=True,
            training_id=training_id,
            config={
                "model_name": request.model_name,
                "training_config": request.training_config,
                "user_id": request.user_id,
                "distributed": request.distributed,
                "world_size": request.world_size,
                "rank": request.rank
            },
            gpu_stats=gpu_stats
        )
    except Exception as e:
        logger.error(f"Training configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training configuration failed: {str(e)}")

# Training endpoints
@router.post("/training/dataparallel", response_model=TrainingResponse)
async def train_dataparallel(
    request: TrainingRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Train model using DataParallel for single-node multi-GPU."""
    try:
        # Prepare dataset
        dataset = await finetuning_service.prepare_dataset(
            texts=request.dataset_config.get("texts", []),
            model_name=request.model_name,
            max_length=request.dataset_config.get("max_length", 512),
            batch_size=request.dataset_config.get("batch_size", 32)
        )
        
        # Train model
        result = await finetuning_service.finetune_model_dataparallel(
            model_name=request.model_name,
            dataset=dataset,
            training_config=request.training_config,
            user_id=request.user_id
        )
        
        return TrainingResponse(
            success=True,
            training_id=f"dataparallel_{int(datetime.now().timestamp())}",
            model_path=result["model_path"],
            training_time=result["training_time"],
            best_loss=result["best_loss"],
            gpu_stats=result["gpu_stats"],
            training_history=result["training_history"]
        )
    except Exception as e:
        logger.error(f"DataParallel training failed: {e}")
        raise HTTPException(status_code=500, detail=f"DataParallel training failed: {str(e)}")

@router.post("/training/distributed", response_model=TrainingResponse)
async def train_distributed(
    request: TrainingRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Train model using DistributedDataParallel for multi-node training."""
    try:
        # Prepare dataset
        dataset = await finetuning_service.prepare_dataset(
            texts=request.dataset_config.get("texts", []),
            model_name=request.model_name,
            max_length=request.dataset_config.get("max_length", 512),
            batch_size=request.dataset_config.get("batch_size", 32)
        )
        
        # Train model
        result = await finetuning_service.finetune_model_distributed(
            model_name=request.model_name,
            dataset=dataset,
            training_config=request.training_config,
            user_id=request.user_id,
            world_size=request.training_config.get("world_size", 4)
        )
        
        return TrainingResponse(
            success=True,
            training_id=f"distributed_{int(datetime.now().timestamp())}",
            model_path=result["model_path"],
            training_time=result["training_time"],
            best_loss=result["best_loss"],
            gpu_stats=result["gpu_stats"],
            training_history=result["training_history"]
        )
    except Exception as e:
        logger.error(f"Distributed training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Distributed training failed: {str(e)}")

@router.post("/training/auto", response_model=TrainingResponse)
async def train_auto(
    request: TrainingRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Automatically choose best training method based on available resources."""
    try:
        # Get GPU stats
        gpu_stats = await finetuning_service.get_gpu_stats()
        available_gpus = gpu_stats.get("available_gpus", [])
        
        # Choose training method
        if len(available_gpus) >= 4:
            # Use distributed training for 4+ GPUs
            training_type = "distributed"
            world_size = min(len(available_gpus), 8)  # Max 8 GPUs
        elif len(available_gpus) >= 2:
            # Use DataParallel for 2-3 GPUs
            training_type = "dataparallel"
            world_size = 1
        else:
            # Single GPU training
            training_type = "single"
            world_size = 1
        
        # Prepare dataset
        dataset = await finetuning_service.prepare_dataset(
            texts=request.dataset_config.get("texts", []),
            model_name=request.model_name,
            max_length=request.dataset_config.get("max_length", 512),
            batch_size=request.dataset_config.get("batch_size", 32)
        )
        
        # Train model
        if training_type == "distributed":
            result = await finetuning_service.finetune_model_distributed(
                model_name=request.model_name,
                dataset=dataset,
                training_config=request.training_config,
                user_id=request.user_id,
                world_size=world_size
            )
        elif training_type == "dataparallel":
            result = await finetuning_service.finetune_model_dataparallel(
                model_name=request.model_name,
                dataset=dataset,
                training_config=request.training_config,
                user_id=request.user_id
            )
        else:
            # Single GPU training
            result = await finetuning_service.finetune_model(
                model_name=request.model_name,
                dataset=dataset,
                training_config=request.training_config,
                user_id=request.user_id
            )
        
        return TrainingResponse(
            success=True,
            training_id=f"auto_{training_type}_{int(datetime.now().timestamp())}",
            model_path=result["model_path"],
            training_time=result["training_time"],
            best_loss=result["best_loss"],
            gpu_stats=result.get("gpu_stats", gpu_stats),
            training_history=result.get("training_history", [])
        )
    except Exception as e:
        logger.error(f"Auto training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto training failed: {str(e)}")

# Resource management endpoints
@router.post("/resources/manage", response_model=ResourceManagementResponse)
async def manage_resources(
    request: ResourceManagementRequest,
    finetuning_service: OptimizedFineTuningService = Depends(get_finetuning_service)
):
    """Manage GPU resources."""
    try:
        if request.action == "cleanup":
            result = await finetuning_service.cleanup_gpu_resources()
            details = {"message": "GPU resources cleaned up successfully"}
        elif request.action == "monitor":
            gpu_stats = await finetuning_service.get_gpu_stats()
            details = {"gpu_stats": gpu_stats}
        elif request.action == "optimize":
            # Optimize GPU memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            details = {"message": "GPU memory optimized"}
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        return ResourceManagementResponse(
            success=True,
            action=request.action,
            details=details
        )
    except Exception as e:
        logger.error(f"Resource management failed: {e}")
        raise HTTPException(status_code=500, detail=f"Resource management failed: {str(e)}")

@router.get("/resources/status")
async def get_resource_status(
    manager: MultiGPUTrainingManager = Depends(get_multi_gpu_manager)
):
    """Get current resource status."""
    try:
        gpu_stats = manager.get_gpu_stats()
        gpu_monitor = GPUMonitor(GPUConfig())
        
        # Calculate resource utilization
        total_memory = 0
        used_memory = 0
        total_gpu_util = 0
        gpu_count = 0
        
        for gpu_id, stats in gpu_stats.get("gpu_info", {}).items():
            total_memory += stats.get("memory_total", 0)
            used_memory += stats.get("memory_used", 0)
            total_gpu_util += stats.get("gpu_utilization", 0)
            gpu_count += 1
        
        avg_gpu_util = total_gpu_util / gpu_count if gpu_count > 0 else 0
        memory_util = (used_memory / total_memory * 100) if total_memory > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "gpu_count": gpu_count,
            "available_gpus": gpu_stats.get("available_gpus", []),
            "average_gpu_utilization": avg_gpu_util,
            "average_memory_utilization": memory_util,
            "gpu_stats": gpu_stats
        }
    except Exception as e:
        logger.error(f"Failed to get resource status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource status: {str(e)}")

# Performance monitoring endpoints
@router.get("/performance/metrics")
async def get_performance_metrics(
    manager: MultiGPUTrainingManager = Depends(get_multi_gpu_manager)
):
    """Get performance metrics for multi-GPU training."""
    try:
        gpu_stats = manager.get_gpu_stats()
        gpu_monitor = GPUMonitor(GPUConfig())
        
        # Calculate performance metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "gpu_count": len(gpu_stats.get("gpu_info", {})),
            "available_gpus": len(gpu_stats.get("available_gpus", [])),
            "gpu_utilization": {},
            "memory_utilization": {},
            "temperature": {},
            "power_draw": {}
        }
        
        for gpu_id, stats in gpu_stats.get("gpu_info", {}).items():
            metrics["gpu_utilization"][gpu_id] = stats.get("gpu_utilization", 0)
            metrics["memory_utilization"][gpu_id] = stats.get("memory_utilization", 0)
            metrics["temperature"][gpu_id] = stats.get("temperature", 0)
            metrics["power_draw"][gpu_id] = stats.get("power_draw", 0)
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

# Training history endpoints
@router.get("/training/history")
async def get_training_history(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of training sessions to return")
):
    """Get training history for a user."""
    try:
        # This would typically query a database for training history
        # For now, return mock data
        history = [
            {
                "training_id": f"training_{i}",
                "user_id": user_id,
                "model_name": "gpt2",
                "training_type": "dataparallel" if i % 2 == 0 else "distributed",
                "start_time": datetime.now().isoformat(),
                "duration": 3600 + i * 300,
                "best_loss": 0.1 + i * 0.01,
                "gpu_count": 4 if i % 2 == 0 else 8
            }
            for i in range(limit)
        ]
        
        return {
            "user_id": user_id,
            "training_history": history,
            "total_sessions": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training history: {str(e)}")

# Utility endpoints
@router.post("/gpu/test")
async def test_gpu_setup(
    gpu_ids: List[int] = Query(..., description="GPU IDs to test")
):
    """Test GPU setup and configuration."""
    try:
        results = {}
        
        for gpu_id in gpu_ids:
            try:
                # Test GPU availability
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    device = torch.device(f"cuda:{gpu_id}")
                    
                    # Test tensor operations
                    x = torch.randn(1000, 1000, device=device)
                    y = torch.randn(1000, 1000, device=device)
                    z = torch.mm(x, y)
                    
                    # Test memory allocation
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_reserved = torch.cuda.memory_reserved(device)
                    
                    results[f"gpu_{gpu_id}"] = {
                        "status": "success",
                        "memory_allocated": memory_allocated,
                        "memory_reserved": memory_reserved,
                        "tensor_operation": "success"
                    }
                    
                    # Cleanup
                    del x, y, z
                    torch.cuda.empty_cache()
                else:
                    results[f"gpu_{gpu_id}"] = {
                        "status": "error",
                        "message": "GPU not available"
                    }
            except Exception as e:
                results[f"gpu_{gpu_id}"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return {
            "test_results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"GPU test failed: {e}")
        raise HTTPException(status_code=500, detail=f"GPU test failed: {str(e)}")

@router.get("/gpu/recommendations")
async def get_gpu_recommendations(
    model_size: str = Query("medium", description="Model size: small, medium, large, xlarge"),
    batch_size: int = Query(8, description="Desired batch size per GPU")
):
    """Get GPU recommendations for training."""
    try:
        # GPU recommendations based on model size and batch size
        recommendations = {
            "small": {
                "min_gpus": 1,
                "recommended_gpus": 2,
                "min_memory_per_gpu": 8,  # GB
                "recommended_memory_per_gpu": 16,
                "batch_size_per_gpu": batch_size
            },
            "medium": {
                "min_gpus": 2,
                "recommended_gpus": 4,
                "min_memory_per_gpu": 16,
                "recommended_memory_per_gpu": 24,
                "batch_size_per_gpu": batch_size
            },
            "large": {
                "min_gpus": 4,
                "recommended_gpus": 8,
                "min_memory_per_gpu": 24,
                "recommended_memory_per_gpu": 32,
                "batch_size_per_gpu": batch_size
            },
            "xlarge": {
                "min_gpus": 8,
                "recommended_gpus": 16,
                "min_memory_per_gpu": 32,
                "recommended_memory_per_gpu": 40,
                "batch_size_per_gpu": batch_size
            }
        }
        
        rec = recommendations.get(model_size, recommendations["medium"])
        
        # Get current GPU stats
        gpu_monitor = GPUMonitor(GPUConfig())
        gpu_info = gpu_monitor.get_gpu_info()
        available_gpus = gpu_monitor.get_available_gpus()
        
        # Check if current setup meets recommendations
        current_gpus = len(available_gpus)
        current_memory = 0
        if gpu_info:
            current_memory = min([stats.get("memory_total", 0) for stats in gpu_info.values()]) / 1024  # GB
        
        meets_recommendations = (
            current_gpus >= rec["min_gpus"] and
            current_memory >= rec["min_memory_per_gpu"]
        )
        
        return {
            "model_size": model_size,
            "batch_size_per_gpu": batch_size,
            "recommendations": rec,
            "current_setup": {
                "gpu_count": current_gpus,
                "min_memory_per_gpu": current_memory,
                "meets_recommendations": meets_recommendations
            },
            "available_gpus": available_gpus,
            "gpu_info": gpu_info
        }
    except Exception as e:
        logger.error(f"Failed to get GPU recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU recommendations: {str(e)}") 