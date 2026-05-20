from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio
import time
from datetime import datetime
from onyx.core.auth import get_current_user
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService
from onyx.server.features.ads.optimized_config import settings
        import torch
from typing import Any, List, Dict, Optional
import logging
"""
API endpoints for fine-tuning with LoRA and P-tuning techniques.
"""


logger = setup_logger()

# Request Models
class FineTuneRequest(BaseModel):
    """Request model for fine-tuning."""
    base_model_name: str = Field("microsoft/DialoGPT-medium", description="Base model to fine-tune")
    epochs: int = Field(3, ge=1, le=10, description="Number of training epochs")
    batch_size: int = Field(4, ge=1, le=16, description="Training batch size")
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-2, description="Learning rate")
    max_length: int = Field(512, ge=256, le=1024, description="Maximum sequence length")
    technique: str = Field("lora", regex="^(lora|p-tuning)$", description="Fine-tuning technique")

class GenerateWithFineTunedRequest(BaseModel):
    """Request model for generation with fine-tuned model."""
    prompt: str = Field(..., min_length=10, max_length=1000, description="Input prompt")
    base_model_name: str = Field("microsoft/DialoGPT-medium", description="Base model name")
    max_length: int = Field(200, ge=50, le=500, description="Maximum generation length")
    temperature: float = Field(0.7, ge=0.1, le=1.0, description="Generation temperature")

class EvaluateModelRequest(BaseModel):
    """Request model for model evaluation."""
    base_model_name: str = Field("microsoft/DialoGPT-medium", description="Base model name")

# Response Models
class FineTuneResponse(BaseModel):
    """Response model for fine-tuning."""
    user_id: int
    technique: str
    base_model_name: str
    status: str
    metrics: Dict[str, Any]
    model_path: Optional[str] = None
    training_time: float
    created_at: datetime

class GenerateResponse(BaseModel):
    """Response model for generation."""
    user_id: int
    prompt: str
    generated_text: str
    model_name: str
    generation_time: float
    cached: bool = False
    created_at: datetime

class EvaluationResponse(BaseModel):
    """Response model for evaluation."""
    user_id: int
    base_model_name: str
    average_loss: float
    total_samples: int
    model_performance: str
    generated_samples: list
    evaluation_time: float
    created_at: datetime

class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    user_id: int
    has_trained_model: bool
    last_training: Optional[Dict[str, Any]] = None
    model_performance: Optional[str] = None
    recommendations: list
    created_at: datetime

# Initialize router and service
router = APIRouter(prefix="/ads/v2/finetuning", tags=["fine-tuning"])
finetuning_service = OptimizedFineTuningService()

# Background task queue for fine-tuning
class FineTuningTaskQueue:
    def __init__(self) -> Any:
        self._tasks = {}
        self._running = False
    
    async def start(self) -> Any:
        """Start the task queue."""
        self._running = True
        logger.info("Fine-tuning task queue started")
    
    async def stop(self) -> Any:
        """Stop the task queue."""
        self._running = False
        logger.info("Fine-tuning task queue stopped")
    
    async def add_task(self, task_id: str, task_func, *args, **kwargs):
        """Add a task to the queue."""
        if task_id in self._tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        self._tasks[task_id] = {
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        # Start task in background
        asyncio.create_task(self._execute_task(task_id, task_func, *args, **kwargs))
    
    async def _execute_task(self, task_id: str, task_func, *args, **kwargs):
        """Execute a task."""
        try:
            self._tasks[task_id]["status"] = "running"
            self._tasks[task_id]["started_at"] = datetime.utcnow()
            
            result = await task_func(*args, **kwargs)
            
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["completed_at"] = datetime.utcnow()
            self._tasks[task_id]["result"] = result
            
            logger.info(f"Fine-tuning task {task_id} completed successfully")
            
        except Exception as e:
            self._tasks[task_id]["status"] = "failed"
            self._tasks[task_id]["completed_at"] = datetime.utcnow()
            self._tasks[task_id]["error"] = str(e)
            
            logger.error(f"Fine-tuning task {task_id} failed: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        return self._tasks.get(task_id)

task_queue = FineTuningTaskQueue()

# Startup and shutdown events
@router.on_event("startup")
async def startup_event():
    """Initialize fine-tuning service."""
    await task_queue.start()
    logger.info("Fine-tuning API started")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup fine-tuning service."""
    await task_queue.stop()
    await finetuning_service.close()
    logger.info("Fine-tuning API stopped")

# API Endpoints
@router.post("/train", response_model=FineTuneResponse)
async def start_fine_tuning(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Start fine-tuning process with LoRA or P-tuning."""
    try:
        user_id = current_user["id"]
        start_time = time.time()
        
        # Check if user has enough training data
        training_data, total_samples = await finetuning_service.prepare_training_data(
            user_id=user_id,
            model_name=request.base_model_name,
            max_samples=100
        )
        
        if total_samples < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient training data. Need at least 10 samples, got {total_samples}"
            )
        
        # Create task ID
        task_id = f"finetune_{user_id}_{int(time.time())}"
        
        # Start fine-tuning in background
        if request.technique == "lora":
            await task_queue.add_task(
                task_id,
                finetuning_service.fine_tune_lora,
                user_id=user_id,
                base_model_name=request.base_model_name,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                max_length=request.max_length
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Fine-tuning technique '{request.technique}' not yet implemented"
            )
        
        training_time = time.time() - start_time
        
        return FineTuneResponse(
            user_id=user_id,
            technique=request.technique,
            base_model_name=request.base_model_name,
            status="started",
            metrics={
                "total_samples": total_samples,
                "task_id": task_id,
                "estimated_duration": "10-30 minutes"
            },
            training_time=training_time,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception("Error starting fine-tuning")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_fine_tuning_status(
    task_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get fine-tuning task status."""
    try:
        task_status = task_queue.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task_id,
            "status": task_status["status"],
            "started_at": task_status["started_at"],
            "completed_at": task_status["completed_at"],
            "result": task_status["result"],
            "error": task_status["error"]
        }
        
    except Exception as e:
        logger.exception("Error getting task status")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=GenerateResponse)
async def generate_with_finetuned_model(
    request: GenerateWithFineTunedRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate ads using fine-tuned model."""
    try:
        user_id = current_user["id"]
        start_time = time.time()
        
        # Check if user has a trained model
        training_status = await finetuning_service.get_training_status(user_id)
        
        if not training_status["has_trained_model"]:
            raise HTTPException(
                status_code=400,
                detail="No fine-tuned model available. Please train a model first."
            )
        
        # Generate with fine-tuned model
        generated_text = await finetuning_service.generate_with_finetuned_model(
            user_id=user_id,
            prompt=request.prompt,
            base_model_name=request.base_model_name,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            user_id=user_id,
            prompt=request.prompt,
            generated_text=generated_text,
            model_name=request.base_model_name,
            generation_time=generation_time,
            cached=False,  # Fine-tuned models don't use generation cache
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception("Error generating with fine-tuned model")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model_performance(
    request: EvaluateModelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Evaluate fine-tuned model performance."""
    try:
        user_id = current_user["id"]
        start_time = time.time()
        
        # Check if user has a trained model
        training_status = await finetuning_service.get_training_status(user_id)
        
        if not training_status["has_trained_model"]:
            raise HTTPException(
                status_code=400,
                detail="No fine-tuned model available for evaluation."
            )
        
        # Evaluate model
        evaluation_metrics = await finetuning_service.evaluate_model_performance(
            user_id=user_id,
            base_model_name=request.base_model_name
        )
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResponse(
            user_id=user_id,
            base_model_name=request.base_model_name,
            average_loss=evaluation_metrics["average_loss"],
            total_samples=evaluation_metrics["total_samples"],
            model_performance=evaluation_metrics["model_performance"],
            generated_samples=evaluation_metrics["generated_samples"],
            evaluation_time=evaluation_time,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception("Error evaluating model")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user's fine-tuning training status."""
    try:
        user_id = current_user["id"]
        
        training_status = await finetuning_service.get_training_status(user_id)
        
        return TrainingStatusResponse(
            user_id=user_id,
            has_trained_model=training_status["has_trained_model"],
            last_training=training_status["last_training"],
            model_performance=training_status["model_performance"],
            recommendations=training_status["recommendations"],
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.exception("Error getting training status")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models")
async def cleanup_old_models(
    days: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Clean up old fine-tuned models."""
    try:
        # Only allow admin users to cleanup models
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Only admin users can cleanup models"
            )
        
        cleanup_stats = await finetuning_service.cleanup_old_models(days=days)
        
        return {
            "message": "Model cleanup completed",
            "cleaned_models": cleanup_stats["cleaned_models"],
            "cutoff_date": cleanup_stats["cutoff_date"]
        }
        
    except Exception as e:
        logger.exception("Error cleaning up models")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for fine-tuning service."""
    try:
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "task_queue_running": task_queue._running,
            "active_tasks": len(task_queue._tasks)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Example usage endpoints
@router.get("/examples")
async def get_fine_tuning_examples():
    """Get examples of fine-tuning configurations."""
    return {
        "examples": [
            {
                "name": "Quick LoRA Training",
                "description": "Fast fine-tuning for small datasets",
                "config": {
                    "base_model_name": "microsoft/DialoGPT-medium",
                    "epochs": 2,
                    "batch_size": 4,
                    "learning_rate": 3e-4,
                    "technique": "lora"
                }
            },
            {
                "name": "High-Quality LoRA Training",
                "description": "Thorough fine-tuning for better results",
                "config": {
                    "base_model_name": "microsoft/DialoGPT-large",
                    "epochs": 5,
                    "batch_size": 2,
                    "learning_rate": 1e-4,
                    "technique": "lora"
                }
            },
            {
                "name": "P-tuning Training",
                "description": "Parameter-efficient fine-tuning",
                "config": {
                    "base_model_name": "microsoft/DialoGPT-medium",
                    "epochs": 3,
                    "batch_size": 4,
                    "learning_rate": 2e-4,
                    "technique": "p-tuning"
                }
            }
        ]
    } 