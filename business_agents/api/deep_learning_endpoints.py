"""
Deep Learning API Endpoints
===========================

API endpoints for deep learning service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.deep_learning_service import (
    DeepLearningService,
    ModelArchitecture,
    TrainingJob,
    ModelInference,
    ModelEvaluation,
    ModelType,
    TrainingStrategy,
    OptimizationAlgorithm
)

logger = logging.getLogger(__name__)

# Create router
deep_learning_router = APIRouter(prefix="/deep-learning", tags=["Deep Learning"])

# Pydantic models for request/response
class ModelArchitectureRequest(BaseModel):
    name: str
    model_type: ModelType
    layers: List[Dict[str, Any]]
    parameters: Dict[str, Any] = {}
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    metadata: Dict[str, Any] = {}

class TrainingJobRequest(BaseModel):
    name: str
    model_id: str
    dataset_id: str
    training_strategy: TrainingStrategy
    optimizer: OptimizationAlgorithm
    hyperparameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class ModelInferenceRequest(BaseModel):
    model_id: str
    input_data: Any
    metadata: Dict[str, Any] = {}

class ModelEvaluationRequest(BaseModel):
    model_id: str
    dataset_id: str
    metadata: Dict[str, Any] = {}

class ModelArchitectureResponse(BaseModel):
    architecture_id: str
    name: str
    model_type: str
    layers: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    total_parameters: int
    created_at: datetime
    metadata: Dict[str, Any]

class TrainingJobResponse(BaseModel):
    job_id: str
    name: str
    model_id: str
    dataset_id: str
    training_strategy: str
    optimizer: str
    hyperparameters: Dict[str, Any]
    status: str
    progress: float
    metrics: Dict[str, float]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ModelInferenceResponse(BaseModel):
    inference_id: str
    model_id: str
    input_data: Any
    output_data: Any
    inference_time: float
    confidence: float
    created_at: datetime
    metadata: Dict[str, Any]

class ModelEvaluationResponse(BaseModel):
    evaluation_id: str
    model_id: str
    dataset_id: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]]
    roc_curve: Optional[Dict[str, Any]]
    precision_recall_curve: Optional[Dict[str, Any]]
    created_at: datetime
    metadata: Dict[str, Any]

class ServiceStatusResponse(BaseModel):
    service_status: str
    total_architectures: int
    total_jobs: int
    total_inferences: int
    total_evaluations: int
    running_jobs: int
    pre_trained_models: int
    training_engines: int
    device: str
    gpu_enabled: bool
    mixed_precision: bool
    distributed_training: bool
    model_serving: bool
    gradio_integration: bool
    tensorboard_logging: bool
    wandb_integration: bool
    timestamp: str

# Dependency to get deep learning service
async def get_deep_learning_service() -> DeepLearningService:
    """Get deep learning service instance."""
    # This would be injected from your dependency injection system
    # For now, we'll create a mock instance
    from ..main import get_deep_learning_service
    return await get_deep_learning_service()

@deep_learning_router.post("/architectures", response_model=Dict[str, str])
async def create_model_architecture(
    request: ModelArchitectureRequest,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Create model architecture."""
    try:
        architecture = ModelArchitecture(
            architecture_id="",
            name=request.name,
            model_type=request.model_type,
            layers=request.layers,
            parameters=request.parameters,
            input_shape=request.input_shape,
            output_shape=request.output_shape,
            total_parameters=0,
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        architecture_id = await deep_learning_service.create_model_architecture(architecture)
        
        return {"architecture_id": architecture_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create model architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/architectures/{architecture_id}", response_model=ModelArchitectureResponse)
async def get_model_architecture(
    architecture_id: str,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Get model architecture."""
    try:
        architecture = await deep_learning_service.get_model_architecture(architecture_id)
        
        if not architecture:
            raise HTTPException(status_code=404, detail="Model architecture not found")
            
        return ModelArchitectureResponse(
            architecture_id=architecture.architecture_id,
            name=architecture.name,
            model_type=architecture.model_type.value,
            layers=architecture.layers,
            parameters=architecture.parameters,
            input_shape=architecture.input_shape,
            output_shape=architecture.output_shape,
            total_parameters=architecture.total_parameters,
            created_at=architecture.created_at,
            metadata=architecture.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/architectures", response_model=List[ModelArchitectureResponse])
async def list_model_architectures(
    model_type: Optional[ModelType] = None,
    limit: int = 100,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """List model architectures."""
    try:
        architectures = await deep_learning_service.list_model_architectures(model_type)
        
        return [
            ModelArchitectureResponse(
                architecture_id=arch.architecture_id,
                name=arch.name,
                model_type=arch.model_type.value,
                layers=arch.layers,
                parameters=arch.parameters,
                input_shape=arch.input_shape,
                output_shape=arch.output_shape,
                total_parameters=arch.total_parameters,
                created_at=arch.created_at,
                metadata=arch.metadata
            )
            for arch in architectures[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list model architectures: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.post("/training", response_model=Dict[str, str])
async def start_training_job(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Start training job."""
    try:
        job = TrainingJob(
            job_id="",
            name=request.name,
            model_id=request.model_id,
            dataset_id=request.dataset_id,
            training_strategy=request.training_strategy,
            optimizer=request.optimizer,
            hyperparameters=request.hyperparameters,
            status="pending",
            progress=0.0,
            metrics={},
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            metadata=request.metadata
        )
        
        job_id = await deep_learning_service.start_training_job(job)
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to start training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/training/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: str,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Get training job."""
    try:
        job = await deep_learning_service.get_training_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
            
        return TrainingJobResponse(
            job_id=job.job_id,
            name=job.name,
            model_id=job.model_id,
            dataset_id=job.dataset_id,
            training_strategy=job.training_strategy.value,
            optimizer=job.optimizer.value,
            hyperparameters=job.hyperparameters,
            status=job.status,
            progress=job.progress,
            metrics=job.metrics,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            metadata=job.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/training", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 100,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """List training jobs."""
    try:
        jobs = await deep_learning_service.list_training_jobs(status)
        
        return [
            TrainingJobResponse(
                job_id=job.job_id,
                name=job.name,
                model_id=job.model_id,
                dataset_id=job.dataset_id,
                training_strategy=job.training_strategy.value,
                optimizer=job.optimizer.value,
                hyperparameters=job.hyperparameters,
                status=job.status,
                progress=job.progress,
                metrics=job.metrics,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                metadata=job.metadata
            )
            for job in jobs[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list training jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.post("/inference", response_model=Dict[str, str])
async def run_model_inference(
    request: ModelInferenceRequest,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Run model inference."""
    try:
        inference_id = await deep_learning_service.run_model_inference(
            request.model_id,
            request.input_data
        )
        
        return {"inference_id": inference_id, "status": "completed"}
        
    except Exception as e:
        logger.error(f"Failed to run model inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/inference/{inference_id}", response_model=ModelInferenceResponse)
async def get_model_inference(
    inference_id: str,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Get model inference."""
    try:
        inference = await deep_learning_service.get_model_inference(inference_id)
        
        if not inference:
            raise HTTPException(status_code=404, detail="Model inference not found")
            
        return ModelInferenceResponse(
            inference_id=inference.inference_id,
            model_id=inference.model_id,
            input_data=inference.input_data,
            output_data=inference.output_data,
            inference_time=inference.inference_time,
            confidence=inference.confidence,
            created_at=inference.created_at,
            metadata=inference.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/inference", response_model=List[ModelInferenceResponse])
async def list_model_inferences(
    model_id: Optional[str] = None,
    limit: int = 100,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """List model inferences."""
    try:
        inferences = await deep_learning_service.list_model_inferences(model_id, limit)
        
        return [
            ModelInferenceResponse(
                inference_id=inference.inference_id,
                model_id=inference.model_id,
                input_data=inference.input_data,
                output_data=inference.output_data,
                inference_time=inference.inference_time,
                confidence=inference.confidence,
                created_at=inference.created_at,
                metadata=inference.metadata
            )
            for inference in inferences
        ]
        
    except Exception as e:
        logger.error(f"Failed to list model inferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.post("/evaluation", response_model=Dict[str, str])
async def evaluate_model(
    request: ModelEvaluationRequest,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Evaluate model."""
    try:
        evaluation_id = await deep_learning_service.evaluate_model(
            request.model_id,
            request.dataset_id
        )
        
        return {"evaluation_id": evaluation_id, "status": "completed"}
        
    except Exception as e:
        logger.error(f"Failed to evaluate model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/evaluation/{evaluation_id}", response_model=ModelEvaluationResponse)
async def get_model_evaluation(
    evaluation_id: str,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Get model evaluation."""
    try:
        evaluation = await deep_learning_service.get_model_evaluation(evaluation_id)
        
        if not evaluation:
            raise HTTPException(status_code=404, detail="Model evaluation not found")
            
        return ModelEvaluationResponse(
            evaluation_id=evaluation.evaluation_id,
            model_id=evaluation.model_id,
            dataset_id=evaluation.dataset_id,
            metrics=evaluation.metrics,
            confusion_matrix=evaluation.confusion_matrix.tolist() if evaluation.confusion_matrix is not None else None,
            roc_curve=evaluation.roc_curve,
            precision_recall_curve=evaluation.precision_recall_curve,
            created_at=evaluation.created_at,
            metadata=evaluation.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/evaluation", response_model=List[ModelEvaluationResponse])
async def list_model_evaluations(
    model_id: Optional[str] = None,
    limit: int = 100,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """List model evaluations."""
    try:
        evaluations = await deep_learning_service.list_model_evaluations(model_id)
        
        return [
            ModelEvaluationResponse(
                evaluation_id=evaluation.evaluation_id,
                model_id=evaluation.model_id,
                dataset_id=evaluation.dataset_id,
                metrics=evaluation.metrics,
                confusion_matrix=evaluation.confusion_matrix.tolist() if evaluation.confusion_matrix is not None else None,
                roc_curve=evaluation.roc_curve,
                precision_recall_curve=evaluation.precision_recall_curve,
                created_at=evaluation.created_at,
                metadata=evaluation.metadata
            )
            for evaluation in evaluations[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list model evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Get deep learning service status."""
    try:
        status = await deep_learning_service.get_service_status()
        
        return ServiceStatusResponse(
            service_status=status["service_status"],
            total_architectures=status["total_architectures"],
            total_jobs=status["total_jobs"],
            total_inferences=status["total_inferences"],
            total_evaluations=status["total_evaluations"],
            running_jobs=status["running_jobs"],
            pre_trained_models=status["pre_trained_models"],
            training_engines=status["training_engines"],
            device=status["device"],
            gpu_enabled=status["gpu_enabled"],
            mixed_precision=status["mixed_precision"],
            distributed_training=status["distributed_training"],
            model_serving=status["model_serving"],
            gradio_integration=status["gradio_integration"],
            tensorboard_logging=status["tensorboard_logging"],
            wandb_integration=status["wandb_integration"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/models", response_model=Dict[str, Any])
async def get_pre_trained_models(
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Get available pre-trained models."""
    try:
        return deep_learning_service.pre_trained_models
        
    except Exception as e:
        logger.error(f"Failed to get pre-trained models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/engines", response_model=Dict[str, Any])
async def get_training_engines(
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Get available training engines."""
    try:
        return deep_learning_service.training_engines
        
    except Exception as e:
        logger.error(f"Failed to get training engines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.get("/model-types", response_model=List[str])
async def get_model_types():
    """Get available model types."""
    return [model_type.value for model_type in ModelType]

@deep_learning_router.get("/training-strategies", response_model=List[str])
async def get_training_strategies():
    """Get available training strategies."""
    return [strategy.value for strategy in TrainingStrategy]

@deep_learning_router.get("/optimizers", response_model=List[str])
async def get_optimization_algorithms():
    """Get available optimization algorithms."""
    return [algorithm.value for algorithm in OptimizationAlgorithm]

@deep_learning_router.delete("/architectures/{architecture_id}")
async def delete_model_architecture(
    architecture_id: str,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Delete model architecture."""
    try:
        if architecture_id not in deep_learning_service.model_architectures:
            raise HTTPException(status_code=404, detail="Model architecture not found")
            
        del deep_learning_service.model_architectures[architecture_id]
        
        return {"status": "deleted", "architecture_id": architecture_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@deep_learning_router.delete("/training/{job_id}")
async def delete_training_job(
    job_id: str,
    deep_learning_service: DeepLearningService = Depends(get_deep_learning_service)
):
    """Delete training job."""
    try:
        if job_id not in deep_learning_service.training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")
            
        del deep_learning_service.training_jobs[job_id]
        
        return {"status": "deleted", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
























