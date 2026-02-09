"""
Deep Learning API
=================

API endpoints para modelos de deep learning.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

from ..core.deep_learning_models import (
    get_dl_model_manager,
    ModelType,
    ModelConfig
)
from ..core.model_training import (
    get_model_trainer,
    TrainingStrategy,
    TrainingConfig
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dl", tags=["Deep Learning"])


# Request/Response Models
class CreateModelRequest(BaseModel):
    """Request para crear modelo."""
    model_type: str = Field(..., description="Tipo de modelo")
    input_size: int = Field(..., description="Tamaño de entrada")
    output_size: int = Field(..., description="Tamaño de salida")
    hidden_sizes: List[int] = Field(default=[128, 64, 32], description="Tamaños de capas ocultas")
    activation: str = Field(default="relu", description="Función de activación")
    dropout: float = Field(default=0.1, description="Tasa de dropout")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size")
    num_epochs: int = Field(default=100, description="Número de épocas")
    use_gpu: bool = Field(default=True, description="Usar GPU")
    mixed_precision: bool = Field(default=False, description="Mixed precision")


class PredictRequest(BaseModel):
    """Request para predicción."""
    model_id: str = Field(..., description="ID del modelo")
    inputs: List[List[float]] = Field(..., description="Datos de entrada")


class TrainingRequest(BaseModel):
    """Request para entrenamiento."""
    model_id: str = Field(..., description="ID del modelo")
    train_inputs: List[List[float]] = Field(..., description="Datos de entrenamiento")
    train_targets: List[List[float]] = Field(..., description="Objetivos de entrenamiento")
    val_inputs: Optional[List[List[float]]] = Field(None, description="Datos de validación")
    val_targets: Optional[List[List[float]]] = Field(None, description="Objetivos de validación")
    strategy: str = Field(default="standard", description="Estrategia de entrenamiento")
    batch_size: int = Field(default=32, description="Batch size")
    num_epochs: int = Field(default=100, description="Número de épocas")
    learning_rate: float = Field(default=0.001, description="Learning rate")


# Endpoints
@router.post("/models", response_model=Dict[str, Any])
async def create_model(request: CreateModelRequest):
    """Crear nuevo modelo."""
    try:
        manager = get_dl_model_manager()
        
        model_type = ModelType(request.model_type)
        
        config = ModelConfig(
            model_id="",  # Se genera en create_model
            model_type=model_type,
            input_size=request.input_size,
            output_size=request.output_size,
            hidden_sizes=request.hidden_sizes,
            activation=request.activation,
            dropout=request.dropout,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            use_gpu=request.use_gpu,
            mixed_precision=request.mixed_precision
        )
        
        model_id = manager.create_model(
            model_type,
            request.input_size,
            request.output_size,
            config
        )
        
        return {
            "model_id": model_id,
            "status": "created",
            "config": {
                "model_type": request.model_type,
                "input_size": request.input_size,
                "output_size": request.output_size
            }
        }
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=Dict[str, Any])
async def list_models():
    """Listar modelos."""
    try:
        manager = get_dl_model_manager()
        stats = manager.get_statistics()
        
        models = []
        for model_id, config in manager.configs.items():
            models.append({
                "model_id": model_id,
                "model_type": config.model_type.value,
                "input_size": config.input_size,
                "output_size": config.output_size
            })
        
        return {
            "models": models,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model(model_id: str):
    """Obtener información de modelo."""
    try:
        manager = get_dl_model_manager()
        
        if model_id not in manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        config = manager.configs[model_id]
        
        return {
            "model_id": model_id,
            "config": {
                "model_type": config.model_type.value,
                "input_size": config.input_size,
                "output_size": config.output_size,
                "hidden_sizes": config.hidden_sizes,
                "activation": config.activation,
                "dropout": config.dropout,
                "learning_rate": config.learning_rate
            },
            "statistics": {
                "checkpoints": len(manager.checkpoints.get(model_id, [])),
                "training_metrics": len(manager.training_metrics.get(model_id, []))
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/predict", response_model=Dict[str, Any])
async def predict(request: PredictRequest):
    """Realizar predicción."""
    try:
        manager = get_dl_model_manager()
        
        if request.model_id not in manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        inputs = np.array(request.inputs)
        predictions = manager.predict(request.model_id, inputs)
        
        return {
            "model_id": request.model_id,
            "predictions": predictions.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/train", response_model=Dict[str, Any])
async def train_model(
    model_id: str,
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Entrenar modelo."""
    try:
        manager = get_dl_model_manager()
        trainer = get_model_trainer()
        
        if model_id not in manager.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Convertir a numpy
        train_inputs = np.array(request.train_inputs)
        train_targets = np.array(request.train_targets)
        
        val_inputs = None
        val_targets = None
        if request.val_inputs and request.val_targets:
            val_inputs = np.array(request.val_inputs)
            val_targets = np.array(request.val_targets)
        
        # Configuración
        strategy = TrainingStrategy(request.strategy)
        config = TrainingConfig(
            training_id="",  # Se genera en start_training
            model_id=model_id,
            strategy=strategy,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate
        )
        
        # Crear DataLoaders
        from torch.utils.data import DataLoader
        from ..core.deep_learning_models import RobotDataset
        
        train_dataset = RobotDataset(train_inputs, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=request.batch_size, shuffle=True)
        
        val_loader = None
        if val_inputs is not None:
            val_dataset = RobotDataset(val_inputs, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=request.batch_size, shuffle=False)
        
        # Iniciar entrenamiento
        training_id = trainer.start_training(
            model_id,
            train_loader,
            val_loader,
            config
        )
        
        return {
            "training_id": training_id,
            "model_id": model_id,
            "status": "started"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trainings/{training_id}", response_model=Dict[str, Any])
async def get_training_progress(training_id: str):
    """Obtener progreso de entrenamiento."""
    try:
        trainer = get_model_trainer()
        
        progress = trainer.get_progress(training_id)
        result = trainer.get_result(training_id)
        
        if not progress and not result:
            raise HTTPException(status_code=404, detail="Training not found")
        
        return {
            "training_id": training_id,
            "progress": [
                {
                    "epoch": p.epoch,
                    "total_epochs": p.total_epochs,
                    "train_loss": p.train_loss,
                    "val_loss": p.val_loss,
                    "learning_rate": p.learning_rate,
                    "elapsed_time": p.elapsed_time,
                    "status": p.status
                }
                for p in progress
            ],
            "result": {
                "final_train_loss": result.final_train_loss,
                "final_val_loss": result.final_val_loss,
                "best_val_loss": result.best_val_loss,
                "completed_epochs": result.completed_epochs,
                "training_time": result.training_time,
                "status": result.status
            } if result else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_statistics():
    """Obtener estadísticas."""
    try:
        manager = get_dl_model_manager()
        trainer = get_model_trainer()
        
        return {
            "models": manager.get_statistics(),
            "training": trainer.get_statistics()
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))




