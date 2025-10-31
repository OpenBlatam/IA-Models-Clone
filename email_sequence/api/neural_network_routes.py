"""
Neural Network Routes for Email Sequence System

This module provides API endpoints for advanced neural network capabilities
including deep learning models and neural architecture search.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from .schemas import ErrorResponse
from ..core.neural_network_engine import (
    neural_network_engine,
    ModelType,
    TaskType,
    TrainingConfig
)
from ..core.dependencies import get_current_user
from ..core.exceptions import NeuralNetworkError

logger = logging.getLogger(__name__)

# Neural Network router
neural_network_router = APIRouter(
    prefix="/api/v1/neural-network",
    tags=["Neural Network"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


@neural_network_router.post("/models")
async def create_neural_network_model(
    model_id: str,
    model_type: ModelType,
    task_type: TaskType,
    architecture: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None
):
    """
    Create a new neural network model.
    
    Args:
        model_id: Unique model identifier
        model_type: Type of neural network
        task_type: Type of ML task
        architecture: Model architecture configuration
        parameters: Model parameters
        
    Returns:
        Model creation result
    """
    try:
        model = await neural_network_engine.create_model(
            model_id=model_id,
            model_type=model_type,
            task_type=task_type,
            architecture=architecture,
            parameters=parameters
        )
        
        return {
            "status": "success",
            "model_id": model.model_id,
            "model_type": model.model_type.value,
            "task_type": model.task_type.value,
            "architecture": model.architecture,
            "created_at": model.created_at.isoformat(),
            "message": "Neural network model created successfully"
        }
        
    except NeuralNetworkError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating neural network model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.post("/models/{model_id}/train")
async def train_neural_network_model(
    model_id: str,
    training_data: UploadFile = File(...),
    target_column: str = "target",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2
):
    """
    Train a neural network model.
    
    Args:
        model_id: Model ID to train
        training_data: Training dataset file
        target_column: Target column name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        validation_split: Validation split ratio
        
    Returns:
        Training results
    """
    try:
        # Read training data
        content = await training_data.read()
        
        # Parse CSV data
        if training_data.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files are supported")
        
        # Create training configuration
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split
        )
        
        # Train model
        results = await neural_network_engine.train_model(
            model_id=model_id,
            training_data=df,
            target_column=target_column,
            config=config
        )
        
        return results
        
    except NeuralNetworkError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error training neural network model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.post("/models/{model_id}/predict")
async def predict_with_neural_network(
    model_id: str,
    prediction_data: List[Dict[str, Any]]
):
    """
    Make predictions using a trained neural network model.
    
    Args:
        model_id: Model ID to use for prediction
        prediction_data: Input data for prediction
        
    Returns:
        Prediction results
    """
    try:
        # Convert prediction data to DataFrame
        df = pd.DataFrame(prediction_data)
        
        # Make predictions
        results = await neural_network_engine.predict(
            model_id=model_id,
            data=df
        )
        
        return results
        
    except NeuralNetworkError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error making neural network predictions: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.post("/models/{model_id}/optimize")
async def optimize_neural_network_hyperparameters(
    model_id: str,
    training_data: UploadFile = File(...),
    target_column: str = "target",
    optimization_config: Dict[str, Any] = None
):
    """
    Optimize hyperparameters using neural architecture search.
    
    Args:
        model_id: Model ID to optimize
        training_data: Training dataset file
        target_column: Target column name
        optimization_config: Optimization configuration
        
    Returns:
        Optimization results
    """
    try:
        # Read training data
        content = await training_data.read()
        
        # Parse CSV data
        if training_data.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files are supported")
        
        # Default optimization configuration
        if optimization_config is None:
            optimization_config = {
                "search_space": {
                    "learning_rate": [0.001, 0.01, 0.1],
                    "batch_size": [16, 32, 64],
                    "hidden_units": [32, 64, 128]
                },
                "max_trials": 10
            }
        
        # Optimize hyperparameters
        results = await neural_network_engine.optimize_hyperparameters(
            model_id=model_id,
            training_data=df,
            target_column=target_column,
            optimization_config=optimization_config
        )
        
        return results
        
    except NeuralNetworkError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error optimizing neural network hyperparameters: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.get("/models/{model_id}/performance")
async def get_neural_network_performance(model_id: str):
    """
    Get neural network model performance metrics.
    
    Args:
        model_id: Model ID
        
    Returns:
        Performance metrics
    """
    try:
        performance = await neural_network_engine.get_model_performance(model_id)
        return performance
        
    except NeuralNetworkError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting neural network performance: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.get("/models")
async def list_neural_network_models():
    """
    List all neural network models.
    
    Returns:
        List of models
    """
    try:
        models = []
        for model_id, model in neural_network_engine.models.items():
            models.append({
                "model_id": model_id,
                "model_type": model.model_type.value,
                "task_type": model.task_type.value,
                "is_trained": model.is_trained,
                "created_at": model.created_at.isoformat(),
                "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                "performance_metrics": model.performance_metrics
            })
        
        return {
            "status": "success",
            "models": models,
            "total_models": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error listing neural network models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.delete("/models/{model_id}")
async def delete_neural_network_model(model_id: str):
    """
    Delete a neural network model.
    
    Args:
        model_id: Model ID to delete
        
    Returns:
        Deletion result
    """
    try:
        if model_id not in neural_network_engine.models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
        
        # Remove model
        del neural_network_engine.models[model_id]
        
        # Remove from cache
        await neural_network_engine.cache_manager.delete(f"neural_model:{model_id}")
        
        return {
            "status": "success",
            "model_id": model_id,
            "message": "Neural network model deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting neural network model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.get("/stats")
async def get_neural_network_stats():
    """
    Get neural network engine statistics.
    
    Returns:
        Engine statistics
    """
    try:
        stats = await neural_network_engine.get_engine_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting neural network stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.post("/models/{model_id}/export")
async def export_neural_network_model(model_id: str):
    """
    Export a neural network model.
    
    Args:
        model_id: Model ID to export
        
    Returns:
        Model export data
    """
    try:
        if model_id not in neural_network_engine.models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
        
        model = neural_network_engine.models[model_id]
        
        # Prepare export data
        export_data = {
            "model_id": model.model_id,
            "model_type": model.model_type.value,
            "task_type": model.task_type.value,
            "architecture": model.architecture,
            "parameters": model.parameters,
            "performance_metrics": model.performance_metrics,
            "training_history": model.training_history,
            "created_at": model.created_at.isoformat(),
            "last_trained": model.last_trained.isoformat() if model.last_trained else None,
            "is_trained": model.is_trained,
            "model_data": model.model_data.decode() if model.model_data else None
        }
        
        return {
            "status": "success",
            "model_id": model_id,
            "export_data": export_data,
            "message": "Model exported successfully"
        }
        
    except Exception as e:
        logger.error(f"Error exporting neural network model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.post("/models/import")
async def import_neural_network_model(
    model_data: Dict[str, Any]
):
    """
    Import a neural network model.
    
    Args:
        model_data: Model data to import
        
    Returns:
        Import result
    """
    try:
        # Validate model data
        required_fields = ["model_id", "model_type", "task_type", "architecture"]
        for field in required_fields:
            if field not in model_data:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Missing required field: {field}")
        
        # Create model from imported data
        model = await neural_network_engine.create_model(
            model_id=model_data["model_id"],
            model_type=ModelType(model_data["model_type"]),
            task_type=TaskType(model_data["task_type"]),
            architecture=model_data["architecture"],
            parameters=model_data.get("parameters", {})
        )
        
        # Update model with imported data
        if "performance_metrics" in model_data:
            model.performance_metrics = model_data["performance_metrics"]
        if "training_history" in model_data:
            model.training_history = model_data["training_history"]
        if "model_data" in model_data and model_data["model_data"]:
            model.model_data = model_data["model_data"].encode()
        if "is_trained" in model_data:
            model.is_trained = model_data["is_trained"]
        
        return {
            "status": "success",
            "model_id": model.model_id,
            "message": "Model imported successfully"
        }
        
    except Exception as e:
        logger.error(f"Error importing neural network model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@neural_network_router.post("/models/{model_id}/clone")
async def clone_neural_network_model(
    model_id: str,
    new_model_id: str
):
    """
    Clone a neural network model.
    
    Args:
        model_id: Source model ID
        new_model_id: New model ID
        
    Returns:
        Clone result
    """
    try:
        if model_id not in neural_network_engine.models:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source model not found")
        
        if new_model_id in neural_network_engine.models:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Target model ID already exists")
        
        source_model = neural_network_engine.models[model_id]
        
        # Create new model with same configuration
        new_model = await neural_network_engine.create_model(
            model_id=new_model_id,
            model_type=source_model.model_type,
            task_type=source_model.task_type,
            architecture=source_model.architecture,
            parameters=source_model.parameters
        )
        
        # Copy training data if available
        if source_model.is_trained and source_model.model_data:
            new_model.model_data = source_model.model_data
            new_model.is_trained = True
            new_model.performance_metrics = source_model.performance_metrics.copy()
            new_model.training_history = source_model.training_history.copy()
        
        return {
            "status": "success",
            "source_model_id": model_id,
            "new_model_id": new_model_id,
            "message": "Model cloned successfully"
        }
        
    except NeuralNetworkError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error cloning neural network model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# Error handlers for neural network routes
@neural_network_router.exception_handler(NeuralNetworkError)
async def neural_network_error_handler(request, exc):
    """Handle neural network errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=f"Neural network error: {exc.message}",
            error_code="NEURAL_NETWORK_ERROR"
        ).dict()
    )






























