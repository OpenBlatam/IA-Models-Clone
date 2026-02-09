"""
Models Router
=============

API endpoints for model management.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
import numpy as np
import torch
import os
import traceback

from ..schemas import (
    CreateModelRequest,
    CreateModelResponse,
    CompileModelRequest,
    TrainRequest,
    TrainResponse,
    EvaluateRequest,
    EvaluateResponse,
    PredictRequest,
    PredictResponse,
    ModelListResponse,
    ModelInfo,
    SaveModelResponse,
    LoadModelResponse
)
from ..services import create_layer_from_config, create_optimizer_from_config, create_loss_from_config, model_store
from ..exceptions import ModelNotFoundError, ModelNotCompiledError
from ..config import settings
from ..utils import (
    serialize_history, validate_array_shape, get_model_summary, 
    serialize_tensor, handle_model_operation_error, parse_iso_date
)
from ..constants import MAX_MODEL_NAME_LENGTH, MAX_SEARCH_RESULTS
from ..metrics import metrics_collector
import truthgpt as tg
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/create", response_model=CreateModelResponse)
async def create_model(request: CreateModelRequest):
    """
    Create a new model.
    
    - **layers**: List of layer configurations
    - **name**: Optional model name
    """
    try:
        if not request.layers:
            raise HTTPException(status_code=400, detail="At least one layer is required")
        
        layers = [create_layer_from_config(layer_config) for layer_config in request.layers]
        model = tg.Sequential(layers, name=request.name)
        model_id = model_store.create(model, name=request.name)
        
        model_data = model_store.get(model_id)
        logger.info(f"Model created successfully: {model_id}")
        metrics_collector.record_model_operation("create", model_id)
        
        return CreateModelResponse(
            model_id=model_id,
            name=model_data["name"],
            status="created",
            message="Model created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        handle_model_operation_error("creating model", error=e)
        raise HTTPException(status_code=400, detail=f"Error creating model: {str(e)}")


@router.post("/{model_id}/compile", response_model=CreateModelResponse)
async def compile_model(model_id: str, request: CompileModelRequest):
    """Compile a model."""
    try:
        model = model_store.get_model(model_id)
        
        optimizer = create_optimizer_from_config(
            request.optimizer,
            request.optimizer_params or {}
        )
        
        loss = create_loss_from_config(
            request.loss,
            request.loss_params or {}
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=request.metrics or []
        )
        
        model_store.mark_compiled(model_id)
        model_store._record_operation(model_id, "compile", {
            "optimizer": request.optimizer,
            "loss": request.loss
        })
        metrics_collector.record_model_operation("compile", model_id)
        
        return CreateModelResponse(
            model_id=model_id,
            name=model_store.get(model_id)["name"],
            status="compiled",
            message="Model compiled successfully"
        )
    except ModelNotFoundError:
        raise
    except Exception as e:
        handle_model_operation_error("compiling model", model_id, e)
        raise HTTPException(status_code=400, detail=f"Error compiling model: {str(e)}")


@router.post("/{model_id}/train", response_model=TrainResponse)
async def train_model(model_id: str, request: TrainRequest):
    """Train a model."""
    try:
        model_store.require_compiled(model_id)
        model = model_store.get_model(model_id)
        
        x_train = validate_array_shape(request.x_train, expected_dim=2)
        if not request.y_train:
            raise HTTPException(status_code=400, detail="y_train cannot be empty")
        y_train = np.array(
            request.y_train,
            dtype=np.int64 if isinstance(request.y_train[0], int) else np.float32
        )
        
        if len(x_train) != len(y_train):
            raise HTTPException(
                status_code=400,
                detail=f"Training data mismatch: x_train has {len(x_train)} samples, "
                       f"y_train has {len(y_train)} samples"
            )
        
        validation_data = None
        if request.validation_data:
            val_y = request.validation_data.get("y", [])
            if not val_y:
                raise HTTPException(status_code=400, detail="Validation y data cannot be empty")
            x_val = validate_array_shape(request.validation_data.get("x", []), expected_dim=2)
            y_val = np.array(
                val_y,
                dtype=np.int64 if isinstance(val_y[0], int) else np.float32
            )
            if len(x_val) != len(y_val):
                raise HTTPException(
                    status_code=400,
                    detail=f"Validation data mismatch: x_val has {len(x_val)} samples, "
                           f"y_val has {len(y_val)} samples"
                )
            validation_data = (x_val, y_val)
        
        logger.info(f"Starting training for model {model_id}: {request.epochs} epochs, "
                   f"batch_size={request.batch_size}")
        
        history = model.fit(
            x_train, y_train,
            epochs=request.epochs,
            batch_size=request.batch_size,
            validation_data=validation_data,
            verbose=request.verbose
        )
        
        history_serializable = serialize_history(history)
        logger.info(f"Training completed for model {model_id}")
        model_store._record_operation(model_id, "train", {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "samples": len(x_train)
        })
        metrics_collector.record_model_operation("train", model_id)
        
        return TrainResponse(
            model_id=model_id,
            status="trained",
            history=history_serializable,
            message="Model trained successfully"
        )
    except (ModelNotFoundError, ModelNotCompiledError):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error training model: {str(e)}\n{traceback.format_exc()}"
        )


@router.post("/{model_id}/evaluate", response_model=EvaluateResponse)
async def evaluate_model(model_id: str, request: EvaluateRequest):
    """Evaluate a model."""
    try:
        model_store.require_compiled(model_id)
        model = model_store.get_model(model_id)
        
        x_test = validate_array_shape(request.x_test, expected_dim=2)
        if not request.y_test:
            raise HTTPException(status_code=400, detail="y_test cannot be empty")
        y_test = np.array(
            request.y_test,
            dtype=np.int64 if isinstance(request.y_test[0], int) else np.float32
        )
        
        if len(x_test) != len(y_test):
            raise HTTPException(
                status_code=400,
                detail=f"Test data mismatch: x_test has {len(x_test)} samples, "
                       f"y_test has {len(y_test)} samples"
            )
        
        results = model.evaluate(x_test, y_test, verbose=request.verbose)
        
        if isinstance(results, tuple):
            results_dict = {
                "loss": float(results[0]),
                "metrics": [float(r) for r in results[1:]]
            }
        else:
            results_dict = {"results": float(results)}
        
        return EvaluateResponse(
            model_id=model_id,
            status="evaluated",
            results=results_dict,
            message="Model evaluated successfully"
        )
    except (ModelNotFoundError, ModelNotCompiledError):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error evaluating model: {str(e)}\n{traceback.format_exc()}"
        )


@router.post("/{model_id}/predict", response_model=PredictResponse)
async def predict(model_id: str, request: PredictRequest):
    """Make predictions with a model."""
    try:
        model = model_store.get_model(model_id)
        
        x = validate_array_shape(request.x, expected_dim=2)
        predictions = model.predict(x, verbose=request.verbose)
        predictions = serialize_tensor(predictions)
        
        return PredictResponse(
            model_id=model_id,
            status="predicted",
            predictions=predictions,
            message="Predictions generated successfully"
        )
    except ModelNotFoundError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error making predictions: {str(e)}\n{traceback.format_exc()}"
        )


@router.get("", response_model=ModelListResponse)
async def list_models(
    compiled_only: bool = Query(False, description="Only return compiled models"),
    name_search: Optional[str] = Query(None, description="Search models by name (case-insensitive partial match)"),
    created_after: Optional[str] = Query(None, description="Filter models created after this date (ISO format)"),
    created_before: Optional[str] = Query(None, description="Filter models created before this date (ISO format)"),
    limit: Optional[int] = Query(None, ge=1, le=MAX_SEARCH_RESULTS, description="Maximum number of models to return"),
    offset: int = Query(0, ge=0, description="Number of models to skip"),
    sort_by: Optional[str] = Query("created_at", description="Sort by field (created_at, updated_at, name)"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)")
):
    """
    List all models with advanced filtering and search.
    
    - **compiled_only**: If True, only return compiled models
    - **name_search**: Search models by name (case-insensitive partial match)
    - **created_after**: Filter models created after this date (ISO format)
    - **created_before**: Filter models created before this date (ISO format)
    - **limit**: Maximum number of models to return
    - **offset**: Number of models to skip
    - **sort_by**: Sort by field (created_at, updated_at, name)
    - **sort_order**: Sort order (asc, desc)
    """
    models_list = []
    all_models = model_store.list_all()
    
    for model_id, model_data in all_models.items():
        if compiled_only and not model_data.get("compiled", False):
            continue
        
        if name_search:
            model_name = model_data.get("name", "").lower()
            if name_search.lower() not in model_name:
                continue
        
        if created_after:
            filter_date = parse_iso_date(created_after)
            model_date = parse_iso_date(model_data.get("created_at", ""))
            if filter_date and model_date and model_date < filter_date:
                continue
        
        if created_before:
            filter_date = parse_iso_date(created_before)
            model_date = parse_iso_date(model_data.get("created_at", ""))
            if filter_date and model_date and model_date > filter_date:
                continue
        
        models_list.append({
            "model_id": model_id,
            "name": model_data["name"],
            "compiled": model_data["compiled"],
            "created_at": model_data.get("created_at"),
            "updated_at": model_data.get("updated_at")
        })
    
    if sort_by in ["created_at", "updated_at", "name"]:
        reverse = sort_order.lower() == "desc"
        sort_key = (
            lambda x: x["name"].lower() 
            if sort_by == "name" 
            else x.get(sort_by, "")
        )
        models_list.sort(key=sort_key, reverse=reverse)
    
    total_count = len(models_list)
    if limit is not None:
        models_list = models_list[offset:offset + limit]
    elif offset > 0:
        models_list = models_list[offset:]
    
    return ModelListResponse(
        models=[ModelInfo(
            model_id=m["model_id"],
            name=m["name"],
            compiled=m["compiled"]
        ) for m in models_list],
        count=total_count
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """
    Get model information.
    
    Returns basic information about the model including:
    - Model ID
    - Model name
    - Compilation status
    """
    model_data = model_store.get(model_id)
    model = model_data["model"]
    
    info = ModelInfo(
        model_id=model_id,
        name=model_data["name"],
        compiled=model_data["compiled"]
    )
    
    summary = get_model_summary(model)
    logger.debug(f"Model info requested: {model_id} - Summary: {summary}")
    
    return info


@router.get("/{model_id}/summary")
async def get_model_summary_endpoint(model_id: str):
    """
    Get detailed model summary.
    
    Returns comprehensive information about the model including:
    - Model architecture
    - Layer details
    - Parameter counts
    - Compilation status
    """
    try:
        model_data = model_store.get(model_id)
        model = model_data["model"]
        
        summary = get_model_summary(model)
        summary.update({
            "model_id": model_id,
            "name": model_data["name"],
            "compiled": model_data["compiled"],
            "created_at": model_data.get("created_at"),
            "updated_at": model_data.get("updated_at")
        })
        
        return summary
    except ModelNotFoundError:
        raise
    except Exception as e:
        handle_model_operation_error("getting model summary", model_id, e)
        raise HTTPException(status_code=400, detail=f"Error getting model summary: {str(e)}")


@router.delete("/{model_id}", response_model=CreateModelResponse)
async def delete_model(model_id: str):
    """Delete a model."""
    model_store.delete(model_id)
    return CreateModelResponse(
        model_id=model_id,
        name="",
        status="deleted",
        message="Model deleted successfully"
    )


@router.post("/{model_id}/save", response_model=SaveModelResponse)
async def save_model_endpoint(
    model_id: str,
    filepath: Optional[str] = Query(None, description="Path where to save the model")
):
    """Save a model to disk."""
    try:
        model = model_store.get_model(model_id)
        
        if not filepath:
            filepath = f"{settings.default_model_dir}/{model_id}.pth"
        
        dir_path = os.path.dirname(filepath) or "."
        os.makedirs(dir_path, exist_ok=True)
        
        model.save(filepath)
        
        return SaveModelResponse(
            model_id=model_id,
            filepath=filepath,
            status="saved",
            message="Model saved successfully"
        )
    except ModelNotFoundError:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving model: {str(e)}")


@router.post("/load", response_model=LoadModelResponse)
async def load_model_endpoint(
    filepath: str = Query(..., description="Path to the model file"),
    model_id: Optional[str] = Query(None, description="Optional model ID (generated if not provided)")
):
    """
    Load a model from disk.
    
    - **filepath**: Path to the model file
    - **model_id**: Optional model ID (generated if not provided)
    """
    try:
        if not os.path.exists(filepath):
            from ..exceptions import ModelFileNotFoundError
            raise ModelFileNotFoundError(filepath)
        
        if not os.path.isfile(filepath):
            raise HTTPException(status_code=400, detail=f"Path is not a file: {filepath}")
        
        from ..constants import BYTES_PER_MB
        file_size = os.path.getsize(filepath) / BYTES_PER_MB
        if file_size > settings.max_model_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"Model file too large: {file_size:.2f}MB. Maximum: {settings.max_model_size_mb}MB"
            )
        
        loaded_model_id = model_store.load_model(filepath, model_id)
        metrics_collector.record_model_operation("load", loaded_model_id)
        
        return LoadModelResponse(
            model_id=loaded_model_id,
            filepath=filepath,
            status="loaded",
            message="Model loaded successfully"
        )
    except (ModelNotFoundError, HTTPException):
        raise
    except Exception as e:
        handle_model_operation_error("loading model", error=e)
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")


@router.post("/{model_id}/clone")
async def clone_model(model_id: str, new_name: Optional[str] = None):
    """
    Clone an existing model.
    
    Creates a copy of the model with a new ID.
    - **model_id**: ID of the model to clone
    - **new_name**: Optional name for the cloned model
    """
    try:
        model_data = model_store.get(model_id)
        original_model = model_data["model"]
        
        import copy
        cloned_model = copy.deepcopy(original_model)
        
        cloned_name = new_name or f"{model_data['name']}_clone"
        cloned_id = model_store.create(cloned_model, name=cloned_name)
        
        if model_data.get("compiled", False):
            model_store.mark_compiled(cloned_id)
        
        metrics_collector.record_model_operation("clone", cloned_id)
        
        return CreateModelResponse(
            model_id=cloned_id,
            name=cloned_name,
            status="cloned",
            message=f"Model cloned successfully from {model_id}"
        )
    except ModelNotFoundError:
        raise
    except Exception as e:
        handle_model_operation_error("cloning model", model_id, e)
        raise HTTPException(status_code=400, detail=f"Error cloning model: {str(e)}")


@router.patch("/{model_id}/rename")
async def rename_model(model_id: str, new_name: str = Query(..., min_length=1, max_length=MAX_MODEL_NAME_LENGTH)):
    """
    Rename a model.
    
    - **model_id**: ID of the model to rename
    - **new_name**: New name for the model (max {MAX_MODEL_NAME_LENGTH} characters)
    """
    try:
        if not new_name or len(new_name.strip()) == 0:
            raise HTTPException(status_code=400, detail="New name cannot be empty")
        
        if len(new_name) > MAX_MODEL_NAME_LENGTH:
            raise HTTPException(status_code=400, detail=f"Model name cannot exceed {MAX_MODEL_NAME_LENGTH} characters")
        
        old_name = model_store.get(model_id)["name"]
        model_store.update(model_id, name=new_name.strip())
        model_store._record_operation(model_id, "rename", {
            "old_name": old_name,
            "new_name": new_name.strip()
        })
        
        return CreateModelResponse(
            model_id=model_id,
            name=new_name.strip(),
            status="renamed",
            message="Model renamed successfully"
        )
    except ModelNotFoundError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        handle_model_operation_error("renaming model", model_id, e)
        raise HTTPException(status_code=400, detail=f"Error renaming model: {str(e)}")

