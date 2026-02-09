"""
Batch Operations Router
======================

API endpoints for batch operations on models.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from ..schemas import BatchDeleteRequest, BatchDeleteResponse, BatchPredictRequest, BatchPredictResponse
from ..services import model_store
from ..exceptions import ModelNotFoundError, ModelNotCompiledError
from ..utils import validate_array_shape, serialize_tensor, handle_model_operation_error, increment_counter
from ..metrics import metrics_collector
import truthgpt as tg
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models/batch", tags=["models", "batch"])


@router.post("/delete", response_model=BatchDeleteResponse)
async def batch_delete(request: BatchDeleteRequest):
    """
    Delete multiple models in a single operation.
    
    - **model_ids**: List of model IDs to delete (max 100)
    
    Returns information about successful and failed deletions.
    """
    deleted = []
    failed = []
    
    for model_id in request.model_ids:
        try:
            model_store.delete(model_id)
            deleted.append(model_id)
            logger.info(f"Model deleted in batch: {model_id}")
        except ModelNotFoundError:
            failed.append({
                "model_id": model_id,
                "error": "Model not found"
            })
        except Exception as e:
            failed.append({
                "model_id": model_id,
                "error": str(e)
            })
            handle_model_operation_error("deleting model in batch", model_id, e)
    
    metrics_collector.record_model_operation("batch_delete", None)
    
    return BatchDeleteResponse(
        deleted=deleted,
        failed=failed,
        total_requested=len(request.model_ids),
        total_deleted=len(deleted),
        total_failed=len(failed)
    )


@router.post("/predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """
    Get predictions from multiple models.
    
    - **model_ids**: List of model IDs (max 10)
    - **x**: Input data for predictions
    - **verbose**: Verbosity level
    
    Returns predictions from all successful models and errors for failed ones.
    """
    predictions = {}
    failed = []
    
    x = validate_array_shape(request.x, expected_dim=2)
    
    for model_id in request.model_ids:
        try:
            model = model_store.get_model(model_id)
            model_predictions = model.predict(x, verbose=request.verbose)
            predictions[model_id] = serialize_tensor(model_predictions)
            logger.debug(f"Batch prediction successful for model {model_id}")
        except ModelNotFoundError:
            failed.append({
                "model_id": model_id,
                "error": "Model not found"
            })
        except Exception as e:
            failed.append({
                "model_id": model_id,
                "error": str(e)
            })
            handle_model_operation_error("getting prediction from model in batch", model_id, e)
    
    metrics_collector.record_model_operation("batch_predict", None)
    
    return BatchPredictResponse(
        predictions=predictions,
        failed=failed,
        total_models=len(request.model_ids),
        successful=len(predictions),
        failed_count=len(failed)
    )


@router.get("/stats")
async def get_batch_stats():
    """
    Get statistics about models in the store.
    
    Returns aggregated statistics including:
    - Total models
    - Compiled vs non-compiled
    - Model types distribution
    - Recent operations
    """
    all_models = model_store.list_all()
    
    compiled_count = sum(1 for m in all_models.values() if m.get("compiled", False))
    stats = {
        "total_models": len(all_models),
        "compiled": compiled_count,
        "not_compiled": len(all_models) - compiled_count,
        "model_types": {},
        "recent_operations": {}
    }
    
    for model_data in all_models.values():
        model_type = type(model_data["model"]).__name__
        increment_counter(stats["model_types"], model_type)
    
    metrics_stats = metrics_collector.get_stats()
    stats["recent_operations"] = metrics_stats.get("model_operations", {})
    
    return stats

