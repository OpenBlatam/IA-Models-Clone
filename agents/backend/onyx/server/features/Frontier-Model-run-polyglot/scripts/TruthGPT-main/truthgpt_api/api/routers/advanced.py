"""
Advanced Operations Router
=========================

Advanced API endpoints for model operations.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging
import numpy as np

from ..services import model_store
from ..exceptions import ModelNotFoundError
from ..utils import (
    get_model_summary, validate_array_shape, handle_model_operation_error, 
    extract_layer_params, increment_counter
)
from ..metrics import metrics_collector
from ..constants import DEFAULT_OPERATION_HISTORY_LIMIT, MAX_OPERATION_HISTORY
import truthgpt as tg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models", "advanced"])


@router.get("/{model_id}/history")
async def get_model_history(
    model_id: str,
    limit: int = Query(DEFAULT_OPERATION_HISTORY_LIMIT, ge=1, le=MAX_OPERATION_HISTORY, description="Maximum number of operations to return")
):
    """
    Get operation history for a model.
    
    Returns a chronological list of all operations performed on the model.
    - **model_id**: Model ID
    - **limit**: Maximum number of operations to return (1-100)
    """
    try:
        model_store.get(model_id)
        history = model_store.get_operation_history(model_id, limit=limit)
        return {
            "model_id": model_id,
            "history": history,
            "total_operations": len(history)
        }
    except ModelNotFoundError:
        raise
    except Exception as e:
        handle_model_operation_error("getting model history", model_id, e)
        raise HTTPException(status_code=400, detail=f"Error getting model history: {str(e)}")


@router.post("/compare")
async def compare_models(
    model_ids: List[str] = Query(..., min_items=2, max_items=10, description="List of model IDs to compare"),
    test_data: Optional[List[List[float]]] = None,
    test_labels: Optional[List[Any]] = None
):
    """
    Compare multiple models.
    
    Compares models based on:
    - Architecture summary
    - Performance metrics (if test data provided)
    - Compilation status
    
    - **model_ids**: List of model IDs to compare (2-10 models)
    - **test_data**: Optional test data for performance comparison
    - **test_labels**: Optional test labels for evaluation
    """
    try:
        comparison = {
            "models": [],
            "comparison": {}
        }
        
        for model_id in model_ids:
            try:
                model_data = model_store.get(model_id)
                model = model_data["model"]
                summary = get_model_summary(model)
                
                model_info = {
                    "model_id": model_id,
                    "name": model_data["name"],
                    "compiled": model_data["compiled"],
                    "summary": summary,
                    "created_at": model_data.get("created_at"),
                    "updated_at": model_data.get("updated_at")
                }
                
                if test_data is not None and test_labels is not None and model_data.get("compiled", False):
                    try:
                        x_test = validate_array_shape(test_data, expected_dim=2)
                        y_test = np.array(test_labels)
                        
                        if len(x_test) != len(y_test):
                            model_info["evaluation_error"] = "Test data length mismatch"
                        else:
                            results = model.evaluate(x_test, y_test, verbose=0)
                            if isinstance(results, tuple):
                                model_info["evaluation"] = {
                                    "loss": float(results[0]),
                                    "metrics": [float(r) for r in results[1:]]
                                }
                            else:
                                model_info["evaluation"] = {"result": float(results)}
                    except Exception as e:
                        model_info["evaluation_error"] = str(e)
                
                comparison["models"].append(model_info)
            except ModelNotFoundError:
                comparison["models"].append({
                    "model_id": model_id,
                    "error": "Model not found"
                })
            except Exception as e:
                comparison["models"].append({
                    "model_id": model_id,
                    "error": str(e)
                })
        
        if len(comparison["models"]) >= 2:
            compiled_models = [m for m in comparison["models"] if m.get("compiled", False)]
            models_with_summary = [m for m in comparison["models"] if "summary" in m]
            total_params = sum(
                m.get("summary", {}).get("trainable_parameters", 0) or 0
                for m in models_with_summary
            )
            
            comparison["comparison"] = {
                "total_models": len(comparison["models"]),
                "compiled_models": len(compiled_models),
                "average_parameters": total_params / len(models_with_summary) if models_with_summary else 0
            }
            
            if test_data is not None:
                evaluated_models = [m for m in comparison["models"] if "evaluation" in m]
                if evaluated_models:
                    losses = [m["evaluation"]["loss"] for m in evaluated_models if "loss" in m.get("evaluation", {})]
                    if losses:
                        comparison["comparison"]["best_loss"] = min(losses)
                        comparison["comparison"]["worst_loss"] = max(losses)
                        comparison["comparison"]["average_loss"] = sum(losses) / len(losses)
        
        metrics_collector.record_model_operation("compare", None)
        return comparison
    except Exception as e:
        handle_model_operation_error("comparing models", error=e)
        raise HTTPException(status_code=400, detail=f"Error comparing models: {str(e)}")


@router.get("/{model_id}/export-config")
async def export_model_config(model_id: str):
    """
    Export model configuration.
    
    Returns a JSON-serializable configuration that can be used to recreate the model.
    - **model_id**: Model ID to export
    """
    try:
        model_data = model_store.get(model_id)
        model = model_data["model"]
        
        config = {
            "model_id": model_id,
            "name": model_data["name"],
            "compiled": model_data.get("compiled", False),
            "created_at": model_data.get("created_at"),
            "layers": []
        }
        
        if hasattr(model, 'layers_list'):
            for layer in model.layers_list:
                layer_info = {
                    "type": type(layer).__name__.lower(),
                    "params": extract_layer_params(layer)
                }
                config["layers"].append(layer_info)
        
        return config
    except ModelNotFoundError:
        raise
    except Exception as e:
        handle_model_operation_error("exporting model config", model_id, e)
        raise HTTPException(status_code=400, detail=f"Error exporting model config: {str(e)}")


@router.get("/stats/detailed")
async def get_detailed_stats():
    """
    Get detailed statistics about all models.
    
    Returns comprehensive statistics including:
    - Model distribution by type
    - Compilation statistics
    - Operation statistics
    - Performance metrics
    """
    all_models = model_store.list_all()
    
    stats = {
        "total_models": len(all_models),
        "compiled_models": sum(1 for m in all_models.values() if m.get("compiled", False)),
        "not_compiled_models": sum(1 for m in all_models.values() if not m.get("compiled", False)),
        "model_types": {},
        "total_parameters": 0,
        "models_by_creation": {},
        "operation_counts": {}
    }
    
    for model_id, model_data in all_models.items():
        model = model_data["model"]
        model_type = type(model).__name__
        increment_counter(stats["model_types"], model_type)
        
        try:
            summary = get_model_summary(model)
            if "trainable_parameters" in summary:
                stats["total_parameters"] += summary["trainable_parameters"]
        except (AttributeError, TypeError, KeyError):
            pass
        
        created_date = model_data.get("created_at", "")[:10]
        if created_date:
            increment_counter(stats["models_by_creation"], created_date)
        
        history = model_store.get_operation_history(model_id)
        for op in history:
            op_type = op.get("operation", "unknown")
            increment_counter(stats["operation_counts"], op_type)
    
    metrics_stats = metrics_collector.get_stats()
    stats["api_metrics"] = {
        "total_requests": metrics_stats.get("total_requests", 0),
        "total_errors": metrics_stats.get("total_errors", 0),
        "error_rate": metrics_stats.get("error_rate", 0),
        "uptime_seconds": metrics_stats.get("uptime_seconds", 0)
    }
    
    return stats

