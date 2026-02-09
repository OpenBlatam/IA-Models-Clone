"""
Search Router
============

Advanced search and filtering endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging

from ..services import model_store
from ..exceptions import ModelNotFoundError
from ..utils import get_model_summary, handle_model_operation_error, parse_iso_date
from ..constants import DEFAULT_SEARCH_LIMIT, MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/models")
async def search_models(
    q: Optional[str] = Query(None, description="Search query (searches in model name and ID)"),
    compiled: Optional[bool] = Query(None, description="Filter by compilation status"),
    min_layers: Optional[int] = Query(None, ge=1, description="Minimum number of layers"),
    max_layers: Optional[int] = Query(None, ge=1, description="Maximum number of layers"),
    min_parameters: Optional[int] = Query(None, ge=0, description="Minimum number of trainable parameters"),
    max_parameters: Optional[int] = Query(None, ge=0, description="Maximum number of trainable parameters"),
    created_after: Optional[str] = Query(None, description="Created after date (ISO format)"),
    created_before: Optional[str] = Query(None, description="Created before date (ISO format)"),
    limit: int = Query(DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_RESULTS, description="Maximum number of results")
):
    """
    Advanced search for models.
    
    Search models using multiple criteria:
    - Text search in names and IDs
    - Filter by compilation status
    - Filter by layer count
    - Filter by parameter count
    - Filter by creation date
    """
    try:
        results = []
        all_models = model_store.list_all()
        
        for model_id, model_data in all_models.items():
            model = model_data["model"]
            
            if q:
                search_text = q.lower()
                if search_text not in model_id.lower() and search_text not in model_data.get("name", "").lower():
                    continue
            
            if compiled is not None:
                if model_data.get("compiled", False) != compiled:
                    continue
            
            summary = get_model_summary(model)
            
            num_layers = summary.get("num_layers", 0)
            if min_layers is not None and num_layers < min_layers:
                continue
            if max_layers is not None and num_layers > max_layers:
                continue
            
            params = summary.get("trainable_parameters", 0) or 0
            if min_parameters is not None and params < min_parameters:
                continue
            if max_parameters is not None and params > max_parameters:
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
            
            results.append({
                "model_id": model_id,
                "name": model_data["name"],
                "compiled": model_data.get("compiled", False),
                "summary": summary,
                "created_at": model_data.get("created_at"),
                "updated_at": model_data.get("updated_at")
            })
            
            if len(results) >= limit:
                break
        
        return {
            "query": {
                "q": q,
                "compiled": compiled,
                "min_layers": min_layers,
                "max_layers": max_layers,
                "min_parameters": min_parameters,
                "max_parameters": max_parameters,
                "created_after": created_after,
                "created_before": created_before
            },
            "results": results,
            "count": len(results),
            "limit": limit
        }
    except Exception as e:
        handle_model_operation_error("searching models", error=e)
        raise HTTPException(status_code=400, detail=f"Error searching models: {str(e)}")


@router.get("/models/by-layer-type")
async def search_by_layer_type(
    layer_type: str = Query(..., description="Layer type to search for (e.g., 'Dense', 'Conv2D')"),
    limit: int = Query(DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_RESULTS, description="Maximum number of results")
):
    """
    Search models that contain a specific layer type.
    
    - **layer_type**: Type of layer to search for
    - **limit**: Maximum number of results
    """
    try:
        results = []
        all_models = model_store.list_all()
        
        for model_id, model_data in all_models.items():
            model = model_data["model"]
            summary = get_model_summary(model)
            
            layers = summary.get("layers", [])
            layer_types = [layer.get("type", "") for layer in layers]
            
            if layer_type in layer_types or layer_type.lower() in [lt.lower() for lt in layer_types]:
                results.append({
                    "model_id": model_id,
                    "name": model_data["name"],
                    "compiled": model_data.get("compiled", False),
                    "layer_count": len(layers),
                    "matching_layers": [
                        i for i, layer in enumerate(layers)
                        if layer.get("type", "").lower() == layer_type.lower()
                    ],
                    "summary": summary
                })
                
                if len(results) >= limit:
                    break
        
        return {
            "layer_type": layer_type,
            "results": results,
            "count": len(results),
            "limit": limit
        }
    except Exception as e:
        handle_model_operation_error("searching by layer type", error=e)
        raise HTTPException(status_code=400, detail=f"Error searching by layer type: {str(e)}")

