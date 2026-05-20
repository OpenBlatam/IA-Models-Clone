"""
Utilities Endpoint
==================
Endpoint para utilidades y herramientas.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel

from ...utils.transformers import (
    transform_dict, flatten_dict, nest_dict, group_by, sort_dict_list
)
from ...utils.string_helpers import (
    camel_to_snake, snake_to_camel, truncate_string,
    extract_emails, extract_urls, normalize_whitespace
)
from ...utils.date_helpers import (
    now_utc, parse_iso_date, format_iso_date, days_between
)
from ...utils.math_helpers import (
    clamp, percentage, round_to, calculate_median, calculate_percentile
)

router = APIRouter(prefix="/api/v1/utils", tags=["utils"])


class TransformRequest(BaseModel):
    """Request para transformar datos."""
    data: Dict[str, Any]
    operation: str
    params: Dict[str, Any] = {}


class StringTransformRequest(BaseModel):
    """Request para transformar strings."""
    text: str
    operation: str
    params: Dict[str, Any] = {}


@router.post("/transform")
async def transform_data(request: TransformRequest) -> Dict[str, Any]:
    """
    Transformar datos usando utilidades disponibles.
    
    Operations:
    - flatten: Aplanar diccionario
    - nest: Anidar diccionario
    - group_by: Agrupar lista por clave
    - sort: Ordenar lista de diccionarios
    """
    try:
        if request.operation == "flatten":
            separator = request.params.get("separator", ".")
            result = flatten_dict(request.data, separator)
        elif request.operation == "nest":
            separator = request.params.get("separator", ".")
            result = nest_dict(request.data, separator)
        elif request.operation == "group_by":
            key = request.params.get("key")
            if not key:
                raise HTTPException(status_code=400, detail="Key parameter required for group_by")
            items = request.data.get("items", [])
            result = group_by(items, key)
        elif request.operation == "sort":
            key = request.params.get("key")
            reverse = request.params.get("reverse", False)
            if not key:
                raise HTTPException(status_code=400, detail="Key parameter required for sort")
            items = request.data.get("items", [])
            result = sort_dict_list(items, key, reverse)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
        
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/string/transform")
async def transform_string(request: StringTransformRequest) -> Dict[str, Any]:
    """
    Transformar strings usando utilidades disponibles.
    
    Operations:
    - camel_to_snake: Convertir camelCase a snake_case
    - snake_to_camel: Convertir snake_case a camelCase
    - truncate: Truncar string
    - extract_emails: Extraer emails
    - extract_urls: Extraer URLs
    - normalize_whitespace: Normalizar espacios
    """
    try:
        if request.operation == "camel_to_snake":
            result = camel_to_snake(request.text)
        elif request.operation == "snake_to_camel":
            result = snake_to_camel(request.text)
        elif request.operation == "truncate":
            max_length = request.params.get("max_length", 100)
            suffix = request.params.get("suffix", "...")
            result = truncate_string(request.text, max_length, suffix)
        elif request.operation == "extract_emails":
            result = extract_emails(request.text)
        elif request.operation == "extract_urls":
            result = extract_urls(request.text)
        elif request.operation == "normalize_whitespace":
            result = normalize_whitespace(request.text)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
        
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/date/now")
async def get_current_time() -> Dict[str, Any]:
    """Obtener fecha/hora actual en UTC."""
    return {
        "timestamp": format_iso_date(now_utc()),
        "iso": now_utc().isoformat()
    }

