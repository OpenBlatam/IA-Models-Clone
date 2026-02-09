"""
Helpers Endpoint
================
Endpoint para helpers y utilidades adicionales.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel, EmailStr

from ...utils.validation_helpers import (
    is_valid_email, is_valid_url, is_valid_phone,
    validate_range, validate_length
)
from ...utils.formatting import (
    format_number, format_bytes, format_percentage,
    format_duration_human, format_datetime_human
)
from ...utils.collection_helpers import (
    unique, filter_by, find_first, partition
)
from datetime import datetime

router = APIRouter(prefix="/api/v1/helpers", tags=["helpers"])


class ValidateRequest(BaseModel):
    """Request para validación."""
    value: Any
    type: str
    params: Dict[str, Any] = {}


class FormatRequest(BaseModel):
    """Request para formateo."""
    value: Any
    format_type: str
    params: Dict[str, Any] = {}


class CollectionRequest(BaseModel):
    """Request para operaciones de colecciones."""
    items: List[Any]
    operation: str
    params: Dict[str, Any] = {}


@router.post("/validate")
async def validate_value(request: ValidateRequest) -> Dict[str, Any]:
    """
    Validar valores.
    
    Types:
    - email: Validar email
    - url: Validar URL
    - phone: Validar teléfono
    - range: Validar rango
    - length: Validar longitud
    """
    try:
        if request.type == "email":
            result = is_valid_email(str(request.value))
        elif request.type == "url":
            result = is_valid_url(str(request.value))
        elif request.type == "phone":
            result = is_valid_phone(str(request.value))
        elif request.type == "range":
            min_val = request.params.get("min")
            max_val = request.params.get("max")
            if min_val is None or max_val is None:
                raise HTTPException(status_code=400, detail="min and max parameters required for range validation")
            result = validate_range(float(request.value), float(min_val), float(max_val))
        elif request.type == "length":
            min_len = request.params.get("min_length", 0)
            max_len = request.params.get("max_length")
            result = validate_length(str(request.value), min_len, max_len)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown validation type: {request.type}")
        
        return {"success": True, "valid": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/format")
async def format_value(request: FormatRequest) -> Dict[str, Any]:
    """
    Formatear valores.
    
    Format types:
    - number: Formatear número
    - bytes: Formatear bytes
    - percentage: Formatear porcentaje
    - duration: Formatear duración
    - datetime: Formatear fecha/hora
    """
    try:
        if request.format_type == "number":
            decimals = request.params.get("decimals", 2)
            separator = request.params.get("thousands_separator", ",")
            result = format_number(float(request.value), decimals, separator)
        elif request.format_type == "bytes":
            result = format_bytes(int(request.value))
        elif request.format_type == "percentage":
            decimals = request.params.get("decimals", 1)
            result = format_percentage(float(request.value), decimals)
        elif request.format_type == "duration":
            result = format_duration_human(float(request.value))
        elif request.format_type == "datetime":
            include_time = request.params.get("include_time", True)
            dt = datetime.fromisoformat(str(request.value)) if isinstance(request.value, str) else request.value
            result = format_datetime_human(dt, include_time)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format type: {request.format_type}")
        
        return {"success": True, "formatted": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collection")
async def collection_operation(request: CollectionRequest) -> Dict[str, Any]:
    """
    Operaciones con colecciones.
    
    Operations:
    - unique: Obtener items únicos
    - filter: Filtrar items
    - find: Encontrar primer item
    - partition: Dividir en dos grupos
    """
    try:
        if request.operation == "unique":
            result = unique(request.items)
        elif request.operation == "filter":
            key = request.params.get("key")
            value = request.params.get("value")
            if not key or value is None:
                raise HTTPException(status_code=400, detail="key and value parameters required for filter")
            result = filter_by(request.items, key, value)
        elif request.operation == "find":
            # Para find necesitaríamos una función de predicado, simplificamos
            result = request.items[0] if request.items else None
        elif request.operation == "partition":
            # Simplificado - necesitaría función de predicado
            mid = len(request.items) // 2
            result = {"true": request.items[:mid], "false": request.items[mid:]}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")
        
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

