"""
MCP Utils - Utilidades para el servidor MCP
===========================================

Funciones utilitarias para el servidor MCP, incluyendo validación JSON-RPC,
helpers para client ID, y utilidades de rate limiting.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Request
from pydantic import ValidationError

from .mcp_models import JSONRPCRequest

logger = logging.getLogger(__name__)

try:
    import orjson
    _json_dumps = lambda obj: orjson.dumps(obj).decode()
    _json_loads = orjson.loads
    _has_orjson = True
except ImportError:
    import json
    _json_dumps = json.dumps
    _json_loads = json.loads
    _has_orjson = False


def validate_jsonrpc(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validar formato JSON-RPC"""
    try:
        request = JSONRPCRequest(**message)
        return request.dict()
    except ValidationError as e:
        logger.warning(f"Invalid JSON-RPC message: {e}")
        return None


def get_client_id(request: Request) -> str:
    """Obtener ID del cliente desde request"""
    return request.client.host if request.client else "unknown"


def get_json_dumps():
    """Obtener función de serialización JSON"""
    return _json_dumps


def get_json_loads():
    """Obtener función de deserialización JSON"""
    return _json_loads




