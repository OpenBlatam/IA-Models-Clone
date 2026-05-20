"""
MCP Transformers - Transformadores de request/response
======================================================
"""

from typing import Any, Dict, Optional, Callable
from functools import wraps
from .contracts import ContextFrame, PromptFrame


class RequestTransformer:
    """
    Transformador de requests
    
    Permite modificar requests antes de procesarlos.
    """
    
    def __init__(self):
        self._transformers: list[Callable] = []
    
    def register(self, transformer: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Registra un transformador
        
        Args:
            transformer: Función que transforma el request
        """
        self._transformers.append(transformer)
    
    def transform(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica todos los transformadores
        
        Args:
            request: Request original
            
        Returns:
            Request transformado
        """
        result = request
        for transformer in self._transformers:
            result = transformer(result)
        return result


class ResponseTransformer:
    """
    Transformador de responses
    
    Permite modificar responses después de procesarlos.
    """
    
    def __init__(self):
        self._transformers: list[Callable] = []
    
    def register(self, transformer: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Registra un transformador
        
        Args:
            transformer: Función que transforma el response
        """
        self._transformers.append(transformer)
    
    def transform(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica todos los transformadores
        
        Args:
            response: Response original
            
        Returns:
            Response transformado
        """
        result = response
        for transformer in self._transformers:
            result = transformer(result)
        return result


# Transformadores comunes

def add_timestamp_transformer(response: Dict[str, Any]) -> Dict[str, Any]:
    """Agrega timestamp a response"""
    from datetime import datetime
    response["processed_at"] = datetime.utcnow().isoformat()
    return response


def mask_sensitive_data_transformer(response: Dict[str, Any]) -> Dict[str, Any]:
    """Enmascara datos sensibles en response"""
    sensitive_keys = ["password", "secret", "token", "api_key", "auth"]
    
    def mask_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in d.items():
            if any(sk in key.lower() for sk in sensitive_keys):
                result[key] = "***MASKED***"
            elif isinstance(value, dict):
                result[key] = mask_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    mask_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    if "data" in response:
        response["data"] = mask_dict(response["data"])
    
    return response


def compress_context_transformer(response: Dict[str, Any]) -> Dict[str, Any]:
    """Comprime contexto grande en response"""
    if "data" in response and isinstance(response["data"], dict):
        data = response["data"]
        if "content" in data and isinstance(data["content"], str):
            content = data["content"]
            if len(content) > 10000:  # 10KB
                data["content"] = content[:1000] + f"... [truncated, original length: {len(content)}]"
                data["compressed"] = True
    
    return response

