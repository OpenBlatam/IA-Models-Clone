"""
Request/Response Transformer - Transformador de Requests/Responses
==================================================================

Transformación de requests y responses:
- Request transformation
- Response transformation
- Data mapping
- Field filtering
- Format conversion
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Tipos de transformación"""
    REQUEST = "request"
    RESPONSE = "response"
    BOTH = "both"


class RequestResponseTransformer:
    """
    Transformador de requests y responses.
    """
    
    def __init__(self) -> None:
        self.transformers: Dict[str, List[Callable]] = {
            "request": [],
            "response": []
        }
        self.field_mappings: Dict[str, Dict[str, str]] = {}
        self.field_filters: Dict[str, List[str]] = {}
    
    def register_transformer(
        self,
        endpoint: str,
        transformer: Callable,
        transformation_type: TransformationType = TransformationType.BOTH
    ) -> None:
        """Registra transformador"""
        if transformation_type in [TransformationType.REQUEST, TransformationType.BOTH]:
            key = f"{endpoint}:request"
            if key not in self.transformers["request"]:
                self.transformers["request"].append((endpoint, transformer))
        
        if transformation_type in [TransformationType.RESPONSE, TransformationType.BOTH]:
            key = f"{endpoint}:response"
            if key not in self.transformers["response"]:
                self.transformers["response"].append((endpoint, transformer))
        
        logger.info(f"Transformer registered for {endpoint}")
    
    def register_field_mapping(
        self,
        endpoint: str,
        mapping: Dict[str, str]
    ) -> None:
        """Registra mapeo de campos"""
        self.field_mappings[endpoint] = mapping
        logger.info(f"Field mapping registered for {endpoint}")
    
    def register_field_filter(
        self,
        endpoint: str,
        allowed_fields: List[str]
    ) -> None:
        """Registra filtro de campos"""
        self.field_filters[endpoint] = allowed_fields
        logger.info(f"Field filter registered for {endpoint}")
    
    async def transform_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transforma request"""
        transformed = data.copy()
        
        # Aplicar transformadores
        for ep, transformer in self.transformers["request"]:
            if ep == endpoint or ep == "*":
                if asyncio.iscoroutinefunction(transformer):
                    transformed = await transformer(transformed)
                else:
                    transformed = transformer(transformed)
        
        # Aplicar mapeo de campos
        if endpoint in self.field_mappings:
            mapping = self.field_mappings[endpoint]
            for old_key, new_key in mapping.items():
                if old_key in transformed:
                    transformed[new_key] = transformed.pop(old_key)
        
        # Aplicar filtro de campos
        if endpoint in self.field_filters:
            allowed = self.field_filters[endpoint]
            transformed = {k: v for k, v in transformed.items() if k in allowed}
        
        return transformed
    
    async def transform_response(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transforma response"""
        transformed = data.copy()
        
        # Aplicar transformadores
        for ep, transformer in self.transformers["response"]:
            if ep == endpoint or ep == "*":
                if asyncio.iscoroutinefunction(transformer):
                    transformed = await transformer(transformed)
                else:
                    transformed = transformer(transformed)
        
        # Aplicar mapeo de campos
        if endpoint in self.field_mappings:
            mapping = self.field_mappings[endpoint]
            # Mapeo inverso para responses
            reverse_mapping = {v: k for k, v in mapping.items()}
            for old_key, new_key in reverse_mapping.items():
                if old_key in transformed:
                    transformed[new_key] = transformed.pop(old_key)
        
        # Aplicar filtro de campos
        if endpoint in self.field_filters:
            allowed = self.field_filters[endpoint]
            transformed = {k: v for k, v in transformed.items() if k in allowed}
        
        return transformed
    
    def add_default_transformer(
        self,
        transformer: Callable,
        transformation_type: TransformationType = TransformationType.BOTH
    ) -> None:
        """Agrega transformador por defecto"""
        self.register_transformer("*", transformer, transformation_type)


import asyncio


def get_request_response_transformer() -> RequestResponseTransformer:
    """Obtiene transformador de requests/responses"""
    return RequestResponseTransformer()















