"""
Request/Response Transformer - Transformación de requests y responses
======================================================================
"""

import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TransformationRule:
    """Regla de transformación"""
    name: str
    request_transformer: Optional[Callable] = None
    response_transformer: Optional[Callable] = None
    condition: Optional[Callable] = None
    priority: int = 0


class RequestResponseTransformer:
    """Transformador de requests y responses"""
    
    def __init__(self):
        self.rules: List[TransformationRule] = []
    
    def add_rule(self, rule: TransformationRule):
        """Agrega una regla de transformación"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def transform_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Any]
    ) -> tuple[str, str, Dict[str, str], Any]:
        """Transforma un request"""
        transformed_method = method
        transformed_path = path
        transformed_headers = headers.copy()
        transformed_body = body
        
        for rule in self.rules:
            if rule.condition and not rule.condition(method, path, headers, body):
                continue
            
            if rule.request_transformer:
                transformed_method, transformed_path, transformed_headers, transformed_body = \
                    rule.request_transformer(transformed_method, transformed_path, transformed_headers, transformed_body)
        
        return transformed_method, transformed_path, transformed_headers, transformed_body
    
    def transform_response(
        self,
        status_code: int,
        headers: Dict[str, str],
        body: Any,
        original_request: Optional[Dict[str, Any]] = None
    ) -> tuple[int, Dict[str, str], Any]:
        """Transforma un response"""
        transformed_status = status_code
        transformed_headers = headers.copy()
        transformed_body = body
        
        for rule in self.rules:
            if rule.response_transformer:
                transformed_status, transformed_headers, transformed_body = \
                    rule.response_transformer(transformed_status, transformed_headers, transformed_body, original_request)
        
        return transformed_status, transformed_headers, transformed_body




