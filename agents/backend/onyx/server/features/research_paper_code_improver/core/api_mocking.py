"""
API Mocking System - Sistema de mocking para APIs
==================================================
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class MockResponse:
    """Response mock"""
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    delay: float = 0.0  # segundos


@dataclass
class MockEndpoint:
    """Endpoint mock"""
    method: str
    path: str
    response: MockResponse
    condition: Optional[Dict[str, Any]] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def matches(self, method: str, path: str, request_data: Optional[Dict[str, Any]] = None) -> bool:
        """Verifica si el endpoint coincide"""
        if not self.enabled:
            return False
        
        if self.method.upper() != method.upper():
            return False
        
        # Path matching simple (puede mejorarse con regex)
        if self.path != path and not path.startswith(self.path):
            return False
        
        # Verificar condiciones
        if self.condition and request_data:
            for key, value in self.condition.items():
                if request_data.get(key) != value:
                    return False
        
        return True


class APIMockingSystem:
    """Sistema de mocking para APIs"""
    
    def __init__(self):
        self.mocks: List[MockEndpoint] = []
    
    def add_mock(
        self,
        method: str,
        path: str,
        response: MockResponse,
        condition: Optional[Dict[str, Any]] = None
    ) -> MockEndpoint:
        """Agrega un mock"""
        mock = MockEndpoint(
            method=method,
            path=path,
            response=response,
            condition=condition
        )
        
        self.mocks.append(mock)
        logger.info(f"Mock agregado: {method} {path}")
        return mock
    
    def find_mock(
        self,
        method: str,
        path: str,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Optional[MockResponse]:
        """Encuentra un mock que coincida"""
        for mock in self.mocks:
            if mock.matches(method, path, request_data):
                return mock.response
        return None
    
    def enable_mock(self, method: str, path: str) -> bool:
        """Habilita un mock"""
        for mock in self.mocks:
            if mock.method == method and mock.path == path:
                mock.enabled = True
                return True
        return False
    
    def disable_mock(self, method: str, path: str) -> bool:
        """Deshabilita un mock"""
        for mock in self.mocks:
            if mock.method == method and mock.path == path:
                mock.enabled = False
                return True
        return False
    
    def list_mocks(self) -> List[Dict[str, Any]]:
        """Lista todos los mocks"""
        return [
            {
                "method": mock.method,
                "path": mock.path,
                "status_code": mock.response.status_code,
                "enabled": mock.enabled,
                "created_at": mock.created_at.isoformat()
            }
            for mock in self.mocks
        ]




