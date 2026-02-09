"""
API Mocking - Mocking de APIs para Testing
==========================================

Mocking de APIs para testing:
- Request mocking
- Response mocking
- Scenario-based mocking
- Dynamic mocking
- Mock validation
"""

from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from enum import Enum
import re

from .shared_utils import get_logger

logger = get_logger(__name__)


class MockMatchType(str, Enum):
    """Tipos de matching para mocks"""
    EXACT = "exact"
    REGEX = "regex"
    PARTIAL = "partial"
    CUSTOM = "custom"


class APIMock:
    """Mock de API"""
    
    def __init__(
        self,
        mock_id: str,
        method: str,
        path: str,
        response: Dict[str, Any],
        status_code: int = 200,
        match_type: MockMatchType = MockMatchType.EXACT,
        condition: Optional[Callable] = None
    ) -> None:
        self.mock_id = mock_id
        self.method = method.upper()
        self.path = path
        self.response = response
        self.status_code = status_code
        self.match_type = match_type
        self.condition = condition
        self.hit_count = 0
        self.created_at = datetime.now()
    
    def matches(self, method: str, path: str, request: Optional[Dict[str, Any]] = None) -> bool:
        """Verifica si el mock coincide"""
        if self.method != method.upper():
            return False
        
        if self.match_type == MockMatchType.EXACT:
            return self.path == path
        elif self.match_type == MockMatchType.REGEX:
            return bool(re.match(self.path, path))
        elif self.match_type == MockMatchType.PARTIAL:
            return self.path in path
        elif self.match_type == MockMatchType.CUSTOM:
            if self.condition:
                return self.condition(method, path, request)
            return False
        
        return False
    
    def get_response(self) -> Dict[str, Any]:
        """Obtiene respuesta del mock"""
        self.hit_count += 1
        return {
            "status_code": self.status_code,
            "body": self.response,
            "headers": {
                "X-Mock-ID": self.mock_id,
                "X-Mock-Hit-Count": str(self.hit_count)
            }
        }


class APIMockServer:
    """
    Servidor de mocks para APIs.
    """
    
    def __init__(self) -> None:
        self.mocks: Dict[str, APIMock] = {}
        self.scenarios: Dict[str, List[str]] = {}
        self.active_scenario: Optional[str] = None
        self.request_history: List[Dict[str, Any]] = []
    
    def register_mock(
        self,
        mock_id: str,
        method: str,
        path: str,
        response: Dict[str, Any],
        status_code: int = 200,
        match_type: MockMatchType = MockMatchType.EXACT,
        condition: Optional[Callable] = None
    ) -> APIMock:
        """Registra mock"""
        mock = APIMock(mock_id, method, path, response, status_code, match_type, condition)
        self.mocks[mock_id] = mock
        logger.info(f"API mock registered: {mock_id} - {method} {path}")
        return mock
    
    def find_mock(
        self,
        method: str,
        path: str,
        request: Optional[Dict[str, Any]] = None
    ) -> Optional[APIMock]:
        """Encuentra mock que coincide"""
        # Si hay scenario activo, buscar solo en ese scenario
        mock_ids_to_check = []
        if self.active_scenario:
            mock_ids_to_check = self.scenarios.get(self.active_scenario, [])
        else:
            mock_ids_to_check = list(self.mocks.keys())
        
        # Buscar mock que coincida
        for mock_id in mock_ids_to_check:
            mock = self.mocks.get(mock_id)
            if mock and mock.matches(method, path, request):
                return mock
        
        return None
    
    async def handle_request(
        self,
        method: str,
        path: str,
        request: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Maneja request y devuelve mock response"""
        mock = self.find_mock(method, path, request)
        
        if mock:
            response = mock.get_response()
            self.request_history.append({
                "method": method,
                "path": path,
                "mock_id": mock.mock_id,
                "timestamp": datetime.now().isoformat(),
                "response_status": response["status_code"]
            })
            return response
        
        logger.warning(f"No mock found for {method} {path}")
        return None
    
    def create_scenario(self, scenario_name: str, mock_ids: List[str]) -> None:
        """Crea scenario con mocks"""
        self.scenarios[scenario_name] = mock_ids
        logger.info(f"Scenario created: {scenario_name} with {len(mock_ids)} mocks")
    
    def activate_scenario(self, scenario_name: str) -> None:
        """Activa scenario"""
        if scenario_name in self.scenarios:
            self.active_scenario = scenario_name
            logger.info(f"Scenario activated: {scenario_name}")
        else:
            logger.warning(f"Scenario not found: {scenario_name}")
    
    def deactivate_scenario(self) -> None:
        """Desactiva scenario"""
        self.active_scenario = None
        logger.info("Scenario deactivated")
    
    def get_mock_stats(self, mock_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de mock"""
        mock = self.mocks.get(mock_id)
        if not mock:
            return None
        
        return {
            "mock_id": mock_id,
            "method": mock.method,
            "path": mock.path,
            "hit_count": mock.hit_count,
            "status_code": mock.status_code,
            "created_at": mock.created_at.isoformat()
        }
    
    def clear_mocks(self) -> None:
        """Limpia todos los mocks"""
        self.mocks.clear()
        self.scenarios.clear()
        self.active_scenario = None
        logger.info("All mocks cleared")


def get_api_mock_server() -> APIMockServer:
    """Obtiene servidor de mocks"""
    return APIMockServer()




