"""
Testing Utilities - Utilidades para Testing
==========================================

Utilidades para testing:
- Test fixtures
- Mock generators
- Test data factories
- Integration test helpers
"""

import logging
from typing import Optional, Dict, Any, List, Callable, TypeVar
from datetime import datetime
import random
import string

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TestDataFactory:
    """Factory para datos de prueba"""
    
    @staticmethod
    def generate_string(length: int = 10, prefix: str = "") -> str:
        """Genera string aleatorio"""
        chars = string.ascii_letters + string.digits
        random_str = ''.join(random.choice(chars) for _ in range(length))
        return f"{prefix}{random_str}" if prefix else random_str
    
    @staticmethod
    def generate_email(domain: str = "test.com") -> str:
        """Genera email de prueba"""
        username = TestDataFactory.generate_string(8)
        return f"{username}@{domain}"
    
    @staticmethod
    def generate_project_data(**overrides: Any) -> Dict[str, Any]:
        """Genera datos de proyecto para testing"""
        return {
            "description": TestDataFactory.generate_string(50),
            "project_name": TestDataFactory.generate_string(10, "test_"),
            "author": TestDataFactory.generate_string(10),
            "version": "1.0.0",
            "priority": random.randint(-10, 10),
            "tags": [TestDataFactory.generate_string(5) for _ in range(3)],
            "metadata": {},
            **overrides
        }
    
    @staticmethod
    def generate_user_data(**overrides: Any) -> Dict[str, Any]:
        """Genera datos de usuario para testing"""
        return {
            "id": TestDataFactory.generate_string(8),
            "email": TestDataFactory.generate_email(),
            "name": TestDataFactory.generate_string(10),
            "created_at": datetime.now().isoformat(),
            **overrides
        }


class MockService:
    """Servicio mock para testing"""
    
    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.calls: List[Dict[str, Any]] = []
        self.responses: Dict[str, Any] = {}
    
    def mock_response(self, method: str, response: Any) -> None:
        """Configura respuesta mock"""
        self.responses[method] = response
    
    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Registra llamada y retorna respuesta mock"""
        self.calls.append({
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat()
        })
        return self.responses.get(method)
    
    def get_calls(self, method: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene llamadas registradas"""
        if method:
            return [c for c in self.calls if c["method"] == method]
        return self.calls
    
    def reset(self) -> None:
        """Resetea mock"""
        self.calls = []
        self.responses = {}


class TestFixture:
    """Fixture para testing"""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.setup_called = False
        self.teardown_called = False
    
    def setup(self) -> None:
        """Setup del fixture"""
        self.setup_called = True
        logger.info(f"Fixture {self.name} setup")
    
    def teardown(self) -> None:
        """Teardown del fixture"""
        self.teardown_called = True
        logger.info(f"Fixture {self.name} teardown")
    
    def __enter__(self) -> 'TestFixture':
        """Context manager entry"""
        self.setup()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit"""
        self.teardown()


class IntegrationTestHelper:
    """Helper para tests de integración"""
    
    def __init__(self) -> None:
        self.services: Dict[str, MockService] = {}
    
    def create_mock_service(self, service_name: str) -> MockService:
        """Crea servicio mock"""
        mock = MockService(service_name)
        self.services[service_name] = mock
        return mock
    
    def get_mock_service(self, service_name: str) -> Optional[MockService]:
        """Obtiene servicio mock"""
        return self.services.get(service_name)
    
    def reset_all(self) -> None:
        """Resetea todos los mocks"""
        for mock in self.services.values():
            mock.reset()


def get_test_data_factory() -> TestDataFactory:
    """Obtiene factory de datos de prueba"""
    return TestDataFactory()


def get_integration_test_helper() -> IntegrationTestHelper:
    """Obtiene helper para tests de integración"""
    return IntegrationTestHelper()















