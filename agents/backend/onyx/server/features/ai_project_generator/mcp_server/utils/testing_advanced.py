"""
Advanced Testing Utilities - Utilidades avanzadas de testing
============================================================

Utilidades avanzadas para testing, incluyendo mocks, fixtures,
y helpers para testing complejo.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable, Awaitable
from contextlib import contextmanager
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MockResponse:
    """Mock de respuesta HTTP."""
    
    def __init__(
        self,
        status_code: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or ""
        self.headers = headers or {}
    
    def json(self) -> Dict[str, Any]:
        """Obtener JSON de respuesta."""
        return self._json_data
    
    def raise_for_status(self) -> None:
        """Lanzar excepción si status code es error."""
        if 400 <= self.status_code < 600:
            raise Exception(f"HTTP {self.status_code}")


class TestDataFactory:
    """
    Factory para crear datos de test.
    
    Facilita la creación de datos de prueba consistentes
    y realistas.
    """
    
    @staticmethod
    def create_resource_manifest(
        resource_id: str = "test-resource",
        name: str = "Test Resource",
        resource_type: str = "filesystem",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Crear manifest de recurso para testing.
        
        Args:
            resource_id: ID del recurso
            name: Nombre del recurso
            resource_type: Tipo de recurso
            **kwargs: Campos adicionales
        
        Returns:
            Diccionario con manifest
        """
        manifest = {
            "resource_id": resource_id,
            "name": name,
            "type": resource_type,
            "connector": resource_type,
            "operations": ["read", "list"],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
            **kwargs
        }
        return manifest
    
    @staticmethod
    def create_user(
        user_id: str = "test-user",
        email: str = "test@example.com",
        scopes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Crear usuario de test.
        
        Args:
            user_id: ID del usuario
            email: Email del usuario
            scopes: Scopes del usuario
            **kwargs: Campos adicionales
        
        Returns:
            Diccionario con usuario
        """
        user = {
            "sub": user_id,
            "email": email,
            "scopes": scopes or ["read"],
            "iat": int(datetime.now().timestamp()),
            "exp": int((datetime.now() + timedelta(hours=1)).timestamp()),
            **kwargs
        }
        return user
    
    @staticmethod
    def create_request(
        resource_id: str = "test-resource",
        operation: str = "read",
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Crear request de test.
        
        Args:
            resource_id: ID del recurso
            operation: Operación a ejecutar
            parameters: Parámetros de la operación
            **kwargs: Campos adicionales
        
        Returns:
            Diccionario con request
        """
        request = {
            "resource_id": resource_id,
            "operation": operation,
            "parameters": parameters or {},
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        return request


class AsyncTestHelper:
    """Helper para testing asíncrono."""
    
    @staticmethod
    def run_async(coro: Awaitable[Any]) -> Any:
        """
        Ejecutar coroutine en loop de eventos.
        
        Args:
            coro: Coroutine a ejecutar
        
        Returns:
            Resultado de la coroutine
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    @staticmethod
    @contextmanager
    def async_context():
        """Context manager para operaciones asíncronas."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield loop
        finally:
            loop.close()


class MockHelper:
    """Helper para crear mocks."""
    
    @staticmethod
    def create_mock_connector(
        connector_type: str = "filesystem",
        operations: Optional[List[str]] = None,
        **kwargs
    ) -> Mock:
        """
        Crear mock de conector.
        
        Args:
            connector_type: Tipo de conector
            operations: Operaciones soportadas
            **kwargs: Configuración adicional
        
        Returns:
            Mock del conector
        """
        mock = MagicMock()
        mock.connector_type = connector_type
        mock.supported_operations = operations or ["read", "list"]
        mock.execute = MagicMock(return_value={"status": "success", "data": {}})
        mock.health_check = MagicMock(return_value=True)
        
        for key, value in kwargs.items():
            setattr(mock, key, value)
        
        return mock
    
    @staticmethod
    def create_mock_security_manager(**kwargs) -> Mock:
        """
        Crear mock de security manager.
        
        Args:
            **kwargs: Configuración adicional
        
        Returns:
            Mock del security manager
        """
        mock = MagicMock()
        mock.validate_token = MagicMock(return_value={"sub": "test-user"})
        mock.create_token = MagicMock(return_value="mock-token")
        mock.check_scope = MagicMock(return_value=True)
        
        for key, value in kwargs.items():
            setattr(mock, key, value)
        
        return mock


class TestAssertions:
    """Assertions personalizadas para testing."""
    
    @staticmethod
    def assert_response_structure(response: Dict[str, Any], required_fields: List[str]) -> None:
        """
        Verificar que la respuesta tenga la estructura esperada.
        
        Args:
            response: Respuesta a verificar
            required_fields: Campos requeridos
        
        Raises:
            AssertionError: Si falta algún campo
        """
        missing = [field for field in required_fields if field not in response]
        if missing:
            raise AssertionError(f"Missing required fields: {missing}")
    
    @staticmethod
    def assert_response_time(response_time: float, max_time: float) -> None:
        """
        Verificar que el tiempo de respuesta sea aceptable.
        
        Args:
            response_time: Tiempo de respuesta en segundos
            max_time: Tiempo máximo permitido
        
        Raises:
            AssertionError: Si el tiempo excede el máximo
        """
        if response_time > max_time:
            raise AssertionError(
                f"Response time {response_time:.3f}s exceeds maximum {max_time:.3f}s"
            )
    
    @staticmethod
    def assert_error_code(response: Dict[str, Any], expected_code: str) -> None:
        """
        Verificar código de error en respuesta.
        
        Args:
            response: Respuesta a verificar
            expected_code: Código de error esperado
        
        Raises:
            AssertionError: Si el código no coincide
        """
        if response.get("error_code") != expected_code:
            raise AssertionError(
                f"Expected error code '{expected_code}', got '{response.get('error_code')}'"
            )


class TestFixtureManager:
    """Gestor de fixtures de testing."""
    
    def __init__(self):
        """Inicializar gestor de fixtures."""
        self._fixtures: Dict[str, Any] = {}
        self._cleanup: List[Callable] = []
    
    def register_fixture(self, name: str, fixture: Any, cleanup: Optional[Callable] = None) -> None:
        """
        Registrar fixture.
        
        Args:
            name: Nombre de la fixture
            fixture: Objeto fixture
            cleanup: Función de limpieza (opcional)
        """
        self._fixtures[name] = fixture
        if cleanup:
            self._cleanup.append(cleanup)
    
    def get_fixture(self, name: str) -> Any:
        """
        Obtener fixture.
        
        Args:
            name: Nombre de la fixture
        
        Returns:
            Objeto fixture
        
        Raises:
            KeyError: Si la fixture no existe
        """
        if name not in self._fixtures:
            raise KeyError(f"Fixture '{name}' not found")
        return self._fixtures[name]
    
    def cleanup(self) -> None:
        """Ejecutar limpieza de todas las fixtures."""
        for cleanup_func in reversed(self._cleanup):
            try:
                cleanup_func()
            except Exception as e:
                logger.warning(f"Error in cleanup: {e}")
        self._fixtures.clear()
        self._cleanup.clear()


def patch_config(config_updates: Dict[str, Any]):
    """
    Decorador/context manager para parchear configuración.
    
    Args:
        config_updates: Actualizaciones de configuración
    
    Example:
        @patch_config({"server.port": 8080})
        def test_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with patch.dict("mcp_server.config.settings", config_updates):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_timeout(timeout: float):
    """
    Decorador para agregar timeout a tests.
    
    Args:
        timeout: Timeout en segundos
    
    Example:
        @with_timeout(5.0)
        def test_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test exceeded timeout of {timeout}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator


__all__ = [
    "MockResponse",
    "TestDataFactory",
    "AsyncTestHelper",
    "MockHelper",
    "TestAssertions",
    "TestFixtureManager",
    "patch_config",
    "with_timeout",
]

