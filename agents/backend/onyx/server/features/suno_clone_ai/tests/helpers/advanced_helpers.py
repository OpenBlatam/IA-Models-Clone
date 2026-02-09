"""
Helpers avanzados para tests más complejos
"""

import asyncio
import time
from typing import List, Dict, Any, Callable, Optional
from unittest.mock import Mock, AsyncMock
from contextlib import asynccontextmanager
import pytest


class AsyncTestHelper:
    """Helper para tests asíncronos avanzados"""
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        error_message: str = "Condition not met"
    ) -> bool:
        """
        Espera hasta que una condición se cumpla
        
        Args:
            condition: Función que retorna True cuando se cumple
            timeout: Tiempo máximo de espera
            interval: Intervalo entre verificaciones
            error_message: Mensaje de error si falla
        
        Returns:
            True si se cumplió, False si timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
        
        if not condition():
            raise TimeoutError(f"{error_message} (timeout: {timeout}s)")
        return False
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_attempts: int = 3,
        delay: float = 0.5,
        exceptions: tuple = (Exception,)
    ) -> Any:
        """
        Reintenta una función async hasta que tenga éxito
        
        Args:
            func: Función async a ejecutar
            max_attempts: Número máximo de intentos
            delay: Delay entre intentos
            exceptions: Excepciones que deben causar retry
        
        Returns:
            Resultado de la función
        """
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return await func()
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                else:
                    raise last_exception
        
        raise last_exception


class MockVerifier:
    """Helper para verificar llamadas a mocks de forma avanzada"""
    
    @staticmethod
    def verify_call_count(mock: Mock, expected_count: int, method_name: str = None):
        """
        Verifica que un mock fue llamado un número específico de veces
        
        Args:
            mock: Mock a verificar
            expected_count: Número esperado de llamadas
            method_name: Nombre del método (opcional)
        """
        if method_name:
            actual_count = getattr(mock, method_name).call_count
        else:
            actual_count = mock.call_count
        
        assert actual_count == expected_count, \
            f"Expected {expected_count} calls, got {actual_count}"
    
    @staticmethod
    def verify_call_with_args(mock: Mock, *args, **kwargs):
        """
        Verifica que un mock fue llamado con argumentos específicos
        
        Args:
            mock: Mock a verificar
            *args: Argumentos posicionales esperados
            **kwargs: Argumentos nombrados esperados
        """
        mock.assert_called_with(*args, **kwargs)
    
    @staticmethod
    def verify_call_contains(mock: Mock, **kwargs):
        """
        Verifica que alguna llamada contenga los kwargs especificados
        
        Args:
            mock: Mock a verificar
            **kwargs: Argumentos que deben estar presentes
        """
        calls = mock.call_args_list
        for call in calls:
            call_kwargs = call.kwargs
            if all(k in call_kwargs and call_kwargs[k] == v for k, v in kwargs.items()):
                return
        
        raise AssertionError(f"No call found with kwargs: {kwargs}")


class ResponseValidator:
    """Helper para validar respuestas HTTP de forma avanzada"""
    
    @staticmethod
    def validate_response_structure(
        response_data: Dict[str, Any],
        required_fields: List[str],
        optional_fields: List[str] = None
    ):
        """
        Valida la estructura de una respuesta
        
        Args:
            response_data: Datos de la respuesta
            required_fields: Campos requeridos
            optional_fields: Campos opcionales
        """
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"
        
        if optional_fields:
            all_fields = set(required_fields) | set(optional_fields)
            unexpected_fields = set(response_data.keys()) - all_fields
            if unexpected_fields:
                pytest.warn(f"Unexpected fields in response: {unexpected_fields}")
    
    @staticmethod
    def validate_status_code(response, expected_code: int, allowed_codes: List[int] = None):
        """
        Valida el código de estado HTTP
        
        Args:
            response: Objeto response
            expected_code: Código esperado
            allowed_codes: Códigos alternativos permitidos
        """
        if allowed_codes:
            assert response.status_code in [expected_code] + allowed_codes, \
                f"Expected {expected_code} or {allowed_codes}, got {response.status_code}"
        else:
            assert response.status_code == expected_code, \
                f"Expected {expected_code}, got {response.status_code}"


class PerformanceHelper:
    """Helper para tests de performance"""
    
    @staticmethod
    def measure_execution_time(func: Callable) -> float:
        """
        Mide el tiempo de ejecución de una función
        
        Args:
            func: Función a medir
        
        Returns:
            Tiempo en segundos
        """
        start_time = time.time()
        func()
        return time.time() - start_time
    
    @staticmethod
    async def measure_async_execution_time(func: Callable) -> float:
        """
        Mide el tiempo de ejecución de una función async
        
        Args:
            func: Función async a medir
        
        Returns:
            Tiempo en segundos
        """
        start_time = time.time()
        await func()
        return time.time() - start_time
    
    @staticmethod
    def assert_execution_time_under(
        func: Callable,
        max_time: float,
        message: str = None
    ):
        """
        Verifica que una función se ejecuta en menos de un tiempo máximo
        
        Args:
            func: Función a verificar
            max_time: Tiempo máximo en segundos
            message: Mensaje personalizado
        """
        execution_time = PerformanceHelper.measure_execution_time(func)
        assert execution_time < max_time, \
            message or f"Execution took {execution_time}s, expected < {max_time}s"


class DataFactory:
    """Factory para crear datos de prueba"""
    
    @staticmethod
    def create_chat_messages(count: int = 1) -> List[Dict[str, Any]]:
        """
        Crea mensajes de chat de prueba
        
        Args:
            count: Número de mensajes a crear
        
        Returns:
            Lista de mensajes
        """
        messages = []
        for i in range(count):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Test message {i}"
            })
        return messages
    
    @staticmethod
    def create_song_requests(count: int = 1, **overrides) -> List[Dict[str, Any]]:
        """
        Crea requests de generación de canciones
        
        Args:
            count: Número de requests
            **overrides: Valores a sobrescribir
        
        Returns:
            Lista de requests
        """
        base_request = {
            "prompt": "A test song",
            "duration": 30,
            "genre": "pop",
            "mood": "happy",
            "user_id": "test-user"
        }
        base_request.update(overrides)
        
        return [base_request.copy() for _ in range(count)]
    
    @staticmethod
    def create_song_ids(count: int = 1) -> List[str]:
        """
        Crea IDs de canciones de prueba
        
        Args:
            count: Número de IDs
        
        Returns:
            Lista de IDs
        """
        import uuid
        return [str(uuid.uuid4()) for _ in range(count)]


@asynccontextmanager
async def async_timeout(timeout: float):
    """
    Context manager para timeout en operaciones async
    
    Args:
        timeout: Tiempo máximo en segundos
    
    Yields:
        None
    """
    try:
        yield
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout}s")


class TestDataBuilder:
    """Builder pattern para crear datos de prueba complejos"""
    
    def __init__(self):
        self.data = {}
    
    def with_prompt(self, prompt: str):
        """Agrega prompt"""
        self.data["prompt"] = prompt
        return self
    
    def with_duration(self, duration: int):
        """Agrega duración"""
        self.data["duration"] = duration
        return self
    
    def with_genre(self, genre: str):
        """Agrega género"""
        self.data["genre"] = genre
        return self
    
    def with_user_id(self, user_id: str):
        """Agrega user_id"""
        self.data["user_id"] = user_id
        return self
    
    def with_chat_history(self, history: List[Dict[str, Any]]):
        """Agrega historial de chat"""
        self.data["chat_history"] = history
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construye el objeto final"""
        return self.data.copy()
    
    @classmethod
    def song_request(cls):
        """Crea un builder para song request"""
        return cls()
    
    @classmethod
    def chat_message(cls):
        """Crea un builder para chat message"""
        return cls()

