"""
Testing Utils - Utilidades de Testing Avanzadas
================================================

Utilidades avanzadas para testing, fixtures, factories y assertions.
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Type
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import string

logger = logging.getLogger(__name__)


class FixtureFactory:
    """
    Factory para crear fixtures de testing.
    """
    
    @staticmethod
    def create_user(
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Crear fixture de usuario"""
        return {
            "id": user_id or f"user_{random.randint(1000, 9999)}",
            "username": username or f"user_{random.randint(100, 999)}",
            "email": email or f"user{random.randint(100, 999)}@example.com",
            "created_at": datetime.now().isoformat(),
            "active": True,
            **kwargs
        }
    
    @staticmethod
    def create_task(
        task_id: Optional[str] = None,
        command: Optional[str] = None,
        status: str = "pending",
        **kwargs
    ) -> Dict[str, Any]:
        """Crear fixture de tarea"""
        return {
            "id": task_id or f"task_{random.randint(1000, 9999)}",
            "command": command or f"echo 'test {random.randint(1, 100)}'",
            "status": status,
            "created_at": datetime.now().isoformat(),
            "priority": 1,
            **kwargs
        }
    
    @staticmethod
    def create_event(
        event_type: str = "test_event",
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Crear fixture de evento"""
        return {
            "type": event_type,
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
            "source": "test",
            **kwargs
        }


class DataGenerator:
    """
    Generador de datos para testing.
    """
    
    @staticmethod
    def random_string(length: int = 10, chars: str = string.ascii_letters + string.digits) -> str:
        """Generar string aleatorio"""
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def random_int(min_val: int = 0, max_val: int = 100) -> int:
        """Generar integer aleatorio"""
        return random.randint(min_val, max_val)
    
    @staticmethod
    def random_float(min_val: float = 0.0, max_val: float = 100.0) -> float:
        """Generar float aleatorio"""
        return random.uniform(min_val, max_val)
    
    @staticmethod
    def random_email() -> str:
        """Generar email aleatorio"""
        return f"{DataGenerator.random_string(8)}@{DataGenerator.random_string(5)}.com"
    
    @staticmethod
    def random_url() -> str:
        """Generar URL aleatoria"""
        return f"https://{DataGenerator.random_string(10)}.example.com/{DataGenerator.random_string(5)}"
    
    @staticmethod
    def random_datetime(
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> datetime:
        """Generar datetime aleatorio"""
        if start is None:
            start = datetime.now() - timedelta(days=365)
        if end is None:
            end = datetime.now()
        
        time_between = end - start
        days_between = time_between.days
        random_days = random.randrange(days_between)
        random_seconds = random.randrange(86400)
        
        return start + timedelta(days=random_days, seconds=random_seconds)
    
    @staticmethod
    def random_list(length: int = 5, generator: Optional[Callable] = None) -> List[Any]:
        """Generar lista aleatoria"""
        generator = generator or (lambda: DataGenerator.random_string())
        return [generator() for _ in range(length)]
    
    @staticmethod
    def random_dict(keys: Optional[List[str]] = None, depth: int = 1) -> Dict[str, Any]:
        """Generar diccionario aleatorio"""
        if keys is None:
            keys = [DataGenerator.random_string(5) for _ in range(3)]
        
        result = {}
        for key in keys:
            if depth > 1 and random.random() > 0.5:
                result[key] = DataGenerator.random_dict(depth=depth - 1)
            else:
                result[key] = DataGenerator.random_string()
        
        return result


class AsyncTestHelper:
    """
    Helper para testing async.
    """
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """
        Esperar hasta que condición sea verdadera.
        
        Args:
            condition: Función que retorna bool
            timeout: Timeout en segundos
            interval: Intervalo de verificación
            
        Returns:
            True si condición se cumplió
        """
        start = time.time()
        while time.time() - start < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    async def wait_for_value(
        getter: Callable[[], Any],
        expected_value: Any,
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """
        Esperar hasta que getter retorne valor esperado.
        
        Args:
            getter: Función que retorna valor
            expected_value: Valor esperado
            timeout: Timeout en segundos
            interval: Intervalo de verificación
            
        Returns:
            True si valor se obtuvo
        """
        return await AsyncTestHelper.wait_for_condition(
            lambda: getter() == expected_value,
            timeout=timeout,
            interval=interval
        )


class AssertionHelper:
    """
    Helper para assertions avanzadas.
    """
    
    @staticmethod
    def assert_dict_contains(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
        """
        Assert que dict1 contiene todas las keys y valores de dict2.
        
        Args:
            dict1: Diccionario que debe contener
            dict2: Diccionario a buscar
        """
        for key, value in dict2.items():
            assert key in dict1, f"Key '{key}' not found in dict1"
            assert dict1[key] == value, f"Value for '{key}' mismatch: {dict1[key]} != {value}"
    
    @staticmethod
    def assert_almost_equal(
        value1: float,
        value2: float,
        delta: float = 0.0001
    ) -> None:
        """
        Assert que dos valores float son casi iguales.
        
        Args:
            value1: Primer valor
            value2: Segundo valor
            delta: Diferencia máxima permitida
        """
        assert abs(value1 - value2) <= delta, f"{value1} != {value2} (delta: {delta})"
    
    @staticmethod
    def assert_in_range(
        value: float,
        min_val: float,
        max_val: float
    ) -> None:
        """
        Assert que valor está en rango.
        
        Args:
            value: Valor a verificar
            min_val: Valor mínimo
            max_val: Valor máximo
        """
        assert min_val <= value <= max_val, f"{value} not in range [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_list_contains(items: List[Any], item: Any) -> None:
        """
        Assert que lista contiene item.
        
        Args:
            items: Lista
            item: Item a buscar
        """
        assert item in items, f"Item {item} not found in list"
    
    @staticmethod
    def assert_list_length(items: List[Any], expected_length: int) -> None:
        """
        Assert que lista tiene longitud esperada.
        
        Args:
            items: Lista
            expected_length: Longitud esperada
        """
        assert len(items) == expected_length, f"List length {len(items)} != {expected_length}"


class MockHelper:
    """
    Helper para crear mocks avanzados.
    """
    
    @staticmethod
    def create_async_mock(return_value: Any = None, side_effect: Any = None) -> AsyncMock:
        """Crear AsyncMock con configuración"""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    @staticmethod
    def create_mock_with_attrs(**attrs) -> Mock:
        """Crear Mock con atributos"""
        mock = Mock()
        for key, value in attrs.items():
            setattr(mock, key, value)
        return mock
    
    @staticmethod
    def patch_multiple(patches: Dict[str, Any]) -> List:
        """Aplicar múltiples patches"""
        return [patch(key, value) for key, value in patches.items()]


class TestTimer:
    """
    Timer para medir tiempo de tests.
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self) -> None:
        """Iniciar timer"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Detener timer y retornar tiempo transcurrido.
        
        Returns:
            Tiempo en segundos
        """
        self.end_time = time.time()
        return self.elapsed
    
    @property
    def elapsed(self) -> float:
        """Obtener tiempo transcurrido"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    def assert_under(self, max_seconds: float) -> None:
        """
        Assert que tiempo está bajo límite.
        
        Args:
            max_seconds: Tiempo máximo en segundos
        """
        elapsed = self.elapsed
        assert elapsed < max_seconds, f"Test took {elapsed}s, expected < {max_seconds}s"




