"""
Helpers avanzados para tests complejos de Lovable Community
"""

import asyncio
import time
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from functools import wraps


class AsyncTestHelper:
    """Helper para tests asíncronos"""
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """Espera hasta que una condición sea verdadera"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_retries: int = 3,
        delay: float = 0.1,
        exceptions: tuple = (Exception,)
    ) -> Any:
        """Reintenta una función asíncrona"""
        last_exception = None
        for attempt in range(max_retries):
            try:
                return await func()
            except exceptions as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise last_exception
        return None


class PerformanceHelper:
    """Helper para medir performance en tests"""
    
    @staticmethod
    def measure_time(func: Callable) -> Callable:
        """Decorador para medir tiempo de ejecución"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            wrapper.elapsed_time = elapsed
            return result
        wrapper.elapsed_time = 0.0
        return wrapper
    
    @staticmethod
    def assert_performance(
        elapsed_time: float,
        max_time: float,
        operation: str = "operation"
    ) -> None:
        """Verifica que una operación cumpla con el tiempo máximo"""
        assert elapsed_time <= max_time, \
            f"{operation} took {elapsed_time:.3f}s, expected <= {max_time:.3f}s"


class DataFactory:
    """Factory para generar datos de prueba"""
    
    @staticmethod
    def create_multiple_chats(
        count: int,
        base_title: str = "Chat",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Crea múltiples chats de prueba"""
        return [
            {
                "id": f"chat-{i}",
                "user_id": kwargs.get("user_id", f"user-{i % 5}"),
                "title": f"{base_title} {i}",
                "description": f"Description {i}",
                "chat_content": '{"messages": []}',
                "tags": kwargs.get("tags", ["test"]),
                "vote_count": kwargs.get("vote_count", i),
                "remix_count": kwargs.get("remix_count", 0),
                "view_count": kwargs.get("view_count", i * 10),
                "score": kwargs.get("score", float(i)),
                "is_public": kwargs.get("is_public", True),
                "is_featured": kwargs.get("is_featured", False),
                "created_at": datetime.utcnow() - timedelta(hours=i),
                **kwargs
            }
            for i in range(count)
        ]
    
    @staticmethod
    def create_votes_for_chat(
        chat_id: str,
        count: int,
        upvote_ratio: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Crea múltiples votos para un chat"""
        votes = []
        upvote_count = int(count * upvote_ratio)
        
        for i in range(count):
            vote_type = "upvote" if i < upvote_count else "downvote"
            votes.append({
                "id": f"vote-{i}",
                "chat_id": chat_id,
                "user_id": f"voter-{i}",
                "vote_type": vote_type,
                "created_at": datetime.utcnow() - timedelta(minutes=i)
            })
        
        return votes
    
    @staticmethod
    def create_search_scenarios() -> List[Dict[str, Any]]:
        """Crea escenarios de búsqueda variados"""
        return [
            {"query": "AI", "tags": None, "user_id": None},
            {"query": None, "tags": ["ai", "ml"], "user_id": None},
            {"query": "test", "tags": None, "user_id": "user-123"},
            {"query": None, "tags": None, "user_id": None},  # Listar todos
        ]


class MockVerifier:
    """Helper para verificar llamadas a mocks"""
    
    @staticmethod
    def verify_call_count(
        mock: Mock,
        expected_count: int,
        method_name: str = None
    ) -> None:
        """Verifica el número de llamadas a un mock"""
        if method_name:
            call_count = getattr(mock, method_name).call_count
        else:
            call_count = mock.call_count
        
        assert call_count == expected_count, \
            f"Expected {expected_count} calls, got {call_count}"
    
    @staticmethod
    def verify_call_args(
        mock: Mock,
        expected_args: tuple = None,
        expected_kwargs: dict = None,
        call_index: int = 0
    ) -> None:
        """Verifica los argumentos de una llamada específica"""
        if not mock.called:
            raise AssertionError("Mock was not called")
        
        call = mock.call_args_list[call_index]
        args, kwargs = call
        
        if expected_args:
            assert args == expected_args, \
                f"Args mismatch: expected {expected_args}, got {args}"
        
        if expected_kwargs:
            for key, value in expected_kwargs.items():
                assert key in kwargs, f"Missing kwarg: {key}"
                assert kwargs[key] == value, \
                    f"Kwarg {key} mismatch: expected {value}, got {kwargs[key]}"


class TestDataBuilder:
    """Builder pattern para datos de prueba complejos"""
    
    def __init__(self):
        self.data = {}
    
    def with_id(self, id: str) -> 'TestDataBuilder':
        """Agrega ID"""
        self.data["id"] = id
        return self
    
    def with_user_id(self, user_id: str) -> 'TestDataBuilder':
        """Agrega user_id"""
        self.data["user_id"] = user_id
        return self
    
    def with_title(self, title: str) -> 'TestDataBuilder':
        """Agrega título"""
        self.data["title"] = title
        return self
    
    def with_tags(self, tags: List[str]) -> 'TestDataBuilder':
        """Agrega tags"""
        self.data["tags"] = tags
        return self
    
    def with_engagement(
        self,
        votes: int = 0,
        remixes: int = 0,
        views: int = 0
    ) -> 'TestDataBuilder':
        """Agrega métricas de engagement"""
        self.data["vote_count"] = votes
        self.data["remix_count"] = remixes
        self.data["view_count"] = views
        return self
    
    def with_timestamp(self, days_ago: int = 0) -> 'TestDataBuilder':
        """Agrega timestamp"""
        self.data["created_at"] = datetime.utcnow() - timedelta(days=days_ago)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construye el diccionario final"""
        # Valores por defecto
        defaults = {
            "id": "default-id",
            "user_id": "default-user",
            "title": "Default Title",
            "description": "Default Description",
            "chat_content": '{"messages": []}',
            "tags": None,
            "vote_count": 0,
            "remix_count": 0,
            "view_count": 0,
            "score": 0.0,
            "is_public": True,
            "is_featured": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        defaults.update(self.data)
        return defaults


class SecurityTestHelper:
    """Helper para tests de seguridad"""
    
    @staticmethod
    def generate_sql_injection_payloads() -> List[str]:
        """Genera payloads de SQL injection"""
        return [
            "' OR '1'='1",
            "'; DROP TABLE chats; --",
            "' UNION SELECT * FROM users --",
            "1' OR '1'='1",
            "admin'--",
            "' OR 1=1--",
        ]
    
    @staticmethod
    def generate_xss_payloads() -> List[str]:
        """Genera payloads de XSS"""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'\"><script>alert('XSS')</script>",
        ]
    
    @staticmethod
    def generate_path_traversal_payloads() -> List[str]:
        """Genera payloads de path traversal"""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "....//....//etc/passwd",
        ]
    
    @staticmethod
    def generate_large_inputs() -> List[str]:
        """Genera inputs grandes para DoS"""
        return [
            "A" * 10000,
            "A" * 100000,
            "A" * 1000000,
        ]


class BatchTestHelper:
    """Helper para tests de operaciones en lote"""
    
    @staticmethod
    def create_batch_operation(
        chat_ids: List[str],
        operation: str,
        expected_success: int = None,
        expected_failures: int = None
    ) -> Dict[str, Any]:
        """Crea una operación en lote para testing"""
        return {
            "chat_ids": chat_ids,
            "operation": operation,
            "expected_success": expected_success,
            "expected_failures": expected_failures
        }
    
    @staticmethod
    def verify_batch_result(
        result: Dict[str, Any],
        expected_operation: str,
        min_success: int = 0,
        max_failures: int = None
    ) -> None:
        """Verifica el resultado de una operación en lote"""
        assert result["operation"] == expected_operation
        assert result["successful"] >= min_success
        assert result["failed"] >= 0
        
        if max_failures is not None:
            assert result["failed"] <= max_failures
        
        assert result["total_requested"] == \
            result["successful"] + result["failed"]

