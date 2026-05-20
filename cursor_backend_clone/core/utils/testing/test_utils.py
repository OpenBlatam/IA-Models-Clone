"""
Test Utils - Utilidades de Testing
===================================

Utilidades para facilitar testing y mocking.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Callable, List
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MockAgent:
    """Mock del agente para testing"""
    
    def __init__(self):
        self.running = False
        self.tasks: Dict[str, Any] = {}
        self.metrics = MockMetrics()
    
    async def start(self):
        """Iniciar agente mock"""
        self.running = True
        logger.debug("Mock agent started")
    
    async def stop(self):
        """Detener agente mock"""
        self.running = False
        logger.debug("Mock agent stopped")
    
    async def add_task(self, command: str, **kwargs) -> str:
        """Agregar tarea mock"""
        task_id = f"task_{len(self.tasks)}"
        self.tasks[task_id] = {
            "id": task_id,
            "command": command,
            "status": "pending",
            **kwargs
        }
        return task_id
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado mock"""
        return {
            "status": "running" if self.running else "stopped",
            "running": self.running,
            "tasks_total": len(self.tasks),
            "tasks_pending": sum(1 for t in self.tasks.values() if t.get("status") == "pending"),
            "tasks_completed": sum(1 for t in self.tasks.values() if t.get("status") == "completed"),
            "tasks_failed": sum(1 for t in self.tasks.values() if t.get("status") == "failed")
        }


class MockMetrics:
    """Mock de métricas para testing"""
    
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
    
    def increment(self, name: str, value: int = 1):
        """Incrementar contador"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float):
        """Establecer gauge"""
        self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """Registrar histograma"""
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)
    
    def get_counter(self, name: str) -> int:
        """Obtener contador"""
        return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Obtener gauge"""
        return self.gauges.get(name)


class MockEventBus:
    """Mock del event bus para testing"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Suscribirse a evento"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def publish(self, event_type: Any, data: Dict[str, Any], source: Optional[str] = None):
        """Publicar evento"""
        event = {
            "type": event_type.value if hasattr(event_type, 'value') else str(event_type),
            "data": data,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
        self.events.append(event)
        
        # Llamar subscribers
        event_type_str = event["type"]
        if event_type_str in self.subscribers:
            for callback in self.subscribers[event_type_str]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}")
    
    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener eventos"""
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events
    
    def clear(self):
        """Limpiar eventos"""
        self.events.clear()


@contextmanager
def mock_time(timestamp: float):
    """
    Mock de tiempo para testing.
    
    Args:
        timestamp: Timestamp a usar
    """
    import time
    original_time = time.time
    
    def mock_time_func():
        return timestamp
    
    time.time = mock_time_func
    try:
        yield
    finally:
        time.time = original_time


@contextmanager
def mock_datetime(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0):
    """
    Mock de datetime para testing.
    
    Args:
        year, month, day, hour, minute, second: Componentes de fecha
    """
    from datetime import datetime as dt
    mock_dt = dt(year, month, day, hour, minute, second)
    
    class MockDateTime:
        @staticmethod
        def now():
            return mock_dt
        
        @staticmethod
        def utcnow():
            return mock_dt
        
        def __getattr__(self, name):
            return getattr(dt, name)
    
    with patch('datetime.datetime', MockDateTime()):
        yield


def create_mock_request(
    method: str = "GET",
    path: str = "/",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None
) -> Mock:
    """
    Crear mock de request HTTP.
    
    Args:
        method: Método HTTP
        path: Path
        headers: Headers
        body: Body
        
    Returns:
        Mock request
    """
    request = Mock()
    request.method = method
    request.url.path = path
    request.headers = headers or {}
    request.body = body
    request.client = Mock()
    request.client.host = "127.0.0.1"
    return request


def create_mock_response(
    status_code: int = 200,
    content: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) -> Mock:
    """
    Crear mock de response HTTP.
    
    Args:
        status_code: Código de estado
        content: Contenido
        headers: Headers
        
    Returns:
        Mock response
    """
    response = Mock()
    response.status_code = status_code
    response.content = content or {}
    response.headers = headers or {}
    return response


async def run_async_test(coro: Callable, timeout: float = 5.0) -> Any:
    """
    Ejecutar test async con timeout.
    
    Args:
        coro: Coroutine a ejecutar
        timeout: Timeout en segundos
        
    Returns:
        Resultado de la coroutine
    """
    return await asyncio.wait_for(coro, timeout=timeout)


def assert_metric_incremented(metrics: MockMetrics, name: str, expected_increment: int = 1):
    """
    Verificar que una métrica se incrementó.
    
    Args:
        metrics: Mock de métricas
        name: Nombre de la métrica
        expected_increment: Incremento esperado
    """
    assert metrics.get_counter(name) >= expected_increment, \
        f"Metric {name} should be at least {expected_increment}"


def assert_event_published(event_bus: MockEventBus, event_type: str, min_count: int = 1):
    """
    Verificar que un evento fue publicado.
    
    Args:
        event_bus: Mock del event bus
        event_type: Tipo de evento
        min_count: Cantidad mínima esperada
    """
    events = event_bus.get_events(event_type)
    assert len(events) >= min_count, \
        f"Expected at least {min_count} events of type {event_type}, got {len(events)}"




