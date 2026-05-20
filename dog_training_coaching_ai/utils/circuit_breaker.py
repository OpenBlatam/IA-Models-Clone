"""
Circuit Breaker
===============
Implementación de circuit breaker pattern.
"""

from typing import Callable, Optional, Dict
from enum import Enum
from datetime import datetime, timedelta
import asyncio


class CircuitState(str, Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker para proteger servicios."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Inicializar circuit breaker.
        
        Args:
            failure_threshold: Número de fallos para abrir
            success_threshold: Número de éxitos para cerrar (half-open)
            timeout: Tiempo en segundos antes de intentar half-open
            expected_exception: Tipo de excepción esperada
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Ejecutar función con circuit breaker.
        
        Args:
            func: Función async a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
            
        Returns:
            Resultado de la función
            
        Raises:
            CircuitBreakerOpenError: Si el circuit está abierto
            Exception: Excepción de la función
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                # Solo permitir un intento a la vez
                pass
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Manejar éxito."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    async def _on_failure(self):
        """Manejar fallo."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Verificar si se debe intentar reset."""
        if self.last_failure_time is None:
            return False
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def get_state(self) -> Dict:
        """Obtener estado del circuit breaker."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }
    
    def reset(self):
        """Resetear circuit breaker manualmente."""
        async def _reset():
            async with self._lock:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.last_failure_time = None
        
        asyncio.create_task(_reset())


class CircuitBreakerOpenError(Exception):
    """Excepción cuando el circuit breaker está abierto."""
    pass

