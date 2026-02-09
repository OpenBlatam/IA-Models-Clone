"""
Circuit Breaker - Patrón Circuit Breaker para Resiliencia
==========================================================

Implementación del patrón Circuit Breaker para prevenir fallos en cascada
en llamadas a servicios externos.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any, TypeVar, Dict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker para prevenir fallos en cascada.
    
    Estados:
    - CLOSED: Funcionando normalmente, permite todas las llamadas
    - OPEN: Demasiados fallos, rechaza todas las llamadas
    - HALF_OPEN: Probando si el servicio se recuperó, permite algunas llamadas
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        name: str = "circuit_breaker"
    ):
        """
        Inicializar circuit breaker.
        
        Args:
            failure_threshold: Número de fallos antes de abrir el circuito
            recovery_timeout: Tiempo en segundos antes de intentar recuperación
            success_threshold: Número de éxitos en HALF_OPEN para cerrar el circuito
            name: Nombre del circuit breaker para logging
        """
        from .constants import (
            DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        )
        
        self.failure_threshold = failure_threshold or DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout or DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self.success_threshold = success_threshold
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        
        self._lock = asyncio.Lock()
    
    def is_open(self) -> bool:
        """Verificar si el circuit breaker está abierto"""
        return self.state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """Verificar si el circuit breaker está cerrado"""
        return self.state == CircuitState.CLOSED
    
    def is_half_open(self) -> bool:
        """Verificar si el circuit breaker está en half-open"""
        return self.state == CircuitState.HALF_OPEN
    
    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Ejecutar función a través del circuit breaker.
        
        Args:
            func: Función async o sync a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si el circuito está abierto o la función falla
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"🔄 Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise Exception(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Last failure: {self.last_failure_time}. "
                        f"Will retry after {self.recovery_timeout}s"
                    )
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self.record_success()
            return result
            
        except Exception as e:
            await self.record_failure()
            raise
    
    async def record_success(self) -> None:
        """Registrar éxito en el circuit breaker"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"✅ Circuit breaker {self.name} CLOSED after recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    async def record_failure(self) -> None:
        """Registrar fallo en el circuit breaker"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now()
                logger.warning(f"⚠️ Circuit breaker {self.name} OPENED (failed during recovery)")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.opened_at = datetime.now()
                    logger.warning(
                        f"⚠️ Circuit breaker {self.name} OPENED after {self.failure_count} failures"
                    )
    
    def _should_attempt_recovery(self) -> bool:
        """Verificar si se debe intentar recuperación"""
        if not self.opened_at:
            return False
        
        elapsed = (datetime.now() - self.opened_at).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """
        Obtener estado actual del circuit breaker.
        
        Returns:
            Diccionario con estado del circuit breaker
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }
    
    def reset(self) -> None:
        """Resetear circuit breaker a estado inicial"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.opened_at = None
        logger.info(f"🔄 Circuit breaker {self.name} reset")

