"""
Circuit Breaker - Sistema de Circuit Breaker
===========================================

Sistema de circuit breaker para proteger servicios externos.
"""

import asyncio
import logging
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estado del circuit breaker."""
    CLOSED = "closed"  # Normal, permitir solicitudes
    OPEN = "open"  # Fallos, rechazar solicitudes
    HALF_OPEN = "half_open"  # Probando recuperación


@dataclass
class CircuitBreakerConfig:
    """Configuración de circuit breaker."""
    failure_threshold: int = 5  # Fallos para abrir
    success_threshold: int = 2  # Éxitos para cerrar
    timeout_seconds: float = 60.0  # Tiempo antes de intentar recuperación
    timeout_window: float = 60.0  # Ventana de tiempo para contar fallos


@dataclass
class CircuitBreakerState:
    """Estado del circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure: Optional[datetime] = None
    opened_at: Optional[datetime] = None


class CircuitBreaker:
    """Circuit breaker."""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.states: Dict[str, CircuitBreakerState] = {}
        self._lock = asyncio.Lock()
    
    async def call(
        self,
        identifier: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Ejecutar función con circuit breaker.
        
        Args:
            identifier: Identificador único del circuit breaker
            func: Función a ejecutar
            *args: Argumentos para la función
            **kwargs: Argumentos keyword para la función
        
        Returns:
            Resultado de la función
        
        Raises:
            Exception: Si el circuit breaker está abierto o la función falla
        """
        state = await self._get_state(identifier)
        
        # Verificar estado
        if state.state == CircuitState.OPEN:
            # Verificar si debemos intentar recuperación
            if state.opened_at:
                time_since_open = (datetime.now() - state.opened_at).total_seconds()
                if time_since_open >= self.config.timeout_seconds:
                    state.state = CircuitState.HALF_OPEN
                    state.successes = 0
                    logger.info(f"Circuit breaker {identifier} entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit breaker {identifier} is OPEN")
        
        # Ejecutar función
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Éxito
            await self._record_success(identifier)
            return result
            
        except Exception as e:
            # Falla
            await self._record_failure(identifier)
            raise
    
    async def _get_state(self, identifier: str) -> CircuitBreakerState:
        """Obtener estado del circuit breaker."""
        async with self._lock:
            if identifier not in self.states:
                self.states[identifier] = CircuitBreakerState()
            return self.states[identifier]
    
    async def _record_success(self, identifier: str):
        """Registrar éxito."""
        async with self._lock:
            state = self.states[identifier]
            
            if state.state == CircuitState.HALF_OPEN:
                state.successes += 1
                if state.successes >= self.config.success_threshold:
                    state.state = CircuitState.CLOSED
                    state.failures = 0
                    state.successes = 0
                    logger.info(f"Circuit breaker {identifier} CLOSED after recovery")
            elif state.state == CircuitState.CLOSED:
                # Reset contador de fallos en ventana de tiempo
                state.failures = 0
    
    async def _record_failure(self, identifier: str):
        """Registrar fallo."""
        async with self._lock:
            state = self.states[identifier]
            state.failures += 1
            state.last_failure = datetime.now()
            
            if state.state == CircuitState.CLOSED:
                if state.failures >= self.config.failure_threshold:
                    state.state = CircuitState.OPEN
                    state.opened_at = datetime.now()
                    logger.warning(f"Circuit breaker {identifier} OPENED after {state.failures} failures")
            elif state.state == CircuitState.HALF_OPEN:
                # Cualquier fallo en HALF_OPEN vuelve a OPEN
                state.state = CircuitState.OPEN
                state.opened_at = datetime.now()
                state.successes = 0
                logger.warning(f"Circuit breaker {identifier} re-OPENED")
    
    async def get_state(self, identifier: str) -> Dict:
        """Obtener estado del circuit breaker."""
        state = await self._get_state(identifier)
        
        return {
            "identifier": identifier,
            "state": state.state.value,
            "failures": state.failures,
            "successes": state.successes,
            "last_failure": state.last_failure.isoformat() if state.last_failure else None,
            "opened_at": state.opened_at.isoformat() if state.opened_at else None,
        }
    
    async def reset(self, identifier: str):
        """Resetear circuit breaker."""
        async with self._lock:
            if identifier in self.states:
                self.states[identifier] = CircuitBreakerState()
                logger.info(f"Circuit breaker {identifier} reset")
















