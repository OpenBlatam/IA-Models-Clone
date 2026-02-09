"""
Circuit Breaker - Architecture Improved
=======================================

Implementación mejorada del patrón Circuit Breaker siguiendo
la nueva arquitectura con Domain-Driven Design.
"""

import time
import asyncio
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
import logging

from .error_handling import (
    InfrastructureError,
    ErrorCode,
    ErrorContext,
    ErrorSeverity
)
from .domain_improved import Entity, DomainEvent

logger = logging.getLogger(__name__)


# ============================================================================
# Domain Layer - Value Objects and Entities
# ============================================================================

class CircuitState(Enum):
    """Estado del circuit breaker."""
    CLOSED = "closed"  # Operación normal
    OPEN = "open"  # Falla detectada, rechazar requests
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuración del circuit breaker (Value Object)."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    monitoring_window: float = 300.0
    call_timeout: Optional[float] = None
    enable_adaptive_timeout: bool = False
    min_timeout: float = 10.0
    max_timeout: float = 300.0
    timeout_multiplier: float = 2.0
    expected_exception: type = Exception  # Excepciones que cuentan como fallos
    
    def __post_init__(self):
        """Validar configuración."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold debe ser >= 1")
        if self.recovery_timeout < 0:
            raise ValueError("recovery_timeout debe ser >= 0")
        if self.success_threshold < 1:
            raise ValueError("success_threshold debe ser >= 1")


@dataclass
class CircuitBreakerMetrics:
    """Métricas del circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_state_change: Optional[datetime] = None
    current_failure_count: int = 0
    current_success_count: int = 0
    failure_timestamps: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calcular tasa de éxito."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calcular tasa de fallo."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "state_changes": self.state_changes,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
            "current_failure_count": self.current_failure_count,
            "current_success_count": self.current_success_count
        }


# ============================================================================
# Domain Events
# ============================================================================

class CircuitBreakerOpenedEvent(DomainEvent):
    """Evento: Circuit breaker abierto."""
    
    def __init__(self, circuit_name: str, failure_count: int, **kwargs):
        super().__init__(**kwargs)
        self.circuit_name = circuit_name
        self.failure_count = failure_count


class CircuitBreakerClosedEvent(DomainEvent):
    """Evento: Circuit breaker cerrado."""
    
    def __init__(self, circuit_name: str, **kwargs):
        super().__init__(**kwargs)
        self.circuit_name = circuit_name


class CircuitBreakerHalfOpenEvent(DomainEvent):
    """Evento: Circuit breaker en half-open."""
    
    def __init__(self, circuit_name: str, **kwargs):
        super().__init__(**kwargs)
        self.circuit_name = circuit_name


# ============================================================================
# Domain Entity - Circuit Breaker
# ============================================================================

class CircuitBreaker(Entity):
    """
    Circuit Breaker como entidad de dominio.
    
    Implementa el patrón Circuit Breaker para proteger servicios
    contra fallos en cascada.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        circuit_id: Optional[str] = None,
        # Backward compatibility - parámetros individuales
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[float] = None,
        expected_exception: Optional[type] = None
    ):
        """
        Inicializar circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
            config: Configuración (tiene precedencia sobre parámetros individuales)
            circuit_id: ID del circuit (se genera si no se proporciona)
            failure_threshold: Threshold de fallos (backward compatibility)
            recovery_timeout: Timeout de recuperación (backward compatibility)
            expected_exception: Excepción esperada (backward compatibility)
        """
        super().__init__(circuit_id)
        self._name = name
        
        # Soporte para backward compatibility
        if config is None:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold or 5,
                recovery_timeout=recovery_timeout or 60.0,
                expected_exception=expected_exception or Exception
            )
        
        self._config = config
        self._state = CircuitState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._last_failure_time: Optional[float] = None
        self._current_timeout = config.recovery_timeout
        self._lock = asyncio.Lock()
        
        # State callbacks para notificaciones
        self._state_callbacks: Dict[CircuitState, List[Callable]] = {
            state: [] for state in CircuitState
        }
    
    @property
    def name(self) -> str:
        """Obtener nombre."""
        return self._name
    
    @property
    def state(self) -> CircuitState:
        """Obtener estado actual."""
        return self._state
    
    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Obtener métricas."""
        return self._metrics
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar función protegida por circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos de la función
            **kwargs: Keyword arguments de la función
            
        Returns:
            Resultado de la función
            
        Raises:
            InfrastructureError: Si el circuit está abierto
        """
        # Verificar si se permite la llamada (fuera del lock para mejor performance)
        if not await self._should_allow_call():
            self._metrics.rejected_requests += 1
            self._metrics.total_requests += 1
            remaining_time = self._get_remaining_timeout()
            raise InfrastructureError(
                f"Circuit breaker '{self._name}' is {self._state.value}. Recovery in {remaining_time:.1f}s",
                ErrorCode.INFRASTRUCTURE_SERVICE_UNAVAILABLE,
                context=ErrorContext(
                    operation=f"circuit_breaker.{self._name}",
                    metadata={
                        "state": self._state.value,
                        "failure_count": self._metrics.current_failure_count,
                        "remaining_timeout": remaining_time
                    }
                )
            )
        
        # Ejecutar función con timeout si está configurado
        try:
            if self._config.call_timeout:
                result = await asyncio.wait_for(
                    self._execute_function(func, *args, **kwargs),
                    timeout=self._config.call_timeout
                )
            else:
                result = await self._execute_function(func, *args, **kwargs)
            
            # Éxito
            await self._on_success()
            return result
        
        except asyncio.TimeoutError:
            await self._on_failure()
            raise InfrastructureError(
                f"Circuit breaker '{self._name}' call timed out after {self._config.call_timeout}s",
                ErrorCode.INFRASTRUCTURE_TIMEOUT,
                context=ErrorContext(
                    operation=f"circuit_breaker.{self._name}",
                    metadata={"timeout": self._config.call_timeout}
                )
            )
        
        except self._config.expected_exception as e:
            # Solo contar como fallo si es la excepción esperada
            await self._on_failure()
            logger.warning(
                f"Circuit breaker '{self._name}' recorded failure: {e}",
                extra={
                    "state": self._state.value,
                    "failure_count": self._metrics.current_failure_count
                }
            )
            raise
        
        except Exception as e:
            # Otras excepciones no cuentan como fallos si expected_exception está configurado
            if self._config.expected_exception == Exception:
                await self._on_failure()
            raise
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecutar función (sync o async)."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Ejecutar función sync en thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    async def _should_allow_call(self) -> bool:
        """
        Determinar si se permite la llamada.
        
        Returns:
            True si se permite, False si se rechaza
        """
        async with self._lock:
            current_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Verificar si el timeout de recuperación ha pasado
                if self._last_failure_time and (current_time - self._last_failure_time >= self._current_timeout):
                    await self._transition_to_half_open()
                    return True
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    def _get_remaining_timeout(self) -> float:
        """
        Obtener tiempo restante antes de intentar reset.
        
        Returns:
            Tiempo restante en segundos
        """
        if self._last_failure_time is None:
            return 0.0
        
        current_time = time.time()
        elapsed = current_time - self._last_failure_time
        remaining = max(0.0, self._current_timeout - elapsed)
        return remaining
    
    async def _on_success(self):
        """Manejar llamada exitosa."""
        async with self._lock:
            self._metrics.total_requests += 1
            self._metrics.successful_requests += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._metrics.current_success_count += 1
                if self._metrics.current_success_count >= self._config.success_threshold:
                    await self._transition_to_closed()
            else:
                # Resetear contador de fallos en sliding window
                self._cleanup_old_failures()
    
    async def _on_failure(self):
        """Manejar llamada fallida."""
        async with self._lock:
            self._metrics.total_requests += 1
            self._metrics.failed_requests += 1
            current_time = time.time()
            
            # Track failure time para sliding window
            self._metrics.failure_timestamps.append(current_time)
            self._last_failure_time = current_time
            
            # Limpiar fallos antiguos fuera del monitoring window
            self._cleanup_old_failures()
            
            # Verificar si debemos abrir el circuit
            if self._state == CircuitState.HALF_OPEN:
                # Cualquier fallo en half-open vuelve a abrir
                await self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                # Verificar threshold
                if self._metrics.current_failure_count >= self._config.failure_threshold:
                    await self._transition_to_open()
    
    def _cleanup_old_failures(self):
        """Limpiar fallos fuera del window."""
        current_time = time.time()
        window_start = current_time - self._config.monitoring_window
        
        # Filtrar fallos dentro del window
        self._metrics.failure_timestamps = [
            ts for ts in self._metrics.failure_timestamps
            if ts >= window_start
        ]
        
        # Actualizar contador
        self._metrics.current_failure_count = len(self._metrics.failure_timestamps)
    
    
    async def _transition_to_open(self):
        """Transición a estado OPEN."""
        if self._state != CircuitState.OPEN:
            old_state = self._state
            self._state = CircuitState.OPEN
            self._metrics.state_changes += 1
            self._metrics.last_state_change = datetime.now()
            self._metrics.current_success_count = 0
            
            # Actualizar adaptive timeout si está habilitado
            if self._config.enable_adaptive_timeout:
                self._update_adaptive_timeout()
            
            # Emitir evento
            self.add_domain_event(
                CircuitBreakerOpenedEvent(
                    circuit_name=self._name,
                    failure_count=self._metrics.current_failure_count
                )
            )
            
            # Invocar callbacks
            self._invoke_callbacks(CircuitState.OPEN, old_state)
            
            logger.warning(
                f"Circuit breaker '{self._name}' OPENED after {self._metrics.current_failure_count} failures"
            )
    
    async def _transition_to_half_open(self):
        """Transición a estado HALF_OPEN."""
        if self._state != CircuitState.HALF_OPEN:
            old_state = self._state
            self._state = CircuitState.HALF_OPEN
            self._metrics.state_changes += 1
            self._metrics.last_state_change = datetime.now()
            self._metrics.current_success_count = 0
            
            # Emitir evento
            self.add_domain_event(
                CircuitBreakerHalfOpenEvent(circuit_name=self._name)
            )
            
            # Invocar callbacks
            self._invoke_callbacks(CircuitState.HALF_OPEN, old_state)
            
            logger.info(f"Circuit breaker '{self._name}' transitioning to HALF_OPEN")
    
    async def _transition_to_closed(self):
        """Transición a estado CLOSED."""
        if self._state != CircuitState.CLOSED:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._metrics.state_changes += 1
            self._metrics.last_state_change = datetime.now()
            self._metrics.current_failure_count = 0
            self._metrics.current_success_count = 0
            self._metrics.failure_timestamps.clear()
            
            # Resetear timeout adaptativo
            if self._config.enable_adaptive_timeout:
                self._current_timeout = self._config.recovery_timeout
            
            # Emitir evento
            self.add_domain_event(
                CircuitBreakerClosedEvent(circuit_name=self._name)
            )
            
            # Invocar callbacks
            self._invoke_callbacks(CircuitState.CLOSED, old_state)
            
            logger.info(f"Circuit breaker '{self._name}' CLOSED after recovery")
    
    def _update_adaptive_timeout(self):
        """Actualizar timeout usando algoritmo adaptativo."""
        if self._config.enable_adaptive_timeout:
            new_timeout = min(
                self._config.max_timeout,
                max(
                    self._config.min_timeout,
                    self._current_timeout * self._config.timeout_multiplier
                )
            )
            self._current_timeout = new_timeout
    
    def register_callback(self, state: CircuitState, callback: Callable):
        """
        Registrar callback para cambio de estado.
        
        Args:
            state: Estado para el cual registrar callback
            callback: Función callback (async o sync)
        """
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)
    
    def _invoke_callbacks(self, new_state: CircuitState, old_state: CircuitState):
        """Invocar callbacks para cambio de estado."""
        callbacks = self._state_callbacks.get(new_state, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Callback async - crear task
                    asyncio.create_task(callback(self, new_state, old_state))
                else:
                    # Callback sync
                    callback(self, new_state, old_state)
            except Exception as e:
                logger.error(f"Error invoking callback for {new_state.value}: {e}")
    
    async def reset(self):
        """Resetear circuit breaker manualmente."""
        await self._transition_to_closed()
        logger.info(f"Circuit breaker '{self._name}' manually reset")
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Obtener información del estado actual.
        
        Returns:
            Diccionario con información del estado
        """
        return {
            "name": self._name,
            "state": self._state.value,
            "metrics": self._metrics.to_dict(),
            "remaining_timeout": self._get_remaining_timeout(),
            "current_timeout": self._current_timeout,
            "config": {
                "failure_threshold": self._config.failure_threshold,
                "recovery_timeout": self._config.recovery_timeout,
                "success_threshold": self._config.success_threshold,
                "monitoring_window": self._config.monitoring_window,
                "enable_adaptive_timeout": self._config.enable_adaptive_timeout
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "name": self._name,
            "state": self._state.value,
            "metrics": self._metrics.to_dict(),
            "config": {
                "failure_threshold": self._config.failure_threshold,
                "recovery_timeout": self._config.recovery_timeout,
                "success_threshold": self._config.success_threshold
            }
        }


# ============================================================================
# Circuit Breaker Manager (Application Service)
# ============================================================================

class CircuitBreakerManager:
    """
    Manager para circuit breakers.
    
    Gestiona múltiples circuit breakers y permite acceso centralizado.
    """
    
    def __init__(self):
        """Inicializar manager."""
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Obtener o crear circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
            config: Configuración (usa default si no se proporciona)
            
        Returns:
            Circuit breaker
        """
        async with self._lock:
            if name not in self._circuits:
                if config is None:
                    config = CircuitBreakerConfig()
                self._circuits[name] = CircuitBreaker(name=name, config=config)
                logger.info(f"Circuit breaker '{name}' creado")
            
            return self._circuits[name]
    
    async def get(self, name: str) -> Optional[CircuitBreaker]:
        """Obtener circuit breaker por nombre."""
        async with self._lock:
            return self._circuits.get(name)
    
    async def list_all(self) -> List[CircuitBreaker]:
        """Listar todos los circuit breakers."""
        async with self._lock:
            return list(self._circuits.values())
    
    async def reset(self, name: str):
        """Resetear circuit breaker."""
        circuit = await self.get(name)
        if circuit:
            circuit.reset()
        else:
            raise ValueError(f"Circuit breaker '{name}' no encontrado")
    
    async def get_metrics(self, name: str) -> Optional[CircuitBreakerMetrics]:
        """Obtener métricas de circuit breaker."""
        circuit = await self.get(name)
        return circuit.metrics if circuit else None


# ============================================================================
# Decorator
# ============================================================================

def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    manager: Optional[CircuitBreakerManager] = None
):
    """
    Decorator para aplicar circuit breaker a función.
    
    Args:
        name: Nombre del circuit breaker
        config: Configuración
        manager: Manager de circuit breakers (opcional)
        
    Returns:
        Decorator function
    """
    _manager = manager or CircuitBreakerManager()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            circuit = await _manager.get_or_create(name, config)
            return await circuit.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================================
# Global Instance
# ============================================================================

_global_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Obtener instancia global del manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = CircuitBreakerManager()
    return _global_manager

