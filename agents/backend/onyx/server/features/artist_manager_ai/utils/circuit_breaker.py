"""
Circuit Breaker
===============

Patrón Circuit Breaker para prevenir cascadas de errores.
"""

import logging
import time
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estado del circuit breaker."""
    CLOSED = "closed"  # Normal
    OPEN = "open"  # Bloqueado
    HALF_OPEN = "half_open"  # Probando


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Inicializar circuit breaker.
        
        Args:
            name: Nombre del circuit breaker
            config: Configuración
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._logger = logger
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecutar función con circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado de la función
        """
        # Verificar estado
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self._logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        # Ejecutar función
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Verificar si se debe intentar resetear."""
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout_seconds
    
    def _on_success(self):
        """Manejar éxito."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self._logger.info(f"Circuit breaker {self.name} CLOSED after success")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Manejar fallo."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._logger.warning(f"Circuit breaker {self.name} OPEN after failure in HALF_OPEN")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._logger.warning(f"Circuit breaker {self.name} OPEN after {self.failure_count} failures")
    
    def reset(self):
        """Resetear circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._logger.info(f"Circuit breaker {self.name} manually reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Obtener estado actual."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }




