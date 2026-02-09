"""
Circuit Breaker - Patrón Circuit Breaker para resiliencia
==========================================================
"""

import logging
import time
from typing import Dict, Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker"""
    failure_threshold: int = 5  # Número de fallos para abrir
    success_threshold: int = 2  # Número de éxitos para cerrar desde half-open
    timeout: float = 60.0  # Tiempo en segundos antes de intentar half-open
    expected_exception: type = Exception  # Excepciones que cuentan como fallos


@dataclass
class CircuitBreakerStats:
    """Estadísticas del circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreaker:
    """Circuit Breaker para proteger llamadas a servicios"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.stats = CircuitBreakerStats()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta una función a través del circuit breaker"""
        self.stats.total_requests += 1
        
        # Verificar estado
        if self.state == CircuitState.OPEN:
            # Verificar si debemos intentar half-open
            if self._should_attempt_half_open():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} movido a HALF_OPEN")
            else:
                self.stats.rejected_requests += 1
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        # Ejecutar función
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
        except Exception as e:
            # Otras excepciones no cuentan como fallos
            self._on_success()
            raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta una función async a través del circuit breaker"""
        self.stats.total_requests += 1
        
        # Verificar estado
        if self.state == CircuitState.OPEN:
            if self._should_attempt_half_open():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} movido a HALF_OPEN")
            else:
                self.stats.rejected_requests += 1
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        # Ejecutar función
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
        except Exception as e:
            self._on_success()
            raise e
    
    def _on_success(self):
        """Maneja un éxito"""
        self.stats.successful_requests += 1
        self.stats.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.stats.state_changes += 1
                logger.info(f"Circuit breaker {self.name} cerrado (recuperado)")
        elif self.state == CircuitState.CLOSED:
            # Resetear contador de fallos en caso de éxito
            self.failure_count = 0
    
    def _on_failure(self):
        """Maneja un fallo"""
        self.stats.failed_requests += 1
        self.stats.last_failure_time = datetime.now()
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            # Cualquier fallo en half-open vuelve a open
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.stats.state_changes += 1
            logger.warning(f"Circuit breaker {self.name} abierto desde HALF_OPEN")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.stats.state_changes += 1
                logger.warning(f"Circuit breaker {self.name} abierto (umbral alcanzado)")
    
    def _should_attempt_half_open(self) -> bool:
        """Verifica si debemos intentar half-open"""
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout
    
    def reset(self):
        """Resetea el circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker {self.name} reseteado")
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "stats": {
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "rejected_requests": self.stats.rejected_requests,
                "state_changes": self.stats.state_changes,
                "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
                "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
            }
        }


class CircuitBreakerManager:
    """Gestor de circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Obtiene o crea un circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Obtiene un circuit breaker"""
        return self.breakers.get(name)
    
    def list_breakers(self) -> List[Dict[str, Any]]:
        """Lista todos los circuit breakers"""
        return [breaker.get_state() for breaker in self.breakers.values()]




