"""
Adaptive Retry System - Sistema de retry inteligente con backoff adaptativo.

Características:
- Backoff exponencial adaptativo
- Jitter para evitar thundering herd
- Circuit breaker integration
- Análisis de patrones de error
- Retry policies configurables
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import random
import math

from config.logging_config import get_logger

logger = get_logger(__name__)


class RetryPolicy(str, Enum):
    """Políticas de retry."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class ErrorType(str, Enum):
    """Tipos de error."""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    AUTH_ERROR = "auth_error"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuración de retry."""
    max_retries: int = 3
    initial_delay: float = 1.0  # segundos
    max_delay: float = 60.0  # segundos
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.3  # 30% de variación
    policy: RetryPolicy = RetryPolicy.ADAPTIVE
    retryable_errors: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.TIMEOUT,
        ErrorType.RATE_LIMIT,
        ErrorType.SERVER_ERROR,
        ErrorType.NETWORK_ERROR
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
            "jitter_range": self.jitter_range,
            "policy": self.policy.value,
            "retryable_errors": [e.value for e in self.retryable_errors]
        }


@dataclass
class RetryAttempt:
    """Intento de retry."""
    attempt_number: int
    delay: float
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "attempt_number": self.attempt_number,
            "delay": self.delay,
            "error_type": self.error_type.value if self.error_type else None,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }


class AdaptiveRetry:
    """
    Sistema de retry adaptativo con backoff inteligente.
    
    Características:
    - Backoff exponencial adaptativo
    - Jitter para evitar sincronización
    - Análisis de patrones de error
    - Políticas configurables
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Inicializar sistema de retry.
        
        Args:
            config: Configuración de retry
        """
        self.config = config or RetryConfig()
        
        # Historial de errores por tipo
        self.error_history: Dict[ErrorType, List[datetime]] = {}
        
        # Estadísticas
        self.stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "total_delay_time": 0.0
        }
    
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Clasificar tipo de error.
        
        Args:
            error: Excepción a clasificar
            
        Returns:
            Tipo de error
        """
        error_str = str(error).lower()
        
        if "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            return ErrorType.TIMEOUT
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "401" in error_str or "403" in error_str or "unauthorized" in error_str:
            return ErrorType.AUTH_ERROR
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            return ErrorType.SERVER_ERROR
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        else:
            return ErrorType.UNKNOWN
    
    def is_retryable(self, error: Exception) -> bool:
        """
        Determinar si un error es retryable.
        
        Args:
            error: Excepción a verificar
            
        Returns:
            True si es retryable
        """
        error_type = self.classify_error(error)
        return error_type in self.config.retryable_errors
    
    def calculate_delay(
        self,
        attempt_number: int,
        error_type: Optional[ErrorType] = None
    ) -> float:
        """
        Calcular delay para un intento.
        
        Args:
            attempt_number: Número de intento (1-indexed)
            error_type: Tipo de error (opcional)
            
        Returns:
            Delay en segundos
        """
        if self.config.policy == RetryPolicy.FIXED:
            delay = self.config.initial_delay
        elif self.config.policy == RetryPolicy.LINEAR:
            delay = self.config.initial_delay * attempt_number
        elif self.config.policy == RetryPolicy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(attempt_number, error_type)
        else:  # EXPONENTIAL
            delay = self.config.initial_delay * (
                self.config.exponential_base ** (attempt_number - 1)
            )
        
        # Aplicar max delay
        delay = min(delay, self.config.max_delay)
        
        # Aplicar jitter
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Mínimo 100ms
        
        return delay
    
    def _calculate_adaptive_delay(
        self,
        attempt_number: int,
        error_type: Optional[ErrorType]
    ) -> float:
        """
        Calcular delay adaptativo basado en historial de errores.
        
        Args:
            attempt_number: Número de intento
            error_type: Tipo de error
            
        Returns:
            Delay calculado
        """
        base_delay = self.config.initial_delay * (
            self.config.exponential_base ** (attempt_number - 1)
        )
        
        if not error_type:
            return base_delay
        
        # Ajustar según tipo de error
        if error_type == ErrorType.RATE_LIMIT:
            # Rate limits requieren más tiempo
            base_delay *= 2.0
        elif error_type == ErrorType.TIMEOUT:
            # Timeouts pueden ser temporales
            base_delay *= 1.5
        elif error_type == ErrorType.SERVER_ERROR:
            # Server errors pueden requerir más tiempo
            base_delay *= 1.8
        
        # Ajustar según frecuencia de errores recientes
        if error_type in self.error_history:
            recent_errors = [
                e for e in self.error_history[error_type]
                if (datetime.now() - e).total_seconds() < 60
            ]
            
            if len(recent_errors) > 5:
                # Muchos errores recientes, aumentar delay
                base_delay *= 1.5
        
        return base_delay
    
    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[Any]],
        context: Optional[str] = None
    ) -> Any:
        """
        Ejecutar función con retry automático.
        
        Args:
            func: Función async a ejecutar
            context: Contexto para logging (opcional)
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si todos los retries fallan
        """
        last_error = None
        attempts = []
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                result = await func()
                
                # Éxito
                if attempt > 1:
                    self.stats["successful_retries"] += 1
                    logger.info(
                        f"Retry exitoso en intento {attempt}"
                        + (f" ({context})" if context else "")
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)
                
                # Registrar error
                if error_type not in self.error_history:
                    self.error_history[error_type] = []
                self.error_history[error_type].append(datetime.now())
                
                # Limpiar historial antiguo (últimos 1000)
                if len(self.error_history[error_type]) > 1000:
                    self.error_history[error_type] = (
                        self.error_history[error_type][-1000:]
                    )
                
                # Verificar si es retryable
                if not self.is_retryable(e) or attempt >= self.config.max_retries:
                    # No retry o último intento
                    if attempt >= self.config.max_retries:
                        self.stats["failed_retries"] += 1
                        logger.error(
                            f"Todos los retries fallaron ({attempt} intentos)"
                            + (f" ({context})" if context else "")
                        )
                    raise
                
                # Calcular delay
                delay = self.calculate_delay(attempt, error_type)
                
                # Registrar intento
                retry_attempt = RetryAttempt(
                    attempt_number=attempt,
                    delay=delay,
                    error_type=error_type,
                    error_message=str(e)
                )
                attempts.append(retry_attempt)
                
                self.stats["total_retries"] += 1
                self.stats["total_delay_time"] += delay
                
                logger.warning(
                    f"Retry {attempt}/{self.config.max_retries} después de {delay:.2f}s "
                    f"(error: {error_type.value})"
                    + (f" ({context})" if context else "")
                )
                
                # Esperar antes de retry
                await asyncio.sleep(delay)
        
        # No debería llegar aquí, pero por si acaso
        raise last_error
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de retry."""
        return {
            **self.stats,
            "avg_delay_time": (
                self.stats["total_delay_time"] / self.stats["total_retries"]
                if self.stats["total_retries"] > 0 else 0.0
            ),
            "success_rate": (
                self.stats["successful_retries"] / self.stats["total_retries"]
                if self.stats["total_retries"] > 0 else 0.0
            ),
            "error_distribution": {
                error_type.value: len(errors)
                for error_type, errors in self.error_history.items()
            }
        }
    
    def reset_stats(self):
        """Resetear estadísticas."""
        self.stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "total_delay_time": 0.0
        }
        self.error_history.clear()


def get_adaptive_retry(config: Optional[RetryConfig] = None) -> AdaptiveRetry:
    """Factory function para obtener instancia del sistema de retry."""
    return AdaptiveRetry(config)



