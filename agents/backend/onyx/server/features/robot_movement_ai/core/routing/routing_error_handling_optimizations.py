"""
Routing Error Handling Optimizations
=====================================

Optimizaciones de manejo de errores y recuperación.
Incluye: Error recovery, Retry logic, Circuit breakers, etc.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, Tuple
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker para prevenir fallos en cascada."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Inicializar circuit breaker.
        
        Args:
            failure_threshold: Umbral de fallos para abrir
            timeout: Tiempo antes de intentar half-open
            success_threshold: Umbral de éxitos para cerrar
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """
        Ejecutar función con circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            (result, error)
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                # Verificar si debemos intentar half-open
                if self.last_failure_time and (time.time() - self.last_failure_time) >= self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    return None, Exception("Circuit breaker is OPEN")
            
            # Intentar ejecutar
            try:
                result = func(*args, **kwargs)
                
                # Éxito
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        logger.info("Circuit breaker CLOSED after recovery")
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = 0
                
                return result, None
            except Exception as e:
                # Falla
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.OPEN
                    logger.warning("Circuit breaker OPENED after half-open failure")
                elif self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
                
                return None, e


class RetryHandler:
    """Manejador de reintentos."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_backoff: float = 60.0
    ):
        """
        Inicializar manejador de reintentos.
        
        Args:
            max_retries: Número máximo de reintentos
            backoff_factor: Factor de backoff exponencial
            max_backoff: Backoff máximo en segundos
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
    
    def execute(self, func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """
        Ejecutar función con reintentos.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            (result, error)
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function succeeded after {attempt} retries")
                return result, None
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = min(
                        self.backoff_factor * (2 ** attempt),
                        self.max_backoff
                    )
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} after {backoff:.2f}s")
                    time.sleep(backoff)
                else:
                    logger.warning(f"Function failed after {self.max_retries} retries")
        
        return None, last_error


class ErrorRecovery:
    """Sistema de recuperación de errores."""
    
    def __init__(self):
        """Inicializar sistema de recuperación."""
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """
        Registrar estrategia de recuperación.
        
        Args:
            error_type: Tipo de error
            strategy: Función de recuperación
        """
        self.recovery_strategies[error_type] = strategy
    
    def recover(self, error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
        """
        Intentar recuperar de un error.
        
        Args:
            error: Error a recuperar
            context: Contexto adicional
        
        Returns:
            Resultado de recuperación o None
        """
        error_type = type(error).__name__
        
        with self.lock:
            self.error_history.append({
                'error_type': error_type,
                'error_message': str(error),
                'timestamp': time.time(),
                'context': context or {}
            })
        
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de errores."""
        with self.lock:
            if not self.error_history:
                return {}
            
            error_counts = {}
            for entry in self.error_history:
                error_type = entry['error_type']
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            return {
                'total_errors': len(self.error_history),
                'error_types': error_counts,
                'recent_errors': list(self.error_history)[-10:]
            }


class ErrorHandlingOptimizer:
    """Optimizador completo de manejo de errores."""
    
    def __init__(self):
        """Inicializar optimizador de manejo de errores."""
        self.circuit_breaker = CircuitBreaker()
        self.retry_handler = RetryHandler()
        self.error_recovery = ErrorRecovery()
    
    def execute_with_protection(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, Optional[Exception]]:
        """
        Ejecutar función con protección completa.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            (result, error)
        """
        # Intentar con circuit breaker
        result, error = self.circuit_breaker.call(func, *args, **kwargs)
        
        if error:
            # Intentar recuperación
            recovery_result = self.error_recovery.recover(error, {'args': args, 'kwargs': kwargs})
            if recovery_result is not None:
                return recovery_result, None
            
            # Intentar con reintentos
            result, error = self.retry_handler.execute(func, *args, **kwargs)
        
        return result, error
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'circuit_breaker_failures': self.circuit_breaker.failure_count,
            'error_statistics': self.error_recovery.get_error_statistics()
        }

