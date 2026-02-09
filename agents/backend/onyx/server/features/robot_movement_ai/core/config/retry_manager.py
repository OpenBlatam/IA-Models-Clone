"""
Retry Manager System
====================

Sistema avanzado de reintentos con múltiples estrategias.
"""

import logging
import asyncio
import time
import random
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Estrategia de reintento."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """Configuración de reintentos."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    custom_delays: Optional[List[float]] = None


@dataclass
class RetryResult:
    """Resultado de reintento."""
    success: bool
    attempts: int
    total_time: float
    last_error: Optional[Exception] = None
    result: Optional[Any] = None


class RetryManager:
    """
    Gestor de reintentos.
    
    Gestiona reintentos con múltiples estrategias.
    """
    
    def __init__(self):
        """Inicializar gestor de reintentos."""
        self.retry_history: List[Dict[str, Any]] = []
        self.max_history = 10000
    
    def calculate_delay(
        self,
        attempt: int,
        config: RetryConfig
    ) -> float:
        """
        Calcular delay para intento.
        
        Args:
            attempt: Número de intento (1-based)
            config: Configuración
            
        Returns:
            Delay en segundos
        """
        if config.strategy == RetryStrategy.FIXED:
            delay = config.initial_delay
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.initial_delay * attempt
        elif config.strategy == RetryStrategy.CUSTOM:
            if config.custom_delays and attempt <= len(config.custom_delays):
                delay = config.custom_delays[attempt - 1]
            else:
                delay = config.max_delay
        else:
            delay = config.initial_delay
        
        # Limitar a max_delay
        delay = min(delay, config.max_delay)
        
        # Agregar jitter
        if config.jitter:
            jitter_amount = delay * config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)
        
        return delay
    
    async def retry(
        self,
        func: Callable,
        config: Optional[RetryConfig] = None,
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Ejecutar función con reintentos.
        
        Args:
            func: Función a ejecutar
            config: Configuración de reintentos
            *args: Argumentos
            **kwargs: Keyword arguments
            
        Returns:
            Resultado de reintento
        """
        if config is None:
            config = RetryConfig()
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(1, config.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                total_time = time.time() - start_time
                
                retry_result = RetryResult(
                    success=True,
                    attempts=attempt,
                    total_time=total_time,
                    result=result
                )
                
                self._record_retry(func.__name__, attempt, True, total_time)
                logger.info(f"Function {func.__name__} succeeded after {attempt} attempts")
                
                return retry_result
            except Exception as e:
                last_error = e
                
                if attempt < config.max_attempts:
                    delay = self.calculate_delay(attempt, config)
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}/{config.max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    total_time = time.time() - start_time
                    self._record_retry(func.__name__, attempt, False, total_time)
                    logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")
        
        total_time = time.time() - start_time
        return RetryResult(
            success=False,
            attempts=config.max_attempts,
            total_time=total_time,
            last_error=last_error
        )
    
    def _record_retry(
        self,
        function_name: str,
        attempts: int,
        success: bool,
        total_time: float
    ) -> None:
        """Registrar reintento en historial."""
        self.retry_history.append({
            "function_name": function_name,
            "attempts": attempts,
            "success": success,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.retry_history) > self.max_history:
            self.retry_history = self.retry_history[-self.max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de reintentos."""
        if not self.retry_history:
            return {
                "total_retries": 0,
                "successful_retries": 0,
                "failed_retries": 0,
                "average_attempts": 0.0
            }
        
        successful = sum(1 for r in self.retry_history if r["success"])
        failed = len(self.retry_history) - successful
        avg_attempts = sum(r["attempts"] for r in self.retry_history) / len(self.retry_history)
        
        return {
            "total_retries": len(self.retry_history),
            "successful_retries": successful,
            "failed_retries": failed,
            "average_attempts": avg_attempts
        }


# Instancia global
_retry_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """Obtener instancia global del gestor de reintentos."""
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = RetryManager()
    return _retry_manager






