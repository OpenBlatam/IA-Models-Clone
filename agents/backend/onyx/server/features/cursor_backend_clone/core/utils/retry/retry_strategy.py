"""
Retry Strategy - Estrategias avanzadas de reintento
===================================================

Sistema mejorado de reintentos con backoff adaptativo, múltiples estrategias,
y métricas integradas.
"""

import asyncio
import functools
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Estrategias de backoff para reintentos"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


class RetryResult:
    """Resultado de una operación con reintentos"""
    
    def __init__(
        self,
        success: bool,
        result: Any = None,
        attempts: int = 0,
        total_time: float = 0.0,
        last_exception: Optional[Exception] = None
    ):
        self.success = success
        self.result = result
        self.attempts = attempts
        self.total_time = total_time
        self.last_exception = last_exception
        self.attempt_times: list[float] = []


def calculate_backoff(
    attempt: int,
    base_delay: float,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    max_delay: float = 300.0,
    jitter: bool = True
) -> float:
    """
    Calcular delay para el siguiente intento según la estrategia.
    
    Args:
        attempt: Número de intento (0-indexed)
        base_delay: Delay base en segundos
        strategy: Estrategia de backoff
        max_delay: Delay máximo permitido
        jitter: Si agregar jitter aleatorio para evitar thundering herd
        
    Returns:
        Delay en segundos
    """
    if strategy == RetryStrategy.EXPONENTIAL:
        delay = base_delay * (2 ** attempt)
    elif strategy == RetryStrategy.LINEAR:
        delay = base_delay * (attempt + 1)
    elif strategy == RetryStrategy.CONSTANT:
        delay = base_delay
    elif strategy == RetryStrategy.FIBONACCI:
        # Fibonacci: 1, 1, 2, 3, 5, 8, 13, ...
        fib = [1, 1]
        for i in range(2, attempt + 2):
            fib.append(fib[i-1] + fib[i-2])
        delay = base_delay * fib[min(attempt, len(fib) - 1)]
    else:
        delay = base_delay * (2 ** attempt)
    
    # Aplicar límite máximo
    delay = min(delay, max_delay)
    
    # Agregar jitter (hasta 25% del delay)
    if jitter:
        jitter_amount = delay * 0.25 * random.random()
        delay = delay + jitter_amount
    
    return delay


def retry_with_strategy(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter: bool = True,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    log_retries: bool = True
):
    """
    Decorador avanzado para reintentos con estrategias configurables.
    
    Args:
        max_attempts: Número máximo de intentos
        base_delay: Delay base en segundos
        max_delay: Delay máximo permitido
        strategy: Estrategia de backoff
        jitter: Si agregar jitter aleatorio
        exceptions: Tupla de excepciones que deben causar reintento
        on_retry: Callback opcional llamado en cada reintento (attempt, exception)
        log_retries: Si registrar logs de reintentos
        
    Example:
        @retry_with_strategy(
            max_attempts=5,
            base_delay=2.0,
            strategy=RetryStrategy.EXPONENTIAL,
            exceptions=(ConnectionError, TimeoutError)
        )
        async def fetch_data():
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            attempt_times = []
            
            for attempt in range(max_attempts):
                attempt_start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    attempt_time = time.time() - attempt_start
                    attempt_times.append(attempt_time)
                    
                    if attempt > 0 and log_retries:
                        logger.info(
                            f"✅ {func.__name__} succeeded after {attempt + 1} attempts "
                            f"(total time: {sum(attempt_times):.2f}s)"
                        )
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    attempt_time = time.time() - attempt_start
                    attempt_times.append(attempt_time)
                    
                    if attempt < max_attempts - 1:
                        delay = calculate_backoff(
                            attempt,
                            base_delay,
                            strategy,
                            max_delay,
                            jitter
                        )
                        
                        if log_retries:
                            logger.warning(
                                f"⚠️ {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): "
                                f"{type(e).__name__}: {str(e)[:100]}. "
                                f"Retrying in {delay:.2f}s..."
                            )
                        
                        if on_retry:
                            try:
                                if asyncio.iscoroutinefunction(on_retry):
                                    await on_retry(attempt + 1, e)
                                else:
                                    on_retry(attempt + 1, e)
                            except Exception as callback_error:
                                logger.error(f"Error in retry callback: {callback_error}")
                        
                        await asyncio.sleep(delay)
                    else:
                        if log_retries:
                            logger.error(
                                f"❌ {func.__name__} failed after {max_attempts} attempts. "
                                f"Total time: {sum(attempt_times):.2f}s"
                            )
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class AdaptiveRetryStrategy:
    """
    Estrategia de reintento adaptativa que ajusta el backoff basado en
    el historial de éxitos/fallos.
    """
    
    def __init__(
        self,
        initial_base_delay: float = 1.0,
        min_delay: float = 0.1,
        max_delay: float = 300.0,
        success_factor: float = 0.8,
        failure_factor: float = 1.5
    ):
        self.base_delay = initial_base_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.success_factor = success_factor
        self.failure_factor = failure_factor
        self.recent_successes = 0
        self.recent_failures = 0
        self._lock = asyncio.Lock()
    
    async def get_delay(self, attempt: int) -> float:
        """Obtener delay para el siguiente intento"""
        async with self._lock:
            # Calcular delay base con backoff exponencial
            delay = self.base_delay * (2 ** attempt)
            delay = min(max(delay, self.min_delay), self.max_delay)
            
            # Agregar jitter
            jitter = delay * 0.25 * random.random()
            return delay + jitter
    
    async def record_success(self):
        """Registrar éxito y ajustar estrategia"""
        async with self._lock:
            self.recent_successes += 1
            if self.recent_successes >= 3:
                # Reducir delay base si hay muchos éxitos
                self.base_delay = max(
                    self.min_delay,
                    self.base_delay * self.success_factor
                )
                self.recent_successes = 0
    
    async def record_failure(self):
        """Registrar fallo y ajustar estrategia"""
        async with self._lock:
            self.recent_failures += 1
            if self.recent_failures >= 3:
                # Aumentar delay base si hay muchos fallos
                self.base_delay = min(
                    self.max_delay,
                    self.base_delay * self.failure_factor
                )
                self.recent_failures = 0




