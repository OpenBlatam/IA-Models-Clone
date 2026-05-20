"""
Backoff Strategies
==================
Estrategias de backoff para reintentos.
"""

import asyncio
import random
from typing import Callable, Optional
from enum import Enum


class BackoffStrategy(str, Enum):
    """Estrategias de backoff."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    POLYNOMIAL = "polynomial"


def linear_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Backoff lineal.
    
    Args:
        attempt: Número de intento
        base_delay: Delay base en segundos
        max_delay: Delay máximo en segundos
        
    Returns:
        Delay en segundos
    """
    delay = base_delay * attempt
    return min(delay, max_delay)


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0) -> float:
    """
    Backoff exponencial.
    
    Args:
        attempt: Número de intento
        base_delay: Delay base en segundos
        max_delay: Delay máximo en segundos
        multiplier: Multiplicador exponencial
        
    Returns:
        Delay en segundos
    """
    delay = base_delay * (multiplier ** attempt)
    return min(delay, max_delay)


def fibonacci_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Backoff de Fibonacci.
    
    Args:
        attempt: Número de intento
        base_delay: Delay base en segundos
        max_delay: Delay máximo en segundos
        
    Returns:
        Delay en segundos
    """
    def fib(n: int) -> int:
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    delay = base_delay * fib(attempt + 1)
    return min(delay, max_delay)


def polynomial_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, power: float = 2.0) -> float:
    """
    Backoff polinomial.
    
    Args:
        attempt: Número de intento
        base_delay: Delay base en segundos
        max_delay: Delay máximo en segundos
        power: Potencia del polinomio
        
    Returns:
        Delay en segundos
    """
    delay = base_delay * (attempt ** power)
    return min(delay, max_delay)


def jitter(delay: float, jitter_factor: float = 0.1) -> float:
    """
    Agregar jitter (variación aleatoria) a un delay.
    
    Args:
        delay: Delay base
        jitter_factor: Factor de jitter (0-1)
        
    Returns:
        Delay con jitter
    """
    jitter_amount = delay * jitter_factor * random.random()
    return delay + jitter_amount


async def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_enabled: bool = True,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Reintentar función con backoff.
    
    Args:
        func: Función async a ejecutar
        max_attempts: Número máximo de intentos
        strategy: Estrategia de backoff
        base_delay: Delay base
        max_delay: Delay máximo
        jitter_enabled: Habilitar jitter
        on_retry: Callback en cada reintento
        
    Returns:
        Resultado de la función
        
    Raises:
        Última excepción si todos los intentos fallan
    """
    strategies = {
        BackoffStrategy.LINEAR: linear_backoff,
        BackoffStrategy.EXPONENTIAL: exponential_backoff,
        BackoffStrategy.FIBONACCI: fibonacci_backoff,
        BackoffStrategy.POLYNOMIAL: polynomial_backoff
    }
    
    backoff_func = strategies[strategy]
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                delay = backoff_func(attempt + 1, base_delay, max_delay)
                
                if jitter_enabled:
                    delay = jitter(delay)
                
                if on_retry:
                    on_retry(attempt + 1, e)
                
                await asyncio.sleep(delay)
            else:
                raise
    
    raise last_exception

