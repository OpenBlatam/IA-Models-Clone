"""
Retry Handler - Lógica de reintentos
"""

from typing import Callable, Any
import asyncio
from functools import wraps


class RetryHandler:
    """Manejador de reintentos para peticiones HTTP"""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        """Inicializa el manejador de reintentos"""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta una función con reintentos"""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise last_exception
        raise last_exception

