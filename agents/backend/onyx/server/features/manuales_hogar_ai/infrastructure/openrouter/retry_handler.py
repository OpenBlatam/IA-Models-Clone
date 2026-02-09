"""
Retry Handler
=============

Manejador especializado para reintentos de llamadas API.
"""

import asyncio
import logging
from typing import Callable, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class RetryHandler:
    """Manejador de reintentos para llamadas API."""
    
    RETRYABLE_STATUS_CODES = [429, 500, 502, 503, 504]
    
    def __init__(self, max_retries: int = 3, base_wait: float = 1.5, max_wait: float = 5.0):
        """
        Inicializar manejador de reintentos.
        
        Args:
            max_retries: Número máximo de reintentos
            base_wait: Tiempo base de espera en segundos
            max_wait: Tiempo máximo de espera en segundos
        """
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.max_wait = max_wait
        self._logger = logger
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecutar función con reintentos.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado de la función
        
        Raises:
            Último error si todos los reintentos fallan
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in self.RETRYABLE_STATUS_CODES and attempt < self.max_retries - 1:
                    wait_time = min((attempt + 1) * self.base_wait, self.max_wait)
                    self._logger.warning(
                        f"Retrying after {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                self._logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise
            
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = min((attempt + 1) * self.base_wait, self.max_wait)
                    self._logger.warning(f"Timeout, retrying after {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
                self._logger.error(f"Timeout: {str(e)}")
                raise
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = min((attempt + 1) * self.base_wait, self.max_wait)
                    await asyncio.sleep(wait_time)
                    continue
                self._logger.error(f"API call failed: {str(e)}")
                raise
        
        if last_error:
            raise last_error

