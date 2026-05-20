"""
Robust Service - Servicio robusto con validación y resiliencia
==============================================================

Servicio base mejorado con validación robusta, timeouts, y manejo de errores avanzado.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta

from .base_service import BaseService
from .exceptions import ServiceUnavailableError, ValidationError
from .decorators import retry_on_failure
from ..circuit_breaker import get_circuit_breaker
from ..retry_logic import RetryConfig

logger = logging.getLogger(__name__)


class RobustService(BaseService):
    """
    Servicio robusto con características avanzadas:
    - Validación robusta
    - Timeouts
    - Circuit breakers
    - Retry logic
    - Health checks
    - Fallbacks
    """
    
    def __init__(
        self,
        cache_service=None,
        event_publisher=None,
        service_name=None,
        timeout: float = 30.0,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True
    ):
        super().__init__(
            cache_service=cache_service,
            event_publisher=event_publisher,
            service_name=service_name
        )
        self.timeout = timeout
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_retry = enable_retry
        self._circuit_breaker = None
        self._retry_config = None
        
        if enable_circuit_breaker:
            self._circuit_breaker = get_circuit_breaker(
                name=service_name or "default",
                failure_threshold=5,
                timeout=60
            )
        
        if enable_retry:
            self._retry_config = RetryConfig(
                max_attempts=3,
                backoff_factor=2.0,
                initial_delay=1.0
            )
    
    async def _execute_with_timeout(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta función con timeout.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            **kwargs: Keyword arguments
        
        Returns:
            Resultado de la función
        
        Raises:
            asyncio.TimeoutError: Si excede timeout
        """
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"{self.service_name} operation timed out after {self.timeout}s")
            raise ServiceUnavailableError(
                f"Service {self.service_name} operation timed out"
            )
    
    async def _execute_with_circuit_breaker(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta función con circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            **kwargs: Keyword arguments
        
        Returns:
            Resultado de la función
        """
        if not self._circuit_breaker:
            return await func(*args, **kwargs)
        
        return await self._circuit_breaker.call_async(func, *args, **kwargs)
    
    async def _execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta función con retry logic.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            **kwargs: Keyword arguments
        
        Returns:
            Resultado de la función
        """
        if not self._retry_config:
            return await func(*args, **kwargs)
        
        from ..retry_logic import retry_async
        return await retry_async(func, config=self._retry_config, *args, **kwargs)
    
    async def _execute_robust(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta función con todas las protecciones (timeout, circuit breaker, retry).
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            **kwargs: Keyword arguments
        
        Returns:
            Resultado de la función
        """
        # Wrapper con todas las protecciones
        async def protected_func():
            if self.enable_circuit_breaker:
                return await self._execute_with_circuit_breaker(func, *args, **kwargs)
            elif self.enable_retry:
                return await self._execute_with_retry(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        return await self._execute_with_timeout(protected_func)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check del servicio.
        
        Returns:
            Estado de salud del servicio
        """
        health = {
            "service": self.service_name,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check cache
        if self.cache_service:
            try:
                await self.cache_service.exists("health_check")
                health["checks"]["cache"] = "healthy"
            except Exception as e:
                health["checks"]["cache"] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
        
        # Check event publisher
        if self.event_publisher:
            health["checks"]["events"] = "available"
        
        # Check circuit breaker
        if self._circuit_breaker:
            stats = self._circuit_breaker.get_stats()
            health["checks"]["circuit_breaker"] = {
                "state": stats["state"],
                "failure_count": stats["failure_count"]
            }
        
        return health
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """
        Valida que las dependencias estén disponibles.
        
        Returns:
            Diccionario con estado de dependencias
        """
        dependencies = {}
        
        if self.cache_service:
            dependencies["cache"] = True
        else:
            dependencies["cache"] = False
            logger.warning(f"{self.service_name}: Cache service not available")
        
        if self.event_publisher:
            dependencies["events"] = True
        else:
            dependencies["events"] = False
            logger.warning(f"{self.service_name}: Event publisher not available")
        
        return dependencies















