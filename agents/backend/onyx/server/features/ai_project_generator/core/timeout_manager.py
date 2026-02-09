"""
Timeout Manager - Gestor de timeouts
====================================

Gestión centralizada de timeouts para operaciones.
"""

import logging
import asyncio
from typing import Optional, Callable, Any, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class TimeoutStrategy(str, Enum):
    """Estrategias de timeout"""
    FAIL = "fail"  # Fallar si excede timeout
    RETURN_DEFAULT = "return_default"  # Retornar valor por defecto
    RETRY = "retry"  # Reintentar
    DEGRADE = "degrade"  # Degradar operación


class TimeoutManager:
    """
    Gestor de timeouts que proporciona:
    - Timeouts configurables
    - Estrategias de manejo
    - Timeouts por operación
    """
    
    def __init__(self):
        self.default_timeout: float = 30.0
        self.operation_timeouts: Dict[str, float] = {}
        self.strategies: Dict[str, TimeoutStrategy] = {}
    
    def set_default_timeout(self, timeout: float):
        """Establece timeout por defecto"""
        self.default_timeout = timeout
    
    def set_operation_timeout(
        self,
        operation: str,
        timeout: float,
        strategy: TimeoutStrategy = TimeoutStrategy.FAIL
    ):
        """
        Establece timeout para una operación específica.
        
        Args:
            operation: Nombre de la operación
            timeout: Timeout en segundos
            strategy: Estrategia de manejo
        """
        self.operation_timeouts[operation] = timeout
        self.strategies[operation] = strategy
    
    async def execute_with_timeout(
        self,
        operation: str,
        func: Callable,
        *args,
        default_value: Any = None,
        **kwargs
    ) -> Any:
        """
        Ejecuta función con timeout.
        
        Args:
            operation: Nombre de la operación
            func: Función a ejecutar
            *args: Argumentos
            default_value: Valor por defecto si falla
            **kwargs: Keyword arguments
        
        Returns:
            Resultado de la función o valor por defecto
        """
        timeout = self.operation_timeouts.get(operation, self.default_timeout)
        strategy = self.strategies.get(operation, TimeoutStrategy.FAIL)
        
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Operation {operation} timed out after {timeout}s, "
                f"using strategy: {strategy.value}"
            )
            
            if strategy == TimeoutStrategy.RETURN_DEFAULT:
                return default_value
            elif strategy == TimeoutStrategy.DEGRADE:
                # Intentar operación degradada
                return await self._degraded_operation(operation, *args, **kwargs)
            else:
                raise TimeoutError(f"Operation {operation} timed out after {timeout}s")
    
    async def _degraded_operation(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """Operación degradada (simplificada)"""
        # Implementación específica según la operación
        logger.info(f"Executing degraded operation for {operation}")
        return None


def get_timeout_manager() -> TimeoutManager:
    """Obtiene gestor de timeouts con configuraciones por defecto"""
    manager = TimeoutManager()
    
    # Timeouts por defecto para operaciones comunes
    manager.set_operation_timeout("generate_project", 300.0)  # 5 minutos
    manager.set_operation_timeout("create_project", 10.0)
    manager.set_operation_timeout("get_project", 5.0)
    manager.set_operation_timeout("list_projects", 10.0)
    
    return manager

