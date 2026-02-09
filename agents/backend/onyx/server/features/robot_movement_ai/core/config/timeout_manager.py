"""
Timeout Manager System
======================

Sistema de gestión de timeouts.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TimeoutConfig:
    """Configuración de timeout."""
    timeout: float
    default_timeout: float = 30.0
    raise_on_timeout: bool = True
    return_on_timeout: Any = None


class TimeoutManager:
    """
    Gestor de timeouts.
    
    Gestiona timeouts para operaciones.
    """
    
    def __init__(self):
        """Inicializar gestor de timeouts."""
        self.timeout_history: List[Dict[str, Any]] = []
        self.max_history = 10000
    
    async def execute_with_timeout(
        self,
        func: Callable,
        timeout: float,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecutar función con timeout.
        
        Args:
            func: Función a ejecutar
            timeout: Timeout en segundos
            *args: Argumentos
            **kwargs: Keyword arguments
            
        Returns:
            Resultado de la función
            
        Raises:
            asyncio.TimeoutError: Si se excede el timeout
        """
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            else:
                # Para funciones síncronas, ejecutar en thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args, **kwargs),
                    timeout=timeout
                )
            
            execution_time = time.time() - start_time
            self._record_timeout(func.__name__, timeout, execution_time, False)
            
            return result
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self._record_timeout(func.__name__, timeout, execution_time, True)
            logger.warning(f"Function {func.__name__} timed out after {timeout}s")
            raise
    
    def _record_timeout(
        self,
        function_name: str,
        timeout: float,
        execution_time: float,
        timed_out: bool
    ) -> None:
        """Registrar timeout en historial."""
        self.timeout_history.append({
            "function_name": function_name,
            "timeout": timeout,
            "execution_time": execution_time,
            "timed_out": timed_out,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.timeout_history) > self.max_history:
            self.timeout_history = self.timeout_history[-self.max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de timeouts."""
        if not self.timeout_history:
            return {
                "total_executions": 0,
                "timeouts": 0,
                "average_execution_time": 0.0
            }
        
        timeouts = sum(1 for t in self.timeout_history if t["timed_out"])
        avg_time = sum(t["execution_time"] for t in self.timeout_history) / len(self.timeout_history)
        
        return {
            "total_executions": len(self.timeout_history),
            "timeouts": timeouts,
            "timeout_rate": timeouts / len(self.timeout_history) if self.timeout_history else 0.0,
            "average_execution_time": avg_time
        }


# Instancia global
_timeout_manager: Optional[TimeoutManager] = None


def get_timeout_manager() -> TimeoutManager:
    """Obtener instancia global del gestor de timeouts."""
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = TimeoutManager()
    return _timeout_manager






