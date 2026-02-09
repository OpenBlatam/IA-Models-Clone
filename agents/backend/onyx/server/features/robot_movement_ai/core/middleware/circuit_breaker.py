"""
Circuit Breaker System
======================

Sistema de circuit breaker para manejo de fallos.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estado del circuit breaker."""
    CLOSED = "closed"  # Normal, permitir requests
    OPEN = "open"  # Fallos detectados, bloquear requests
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó


@dataclass
class CircuitBreaker:
    """Circuit breaker."""
    name: str
    failure_threshold: int = 5  # Número de fallos para abrir
    success_threshold: int = 2  # Número de éxitos para cerrar
    timeout: float = 60.0  # Tiempo en segundos antes de intentar half-open
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreakerManager:
    """
    Gestor de circuit breakers.
    
    Gestiona múltiples circuit breakers.
    """
    
    def __init__(self):
        """Inicializar gestor de circuit breakers."""
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def create_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CircuitBreaker:
        """
        Crear circuit breaker.
        
        Args:
            name: Nombre del breaker
            failure_threshold: Umbral de fallos
            success_threshold: Umbral de éxitos
            timeout: Timeout en segundos
            metadata: Metadata adicional
            
        Returns:
            Circuit breaker creado
        """
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            metadata=metadata or {}
        )
        
        self.breakers[name] = breaker
        logger.info(f"Created circuit breaker: {name}")
        
        return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Obtener circuit breaker por nombre."""
        return self.breakers.get(name)
    
    def can_execute(self, name: str) -> bool:
        """
        Verificar si se puede ejecutar.
        
        Args:
            name: Nombre del breaker
            
        Returns:
            True si se puede ejecutar, False si está abierto
        """
        breaker = self.get_breaker(name)
        if not breaker:
            return True  # Si no existe, permitir
        
        if breaker.state == CircuitState.CLOSED:
            return True
        
        if breaker.state == CircuitState.OPEN:
            # Verificar si ha pasado el timeout
            if breaker.last_failure_time:
                elapsed = time.time() - breaker.last_failure_time
                if elapsed >= breaker.timeout:
                    # Cambiar a half-open
                    breaker.state = CircuitState.HALF_OPEN
                    breaker.success_count = 0
                    breaker.last_state_change = datetime.now().isoformat()
                    logger.info(f"Circuit breaker {name} moved to HALF_OPEN")
                    return True
            return False
        
        if breaker.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self, name: str) -> None:
        """Registrar éxito."""
        breaker = self.get_breaker(name)
        if not breaker:
            return
        
        breaker.success_count += 1
        
        if breaker.state == CircuitState.HALF_OPEN:
            if breaker.success_count >= breaker.success_threshold:
                # Cerrar el circuit breaker
                breaker.state = CircuitState.CLOSED
                breaker.failure_count = 0
                breaker.success_count = 0
                breaker.last_state_change = datetime.now().isoformat()
                logger.info(f"Circuit breaker {name} moved to CLOSED")
    
    def record_failure(self, name: str) -> None:
        """Registrar fallo."""
        breaker = self.get_breaker(name)
        if not breaker:
            return
        
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        
        if breaker.state == CircuitState.CLOSED:
            if breaker.failure_count >= breaker.failure_threshold:
                # Abrir el circuit breaker
                breaker.state = CircuitState.OPEN
                breaker.last_state_change = datetime.now().isoformat()
                logger.warning(f"Circuit breaker {name} moved to OPEN")
        
        elif breaker.state == CircuitState.HALF_OPEN:
            # Volver a abrir
            breaker.state = CircuitState.OPEN
            breaker.success_count = 0
            breaker.last_state_change = datetime.now().isoformat()
            logger.warning(f"Circuit breaker {name} moved back to OPEN")
    
    async def execute(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecutar función con circuit breaker.
        
        Args:
            name: Nombre del breaker
            func: Función a ejecutar
            *args: Argumentos
            **kwargs: Keyword arguments
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si el circuit breaker está abierto o la función falla
        """
        if not self.can_execute(name):
            raise Exception(f"Circuit breaker {name} is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.record_success(name)
            return result
        except Exception as e:
            self.record_failure(name)
            raise
    
    def get_statistics(self, name: str) -> Dict[str, Any]:
        """Obtener estadísticas de circuit breaker."""
        breaker = self.get_breaker(name)
        if not breaker:
            return {"error": "Circuit breaker not found"}
        
        return {
            "name": breaker.name,
            "state": breaker.state.value,
            "failure_count": breaker.failure_count,
            "success_count": breaker.success_count,
            "failure_threshold": breaker.failure_threshold,
            "success_threshold": breaker.success_threshold,
            "timeout": breaker.timeout,
            "last_state_change": breaker.last_state_change,
            "last_failure_time": breaker.last_failure_time
        }


# Instancia global
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Obtener instancia global del gestor de circuit breakers."""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager


# Importar asyncio para verificar funciones async
import asyncio






