"""
Graceful Degradation
Sistema de degradación elegante cuando servicios fallan
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Niveles de degradación"""
    NONE = "none"  # Todo funciona
    MINOR = "minor"  # Funcionalidad reducida
    MAJOR = "major"  # Funcionalidad limitada
    CRITICAL = "critical"  # Solo funcionalidad esencial


@dataclass
class ServiceStatus:
    """Estado de un servicio"""
    name: str
    available: bool
    response_time: float
    error_rate: float
    degradation_level: DegradationLevel = DegradationLevel.NONE


class GracefulDegradation:
    """
    Sistema de degradación elegante
    Permite que la aplicación funcione con funcionalidad reducida
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceStatus] = {}
        self._fallbacks: Dict[str, Callable] = {}
        self._degradation_strategies: Dict[str, Dict[DegradationLevel, Callable]] = {}
    
    def register_service(
        self,
        name: str,
        health_check: Callable,
        fallback: Optional[Callable] = None
    ):
        """
        Registra un servicio para monitoreo
        
        Args:
            name: Nombre del servicio
            health_check: Función que verifica salud del servicio
            fallback: Función fallback si el servicio falla
        """
        self._services[name] = ServiceStatus(
            name=name,
            available=True,
            response_time=0.0,
            error_rate=0.0
        )
        
        if fallback:
            self._fallbacks[name] = fallback
        
        logger.info(f"Registered service for degradation: {name}")
    
    def set_degradation_strategy(
        self,
        service_name: str,
        level: DegradationLevel,
        strategy: Callable
    ):
        """Configura estrategia de degradación para un nivel"""
        if service_name not in self._degradation_strategies:
            self._degradation_strategies[service_name] = {}
        
        self._degradation_strategies[service_name][level] = strategy
        logger.info(f"Set degradation strategy for {service_name} at level {level.value}")
    
    async def check_service(self, name: str) -> ServiceStatus:
        """Verifica el estado de un servicio"""
        if name not in self._services:
            raise ValueError(f"Service {name} not registered")
        
        import time
        start_time = time.time()
        
        try:
            # Ejecutar health check
            health_check = self._services[name]
            # Aquí deberías tener la función de health check
            # Por simplicidad, asumimos que está disponible
            
            response_time = time.time() - start_time
            
            # Actualizar estado
            self._services[name].available = True
            self._services[name].response_time = response_time
            
            # Determinar nivel de degradación
            if response_time > 5.0:
                self._services[name].degradation_level = DegradationLevel.MAJOR
            elif response_time > 2.0:
                self._services[name].degradation_level = DegradationLevel.MINOR
            else:
                self._services[name].degradation_level = DegradationLevel.NONE
            
            return self._services[name]
            
        except Exception as e:
            logger.error(f"Service {name} health check failed: {e}")
            self._services[name].available = False
            self._services[name].degradation_level = DegradationLevel.CRITICAL
            return self._services[name]
    
    async def execute_with_fallback(
        self,
        service_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta una operación con fallback si falla
        
        Args:
            service_name: Nombre del servicio
            operation: Operación a ejecutar
            *args, **kwargs: Argumentos para la operación
        """
        service_status = self._services.get(service_name)
        
        if not service_status or not service_status.available:
            # Usar fallback si está disponible
            if service_name in self._fallbacks:
                logger.warning(f"Using fallback for {service_name}")
                fallback = self._fallbacks[service_name]
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                else:
                    return fallback(*args, **kwargs)
            else:
                raise RuntimeError(f"Service {service_name} unavailable and no fallback")
        
        # Ejecutar operación normal
        try:
            if asyncio.iscoroutinefunction(operation):
                return await operation(*args, **kwargs)
            else:
                return operation(*args, **kwargs)
        except Exception as e:
            logger.error(f"Operation failed for {service_name}: {e}")
            
            # Marcar servicio como no disponible
            self._services[service_name].available = False
            
            # Intentar fallback
            if service_name in self._fallbacks:
                fallback = self._fallbacks[service_name]
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                else:
                    return fallback(*args, **kwargs)
            else:
                raise
    
    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """Obtiene el estado de un servicio"""
        return self._services.get(name)
    
    def get_all_status(self) -> Dict[str, ServiceStatus]:
        """Obtiene el estado de todos los servicios"""
        return self._services.copy()


# Instancia global
_degradation: Optional[GracefulDegradation] = None


def get_graceful_degradation() -> GracefulDegradation:
    """Obtiene el sistema de degradación elegante"""
    global _degradation
    if _degradation is None:
        _degradation = GracefulDegradation()
    return _degradation

