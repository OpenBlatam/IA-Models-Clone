"""
Fallback Manager - Gestor de fallbacks
======================================

Sistema de fallbacks para degradación graceful cuando servicios fallan.
"""

import logging
from typing import Optional, Callable, Any, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackStrategy(str, Enum):
    """Estrategias de fallback"""
    NONE = "none"  # No fallback, fallar
    CACHE = "cache"  # Usar cache
    DEFAULT = "default"  # Usar valor por defecto
    RETRY = "retry"  # Reintentar
    DEGRADE = "degrade"  # Degradar funcionalidad


class FallbackManager:
    """
    Gestor de fallbacks que proporciona:
    - Fallbacks automáticos
    - Degradación graceful
    - Valores por defecto
    """
    
    def __init__(self):
        self.fallbacks: Dict[str, Dict[str, Any]] = {}
    
    def register_fallback(
        self,
        service_name: str,
        strategy: FallbackStrategy,
        fallback_func: Optional[Callable] = None,
        default_value: Any = None
    ):
        """
        Registra un fallback para un servicio.
        
        Args:
            service_name: Nombre del servicio
            strategy: Estrategia de fallback
            fallback_func: Función de fallback (opcional)
            default_value: Valor por defecto (opcional)
        """
        self.fallbacks[service_name] = {
            "strategy": strategy,
            "func": fallback_func,
            "default": default_value
        }
    
    async def execute_with_fallback(
        self,
        service_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Ejecuta operación con fallback si falla.
        
        Args:
            service_name: Nombre del servicio
            operation: Operación a ejecutar
            *args: Argumentos
            **kwargs: Keyword arguments
        
        Returns:
            Resultado de la operación o fallback
        """
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{service_name} operation failed, using fallback: {e}")
            
            fallback = self.fallbacks.get(service_name)
            if not fallback:
                raise
            
            strategy = fallback["strategy"]
            
            if strategy == FallbackStrategy.CACHE:
                # Intentar obtener de cache
                return await self._fallback_cache(service_name, *args, **kwargs)
            
            elif strategy == FallbackStrategy.DEFAULT:
                # Retornar valor por defecto
                return fallback["default"]
            
            elif strategy == FallbackStrategy.DEGRADE:
                # Degradar funcionalidad
                if fallback["func"]:
                    return await fallback["func"](*args, **kwargs)
                return fallback["default"]
            
            else:
                # No fallback, re-raise
                raise
    
    async def _fallback_cache(
        self,
        service_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Fallback usando cache"""
        try:
            from ..infrastructure.cache import get_cache_service
            cache = get_cache_service()
            
            # Generar clave de cache
            cache_key = f"{service_name}:{str(args)}:{str(kwargs)}"
            cached = await cache.get(cache_key)
            
            if cached:
                logger.info(f"Using cached value for {service_name}")
                return cached
            
            # Si no hay cache, usar default
            fallback = self.fallbacks.get(service_name)
            return fallback.get("default") if fallback else None
            
        except Exception as e:
            logger.error(f"Cache fallback failed: {e}")
            fallback = self.fallbacks.get(service_name)
            return fallback.get("default") if fallback else None


def get_fallback_manager() -> FallbackManager:
    """Obtiene gestor de fallbacks con configuraciones por defecto"""
    manager = FallbackManager()
    
    # Configurar fallbacks por defecto
    manager.register_fallback(
        "cache",
        FallbackStrategy.DEGRADE,
        default_value={}
    )
    
    manager.register_fallback(
        "events",
        FallbackStrategy.NONE,  # Events no son críticos
        default_value=None
    )
    
    return manager















