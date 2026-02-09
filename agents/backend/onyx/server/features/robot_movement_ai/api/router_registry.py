"""
Router Registry
===============

Sistema de registro centralizado para routers de FastAPI.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)


class RouterRegistry:
    """Registro centralizado de routers."""
    
    def __init__(self):
        """Inicializar registro."""
        self._routers: List[Dict[str, Any]] = []
        self._failed_routers: List[str] = []
    
    def register(
        self,
        router: APIRouter,
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        required: bool = False
    ) -> bool:
        """
        Registrar un router.
        
        Args:
            router: Router de FastAPI
            prefix: Prefijo opcional para el router
            tags: Tags opcionales
            required: Si es requerido (falla si no se puede importar)
        
        Returns:
            True si se registró exitosamente
        """
        try:
            self._routers.append({
                "router": router,
                "prefix": prefix,
                "tags": tags,
                "required": required
            })
            return True
        except Exception as e:
            if required:
                logger.error(f"Failed to register required router: {e}")
                raise
            else:
                logger.warning(f"Failed to register optional router: {e}")
                self._failed_routers.append(str(router))
                return False
    
    def register_lazy(
        self,
        module_path: str,
        router_name: str = "router",
        prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        required: bool = False
    ) -> bool:
        """
        Registrar router con import diferido.
        
        Args:
            module_path: Ruta del módulo (ej: 'api.metrics_api')
            router_name: Nombre del router en el módulo
            prefix: Prefijo opcional
            tags: Tags opcionales
            required: Si es requerido
        
        Returns:
            True si se registró exitosamente
        """
        try:
            import importlib
            module = importlib.import_module(module_path)
            router = getattr(module, router_name)
            
            if not isinstance(router, APIRouter):
                raise TypeError(f"{router_name} is not an APIRouter")
            
            return self.register(router, prefix=prefix, tags=tags, required=required)
        except Exception as e:
            if required:
                logger.error(f"Failed to register required router {module_path}: {e}")
                raise
            else:
                logger.warning(f"Failed to register optional router {module_path}: {e}")
                self._failed_routers.append(module_path)
                return False
    
    def get_routers(self) -> List[Dict[str, Any]]:
        """
        Obtener todos los routers registrados.
        
        Returns:
            Lista de diccionarios con información de routers
        """
        return self._routers.copy()
    
    def get_failed_routers(self) -> List[str]:
        """
        Obtener lista de routers que fallaron al registrarse.
        
        Returns:
            Lista de nombres de routers fallidos
        """
        return self._failed_routers.copy()
    
    def clear(self) -> None:
        """Limpiar registro."""
        self._routers.clear()
        self._failed_routers.clear()


# Instancia global
_router_registry: Optional[RouterRegistry] = None


def get_router_registry() -> RouterRegistry:
    """Obtener instancia global del registro."""
    global _router_registry
    if _router_registry is None:
        _router_registry = RouterRegistry()
    return _router_registry

