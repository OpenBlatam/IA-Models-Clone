"""
Clase base para plugins

Proporciona una implementación base para plugins del sistema.
"""

from abc import ABC
from typing import Dict, Any
from core.interfaces import IPlugin


class BasePlugin(IPlugin, ABC):
    """Clase base para plugins con implementación por defecto"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Args:
            name: Nombre del plugin
            version: Versión del plugin
        """
        self._name = name
        self._version = version
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Nombre del plugin"""
        return self._name
    
    @property
    def version(self) -> str:
        """Versión del plugin"""
        return self._version
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Inicializa el plugin
        
        Args:
            config: Configuración del plugin
        
        Returns:
            True si se inicializó exitosamente
        """
        if self._initialized:
            return True
        
        try:
            success = await self._on_initialize(config)
            self._initialized = success
            return success
        except Exception as e:
            print(f"Error initializing plugin {self.name}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Cierra el plugin"""
        if not self._initialized:
            return
        
        try:
            await self._on_shutdown()
            self._initialized = False
        except Exception as e:
            print(f"Error shutting down plugin {self.name}: {e}")
    
    async def _on_initialize(self, config: Dict[str, Any]) -> bool:
        """
        Hook para inicialización personalizada
        
        Args:
            config: Configuración
        
        Returns:
            True si se inicializó exitosamente
        """
        return True
    
    async def _on_shutdown(self) -> None:
        """Hook para cierre personalizado"""
        pass

