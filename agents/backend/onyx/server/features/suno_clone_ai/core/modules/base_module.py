"""
Clase base para módulos

Proporciona funcionalidad común para todos los módulos del sistema.
"""

from abc import ABC
from typing import Dict, Any, Optional
from datetime import datetime


class BaseModule(ABC):
    """Clase base para módulos del sistema"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Args:
            name: Nombre del módulo
            version: Versión del módulo
        """
        self.name = name
        self.version = version
        self.initialized = False
        self.initialized_at: Optional[datetime] = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Inicializa el módulo
        
        Args:
            config: Configuración del módulo
        
        Returns:
            True si se inicializó exitosamente
        """
        if self.initialized:
            return True
        
        try:
            success = await self._on_initialize(config)
            if success:
                self.initialized = True
                self.initialized_at = datetime.now()
            return success
        except Exception as e:
            print(f"Error initializing module {self.name}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Cierra el módulo"""
        if not self.initialized:
            return
        
        try:
            await self._on_shutdown()
            self.initialized = False
        except Exception as e:
            print(f"Error shutting down module {self.name}: {e}")
    
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
    
    def get_info(self) -> Dict[str, Any]:
        """Obtiene información del módulo"""
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self.initialized,
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None
        }

