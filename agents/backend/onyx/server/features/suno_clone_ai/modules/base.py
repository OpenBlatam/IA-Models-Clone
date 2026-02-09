"""
Base Module Interface
Define la interfaz base para todos los módulos
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModuleStatus(Enum):
    """Estados de un módulo"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ModuleConfig:
    """Configuración de un módulo"""
    name: str
    version: str
    enabled: bool = True
    dependencies: List[str] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}


class BaseModule(ABC):
    """
    Clase base para todos los módulos
    Cada módulo es independiente y puede funcionar como microservicio
    """
    
    def __init__(self, config: ModuleConfig):
        self.config = config
        self.status = ModuleStatus.UNINITIALIZED
        self.logger = logging.getLogger(f"module.{config.name}")
        self._dependencies: Dict[str, 'BaseModule'] = {}
    
    @property
    def name(self) -> str:
        """Nombre del módulo"""
        return self.config.name
    
    @property
    def version(self) -> str:
        """Versión del módulo"""
        return self.config.version
    
    @property
    def is_enabled(self) -> bool:
        """Si el módulo está habilitado"""
        return self.config.enabled
    
    async def initialize(self, dependencies: Dict[str, 'BaseModule'] = None) -> bool:
        """
        Inicializa el módulo
        
        Args:
            dependencies: Diccionario de módulos dependientes
            
        Returns:
            True si la inicialización fue exitosa
        """
        if not self.config.enabled:
            self.logger.info(f"Module {self.name} is disabled")
            return False
        
        if self.status != ModuleStatus.UNINITIALIZED:
            self.logger.warning(f"Module {self.name} already initialized")
            return False
        
        self.status = ModuleStatus.INITIALIZING
        self.logger.info(f"Initializing module {self.name} v{self.version}")
        
        try:
            # Guardar dependencias
            if dependencies:
                self._dependencies = dependencies
            
            # Verificar dependencias
            missing = self._check_dependencies()
            if missing:
                raise ValueError(f"Missing dependencies: {missing}")
            
            # Inicializar módulo
            await self._initialize()
            
            self.status = ModuleStatus.ACTIVE
            self.logger.info(f"Module {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.status = ModuleStatus.ERROR
            self.logger.error(f"Failed to initialize module {self.name}: {e}", exc_info=True)
            raise
    
    async def shutdown(self) -> bool:
        """
        Cierra el módulo
        
        Returns:
            True si el cierre fue exitoso
        """
        if self.status == ModuleStatus.SHUTDOWN:
            return True
        
        self.logger.info(f"Shutting down module {self.name}")
        
        try:
            await self._shutdown()
            self.status = ModuleStatus.SHUTDOWN
            self.logger.info(f"Module {self.name} shut down successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error shutting down module {self.name}: {e}", exc_info=True)
            return False
    
    async def pause(self):
        """Pausa el módulo temporalmente"""
        if self.status == ModuleStatus.ACTIVE:
            await self._pause()
            self.status = ModuleStatus.PAUSED
            self.logger.info(f"Module {self.name} paused")
    
    async def resume(self):
        """Reanuda el módulo"""
        if self.status == ModuleStatus.PAUSED:
            await self._resume()
            self.status = ModuleStatus.ACTIVE
            self.logger.info(f"Module {self.name} resumed")
    
    def get_dependency(self, name: str) -> Optional['BaseModule']:
        """Obtiene una dependencia por nombre"""
        return self._dependencies.get(name)
    
    def _check_dependencies(self) -> List[str]:
        """Verifica que todas las dependencias estén disponibles"""
        missing = []
        for dep_name in self.config.dependencies:
            if dep_name not in self._dependencies:
                missing.append(dep_name)
        return missing
    
    @abstractmethod
    async def _initialize(self):
        """Inicialización específica del módulo (implementar en subclases)"""
        pass
    
    @abstractmethod
    async def _shutdown(self):
        """Cierre específico del módulo (implementar en subclases)"""
        pass
    
    async def _pause(self):
        """Pausa específica del módulo (opcional)"""
        pass
    
    async def _resume(self):
        """Reanudación específica del módulo (opcional)"""
        pass
    
    def get_health(self) -> Dict[str, Any]:
        """Obtiene el estado de salud del módulo"""
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "enabled": self.is_enabled
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del módulo (implementar en subclases)"""
        return {
            "name": self.name,
            "status": self.status.value
        }















