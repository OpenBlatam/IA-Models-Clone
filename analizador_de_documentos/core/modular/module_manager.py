"""
Module Manager - Gestor de Módulos
==================================

Sistema de gestión de módulos para arquitectura modular.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModuleStatus(Enum):
    """Estado del módulo."""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ModuleInfo:
    """Información de módulo."""
    module_id: str
    name: str
    version: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: ModuleStatus = ModuleStatus.UNREGISTERED
    instance: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModuleRegistry:
    """Registro de módulos."""
    
    def __init__(self):
        """Inicializar registro."""
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_order: List[str] = []
    
    def register(
        self,
        module_id: str,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar módulo."""
        if module_id in self.modules:
            logger.warning(f"Módulo {module_id} ya registrado, actualizando...")
        
        self.modules[module_id] = ModuleInfo(
            module_id=module_id,
            name=name,
            version=version,
            description=description,
            dependencies=dependencies or [],
            status=ModuleStatus.REGISTERED,
            metadata=metadata or {}
        )
        
        if module_id not in self.module_order:
            self.module_order.append(module_id)
        
        logger.info(f"Módulo registrado: {module_id} ({name})")
    
    def get_module(self, module_id: str) -> Optional[ModuleInfo]:
        """Obtener información de módulo."""
        return self.modules.get(module_id)
    
    def get_all_modules(self) -> List[ModuleInfo]:
        """Obtener todos los módulos."""
        return list(self.modules.values())
    
    def get_dependencies(self, module_id: str) -> List[str]:
        """Obtener dependencias de un módulo."""
        module = self.modules.get(module_id)
        return module.dependencies if module else []
    
    def get_load_order(self) -> List[str]:
        """Obtener orden de carga basado en dependencias."""
        # Topological sort
        visited = set()
        result = []
        
        def visit(module_id: str):
            if module_id in visited:
                return
            
            module = self.modules.get(module_id)
            if not module:
                return
            
            visited.add(module_id)
            
            for dep in module.dependencies:
                if dep in self.modules:
                    visit(dep)
            
            result.append(module_id)
        
        for module_id in self.module_order:
            if module_id not in visited:
                visit(module_id)
        
        return result


class ModuleManager:
    """Gestor de módulos."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.registry = ModuleRegistry()
        self.loaded_modules: Dict[str, Any] = {}
        self.module_factories: Dict[str, Callable] = {}
    
    def register_module(
        self,
        module_id: str,
        name: str,
        factory: Callable,
        version: str = "1.0.0",
        description: str = "",
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar módulo con factory."""
        self.registry.register(
            module_id, name, version, description, dependencies, metadata
        )
        self.module_factories[module_id] = factory
    
    async def load_module(self, module_id: str, **kwargs) -> Any:
        """Cargar módulo."""
        if module_id in self.loaded_modules:
            return self.loaded_modules[module_id]
        
        module_info = self.registry.get_module(module_id)
        if not module_info:
            raise ValueError(f"Módulo {module_id} no registrado")
        
        # Verificar dependencias
        for dep_id in module_info.dependencies:
            if dep_id not in self.loaded_modules:
                await self.load_module(dep_id, **kwargs)
        
        # Cargar módulo
        module_info.status = ModuleStatus.LOADING
        
        try:
            factory = self.module_factories.get(module_id)
            if not factory:
                raise ValueError(f"Factory no encontrada para módulo {module_id}")
            
            instance = factory(**kwargs)
            
            # Si es async, esperar
            if asyncio.iscoroutine(instance):
                instance = await instance
            
            self.loaded_modules[module_id] = instance
            module_info.instance = instance
            module_info.status = ModuleStatus.LOADED
            
            logger.info(f"Módulo cargado: {module_id}")
            
            return instance
        except Exception as e:
            module_info.status = ModuleStatus.ERROR
            logger.error(f"Error cargando módulo {module_id}: {e}")
            raise
    
    async def load_all_modules(self, **kwargs) -> Dict[str, Any]:
        """Cargar todos los módulos en orden."""
        load_order = self.registry.get_load_order()
        loaded = {}
        
        for module_id in load_order:
            try:
                instance = await self.load_module(module_id, **kwargs)
                loaded[module_id] = instance
            except Exception as e:
                logger.error(f"Error cargando {module_id}: {e}")
        
        return loaded
    
    def get_module(self, module_id: str) -> Optional[Any]:
        """Obtener módulo cargado."""
        return self.loaded_modules.get(module_id)
    
    def unload_module(self, module_id: str):
        """Descargar módulo."""
        if module_id in self.loaded_modules:
            module_info = self.registry.get_module(module_id)
            if module_info:
                module_info.status = ModuleStatus.UNLOADING
            
            del self.loaded_modules[module_id]
            
            if module_info:
                module_info.status = ModuleStatus.REGISTERED
                module_info.instance = None
            
            logger.info(f"Módulo descargado: {module_id}")
    
    def get_module_status(self, module_id: str) -> Optional[ModuleStatus]:
        """Obtener estado de módulo."""
        module = self.registry.get_module(module_id)
        return module.status if module else None
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """Listar todos los módulos."""
        return [
            {
                "id": info.module_id,
                "name": info.name,
                "version": info.version,
                "status": info.status.value,
                "dependencies": info.dependencies,
                "loaded": info.module_id in self.loaded_modules
            }
            for info in self.registry.get_all_modules()
        ]


__all__ = [
    "ModuleManager",
    "ModuleRegistry",
    "ModuleStatus",
    "ModuleInfo"
]


