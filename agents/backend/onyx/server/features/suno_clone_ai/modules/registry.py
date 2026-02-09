"""
Module Registry
Gestiona el registro y descubrimiento de módulos
"""

import logging
from typing import Dict, Optional, List, Type
from modules.base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """
    Registro central de módulos
    Permite registrar, descubrir y gestionar módulos
    """
    
    def __init__(self):
        self._modules: Dict[str, BaseModule] = {}
        self._module_classes: Dict[str, Type[BaseModule]] = {}
        self._initialized = False
    
    def register(
        self,
        module_class: Type[BaseModule],
        config: ModuleConfig
    ) -> bool:
        """
        Registra una clase de módulo
        
        Args:
            module_class: Clase del módulo
            config: Configuración del módulo
            
        Returns:
            True si el registro fue exitoso
        """
        if config.name in self._modules:
            logger.warning(f"Module {config.name} already registered, overwriting")
        
        try:
            module = module_class(config)
            self._modules[config.name] = module
            self._module_classes[config.name] = module_class
            logger.info(f"Module {config.name} registered")
            return True
        except Exception as e:
            logger.error(f"Failed to register module {config.name}: {e}")
            return False
    
    def get_module(self, name: str) -> Optional[BaseModule]:
        """Obtiene un módulo por nombre"""
        return self._modules.get(name)
    
    def get_all_modules(self) -> Dict[str, BaseModule]:
        """Obtiene todos los módulos registrados"""
        return self._modules.copy()
    
    def get_active_modules(self) -> Dict[str, BaseModule]:
        """Obtiene solo los módulos activos"""
        return {
            name: module
            for name, module in self._modules.items()
            if module.status == ModuleStatus.ACTIVE
        }
    
    async def initialize_all(self) -> bool:
        """
        Inicializa todos los módulos en orden de dependencias
        
        Returns:
            True si todos los módulos se inicializaron correctamente
        """
        if self._initialized:
            logger.warning("Modules already initialized")
            return True
        
        logger.info("Initializing all modules...")
        
        # Ordenar módulos por dependencias
        ordered = self._topological_sort()
        
        # Inicializar en orden
        for module_name in ordered:
            module = self._modules[module_name]
            if not module.is_enabled:
                continue
            
            dependencies = {
                dep: self._modules[dep]
                for dep in module.config.dependencies
                if dep in self._modules
            }
            
            try:
                await module.initialize(dependencies)
            except Exception as e:
                logger.error(f"Failed to initialize module {module_name}: {e}")
                return False
        
        self._initialized = True
        logger.info("All modules initialized successfully")
        return True
    
    async def shutdown_all(self):
        """Cierra todos los módulos en orden inverso"""
        logger.info("Shutting down all modules...")
        
        # Cerrar en orden inverso
        ordered = self._topological_sort()
        for module_name in reversed(ordered):
            module = self._modules[module_name]
            if module.status != ModuleStatus.SHUTDOWN:
                await module.shutdown()
        
        self._initialized = False
        logger.info("All modules shut down")
    
    def _topological_sort(self) -> List[str]:
        """
        Ordena los módulos topológicamente según sus dependencias
        
        Returns:
            Lista de nombres de módulos en orden de inicialización
        """
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            module = self._modules.get(name)
            if module:
                for dep in module.config.dependencies:
                    if dep in self._modules:
                        visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            result.append(name)
        
        for module_name in self._modules.keys():
            if module_name not in visited:
                visit(module_name)
        
        return result
    
    def get_health_report(self) -> Dict[str, Any]:
        """Obtiene un reporte de salud de todos los módulos"""
        modules_health = {}
        for name, module in self._modules.items():
            modules_health[name] = module.get_health()
        
        active_count = sum(
            1 for m in self._modules.values()
            if m.status == ModuleStatus.ACTIVE
        )
        
        return {
            "total_modules": len(self._modules),
            "active_modules": active_count,
            "modules": modules_health
        }


# Instancia global del registro
_registry: Optional[ModuleRegistry] = None


def get_module_registry() -> ModuleRegistry:
    """Obtiene la instancia global del registro de módulos"""
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
    return _registry















