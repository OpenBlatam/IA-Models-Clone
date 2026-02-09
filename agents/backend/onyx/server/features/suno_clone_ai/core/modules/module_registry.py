"""
Registro de Módulos

Gestiona el registro y carga de módulos del sistema.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Registro centralizado de módulos"""
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
        self._module_configs: Dict[str, Dict[str, Any]] = {}
        logger.info("ModuleRegistry initialized")
    
    def register_module(
        self,
        module_name: str,
        module: Any,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ):
        """
        Registra un módulo
        
        Args:
            module_name: Nombre del módulo
            module: Instancia del módulo
            config: Configuración del módulo
            dependencies: Lista de dependencias
        """
        if module_name in self._modules:
            logger.warning(f"Module already registered: {module_name}")
            return
        
        self._modules[module_name] = {
            "module": module,
            "dependencies": dependencies or [],
            "initialized": False
        }
        self._module_configs[module_name] = config or {}
        logger.info(f"Module registered: {module_name}")
    
    async def initialize_module(self, module_name: str) -> bool:
        """
        Inicializa un módulo y sus dependencias
        
        Args:
            module_name: Nombre del módulo
        
        Returns:
            True si se inicializó exitosamente
        """
        if module_name not in self._modules:
            logger.error(f"Module not found: {module_name}")
            return False
        
        module_info = self._modules[module_name]
        
        if module_info["initialized"]:
            return True
        
        # Inicializar dependencias primero
        for dep in module_info["dependencies"]:
            if not await self.initialize_module(dep):
                logger.error(f"Failed to initialize dependency: {dep}")
                return False
        
        # Inicializar módulo
        module = module_info["module"]
        config = self._module_configs.get(module_name, {})
        
        try:
            if hasattr(module, "initialize"):
                if asyncio.iscoroutinefunction(module.initialize):
                    success = await module.initialize(config)
                else:
                    success = module.initialize(config)
            else:
                success = True
            
            if success:
                module_info["initialized"] = True
                logger.info(f"Module initialized: {module_name}")
            else:
                logger.error(f"Module initialization failed: {module_name}")
            
            return success
        except Exception as e:
            logger.error(f"Error initializing module {module_name}: {e}")
            return False
    
    async def initialize_all(self) -> Dict[str, bool]:
        """Inicializa todos los módulos registrados"""
        results = {}
        for module_name in self._modules:
            results[module_name] = await self.initialize_module(module_name)
        return results
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """Obtiene un módulo"""
        module_info = self._modules.get(module_name)
        return module_info["module"] if module_info else None
    
    def list_modules(self) -> List[str]:
        """Lista todos los módulos registrados"""
        return list(self._modules.keys())
    
    def get_dependencies(self, module_name: str) -> List[str]:
        """Obtiene dependencias de un módulo"""
        module_info = self._modules.get(module_name)
        return module_info["dependencies"] if module_info else []


# Instancia global
_module_registry: Optional[ModuleRegistry] = None


def get_module_registry() -> ModuleRegistry:
    """Obtiene la instancia global del registro de módulos"""
    global _module_registry
    if _module_registry is None:
        _module_registry = ModuleRegistry()
    return _module_registry

