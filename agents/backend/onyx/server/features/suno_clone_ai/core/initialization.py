"""
Sistema de Inicialización

Gestiona la inicialización ordenada de todos los componentes del sistema.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.dependency_injection import get_container
from core.factories import (
    MusicGeneratorFactory,
    CacheFactory,
    StorageFactory
)
from core.plugins import get_plugin_manager
from core.modules import get_module_registry
from core.events import get_event_bus
from config.settings import settings

logger = logging.getLogger(__name__)


class SystemInitializer:
    """Inicializador del sistema"""
    
    def __init__(self):
        self.initialized = False
        self.initialization_order: List[str] = []
        self.initialization_times: Dict[str, float] = {}
        logger.info("SystemInitializer created")
    
    async def initialize_all(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Inicializa todos los componentes del sistema
        
        Args:
            config: Configuración opcional
        
        Returns:
            Diccionario con resultados de inicialización
        """
        if self.initialized:
            logger.warning("System already initialized")
            return {}
        
        import time
        results = {}
        config = config or {}
        
        try:
            # 1. Inicializar contenedor de dependencias
            start = time.time()
            container = get_container()
            self._register_core_services(container, config)
            results["dependency_container"] = True
            self.initialization_order.append("dependency_container")
            self.initialization_times["dependency_container"] = time.time() - start
            
            # 2. Inicializar storage
            start = time.time()
            storage = StorageFactory.create_storage(
                storage_type=config.get("storage_type", "local"),
                base_path=config.get("storage_path", "storage")
            )
            container.register("storage", storage)
            results["storage"] = True
            self.initialization_order.append("storage")
            self.initialization_times["storage"] = time.time() - start
            
            # 3. Inicializar caché
            start = time.time()
            cache = CacheFactory.create_cache(
                cache_type=config.get("cache_type", "memory")
            )
            container.register("cache", cache)
            results["cache"] = True
            self.initialization_order.append("cache")
            self.initialization_times["cache"] = time.time() - start
            
            # 4. Inicializar generador de música
            start = time.time()
            generator = MusicGeneratorFactory.create_generator(
                generator_type=config.get("generator_type", "default")
            )
            container.register("music_generator", generator)
            results["music_generator"] = True
            self.initialization_order.append("music_generator")
            self.initialization_times["music_generator"] = time.time() - start
            
            # 5. Inicializar bus de eventos
            start = time.time()
            event_bus = get_event_bus()
            container.register("event_bus", event_bus)
            results["event_bus"] = True
            self.initialization_order.append("event_bus")
            self.initialization_times["event_bus"] = time.time() - start
            
            # 6. Inicializar plugins
            start = time.time()
            plugin_manager = get_plugin_manager()
            plugin_results = await plugin_manager.initialize_all()
            results["plugins"] = all(plugin_results.values())
            self.initialization_order.append("plugins")
            self.initialization_times["plugins"] = time.time() - start
            
            # 7. Inicializar módulos
            start = time.time()
            module_registry = get_module_registry()
            module_results = await module_registry.initialize_all()
            results["modules"] = all(module_results.values())
            self.initialization_order.append("modules")
            self.initialization_times["modules"] = time.time() - start
            
            self.initialized = True
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during system initialization: {e}", exc_info=True)
            results["error"] = str(e)
        
        return results
    
    def _register_core_services(self, container, config: Dict[str, Any]):
        """Registra servicios core"""
        # Servicios adicionales pueden registrarse aquí
        pass
    
    def get_initialization_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de inicialización"""
        total_time = sum(self.initialization_times.values())
        
        return {
            "initialized": self.initialized,
            "order": self.initialization_order,
            "times": self.initialization_times,
            "total_time": total_time,
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Cierra todos los componentes"""
        if not self.initialized:
            return
        
        try:
            # Cerrar plugins
            plugin_manager = get_plugin_manager()
            await plugin_manager.shutdown_all()
            
            # Cerrar módulos
            module_registry = get_module_registry()
            for module_name in module_registry.list_modules():
                module = module_registry.get_module(module_name)
                if module and hasattr(module, "shutdown"):
                    await module.shutdown()
            
            self.initialized = False
            logger.info("System shutdown completed")
        
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}", exc_info=True)


# Instancia global
_system_initializer: Optional[SystemInitializer] = None


def get_system_initializer() -> SystemInitializer:
    """Obtiene la instancia global del inicializador"""
    global _system_initializer
    if _system_initializer is None:
        _system_initializer = SystemInitializer()
    return _system_initializer


async def initialize_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """
    Inicializa el sistema (función helper)
    
    Args:
        config: Configuración opcional
    
    Returns:
        Resultados de inicialización
    """
    initializer = get_system_initializer()
    return await initializer.initialize_all(config)

