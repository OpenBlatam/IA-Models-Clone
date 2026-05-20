"""
MCP Plugins - Sistema de plugins para conectores
================================================
"""

import importlib
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from abc import ABC, abstractmethod

from .connectors import BaseConnector
from .exceptions import MCPError

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """
    Clase base para plugins MCP
    
    Los plugins pueden extender funcionalidad del servidor MCP.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Args:
            name: Nombre del plugin
            version: Versión del plugin
        """
        self.name = name
        self.version = version
        self.enabled = True
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """
        Inicializa el plugin
        
        Args:
            context: Contexto con acceso a componentes MCP
            
        Returns:
            True si se inicializó correctamente
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Limpia recursos del plugin"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtiene información del plugin
        
        Returns:
            Diccionario con información
        """
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
        }


class ConnectorPlugin(Plugin):
    """
    Plugin que agrega un nuevo conector
    """
    
    def __init__(self, name: str, connector: BaseConnector):
        """
        Args:
            name: Nombre del plugin
            connector: Instancia del conector
        """
        super().__init__(name)
        self.connector = connector
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Registra el conector"""
        registry = context.get("connector_registry")
        if not registry:
            logger.error(f"Plugin {self.name}: connector_registry not found in context")
            return False
        
        registry.register(self.name, self.connector)
        logger.info(f"Plugin {self.name}: connector registered")
        return True
    
    def cleanup(self):
        """Limpia el plugin"""
        pass


class MiddlewarePlugin(Plugin):
    """
    Plugin que agrega middleware
    """
    
    def __init__(self, name: str, middleware: Callable):
        """
        Args:
            name: Nombre del plugin
            middleware: Función middleware
        """
        super().__init__(name)
        self.middleware = middleware
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Registra el middleware"""
        app = context.get("app")
        if not app:
            logger.error(f"Plugin {self.name}: app not found in context")
            return False
        
        app.add_middleware(self.middleware)
        logger.info(f"Plugin {self.name}: middleware registered")
        return True
    
    def cleanup(self):
        """Limpia el plugin"""
        pass


class PluginManager:
    """
    Gestor de plugins MCP
    
    Permite cargar y gestionar plugins dinámicamente.
    """
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._loaded_modules: Dict[str, Any] = {}
    
    def register(self, plugin: Plugin):
        """
        Registra un plugin
        
        Args:
            plugin: Instancia del plugin
        """
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
    
    def load_from_module(self, module_path: str, plugin_name: Optional[str] = None) -> bool:
        """
        Carga plugin desde módulo Python
        
        Args:
            module_path: Ruta al módulo (ej: "my_plugin.module")
            plugin_name: Nombre del plugin (opcional)
            
        Returns:
            True si se cargó correctamente
        """
        try:
            module = importlib.import_module(module_path)
            
            # Buscar clase Plugin en el módulo
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, Plugin) and 
                    attr != Plugin):
                    plugin_class = attr
                    break
            
            if not plugin_class:
                logger.error(f"No Plugin class found in {module_path}")
                return False
            
            # Instanciar plugin
            plugin = plugin_class()
            plugin_name = plugin_name or plugin.name
            
            self.register(plugin)
            self._loaded_modules[module_path] = module
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin from {module_path}: {e}")
            return False
    
    def load_from_directory(self, directory: str, pattern: str = "*.py") -> int:
        """
        Carga plugins desde directorio
        
        Args:
            directory: Directorio con plugins
            pattern: Patrón de archivos a cargar
            
        Returns:
            Número de plugins cargados
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Plugin directory not found: {directory}")
            return 0
        
        loaded = 0
        for file_path in dir_path.glob(pattern):
            if file_path.name.startswith("_"):
                continue
            
            module_name = file_path.stem
            module_path = f"{directory.replace('/', '.')}.{module_name}"
            
            if self.load_from_module(module_path):
                loaded += 1
        
        logger.info(f"Loaded {loaded} plugins from {directory}")
        return loaded
    
    def initialize_all(self, context: Dict[str, Any]) -> int:
        """
        Inicializa todos los plugins
        
        Args:
            context: Contexto para plugins
            
        Returns:
            Número de plugins inicializados
        """
        initialized = 0
        for plugin in self._plugins.values():
            if plugin.enabled:
                try:
                    if plugin.initialize(context):
                        initialized += 1
                except Exception as e:
                    logger.error(f"Error initializing plugin {plugin.name}: {e}")
        
        logger.info(f"Initialized {initialized}/{len(self._plugins)} plugins")
        return initialized
    
    def cleanup_all(self):
        """Limpia todos los plugins"""
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.name}: {e}")
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Obtiene un plugin por nombre
        
        Args:
            name: Nombre del plugin
            
        Returns:
            Plugin o None
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        Lista todos los plugins
        
        Returns:
            Lista de información de plugins
        """
        return [plugin.get_info() for plugin in self._plugins.values()]

