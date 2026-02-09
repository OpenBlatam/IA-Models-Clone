"""
Plugin System - Sistema de plugins y extensions
===============================================
"""

import logging
import importlib
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class para plugins"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.enabled = True
    
    @abstractmethod
    def initialize(self):
        """Inicializa el plugin"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cierra el plugin"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Obtiene información del plugin"""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled
        }


class MaterialPlugin(Plugin):
    """Plugin para agregar materiales personalizados"""
    
    def __init__(self, name: str, version: str, materials: Dict[str, Any]):
        super().__init__(name, version)
        self.materials = materials
    
    def initialize(self):
        """Inicializa el plugin de materiales"""
        logger.info(f"Plugin de materiales {self.name} inicializado")
    
    def shutdown(self):
        """Cierra el plugin"""
        logger.info(f"Plugin de materiales {self.name} cerrado")
    
    def get_materials(self) -> Dict[str, Any]:
        """Obtiene materiales del plugin"""
        return self.materials


class ProcessorPlugin(Plugin):
    """Plugin para procesamiento personalizado"""
    
    def __init__(self, name: str, version: str, processor_func: callable):
        super().__init__(name, version)
        self.processor_func = processor_func
    
    def initialize(self):
        """Inicializa el plugin de procesamiento"""
        logger.info(f"Plugin de procesamiento {self.name} inicializado")
    
    def shutdown(self):
        """Cierra el plugin"""
        logger.info(f"Plugin de procesamiento {self.name} cerrado")
    
    def process(self, data: Any) -> Any:
        """Procesa datos"""
        return self.processor_func(data)


class PluginSystem:
    """Sistema de gestión de plugins"""
    
    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = Path(plugins_dir) if plugins_dir else Path("plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_types: Dict[str, List[str]] = {
            "material": [],
            "processor": [],
            "exporter": [],
            "validator": []
        }
    
    def register_plugin(self, plugin: Plugin, plugin_type: str = "general"):
        """Registra un plugin"""
        self.plugins[plugin.name] = plugin
        
        if plugin_type in self.plugin_types:
            self.plugin_types[plugin_type].append(plugin.name)
        
        plugin.initialize()
        logger.info(f"Plugin registrado: {plugin.name} (tipo: {plugin_type})")
    
    def unregister_plugin(self, plugin_name: str):
        """Desregistra un plugin"""
        plugin = self.plugins.get(plugin_name)
        if plugin:
            plugin.shutdown()
            del self.plugins[plugin_name]
            
            # Remover de tipos
            for plugin_list in self.plugin_types.values():
                if plugin_name in plugin_list:
                    plugin_list.remove(plugin_name)
            
            logger.info(f"Plugin desregistrado: {plugin_name}")
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Obtiene un plugin"""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista plugins"""
        plugins = list(self.plugins.values())
        
        if plugin_type:
            plugin_names = self.plugin_types.get(plugin_type, [])
            plugins = [p for p in plugins if p.name in plugin_names]
        
        return [p.get_info() for p in plugins]
    
    def enable_plugin(self, plugin_name: str):
        """Habilita un plugin"""
        plugin = self.plugins.get(plugin_name)
        if plugin:
            plugin.enabled = True
            logger.info(f"Plugin habilitado: {plugin_name}")
    
    def disable_plugin(self, plugin_name: str):
        """Deshabilita un plugin"""
        plugin = self.plugins.get(plugin_name)
        if plugin:
            plugin.enabled = False
            logger.info(f"Plugin deshabilitado: {plugin_name}")
    
    def load_plugin_from_file(self, file_path: str) -> bool:
        """Carga un plugin desde un archivo Python"""
        try:
            # En producción, esto cargaría el módulo dinámicamente
            # Por ahora, solo registra que se intentó cargar
            logger.info(f"Intentando cargar plugin desde: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error cargando plugin: {e}")
            return False




