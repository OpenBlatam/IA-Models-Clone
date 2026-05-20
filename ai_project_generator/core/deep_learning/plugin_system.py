"""
Plugin System
=============

Sistema de plugins para extender el generador de Deep Learning.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadatos de plugin."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    hooks: List[str] = field(default_factory=list)


class Plugin(ABC):
    """Interfaz base para plugins."""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Obtener metadatos del plugin."""
        pass
    
    @abstractmethod
    def on_generation_start(
        self,
        generator_key: str,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """Llamado al inicio de generación."""
        pass
    
    @abstractmethod
    def on_generation_end(
        self,
        generator_key: str,
        project_dir: Path,
        success: bool,
        files_generated: List[str]
    ) -> None:
        """Llamado al final de generación."""
        pass
    
    def on_file_generated(
        self,
        file_path: Path,
        content: str
    ) -> Optional[str]:
        """
        Llamado cuando se genera un archivo.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido generado
            
        Returns:
            Contenido modificado o None para mantener original
        """
        return None


class PluginManager:
    """
    Gestor de plugins.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {
            'generation_start': [],
            'generation_end': [],
            'file_generated': []
        }
    
    def register_plugin(self, plugin: Plugin) -> None:
        """
        Registrar plugin.
        
        Args:
            plugin: Instancia del plugin
        """
        metadata = plugin.get_metadata()
        self.plugins[metadata.name] = plugin
        
        # Registrar hooks
        if 'generation_start' in metadata.hooks:
            self.hooks['generation_start'].append(plugin.on_generation_start)
        
        if 'generation_end' in metadata.hooks:
            self.hooks['generation_end'].append(plugin.on_generation_end)
        
        if 'file_generated' in metadata.hooks:
            self.hooks['file_generated'].append(plugin.on_file_generated)
        
        logger.info(f"Plugin registrado: {metadata.name} v{metadata.version}")
    
    def unregister_plugin(self, name: str) -> None:
        """
        Desregistrar plugin.
        
        Args:
            name: Nombre del plugin
        """
        if name in self.plugins:
            plugin = self.plugins[name]
            metadata = plugin.get_metadata()
            
            # Remover hooks
            for hook_name in metadata.hooks:
                if hook_name in self.hooks:
                    # Remover métodos del plugin
                    self.hooks[hook_name] = [
                        h for h in self.hooks[hook_name]
                        if not hasattr(h, '__self__') or h.__self__ != plugin
                    ]
            
            del self.plugins[name]
            logger.info(f"Plugin desregistrado: {name}")
    
    def call_hook(
        self,
        hook_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Llamar hook.
        
        Args:
            hook_name: Nombre del hook
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado del último hook o None
        """
        if hook_name not in self.hooks:
            return None
        
        result = None
        for hook in self.hooks[hook_name]:
            try:
                result = hook(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en hook '{hook_name}': {e}")
        
        return result
    
    def list_plugins(self) -> List[PluginMetadata]:
        """
        Listar plugins registrados.
        
        Returns:
            Lista de metadatos de plugins
        """
        return [plugin.get_metadata() for plugin in self.plugins.values()]


# Instancia global
_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """
    Obtener instancia global del gestor de plugins.
    
    Returns:
        Instancia del gestor
    """
    global _global_plugin_manager
    
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    
    return _global_plugin_manager

