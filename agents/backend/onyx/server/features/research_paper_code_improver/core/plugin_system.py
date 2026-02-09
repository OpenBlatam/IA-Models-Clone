"""
Plugin System - Sistema de plugins extensible
==============================================
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class Plugin:
    """Base class para plugins"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
    
    def on_paper_uploaded(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Hook cuando se sube un paper"""
        return None
    
    def on_code_improved(self, improvement_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Hook cuando se mejora código"""
        return None
    
    def on_model_trained(self, model_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Hook cuando se entrena un modelo"""
        return None


class PluginManager:
    """
    Gestiona plugins del sistema.
    """
    
    def __init__(self, plugins_dir: str = "plugins"):
        """
        Inicializar gestor de plugins.
        
        Args:
            plugins_dir: Directorio de plugins
        """
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {
            "paper_uploaded": [],
            "code_improved": [],
            "model_trained": []
        }
    
    def register_plugin(self, plugin: Plugin) -> bool:
        """
        Registra un plugin.
        
        Args:
            plugin: Instancia del plugin
            
        Returns:
            True si se registró exitosamente
        """
        try:
            self.plugins[plugin.name] = plugin
            
            # Registrar hooks
            if hasattr(plugin, "on_paper_uploaded"):
                self.hooks["paper_uploaded"].append(plugin.on_paper_uploaded)
            
            if hasattr(plugin, "on_code_improved"):
                self.hooks["code_improved"].append(plugin.on_code_improved)
            
            if hasattr(plugin, "on_model_trained"):
                self.hooks["model_trained"].append(plugin.on_model_trained)
            
            logger.info(f"Plugin registrado: {plugin.name} v{plugin.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error registrando plugin: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Desregistra un plugin.
        
        Args:
            plugin_name: Nombre del plugin
            
        Returns:
            True si se desregistró exitosamente
        """
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        
        # Remover hooks
        for hook_name, hooks in self.hooks.items():
            hook_method = getattr(plugin, f"on_{hook_name}", None)
            if hook_method and hook_method in hooks:
                hooks.remove(hook_method)
        
        del self.plugins[plugin_name]
        
        logger.info(f"Plugin desregistrado: {plugin_name}")
        return True
    
    def trigger_hook(self, hook_name: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dispara un hook.
        
        Args:
            hook_name: Nombre del hook
            data: Datos para el hook
            
        Returns:
            Resultados de los hooks
        """
        if hook_name not in self.hooks:
            logger.warning(f"Hook desconocido: {hook_name}")
            return []
        
        results = []
        for hook in self.hooks[hook_name]:
            try:
                result = hook(data)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error en hook {hook_name}: {e}")
        
        return results
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """Lista plugins registrados"""
        return [
            {
                "name": plugin.name,
                "version": plugin.version
            }
            for plugin in self.plugins.values()
        ]
    
    def load_plugin_from_file(self, file_path: str) -> bool:
        """
        Carga un plugin desde archivo.
        
        Args:
            file_path: Ruta al archivo del plugin
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            # En producción, aquí se cargaría el plugin dinámicamente
            # Por ahora, es un placeholder
            logger.info(f"Intentando cargar plugin desde: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error cargando plugin: {e}")
            return False




