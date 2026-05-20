"""
Plugins - Sistema de plugins y extensibilidad
"""

import logging
import importlib
import importlib.util
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Clase base para plugins"""

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Inicializar plugin.

        Args:
            name: Nombre del plugin
            version: Versión del plugin
        """
        self.name = name
        self.version = version
        self.enabled = True

    @abstractmethod
    def on_add(self, content: str, addition: str, **kwargs) -> Optional[str]:
        """
        Hook llamado antes de agregar contenido.

        Args:
            content: Contenido original
            addition: Contenido a agregar
            **kwargs: Argumentos adicionales

        Returns:
            Contenido modificado o None
        """
        pass

    @abstractmethod
    def on_remove(self, content: str, pattern: str, **kwargs) -> Optional[str]:
        """
        Hook llamado antes de eliminar contenido.

        Args:
            content: Contenido original
            pattern: Patrón a eliminar
            **kwargs: Argumentos adicionales

        Returns:
            Contenido modificado o None
        """
        pass

    def on_after_add(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook llamado después de agregar contenido.

        Args:
            result: Resultado de la operación

        Returns:
            Resultado modificado
        """
        return result

    def on_after_remove(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook llamado después de eliminar contenido.

        Args:
            result: Resultado de la operación

        Returns:
            Resultado modificado
        """
        return result


class PluginManager:
    """Gestor de plugins"""

    def __init__(self):
        """Inicializar el gestor de plugins"""
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {
            "before_add": [],
            "after_add": [],
            "before_remove": [],
            "after_remove": []
        }

    def register_plugin(self, plugin: Plugin):
        """
        Registrar un plugin.

        Args:
            plugin: Instancia del plugin
        """
        if plugin.name in self.plugins:
            logger.warning(f"Plugin {plugin.name} ya está registrado, reemplazando...")
        
        self.plugins[plugin.name] = plugin
        logger.info(f"Plugin registrado: {plugin.name} v{plugin.version}")

    def unregister_plugin(self, plugin_name: str):
        """
        Desregistrar un plugin.

        Args:
            plugin_name: Nombre del plugin
        """
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            logger.info(f"Plugin desregistrado: {plugin_name}")

    def load_plugin_from_module(self, module_path: str, plugin_class: str):
        """
        Cargar plugin desde un módulo.

        Args:
            module_path: Ruta del módulo
            plugin_class: Nombre de la clase del plugin
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class_obj = getattr(module, plugin_class)
            plugin = plugin_class_obj()
            self.register_plugin(plugin)
        except Exception as e:
            logger.error(f"Error cargando plugin {module_path}.{plugin_class}: {e}")

    def load_plugins_from_directory(self, directory: Path):
        """
        Cargar plugins desde un directorio.

        Args:
            directory: Directorio con plugins
        """
        if not directory.exists():
            logger.warning(f"Directorio de plugins no existe: {directory}")
            return
        
        for file_path in directory.glob("*.py"):
            try:
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Buscar clases que hereden de Plugin
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, Plugin) and 
                        attr != Plugin):
                        plugin = attr()
                        self.register_plugin(plugin)
            except Exception as e:
                logger.error(f"Error cargando plugin de {file_path}: {e}")

    def execute_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """
        Ejecutar un hook.

        Args:
            hook_name: Nombre del hook
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre

        Returns:
            Resultado del hook
        """
        if hook_name not in self.hooks:
            return None
        
        result = None
        for hook in self.hooks[hook_name]:
            try:
                result = hook(*args, **kwargs)
                if result is not None:
                    break  # Si un hook retorna algo, usar ese resultado
            except Exception as e:
                logger.error(f"Error ejecutando hook {hook_name}: {e}")
        
        return result

    def get_plugins(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de plugins registrados.

        Returns:
            Lista de información de plugins
        """
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "enabled": plugin.enabled
            }
            for plugin in self.plugins.values()
        ]


# Plugin de ejemplo: Sanitizador
class SanitizerPlugin(Plugin):
    """Plugin para sanitizar contenido"""

    def __init__(self):
        super().__init__("sanitizer", "1.0.0")

    def on_add(self, content: str, addition: str, **kwargs) -> Optional[str]:
        """Sanitizar contenido antes de agregar"""
        import re
        # Remover caracteres de control
        addition = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', addition)
        return addition

    def on_remove(self, content: str, pattern: str, **kwargs) -> Optional[str]:
        """No modificar en remove"""
        return None


# Plugin de ejemplo: Logger
class LoggerPlugin(Plugin):
    """Plugin para logging avanzado"""

    def __init__(self):
        super().__init__("logger", "1.0.0")

    def on_add(self, content: str, addition: str, **kwargs) -> Optional[str]:
        """Log antes de agregar"""
        logger.info(f"Agregando contenido: {addition[:50]}...")
        return None

    def on_remove(self, content: str, pattern: str, **kwargs) -> Optional[str]:
        """Log antes de eliminar"""
        logger.info(f"Eliminando patrón: {pattern}")
        return None

    def on_after_add(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Log después de agregar"""
        logger.info(f"Adición completada: {result.get('success')}")
        return result

    def on_after_remove(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Log después de eliminar"""
        logger.info(f"Eliminación completada: {result.get('success')}")
        return result

