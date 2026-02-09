"""
Dynamic Configuration System
==============================

Sistema de configuración dinámica con hot-reload.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Cambio de configuración."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


class ConfigFileHandler(FileSystemEventHandler):
    """Handler para cambios en archivos de configuración."""
    
    def __init__(self, config_manager: "DynamicConfigManager"):
        """Inicializar handler."""
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Manejar modificación de archivo."""
        if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Config file modified: {event.src_path}")
            self.config_manager.reload_config()


class DynamicConfigManager:
    """
    Gestor de configuración dinámica.
    
    Permite cambios en tiempo real sin reiniciar.
    """
    
    def __init__(self, config_file: str = "config/dynamic_config.json"):
        """
        Inicializar gestor de configuración dinámica.
        
        Args:
            config_file: Archivo de configuración
        """
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config: Dict[str, Any] = {}
        self.change_history: List[ConfigChange] = []
        self.watchers: List[Callable[[str, Any, Any], None]] = []
        self.observer: Optional[Observer] = None
        
        # Cargar configuración inicial
        self.load_config()
        
        # Iniciar observador de archivos
        self.start_watching()
    
    def load_config(self) -> None:
        """Cargar configuración desde archivo."""
        if self.config_file.exists():
            try:
                if self.config_file.suffix in ['.yaml', '.yml']:
                    with open(self.config_file, 'r') as f:
                        self.config = yaml.safe_load(f) or {}
                else:
                    with open(self.config_file, 'r') as f:
                        self.config = json.load(f)
                
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.config = {}
        else:
            # Crear configuración por defecto
            self.config = self._default_config()
            self.save_config()
    
    def save_config(self) -> None:
        """Guardar configuración en archivo."""
        try:
            if self.config_file.suffix in ['.yaml', '.yml']:
                with open(self.config_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Configuración por defecto."""
        return {
            "optimization": {
                "max_iterations": 100,
                "learning_rate": 0.001,
                "cache_enabled": True
            },
            "performance": {
                "enable_caching": True,
                "cache_size": 128,
                "enable_profiling": False
            },
            "monitoring": {
                "metrics_enabled": True,
                "health_checks_enabled": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración.
        
        Args:
            key: Clave (puede usar dot notation, ej: "optimization.max_iterations")
            default: Valor por defecto
            
        Returns:
            Valor de configuración
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, save: bool = True) -> None:
        """
        Establecer valor de configuración.
        
        Args:
            key: Clave (puede usar dot notation)
            value: Nuevo valor
            save: Guardar en archivo
        """
        keys = key.split('.')
        old_value = self.get(key)
        
        # Crear estructura anidada si no existe
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Establecer valor
        config[keys[-1]] = value
        
        # Registrar cambio
        change = ConfigChange(key=key, old_value=old_value, new_value=value)
        self.change_history.append(change)
        
        # Notificar watchers
        for watcher in self.watchers:
            try:
                watcher(key, old_value, value)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
        
        # Guardar si se solicita
        if save:
            self.save_config()
    
    def watch(self, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Registrar callback para cambios de configuración.
        
        Args:
            callback: Función callback(key, old_value, new_value)
        """
        self.watchers.append(callback)
    
    def reload_config(self) -> None:
        """Recargar configuración desde archivo."""
        old_config = self.config.copy()
        self.load_config()
        
        # Detectar cambios
        self._detect_changes(old_config, self.config)
    
    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], prefix: str = "") -> None:
        """Detectar cambios en configuración."""
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old_config:
                # Nueva clave
                change = ConfigChange(key=full_key, old_value=None, new_value=new_config[key])
                self.change_history.append(change)
                for watcher in self.watchers:
                    try:
                        watcher(full_key, None, new_config[key])
                    except Exception as e:
                        logger.error(f"Error in config watcher: {e}")
            
            elif key not in new_config:
                # Clave eliminada
                change = ConfigChange(key=full_key, old_value=old_config[key], new_value=None)
                self.change_history.append(change)
                for watcher in self.watchers:
                    try:
                        watcher(full_key, old_config[key], None)
                    except Exception as e:
                        logger.error(f"Error in config watcher: {e}")
            
            elif isinstance(old_config[key], dict) and isinstance(new_config[key], dict):
                # Recursión para diccionarios anidados
                self._detect_changes(old_config[key], new_config[key], full_key)
            
            elif old_config[key] != new_config[key]:
                # Valor cambiado
                change = ConfigChange(key=full_key, old_value=old_config[key], new_value=new_config[key])
                self.change_history.append(change)
                for watcher in self.watchers:
                    try:
                        watcher(full_key, old_config[key], new_config[key])
                    except Exception as e:
                        logger.error(f"Error in config watcher: {e}")
    
    def start_watching(self) -> None:
        """Iniciar observación de archivos."""
        try:
            self.observer = Observer()
            handler = ConfigFileHandler(self)
            self.observer.schedule(handler, str(self.config_file.parent), recursive=False)
            self.observer.start()
            logger.info(f"Started watching config file: {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not start file watcher: {e}")
    
    def stop_watching(self) -> None:
        """Detener observación de archivos."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching config file")
    
    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        """
        Obtener historial de cambios.
        
        Args:
            limit: Límite de cambios
            
        Returns:
            Lista de cambios
        """
        return self.change_history[-limit:]


# Instancia global
_dynamic_config_manager: Optional[DynamicConfigManager] = None


def get_dynamic_config_manager(config_file: str = "config/dynamic_config.json") -> DynamicConfigManager:
    """Obtener instancia global del gestor de configuración dinámica."""
    global _dynamic_config_manager
    if _dynamic_config_manager is None:
        _dynamic_config_manager = DynamicConfigManager(config_file)
    return _dynamic_config_manager






