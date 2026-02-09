"""
Configuration Manager
=====================

Gestor avanzado de configuración con validación y hot-reload.
"""

import json
import yaml
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """Sección de configuración."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    validators: Dict[str, Callable] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validar sección."""
        for key, validator in self.validators.items():
            if key in self.data:
                try:
                    validator(self.data[key])
                except Exception as e:
                    raise ConfigurationError(
                        f"Validation failed for {self.name}.{key}: {e}"
                    )
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Establecer valor de configuración."""
        if key in self.validators:
            self.validators[key](value)
        self.data[key] = value


class ConfigFileHandler(FileSystemEventHandler):
    """Handler para cambios en archivos de configuración."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        """Inicializar handler."""
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Manejar modificación de archivo."""
        if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Configuration file changed: {event.src_path}")
            self.config_manager.reload()


class ConfigManager:
    """
    Gestor avanzado de configuración.
    
    Características:
    - Carga desde múltiples formatos (JSON, YAML)
    - Validación de configuración
    - Hot-reload automático
    - Secciones organizadas
    - Callbacks para cambios
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializar gestor de configuración.
        
        Args:
            config_file: Ruta al archivo de configuración
        """
        self.config_file = config_file
        self.sections: Dict[str, ConfigSection] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.observer: Optional[Observer] = None
        
        if config_file:
            self.load_from_file(config_file)
        
        if self.config_file:
            self._setup_file_watcher()
    
    def _setup_file_watcher(self):
        """Configurar observador de archivos para hot-reload."""
        try:
            self.observer = Observer()
            handler = ConfigFileHandler(self)
            config_path = Path(self.config_file)
            self.observer.schedule(handler, str(config_path.parent), recursive=False)
            self.observer.start()
            logger.info(f"File watcher started for {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not setup file watcher: {e}")
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Cargar configuración desde archivo.
        
        Args:
            filepath: Ruta al archivo
            
        Returns:
            True si fue exitoso
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"Config file {filepath} does not exist")
                return False
            
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._load_data(data)
            self.config_file = filepath
            logger.info(f"Configuration loaded from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading config from {filepath}: {e}")
            return False
    
    def _load_data(self, data: Dict[str, Any]) -> None:
        """Cargar datos de configuración."""
        for section_name, section_data in data.items():
            if section_name not in self.sections:
                self.sections[section_name] = ConfigSection(name=section_name)
            self.sections[section_name].data.update(section_data)
    
    def reload(self) -> bool:
        """Recargar configuración desde archivo."""
        if self.config_file:
            return self.load_from_file(self.config_file)
        return False
    
    def get_section(self, name: str) -> Optional[ConfigSection]:
        """Obtener sección de configuración."""
        return self.sections.get(name)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración.
        
        Args:
            section: Nombre de la sección
            key: Clave
            default: Valor por defecto
            
        Returns:
            Valor de configuración
        """
        section_obj = self.sections.get(section)
        if section_obj:
            return section_obj.get(key, default)
        return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Establecer valor de configuración.
        
        Args:
            section: Nombre de la sección
            key: Clave
            value: Valor
        """
        if section not in self.sections:
            self.sections[section] = ConfigSection(name=section)
        
        self.sections[section].set(key, value)
        self._notify_callbacks(section, key, value)
    
    def register_callback(
        self,
        section: str,
        callback: Callable[[str, Any], None]
    ) -> None:
        """
        Registrar callback para cambios en sección.
        
        Args:
            section: Nombre de la sección
            callback: Función callback
        """
        if section not in self.callbacks:
            self.callbacks[section] = []
        self.callbacks[section].append(callback)
    
    def _notify_callbacks(self, section: str, key: str, value: Any) -> None:
        """Notificar callbacks de cambios."""
        if section in self.callbacks:
            for callback in self.callbacks[section]:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.error(f"Error in config callback: {e}")
    
    def save_to_file(self, filepath: Optional[str] = None) -> bool:
        """
        Guardar configuración a archivo.
        
        Args:
            filepath: Ruta al archivo (usa self.config_file si None)
            
        Returns:
            True si fue exitoso
        """
        filepath = filepath or self.config_file
        if not filepath:
            logger.error("No filepath specified for saving config")
            return False
        
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                section.name: section.data
                for section in self.sections.values()
            }
            
            with open(path, 'w') as f:
                if path.suffix in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    json.dump(data, f, indent=2)
            
            logger.info(f"Configuration saved to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving config to {filepath}: {e}")
            return False
    
    def validate_all(self) -> bool:
        """Validar toda la configuración."""
        for section in self.sections.values():
            section.validate()
        return True
    
    def shutdown(self):
        """Cerrar gestor de configuración."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

