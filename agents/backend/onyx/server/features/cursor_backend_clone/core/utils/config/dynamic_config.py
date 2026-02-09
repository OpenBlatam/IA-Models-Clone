"""
Dynamic Configuration - Configuración Dinámica
==============================================

Sistema para gestión dinámica de configuración con validación y hot-reload.
"""

import logging
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """Tipos de configuración"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


@dataclass
class ConfigItem:
    """Item de configuración"""
    key: str
    value: Any
    config_type: ConfigType
    description: Optional[str] = None
    default_value: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validar valor de configuración.
        
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        # Validar tipo
        if self.config_type == ConfigType.STRING and not isinstance(self.value, str):
            return False, f"Expected string, got {type(self.value).__name__}"
        elif self.config_type == ConfigType.INTEGER and not isinstance(self.value, int):
            return False, f"Expected integer, got {type(self.value).__name__}"
        elif self.config_type == ConfigType.FLOAT and not isinstance(self.value, (int, float)):
            return False, f"Expected float, got {type(self.value).__name__}"
        elif self.config_type == ConfigType.BOOLEAN and not isinstance(self.value, bool):
            return False, f"Expected boolean, got {type(self.value).__name__}"
        elif self.config_type == ConfigType.LIST and not isinstance(self.value, list):
            return False, f"Expected list, got {type(self.value).__name__}"
        elif self.config_type == ConfigType.DICT and not isinstance(self.value, dict):
            return False, f"Expected dict, got {type(self.value).__name__}"
        
        # Validar con función personalizada
        if self.validator:
            try:
                if not self.validator(self.value):
                    return False, "Custom validation failed"
            except Exception as e:
                return False, f"Validation error: {str(e)}"
        
        return True, None


class DynamicConfigManager:
    """
    Gestor de configuración dinámica.
    
    Permite cambiar configuración en tiempo de ejecución con validación.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config: Dict[str, ConfigItem] = {}
        self.config_file = Path(config_file) if config_file else None
        self.watchers: List[Callable[[str, Any, Any], None]] = []  # key, old_value, new_value
        self._load_from_file()
    
    def register(
        self,
        key: str,
        value: Any,
        config_type: ConfigType,
        description: Optional[str] = None,
        default_value: Optional[Any] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        **metadata
    ) -> ConfigItem:
        """
        Registrar item de configuración.
        
        Args:
            key: Clave de configuración
            value: Valor inicial
            config_type: Tipo de configuración
            description: Descripción
            default_value: Valor por defecto
            validator: Función de validación
            **metadata: Metadata adicional
            
        Returns:
            ConfigItem creado
        """
        item = ConfigItem(
            key=key,
            value=value,
            config_type=config_type,
            description=description,
            default_value=default_value or value,
            validator=validator,
            metadata=metadata
        )
        
        # Validar
        is_valid, error = item.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration value for {key}: {error}")
        
        self.config[key] = item
        logger.info(f"⚙️ Configuration registered: {key}")
        return item
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración.
        
        Args:
            key: Clave de configuración
            default: Valor por defecto si no existe
            
        Returns:
            Valor de configuración
        """
        if key in self.config:
            return self.config[key].value
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        validate: bool = True
    ) -> bool:
        """
        Establecer valor de configuración.
        
        Args:
            key: Clave de configuración
            value: Nuevo valor
            validate: Si validar antes de establecer
            
        Returns:
            True si se estableció correctamente
        """
        if key not in self.config:
            logger.warning(f"Configuration key not found: {key}")
            return False
        
        old_value = self.config[key].value
        
        # Validar
        if validate:
            self.config[key].value = value
            is_valid, error = self.config[key].validate()
            if not is_valid:
                self.config[key].value = old_value
                logger.error(f"Invalid configuration value for {key}: {error}")
                return False
        
        # Establecer valor
        self.config[key].value = value
        self.config[key].updated_at = datetime.now()
        
        # Notificar watchers
        for watcher in self.watchers:
            try:
                watcher(key, old_value, value)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
        
        # Guardar en archivo
        self._save_to_file()
        
        logger.info(f"⚙️ Configuration updated: {key} = {value}")
        return True
    
    def watch(self, watcher: Callable[[str, Any, Any], None]) -> None:
        """
        Registrar watcher para cambios de configuración.
        
        Args:
            watcher: Función que se llama cuando cambia una configuración
        """
        self.watchers.append(watcher)
        logger.debug("⚙️ Configuration watcher registered")
    
    def get_all(self) -> Dict[str, Any]:
        """Obtener todas las configuraciones"""
        return {key: item.value for key, item in self.config.items()}
    
    def get_item(self, key: str) -> Optional[ConfigItem]:
        """Obtener item de configuración completo"""
        return self.config.get(key)
    
    def _load_from_file(self) -> None:
        """Cargar configuración desde archivo"""
        if not self.config_file or not self.config_file.exists():
            return
        
        try:
            content = self.config_file.read_text(encoding="utf-8")
            data = json.loads(content)
            
            for key, value in data.items():
                if key in self.config:
                    self.config[key].value = value
                    self.config[key].updated_at = datetime.now()
            
            logger.info(f"⚙️ Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}")
    
    def _save_to_file(self) -> None:
        """Guardar configuración en archivo"""
        if not self.config_file:
            return
        
        try:
            data = {key: item.value for key, item in self.config.items()}
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            logger.debug(f"⚙️ Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration to file: {e}")




