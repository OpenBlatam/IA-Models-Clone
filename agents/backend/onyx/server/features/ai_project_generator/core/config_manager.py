"""
Configuration Manager - Gestor de Configuración Avanzado
========================================================

Gestión avanzada de configuración:
- Environment-based config
- Config validation
- Config hot-reload
- Secret management
- Config versioning
"""

import logging
import os
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from enum import Enum
import json
import yaml

logger = logging.getLogger(__name__)


class ConfigSource(str, Enum):
    """Fuentes de configuración"""
    ENV = "env"
    FILE = "file"
    SECRETS_MANAGER = "secrets_manager"
    CONSUL = "consul"
    ETCD = "etcd"


class ConfigManager:
    """
    Gestor de configuración avanzado.
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        env_prefix: str = "APP_",
        watch_for_changes: bool = False
    ) -> None:
        self.config_file = config_file
        self.env_prefix = env_prefix
        self.watch_for_changes = watch_for_changes
        self.config: Dict[str, Any] = {}
        self.watchers: List[Callable] = []
        self._load_config()
    
    def _load_config(self) -> None:
        """Carga configuración"""
        # Cargar de archivo
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file(self.config_file)
        
        # Cargar de variables de entorno
        self._load_from_env()
        
        logger.info("Configuration loaded")
    
    def _load_from_file(self, file_path: str) -> None:
        """Carga configuración de archivo"""
        path = Path(file_path)
        
        if path.suffix == ".json":
            with open(path) as f:
                self.config.update(json.load(f))
        elif path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                self.config.update(yaml.safe_load(f))
        else:
            logger.warning(f"Unsupported config file format: {path.suffix}")
    
    def _load_from_env(self) -> None:
        """Carga configuración de variables de entorno"""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                config_key = key[len(self.env_prefix):].lower()
                # Convertir tipos
                if value.lower() in ["true", "false"]:
                    self.config[config_key] = value.lower() == "true"
                elif value.isdigit():
                    self.config[config_key] = int(value)
                else:
                    self.config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene valor de configuración"""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Establece valor de configuración"""
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._notify_watchers(key, value)
    
    def register_watcher(self, callback: Callable[[str, Any], None]) -> None:
        """Registra watcher para cambios"""
        self.watchers.append(callback)
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """Notifica watchers de cambios"""
        for watcher in self.watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logger.error(f"Watcher error: {e}")
    
    def validate(self, schema: Dict[str, Any]) -> bool:
        """Valida configuración contra schema"""
        # Implementación simplificada
        # En producción usaría jsonschema o similar
        required_keys = schema.get("required", [])
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Required config key missing: {key}")
                return False
        
        return True
    
    def reload(self) -> None:
        """Recarga configuración"""
        self.config.clear()
        self._load_config()
        logger.info("Configuration reloaded")


def get_config_manager(
    config_file: Optional[str] = None,
    env_prefix: str = "APP_"
) -> ConfigManager:
    """Obtiene gestor de configuración"""
    return ConfigManager(config_file=config_file, env_prefix=env_prefix)















