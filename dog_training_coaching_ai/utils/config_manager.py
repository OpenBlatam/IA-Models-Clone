"""
Configuration Manager
=====================
Utilidades avanzadas para gestión de configuración.
"""

import os
from typing import Any, Dict, Optional, Callable
from pathlib import Path
import json
from datetime import datetime

from .logger import get_logger
from ..config.app_config import get_config

logger = get_logger(__name__)


class ConfigManager:
    """Manager avanzado para configuración."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializar config manager.
        
        Args:
            config_file: Ruta opcional a archivo de configuración
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.watchers: Dict[str, List[Callable]] = {}
        self.load_config()
    
    def load_config(self):
        """Cargar configuración."""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded config from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        else:
            # Cargar desde variables de entorno
            self.config = self._load_from_env()
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Cargar configuración desde variables de entorno."""
        config = {}
        
        # Mapeo de variables de entorno
        env_mapping = {
            "DOG_TRAINING_AI_HOST": "host",
            "DOG_TRAINING_AI_PORT": "port",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "OPENROUTER_BASE_URL": "openrouter_base_url",
            "OPENROUTER_MODEL": "openrouter_model",
            "DEBUG": "debug",
            "REQUEST_TIMEOUT": "request_timeout",
            "MAX_RETRIES": "max_retries"
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                # Convertir tipos
                if config_key in ["port", "request_timeout", "max_retries"]:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif config_key == "debug":
                    value = value.lower() == "true"
                
                config[config_key] = value
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de configuración.
        
        Args:
            key: Clave de configuración
            default: Valor por defecto
            
        Returns:
            Valor de configuración
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any, notify: bool = True):
        """
        Establecer valor de configuración.
        
        Args:
            key: Clave de configuración
            value: Valor
            notify: Notificar watchers
        """
        old_value = self.config.get(key)
        self.config[key] = value
        
        if notify and key in self.watchers:
            for watcher in self.watchers[key]:
                try:
                    watcher(key, value, old_value)
                except Exception as e:
                    logger.error(f"Error in config watcher: {e}")
    
    def watch(self, key: str, callback: Callable):
        """
        Observar cambios en clave de configuración.
        
        Args:
            key: Clave a observar
            callback: Callback a ejecutar en cambios
        """
        if key not in self.watchers:
            self.watchers[key] = []
        
        self.watchers[key].append(callback)
    
    def save_config(self, file_path: Optional[str] = None):
        """
        Guardar configuración a archivo.
        
        Args:
            file_path: Ruta opcional del archivo
        """
        target_file = file_path or self.config_file
        if not target_file:
            logger.warning("No config file specified for saving")
            return
        
        try:
            with open(target_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved config to {target_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validar configuración.
        
        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        
        # Validar campos requeridos
        required_fields = ["openrouter_api_key"]
        for field in required_fields:
            if not self.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validar tipos
        if self.get("port") and not isinstance(self.get("port"), int):
            errors.append("Port must be an integer")
        
        if self.get("debug") and not isinstance(self.get("debug"), bool):
            warnings.append("Debug should be a boolean")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_all(self) -> Dict[str, Any]:
        """Obtener toda la configuración."""
        return self.config.copy()
    
    def reload(self):
        """Recargar configuración."""
        self.load_config()
        logger.info("Configuration reloaded")


# Instancia global
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Obtener instancia global de config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager



