"""
Config Loader
=============

Cargador de configuraciones desde YAML.
"""

import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Cargador de configuraciones."""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Cargar configuración desde YAML.
        
        Args:
            file_path: Ruta al archivo YAML
            
        Returns:
            Diccionario con configuración
        """
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            raise
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], file_path: str):
        """
        Guardar configuración a YAML.
        
        Args:
            config: Configuración
            file_path: Ruta al archivo YAML
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {file_path}")
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
            raise


def load_config(file_path: str) -> Dict[str, Any]:
    """Cargar configuración."""
    return ConfigLoader.load_yaml(file_path)


def save_config(config: Dict[str, Any], file_path: str):
    """Guardar configuración."""
    ConfigLoader.save_yaml(config, file_path)


