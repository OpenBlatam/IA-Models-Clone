"""
Configuration Loader
====================

Cargador de configuraciones YAML para modelos y entrenamiento.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Cargador de configuraciones YAML.
    """
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """
        Cargar archivo YAML.
        
        Args:
            path: Ruta al archivo YAML
            
        Returns:
            Diccionario con configuración
        """
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], path: str):
        """
        Guardar configuración en YAML.
        
        Args:
            config: Diccionario de configuración
            path: Ruta donde guardar
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combinar configuraciones.
        
        Args:
            base_config: Configuración base
            override_config: Configuración a sobreescribir
            
        Returns:
            Configuración combinada
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged


def load_config(config_path: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Cargar configuración desde archivo.
    
    Args:
        config_path: Ruta al archivo de configuración
        default_config: Configuración por defecto (opcional)
        
    Returns:
        Configuración cargada
    """
    if default_config is None:
        default_config = {}
    
    if os.path.exists(config_path):
        loaded_config = ConfigLoader.load_yaml(config_path)
        config = ConfigLoader.merge_configs(default_config, loaded_config)
        logger.info(f"Configuración cargada desde: {config_path}")
    else:
        config = default_config
        logger.warning(f"Archivo de configuración no encontrado: {config_path}, usando defaults")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Guardar configuración en archivo.
    
    Args:
        config: Configuración a guardar
        config_path: Ruta donde guardar
    """
    ConfigLoader.save_yaml(config, config_path)
    logger.info(f"Configuración guardada en: {config_path}")


