"""
Config Manager - Gestor de configuración
=========================================

Sistema para gestionar configuraciones usando YAML y dataclasses.
Sigue mejores prácticas de configuración de proyectos.
"""

import logging
from typing import Dict, List, Optional, Any, Type, TypeVar
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available. Install with: pip install pyyaml")

T = TypeVar('T')


class ConfigManager:
    """Gestor de configuración"""
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo YAML.
        
        Args:
            filepath: Ruta al archivo YAML
        
        Returns:
            Diccionario con configuración
        """
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML not available. Install with: pip install pyyaml")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return config or {}
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}", exc_info=True)
            raise
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], filepath: str) -> bool:
        """
        Guardar configuración a archivo YAML.
        
        Args:
            config: Diccionario de configuración
            filepath: Ruta donde guardar
        
        Returns:
            True si se guardó exitosamente
        """
        if not YAML_AVAILABLE:
            raise RuntimeError("PyYAML not available")
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving YAML config: {e}", exc_info=True)
            return False
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo JSON.
        
        Args:
            filepath: Ruta al archivo JSON
        
        Returns:
            Diccionario con configuración
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return config
        except Exception as e:
            logger.error(f"Error loading JSON config: {e}", exc_info=True)
            raise
    
    @staticmethod
    def save_json(config: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
        """
        Guardar configuración a archivo JSON.
        
        Args:
            config: Diccionario de configuración
            filepath: Ruta donde guardar
            indent: Indentación del JSON
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=indent, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON config: {e}", exc_info=True)
            return False
    
    @staticmethod
    def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
        """
        Convertir dataclass a diccionario.
        
        Args:
            obj: Instancia de dataclass
        
        Returns:
            Diccionario
        """
        return asdict(obj)
    
    @staticmethod
    def dict_to_dataclass(data: Dict[str, Any], dataclass_type: Type[T]) -> T:
        """
        Convertir diccionario a dataclass.
        
        Args:
            data: Diccionario de datos
            dataclass_type: Tipo de dataclass
        
        Returns:
            Instancia de dataclass
        """
        # Filter data to only include fields from dataclass
        field_names = {f.name for f in dataclass_type.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        return dataclass_type(**filtered_data)
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusionar dos configuraciones (override tiene prioridad).
        
        Args:
            base: Configuración base
            override: Configuración que sobrescribe
        
        Returns:
            Configuración fusionada
        """
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(
        config: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Validar que la configuración tenga las claves requeridas.
        
        Args:
            config: Configuración a validar
            required_keys: Claves requeridas
            optional_keys: Claves opcionales (para logging)
        
        Returns:
            True si es válida
        
        Raises:
            ValueError: Si faltan claves requeridas
        """
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        if optional_keys:
            unknown_keys = [key for key in config.keys() if key not in required_keys + optional_keys]
            if unknown_keys:
                logger.warning(f"Unknown config keys: {unknown_keys}")
        
        return True




