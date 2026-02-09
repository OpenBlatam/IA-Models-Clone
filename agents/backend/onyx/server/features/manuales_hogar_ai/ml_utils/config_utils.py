"""
Config Utils - Utilidades de Gestión de Configuraciones
========================================================

Utilidades para gestión de configuraciones YAML/JSON.
"""

import logging
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
import copy

logger = logging.getLogger(__name__)

# Intentar importar yaml
try:
    import yaml
    _has_yaml = True
except ImportError:
    _has_yaml = False
    logger.warning("yaml not available, YAML features will be limited")


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    gradient_clip: float = 1.0
    use_amp: bool = True
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Crear desde diccionario."""
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """Configuración de modelo."""
    model_type: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    activation: str = "gelu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Crear desde diccionario."""
        return cls(**config_dict)


class ConfigManager:
    """
    Gestor de configuraciones.
    """
    
    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Cargar configuración YAML.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Configuración
        """
        if not _has_yaml:
            raise ImportError("yaml is required for YAML loading")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], filepath: Union[str, Path]):
        """
        Guardar configuración YAML.
        
        Args:
            config: Configuración
            filepath: Ruta del archivo
        """
        if not _has_yaml:
            raise ImportError("yaml is required for YAML saving")
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Cargar configuración JSON.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Configuración
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def save_json(config: Dict[str, Any], filepath: Union[str, Path]):
        """
        Guardar configuración JSON.
        
        Args:
            config: Configuración
            filepath: Ruta del archivo
        """
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fusionar configuraciones.
        
        Args:
            base_config: Configuración base
            override_config: Configuración a aplicar
            
        Returns:
            Configuración fusionada
        """
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def validate_config(
        config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validar configuración.
        
        Args:
            config: Configuración
            schema: Esquema de validación
            
        Returns:
            Tupla (es_válido, errores)
        """
        errors = []
        
        for key, expected_type in schema.items():
            if key not in config:
                errors.append(f"Missing key: {key}")
            elif not isinstance(config[key], expected_type):
                errors.append(
                    f"Key {key} has wrong type: expected {expected_type}, got {type(config[key])}"
                )
        
        return len(errors) == 0, errors


def create_default_config() -> Dict[str, Any]:
    """
    Crear configuración por defecto.
    
    Returns:
        Configuración por defecto
    """
    return {
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'use_amp': True
        },
        'model': {
            'model_type': 'transformer',
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'dropout': 0.1
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
    }




