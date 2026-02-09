"""
Routing YAML Configuration
==========================

Sistema de configuración usando YAML para hyperparameters y settings.
Sigue mejores prácticas de configuración para proyectos ML.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available. YAML config features will be disabled.")


class YAMLConfigLoader:
    """Cargador de configuración YAML."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo YAML.
        
        Args:
            config_path: Ruta al archivo YAML
        
        Returns:
            Diccionario de configuración
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML config loading")
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config or {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        Guardar configuración a archivo YAML.
        
        Args:
            config: Diccionario de configuración
            config_path: Ruta donde guardar
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML config saving")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    @staticmethod
    def create_default_config(output_path: str):
        """
        Crear archivo de configuración por defecto.
        
        Args:
            output_path: Ruta donde crear el archivo
        """
        default_config = {
            'model': {
                'gnn': {
                    'input_dim': 10,
                    'hidden_dim': 128,
                    'output_dim': 64,
                    'num_layers': 3,
                    'num_heads': 8,
                    'dropout': 0.1
                },
                'transformer': {
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 6,
                    'dim_feedforward': 512,
                    'dropout': 0.1
                },
                'deep_optimizer': {
                    'input_dim': 10,
                    'hidden_dims': [256, 128, 64],
                    'dropout': 0.2
                }
            },
            'training': {
                'num_epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'gradient_clip_norm': 1.0,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 1e-6,
                'validation_split': 0.2,
                'use_mixed_precision': True,
                'gradient_accumulation_steps': 1
            },
            'routing': {
                'cache_max_size': 10000,
                'cache_ttl': 3600.0,
                'precomputation_max_nodes': 100,
                'enable_precomputation': False,
                'batch_processing_enabled': True,
                'enable_performance_monitoring': True
            },
            'hardware': {
                'use_gpu': True,
                'num_workers': 4,
                'pin_memory': True
            },
            'experiment': {
                'project_name': 'routing_ai',
                'use_wandb': True,
                'use_tensorboard': True,
                'checkpoint_dir': 'checkpoints'
            }
        }
        
        YAMLConfigLoader.save_config(default_config, output_path)
        logger.info(f"Default configuration created at {output_path}")

