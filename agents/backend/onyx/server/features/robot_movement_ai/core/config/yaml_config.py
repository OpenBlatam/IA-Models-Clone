"""
YAML Configuration Management
==============================

Gestión de configuración usando archivos YAML para modelos y entrenamiento.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass, field, asdict
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuración de modelo."""
    name: str
    type: str  # 'transformer', 'diffusion', 'mlp', etc.
    input_size: int
    output_size: int
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = 'gelu'
    use_batch_norm: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 10
    save_best: bool = True
    checkpoint_dir: str = 'checkpoints'
    experiment_name: str = 'experiment'


@dataclass
class DataConfig:
    """Configuración de datos."""
    data_path: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    trajectory_length: int = 100
    trajectory_dim: int = 3
    normalize: bool = True
    augment: bool = False
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """Configuración completa de experimento."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str = 'experiment'
    description: str = ''
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class YAMLConfigManager:
    """
    Gestor de configuración YAML.
    """
    
    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        """
        Cargar configuración desde archivo YAML.
        
        Args:
            config_path: Ruta al archivo YAML
            
        Returns:
            Configuración del experimento
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parsear configuración
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        experiment_config = ExperimentConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            experiment_name=config_dict.get('experiment_name', 'experiment'),
            description=config_dict.get('description', ''),
            tags=config_dict.get('tags', []),
            metadata=config_dict.get('metadata', {})
        )
        
        logger.info(f"Configuration loaded from {config_path}")
        return experiment_config
    
    @staticmethod
    def save_config(
        config: ExperimentConfig,
        config_path: str
    ):
        """
        Guardar configuración a archivo YAML.
        
        Args:
            config: Configuración del experimento
            config_path: Ruta donde guardar
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    @staticmethod
    def create_default_config(config_path: str):
        """
        Crear archivo de configuración por defecto.
        
        Args:
            config_path: Ruta donde crear el archivo
        """
        default_config = ExperimentConfig(
            model=ModelConfig(
                name='default_model',
                type='transformer',
                input_size=3,
                output_size=3
            ),
            training=TrainingConfig(),
            data=DataConfig(data_path='data/trajectories.json')
        )
        
        YAMLConfigManager.save_config(default_config, config_path)
        logger.info(f"Default configuration created at {config_path}")


def load_yaml_config(config_path: str) -> ExperimentConfig:
    """
    Función helper para cargar configuración.
    
    Args:
        config_path: Ruta al archivo YAML
        
    Returns:
        Configuración del experimento
    """
    return YAMLConfigManager.load_config(config_path)













