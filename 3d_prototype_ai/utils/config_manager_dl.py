"""
Deep Learning Config Manager - Gestor de configuración para DL
==================================================================
Gestión de configuraciones YAML para modelos y entrenamiento
"""

import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuración de modelo"""
    name: str
    type: str  # "transformer", "cnn", "rnn", etc.
    architecture: Dict[str, Any]
    pretrained: bool = True
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "linear"
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class DataConfig:
    """Configuración de datos"""
    dataset_path: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class ExperimentConfig:
    """Configuración completa de experimento"""
    experiment_name: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    use_wandb: bool = True
    use_tensorboard: bool = True
    output_dir: str = "./outputs"
    seed: int = 42


class DLConfigManager:
    """Gestor de configuración para Deep Learning"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, ExperimentConfig] = {}
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """Carga configuración desde archivo YAML"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Convertir a dataclasses
        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])
        data_config = DataConfig(**config_dict["data"])
        
        experiment_config = ExperimentConfig(
            experiment_name=config_dict.get("experiment_name", "default"),
            model=model_config,
            training=training_config,
            data=data_config,
            use_wandb=config_dict.get("use_wandb", True),
            use_tensorboard=config_dict.get("use_tensorboard", True),
            output_dir=config_dict.get("output_dir", "./outputs"),
            seed=config_dict.get("seed", 42)
        )
        
        config_id = config_path.stem
        self.configs[config_id] = experiment_config
        
        logger.info(f"Loaded config: {config_id}")
        return experiment_config
    
    def save_config(self, config: ExperimentConfig, config_name: str):
        """Guarda configuración a archivo YAML"""
        config_dict = {
            "experiment_name": config.experiment_name,
            "model": asdict(config.model),
            "training": asdict(config.training),
            "data": asdict(config.data),
            "use_wandb": config.use_wandb,
            "use_tensorboard": config.use_tensorboard,
            "output_dir": config.output_dir,
            "seed": config.seed
        }
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.configs[config_name] = config
        logger.info(f"Saved config: {config_name}")
    
    def create_default_config(self, experiment_name: str) -> ExperimentConfig:
        """Crea configuración por defecto"""
        model_config = ModelConfig(
            name="gpt2",
            type="transformer",
            architecture={"hidden_size": 768, "num_layers": 12}
        )
        
        training_config = TrainingConfig()
        data_config = DataConfig(dataset_path="./data")
        
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            model=model_config,
            training=training_config,
            data=data_config
        )
        
        return experiment_config
    
    def get_config(self, config_name: str) -> Optional[ExperimentConfig]:
        """Obtiene configuración"""
        return self.configs.get(config_name)




