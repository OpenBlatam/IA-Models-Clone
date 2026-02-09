"""
YAML-based Configuration Management for Training
"""

import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "autoencoder"  # "autoencoder", "vit", "diffusion"
    input_channels: int = 3
    input_size: tuple = (224, 224)
    latent_dim: int = 128
    num_classes: int = 10
    pretrained: bool = True
    model_name: str = "google/vit-base-patch16-224"


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer_type: str = "adam"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau", "none"
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100
    eta_min: float = 0.0


@dataclass
class DataConfig:
    """Data configuration"""
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    test_data_path: Optional[str] = None
    image_size: tuple = (224, 224)
    normalize: bool = True
    augmentation: bool = True


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    use_wandb: bool = False
    wandb_project: str = "quality-control-ai"
    wandb_entity: Optional[str] = None
    use_tensorboard: bool = True


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    data: DataConfig
    experiment: ExperimentConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load configuration from YAML file
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
            scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
            data=DataConfig(**config_dict.get("data", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file
        
        Args:
            yaml_path: Path to save YAML file
        """
        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "optimizer": asdict(self.optimizer),
            "scheduler": asdict(self.scheduler),
            "data": asdict(self.data),
            "experiment": asdict(self.experiment)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {yaml_path}")
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration"""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            optimizer=OptimizerConfig(),
            scheduler=SchedulerConfig(),
            data=DataConfig(),
            experiment=ExperimentConfig()
        )


def create_default_config_file(path: str = "config.yaml"):
    """Create a default configuration YAML file"""
    config = Config.default()
    config.to_yaml(path)
    logger.info(f"Default configuration created at {path}")

