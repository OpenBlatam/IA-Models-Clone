"""
Training Configuration Management
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
    model_type: str = "transformer"
    model_name: str = "bert-base-uncased"
    num_classes: int = 3
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer_type: str = "adamw"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""
    scheduler_type: str = "linear"
    warmup_steps: int = 100
    num_training_steps: int = 1000


@dataclass
class DataConfig:
    """Data configuration"""
    train_data_path: str = "./data/train"
    val_data_path: str = "./data/val"
    test_data_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    use_wandb: bool = False
    wandb_project: str = "addition-removal-ai"
    use_tensorboard: bool = True


@dataclass
class Config:
    """Complete training configuration"""
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    data: DataConfig
    experiment: ExperimentConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load from YAML"""
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
        """Save to YAML"""
        config_dict = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "optimizer": asdict(self.optimizer),
            "scheduler": asdict(self.scheduler),
            "data": asdict(self.data),
            "experiment": asdict(self.experiment)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Config saved to {yaml_path}")
    
    @classmethod
    def default(cls) -> "Config":
        """Create default config"""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            optimizer=OptimizerConfig(),
            scheduler=SchedulerConfig(),
            data=DataConfig(),
            experiment=ExperimentConfig()
        )


def create_default_config_file(path: str = "training_config.yaml"):
    """Create default config file"""
    config = Config.default()
    config.to_yaml(path)
    logger.info(f"Default config created at {path}")

