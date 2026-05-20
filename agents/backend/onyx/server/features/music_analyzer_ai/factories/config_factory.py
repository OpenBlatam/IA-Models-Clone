"""
Config Factory - Create configuration instances
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    input_dim: int
    hidden_dims: list
    output_dim: int
    dropout: float = 0.1
    activation: str = "relu"


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    compile_model: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


class ConfigFactory:
    """
    Factory for creating configuration instances
    """
    
    @staticmethod
    def create_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
        """Create model config from dict"""
        return ModelConfig(**config_dict)
    
    @staticmethod
    def create_training_config(config_dict: Dict[str, Any]) -> TrainingConfig:
        """Create training config from dict"""
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def create_data_config(config_dict: Dict[str, Any]) -> DataConfig:
        """Create data config from dict"""
        return DataConfig(**config_dict)
    
    @staticmethod
    def load_from_yaml(path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def save_to_yaml(config: Any, path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def create_config(config_type: str, config_dict: Dict[str, Any]):
    """
    Convenience function to create config
    
    Args:
        config_type: Type of config (model, training, data)
        config_dict: Configuration dictionary
    
    Returns:
        Config instance
    """
    if config_type == "model":
        return ConfigFactory.create_model_config(config_dict)
    elif config_type == "training":
        return ConfigFactory.create_training_config(config_dict)
    elif config_type == "data":
        return ConfigFactory.create_data_config(config_dict)
    else:
        raise ValueError(f"Unknown config type: {config_type}")













