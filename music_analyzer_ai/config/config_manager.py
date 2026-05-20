"""
Configuration Management
YAML-based configuration for hyperparameters and model settings
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available, config loading will not work")


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    use_mixed_precision: bool = True
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "DeepGenreClassifier"
    input_size: int = 169
    num_genres: int = 10
    hidden_layers: list = None
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    use_residual: bool = True
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [512, 512, 256, 256, 128, 128]


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "./data"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True
    cache_dir: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_name: str = "experiment_001"
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_dir: str = "./logs"
    project_name: Optional[str] = None


class ConfigManager:
    """
    Configuration manager for loading and saving YAML configs
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for config loading")
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for config saving")
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved config to {config_path}")
    
    @staticmethod
    def create_default_config(output_path: str):
        """Create default configuration file"""
        config = {
            "training": asdict(TrainingConfig()),
            "model": asdict(ModelConfig()),
            "data": asdict(DataConfig()),
            "experiment": asdict(ExperimentConfig())
        }
        
        ConfigManager.save_config(config, output_path)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration"""
        required_sections = ["training", "model", "data", "experiment"]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Validate training config
        training = config["training"]
        if training.get("epochs", 0) <= 0:
            logger.error("epochs must be > 0")
            return False
        
        if training.get("batch_size", 0) <= 0:
            logger.error("batch_size must be > 0")
            return False
        
        # Validate data config
        data = config["data"]
        ratios = [data.get("train_ratio", 0), data.get("val_ratio", 0), data.get("test_ratio", 0)]
        if abs(sum(ratios) - 1.0) > 1e-6:
            logger.error("Data ratios must sum to 1.0")
            return False
        
        logger.info("Configuration validated successfully")
        return True


def load_training_config(config_path: str) -> TrainingConfig:
    """Load training configuration"""
    config = ConfigManager.load_config(config_path)
    return TrainingConfig(**config["training"])


def load_model_config(config_path: str) -> ModelConfig:
    """Load model configuration"""
    config = ConfigManager.load_config(config_path)
    return ModelConfig(**config["model"])


def load_data_config(config_path: str) -> DataConfig:
    """Load data configuration"""
    config = ConfigManager.load_config(config_path)
    return DataConfig(**config["data"])

