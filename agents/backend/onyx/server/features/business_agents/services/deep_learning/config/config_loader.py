"""
Configuration Loader - YAML-based Configuration Management
===========================================================

Loads and manages configuration from YAML files for hyperparameters,
model settings, and training configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "custom_model"
    model_type: str = "transformer"  # transformer, cnn, lstm, diffusion
    architecture: Dict[str, Any] = field(default_factory=dict)
    initialization: str = "xavier_uniform"  # xavier_uniform, kaiming_uniform, etc.
    device: str = "auto"  # auto, cuda, cpu
    torch_dtype: str = "auto"  # auto, float16, bfloat16, float32


@dataclass
class DataConfig:
    """Data loading configuration."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    shuffle: bool = True
    validation_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration with best practices."""
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # adamw, adam, sgd, etc.
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # float16, bfloat16
    
    # Gradient handling
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Learning rate scheduling
    learning_rate_scheduler: str = "cosine"  # cosine, step, reduce_on_plateau, exponential, onecycle
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 0
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    early_stopping_mode: str = "min"  # min, max
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 1  # Save every N epochs
    save_best_only: bool = True
    
    # Experiment tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    experiment_name: str = "default_experiment"
    log_dir: str = "./logs"
    
    # Distributed training
    use_ddp: bool = False
    ddp_backend: str = "nccl"  # nccl, gloo
    world_size: int = 1
    rank: int = 0
    
    # Fine-tuning (LoRA/P-tuning)
    use_lora: bool = False
    lora_config: Dict[str, Any] = field(default_factory=dict)
    use_p_tuning: bool = False
    p_tuning_config: Dict[str, Any] = field(default_factory=dict)
    
    # Debugging
    detect_anomaly: bool = False
    log_interval: int = 100


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1_score"])
    compute_confusion_matrix: bool = True
    compute_roc_curve: bool = True
    compute_pr_curve: bool = True
    save_predictions: bool = False
    output_dir: str = "./evaluation_results"


class ConfigLoader:
    """
    Load and manage configuration from YAML files.
    
    Example:
        >>> loader = ConfigLoader("config/training.yaml")
        >>> config = loader.get_training_config()
        >>> model_config = loader.get_model_config()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default config path."""
        config_dir = Path(__file__).parent
        config_file = config_dir / "default_config.yaml"
        return str(config_file)
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(
                    f"Config file not found: {self.config_path}. "
                    "Using default configuration."
                )
                self.config = self._get_default_config()
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            
            logger.info(f"✅ Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}", exc_info=True)
            logger.warning("Using default configuration")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "name": "custom_model",
                "model_type": "transformer",
                "architecture": {},
                "initialization": "xavier_uniform",
                "device": "auto",
                "torch_dtype": "auto"
            },
            "data": {
                "batch_size": 32,
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
                "shuffle": True,
                "validation_split": 0.2,
                "test_split": 0.1,
                "seed": 42
            },
            "training": {
                "epochs": 10,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "optimizer": "adamw",
                "use_mixed_precision": True,
                "mixed_precision_dtype": "float16",
                "gradient_clip_norm": 1.0,
                "gradient_accumulation_steps": 1,
                "learning_rate_scheduler": "cosine",
                "early_stopping_patience": 5,
                "early_stopping_min_delta": 0.001,
                "save_checkpoints": True,
                "checkpoint_dir": "./checkpoints",
                "use_tensorboard": True,
                "use_wandb": False,
                "experiment_name": "default_experiment"
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "compute_confusion_matrix": True,
                "compute_roc_curve": True,
                "compute_pr_curve": True
            }
        }
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model_dict = self.config.get("model", {})
        return ModelConfig(**model_dict)
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        data_dict = self.config.get("data", {})
        return DataConfig(**data_dict)
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        training_dict = self.config.get("training", {})
        return TrainingConfig(**training_dict)
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        eval_dict = self.config.get("evaluation", {})
        return EvaluationConfig(**eval_dict)
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self.config
    
    def save_config(self, path: str) -> None:
        """Save current configuration to YAML file."""
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✅ Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")
            raise
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
            """Recursively update nested dictionaries."""
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        self.config = deep_update(self.config, updates)
        logger.info("✅ Configuration updated")



