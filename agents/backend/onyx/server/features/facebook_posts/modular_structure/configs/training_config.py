"""
🚀 Training Configuration

YAML-based configuration for training hyperparameters and settings.
Implements the key convention: "Use configuration files (e.g., YAML) for hyperparameters and model settings."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration class.
    
    This class defines all training-related hyperparameters and settings
    that can be configured via YAML files.
    """
    
    # Basic Training Parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimizer Settings
    optimizer: str = "adam"  # adam, sgd, adamw, rmsprop
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Learning Rate Scheduling
    scheduler: Optional[str] = "cosine"  # cosine, step, exponential, plateau, cyclic, none
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_epochs: int = 0
    warmup_lr: float = 0.0001
    
    # Loss Function
    loss_function: str = "cross_entropy"  # cross_entropy, mse, mae, focal, etc.
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # Gradient Settings
    gradient_clip_norm: Optional[float] = 1.0
    gradient_clip_value: Optional[float] = None
    gradient_accumulation_steps: int = 1
    
    # Numerical Stability
    eps: float = 1e-8
    detect_anomaly: bool = False
    nan_to_num: bool = False
    
    # Training Monitoring
    log_interval: int = 100  # Log every N steps
    eval_interval: int = 1   # Evaluate every N epochs
    save_interval: int = 10  # Save checkpoint every N epochs
    
    # Early Stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor_metric: str = "val_loss"  # val_loss, val_accuracy, etc.
    mode: str = "min"  # min or max
    
    # Checkpointing
    save_best_only: bool = True
    save_last_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Data Loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Distributed Training
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    dist_backend: str = "nccl"
    dist_url: str = "env://"
    
    # Mixed Precision
    mixed_precision: bool = False
    amp_backend: str = "native"  # native, apex
    loss_scale: Union[str, float] = "dynamic"  # dynamic or float value
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    
    # Validation
    validation_split: float = 0.2
    test_split: float = 0.1
    stratify: bool = True
    
    # Data Augmentation
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment Tracking
    experiment_name: str = "training_experiment"
    project_name: str = "ml_project"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Advanced Training Techniques
    use_ema: bool = False  # Exponential Moving Average
    ema_decay: float = 0.999
    use_swa: bool = False  # Stochastic Weight Averaging
    swa_lr: float = 0.01
    swa_start_epoch: int = 10
    
    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        self._validate_config()
        self._setup_optimizer_params()
        self._setup_scheduler_params()
        self._setup_augmentation_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate epochs and batch size
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        # Validate weight decay
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        # Validate dropout rate
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {self.dropout_rate}")
        
        # Validate patience
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        
        # Validate splits
        if not 0.0 <= self.validation_split <= 1.0:
            raise ValueError(f"validation_split must be between 0.0 and 1.0, got {self.validation_split}")
        
        if not 0.0 <= self.test_split <= 1.0:
            raise ValueError(f"test_split must be between 0.0 and 1.0, got {self.test_split}")
        
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be less than 1.0")
        
        logger.debug("Training configuration validation passed")
    
    def _setup_optimizer_params(self) -> None:
        """Setup default optimizer parameters."""
        default_params = {
            "adam": {"betas": (0.9, 0.999), "eps": self.eps},
            "adamw": {"betas": (0.9, 0.999), "eps": self.eps},
            "sgd": {"momentum": 0.9, "nesterov": True},
            "rmsprop": {"alpha": 0.99, "eps": self.eps}
        }
        
        if self.optimizer in default_params:
            optimizer_defaults = default_params[self.optimizer]
            self.optimizer_params = {**optimizer_defaults, **self.optimizer_params}
    
    def _setup_scheduler_params(self) -> None:
        """Setup default scheduler parameters."""
        default_params = {
            "cosine": {"T_max": self.num_epochs, "eta_min": 1e-6},
            "step": {"step_size": 30, "gamma": 0.1},
            "exponential": {"gamma": 0.95},
            "plateau": {"mode": "min", "factor": 0.1, "patience": 10},
            "cyclic": {"base_lr": self.learning_rate * 0.1, "max_lr": self.learning_rate}
        }
        
        if self.scheduler and self.scheduler in default_params:
            scheduler_defaults = default_params[self.scheduler]
            self.scheduler_params = {**scheduler_defaults, **self.scheduler_params}
    
    def _setup_augmentation_config(self) -> None:
        """Setup default augmentation configuration."""
        if not self.augmentation_config:
            self.augmentation_config = {
                "horizontal_flip": 0.5,
                "vertical_flip": 0.0,
                "rotation": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "hue": 0.0,
                "noise": 0.0
            }
    
    def get_training_summary(self) -> str:
        """
        Generate a summary of the training configuration.
        
        Returns:
            String summary of the training configuration
        """
        summary = f"""
Training Configuration Summary:
{'=' * 40}
Epochs: {self.num_epochs}
Batch Size: {self.batch_size}
Learning Rate: {self.learning_rate}
Weight Decay: {self.weight_decay}
Optimizer: {self.optimizer}
Scheduler: {self.scheduler}
Loss Function: {self.loss_function}

Regularization:
  Dropout Rate: {self.dropout_rate}
  Label Smoothing: {self.label_smoothing}
  Gradient Clip Norm: {self.gradient_clip_norm}

Early Stopping:
  Enabled: {self.early_stopping}
  Patience: {self.patience}
  Monitor: {self.monitor_metric}

Mixed Precision: {self.mixed_precision}
Distributed: {self.distributed}
Seed: {self.seed}
        """.strip()
        
        return summary
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps * self.world_size
    
    def get_total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps."""
        steps_per_epoch = dataset_size // self.get_effective_batch_size()
        return steps_per_epoch * self.num_epochs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary."""
        return cls(**config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after updates
        self._validate_config()
        self._setup_optimizer_params()
        self._setup_scheduler_params()
        self._setup_augmentation_config()
    
    def clone(self) -> 'TrainingConfig':
        """Create a copy of the configuration."""
        from copy import deepcopy
        return deepcopy(self)


# Predefined training configurations for common scenarios
TRAINING_CONFIGS = {
    "quick_test": TrainingConfig(
        num_epochs=5,
        batch_size=16,
        learning_rate=0.01,
        log_interval=10,
        eval_interval=1,
        early_stopping=False
    ),
    
    "standard_training": TrainingConfig(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        scheduler="cosine",
        early_stopping=True,
        patience=15,
        mixed_precision=True
    ),
    
    "fine_tuning": TrainingConfig(
        num_epochs=50,
        batch_size=16,
        learning_rate=0.0001,
        weight_decay=0.01,
        optimizer="adamw",
        scheduler="cosine",
        warmup_epochs=5,
        gradient_clip_norm=1.0,
        early_stopping=True,
        patience=10
    ),
    
    "heavy_regularization": TrainingConfig(
        num_epochs=200,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.001,
        dropout_rate=0.3,
        label_smoothing=0.1,
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
        early_stopping=True,
        patience=20
    ),
    
    "distributed_training": TrainingConfig(
        num_epochs=100,
        batch_size=64,
        learning_rate=0.001,
        optimizer="adamw",
        scheduler="cosine",
        distributed=True,
        mixed_precision=True,
        gradient_accumulation_steps=2,
        num_workers=8
    )
}






