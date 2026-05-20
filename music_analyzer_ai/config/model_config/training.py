"""
Training Configuration Module

Training configuration dataclasses.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    scheduler_type: str = "cosine"  # "cosine", "linear", "plateau", "onecycle"
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    use_mixed_precision: bool = True
    compile_model: bool = True
    enable_tf32: bool = True
    early_stopping_patience: int = 10
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10



