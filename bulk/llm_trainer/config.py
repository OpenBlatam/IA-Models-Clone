"""
Training Configuration Module
==============================

Manages training arguments and configuration for LLM training.
Provides default values and automatic configuration based on device.

Author: BUL System
Date: 2024
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from transformers import TrainingArguments
from .device_manager import DeviceManager

logger = logging.getLogger(__name__)


class TrainingConfig:
    """
    Manages training configuration and arguments.
    
    Provides default values and automatic configuration based on:
    - Device capabilities (GPU/TPU/CPU)
    - Dataset size
    - Model requirements
    
    Attributes:
        training_args: TrainingArguments instance
        device_manager: DeviceManager instance
        
    Example:
        >>> config = TrainingConfig(
        ...     output_dir="./checkpoints",
        ...     learning_rate=3e-5,
        ...     num_train_epochs=3,
        ...     batch_size=8,
        ...     device_manager=device_manager
        ... )
        >>> args = config.get_training_args()
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        learning_rate: float = 3e-5,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        device_manager: Optional[DeviceManager] = None,
        gradient_accumulation_steps: int = 1,
        warmup_steps: Optional[int] = None,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: Optional[int] = None,
        evaluation_strategy: str = "no",
        load_best_model_at_end: bool = False,
        save_total_limit: int = 3,
        fp16: bool = False,
        bf16: bool = False,
        dataloader_num_workers: int = 4,
        num_train_samples: Optional[int] = None,
        optimizer: str = "adamw_torch",
        lr_scheduler_type: str = "linear",
        gradient_checkpointing: bool = False,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize TrainingConfig.
        
        Args:
            output_dir: Directory to save checkpoints and logs
            learning_rate: Learning rate for training (default: 3e-5)
            num_train_epochs: Number of training epochs (default: 3)
            batch_size: Training batch size (default: 8)
            device_manager: DeviceManager instance (optional)
            gradient_accumulation_steps: Gradient accumulation steps (default: 1)
            warmup_steps: Warmup steps (auto-calculated if None)
            weight_decay: Weight decay for regularization (default: 0.01)
            logging_steps: Steps between logging (default: 10)
            save_steps: Steps between checkpoints (default: 500)
            optimizer: Optimizer type (default: "adamw_torch")
            lr_scheduler_type: Learning rate scheduler type (default: "linear")
            gradient_checkpointing: Enable gradient checkpointing (default: False)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            seed: Random seed for reproducibility (default: 42)
            eval_steps: Steps between evaluations (default: None)
            evaluation_strategy: Evaluation strategy (default: "no")
            load_best_model_at_end: Load best model at end (default: False)
            save_total_limit: Maximum checkpoints to keep (default: 3)
            fp16: Use FP16 mixed precision (default: False)
            bf16: Use BF16 mixed precision (default: False)
            dataloader_num_workers: Data loader workers (default: 4)
            num_train_samples: Number of training samples (for warmup calculation)
            **kwargs: Additional arguments for TrainingArguments
        """
        self.output_dir = Path(output_dir)
        self.device_manager = device_manager or DeviceManager()
        self.num_train_samples = num_train_samples
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate warmup steps if not provided
        if warmup_steps is None and num_train_samples:
            total_steps = (num_train_samples // batch_size) * num_train_epochs
            warmup_steps = max(1, int(total_steps * 0.1))
        
        # Configure mixed precision based on device
        fp16, bf16 = self._configure_mixed_precision(fp16, bf16)
        
        # Adjust dataloader workers for CPU
        if not self.device_manager.is_cuda_available():
            dataloader_num_workers = min(dataloader_num_workers, 2)
        
        # Create TrainingArguments
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps or 0,
            weight_decay=weight_decay,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy=evaluation_strategy,
            load_best_model_at_end=load_best_model_at_end,
            save_total_limit=save_total_limit,
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=dataloader_num_workers,
            optimizer=optimizer,
            lr_scheduler_type=lr_scheduler_type,
            gradient_checkpointing=gradient_checkpointing,
            max_grad_norm=max_grad_norm,
            seed=seed,
            report_to="none",  # Disable wandb/tensorboard by default
            remove_unused_columns=False,
            **kwargs
        )
        
        logger.info(f"Training configuration created: {self.training_args}")
    
    def _configure_mixed_precision(
        self,
        fp16: bool,
        bf16: bool
    ) -> Tuple[bool, bool]:
        """
        Configure mixed precision based on device capabilities.
        
        Args:
            fp16: Requested FP16 setting
            bf16: Requested BF16 setting
            
        Returns:
            Tuple of (fp16, bf16) settings
        """
        if not self.device_manager.is_cuda_available():
            return False, False
        
        # Check BF16 support
        if bf16 and self.device_manager.supports_bf16:
            fp16 = False
            logger.info("Using BF16 mixed precision (device supports it)")
            return False, True
        elif fp16:
            logger.info("Using FP16 mixed precision")
            return True, False
        else:
            logger.info("Using FP32 precision")
            return False, False
    
    def get_training_args(self) -> TrainingArguments:
        """Get the TrainingArguments instance."""
        return self.training_args
    
    def get_output_dir(self) -> Path:
        """Get the output directory."""
        return self.output_dir
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.training_args, key):
                setattr(self.training_args, key, value)
                logger.info(f"Updated {key} to {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")

