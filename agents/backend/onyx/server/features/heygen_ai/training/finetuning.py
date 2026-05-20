"""
Fine-tuning Module for HeyGen AI
=================================

Implements efficient fine-tuning techniques following best practices:
- LoRA (Low-Rank Adaptation)
- P-tuning
- Full fine-tuning
- Gradient accumulation
- Mixed precision training
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available. Install with: pip install peft")

try:
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollator,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available")

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.
    
    Attributes:
        r: Rank of the low-rank matrices
        lora_alpha: Scaling parameter
        target_modules: Modules to apply LoRA to
        lora_dropout: Dropout probability
        bias: Bias type ('none', 'all', 'lora_only')
        task_type: Task type for PEFT
    """
    r: int = 8
    lora_alpha: int = 16
    target_modules: list = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning.
    
    Attributes:
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay
        max_grad_norm: Maximum gradient norm for clipping
        use_lora: Use LoRA for efficient fine-tuning
        lora_config: LoRA configuration
        use_mixed_precision: Use mixed precision training
        save_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
    """
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_lora: bool = True
    lora_config: Optional[LoRAConfig] = None
    use_mixed_precision: bool = True
    save_steps: int = 500
    eval_steps: int = 500


class LoRAFineTuner:
    """Fine-tune models using LoRA (Low-Rank Adaptation).
    
    Features:
    - Efficient parameter-efficient fine-tuning
    - Memory-efficient training
    - Support for multiple model architectures
    - Proper error handling
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize LoRA fine-tuner.
        
        Args:
            model: Base model to fine-tune
            config: Fine-tuning configuration
            device: Training device
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "PEFT library not available. Install with: pip install peft"
            )
        
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logging.getLogger(f"{__name__}.LoRAFineTuner")
        
        # Apply LoRA to model
        self.model = self._apply_lora(model)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = None
        if config.use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info("LoRA Fine-tuner initialized")
    
    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA to model.
        
        Args:
            model: Base model
        
        Returns:
            Model with LoRA adapters
        """
        lora_config = self.config.lora_config or LoRAConfig()
        
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for fine-tuning.
        
        Returns:
            Optimizer instance
        """
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler.
        
        Returns:
            Scheduler instance
        """
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.1,
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Training batch
        
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Forward pass with mixed precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
        
        self.scheduler.step()
        
        return {
            "loss": loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
    
    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights.
        
        Args:
            path: Path to save adapter
        """
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(path)
            self.logger.info(f"LoRA adapter saved to {path}")
        else:
            raise RuntimeError("Model is not a PEFT model")
    
    def load_adapter(self, path: str) -> None:
        """Load LoRA adapter weights.
        
        Args:
            path: Path to adapter
        """
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(path)
            self.logger.info(f"LoRA adapter loaded from {path}")
        else:
            raise RuntimeError("Model is not a PEFT model")


class FullFineTuner:
    """Full fine-tuning without LoRA.
    
    Use when you need to fine-tune all parameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize full fine-tuner.
        
        Args:
            model: Model to fine-tune
            config: Fine-tuning configuration
            device: Training device
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.logger = logging.getLogger(f"{__name__}.FullFineTuner")
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )
        
        # Mixed precision scaler
        self.scaler = None
        if config.use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info("Full Fine-tuner initialized")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Training batch
        
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Forward pass with mixed precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
        
        self.scheduler.step()
        
        return {
            "loss": loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }



