from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import os
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import wandb
                from lion_pytorch import Lion
            from transformers import get_cosine_schedule_with_warmup
from typing import Any, List, Dict, Optional
import asyncio
"""
PyTorch Training Pipeline for HeyGen AI.

Advanced PyTorch-based training pipeline with mixed precision, distributed
training, and optimization techniques following PEP 8 style guidelines.
"""



logger = logging.getLogger(__name__)


@dataclass
class PyTorchTrainingConfig:
    """Configuration for PyTorch training pipeline."""

    # Model configuration
    model_type: str
    model_parameters: Dict[str, Any]
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    scheduler_type: str = "cosine"
    
    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_dtype: torch.dtype = torch.float16
    
    # Distributed training
    use_distributed_training: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    
    # Checkpointing
    save_checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "heygen_ai"
    
    # Device
    device: str = "cuda"
    
    # Data
    train_data_path: str = ""
    val_data_path: str = ""
    num_workers: int = 4
    
    # Advanced features
    use_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 1
    use_ema: bool = False
    ema_decay: float = 0.999


class PyTorchTrainingMetrics:
    """PyTorch training metrics tracking."""

    def __init__(self) -> Any:
        """Initialize PyTorch training metrics."""
        self.training_losses = []
        self.validation_losses = []
        self.learning_rates = []
        self.training_times = []
        self.gradient_norms = []
        self.memory_usage = []
        self.epoch_losses = []
        self.step_losses = []

    def update(
        self,
        training_loss: float,
        validation_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        training_time: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        memory_usage: Optional[float] = None,
        epoch_loss: Optional[float] = None,
        step_loss: Optional[float] = None
    ):
        """Update PyTorch training metrics.

        Args:
            training_loss: Current training loss.
            validation_loss: Current validation loss.
            learning_rate: Current learning rate.
            training_time: Training time for current step.
            gradient_norm: Current gradient norm.
            memory_usage: Current memory usage.
            epoch_loss: Current epoch loss.
            step_loss: Current step loss.
        """
        self.training_losses.append(training_loss)
        
        if validation_loss is not None:
            self.validation_losses.append(validation_loss)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        if training_time is not None:
            self.training_times.append(training_time)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        if memory_usage is not None:
            self.memory_usage.append(memory_usage)
        if epoch_loss is not None:
            self.epoch_losses.append(epoch_loss)
        if step_loss is not None:
            self.step_losses.append(step_loss)

    def get_average_loss(self, window_size: int = 100) -> float:
        """Get average training loss over recent steps.

        Args:
            window_size: Number of recent steps to average.

        Returns:
            float: Average training loss.
        """
        if len(self.training_losses) == 0:
            return 0.0
        recent_losses = self.training_losses[-window_size:]
        return sum(recent_losses) / len(recent_losses)

    def save_metrics(self, file_path: str):
        """Save metrics to file.

        Args:
            file_path: Path to save metrics.
        """
        metrics_dict = {
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "learning_rates": self.learning_rates,
            "training_times": self.training_times,
            "gradient_norms": self.gradient_norms,
            "memory_usage": self.memory_usage,
            "epoch_losses": self.epoch_losses,
            "step_losses": self.step_losses
        }
        
        with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metrics_dict, f, indent=2)


class ExponentialMovingAverage:
    """Exponential moving average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize exponential moving average.

        Args:
            model: PyTorch model.
            decay: Decay rate.
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA parameters.

        Args:
            model: PyTorch model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to model.

        Args:
            model: PyTorch model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original parameters.

        Args:
            model: PyTorch model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PyTorchAdvancedTrainingPipeline:
    """Advanced PyTorch training pipeline with mixed precision and distributed training."""

    def __init__(self, config: PyTorchTrainingConfig):
        """Initialize PyTorch training pipeline.

        Args:
            config: PyTorch training configuration.
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_mixed_precision else None
        self.metrics = PyTorchTrainingMetrics()
        self.ema = ExponentialMovingAverage(self.model, config.ema_decay) if config.use_ema else None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_validation_loss = float('inf')
        
        # Logging
        self.writer = SummaryWriter() if config.use_tensorboard else None
        if config.use_wandb:
            wandb.init(project=config.wandb_project)
        
        # Setup distributed training
        if config.use_distributed_training:
            self._setup_distributed_training()

    def _setup_distributed_training(self) -> Any:
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend=self.config.backend)
        
        self.config.rank = dist.get_rank()
        self.config.world_size = dist.get_world_size()
        
        # Set device based on rank
        torch.cuda.set_device(self.config.rank)
        self.device = torch.device(f'cuda:{self.config.rank}')

    def setup_model(self, model: nn.Module):
        """Setup PyTorch model for training.

        Args:
            model: PyTorch neural network model.
        """
        self.model = model.to(self.device)
        
        if self.config.use_distributed_training:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.config.rank],
                output_device=self.config.rank
            )
        
        # Initialize EMA if enabled
        if self.config.use_ema:
            self.ema = ExponentialMovingAverage(self.model, self.config.ema_decay)

    def setup_optimizer(self, model_parameters) -> Any:
        """Setup PyTorch optimizer.

        Args:
            model_parameters: Model parameters to optimize.
        """
        if self.config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                model_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                model_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                model_parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "lion":
            # Lion optimizer (if available)
            try:
                self.optimizer = Lion(
                    model_parameters,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            except ImportError:
                logger.warning("Lion optimizer not available, falling back to AdamW")
                self.optimizer = optim.AdamW(
                    model_parameters,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")

    def setup_scheduler(self, total_steps: int):
        """Setup PyTorch learning rate scheduler.

        Args:
            total_steps: Total number of training steps.
        """
        if self.config.scheduler_type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        elif self.config.scheduler_type.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        elif self.config.scheduler_type.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_steps // 10,
                gamma=0.9
            )
        elif self.config.scheduler_type.lower() == "cosine_with_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")

    def train_step(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        loss_function: Callable
    ) -> Dict[str, float]:
        """Single PyTorch training step.

        Args:
            batch_data: Training batch data.
            loss_function: Loss function to compute.

        Returns:
            Dict[str, float]: Training step metrics.
        """
        self.model.train()
        
        # Move data to device
        batch_data = tuple(data.to(self.device) for data in batch_data)
        
        # Forward pass with mixed precision
        if self.config.use_mixed_precision:
            with autocast(dtype=self.config.mixed_precision_dtype):
                loss = loss_function(self.model, *batch_data)
            
            # Scale loss for gradient accumulation
            if self.config.use_gradient_accumulation:
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if self.config.use_gradient_accumulation:
                if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    if self.config.use_ema:
                        self.ema.update(self.model)
                else:
                    gradient_norm = torch.tensor(0.0)
            else:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.config.use_ema:
                    self.ema.update(self.model)
        else:
            # Standard forward pass
            loss = loss_function(self.model, *batch_data)
            
            # Scale loss for gradient accumulation
            if self.config.use_gradient_accumulation:
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if self.config.use_gradient_accumulation:
                if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    if self.config.use_ema:
                        self.ema.update(self.model)
                else:
                    gradient_norm = torch.tensor(0.0)
            else:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.config.use_ema:
                    self.ema.update(self.model)

        # Update scheduler
        if self.scheduler is not None and not self.config.use_gradient_accumulation:
            self.scheduler.step()

        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']

        # Get memory usage
        memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        return {
            "loss": loss.item(),
            "gradient_norm": gradient_norm.item(),
            "learning_rate": current_lr,
            "memory_usage": memory_usage
        }

    def validation_step(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        loss_function: Callable
    ) -> float:
        """Single PyTorch validation step.

        Args:
            batch_data: Validation batch data.
            loss_function: Loss function to compute.

        Returns:
            float: Validation loss.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            batch_data = tuple(data.to(self.device) for data in batch_data)
            
            # Forward pass
            if self.config.use_mixed_precision:
                with autocast(dtype=self.config.mixed_precision_dtype):
                    loss = loss_function(self.model, *batch_data)
            else:
                loss = loss_function(self.model, *batch_data)
            
            return loss.item()

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        loss_function: Callable,
        epoch: int
    ):
        """Train for one epoch using PyTorch.

        Args:
            train_dataloader: Training data loader.
            loss_function: Loss function.
            epoch: Current epoch number.
        """
        epoch_start_time = time.time()
        epoch_losses = []
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=self.config.rank != 0
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch_data, loss_function)
            
            # Update metrics
            self.metrics.update(
                training_loss=step_metrics["loss"],
                learning_rate=step_metrics["learning_rate"],
                training_time=time.time() - step_start_time,
                gradient_norm=step_metrics["gradient_norm"],
                memory_usage=step_metrics["memory_usage"],
                step_loss=step_metrics["loss"]
            )
            
            epoch_losses.append(step_metrics["loss"])
            self.current_step += 1
            
            # Logging
            if self.current_step % self.config.log_every == 0:
                avg_loss = self.metrics.get_average_loss()
                logger.info(
                    f"Step {self.current_step}: Loss = {avg_loss:.4f}, "
                    f"LR = {step_metrics['learning_rate']:.6f}, "
                    f"Memory = {step_metrics['memory_usage']:.2f}GB"
                )
                
                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('Loss/Train', avg_loss, self.current_step)
                    self.writer.add_scalar('Learning_Rate', step_metrics['learning_rate'], self.current_step)
                    self.writer.add_scalar('Memory_Usage', step_metrics['memory_usage'], self.current_step)
                
                # WandB logging
                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': avg_loss,
                        'learning_rate': step_metrics['learning_rate'],
                        'memory_usage': step_metrics['memory_usage'],
                        'step': self.current_step
                    })
            
            # Checkpointing
            if self.current_step % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_step_{self.current_step}.pt")
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{step_metrics['loss']:.4f}",
                'lr': f"{step_metrics['learning_rate']:.6f}"
            })
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Update epoch metrics
        self.metrics.update(epoch_loss=avg_epoch_loss)
        
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s. "
            f"Average loss: {avg_epoch_loss:.4f}"
        )

    def validate(
        self,
        val_dataloader: DataLoader,
        loss_function: Callable
    ) -> float:
        """Run PyTorch validation.

        Args:
            val_dataloader: Validation data loader.
            loss_function: Loss function.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        validation_losses = []
        
        # Use EMA for validation if enabled
        if self.config.use_ema:
            self.ema.apply_shadow(self.model)
        
        with torch.no_grad():
            for batch_data in tqdm(val_dataloader, desc="Validation"):
                loss = self.validation_step(batch_data, loss_function)
                validation_losses.append(loss)
        
        # Restore original parameters if using EMA
        if self.config.use_ema:
            self.ema.restore(self.model)
        
        avg_validation_loss = sum(validation_losses) / len(validation_losses)
        
        # Update best validation loss
        if avg_validation_loss < self.best_validation_loss:
            self.best_validation_loss = avg_validation_loss
            self.save_checkpoint("best_model.pt")
        
        logger.info(f"Validation loss: {avg_validation_loss:.4f}")
        return avg_validation_loss

    def save_checkpoint(self, filename: str):
        """Save PyTorch training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "ema_state_dict": self.ema.shadow if self.ema else None,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "best_validation_loss": self.best_validation_loss,
            "config": self.config.__dict__,
            "metrics": {
                "training_losses": self.metrics.training_losses,
                "validation_losses": self.metrics.validation_losses,
                "learning_rates": self.metrics.learning_rates
            }
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load PyTorch training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        # Load scheduler state
        if checkpoint_data["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        
        # Load scaler state
        if checkpoint_data["scaler_state_dict"] and self.scaler:
            self.scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
        
        # Load EMA state
        if checkpoint_data["ema_state_dict"] and self.ema:
            self.ema.shadow = checkpoint_data["ema_state_dict"]
        
        # Load training state
        self.current_epoch = checkpoint_data["current_epoch"]
        self.current_step = checkpoint_data["current_step"]
        self.best_validation_loss = checkpoint_data["best_validation_loss"]
        
        # Load metrics
        self.metrics.training_losses = checkpoint_data["metrics"]["training_losses"]
        self.metrics.validation_losses = checkpoint_data["metrics"]["validation_losses"]
        self.metrics.learning_rates = checkpoint_data["metrics"]["learning_rates"]
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        loss_function: Callable
    ):
        """Complete PyTorch training loop.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            loss_function: Loss function.
        """
        logger.info("Starting PyTorch training...")
        
        # Setup scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs
        if self.config.use_gradient_accumulation:
            total_steps = total_steps // self.config.gradient_accumulation_steps
        self.setup_scheduler(total_steps)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            self.train_epoch(train_dataloader, loss_function, epoch)
            
            # Validation
            if val_dataloader and epoch % 5 == 0:
                validation_loss = self.validate(val_dataloader, loss_function)
                self.metrics.update(validation_loss=validation_loss)
                
                # Log validation metrics
                if self.writer:
                    self.writer.add_scalar('Loss/Validation', validation_loss, epoch)
                if self.config.use_wandb:
                    wandb.log({'val_loss': validation_loss, 'epoch': epoch})
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        logger.info("PyTorch training completed!")


def create_pytorch_training_pipeline(config: PyTorchTrainingConfig) -> PyTorchAdvancedTrainingPipeline:
    """Factory function to create PyTorch training pipeline.

    Args:
        config: PyTorch training configuration.

    Returns:
        PyTorchAdvancedTrainingPipeline: Created PyTorch training pipeline.
    """
    return PyTorchAdvancedTrainingPipeline(config) 