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
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import os
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import bitsandbytes as bnb
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments
import wandb
        from accelerate import Accelerator
        import wandb
        import bitsandbytes as bnb
        import torch.optim as optim
                import wandb
from typing import Any, List, Dict, Optional
import asyncio
"""
Deep Learning Training Pipeline for HeyGen AI.

Advanced training pipeline with mixed precision, distributed training,
and optimization techniques following PEP 8 style guidelines.
"""



logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Model configuration
    model_architecture: str
    model_config: Dict[str, Any]
    
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
    
    # Checkpointing
    save_checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 100
    eval_every: int = 1000
    
    # Device
    device: str = "cuda"
    
    # Data
    train_data_path: str = ""
    val_data_path: str = ""
    num_workers: int = 4


class TrainingMetrics:
    """Training metrics tracking."""

    def __init__(self) -> Any:
        """Initialize training metrics."""
        self.training_losses = []
        self.validation_losses = []
        self.learning_rates = []
        self.training_times = []
        self.gradient_norms = []
        self.memory_usage = []

    def update(
        self,
        training_loss: float,
        validation_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        training_time: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        memory_usage: Optional[float] = None
    ):
        """Update training metrics.

        Args:
            training_loss: Current training loss.
            validation_loss: Current validation loss.
            learning_rate: Current learning rate.
            training_time: Training time for current step.
            gradient_norm: Current gradient norm.
            memory_usage: Current memory usage.
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
            "memory_usage": self.memory_usage
        }
        
        with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metrics_dict, f, indent=2)


class LibraryOptimizer:
    """Gestor de optimización de librerías para deep learning."""
    def __init__(self, config) -> Any:
        self.config = config
        self.accelerator = None
        self.wandb_run = None

    def setup_accelerate(self) -> Any:
        self.accelerator = Accelerator()
        return self.accelerator

    def setup_wandb(self) -> Any:
        self.wandb_run = wandb.init(project="heygen-ai-training", config=self.config.__dict__)
        return self.wandb_run

    def setup_optimizer(self, model_parameters) -> Any:
        opt_type = self.config.optimizer_type.lower()
        if opt_type == "adamw":
            try:
                return bnb.optim.AdamW8bit(
                    model_parameters,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            except Exception:
                return optim.AdamW(
                    model_parameters,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
        elif opt_type == "adam":
            try:
                return bnb.optim.Adam8bit(
                    model_parameters,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            except Exception:
                return optim.Adam(
                    model_parameters,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
        elif opt_type == "sgd":
            return optim.SGD(
                model_parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")

    def finish_wandb(self) -> Any:
        if self.wandb_run:
            self.wandb_run.finish()


class AdvancedTrainingPipeline:
    """Advanced training pipeline with mixed precision and distributed training."""

    def __init__(self, config: TrainingConfig):
        """Initialize training pipeline.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_mixed_precision else None
        self.metrics = TrainingMetrics()
        self.dynamic_loss_scaler = GradScaler(init_scale=2.**16) if config.use_mixed_precision else None
        self.profiler = None
        self.batch_size = config.batch_size
        self._auto_batch_scaling_done = False
        self.library_optimizer = LibraryOptimizer(config)
        self.accelerator = self.library_optimizer.setup_accelerate()
        self.wandb_run = self.library_optimizer.setup_wandb()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_validation_loss = float('inf')
        
        # Setup distributed training
        if config.use_distributed_training:
            self._setup_distributed_training()

    def _setup_distributed_training(self) -> Any:
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.config.rank = dist.get_rank()
        self.config.world_size = dist.get_world_size()
        
        # Set device based on rank
        torch.cuda.set_device(self.config.rank)
        self.device = torch.device(f'cuda:{self.config.rank}')

    def setup_model(self, model: nn.Module):
        """Setup model for training.

        Args:
            model: Neural network model.
        """
        self.model = model.to(self.device)
        
        if self.config.use_distributed_training:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.config.rank],
                output_device=self.config.rank
            )

    def setup_optimizer(self, model_parameters) -> Any:
        """Setup optimizer with bitsandbytes if available."""
        self.optimizer = self.library_optimizer.setup_optimizer(model_parameters)

    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler.

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
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")

    def _auto_scale_batch_size(self, train_dataloader, loss_function) -> Any:
        """Automatically scale batch size to fit GPU memory."""
        if self._auto_batch_scaling_done:
            return
        max_batch_size = self.batch_size
        min_batch_size = 1
        best_batch_size = min_batch_size
        for bs in [max_batch_size, max_batch_size//2, max_batch_size//4, min_batch_size]:
            try:
                loader = DataLoader(
                    train_dataloader.dataset,
                    batch_size=bs,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    prefetch_factor=4,
                    persistent_workers=True
                )
                batch = next(iter(loader))
                with autocast(enabled=self.config.use_mixed_precision):
                    loss = loss_function(self.model, *batch)
                best_batch_size = bs
                break
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        self.batch_size = best_batch_size
        self._auto_batch_scaling_done = True

    def train_step(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        loss_function: Callable
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch_data: Training batch data.
            loss_function: Loss function to compute.

        Returns:
            Dict[str, float]: Training step metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        batch_data = tuple(data.to(self.device, non_blocking=True) for data in batch_data)

        # Forward pass with mixed precision
        if self.config.use_mixed_precision:
            with autocast(dtype=self.config.mixed_precision_dtype):
                loss = loss_function(self.model, *batch_data)
            
            # Backward pass with gradient scaling
            if self.dynamic_loss_scaler:
                self.dynamic_loss_scaler.scale(loss).backward()
                self.dynamic_loss_scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.dynamic_loss_scaler.step(self.optimizer)
                self.dynamic_loss_scaler.update()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            # Standard forward pass
            loss = loss_function(self.model, *batch_data)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()

        # Update scheduler
        if self.scheduler is not None:
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
        """Single validation step.

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
        """Train for one epoch with profiler and async prefetch."""
        epoch_start_time = time.time()
        epoch_losses = []
        self._auto_scale_batch_size(train_dataloader, loss_function)
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=self.config.rank != 0
        )
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            for batch_idx, batch_data in enumerate(progress_bar):
                step_start_time = time.time()
                with record_function("train_step"):
                    step_metrics = self.train_step(batch_data, loss_function)
                self.metrics.update(
                    training_loss=step_metrics["loss"],
                    learning_rate=step_metrics["learning_rate"],
                    training_time=time.time() - step_start_time,
                    gradient_norm=step_metrics["gradient_norm"],
                    memory_usage=step_metrics["memory_usage"]
                )
                epoch_losses.append(step_metrics["loss"])
                self.current_step += 1
                if self.current_step % self.config.log_every == 0:
                    avg_loss = self.metrics.get_average_loss()
                    logger.info(
                        f"Step {self.current_step}: Loss = {avg_loss:.4f}, "
                        f"LR = {step_metrics['learning_rate']:.6f}, "
                        f"Memory = {step_metrics['memory_usage']:.2f}GB"
                    )
                if self.current_step % self.config.save_checkpoint_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.current_step}.pt")
                progress_bar.set_postfix({
                    'loss': f"{step_metrics['loss']:.4f}",
                    'lr': f"{step_metrics['learning_rate']:.6f}"
                })
            prof.export_chrome_trace(f"profiler_epoch_{epoch}.json")
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.2f}s. "
            f"Average loss: {avg_epoch_loss:.4f}"
        )

    def validate(
        self,
        val_dataloader: DataLoader,
        loss_function: Callable
    ) -> float:
        """Run validation.

        Args:
            val_dataloader: Validation data loader.
            loss_function: Loss function.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        validation_losses = []
        
        with torch.no_grad():
            for batch_data in tqdm(val_dataloader, desc="Validation"):
                loss = self.validation_step(batch_data, loss_function)
                validation_losses.append(loss)
        
        avg_validation_loss = sum(validation_losses) / len(validation_losses)
        
        # Update best validation loss
        if avg_validation_loss < self.best_validation_loss:
            self.best_validation_loss = avg_validation_loss
            self.save_checkpoint("best_model.pt")
        
        logger.info(f"Validation loss: {avg_validation_loss:.4f}")
        return avg_validation_loss

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

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
        """Load training checkpoint.

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
        
    """train function."""
logger.info("Starting training...")
        total_steps = len(train_dataloader) * self.config.num_epochs
        self.setup_scheduler(total_steps)
        self.model, self.optimizer, train_dataloader, val_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, val_dataloader, self.scheduler
        )
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch(train_dataloader, loss_function, epoch)
            if val_dataloader and epoch % 5 == 0:
                validation_loss = self.validate(val_dataloader, loss_function)
                self.metrics.update(validation_loss=validation_loss)
                wandb.log({"validation_loss": validation_loss, "epoch": epoch})
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        logger.info("Training completed!")
        self.library_optimizer.finish_wandb()

    def train_with_hf_trainer(self, train_dataset, eval_dataset, model, compute_metrics=None) -> Any:
        """Entrenamiento alternativo usando HuggingFace Trainer y mixed precision."""
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=self.config.use_mixed_precision,
            report_to=["wandb"],
            logging_dir="./logs",
            logging_steps=self.config.log_every,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )
        trainer.train()


def create_training_pipeline(config: TrainingConfig) -> AdvancedTrainingPipeline:
    """Factory function to create training pipeline.

    Args:
        config: Training configuration.

    Returns:
        AdvancedTrainingPipeline: Created training pipeline.
    """
    return AdvancedTrainingPipeline(config) 