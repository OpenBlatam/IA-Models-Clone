from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import structlog
from tqdm import tqdm
import time
import os
from pathlib import Path
import json
from dataclasses import dataclass
import pickle
from .models import BaseMessageModel, ModelConfig
from .data_loader import MessageDataset, DataManager
        from .models import ModelFactory
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Training Module for Key Messages Feature - Modular Architecture
"""


logger = structlog.get_logger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model configuration
    model_type: str = "gpt2"
    model_name: str = "gpt2"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    weight_decay: float = 0.01
    scheduler_type: str = "cosine"  # cosine, linear, step
    gradient_accumulation_steps: int = 4
    
    # Mixed precision
    use_mixed_precision: bool = True
    fp16: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    use_tensorboard: bool = True
    experiment_name: str = "key_messages_training"
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Data
    max_length: int = 512
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    """Main training class with advanced features."""
    
    def __init__(self, config: TrainingConfig, model: BaseMessageModel, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 test_loader: Optional[DataLoader] = None):
        
    """__init__ function."""
# Guard clauses for early validation
        if not config:
            raise ValueError("TrainingConfig cannot be None")
        
        if not model:
            raise ValueError("Model cannot be None")
        
        if not train_loader:
            raise ValueError("Train loader cannot be None")
        
        if not val_loader:
            raise ValueError("Validation loader cannot be None")
        
        if config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if config.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if config.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Setup experiment tracking
        self.writer = None
        self.wandb_run = None
        self._setup_experiment_tracking()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Setup checkpointing
        self.checkpoint_dir = Path(f"checkpoints/{config.experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Trainer initialized", 
                   model_type=config.model_type,
                   device=str(self.device),
                   mixed_precision=config.use_mixed_precision,
                   gradient_accumulation=config.gradient_accumulation_steps)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with weight decay."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps,
                eta_min=0
            )
        elif self.config.scheduler_type == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_steps // 3,
                gamma=0.1
            )
        else:
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
    
    def _setup_experiment_tracking(self) -> Any:
        """Setup experiment tracking with TensorBoard and/or Weights & Biases."""
        if self.config.use_tensorboard:
            log_dir = f"logs/{self.config.experiment_name}"
            self.writer = SummaryWriter(log_dir)
            logger.info("TensorBoard logging enabled", log_dir=log_dir)
        
        if self.config.use_wandb:
            try:
                wandb.init(
                    project="key-messages",
                    name=self.config.experiment_name,
                    config=vars(self.config)
                )
                self.wandb_run = wandb.run
                logger.info("Weights & Biases logging enabled")
            except Exception as e:
                logger.warning("Failed to initialize Weights & Biases", error=str(e))
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with gradient accumulation and mixed precision."""
        logger.info("Starting training", epochs=self.config.num_epochs)
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Validation phase
            val_loss = self._validate_epoch(epoch)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_metrics(epoch, train_loss, val_loss, epoch_time)
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(f"best_model_epoch_{epoch}.pt", epoch)
            
            # Regular checkpointing
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch)
        
        # Final evaluation
        if self.test_loader:
            test_metrics = self._evaluate_model()
            logger.info("Training completed", test_metrics=test_metrics)
        
        return self._get_training_summary()
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            loss = self._forward_pass(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
                
                # Log training metrics
                if self.global_step % 100 == 0:
                    self._log_training_metrics(loss.item() * self.config.gradient_accumulation_steps)
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                batch = self._move_batch_to_device(batch)
                loss = self._forward_pass(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with mixed precision support."""
        if self.config.use_mixed_precision:
            with autocast():
                return self._compute_loss(batch)
        else:
            return self._compute_loss(batch)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for the batch."""
        # This is a simplified loss computation
        # In practice, you'd implement the specific loss for your task
        
        if 'input_ids' in batch and 'labels' in batch:
            # For language modeling tasks
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask', None)
            )
            
            # Shift sequences for next token prediction
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        else:
            # For other tasks, implement appropriate loss
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return loss
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_training_metrics(self, loss: float):
        """Log training metrics to TensorBoard and/or W&B."""
        metrics = {
            'train/loss': loss,
            'train/learning_rate': self.scheduler.get_last_lr()[0],
            'train/global_step': self.global_step
        }
        
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
        
        if self.wandb_run:
            self.wandb_run.log(metrics, step=self.global_step)
    
    def _log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float, epoch_time: float):
        """Log epoch-level metrics."""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'best_val_loss': self.best_val_loss
        }
        
        self.training_history.append(metrics)
        
        if self.writer:
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/best_val_loss', self.best_val_loss, epoch)
        
        if self.wandb_run:
            self.wandb_run.log(metrics, step=epoch)
        
        logger.info("Epoch completed", **metrics)
    
    def _save_checkpoint(self, filename: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
            'training_history': self.training_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        logger.info("Checkpoint saved", path=str(checkpoint_path), epoch=epoch)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info("Checkpoint loaded", path=checkpoint_path)
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = self._move_batch_to_device(batch)
                loss = self._forward_pass(batch)
                total_loss += loss.item()
                num_batches += 1
        
        test_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {
            'test_loss': test_loss,
            'best_val_loss': self.best_val_loss
        }
        
        # Log test metrics
        if self.writer:
            self.writer.add_scalar('test/loss', test_loss, 0)
        
        if self.wandb_run:
            self.wandb_run.log(metrics)
        
        return metrics
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': len(self.training_history),
            'total_steps': self.global_step,
            'training_history': self.training_history,
            'config': vars(self.config)
        }
    
    def cleanup(self) -> Any:
        """Cleanup resources."""
        if self.writer:
            self.writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()

class TrainingManager:
    """High-level training management class."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.data_manager = DataManager({'max_length': config.max_length})
    
    def prepare_training(self, data_path: str, tokenizer=None) -> Tuple[Trainer, Dict[str, Any]]:
        """Prepare everything for training."""
        # Load and preprocess data
        data = self.data_manager.load_data(data_path)
        
        # Create dataset
        dataset = self.data_manager.create_dataset(data, tokenizer)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.data_manager.get_data_loaders(
            dataset,
            batch_size=self.config.batch_size,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio
        )
        
        # Create model
        model_config = ModelConfig(
            model_name=self.config.model_name,
            max_length=self.config.max_length,
            device=self.config.device
        )
        
        model = ModelFactory.create_model(self.config.model_type, model_config)
        
        # Create trainer
        trainer = Trainer(self.config, model, train_loader, val_loader, test_loader)
        
        return trainer, {
            'dataset_size': len(dataset),
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'test_batches': len(test_loader) if test_loader else 0
        }
    
    def train_model(self, data_path: str, tokenizer=None) -> Dict[str, Any]:
        """Complete training pipeline."""
        logger.info("Starting training pipeline", data_path=data_path)
        
        # Prepare training
        trainer, data_info = self.prepare_training(data_path, tokenizer)
        logger.info("Training prepared", **data_info)
        
        try:
            # Train model
            training_summary = trainer.train()
            
            # Save final model
            trainer._save_checkpoint("final_model.pt", training_summary['final_epoch'])
            
            return {
                'training_summary': training_summary,
                'data_info': data_info,
                'model_path': str(trainer.checkpoint_dir / "final_model.pt")
            }
            
        finally:
            trainer.cleanup()

# Default training configuration
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    model_type="gpt2",
    model_name="gpt2",
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=5,
    gradient_accumulation_steps=4,
    use_mixed_precision=True,
    use_tensorboard=True,
    use_wandb=False,
    experiment_name="key_messages_baseline"
) 