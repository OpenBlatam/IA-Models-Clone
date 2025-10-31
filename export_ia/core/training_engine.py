"""
Advanced Training Engine for Export IA
Refactored training pipeline with PyTorch best practices and optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import os
import json
from pathlib import Path
import wandb
from tqdm import tqdm
import math

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Model parameters
    model_name: str
    model_type: str
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_every_n_epochs: int = 5
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Logging and monitoring
    log_every_n_steps: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "export-ia"
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

class MetricsTracker:
    """Advanced metrics tracking and computation"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
        
    def update(self, metrics: Dict[str, float], step: int = None):
        """Update metrics"""
        if step is not None:
            metrics['step'] = step
        self.history.append(metrics.copy())
        
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def get_average(self, metric_name: str, last_n: int = None) -> float:
        """Get average of metric over last n steps"""
        if metric_name not in self.metrics:
            return 0.0
            
        values = self.metrics[metric_name]
        if last_n is not None:
            values = values[-last_n:]
            
        return np.mean(values) if values else 0.0
        
    def get_best(self, metric_name: str, mode: str = "min") -> Tuple[float, int]:
        """Get best value and step for metric"""
        if metric_name not in self.metrics:
            return 0.0, 0
            
        values = self.metrics[metric_name]
        if not values:
            return 0.0, 0
            
        if mode == "min":
            best_value = min(values)
            best_step = values.index(best_value)
        else:
            best_value = max(values)
            best_step = values.index(best_value)
            
        return best_value, best_step
        
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.history = []

class EarlyStopping:
    """Early stopping mechanism"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop early"""
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
        
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best"""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

class LearningRateScheduler:
    """Advanced learning rate scheduling"""
    
    def __init__(self, optimizer: optim.Optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.scheduler = self._create_scheduler()
        
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler == "cosine_with_warmup":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            return None
            
    def step(self, metric: float = None):
        """Step the scheduler"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
                
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

class TrainingEngine:
    """Advanced training engine with best practices"""
    
    def __init__(self, config: TrainingConfig, model: nn.Module, 
                 train_dataset: Dataset, val_dataset: Dataset = None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Setup device and distributed training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_distributed()
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = LearningRateScheduler(self.optimizer, config)
        self.scaler = GradScaler() if config.mixed_precision else None
        self.early_stopping = EarlyStopping(
            config.early_stopping_patience, 
            config.early_stopping_min_delta
        )
        self.metrics_tracker = MetricsTracker()
        
        # Setup logging
        self._setup_logging()
        
        # Create data loaders
        self.train_loader = self._create_data_loader(train_dataset, shuffle=True)
        self.val_loader = self._create_data_loader(val_dataset, shuffle=False) if val_dataset else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _setup_distributed(self):
        """Setup distributed training"""
        if self.config.distributed:
            dist.init_process_group(backend='nccl')
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
            torch.cuda.set_device(self.config.local_rank)
            
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _create_data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create data loader"""
        if dataset is None:
            return None
            
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
    def _setup_logging(self):
        """Setup logging and monitoring"""
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # TensorBoard
        if self.config.use_tensorboard and self.config.local_rank == 0:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.config.checkpoint_dir, "tensorboard")
            )
        else:
            self.writer = None
            
        # Weights & Biases
        if self.config.use_wandb and self.config.local_rank == 0:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.model_name,
                config=self.config.__dict__
            )
            
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {}
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch}",
            disable=self.config.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            loss, metrics = self._train_step(batch)
            
            # Update metrics
            self.metrics_tracker.update(metrics, self.global_step)
            
            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics(metrics, "train")
                
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss:.4f}",
                'lr': f"{self.scheduler.get_lr():.2e}"
            })
            
            self.global_step += 1
            
        # Calculate epoch averages
        for key in self.metrics_tracker.metrics:
            if key != 'step':
                epoch_metrics[f"epoch_{key}"] = self.metrics_tracker.get_average(key)
                
        return epoch_metrics
        
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.config.mixed_precision:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        else:
            outputs = self.model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
        # Calculate metrics
        metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'learning_rate': self.scheduler.get_lr()
        }
        
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key != 'loss' and isinstance(value, torch.Tensor):
                    metrics[key] = value.item()
                    
        return loss.item() * self.config.gradient_accumulation_steps, metrics
        
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", 
                            disable=self.config.local_rank != 0):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.config.mixed_precision:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                    
                # Accumulate metrics
                val_metrics['val_loss'] = val_metrics.get('val_loss', 0) + loss.item()
                
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if key != 'loss' and isinstance(value, torch.Tensor):
                            val_key = f"val_{key}"
                            val_metrics[val_key] = val_metrics.get(val_key, 0) + value.item()
                            
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return val_metrics
        
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics.get('val_loss', 0))
            
            # Logging
            epoch_time = time.time() - start_time
            self._log_epoch_metrics(train_metrics, val_metrics, epoch_time)
            
            # Update training history
            training_history['train_loss'].append(train_metrics.get('epoch_loss', 0))
            training_history['val_loss'].append(val_metrics.get('val_loss', 0))
            training_history['learning_rates'].append(self.scheduler.get_lr())
            training_history['epochs'].append(epoch)
            
            # Checkpointing
            if self._should_save_checkpoint(val_metrics):
                self._save_checkpoint(val_metrics)
                
            # Early stopping
            if self.early_stopping(val_metrics.get('val_loss', float('inf'))):
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        # Final checkpoint
        self._save_checkpoint(val_metrics, is_final=True)
        
        # Close logging
        if self.writer:
            self.writer.close()
        if self.config.use_wandb and self.config.local_rank == 0:
            wandb.finish()
            
        return training_history
        
    def _should_save_checkpoint(self, val_metrics: Dict[str, float]) -> bool:
        """Check if checkpoint should be saved"""
        val_loss = val_metrics.get('val_loss', float('inf'))
        
        # Save best model
        if self.config.save_best_model and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
            
        # Save every n epochs
        if self.current_epoch % self.config.save_every_n_epochs == 0:
            return True
            
        return False
        
    def _save_checkpoint(self, val_metrics: Dict[str, float], is_final: bool = False):
        """Save model checkpoint"""
        if self.config.local_rank != 0:
            return
            
        checkpoint_name = f"checkpoint_epoch_{self.current_epoch}.pt"
        if is_final:
            checkpoint_name = "final_checkpoint.pt"
        elif val_metrics.get('val_loss', float('inf')) < self.best_val_loss:
            checkpoint_name = "best_model.pt"
            
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        
        # Get model state dict
        model_state_dict = self.model.state_dict()
        if hasattr(self.model, 'module'):  # DDP wrapper
            model_state_dict = self.model.module.state_dict()
            
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.scheduler.state_dict() if self.scheduler.scheduler else None,
            'val_metrics': val_metrics,
            'config': self.config.__dict__,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def _log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to TensorBoard and WandB"""
        if self.config.local_rank != 0:
            return
            
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, self.global_step)
                
        # WandB
        if self.config.use_wandb:
            wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()}, 
                     step=self.global_step)
                     
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float], epoch_time: float):
        """Log epoch-level metrics"""
        if self.config.local_rank != 0:
            return
            
        logger.info(f"Epoch {self.current_epoch}:")
        logger.info(f"  Train Loss: {train_metrics.get('epoch_loss', 0):.4f}")
        logger.info(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
        logger.info(f"  Learning Rate: {self.scheduler.get_lr():.2e}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("epoch/train_loss", train_metrics.get('epoch_loss', 0), self.current_epoch)
            self.writer.add_scalar("epoch/val_loss", val_metrics.get('val_loss', 0), self.current_epoch)
            self.writer.add_scalar("epoch/learning_rate", self.scheduler.get_lr(), self.current_epoch)
            self.writer.add_scalar("epoch/time", epoch_time, self.current_epoch)
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):  # DDP wrapper
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if checkpoint.get('scheduler_state_dict') and self.scheduler.scheduler:
            self.scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_metrics', {}).get('val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy dataset for testing
    class DummyDataset(Dataset):
        def __init__(self, size: int = 1000):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128),
                'labels': torch.randint(0, 10, (1,))
            }
    
    # Create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 512)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(512, 8, batch_first=True), 6
            )
            self.classifier = nn.Linear(512, 10)
            self.loss_fn = nn.CrossEntropyLoss()
            
        def forward(self, input_ids, attention_mask, labels=None):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global average pooling
            logits = self.classifier(x)
            
            if labels is not None:
                loss = self.loss_fn(logits, labels.squeeze())
                return {'loss': loss, 'logits': logits}
            return {'logits': logits}
    
    # Test training engine
    print("Testing Training Engine...")
    
    # Create datasets
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)
    
    # Create model
    model = DummyModel()
    
    # Create training config
    config = TrainingConfig(
        model_name="test_model",
        model_type="transformer",
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=2,
        checkpoint_dir="./test_checkpoints",
        use_tensorboard=False,
        use_wandb=False
    )
    
    # Create training engine
    engine = TrainingEngine(config, model, train_dataset, val_dataset)
    
    # Test training
    print("Starting training...")
    history = engine.train()
    
    print(f"Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    print("\nTraining engine refactored successfully!")
























