from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
import yaml
import json
import logging
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from torch.utils.tensorboard import SummaryWriter
import pickle
import hashlib
from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
Advanced Model Training for HeyGen AI.
Implements configuration management, experiment tracking, and model checkpointing.
"""


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model settings
    model_name: str = "gpt2"
    model_type: str = "transformer"
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Optimization settings
    use_fp16: bool = True
    use_mixed_precision: bool = True
    use_distributed: bool = False
    num_workers: int = 4
    
    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_length: int = 512
    
    # Checkpointing settings
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    checkpoint_dir: str = "./checkpoints"
    
    # Logging settings
    logging_steps: int = 10
    log_dir: str = "./logs"
    use_wandb: bool = True
    use_tensorboard: bool = True
    
    # Early stopping settings
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Cross-validation settings
    use_cross_validation: bool = False
    n_folds: int = 5
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConfigManager:
    """Manages configuration files and experiment settings."""
    
    def __init__(self, config_path: str = None):
        
    """__init__ function."""
self.config_path = config_path
        self.config = TrainingConfig()
        self.experiment_id = self._generate_experiment_id()
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{random_suffix}"
    
    def load_config(self, config_path: str = None) -> TrainingConfig:
        """Load configuration from YAML file."""
        config_file = config_path or self.config_path
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config_dict = yaml.safe_load(f)
                
                # Update config with loaded values
                for key, value in config_dict.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")
        
        return self.config
    
    def save_config(self, save_path: str = None) -> str:
        """Save current configuration to YAML file."""
        if save_path is None:
            save_path = f"configs/{self.experiment_id}_config.yaml"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            config_dict = asdict(self.config)
            with open(save_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            raise
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path."""
        return f"experiments/{self.experiment_id}"


class ExperimentTracker:
    """Manages experiment tracking and logging."""
    
    def __init__(self, config: TrainingConfig, experiment_id: str):
        
    """__init__ function."""
self.config = config
        self.experiment_id = experiment_id
        self.writer = None
        self.wandb_run = None
        self.setup_logging()
    
    def setup_logging(self) -> Any:
        """Setup logging and experiment tracking."""
        try:
            # Create experiment directory
            experiment_dir = f"experiments/{self.experiment_id}"
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Setup TensorBoard
            if self.config.use_tensorboard:
                log_dir = f"{experiment_dir}/tensorboard"
                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging to {log_dir}")
            
            # Setup Weights & Biases
            if self.config.use_wandb:
                wandb.init(
                    project="heygen-ai-training",
                    name=self.experiment_id,
                    config=asdict(self.config),
                    dir=experiment_dir
                )
                self.wandb_run = wandb.run
                logger.info("Weights & Biases logging initialized")
                
        except Exception as e:
            logger.error(f"Failed to setup logging: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to all configured logging systems."""
        try:
            # Add prefix to metric names
            prefixed_metrics = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
            
            # Log to TensorBoard
            if self.writer:
                for name, value in prefixed_metrics.items():
                    self.writer.add_scalar(name, value, step)
            
            # Log to Weights & Biases
            if self.wandb_run:
                wandb.log(prefixed_metrics, step=step)
                
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model_parameters(self, model: nn.Module):
        """Log model parameters and architecture."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
            }
            
            if self.wandb_run:
                wandb.config.update(model_info)
            
            logger.info(f"Model parameters: {model_info}")
            
        except Exception as e:
            logger.error(f"Failed to log model parameters: {e}")
    
    def close(self) -> Any:
        """Close all logging systems."""
        try:
            if self.writer:
                self.writer.close()
            if self.wandb_run:
                wandb.finish()
        except Exception as e:
            logger.error(f"Failed to close logging: {e}")


class ModelCheckpointer:
    """Manages model checkpointing and loading."""
    
    def __init__(self, config: TrainingConfig, experiment_id: str):
        
    """__init__ function."""
self.config = config
        self.experiment_id = experiment_id
        self.checkpoint_dir = f"experiments/{experiment_id}/checkpoints"
        self.best_metric = float('inf')
        self.checkpoint_history = []
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: Any, epoch: int, step: int, metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': metrics,
                'config': asdict(self.config),
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save checkpoint
            checkpoint_path = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = f"{self.checkpoint_dir}/best_model.pt"
                shutil.copy(checkpoint_path, best_path)
                logger.info(f"Saved best model to {best_path}")
            
            # Update checkpoint history
            self.checkpoint_history.append({
                'path': checkpoint_path,
                'step': step,
                'metrics': metrics,
                'is_best': is_best
            })
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: optim.Optimizer, scheduler: Any = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def load_best_model(self, model: nn.Module) -> Dict[str, Any]:
        """Load the best model checkpoint."""
        best_path = f"{self.checkpoint_dir}/best_model.pt"
        if os.path.exists(best_path):
            return self.load_checkpoint(best_path, model, None)
        else:
            logger.warning("No best model checkpoint found")
            return {}
    
    def _cleanup_old_checkpoints(self) -> Any:
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoint_history) > self.config.save_total_limit:
            # Sort by step and remove oldest
            self.checkpoint_history.sort(key=lambda x: x['step'])
            checkpoints_to_remove = self.checkpoint_history[:-self.config.save_total_limit]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    if os.path.exists(checkpoint['path']):
                        os.remove(checkpoint['path'])
                        logger.info(f"Removed old checkpoint: {checkpoint['path']}")
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {checkpoint['path']}: {e}")


class EarlyStopping:
    """Implements early stopping mechanism."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        
    """__init__ function."""
self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, current_score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == "min":
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CrossValidator:
    """Implements k-fold cross-validation."""
    
    def __init__(self, n_folds: int = 5):
        
    """__init__ function."""
self.n_folds = n_folds
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    def split_data(self, dataset_size: int) -> List[Tuple[List[int], List[int]]]:
        """Split data into k folds."""
        indices = list(range(dataset_size))
        return list(self.kfold.split(indices))
    
    def train_fold(self, train_indices: List[int], val_indices: List[int],
                   model: nn.Module, dataset: Any, config: TrainingConfig) -> Dict[str, float]:
        """Train model on a single fold."""
        # Create data loaders for this fold
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            dataset, batch_size=config.batch_size, sampler=train_sampler,
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            dataset, batch_size=config.batch_size, sampler=val_sampler,
            num_workers=config.num_workers
        )
        
        # Train model
        trainer = ModelTrainer(config)
        results = trainer.train(model, train_loader, val_loader)
        
        return results


class ModelTrainer:
    """Main trainer class with all training functionality."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        self.scaler = GradScaler() if config.use_fp16 else None
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
        
        # Setup distributed training
        if config.use_distributed and torch.cuda.device_count() > 1:
            self.setup_distributed_training()
    
    def setup_distributed_training(self) -> Any:
        """Setup distributed training."""
        try:
            dist.init_process_group(backend='nccl')
            logger.info(f"Using {torch.cuda.device_count()} GPUs for distributed training")
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader = None) -> Dict[str, float]:
        """Main training loop."""
        model = model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        # Setup experiment tracking
        experiment_id = f"train_{int(time.time())}"
        tracker = ExperimentTracker(self.config, experiment_id)
        checkpointer = ModelCheckpointer(self.config, experiment_id)
        
        # Log model parameters
        tracker.log_model_parameters(model)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler)
            
            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate_epoch(model, val_loader)
                val_loss = val_metrics.get('val_loss', float('inf'))
                
                # Check for best model
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                # Early stopping
                if self.early_stopping(val_loss):
                    logger.info("Early stopping triggered")
                    break
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            tracker.log_metrics(all_metrics, epoch, prefix="epoch")
            
            # Save checkpoint
            if epoch % self.config.save_steps == 0:
                checkpointer.save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch, all_metrics, is_best
                )
        
        tracker.close()
        return all_metrics
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader,
                   optimizer: optim.Optimizer, scheduler: Any) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(self.device) if self.config.use_fp16 else torch.no_grad():
                    loss = model(**batch)
                
                # Backward pass
                if self.config.use_fp16:
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.max_grad_norm
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Log progress
                if batch_idx % self.config.logging_steps == 0:
                    logger.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    loss = model(**batch)
                    total_loss += loss.item()
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    continue
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}


def create_default_config() -> str:
    """Create a default configuration file."""
    config = TrainingConfig()
    config_path = "configs/default_training_config.yaml"
    
    os.makedirs("configs", exist_ok=True)
    
    with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        yaml.dump(asdict(config), f, default_flow_style=False, indent=2)
    
    logger.info(f"Created default config at {config_path}")
    return config_path


def main():
    """Main function demonstrating usage."""
    try:
        # Create default config if it doesn't exist
        config_path = "configs/default_training_config.yaml"
        if not os.path.exists(config_path):
            create_default_config()
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        logger.info("Model training system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize training system: {e}")
        raise


match __name__:
    case "__main__":
    main() 