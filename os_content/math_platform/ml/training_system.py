from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import os
from pathlib import Path
import json
import pickle
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.autograd
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import optuna
from optuna import Trial, create_study
import mlflow
import mlflow.pytorch
import wandb
from tqdm import tqdm
import yaml
import warnings
    import cupy as cp
    import ray
    from ray import tune
import jax
import jax.numpy as jnp
from jax import grad, jit as jax_jit, vmap
import flax
from flax import linen as nn as flax_nn
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Training System
Comprehensive ML training pipeline with distributed training, advanced optimization, and production features.
"""


# Advanced ML Libraries


# GPU and Distributed Computing
try:
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Advanced Optimization

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types for training."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    TRANSLATION = "translation"
    REINFORCEMENT_LEARNING = "rl"
    MULTI_TASK = "multi_task"


class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Advanced configuration for training parameters."""
    # Model parameters
    model_name: str = "transformer"
    task_type: TaskType = TaskType.CLASSIFICATION
    vocab_size: int = 30000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    activation: str = "gelu"
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_sequence_length: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    
    # Optimization parameters
    use_amp: bool = True
    use_multi_gpu: bool = False
    use_distributed: bool = False
    use_ray: bool = False
    use_jax: bool = False
    
    # Advanced optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    lr_scheduler_warmup_steps: int = 1000
    lr_scheduler_decay_steps: int = 10000
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    early_stopping_restore_best_weights: bool = True
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 1000
    max_checkpoints: int = 5
    save_best_only: bool = True
    
    # Logging and monitoring
    log_every: int = 100
    use_wandb: bool = True
    use_mlflow: bool = True
    project_name: str = "math_platform_training"
    experiment_name: str = "default"
    
    # Cross-validation
    use_cross_validation: bool = False
    n_folds: int = 5
    cv_strategy: str = "stratified"
    
    # Hyperparameter optimization
    use_hyperopt: bool = False
    hyperopt_trials: int = 100
    hyperopt_timeout: float = 3600.0
    
    # Advanced features
    use_curriculum_learning: bool = False
    use_meta_learning: bool = False
    use_federated_learning: bool = False
    use_continual_learning: bool = False
    
    # Production features
    model_versioning: bool = True
    model_registry: bool = True
    a_b_testing: bool = False
    model_monitoring: bool = True
    
    def __post_init__(self) -> Any:
        """Validate configuration."""
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")


@dataclass
class TrainingMetrics:
    """Enhanced training metrics tracking."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_metric(self, metric_name: str, value: float):
        """Add a custom metric."""
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = []
        self.custom_metrics[metric_name].append(value)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest values for all metrics."""
        latest = {}
        
        if self.train_loss:
            latest["train_loss"] = self.train_loss[-1]
        if self.val_loss:
            latest["val_loss"] = self.val_loss[-1]
        if self.train_accuracy:
            latest["train_accuracy"] = self.train_accuracy[-1]
        if self.val_accuracy:
            latest["val_accuracy"] = self.val_accuracy[-1]
        if self.learning_rate:
            latest["learning_rate"] = self.learning_rate[-1]
        
        # Add custom metrics
        for metric_name, values in self.custom_metrics.items():
            if values:
                latest[metric_name] = values[-1]
        
        return latest


@dataclass
class ModelCheckpoint:
    """Enhanced model checkpoint with metadata."""
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    epoch: int
    step: int
    metrics: TrainingMetrics
    config: TrainingConfig
    model_hash: str
    timestamp: float
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingError(Exception):
    """Base exception for training errors."""
    pass


class DataError(TrainingError):
    """Exception for data-related errors."""
    pass


class ModelError(TrainingError):
    """Exception for model-related errors."""
    pass


class ValidationError(TrainingError):
    """Exception for validation-related errors."""
    pass


class AdvancedDataset(Dataset):
    """Advanced dataset with caching and preprocessing."""
    
    def __init__(self, data: Union[List, np.ndarray, pd.DataFrame], 
                 labels: Optional[Union[List, np.ndarray]] = None,
                 transform: Optional[Callable] = None,
                 cache: bool = True):
        
    """__init__ function."""
self.data = data
        self.labels = labels
        self.transform = transform
        self.cache = cache
        self.cached_items = {}
        
        if cache:
            logger.info("Dataset caching enabled")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache and idx in self.cached_items:
            return self.cached_items[idx]
        
        item = {"data": self.data[idx]}
        
        if self.labels is not None:
            item["label"] = self.labels[idx]
        
        if self.transform:
            item = self.transform(item)
        
        if self.cache:
            self.cached_items[idx] = item
        
        return item
    
    def clear_cache(self) -> Any:
        """Clear the cache."""
        self.cached_items.clear()


class AdvancedDataLoader:
    """Advanced data loader with prefetching and caching."""
    
    def __init__(self, dataset: Dataset, config: TrainingConfig):
        
    """__init__ function."""
self.dataset = dataset
        self.config = config
        self.prefetch_factor = 2
        self.pin_memory = config.pin_memory and torch.cuda.is_available()
        
    def create_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        total_size = len(self.dataset)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )
        
        return train_loader, val_loader, test_loader


class AdvancedModel(nn.Module):
    """Advanced model with enhanced features."""
    
    def __init__(self, config: TrainingConfig, num_classes: int):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Model architecture
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_encoding = nn.Parameter(torch.randn(1, config.max_sequence_length, config.hidden_size))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout_rate,
            activation=config.activation,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Model versioning
        self.model_hash = self._generate_model_hash()
        self.version = "1.0.0"
    
    def _init_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _generate_model_hash(self) -> str:
        """Generate a hash for the model architecture."""
        model_str = str(self.state_dict().keys())
        return hashlib.md5(model_str.encode()).hexdigest()[:8]
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with enhanced features."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.position_encoding[:, :seq_len, :]
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Transformer
        transformer_output = self.transformer(embeddings, src_key_padding_mask=~attention_mask.bool())
        
        # Pooling (mean pooling)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_output = (transformer_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_hash": self.model_hash,
            "version": self.version,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "architecture": self.config.model_name,
            "task_type": self.config.task_type.value
        }


class AdvancedOptimizer:
    """Advanced optimizer with multiple optimization strategies."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        
        # Parameter grouping
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        # Optimizer selection
        if config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                eps=1e-8
            )
        elif config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # Learning rate scheduler
        if config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.num_epochs
            )
        elif config.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.num_epochs // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
    
    def step(self, loss: torch.Tensor) -> float:
        """Perform optimization step with mixed precision."""
        if self.config.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Return gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        return total_norm
    
    def step_scheduler(self) -> Any:
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()


class EarlyStopping:
    """Enhanced early stopping with multiple strategies."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 restore_best_weights: bool = True, mode: str = "min"):
        
    """__init__ function."""
self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        
        if self.mode == "min":
            improved = val_score < self.best_score - self.min_delta
        else:
            improved = val_score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class ModelCheckpointer:
    """Enhanced model checkpointer with versioning and metadata."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.checkpoints = []
    
    def save_checkpoint(self, model: nn.Module, optimizer: AdvancedOptimizer,
                       epoch: int, step: int, metrics: TrainingMetrics,
                       is_best: bool = False) -> str:
        """Save model checkpoint with metadata."""
        checkpoint = ModelCheckpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.optimizer.state_dict(),
            scheduler_state=optimizer.scheduler.state_dict() if optimizer.scheduler else None,
            epoch=epoch,
            step=step,
            metrics=metrics,
            config=self.config,
            model_hash=model.model_hash,
            timestamp=time.time(),
            version=model.version,
            metadata={
                "is_best": is_best,
                "model_info": model.get_model_info(),
                "gpu_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            }
        )
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}_step_{step}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
        
        # Update checkpoint list
        self.checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.config.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, model: nn.Module, optimizer: AdvancedOptimizer,
                       checkpoint_path: str) -> Tuple[int, int, TrainingMetrics]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint.model_state)
        optimizer.optimizer.load_state_dict(checkpoint.optimizer_state)
        
        if checkpoint.scheduler_state and optimizer.scheduler:
            optimizer.scheduler.load_state_dict(checkpoint.scheduler_state)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint.epoch, checkpoint.step, checkpoint.metrics


class AdvancedTrainingSystem:
    """Advanced training system with comprehensive features."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = TrainingMetrics()
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            restore_best_weights=config.early_stopping_restore_best_weights
        )
        self.checkpointer = ModelCheckpointer(config)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model and optimizer
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        logger.info(f"AdvancedTrainingSystem initialized on {self.device}")
    
    def _setup_logging(self) -> Any:
        """Setup logging and monitoring."""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
        
        if self.config.use_mlflow:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.log_params(self.config.__dict__)
    
    def setup_data(self, dataset: Dataset):
        """Setup data loaders."""
        data_loader = AdvancedDataLoader(dataset, self.config)
        self.train_loader, self.val_loader, self.test_loader = data_loader.create_loaders()
        
        logger.info(f"Data loaders created: {len(self.train_loader)} train batches, "
                   f"{len(self.val_loader)} val batches, {len(self.test_loader)} test batches")
    
    def setup_model(self, num_classes: int):
        """Setup model and optimizer."""
        self.model = AdvancedModel(self.config, num_classes).to(self.device)
        
        # Multi-GPU setup
        if self.config.use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        self.optimizer = AdvancedOptimizer(self.model, self.config)
        
        # Log model info
        model_info = self.model.get_model_info()
        logger.info(f"Model setup complete: {model_info}")
        
        if self.config.use_wandb:
            wandb.log({"model_info": model_info})
    
    async def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with enhanced features."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            with autocast() if self.config.use_amp else torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Backward pass
            gradient_norm = self.optimizer.step(loss)
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "accuracy": f"{accuracy:.4f}",
                "grad_norm": f"{gradient_norm:.4f}"
            })
            
            # Log metrics
            if batch_idx % self.config.log_every == 0:
                self.metrics.train_loss.append(loss.item())
                self.metrics.train_accuracy.append(accuracy)
                self.metrics.gradient_norms.append(gradient_norm)
                
                if self.config.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_accuracy": accuracy,
                        "gradient_norm": gradient_norm,
                        "learning_rate": self.optimizer.optimizer.param_groups[0]['lr']
                    })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    async def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean().item()
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    async def train(self) -> TrainingMetrics:
        """Main training loop with comprehensive features."""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_accuracy = await self.train_epoch()
            
            # Validation
            val_loss, val_accuracy = await self.validate_epoch()
            
            # Update metrics
            self.metrics.val_loss.append(val_loss)
            self.metrics.val_accuracy.append(val_accuracy)
            self.metrics.learning_rate.append(self.optimizer.optimizer.param_groups[0]['lr'])
            self.metrics.epoch_times.append(time.time() - epoch_start_time)
            
            # Learning rate scheduling
            self.optimizer.step_scheduler()
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch_time": time.time() - epoch_start_time
                })
            
            # Checkpointing
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % self.config.save_every == 0 or is_best:
                checkpoint_path = self.checkpointer.save_checkpoint(
                    self.model, self.optimizer, epoch + 1, 0, self.metrics, is_best
                )
                
                if self.config.use_mlflow:
                    mlflow.pytorch.log_model(self.model, "model")
                    mlflow.log_artifact(checkpoint_path)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info("Early stopping triggered")
                break
        
        total_training_time = time.time() - start_time
        logger.info(f"Training completed in {total_training_time:.2f} seconds")
        
        return self.metrics
    
    async def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        if self.test_loader is None:
            raise ValueError("Test loader not set up")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            "test_loss": total_loss / len(self.test_loader),
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }
        
        logger.info(f"Test metrics: {metrics}")
        
        if self.config.use_wandb:
            wandb.log(metrics)
        
        return metrics


async def main():
    """Main function for testing the advanced training system."""
    # Create configuration
    config = TrainingConfig(
        model_name="transformer",
        task_type=TaskType.CLASSIFICATION,
        batch_size=8,
        num_epochs=3,
        use_wandb=False,
        use_mlflow=False
    )
    
    # Create training system
    training_system = AdvancedTrainingSystem(config)
    
    # Create dummy dataset
    num_samples = 1000
    num_classes = 3
    dummy_data = torch.randint(0, config.vocab_size, (num_samples, config.max_sequence_length))
    dummy_labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = AdvancedDataset(dummy_data, dummy_labels)
    
    # Setup data and model
    training_system.setup_data(dataset)
    training_system.setup_model(num_classes)
    
    # Train
    metrics = await training_system.train()
    
    # Evaluate
    test_metrics = await training_system.evaluate()
    
    print(f"Training completed. Test accuracy: {test_metrics['test_accuracy']:.4f}")


match __name__:
    case "__main__":
    asyncio.run(main()) 