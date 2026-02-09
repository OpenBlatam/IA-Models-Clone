from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings
import os
from functools import partial, reduce
from operator import itemgetter
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import (
from sklearn.model_selection import train_test_split, KFold
import optuna
from optuna.samplers import TPESampler
import mlflow
import wandb
from tqdm import tqdm
import torch.profiler
import cProfile
import pstats
import io
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup
    import yaml
from typing import Any, List, Dict, Optional
"""
🚀 Functional Model Training & Evaluation System
===============================================

Pure functional, declarative approach to model training and evaluation.
Uses data transformations, pure functions, and functional patterns instead of classes.

Key Principles:
- Pure functions with no side effects
- Data transformations over mutable state
- Composition over inheritance
- Immutable data structures
- Declarative configuration
"""


    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pure Data Structures (Immutable)
# ============================================================================

class TrainingMode(Enum):
    """Available training modes."""
    FINE_TUNE = "fine_tune"
    TRANSFER_LEARNING = "transfer_learning"
    FROM_SCRATCH = "from_scratch"
    LORA = "lora"
    P_TUNING = "p_tuning"

class ModelType(Enum):
    """Available model types."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    LLM = "llm"
    CUSTOM = "custom"

@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""
    model_type: ModelType
    training_mode: TrainingMode
    model_name: str
    dataset_path: str
    output_dir: str = "models"
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Advanced training
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = -1
    early_stopping_patience: int = 5
    save_steps: int = 500
    eval_steps: int = 500
    
    # Multi-GPU training
    distributed: bool = False
    num_gpus: int = 1
    use_data_parallel: bool = False
    use_distributed_data_parallel: bool = False
    
    # Evaluation
    eval_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    
    # Logging
    log_to_tensorboard: bool = True
    log_to_wandb: bool = False
    log_to_mlflow: bool = False
    
    # Debugging and monitoring
    debug_mode: bool = False
    detect_anomaly: bool = False
    gradient_checking: bool = False
    memory_profiling: bool = False
    performance_profiling: bool = False
    
    # Performance optimization
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_batch_optimization: bool = True
    enable_compilation: bool = False
    enable_amp: bool = True
    enable_gradient_checkpointing: bool = False
    enable_dynamic_batching: bool = True
    enable_pin_memory: bool = True
    enable_persistent_workers: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Advanced performance
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True
    enable_channels_last: bool = False
    enable_compile_mode: str = "default"

@dataclass(frozen=True)
class TrainingMetrics:
    """Immutable training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    training_time: float
    train_f1: Optional[float] = None
    val_f1: Optional[float] = None
    train_precision: Optional[float] = None
    val_precision: Optional[float] = None
    train_recall: Optional[float] = None
    val_recall: Optional[float] = None

@dataclass(frozen=True)
class EvaluationResult:
    """Immutable evaluation results."""
    model_name: str
    test_accuracy: float
    test_f1: float
    test_precision: float
    test_recall: float
    confusion_matrix: np.ndarray
    classification_report: str
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    inference_time_ms: float = 0.0

@dataclass(frozen=True)
class TrainingState:
    """Immutable training state."""
    config: TrainingConfig
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[Any]
    scaler: Optional[Any]
    device: torch.device
    current_epoch: int = 0
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    training_history: List[TrainingMetrics] = field(default_factory=list)

@dataclass(frozen=True)
class DeviceInfo:
    """Immutable device information."""
    device: torch.device
    gpu_available: bool
    num_gpus: int
    device_name: str
    memory_total: Optional[int] = None
    memory_allocated: Optional[int] = None

# ============================================================================
# Pure Functions - Device Management
# ============================================================================

def get_device_info() -> DeviceInfo:
    """Get device information in a pure functional way."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if gpu_available else 0
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    
    memory_total = None
    memory_allocated = None
    if gpu_available:
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_allocated = torch.cuda.memory_allocated(0)
    
    return DeviceInfo(
        device=device,
        gpu_available=gpu_available,
        num_gpus=num_gpus,
        device_name=device_name,
        memory_total=memory_total,
        memory_allocated=memory_allocated
    )

def setup_device_optimization(device_info: DeviceInfo, config: TrainingConfig) -> None:
    """Setup device optimization in a pure functional way."""
    if not device_info.gpu_available or not config.enable_gpu_optimization:
        return
    
    if config.enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    
    if config.enable_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    
    if config.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ============================================================================
# Pure Functions - Configuration Management
# ============================================================================

def create_default_config(
    model_name: str,
    dataset_path: str,
    model_type: ModelType = ModelType.TRANSFORMER,
    training_mode: TrainingMode = TrainingMode.FINE_TUNE
) -> TrainingConfig:
    """Create default configuration in a pure functional way."""
    device_info = get_device_info()
    
    # Auto-detect multi-GPU settings
    num_gpus = device_info.num_gpus
    use_data_parallel = num_gpus > 1
    
    # Calculate effective batch size
    effective_batch_size = 16 * 1 * num_gpus  # batch_size * gradient_accumulation * num_gpus
    
    return TrainingConfig(
        model_type=model_type,
        training_mode=training_mode,
        model_name=model_name,
        dataset_path=dataset_path,
        num_gpus=num_gpus,
        use_data_parallel=use_data_parallel,
        effective_batch_size=effective_batch_size
    )

def update_config(config: TrainingConfig, **updates) -> TrainingConfig:
    """Create a new config with updates (immutable update)."""
    return TrainingConfig(**{**config.__dict__, **updates})

def validate_config(config: TrainingConfig) -> Tuple[bool, List[str]]:
    """Validate configuration and return (is_valid, errors)."""
    errors = []
    
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    
    if config.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    
    if config.num_epochs <= 0:
        errors.append("num_epochs must be positive")
    
    if config.gradient_accumulation_steps < 1:
        errors.append("gradient_accumulation_steps must be >= 1")
    
    if config.eval_split <= 0 or config.eval_split >= 1:
        errors.append("eval_split must be between 0 and 1")
    
    if config.test_split <= 0 or config.test_split >= 1:
        errors.append("test_split must be between 0 and 1")
    
    if config.eval_split + config.test_split >= 1:
        errors.append("eval_split + test_split must be < 1")
    
    return len(errors) == 0, errors

# ============================================================================
# Pure Functions - Model Creation
# ============================================================================

def create_model(config: TrainingConfig, num_classes: int) -> nn.Module:
    """Create model in a pure functional way."""
    if config.model_type == ModelType.TRANSFORMER:
        return create_transformer_model(config.model_name, num_classes)
    elif config.model_type == ModelType.CUSTOM:
        return create_custom_model(num_classes)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

def create_transformer_model(model_name: str, num_classes: int) -> nn.Module:
    """Create transformer model in a pure functional way."""
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    return model

def create_custom_model(num_classes: int) -> nn.Module:
    """Create custom model in a pure functional way."""
    class CustomTransformer(nn.Module):
        def __init__(self, num_classes: int, vocab_size: int = 30522, hidden_size: int = 768):
            
    """__init__ function."""
super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1
            )
            self.classifier = nn.Linear(hidden_size, num_classes)
        
        def forward(self, input_ids, attention_mask=None) -> Any:
            embeddings = self.embedding(input_ids)
            if attention_mask is not None:
                embeddings = embeddings * attention_mask.unsqueeze(-1)
            encoded = self.transformer(embeddings)
            pooled = encoded.mean(dim=1)
            return self.classifier(pooled)
    
    return CustomTransformer(num_classes)

# ============================================================================
# Pure Functions - Optimizer and Scheduler Creation
# ============================================================================

def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Create optimizer in a pure functional way."""
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
    
    return optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=1e-8
    )

def create_scheduler(
    optimizer: optim.Optimizer, 
    config: TrainingConfig, 
    num_training_steps: int
) -> Any:
    """Create scheduler in a pure functional way."""
    
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

def create_scaler(config: TrainingConfig) -> Optional[Any]:
    """Create scaler for mixed precision in a pure functional way."""
    if config.enable_amp and config.mixed_precision:
        return torch.cuda.amp.GradScaler()
    return None

# ============================================================================
# Pure Functions - Data Loading
# ============================================================================

def create_dataset(texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512) -> Dataset:
    """Create dataset in a pure functional way."""
    class FunctionalDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length) -> Any:
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self) -> Any:
            return len(self.texts)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            text = self.texts[idx]
            label = self.labels[idx]
            
            if self.tokenizer:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
            else:
                return {
                    'text': text,
                    'labels': torch.tensor(label, dtype=torch.long)
                }
    
    return FunctionalDataset(texts, labels, tokenizer, max_length)

def load_and_split_data(config: TrainingConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """Load and split data in a pure functional way."""
    # Load data (simplified - replace with your data loading logic)
    data = pd.read_csv(config.dataset_path)
    texts = data['text'].tolist()
    labels = data['label'].tolist()
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=config.eval_split + config.test_split, random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, 
        test_size=config.test_split / (config.eval_split + config.test_split), 
        random_state=42
    )
    
    # Create datasets
    train_dataset = create_dataset(train_texts, train_labels)
    val_dataset = create_dataset(val_texts, val_labels)
    test_dataset = create_dataset(test_texts, test_labels)
    
    return train_dataset, val_dataset, test_dataset

def create_dataloader(
    dataset: Dataset, 
    config: TrainingConfig, 
    shuffle: bool = True
) -> DataLoader:
    """Create dataloader in a pure functional way."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.enable_pin_memory,
        persistent_workers=config.enable_persistent_workers,
        prefetch_factor=config.prefetch_factor
    )

# ============================================================================
# Pure Functions - Training Logic
# ============================================================================

def setup_debugging(config: TrainingConfig) -> None:
    """Setup debugging in a pure functional way."""
    if config.debug_mode:
        if config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        if config.gradient_checking:
            torch.autograd.set_grad_enabled(True)
        
        if config.memory_profiling:
            torch.cuda.empty_cache()

def calculate_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    task_type: str = "classification",
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate metrics in a pure functional way."""
    if task_type == "classification":
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        if probabilities is not None and len(np.unique(targets)) == 2:
            try:
                roc_auc = roc_auc_score(targets, probabilities[:, 1])
                metrics['roc_auc'] = roc_auc
            except:
                pass
        
        return metrics
    
    elif task_type == "regression":
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    config: TrainingConfig,
    device: torch.device
) -> Dict[str, float]:
    """Train one epoch in a pure functional way."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast() if scaler else torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        optimizer.zero_grad()
        
        # Collect metrics
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=-1) if hasattr(outputs, 'logits') else outputs
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(batch['labels'].cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate one epoch in a pure functional way."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Collect metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1) if hasattr(outputs, 'logits') else outputs
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def should_stop_early(
    current_val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    patience: int
) -> Tuple[bool, float, int]:
    """Check if training should stop early in a pure functional way."""
    if current_val_loss < best_val_loss:
        return False, current_val_loss, 0
    else:
        new_patience_counter = patience_counter + 1
        return new_patience_counter >= patience, best_val_loss, new_patience_counter

def save_model(
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    metrics: Dict[str, float]
) -> str:
    """Save model in a pure functional way."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / f"{config.model_name}_epoch_{epoch}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': epoch,
        'metrics': metrics
    }, model_path)
    
    return str(model_path)

# ============================================================================
# Pure Functions - Main Training Pipeline
# ============================================================================

def create_training_state(config: TrainingConfig) -> TrainingState:
    """Create initial training state in a pure functional way."""
    device_info = get_device_info()
    setup_device_optimization(device_info, config)
    setup_debugging(config)
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_and_split_data(config)
    
    # Create model
    num_classes = len(set(train_dataset.labels)) if hasattr(train_dataset, 'labels') else 2
    model = create_model(config, num_classes)
    model.to(device_info.device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    num_training_steps = len(train_dataset) // config.batch_size * config.num_epochs
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Create scaler
    scaler = create_scaler(config)
    
    return TrainingState(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device_info.device
    )

def train_model(config: TrainingConfig) -> Dict[str, Any]:
    """Main training function using pure functional approach."""
    logger.info("Starting functional training pipeline")
    
    # Create initial state
    state = create_training_state(config)
    
    # Create dataloaders
    train_dataloader = create_dataloader(state.config, train_dataset, shuffle=True)
    val_dataloader = create_dataloader(state.config, val_dataset, shuffle=False)
    
    # Training loop
    for epoch in range(state.config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{state.config.num_epochs}")
        
        # Train epoch
        train_metrics = train_epoch(
            state.model,
            train_dataloader,
            state.optimizer,
            state.scheduler,
            state.scaler,
            state.config,
            state.device
        )
        
        # Validate epoch
        val_metrics = validate_epoch(
            state.model,
            val_dataloader,
            state.device
        )
        
        # Create metrics record
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['loss'],
            train_accuracy=train_metrics['accuracy'],
            val_accuracy=val_metrics['accuracy'],
            learning_rate=state.optimizer.param_groups[0]['lr'],
            training_time=time.time(),
            train_f1=train_metrics.get('f1'),
            val_f1=val_metrics.get('f1'),
            train_precision=train_metrics.get('precision'),
            val_precision=val_metrics.get('precision'),
            train_recall=train_metrics.get('recall'),
            val_recall=val_metrics.get('recall')
        )
        
        # Update state
        state = TrainingState(
            config=state.config,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            device=state.device,
            current_epoch=epoch + 1,
            best_val_loss=min(state.best_val_loss, val_metrics['loss']),
            patience_counter=state.patience_counter,
            training_history=state.training_history + [metrics]
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_metrics['loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save model if best so far
        if val_metrics['loss'] == state.best_val_loss:
            model_path = save_model(state.model, state.config, epoch + 1, val_metrics)
            logger.info(f"Saved best model to {model_path}")
        
        # Early stopping check
        should_stop, new_best_loss, new_patience = should_stop_early(
            val_metrics['loss'],
            state.best_val_loss,
            state.patience_counter,
            state.config.early_stopping_patience
        )
        
        if should_stop:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Update patience counter
        state = TrainingState(
            config=state.config,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            device=state.device,
            current_epoch=state.current_epoch,
            best_val_loss=new_best_loss,
            patience_counter=new_patience,
            training_history=state.training_history
        )
    
    # Return results
    return {
        'config': state.config,
        'training_history': state.training_history,
        'best_val_loss': state.best_val_loss,
        'final_epoch': state.current_epoch,
        'model_path': str(Path(state.config.output_dir) / f"{state.config.model_name}_final.pt")
    }

# ============================================================================
# Quick Training Functions
# ============================================================================

async def quick_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5
) -> Dict[str, Any]:
    """Quick training function using functional approach."""
    config = create_default_config(model_name, dataset_path)
    config = update_config(config, output_dir=output_dir, num_epochs=num_epochs)
    
    is_valid, errors = validate_config(config)
    if not is_valid:
        raise ValueError(f"Invalid config: {errors}")
    
    return train_model(config)

async def functional_train_with_config(
    config_path: str
) -> Dict[str, Any]:
    """Train using YAML config file with functional approach."""
    
    with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        config_dict = yaml.safe_load(f)
    
    # Convert string enums
    if 'model_type' in config_dict:
        config_dict['model_type'] = ModelType(config_dict['model_type'])
    if 'training_mode' in config_dict:
        config_dict['training_mode'] = TrainingMode(config_dict['training_mode'])
    
    config = TrainingConfig(**config_dict)
    
    is_valid, errors = validate_config(config)
    if not is_valid:
        raise ValueError(f"Invalid config: {errors}")
    
    return train_model(config)

# ============================================================================
# Utility Functions
# ============================================================================

def get_training_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get training summary in a pure functional way."""
    history = results['training_history']
    if not history:
        return {'error': 'No training history available'}
    
    final_metrics = history[-1]
    best_metrics = min(history, key=lambda x: x.val_loss)
    
    return {
        'final_epoch': results['final_epoch'],
        'best_val_loss': results['best_val_loss'],
        'final_val_accuracy': final_metrics.val_accuracy,
        'best_val_accuracy': best_metrics.val_accuracy,
        'total_training_time': sum(m.training_time for m in history),
        'model_path': results['model_path']
    }

def demo():
    """Demo the functional training system."""
    async def run_demo():
        
    """run_demo function."""
print("🚀 Functional Training Demo")
        print("=" * 50)
        
        # Quick training example
        results = await quick_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sample_dataset.csv",
            num_epochs=3
        )
        
        summary = get_training_summary(results)
        print(f"Training completed!")
        print(f"Final validation accuracy: {summary['final_val_accuracy']:.4f}")
        print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
        print(f"Model saved to: {summary['model_path']}")
    
    asyncio.run(run_demo())

match __name__:
    case "__main__":
    demo() 