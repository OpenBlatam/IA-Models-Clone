from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from functional_utils import (
        from transformers import AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from pathlib import Path
    import yaml
from typing import Any, List, Dict, Optional
"""
🔧 Modular Training Framework
============================

Modular training framework using iteration and modularization.
Eliminates code duplication through reusable, composable training functions.

Key Principles:
- Iteration over duplication
- Modularization over repetition
- Composition over inheritance
- Pure functions with no side effects
- Immutable data transformations
"""


    Result, ValidationResult, safe_execute, timer_context,
    transform_list, filter_list, group_by, sort_by,
    pipe, compose, log_function_call, log_data_info,
    iterate_with_progress, iterate_batches
)

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# Generic Types
# ============================================================================

T = TypeVar('T')
U = TypeVar('U')

# ============================================================================
# Core Data Structures
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
    
    # Performance optimization
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_batch_optimization: bool = True
    enable_compilation: bool = False
    enable_amp: bool = True
    num_workers: int = 4

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
# Device Management Modules
# ============================================================================

class DeviceManager:
    """Modular device management using iteration and composition."""
    
    @staticmethod
    def get_device_info() -> DeviceInfo:
        """Get device information in a pure functional way."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_gpu_available = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if is_gpu_available else 0
        device_name = torch.cuda.get_device_name(0) if is_gpu_available else "CPU"
        
        memory_total = None
        memory_allocated = None
        if is_gpu_available:
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_allocated = torch.cuda.memory_allocated(0)
        
        return DeviceInfo(
            device=device,
            gpu_available=is_gpu_available,
            num_gpus=num_gpus,
            device_name=device_name,
            memory_total=memory_total,
            memory_allocated=memory_allocated
        )
    
    @staticmethod
    def setup_device_optimization(device_info: DeviceInfo, config: TrainingConfig) -> None:
        """Setup device optimization using modular approach."""
        if not device_info.gpu_available or not config.enable_gpu_optimization:
            return
        
        optimization_settings = [
            (config.enable_cudnn_benchmark, lambda: setattr(torch.backends.cudnn, 'benchmark', True)),
            (config.enable_cudnn_deterministic, lambda: setattr(torch.backends.cudnn, 'deterministic', True)),
            (config.enable_tf32, lambda: setattr(torch.backends.cuda.matmul, 'allow_tf32', True)),
            (config.enable_tf32, lambda: setattr(torch.backends.cudnn, 'allow_tf32', True))
        ]
        
        # Apply optimizations using iteration
        for should_apply_optimization, optimization_fn in optimization_settings:
            if should_apply_optimization:
                safe_execute(optimization_fn)
    
    @staticmethod
    def setup_debugging(config: TrainingConfig) -> None:
        """Setup debugging using modular approach."""
        if not config.debug_mode:
            return
        
        debugging_settings = [
            (config.detect_anomaly, lambda: torch.autograd.set_detect_anomaly(True)),
            (config.gradient_checking, lambda: torch.autograd.set_grad_enabled(True)),
            (config.memory_profiling, lambda: torch.cuda.empty_cache())
        ]
        
        # Apply debugging settings using iteration
        for should_apply_debugging, debugging_fn in debugging_settings:
            if should_apply_debugging:
                safe_execute(debugging_fn)

# ============================================================================
# Model Creation Modules
# ============================================================================

class ModelFactory:
    """Modular model factory using iteration and composition."""
    
    @staticmethod
    def create_model(config: TrainingConfig, num_classes: int) -> nn.Module:
        """Create model using modular approach."""
        model_creators = {
            ModelType.TRANSFORMER: ModelFactory._create_transformer_model,
            ModelType.CUSTOM: ModelFactory._create_custom_model,
        }
        
        if config.model_type not in model_creators:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        return model_creators[config.model_type](config.model_name, num_classes)
    
    @staticmethod
    def _create_transformer_model(model_name: str, num_classes: int) -> nn.Module:
        """Create transformer model."""
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        return model
    
    @staticmethod
    def _create_custom_model(num_classes: int) -> nn.Module:
        """Create custom model."""
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
# Optimizer and Scheduler Modules
# ============================================================================

class OptimizerFactory:
    """Modular optimizer factory using iteration and composition."""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """Create optimizer using modular approach."""
        # Define parameter groups
        no_decay_parameters = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [param for name, param in model.named_parameters() 
                          if not any(no_decay_param in name for no_decay_param in no_decay_parameters)],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [param for name, param in model.named_parameters() 
                          if any(no_decay_param in name for no_decay_param in no_decay_parameters)],
                'weight_decay': 0.0,
            }
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            eps=1e-8
        )

class SchedulerFactory:
    """Modular scheduler factory using iteration and composition."""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer, 
        config: TrainingConfig, 
        num_training_steps: int
    ) -> Any:
        """Create scheduler using modular approach."""
        
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

class ScalerFactory:
    """Modular scaler factory using iteration and composition."""
    
    @staticmethod
    def create_scaler(config: TrainingConfig) -> Optional[Any]:
        """Create scaler using modular approach."""
        if config.enable_amp and config.mixed_precision:
            return torch.cuda.amp.GradScaler()
        return None

# ============================================================================
# Data Loading Modules
# ============================================================================

class DataLoaderFactory:
    """Modular data loader factory using iteration and composition."""
    
    @staticmethod
    def create_dataset(
        texts: List[str], 
        labels: List[int], 
        tokenizer=None, 
        max_length: int = 512
    ) -> Dataset:
        """Create dataset using modular approach."""
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
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset, 
        config: TrainingConfig, 
        shuffle: bool = True
    ) -> DataLoader:
        """Create dataloader using modular approach."""
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
# Training Pipeline Modules
# ============================================================================

class TrainingPipeline:
    """Modular training pipeline using iteration and composition."""
    
    @staticmethod
    @log_function_call
    def create_training_state(config: TrainingConfig) -> TrainingState:
        """Create training state using modular approach."""
        with timer_context("Creating training state"):
            # Get device info
            device_info = DeviceManager.get_device_info()
            
            # Setup optimizations
            DeviceManager.setup_device_optimization(device_info, config)
            DeviceManager.setup_debugging(config)
            
            # Load data
            train_dataset, val_dataset, test_dataset = TrainingPipeline._load_and_split_data(config)
            
            # Create model
            num_classes = len(set(train_dataset.labels)) if hasattr(train_dataset, 'labels') else 2
            model = ModelFactory.create_model(config, num_classes)
            model.to(device_info.device)
            
            # Create optimizer and scheduler
            optimizer = OptimizerFactory.create_optimizer(model, config)
            num_training_steps = len(train_dataset) // config.batch_size * config.num_epochs
            scheduler = SchedulerFactory.create_scheduler(optimizer, config, num_training_steps)
            
            # Create scaler
            scaler = ScalerFactory.create_scaler(config)
            
            return TrainingState(
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device_info.device
            )
    
    @staticmethod
    def _load_and_split_data(config: TrainingConfig) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split data using modular approach."""
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
        train_dataset = DataLoaderFactory.create_dataset(train_texts, train_labels)
        val_dataset = DataLoaderFactory.create_dataset(val_texts, val_labels)
        test_dataset = DataLoaderFactory.create_dataset(test_texts, test_labels)
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    @log_function_call
    def train_model(config: TrainingConfig) -> Dict[str, Any]:
        """Train model using modular pipeline."""
        logger.info("Starting modular training pipeline")
        
        # Create initial state
        state = TrainingPipeline.create_training_state(config)
        
        # Create dataloaders
        train_dataloader = DataLoaderFactory.create_dataloader(state.config, train_dataset, shuffle=True)
        val_dataloader = DataLoaderFactory.create_dataloader(state.config, val_dataset, shuffle=False)
        
        # Training loop using iteration
        for epoch in TrainingPipeline._iterate_epochs(state.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{state.config.num_epochs}")
            
            # Train epoch
            train_metrics = TrainingPipeline._train_epoch(state, train_dataloader)
            
            # Validate epoch
            val_metrics = TrainingPipeline._validate_epoch(state, val_dataloader)
            
            # Create metrics record
            metrics = TrainingPipeline._create_training_metrics(epoch, train_metrics, val_metrics, state)
            
            # Update state
            state = TrainingPipeline._update_training_state(state, metrics, val_metrics)
            
            # Log metrics
            TrainingPipeline._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save model if best so far
            if val_metrics['loss'] == state.best_val_loss:
                model_path = TrainingPipeline._save_model(state, epoch, val_metrics)
                logger.info(f"Saved best model to {model_path}")
            
            # Early stopping check
            should_stop, new_best_loss, new_patience = TrainingPipeline._check_early_stopping(
                val_metrics['loss'], state, config
            )
            
            if should_stop:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Update patience counter
            state = TrainingPipeline._update_patience_counter(state, new_best_loss, new_patience)
        
        # Return results
        return TrainingPipeline._create_training_results(state)
    
    @staticmethod
    def _iterate_epochs(num_epochs: int):
        """Iterate over epochs."""
        return range(num_epochs)
    
    @staticmethod
    def _train_epoch(state: TrainingState, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch using modular approach."""
        state.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Process batches using iteration
        for batch_idx, batch in enumerate(iterate_with_progress(dataloader, desc="Training")):
            # Move batch to device
            batch = TrainingPipeline._move_batch_to_device(batch, state.device)
            
            # Forward pass
            outputs = TrainingPipeline._forward_pass(state, batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Backward pass
            TrainingPipeline._backward_pass(state, loss)
            
            # Collect metrics
            total_loss += loss.item()
            predictions = TrainingPipeline._extract_predictions(outputs)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        metrics = TrainingPipeline._calculate_epoch_metrics(all_predictions, all_targets, total_loss, len(dataloader))
        return metrics
    
    @staticmethod
    def _validate_epoch(state: TrainingState, dataloader: DataLoader) -> Dict[str, float]:
        """Validate one epoch using modular approach."""
        state.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Process batches using iteration
            for batch in iterate_with_progress(dataloader, desc="Validation"):
                # Move batch to device
                batch = TrainingPipeline._move_batch_to_device(batch, state.device)
                
                # Forward pass
                outputs = TrainingPipeline._forward_pass(state, batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Collect metrics
                total_loss += loss.item()
                predictions = TrainingPipeline._extract_predictions(outputs)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        metrics = TrainingPipeline._calculate_epoch_metrics(all_predictions, all_targets, total_loss, len(dataloader))
        return metrics
    
    @staticmethod
    def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
        """Move batch to device."""
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    @staticmethod
    def _forward_pass(state: TrainingState, batch: Dict):
        """Perform forward pass."""
        with torch.cuda.amp.autocast() if state.scaler else torch.no_grad():
            return state.model(**batch)
    
    @staticmethod
    def _backward_pass(state: TrainingState, loss: torch.Tensor):
        """Perform backward pass."""
        if state.scaler:
            state.scaler.scale(loss).backward()
            state.scaler.step(state.optimizer)
            state.scaler.update()
        else:
            loss.backward()
            state.optimizer.step()
        
        if state.scheduler:
            state.scheduler.step()
        
        state.optimizer.zero_grad()
    
    @staticmethod
    def _extract_predictions(outputs) -> torch.Tensor:
        """Extract predictions from outputs."""
        return torch.argmax(outputs.logits, dim=-1) if hasattr(outputs, 'logits') else outputs
    
    @staticmethod
    def _calculate_epoch_metrics(predictions: List, targets: List, total_loss: float, num_batches: int) -> Dict[str, float]:
        """Calculate epoch metrics."""
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'loss': total_loss / num_batches
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        return metrics
    
    @staticmethod
    def _create_training_metrics(epoch: int, train_metrics: Dict, val_metrics: Dict, state: TrainingState) -> TrainingMetrics:
        """Create training metrics."""
        return TrainingMetrics(
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
    
    @staticmethod
    def _update_training_state(state: TrainingState, metrics: TrainingMetrics, val_metrics: Dict) -> TrainingState:
        """Update training state."""
        return TrainingState(
            config=state.config,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            device=state.device,
            current_epoch=state.current_epoch + 1,
            best_val_loss=min(state.best_val_loss, val_metrics['loss']),
            patience_counter=state.patience_counter,
            training_history=state.training_history + [metrics]
        )
    
    @staticmethod
    def _log_epoch_metrics(epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_metrics['loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    
    @staticmethod
    def _save_model(state: TrainingState, epoch: int, metrics: Dict) -> str:
        """Save model."""
        output_dir = Path(state.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        model_path = output_dir / f"{state.config.model_name}_epoch_{epoch + 1}.pt"
        torch.save({
            'model_state_dict': state.model.state_dict(),
            'config': state.config,
            'epoch': epoch + 1,
            'metrics': metrics
        }, model_path)
        
        return str(model_path)
    
    @staticmethod
    def _check_early_stopping(current_val_loss: float, state: TrainingState, config: TrainingConfig) -> Tuple[bool, float, int]:
        """Check early stopping."""
        if current_val_loss < state.best_val_loss:
            return False, current_val_loss, 0
        else:
            new_patience_counter = state.patience_counter + 1
            return new_patience_counter >= config.early_stopping_patience, state.best_val_loss, new_patience_counter
    
    @staticmethod
    def _update_patience_counter(state: TrainingState, new_best_loss: float, new_patience: int) -> TrainingState:
        """Update patience counter."""
        return TrainingState(
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
    
    @staticmethod
    def _create_training_results(state: TrainingState) -> Dict[str, Any]:
        """Create training results."""
        return {
            'config': state.config,
            'training_history': state.training_history,
            'best_val_loss': state.best_val_loss,
            'final_epoch': state.current_epoch,
            'model_path': str(Path(state.config.output_dir) / f"{state.config.model_name}_final.pt")
        }

# ============================================================================
# Quick Training Functions (Composed from modules)
# ============================================================================

async def quick_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5
) -> Dict[str, Any]:
    """Quick training using modular approach."""
    config = TrainingPipeline._create_default_config(model_name, dataset_path)
    config = TrainingPipeline._update_config(config, output_dir=output_dir, num_epochs=num_epochs)
    
    is_valid, errors = TrainingPipeline._validate_config(config)
    if not is_valid:
        raise ValueError(f"Invalid config: {errors}")
    
    return TrainingPipeline.train_model(config)

async def modular_train_with_config(config_path: str) -> Dict[str, Any]:
    """Train using YAML config file with modular approach."""
    
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
    
    is_valid, errors = TrainingPipeline._validate_config(config)
    if not is_valid:
        raise ValueError(f"Invalid config: {errors}")
    
    return TrainingPipeline.train_model(config)

# ============================================================================
# Configuration Utilities (Composed from modules)
# ============================================================================

class ConfigManager:
    """Modular configuration manager using iteration and composition."""
    
    @staticmethod
    def create_default_config(
        model_name: str,
        dataset_path: str,
        model_type: ModelType = ModelType.TRANSFORMER,
        training_mode: TrainingMode = TrainingMode.FINE_TUNE
    ) -> TrainingConfig:
        """Create default configuration using modular approach."""
        device_info = DeviceManager.get_device_info()
        
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
    
    @staticmethod
    def update_config(config: TrainingConfig, **updates) -> TrainingConfig:
        """Update configuration using modular approach."""
        return TrainingConfig(**{**config.__dict__, **updates})
    
    @staticmethod
    def validate_config(config: TrainingConfig) -> Tuple[bool, List[str]]:
        """Validate configuration using modular approach."""
        errors = []
        
        validation_rules = [
            (config.batch_size <= 0, "batch_size must be positive"),
            (config.learning_rate <= 0, "learning_rate must be positive"),
            (config.num_epochs <= 0, "num_epochs must be positive"),
            (config.gradient_accumulation_steps < 1, "gradient_accumulation_steps must be >= 1"),
            (config.eval_split <= 0 or config.eval_split >= 1, "eval_split must be between 0 and 1"),
            (config.test_split <= 0 or config.test_split >= 1, "test_split must be between 0 and 1"),
            (config.eval_split + config.test_split >= 1, "eval_split + test_split must be < 1")
        ]
        
        # Apply validation rules using iteration
        for condition, error_message in validation_rules:
            if condition:
                errors.append(error_message)
        
        return len(errors) == 0, errors

# ============================================================================
# Demo Functions
# ============================================================================

def demo_modular_training():
    """Demo the modular training framework."""
    print("🔧 Modular Training Demo")
    print("=" * 50)
    
    # Test device management
    print("Device Management:")
    device_info = DeviceManager.get_device_info()
    print(f"  Device: {device_info.device}")
    print(f"  GPU Available: {device_info.gpu_available}")
    print(f"  Num GPUs: {device_info.num_gpus}")
    
    # Test configuration management
    print("\nConfiguration Management:")
    config = ConfigManager.create_default_config("test-model", "data/test.csv")
    print(f"  Created config: {config.model_name}")
    
    updated_config = ConfigManager.update_config(config, batch_size=32)
    print(f"  Updated batch size: {updated_config.batch_size}")
    
    is_valid, errors = ConfigManager.validate_config(updated_config)
    print(f"  Config valid: {is_valid}")
    
    # Test model factory
    print("\nModel Factory:")
    model = ModelFactory.create_model(config, num_classes=3)
    print(f"  Created model: {type(model).__name__}")
    
    # Test optimizer factory
    print("\nOptimizer Factory:")
    optimizer = OptimizerFactory.create_optimizer(model, config)
    print(f"  Created optimizer: {type(optimizer).__name__}")
    
    print("\n🎉 All modular training functions working correctly!")

match __name__:
    case "__main__":
    demo_modular_training() 