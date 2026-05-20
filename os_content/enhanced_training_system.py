from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

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
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import os
from pathlib import Path
import json
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import yaml
import wandb
from tqdm import tqdm
import warnings
import gradio as gr
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Enhanced Training System for OS Content
Comprehensive training pipeline with efficient data loading, validation splits,
early stopping, learning rate scheduling, evaluation metrics, and error handling.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

class TaskType(Enum):
    """Supported task types for training."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    TRANSLATION = "translation"

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Model parameters
    model_name: str = "transformer"
    task_type: TaskType = TaskType.CLASSIFICATION
    vocab_size: int = 30000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_sequence_length: int = 512
    num_workers: int = 4
    
    # Optimization parameters
    use_amp: bool = True
    use_multi_gpu: bool = False
    use_distributed: bool = False
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 1000
    max_checkpoints: int = 5
    
    # Logging
    log_every: int = 100
    use_wandb: bool = True
    project_name: str = "os_content_training"
    
    # Cross-validation
    use_cross_validation: bool = False
    n_folds: int = 5

@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

# ============================================================================
# ERROR HANDLING AND VALIDATION
# ============================================================================

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
    """Exception for validation errors."""
    pass

def validate_config(config: TrainingConfig) -> None:
    """Validate training configuration."""
    if config.batch_size <= 0:
        raise ValidationError("Batch size must be positive")
    
    if config.learning_rate <= 0:
        raise ValidationError("Learning rate must be positive")
    
    if not (0 < config.train_split < 1):
        raise ValidationError("Train split must be between 0 and 1")
    
    if abs(config.train_split + config.val_split + config.test_split - 1.0) > 1e-6:
        raise ValidationError("Data splits must sum to 1.0")
    
    if config.num_epochs <= 0:
        raise ValidationError("Number of epochs must be positive")

def handle_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Handle NaN and Inf values in tensors."""
    if torch.isnan(tensor).any():
        logger.warning(f"NaN detected in {name}, replacing with 0")
        tensor = torch.nan_to_num(tensor, nan=0.0)
    
    if torch.isinf(tensor).any():
        logger.warning(f"Inf detected in {name}, replacing with large values")
        tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
    
    return tensor

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

class TextDataset(Dataset):
    """Text dataset for training."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer=None, max_length: int = 512):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if not texts:
            raise DataError("Texts list cannot be empty")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            text = self.texts[idx]
            
            if self.tokenizer:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Remove batch dimension
                item = {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0)
                }
            else:
                # Simple character-level encoding
                item = {'text': torch.tensor([ord(c) for c in text[:self.max_length]], dtype=torch.long)}
            
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return item
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise DataError(f"Failed to load item {idx}")

class DataLoaderManager:
    """Manages data loading and splitting."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def create_data_loaders(self, dataset: Dataset, tokenizer=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders with proper splitting."""
        try:
            total_size = len(dataset)
            train_size = int(self.config.train_split * total_size)
            val_size = int(self.config.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create samplers for distributed training
            train_sampler = None
            val_sampler = None
            test_sampler = None
            
            if self.config.use_distributed:
                train_sampler = DistributedSampler(train_dataset)
                val_sampler = DistributedSampler(val_dataset, shuffle=False)
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                sampler=test_sampler,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            return self.train_loader, self.val_loader, self.test_loader
            
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            raise DataError(f"Failed to create data loaders: {e}")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class TransformerClassifier(nn.Module):
    """Transformer-based classifier."""
    
    def __init__(self, config: TrainingConfig, num_classes: int):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_sequence_length, config.hidden_size))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        try:
            # Handle NaN/Inf values
            input_ids = handle_nan_inf(input_ids, "input_ids")
            
            # Embeddings
            embeddings = self.embedding(input_ids)
            embeddings = embeddings + self.pos_encoding[:, :embeddings.size(1), :]
            
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            
            # Transformer encoding
            encoded = self.transformer(embeddings, src_key_padding_mask=(attention_mask == 0))
            
            # Global average pooling
            pooled = encoded.mean(dim=1)
            
            # Classification
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise ModelError(f"Forward pass failed: {e}")

# ============================================================================
# TRAINING COMPONENTS
# ============================================================================

class EarlyStopping:
    """Early stopping mechanism."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        
    """__init__ function."""
self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop

class ModelCheckpointer:
    """Model checkpointing utility."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.checkpoints = []
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler, epoch: int, metrics: TrainingMetrics, 
                       is_best: bool = False) -> str:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': metrics,
                'config': self.config
            }
            
            # Save checkpoint
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = self.save_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
            
            # Manage checkpoint history
            self.checkpoints.append(checkpoint_path)
            if len(self.checkpoints) > self.config.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise TrainingError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler, checkpoint_path: str) -> Tuple[int, TrainingMetrics]:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            return checkpoint['epoch'], checkpoint['metrics']
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise TrainingError(f"Failed to load checkpoint: {e}")

# ============================================================================
# MAIN TRAINING SYSTEM
# ============================================================================

class EnhancedTrainingSystem:
    """Enhanced training system with comprehensive features."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
# Validate configuration
        validate_config(config)
        self.config = config
        
        # Setup device and distributed training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler() if config.use_amp else None
        
        # Initialize components
        self.metrics = TrainingMetrics()
        self.checkpointer = ModelCheckpointer(config)
        self.data_manager = DataLoaderManager(config)
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
    
    def _setup_logging(self) -> Any:
        """Setup logging and experiment tracking."""
        if self.config.use_wandb:
            try:
                wandb.init(project=self.config.project_name, config=vars(self.config))
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.config.use_wandb = False
        
        # Setup file logging
        log_file = Path("training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    def setup_model(self, num_classes: int, tokenizer=None):
        """Setup model, optimizer, and loss function."""
        try:
            # Create model
            self.model = TransformerClassifier(self.config, num_classes)
            
            # Move to device and setup multi-GPU
            if self.config.use_multi_gpu and torch.cuda.device_count() > 1:
                self.model = DataParallel(self.model)
            
            if self.config.use_distributed:
                self.model = DistributedDataParallel(self.model)
            
            self.model.to(self.device)
            
            # Setup optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Setup scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.config.warmup_steps
            )
            
            # Setup loss function
            if self.config.task_type == TaskType.CLASSIFICATION:
                self.criterion = nn.CrossEntropyLoss()
            elif self.config.task_type == TaskType.REGRESSION:
                self.criterion = nn.MSELoss()
            else:
                raise ValueError(f"Unsupported task type: {self.config.task_type}")
            
            logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise ModelError(f"Failed to setup model: {e}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision
                if self.config.use_amp:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                
                # Handle NaN/Inf values
                loss = handle_nan_inf(loss, "loss")
                
                # Backward pass
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    if self.config.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                if self.config.task_type == TaskType.CLASSIFICATION:
                    predictions = torch.argmax(outputs, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    if self.config.use_amp:
                        with autocast():
                            outputs = self.model(input_ids, attention_mask)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, labels)
                    
                    # Handle NaN/Inf values
                    loss = handle_nan_inf(loss, "val_loss")
                    
                    # Update metrics
                    total_loss += loss.item()
                    if self.config.task_type == TaskType.CLASSIFICATION:
                        predictions = torch.argmax(outputs, dim=-1)
                        correct_predictions += (predictions == labels).sum().item()
                        total_predictions += labels.size(0)
                
                except Exception as e:
                    logger.error(f"Error in validation step: {e}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              start_epoch: int = 0) -> TrainingMetrics:
        """Main training loop."""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            try:
                # Training
                train_loss, train_accuracy = self.train_epoch(train_loader)
                
                # Validation
                val_loss, val_accuracy = self.validate_epoch(val_loader)
                
                # Update metrics
                self.metrics.train_loss.append(train_loss)
                self.metrics.val_loss.append(val_loss)
                self.metrics.train_accuracy.append(train_accuracy)
                self.metrics.val_accuracy.append(val_accuracy)
                self.metrics.learning_rate.append(self.scheduler.get_last_lr()[0])
                self.metrics.epoch_times.append(time.time() - epoch_start_time)
                
                # Logging
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )
                
                # Wandb logging
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })
                
                # Checkpointing
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                if (epoch + 1) % self.config.save_every == 0 or is_best:
                    self.checkpointer.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.metrics, is_best
                    )
                
                # Early stopping
                if self.early_stopping(val_loss):
                    logger.info("Early stopping triggered")
                    break
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                continue
        
        logger.info("Training completed!")
        return self.metrics

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = yaml.safe_load(f)
        
        return TrainingConfig(**config_dict)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise ValidationError(f"Failed to load config: {e}")

def save_config(config: TrainingConfig, config_path: str):
    """Save configuration to YAML file."""
    try:
        config_dict = vars(config)
        # Convert enum to string
        config_dict['task_type'] = config_dict['task_type'].value
        
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_dict, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise ValidationError(f"Failed to save config: {e}")

def gradio_predict(text: str, model: nn.Module, tokenizer, device: str) -> str:
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred].item()
        label = f"Class {pred} (confidence: {confidence:.2f})"
        return label

def launch_gradio_demo(model: nn.Module, tokenizer, device: str = 'cpu'):
    
    """launch_gradio_demo function."""
interface = gr.Interface(
        fn=lambda text: gradio_predict(text, model, tokenizer, device),
        inputs=gr.Textbox(lines=2, placeholder="Enter text for classification..."),
        outputs=gr.Textbox(label="Prediction"),
        title="Text Classification Demo",
        description="Enter text to see the model's prediction."
    )
    interface.launch()

def gradio_demo(model: nn.Module, tokenizer, config: TrainingConfig):
    """Gradio demo for model inference and visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    def predict(text: str):
        
    """predict function."""
with torch.no_grad():
            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=config.max_sequence_length,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
            return {f"Class {i}": float(prob) for i, prob in enumerate(probs)}

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=2, placeholder="Enter text for inference..."),
        outputs=gr.Label(num_top_classes=3),
        title="Transformer Model Inference Demo",
        description="Enter text to see model predictions."
    )
    interface.launch()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        model_name="transformer",
        task_type=TaskType.CLASSIFICATION,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=5,
        use_amp=True
    )
    
    # Create training system
    trainer = EnhancedTrainingSystem(config)
    
    # Example data (replace with actual data)
    texts = ["This is a positive review", "This is a negative review"] * 100
    labels = [1, 0] * 100
    
    # Create dataset
    dataset = TextDataset(texts, labels)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.data_manager.create_data_loaders(dataset)
    
    # Setup model
    trainer.setup_model(num_classes=2)
    
    # Train
    metrics = trainer.train(train_loader, val_loader)
    
    print("Training completed successfully!") 
    # Example Gradio demo usage (uncomment and adapt as needed)
    # gradio_demo(trainer.model, tokenizer, config) 