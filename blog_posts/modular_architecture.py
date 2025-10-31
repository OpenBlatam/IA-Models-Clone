from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd
import numpy as np
from PIL import Image
import structlog
from transformers import AutoModel, AutoTokenizer, AutoConfig
from diffusers import DiffusionPipeline, UNet2DConditionModel
import wandb
import tensorboard
from tensorboard import program
import matplotlib.pyplot as plt
import seaborn as sns
            from torch.utils.tensorboard import SummaryWriter
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
            from sklearn.metrics import roc_curve
from typing import Any, List, Dict, Optional
import asyncio
"""
Modular Architecture Framework
==============================

This module provides a comprehensive modular code structure with separate files
for models, data loading, training, and evaluation. This follows production-grade
standards and best practices for deep learning projects.

Key Components:
1. Model Architecture Module
2. Data Loading Module
3. Training Module
4. Evaluation Module
5. Configuration Management
6. Utility Functions
"""


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: str  # transformer, cnn, diffusion, etc.
    model_name: str
    num_classes: Optional[int] = None
    input_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    dropout: float = 0.1
    activation: str = "relu"
    pretrained: bool = True
    freeze_backbone: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    pin_memory: bool = True
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    transform_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "cosine"
    loss_function: str = "cross_entropy"
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 10
    save_best_model: bool = True
    checkpoint_dir: str = "checkpoints"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation process."""
    metrics: List[str]
    save_predictions: bool = True
    save_plots: bool = True
    output_dir: str = "evaluation_results"
    threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str
    model_config: ModelConfig
    data_config: DataConfig
    training_config: TrainingConfig
    evaluation_config: EvaluationConfig
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "model_config": self.model_config.to_dict(),
            "data_config": self.data_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "evaluation_config": self.evaluation_config.to_dict()
        }
    
    def save(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = yaml.safe_load(f)
        
        return cls(
            experiment_name=data["experiment_name"],
            model_config=ModelConfig(**data["model_config"]),
            data_config=DataConfig(**data["data_config"]),
            training_config=TrainingConfig(**data["training_config"]),
            evaluation_config=EvaluationConfig(**data["evaluation_config"])
        )


# =============================================================================
# MODEL ARCHITECTURE MODULE
# =============================================================================

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation."""
        pass
    
    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, filepath)
        self.logger.info("Model saved", filepath=filepath)
    
    @classmethod
    def load_model(cls, filepath: str, config: ModelConfig) -> 'BaseModel':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class TransformerModel(BaseModel):
    """Transformer-based model for various tasks."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Load pretrained transformer
        if config.pretrained:
            self.transformer = AutoModel.from_pretrained(config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        else:
            transformer_config = AutoConfig.from_pretrained(config.model_name)
            self.transformer = AutoModel.from_config(transformer_config)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Classification head
        if config.num_classes:
            hidden_size = self.transformer.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, config.hidden_dim or hidden_size),
                nn.ReLU() if config.activation == "relu" else nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim or hidden_size, config.num_classes)
            )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        
        if hasattr(self, 'classifier'):
            return self.classifier(pooled_output)
        return pooled_output


class CNNModel(BaseModel):
    """CNN-based model for image classification."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        if config.num_classes:
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(256, config.hidden_dim or 512),
                nn.ReLU() if config.activation == "relu" else nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim or 512, config.num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        if hasattr(self, 'classifier'):
            x = self.classifier(x)
        return x


class DiffusionModel(BaseModel):
    """Diffusion model for image generation."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # UNet backbone for diffusion
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=768,
        )
        
        # Text encoder for conditioning
        if config.pretrained:
            self.text_encoder = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.unet(x, timesteps, encoder_hidden_states).sample


class ModelFactory:
    """Factory for creating model instances."""
    
    _models = {
        "transformer": TransformerModel,
        "cnn": CNNModel,
        "diffusion": DiffusionModel
    }
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseModel:
        """Create model instance based on configuration."""
        if config.model_type not in cls._models:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        model_class = cls._models[config.model_type]
        return model_class(config)
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """Register a new model type."""
        cls._models[model_type] = model_class


# =============================================================================
# DATA LOADING MODULE
# =============================================================================

class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets."""
    
    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        
    """__init__ function."""
self.data_path = Path(data_path)
        self.transform = transform
        self.logger = structlog.get_logger(__name__)
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        pass


class TabularDataset(BaseDataset):
    """Dataset for tabular data."""
    
    def __init__(self, data_path: str, target_column: str, feature_columns: Optional[List[str]] = None, transform: Optional[Callable] = None):
        
    """__init__ function."""
super().__init__(data_path, transform)
        
        # Load data
        if self.data_path.suffix.lower() == '.csv':
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.suffix.lower() in ['.xlsx', '.xls']:
            self.data = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        self.target_column = target_column
        self.feature_columns = feature_columns or [col for col in self.data.columns if col != target_column]
        
        # Prepare features and targets
        self.features = self.data[self.feature_columns].values.astype(np.float32)
        self.targets = self.data[self.target_column].values
        
        self.logger.info("Tabular dataset loaded", 
                        samples=len(self.data),
                        features=len(self.feature_columns),
                        target_column=target_column)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
        
        return features, target


class ImageDataset(BaseDataset):
    """Dataset for image data."""
    
    def __init__(self, data_path: str, target_column: Optional[str] = None, transform: Optional[Callable] = None):
        
    """__init__ function."""
super().__init__(data_path, transform)
        
        # Get image files
        self.image_files = []
        self.targets = []
        
        if self.data_path.is_file():
            # Single image file
            self.image_files = [self.data_path]
            self.targets = [0]  # Default target
        else:
            # Directory of images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            for img_path in self.data_path.rglob('*'):
                if img_path.suffix.lower() in image_extensions:
                    self.image_files.append(img_path)
                    
                    # Extract target from filename or directory structure
                    if target_column:
                        # Assuming target is in filename or path
                        target = self._extract_target_from_path(img_path, target_column)
                    else:
                        target = 0  # Default target
                    
                    self.targets.append(target)
        
        self.logger.info("Image dataset loaded", 
                        samples=len(self.image_files),
                        target_column=target_column)
    
    def _extract_target_from_path(self, img_path: Path, target_column: str) -> int:
        """Extract target from image path."""
        # This is a simplified implementation
        # In practice, you might have a mapping file or specific naming convention
        return hash(str(img_path)) % 10  # Simple hash-based target
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        
        return image, target


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_loaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        
        # Create dataset
        if config.data_path.endswith(('.csv', '.xlsx', '.xls')):
            dataset = TabularDataset(
                data_path=config.data_path,
                target_column=config.target_column,
                feature_columns=config.feature_columns,
                transform=config.transform_config
            )
        else:
            dataset = ImageDataset(
                data_path=config.data_path,
                target_column=config.target_column,
                transform=config.transform_config
            )
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(config.train_split * total_size)
        val_size = int(config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        return train_loader, val_loader, test_loader


# =============================================================================
# TRAINING MODULE
# =============================================================================

class LossFunctionFactory:
    """Factory for creating loss functions."""
    
    _loss_functions = {
        "cross_entropy": nn.CrossEntropyLoss,
        "mse": nn.MSELoss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "kl_divergence": nn.KLDivLoss,
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss
    }
    
    @classmethod
    def create_loss_function(cls, loss_name: str, **kwargs) -> nn.Module:
        """Create loss function instance."""
        if loss_name not in cls._loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}")
        
        return cls._loss_functions[loss_name](**kwargs)


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    _optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad
    }
    
    @classmethod
    def create_optimizer(cls, optimizer_name: str, model: nn.Module, **kwargs) -> Optimizer:
        """Create optimizer instance."""
        if optimizer_name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return cls._optimizers[optimizer_name](model.parameters(), **kwargs)


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(scheduler_name: str, optimizer: Optimizer, **kwargs) -> _LRScheduler:
        """Create scheduler instance."""
        schedulers = {
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
            "step": torch.optim.lr_scheduler.StepLR,
            "exponential": torch.optim.lr_scheduler.ExponentialLR,
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "one_cycle": torch.optim.lr_scheduler.OneCycleLR
        }
        
        if scheduler_name not in schedulers:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return schedulers[scheduler_name](optimizer, **kwargs)


class Trainer:
    """Main training class."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _setup_experiment_tracking(self) -> Any:
        """Setup experiment tracking with WandB and TensorBoard."""
        try:
            wandb.init(project="deep-learning-project", config=self.config.to_dict())
            self.logger.info("WandB initialized successfully")
        except Exception as e:
            self.logger.warning("Failed to initialize WandB", error=str(e))
        
        # TensorBoard setup
        self.tensorboard_writer = None
        try:
            self.tensorboard_writer = SummaryWriter(f"logs/tensorboard/{self.config.checkpoint_dir}")
            self.logger.info("TensorBoard initialized successfully")
        except Exception as e:
            self.logger.warning("Failed to initialize TensorBoard", error=str(e))
    
    def train(self, model: BaseModel, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Main training loop."""
        self.logger.info("Starting training", epochs=self.config.epochs)
        
        # Setup model, optimizer, loss, and scheduler
        model = model.to(self.device)
        optimizer = OptimizerFactory.create_optimizer(
            self.config.optimizer,
            model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        loss_function = LossFunctionFactory.create_loss_function(self.config.loss_function)
        scheduler = SchedulerFactory.create_scheduler(
            self.config.scheduler,
            optimizer,
            T_max=self.config.epochs
        )
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, loss_function, scaler
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, loss_function)
            
            # Learning rate scheduling
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
            
            # Save best model
            if self.config.save_best_model and val_loss < self.best_val_loss:
                self._save_best_model(model, epoch, val_loss)
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.info("Early stopping triggered", epoch=epoch)
                break
        
        self.logger.info("Training completed")
        return self.training_history
    
    def _train_epoch(self, model: BaseModel, train_loader: DataLoader, optimizer: Optimizer, 
                    loss_function: nn.Module, scaler: Optional[torch.cuda.amp.GradScaler]) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = loss_function(output, target)
                
                scaler.scale(loss).backward()
                
                if self.config.gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
                
                optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: BaseModel, val_loader: DataLoader, 
                       loss_function: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = loss_function(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                    train_acc: float, val_acc: float):
        """Log training metrics."""
        # Update history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_acc'].append(val_acc)
        
        # Log to console
        self.logger.info("Epoch completed",
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc)
        
        # Log to WandB
        try:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
        except Exception as e:
            self.logger.warning("Failed to log to WandB", error=str(e))
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch)
                self.tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.tensorboard_writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.tensorboard_writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            except Exception as e:
                self.logger.warning("Failed to log to TensorBoard", error=str(e))
    
    def _save_best_model(self, model: BaseModel, epoch: int, val_loss: float):
        """Save the best model."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}_val_loss_{val_loss:.4f}.pt"
        model.save_model(str(checkpoint_path))
        
        self.logger.info("Best model saved", 
                        checkpoint_path=str(checkpoint_path),
                        epoch=epoch,
                        val_loss=val_loss)


# =============================================================================
# EVALUATION MODULE
# =============================================================================

class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }


class Evaluator:
    """Main evaluation class."""
    
    def __init__(self, config: EvaluationConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
        self.metrics_calculator = MetricsCalculator()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, model: BaseModel, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set."""
        self.logger.info("Starting model evaluation")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Get predictions and probabilities
                probabilities = F.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_classification_metrics(y_true, y_pred, y_prob)
        
        # Save results
        results = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'targets': y_true.tolist(),
            'probabilities': y_prob.tolist()
        }
        
        if self.config.save_predictions:
            self._save_predictions(results)
        
        if self.config.save_plots:
            self._create_evaluation_plots(y_true, y_pred, y_prob)
        
        self.logger.info("Evaluation completed", metrics=metrics)
        return results
    
    def _save_predictions(self, results: Dict[str, Any]):
        """Save predictions to file."""
        predictions_file = self.output_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2)
        
        self.logger.info("Predictions saved", file=str(predictions_file))
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        """Create evaluation plots."""
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve (for binary classification)
        if len(np.unique(y_true)) == 2:
            plt.figure(figsize=(8, 6))
            
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.metrics_calculator.calculate_classification_metrics(y_true, y_pred, y_prob).get("roc_auc", 0):.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(self.output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Metrics summary
        metrics = self.metrics_calculator.calculate_classification_metrics(y_true, y_pred, y_prob)
        plt.figure(figsize=(10, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        plt.bar(metric_names, metric_values)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Evaluation plots created", output_dir=str(self.output_dir))


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Main class to run complete experiments."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete experiment pipeline."""
        self.logger.info("Starting experiment", experiment_name=self.config.experiment_name)
        
        # Step 1: Create data loaders
        train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(self.config.data_config)
        
        # Step 2: Create model
        model = ModelFactory.create_model(self.config.model_config)
        
        # Step 3: Train model
        trainer = Trainer(self.config.training_config)
        training_history = trainer.train(model, train_loader, val_loader)
        
        # Step 4: Evaluate model
        evaluator = Evaluator(self.config.evaluation_config)
        evaluation_results = evaluator.evaluate(model, test_loader)
        
        # Step 5: Save experiment results
        experiment_results = {
            'experiment_name': self.config.experiment_name,
            'config': self.config.to_dict(),
            'training_history': training_history,
            'evaluation_results': evaluation_results
        }
        
        self._save_experiment_results(experiment_results)
        
        self.logger.info("Experiment completed successfully")
        return experiment_results
    
    def _save_experiment_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        results_dir = Path("experiment_results") / self.config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = results_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        # Save configuration
        config_file = results_dir / "config.yaml"
        self.config.save(str(config_file))
        
        self.logger.info("Experiment results saved", results_dir=str(results_dir))


def main():
    """Example usage of the modular architecture."""
    
    # Create configuration
    model_config = ModelConfig(
        model_type="transformer",
        model_name="bert-base-uncased",
        num_classes=10,
        hidden_dim=512,
        dropout=0.1
    )
    
    data_config = DataConfig(
        data_path="data/dataset.csv",
        batch_size=32,
        target_column="label"
    )
    
    training_config = TrainingConfig(
        epochs=50,
        learning_rate=0.001,
        optimizer="adam",
        scheduler="cosine"
    )
    
    evaluation_config = EvaluationConfig(
        metrics=["accuracy", "precision", "recall", "f1_score"],
        save_predictions=True,
        save_plots=True
    )
    
    experiment_config = ExperimentConfig(
        experiment_name="transformer_classification_experiment",
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        evaluation_config=evaluation_config
    )
    
    # Run experiment
    runner = ExperimentRunner(experiment_config)
    results = runner.run_experiment()
    
    print("Experiment completed!")
    print(f"Results saved in: experiment_results/{experiment_config.experiment_name}")


match __name__:
    case "__main__":
    main() 