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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import numpy as np
import json
import os
import time
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod
from gpu_optimization import GPUConfig, GPUManager, MixedPrecisionTrainer
from model_architectures import BaseModel, ModelConfig, ModelFactory
from data_pipelines import TextData, ProcessedData
from pytorch_configuration import PyTorchConfig, PyTorchManager, PyTorchTrainer, setup_pytorch_environment
from custom_models import CustomSEOModel, CustomModelConfig, create_custom_model
from autograd_utils import AutogradMonitor, AutogradProfiler, AutogradDebugger, GradientClipper
from weight_initialization import (
from loss_functions import (
from transformer_models import (
from tokenization_utils import (
from model_training_evaluation import (
from data_splitting_cross_validation import (
from early_stopping_lr_scheduling import (
        from transformers import AutoTokenizer
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Deep Learning Framework for SEO Service
Advanced model development, training, and deployment capabilities
"""


# Import our existing modules
# Import our custom modules
    WeightInitializationManager, InitializationConfig, 
    NormalizationConfig, WeightAnalysis
)
    LossFunctionManager, LossConfig, OptimizerConfig, SchedulerConfig,
    AdvancedOptimizer, AdvancedScheduler, SEOSpecificLoss, FocalLoss,
    LabelSmoothingLoss, RankingLoss, ContrastiveLoss, MultiTaskLoss
)
    TransformerManager, TransformerConfig, LLMConfig,
    SEOSpecificTransformer, MultiTaskTransformer, LLMIntegration
)
# Import tokenization utilities
    AdvancedTokenizer, SequenceHandler, TokenizedDataset, TokenizationPipeline,
    TokenizationConfig, SequenceConfig, analyze_tokenization_quality,
    optimize_tokenization_config, create_data_collator
)
# Import training and evaluation framework
    TrainingConfig as AdvancedTrainingConfig,
    ModelTrainer as AdvancedModelTrainer,
    ModelEvaluator as AdvancedModelEvaluator,
    EfficientDataLoader, TrainingMetrics
)
# Import data splitting and cross-validation framework
    DataSplitConfig, DataSplitManager, DataSplit, CrossValidationSplit,
    SEOSpecificSplitter
)
# Import early stopping and learning rate scheduling framework
    EarlyStoppingConfig, LRSchedulerConfig, TrainingMetrics as EarlyStoppingMetrics,
    EarlyStopping, AdvancedLRScheduler, TrainingOptimizer
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for deep learning training"""
    # Model configuration
    model_type: str = "transformer"
    model_name: str = "bert-base-uncased"
    num_classes: int = 2
    max_length: int = 512
    dropout_rate: float = 0.1
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimization configuration
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine_with_warmup"
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    
    # PyTorch configuration
    pytorch_config: PyTorchConfig = None
    
    # Data configuration
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    use_stratified_split: bool = True
    
    # Monitoring configuration
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    early_stopping_patience: int = 5
    
    def __post_init__(self) -> Any:
        """Initialize PyTorch config if not provided"""
        if self.pytorch_config is None:
            self.pytorch_config = PyTorchConfig(
                device="auto",
                enable_amp=self.use_mixed_precision,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                max_grad_norm=self.max_grad_norm
            )

@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    step: int = 0
    timestamp: float = field(default_factory=time.time)

class DeepLearningDataset(Dataset):
    """Custom dataset for deep learning training"""
    
    def __init__(self, data: List[ProcessedData], tokenizer=None):
        
    """__init__ function."""
self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self) -> Any:
        return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        item = self.data[idx]
        result = {
            'input_ids': item.input_ids,
            'attention_mask': item.attention_mask
        }
        if item.labels is not None:
            result['labels'] = item.labels
        return result

class ModelTrainer(ABC):
    """Abstract base class for model training"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.pytorch_manager = setup_pytorch_environment(config.pytorch_config)
        self.device = self.pytorch_manager.device
        self.pytorch_trainer = PyTorchTrainer(self.pytorch_manager)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = self.pytorch_manager.scaler
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Model trainer initialized on device: {self.device}")
        logger.info(f"PyTorch configuration: {self.pytorch_manager.get_device_info()}")
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the neural network model"""
        pass
    
    @abstractmethod
    def create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer"""
        pass
    
    @abstractmethod
    def create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create the learning rate scheduler"""
        pass
    
    def setup_training(self) -> Any:
        """Setup model, optimizer, and scheduler with PyTorch optimizations and weight initialization"""
        self.model = self.create_model()
        
        # Initialize weights with advanced strategies
        init_config = InitializationConfig(
            method=self.config.pytorch_config.initialization_method if hasattr(self.config.pytorch_config, 'initialization_method') else "xavier_uniform",
            gain=1.0,
            fan_mode="fan_avg",
            nonlinearity="relu"
        )
        self.weight_manager.initialize_model(self.model, init_config)
        
        # Apply normalization if specified
        if hasattr(self.config.pytorch_config, 'normalization_method'):
            norm_config = NormalizationConfig(
                method=self.config.pytorch_config.normalization_method,
                eps=1e-5,
                affine=True
            )
            self.model = self.weight_manager.apply_normalization(self.model, norm_config)
        
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Apply PyTorch optimizations to model
        self.model = self.pytorch_manager.optimize_model(self.model)
        
        # Analyze weights after initialization
        weight_summary = self.model.get_weight_summary() if hasattr(self.model, 'get_weight_summary') else None
        if weight_summary:
            logger.info(f"Weight analysis: {weight_summary['health']['is_healthy']}")
            if not weight_summary['health']['is_healthy']:
                logger.warning(f"Weight health issues: {weight_summary['health']['issues']}")
        
        logger.info(f"Training setup completed. Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Model optimized for device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch using PyTorch optimizations"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step with PyTorch optimizations
            step_result = self.pytorch_trainer.train_step(
                self.model,
                self.optimizer,
                batch,
                F.cross_entropy,
                self.config.gradient_accumulation_steps
            )
            
            total_loss += step_result['loss']
            
            # Calculate accuracy
            if 'labels' in batch:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device)
                    )
                    predictions = torch.argmax(outputs, dim=1)
                    labels = batch['labels'].to(self.device)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {step_result['loss']:.4f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch using PyTorch optimizations"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Validation step with PyTorch optimizations
                step_result = self.pytorch_trainer.validation_step(
                    self.model,
                    batch,
                    F.cross_entropy
                )
                
                total_loss += step_result['loss']
                
                # Calculate accuracy
                if 'labels' in batch:
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device)
                    )
                    predictions = torch.argmax(outputs, dim=1)
                    labels = batch['labels'].to(self.device)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Complete training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.learning_rates.append(self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} completed in {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Memory cleanup
            self.pytorch_manager.clear_memory()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epoch': len(self.train_losses)
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded: {filename}")

class SEOModelTrainer(ModelTrainer):
    """Specialized trainer for SEO models"""
    
    def create_model(self) -> nn.Module:
        """Create SEO model using factory pattern"""
        model_config = ModelConfig(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            max_length=self.config.max_length,
            dropout_rate=self.config.dropout_rate
        )
        
        model = ModelFactory.create_model(self.config.model_type, model_config)
        return model

class CustomSEOModelTrainer(ModelTrainer):
    """Advanced trainer for custom SEO models with autograd monitoring"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Initialize autograd monitoring tools
        self.autograd_monitor = AutogradMonitor()
        self.autograd_profiler = AutogradProfiler()
        self.autograd_debugger = AutogradDebugger()
        self.gradient_clipper = GradientClipper()
        
        # Initialize weight management tools
        self.weight_manager = WeightInitializationManager()
        
        # Initialize loss function and optimization management
        self.loss_manager = LossFunctionManager()
        
        logger.info("Custom SEO model trainer initialized with autograd monitoring, weight management, and loss optimization")
    
    def create_model(self) -> nn.Module:
        """Create custom SEO model with advanced architecture"""
        custom_config = CustomModelConfig(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            hidden_size=768,
            num_layers=6,
            num_heads=12,
            dropout_rate=self.config.dropout_rate,
            max_length=self.config.max_length,
            use_layer_norm=True,
            use_residual_connections=True,
            activation_function="gelu",
            initialization_method="xavier",
            gradient_checkpointing=self.config.use_gradient_checkpointing
        )
        
        model = create_custom_model(custom_config)
        logger.info(f"Created custom SEO model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def create_loss_function(self) -> nn.Module:
        """Create loss function with advanced configuration"""
        # Create loss configuration
        loss_config = LossConfig(
            loss_type="seo_specific",  # Use SEO-specific loss by default
            alpha=1.0,
            gamma=2.0,
            smoothing=0.1,
            margin=1.0,
            temperature=0.1
        )
        
        # Create loss function using loss manager
        loss_function = self.loss_manager.create_loss_function(loss_config)
        
        return loss_function
    
    def train_epoch_with_monitoring(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with comprehensive autograd monitoring and advanced loss functions"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Create loss function
        loss_function = self.create_loss_function()
        
        for batch_idx, batch in enumerate(train_loader):
            # Profile autograd operations
            with self.autograd_profiler.profile_autograd(f"training_step_{batch_idx}"):
                # Training step with PyTorch optimizations
                step_result = self.pytorch_trainer.train_step(
                    self.model,
                    self.optimizer,
                    batch,
                    loss_function,
                    self.config.gradient_accumulation_steps
                )
            
            total_loss += step_result['loss']
            
            # Monitor gradients
            gradient_info = self.autograd_monitor.monitor_gradients(self.model)
            
            # Check for gradient issues
            gradient_issues = self.autograd_debugger.check_gradients(self.model)
            if gradient_issues['has_issues']:
                logger.warning(f"Gradient issues detected: {gradient_issues['issues']}")
            
            # Apply gradient clipping if needed
            if gradient_info['total_norm'] > 1.0:
                self.gradient_clipper.clip_grad_norm_(self.model, max_norm=1.0)
                logger.info(f"Applied gradient clipping. Norm: {gradient_info['total_norm']:.4f}")
            
            # Calculate accuracy
            if 'labels' in batch:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device)
                    )
                    predictions = torch.argmax(outputs, dim=1)
                    labels = batch['labels'].to(self.device)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
            
            # Logging with autograd information
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {step_result['loss']:.4f}, "
                           f"LR: {current_lr:.2e}, "
                           f"Grad Norm: {gradient_info['total_norm']:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss, 
            'accuracy': accuracy,
            'gradient_stats': self.autograd_monitor.get_gradient_statistics()
        }
    
    def get_autograd_summary(self) -> Dict[str, Any]:
        """Get comprehensive autograd summary"""
        return {
            'gradient_statistics': self.autograd_monitor.get_gradient_statistics(),
            'profiling_summary': self.autograd_profiler.get_profile_summary(),
            'loss_summary': self.loss_manager.get_loss_summary(),
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(self.device)
            }
        }
    
    def create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with advanced configuration using loss function manager"""
        # Get model parameters
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # Create optimizer configuration
        optimizer_config = OptimizerConfig(
            optimizer_type=self.config.optimizer_type,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8
        )
        
        # Create optimizer using advanced optimizer
        optimizer = AdvancedOptimizer.create_optimizer(self.model, optimizer_config)
        
        return optimizer
    
    def create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with advanced configuration"""
        # Create scheduler configuration
        scheduler_config = SchedulerConfig(
            scheduler_type=self.config.scheduler_type,
            T_max=self.config.num_epochs,
            min_lr=1e-6,
            gamma=0.1,
            step_size=30
        )
        
        # Create scheduler using advanced scheduler
        scheduler = AdvancedScheduler.create_scheduler(self.optimizer, scheduler_config)
        
        return scheduler

class ModelEvaluator:
    """Model evaluation and testing utilities"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        
    """__init__ function."""
self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_accuracy(self, test_loader: DataLoader) -> float:
        """Evaluate model accuracy"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                predictions = torch.argmax(outputs, dim=1)
                labels = batch['labels'].to(self.device)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def predict_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Make predictions on a batch"""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
            )
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions, probabilities
    
    def get_model_predictions(self, test_loader: DataLoader) -> Tuple[List[int], List[int]]:
        """Get model predictions and true labels"""
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                batch_predictions = torch.argmax(outputs, dim=1)
                predictions.extend(batch_predictions.cpu().numpy())
                
                if 'labels' in batch:
                    true_labels.extend(batch['labels'].numpy())
        
        return predictions, true_labels

class ModelDeployment:
    """Model deployment and serving utilities"""
    
    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        
    """__init__ function."""
self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def preprocess_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Preprocess text for inference"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict on a single text"""
        encoding = self.preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            )
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        
        return {
            'text': text,
            'prediction': prediction.item(),
            'confidence': probabilities.max().item(),
            'probabilities': probabilities.cpu().numpy().tolist()
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict on a batch of texts"""
        results = []
        
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def save_model_for_serving(self, save_path: str):
        """Save model for production serving"""
        # Save model
        torch.save(self.model.state_dict(), f"{save_path}/model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model.__class__.__name__,
            'device': str(self.device),
            'num_classes': self.model.config.num_labels if hasattr(self.model, 'config') else None
        }
        
        with open(f"{save_path}/metadata.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved for serving at: {save_path}")

class DeepLearningFramework:
    """Main deep learning framework orchestrator"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.trainer = SEOModelTrainer(config)
        self.evaluator = None
        self.deployment = None
        
        # Initialize transformer and LLM managers
        self.transformer_manager = TransformerManager()
        self.transformers = {}
        self.llm_models = {}
    
    async def train_model(
        self,
        train_data: List[ProcessedData],
        val_data: List[ProcessedData],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete model training pipeline"""
        
        # Setup training
        self.trainer.setup_training()
        
        # Create dataloaders
        train_dataset = DeepLearningDataset(train_data)
        val_dataset = DeepLearningDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Train model
        training_results = self.trainer.train(train_loader, val_loader)
        
        # Save model if path provided
        if save_path:
            self.trainer.save_checkpoint(f"{save_path}/final_model.pt")
        
        # Create evaluator
        self.evaluator = ModelEvaluator(self.trainer.model, self.trainer.device)
        
        return training_results
    
    def evaluate_model(self, test_data: List[ProcessedData]) -> Dict[str, float]:
        """Evaluate trained model"""
        if self.evaluator is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        test_dataset = DeepLearningDataset(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        accuracy = self.evaluator.evaluate_accuracy(test_loader)
        predictions, true_labels = self.evaluator.get_model_predictions(test_loader)
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def deploy_model(self, tokenizer, save_path: str):
        """Deploy model for production"""
        if self.trainer.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        self.deployment = ModelDeployment(
            self.trainer.model,
            tokenizer,
            self.trainer.device
        )
        
        self.deployment.save_model_for_serving(save_path)
        
        return self.deployment
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'config': self.config,
            'training_metrics': {
                'train_losses': self.trainer.train_losses,
                'val_losses': self.trainer.val_losses,
                'train_accuracies': self.trainer.train_accuracies,
                'val_accuracies': self.trainer.val_accuracies,
                'learning_rates': self.trainer.learning_rates
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.trainer.model.parameters()) if self.trainer.model else 0,
                'trainable_parameters': sum(p.numel() for p in self.trainer.model.parameters() if p.requires_grad) if self.trainer.model else 0,
                'device': str(self.trainer.device)
            },
            'gpu_info': self.trainer.pytorch_manager.get_memory_info(),
            'transformer_models': list(self.transformers.keys()),
            'llm_models': list(self.llm_models.keys())
        }
    
    def create_transformer(self, config: TransformerConfig, model_name: str) -> SEOSpecificTransformer:
        """Create a new SEO-specific transformer model"""
        transformer = self.transformer_manager.create_transformer(config, model_name)
        self.transformers[model_name] = transformer
        logger.info(f"Created transformer model: {model_name}")
        return transformer
    
    def create_multi_task_transformer(self, config: TransformerConfig, task_configs: Dict[str, Dict[str, Any]], 
                                    model_name: str) -> MultiTaskTransformer:
        """Create a new multi-task transformer model"""
        transformer = self.transformer_manager.create_multi_task_transformer(config, task_configs, model_name)
        self.transformers[model_name] = transformer
        logger.info(f"Created multi-task transformer model: {model_name}")
        return transformer
    
    def load_pretrained_transformer(self, model_name: str, pretrained_name: str) -> nn.Module:
        """Load a pretrained transformer model"""
        transformer = self.transformer_manager.load_pretrained_transformer(model_name, pretrained_name)
        self.transformers[model_name] = transformer
        logger.info(f"Loaded pretrained transformer: {model_name} from {pretrained_name}")
        return transformer
    
    def create_llm_integration(self, config: LLMConfig, model_name: str) -> LLMIntegration:
        """Create LLM integration for text generation and analysis"""
        llm = self.transformer_manager.create_llm_integration(config, model_name)
        self.llm_models[model_name] = llm
        logger.info(f"Created LLM integration: {model_name}")
        return llm
    
    def get_transformer(self, model_name: str) -> Optional[nn.Module]:
        """Get a transformer model by name"""
        return self.transformer_manager.get_model(model_name)
    
    def get_llm(self, model_name: str) -> Optional[LLMIntegration]:
        """Get an LLM model by name"""
        return self.transformer_manager.get_llm(model_name)
    
    def save_transformer(self, model_name: str, save_path: str):
        """Save a transformer model"""
        if model_name not in self.transformers:
            raise ValueError(f"Transformer model '{model_name}' not found")
        
        self.transformer_manager.save_model(model_name, save_path)
        logger.info(f"Saved transformer model: {model_name} to {save_path}")
    
    def load_transformer(self, model_name: str, load_path: str) -> nn.Module:
        """Load a saved transformer model"""
        transformer = self.transformer_manager.load_model(model_name, load_path)
        self.transformers[model_name] = transformer
        logger.info(f"Loaded transformer model: {model_name} from {load_path}")
        return transformer
    
    def generate_text_with_llm(self, model_name: str, prompt: str, max_length: Optional[int] = None) -> str:
        """Generate text using an LLM"""
        if model_name not in self.llm_models:
            raise ValueError(f"LLM model '{model_name}' not found")
        
        llm = self.llm_models[model_name]
        return llm.generate_text(prompt, max_length)
    
    def analyze_seo_content_with_llm(self, model_name: str, content: str) -> Dict[str, Any]:
        """Analyze SEO content using an LLM"""
        if model_name not in self.llm_models:
            raise ValueError(f"LLM model '{model_name}' not found")
        
        llm = self.llm_models[model_name]
        return llm.analyze_seo_content(content)
    
    def get_embeddings_with_llm(self, model_name: str, text: str) -> torch.Tensor:
        """Get embeddings for text using an LLM"""
        if model_name not in self.llm_models:
            raise ValueError(f"LLM model '{model_name}' not found")
        
        llm = self.llm_models[model_name]
        return llm.get_embeddings(text)
    
    # Tokenization utility methods
    def create_advanced_tokenizer(self, config: TokenizationConfig) -> AdvancedTokenizer:
        """Create an advanced tokenizer with caching and optimization"""
        tokenizer = AdvancedTokenizer(config)
        logger.info(f"Created advanced tokenizer: {config.model_name}")
        return tokenizer
    
    def create_sequence_handler(self, config: SequenceConfig) -> SequenceHandler:
        """Create a sequence handler for long text processing"""
        handler = SequenceHandler(config)
        logger.info(f"Created sequence handler with strategy: {config.chunk_strategy}")
        return handler
    
    def create_tokenization_pipeline(self, tokenizer_config: TokenizationConfig, 
                                   sequence_config: SequenceConfig = None) -> TokenizationPipeline:
        """Create a complete tokenization pipeline"""
        pipeline = TokenizationPipeline(tokenizer_config, sequence_config)
        logger.info("Created tokenization pipeline")
        return pipeline
    
    def create_tokenized_dataset(self, texts: List[str], labels: Optional[List[int]] = None,
                                tokenizer: AdvancedTokenizer = None, max_length: int = 512,
                                cache_dir: Optional[str] = None) -> TokenizedDataset:
        """Create a tokenized dataset with caching"""
        if tokenizer is None:
            # Create default tokenizer
            config = TokenizationConfig(
                model_name=self.config.model_name,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            tokenizer = AdvancedTokenizer(config)
        
        dataset = TokenizedDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
            cache_dir=cache_dir
        )
        
        logger.info(f"Created tokenized dataset with {len(texts)} samples")
        return dataset
    
    def analyze_tokenization_quality(self, texts: List[str], tokenizer_name: str = None) -> Dict[str, Any]:
        """Analyze tokenization quality for a set of texts"""
        if tokenizer_name is None:
            tokenizer_name = self.config.model_name
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        analysis = analyze_tokenization_quality(tokenizer, texts)
        
        logger.info(f"Tokenization quality analysis completed for {len(texts)} texts")
        logger.info(f"Average tokens per text: {analysis['avg_tokens_per_text']:.2f}")
        logger.info(f"Vocabulary coverage ratio: {analysis['vocabulary_coverage_ratio']:.4f}")
        
        return analysis
    
    def optimize_tokenization_config(self, texts: List[str], 
                                   base_config: TokenizationConfig = None) -> TokenizationConfig:
        """Optimize tokenization configuration based on data analysis"""
        if base_config is None:
            base_config = TokenizationConfig(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        
        optimized_config = optimize_tokenization_config(texts, base_config)
        
        logger.info(f"Optimized tokenization config:")
        logger.info(f"  Model: {optimized_config.model_name}")
        logger.info(f"  Max length: {optimized_config.max_length}")
        logger.info(f"  Truncation: {optimized_config.truncation}")
        
        return optimized_config
    
    def process_text_with_advanced_tokenization(self, text: str, 
                                              tokenizer_config: TokenizationConfig = None,
                                              sequence_config: SequenceConfig = None) -> Dict[str, Any]:
        """Process text using advanced tokenization pipeline"""
        if tokenizer_config is None:
            tokenizer_config = TokenizationConfig(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        
        pipeline = self.create_tokenization_pipeline(tokenizer_config, sequence_config)
        result = pipeline.process_text(text)
        
        # Get statistics
        stats = pipeline.get_statistics()
        
        return {
            'tokenization_result': result,
            'statistics': stats,
            'text_length': len(text),
            'token_count': result['input_ids'].shape[-1] if 'input_ids' in result else 0
        }
    
    def process_batch_with_advanced_tokenization(self, texts: List[str],
                                               tokenizer_config: TokenizationConfig = None,
                                               sequence_config: SequenceConfig = None) -> Dict[str, Any]:
        """Process batch of texts using advanced tokenization pipeline"""
        if tokenizer_config is None:
            tokenizer_config = TokenizationConfig(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        
        pipeline = self.create_tokenization_pipeline(tokenizer_config, sequence_config)
        batch_result = pipeline.process_batch(texts)
        
        # Get statistics
        stats = pipeline.get_statistics()
        
        return {
            'batch_result': batch_result,
            'statistics': stats,
            'batch_size': len(texts),
            'total_tokens': batch_result['input_ids'].shape[0] * batch_result['input_ids'].shape[1] if 'input_ids' in batch_result else 0
        }
    
    def create_data_loader_with_tokenization(self, texts: List[str], labels: Optional[List[int]] = None,
                                           batch_size: int = None, shuffle: bool = True,
                                           tokenizer_config: TokenizationConfig = None,
                                           cache_dir: Optional[str] = None) -> DataLoader:
        """Create a data loader with integrated tokenization"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if tokenizer_config is None:
            tokenizer_config = TokenizationConfig(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        
        # Create tokenized dataset
        dataset = self.create_tokenized_dataset(
            texts=texts,
            labels=labels,
            tokenizer=AdvancedTokenizer(tokenizer_config),
            max_length=tokenizer_config.max_length,
            cache_dir=cache_dir
        )
        
        # Create data collator
        data_collator = create_data_collator(
            dataset.tokenizer.tokenizer if dataset.tokenizer else None,
            "sequence_classification"
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Created data loader with {len(dataset)} samples, batch size: {batch_size}")
        return dataloader
    
    def get_tokenization_statistics(self, tokenizer: AdvancedTokenizer = None) -> Dict[str, Any]:
        """Get comprehensive tokenization statistics"""
        if tokenizer is None:
            # Create default tokenizer
            config = TokenizationConfig(
                model_name=self.config.model_name,
                max_length=self.config.max_length
            )
            tokenizer = AdvancedTokenizer(config)
        
        stats = tokenizer.get_stats()
        
        return {
            'total_tokens': stats.total_tokens,
            'unique_tokens': stats.unique_tokens,
            'avg_sequence_length': stats.avg_sequence_length,
            'max_sequence_length': stats.max_sequence_length,
            'min_sequence_length': stats.min_sequence_length,
            'vocabulary_size': stats.vocabulary_size,
            'oov_rate': stats.oov_rate,
            'padding_ratio': stats.padding_ratio,
            'truncation_ratio': stats.truncation_ratio,
            'token_distribution': dict(list(stats.token_distribution.items())[:10]),  # Top 10
            'sequence_length_distribution': dict(list(stats.sequence_length_distribution.items())[:10]),  # Top 10
            'cache_size': len(tokenizer.cache)
        }
    
    # Advanced Training and Evaluation Integration Methods
    
    def create_advanced_training_config(self, 
                                      model: nn.Module,
                                      train_dataset: Dataset,
                                      val_dataset: Dataset = None,
                                      test_dataset: Dataset = None,
                                      **kwargs) -> AdvancedTrainingConfig:
        """Create advanced training configuration"""
        config = AdvancedTrainingConfig(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=kwargs.get('batch_size', self.config.batch_size),
            learning_rate=kwargs.get('learning_rate', self.config.learning_rate),
            epochs=kwargs.get('epochs', self.config.num_epochs),
            optimizer=kwargs.get('optimizer', 'adamw'),
            scheduler=kwargs.get('scheduler', 'cosine'),
            use_mixed_precision=kwargs.get('use_mixed_precision', self.config.use_mixed_precision),
            device=kwargs.get('device', str(self.pytorch_manager.device)),
            **kwargs
        )
        
        logger.info(f"Created advanced training config: {config}")
        return config
    
    def create_advanced_trainer(self, config: AdvancedTrainingConfig) -> AdvancedModelTrainer:
        """Create advanced model trainer"""
        trainer = AdvancedModelTrainer(config)
        logger.info(f"Created advanced model trainer with {sum(p.numel() for p in config.model.parameters()):,} parameters")
        return trainer
    
    def create_advanced_evaluator(self, model: nn.Module) -> AdvancedModelEvaluator:
        """Create advanced model evaluator"""
        evaluator = AdvancedModelEvaluator(model, device=str(self.pytorch_manager.device))
        logger.info(f"Created advanced model evaluator")
        return evaluator
    
    def train_with_advanced_framework(self, 
                                    model: nn.Module,
                                    train_dataset: Dataset,
                                    val_dataset: Dataset = None,
                                    test_dataset: Dataset = None,
                                    **kwargs) -> TrainingMetrics:
        """Train model using advanced training framework"""
        # Create advanced training config
        config = self.create_advanced_training_config(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **kwargs
        )
        
        # Create trainer
        trainer = self.create_advanced_trainer(config)
        
        # Train model
        logger.info("Starting advanced training...")
        metrics = trainer.train()
        
        logger.info(f"Advanced training completed!")
        logger.info(f"Best validation loss: {metrics.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {metrics.best_val_accuracy:.4f}")
        
        return metrics
    
    def evaluate_with_advanced_framework(self, 
                                       model: nn.Module,
                                       test_dataset: Dataset,
                                       task_type: str = "classification") -> Dict[str, float]:
        """Evaluate model using advanced evaluation framework"""
        # Create evaluator
        evaluator = self.create_advanced_evaluator(model)
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate
        logger.info(f"Starting advanced evaluation for {task_type} task...")
        metrics = evaluator.evaluate(test_loader, task_type=task_type)
        
        logger.info(f"Advanced evaluation completed!")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def create_efficient_data_loader(self, 
                                   train_dataset: Dataset,
                                   val_dataset: Dataset = None,
                                   test_dataset: Dataset = None,
                                   **kwargs) -> EfficientDataLoader:
        """Create efficient data loader"""
        config = AdvancedTrainingConfig(
            model=None,  # Not needed for data loader
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=kwargs.get('batch_size', self.config.batch_size),
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            **kwargs
        )
        
        data_loader = EfficientDataLoader(config)
        logger.info(f"Created efficient data loader")
        return data_loader
    
    def run_comprehensive_training(self, 
                                 model: nn.Module,
                                 train_dataset: Dataset,
                                 val_dataset: Dataset,
                                 test_dataset: Dataset = None,
                                 **kwargs) -> Dict[str, Any]:
        """Run comprehensive training with advanced framework"""
        logger.info("Starting comprehensive training pipeline...")
        
        # Train with advanced framework
        training_metrics = self.train_with_advanced_framework(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **kwargs
        )
        
        # Evaluate if test dataset provided
        evaluation_metrics = None
        if test_dataset is not None:
            evaluation_metrics = self.evaluate_with_advanced_framework(
                model=model,
                test_dataset=test_dataset,
                task_type=kwargs.get('task_type', 'classification')
            )
        
        # Create comprehensive results
        results = {
            'training_metrics': training_metrics,
            'evaluation_metrics': evaluation_metrics,
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            },
            'training_config': kwargs
        }
        
        logger.info("Comprehensive training pipeline completed!")
        return results
    
    def compare_models_with_advanced_framework(self, 
                                             models: Dict[str, nn.Module],
                                             train_dataset: Dataset,
                                             val_dataset: Dataset,
                                             test_dataset: Dataset = None,
                                             **kwargs) -> Dict[str, Any]:
        """Compare multiple models using advanced framework"""
        logger.info(f"Starting model comparison for {len(models)} models...")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training and evaluating {model_name}...")
            
            try:
                results = self.run_comprehensive_training(
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    **kwargs
                )
                
                comparison_results[model_name] = results
                
                logger.info(f"{model_name} - Best val loss: {results['training_metrics'].best_val_loss:.4f}")
                if results['evaluation_metrics']:
                    if 'accuracy' in results['evaluation_metrics']:
                        logger.info(f"{model_name} - Test accuracy: {results['evaluation_metrics']['accuracy']:.4f}")
                    elif 'mse' in results['evaluation_metrics']:
                        logger.info(f"{model_name} - Test MSE: {results['evaluation_metrics']['mse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
        
        # Create comparison summary
        summary = {
            'model_comparison': comparison_results,
            'best_model': None,
            'best_metric': float('inf')
        }
        
        # Find best model
        for model_name, results in comparison_results.items():
            if 'error' not in results and results['training_metrics']:
                val_loss = results['training_metrics'].best_val_loss
                if val_loss < summary['best_metric']:
                    summary['best_metric'] = val_loss
                    summary['best_model'] = model_name
        
        logger.info(f"Model comparison completed! Best model: {summary['best_model']}")
        return summary
    
    def hyperparameter_tuning_with_advanced_framework(self, 
                                                    model_factory: Callable,
                                                    train_dataset: Dataset,
                                                    val_dataset: Dataset,
                                                    test_dataset: Dataset = None,
                                                    hyperparameter_configs: List[Dict] = None,
                                                    **kwargs) -> Dict[str, Any]:
        """Perform hyperparameter tuning using advanced framework"""
        if hyperparameter_configs is None:
            # Default hyperparameter configurations
            hyperparameter_configs = [
                {'learning_rate': 1e-3, 'batch_size': 32, 'optimizer': 'adamw'},
                {'learning_rate': 5e-4, 'batch_size': 64, 'optimizer': 'adam'},
                {'learning_rate': 1e-4, 'batch_size': 16, 'optimizer': 'sgd'},
            ]
        
        logger.info(f"Starting hyperparameter tuning with {len(hyperparameter_configs)} configurations...")
        
        tuning_results = []
        
        for i, hp_config in enumerate(hyperparameter_configs):
            logger.info(f"Testing configuration {i+1}: {hp_config}")
            
            try:
                # Create model with current configuration
                model = model_factory(**hp_config)
                
                # Train and evaluate
                results = self.run_comprehensive_training(
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    **hp_config,
                    **kwargs
                )
                
                tuning_results.append({
                    'config_id': i + 1,
                    'hyperparameters': hp_config,
                    'results': results
                })
                
                logger.info(f"Config {i+1} - Best val loss: {results['training_metrics'].best_val_loss:.4f}")
                
            except Exception as e:
                logger.error(f"Error with configuration {i+1}: {e}")
                tuning_results.append({
                    'config_id': i + 1,
                    'hyperparameters': hp_config,
                    'error': str(e)
                })
        
        # Find best configuration
        best_config = None
        best_metric = float('inf')
        
        for result in tuning_results:
            if 'error' not in result and result['results']['training_metrics']:
                val_loss = result['results']['training_metrics'].best_val_loss
                if val_loss < best_metric:
                    best_metric = val_loss
                    best_config = result
        
        summary = {
            'tuning_results': tuning_results,
            'best_config': best_config,
            'best_metric': best_metric,
            'total_configurations': len(hyperparameter_configs)
        }
        
        logger.info(f"Hyperparameter tuning completed! Best config: {best_config['config_id'] if best_config else None}")
        return summary
    
    # Data Splitting and Cross-Validation Integration Methods
    
    def create_data_split_config(self, 
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                use_cross_validation: bool = False,
                                cv_folds: int = 5,
                                cv_strategy: str = "stratified",
                                stratify_by: Optional[str] = None,
                                group_by: Optional[str] = None,
                                **kwargs) -> DataSplitConfig:
        """Create data splitting configuration"""
        config = DataSplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            use_cross_validation=use_cross_validation,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            stratify_by=stratify_by,
            group_by=group_by,
            random_state=kwargs.get('random_state', 42),
            **kwargs
        )
        
        logger.info(f"Created data split config: {config}")
        return config
    
    def create_data_split_manager(self, config: DataSplitConfig) -> DataSplitManager:
        """Create data split manager"""
        split_manager = DataSplitManager(config)
        logger.info(f"Created data split manager with strategy: {config.cv_strategy if config.use_cross_validation else 'standard'}")
        return split_manager
    
    def split_dataset_with_config(self, 
                                 dataset: Dataset,
                                 config: DataSplitConfig,
                                 labels: Optional[List] = None,
                                 groups: Optional[List] = None) -> Union[DataSplit, CrossValidationSplit]:
        """Split dataset using specified configuration"""
        split_manager = self.create_data_split_manager(config)
        
        # Extract labels if not provided
        if labels is None and config.stratify_by:
            labels = self._extract_labels_from_dataset(dataset, config.stratify_by)
        
        # Extract groups if not provided
        if groups is None and config.group_by:
            groups = self._extract_groups_from_dataset(dataset, config.group_by)
        
        # Create splits
        splits = split_manager.create_splits(dataset, labels=labels, groups=groups)
        
        logger.info(f"Dataset split successfully")
        if isinstance(splits, DataSplit):
            logger.info(f"  Train: {len(splits.train_indices)}, Val: {len(splits.val_indices)}, Test: {len(splits.test_indices)}")
        else:
            logger.info(f"  Cross-validation with {len(splits.fold_splits)} folds")
        
        return splits
    
    def create_seo_specific_splits(self, 
                                 dataset: Dataset,
                                 split_strategy: str = "domain",
                                 **kwargs) -> DataSplit:
        """Create SEO-specific data splits"""
        config = self.create_data_split_config(**kwargs)
        split_manager = self.create_data_split_manager(config)
        
        splits = split_manager.create_seo_splits(dataset, split_strategy=split_strategy)
        
        logger.info(f"Created SEO-specific splits using {split_strategy} strategy")
        logger.info(f"  Train: {len(splits.train_indices)}, Val: {len(splits.val_indices)}, Test: {len(splits.test_indices)}")
        
        return splits
    
    def create_datasets_from_splits(self, 
                                  dataset: Dataset,
                                  splits: Union[DataSplit, CrossValidationSplit]) -> Union[Tuple[Dataset, Dataset, Dataset], List[Tuple[Dataset, Dataset, Dataset]]]:
        """Create PyTorch datasets from splits"""
        config = self.create_data_split_config()
        split_manager = self.create_data_split_manager(config)
        
        datasets = split_manager.create_datasets(dataset, splits)
        
        if isinstance(datasets, tuple):
            logger.info(f"Created datasets: Train({len(datasets[0])}), Val({len(datasets[1])}), Test({len(datasets[2])})")
        else:
            logger.info(f"Created {len(datasets)} fold datasets")
        
        return datasets
    
    def analyze_data_splits(self, 
                           splits: Union[DataSplit, CrossValidationSplit],
                           labels: Optional[List] = None) -> Dict[str, Any]:
        """Analyze the quality of data splits"""
        config = self.create_data_split_config()
        split_manager = self.create_data_split_manager(config)
        
        analysis = split_manager.analyze_splits(splits, labels=labels)
        
        logger.info("Data split analysis completed")
        if 'stratification_quality' in analysis:
            logger.info(f"  Stratification quality: {analysis['stratification_quality']['overall_quality']:.4f}")
        
        return analysis
    
    def train_with_proper_splits(self, 
                                model: nn.Module,
                                dataset: Dataset,
                                split_config: DataSplitConfig,
                                labels: Optional[List] = None,
                                groups: Optional[List] = None,
                                **kwargs) -> Dict[str, Any]:
        """Train model with proper data splitting"""
        logger.info("Starting training with proper data splits...")
        
        # Create splits
        splits = self.split_dataset_with_config(dataset, split_config, labels, groups)
        
        # Create datasets
        datasets = self.create_datasets_from_splits(dataset, splits)
        
        # Analyze splits
        analysis = self.analyze_data_splits(splits, labels)
        
        if isinstance(datasets, tuple):
            # Standard split
            train_dataset, val_dataset, test_dataset = datasets
            
            # Train model
            results = self.run_comprehensive_training(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                **kwargs
            )
            
            results['split_analysis'] = analysis
            results['split_type'] = 'standard'
            
        else:
            # Cross-validation
            fold_results = []
            
            for fold_idx, (train_dataset, val_dataset, test_dataset) in enumerate(datasets):
                logger.info(f"Training fold {fold_idx + 1}/{len(datasets)}...")
                
                # Create new model for each fold
                fold_model = type(model)()  # Create new instance
                
                # Train on this fold
                fold_result = self.run_comprehensive_training(
                    model=fold_model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    **kwargs
                )
                
                fold_results.append({
                    'fold': fold_idx + 1,
                    'results': fold_result
                })
            
            # Aggregate results
            test_metrics = [result['results']['evaluation_metrics'] for result in fold_results if result['results']['evaluation_metrics']]
            
            if test_metrics:
                if 'accuracy' in test_metrics[0]:
                    accuracies = [metrics['accuracy'] for metrics in test_metrics]
                    mean_accuracy = np.mean(accuracies)
                    std_accuracy = np.std(accuracies)
                    
                    results = {
                        'fold_results': fold_results,
                        'aggregate_metrics': {
                            'mean_accuracy': mean_accuracy,
                            'std_accuracy': std_accuracy,
                            'min_accuracy': min(accuracies),
                            'max_accuracy': max(accuracies)
                        },
                        'split_analysis': analysis,
                        'split_type': 'cross_validation'
                    }
                else:
                    # Handle regression metrics
                    mse_values = [metrics['mse'] for metrics in test_metrics]
                    mean_mse = np.mean(mse_values)
                    std_mse = np.std(mse_values)
                    
                    results = {
                        'fold_results': fold_results,
                        'aggregate_metrics': {
                            'mean_mse': mean_mse,
                            'std_mse': std_mse,
                            'min_mse': min(mse_values),
                            'max_mse': max(mse_values)
                        },
                        'split_analysis': analysis,
                        'split_type': 'cross_validation'
                    }
            else:
                results = {
                    'fold_results': fold_results,
                    'split_analysis': analysis,
                    'split_type': 'cross_validation'
                }
        
        logger.info("Training with proper splits completed!")
        return results
    
    def compare_models_with_proper_splits(self, 
                                        models: Dict[str, nn.Module],
                                        dataset: Dataset,
                                        split_config: DataSplitConfig,
                                        labels: Optional[List] = None,
                                        groups: Optional[List] = None,
                                        **kwargs) -> Dict[str, Any]:
        """Compare multiple models with proper data splitting"""
        logger.info(f"Starting model comparison with proper splits for {len(models)} models...")
        
        # Create splits once for all models
        splits = self.split_dataset_with_config(dataset, split_config, labels, groups)
        datasets = self.create_datasets_from_splits(dataset, splits)
        analysis = self.analyze_data_splits(splits, labels)
        
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training and evaluating {model_name}...")
            
            try:
                if isinstance(datasets, tuple):
                    # Standard split
                    train_dataset, val_dataset, test_dataset = datasets
                    
                    results = self.run_comprehensive_training(
                        model=model,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        test_dataset=test_dataset,
                        **kwargs
                    )
                    
                    comparison_results[model_name] = results
                    
                else:
                    # Cross-validation
                    fold_results = []
                    
                    for fold_idx, (train_dataset, val_dataset, test_dataset) in enumerate(datasets):
                        # Create new model for each fold
                        fold_model = type(model)()
                        
                        fold_result = self.run_comprehensive_training(
                            model=fold_model,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            test_dataset=test_dataset,
                            **kwargs
                        )
                        
                        fold_results.append({
                            'fold': fold_idx + 1,
                            'results': fold_result
                        })
                    
                    # Aggregate fold results
                    test_metrics = [result['results']['evaluation_metrics'] for result in fold_results if result['results']['evaluation_metrics']]
                    
                    if test_metrics and 'accuracy' in test_metrics[0]:
                        accuracies = [metrics['accuracy'] for metrics in test_metrics]
                        comparison_results[model_name] = {
                            'fold_results': fold_results,
                            'mean_accuracy': np.mean(accuracies),
                            'std_accuracy': np.std(accuracies)
                        }
                    else:
                        comparison_results[model_name] = {
                            'fold_results': fold_results
                        }
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
        
        # Create comparison summary
        summary = {
            'model_comparison': comparison_results,
            'split_analysis': analysis,
            'best_model': None,
            'best_metric': float('inf')
        }
        
        # Find best model
        for model_name, results in comparison_results.items():
            if 'error' not in results:
                if 'mean_accuracy' in results:
                    if results['mean_accuracy'] > summary['best_metric']:
                        summary['best_metric'] = results['mean_accuracy']
                        summary['best_model'] = model_name
                elif 'training_metrics' in results:
                    val_loss = results['training_metrics'].best_val_loss
                    if val_loss < summary['best_metric']:
                        summary['best_metric'] = val_loss
                        summary['best_model'] = model_name
        
        logger.info(f"Model comparison with proper splits completed! Best model: {summary['best_model']}")
        return summary
    
    def _extract_labels_from_dataset(self, dataset: Dataset, label_key: str) -> List:
        """Extract labels from dataset"""
        labels = []
        for i in range(min(1000, len(dataset))):  # Sample first 1000 items
            try:
                item = dataset[i]
                if isinstance(item, dict) and label_key in item:
                    labels.append(item[label_key])
                else:
                    labels.append(0)  # Default label
            except:
                labels.append(0)
        
        # Extend with default labels if needed
        while len(labels) < len(dataset):
            labels.append(0)
        
        return labels[:len(dataset)]
    
    def _extract_groups_from_dataset(self, dataset: Dataset, group_key: str) -> List:
        """Extract groups from dataset"""
        groups = []
        for i in range(min(1000, len(dataset))):  # Sample first 1000 items
            try:
                item = dataset[i]
                if isinstance(item, dict) and group_key in item:
                    groups.append(item[group_key])
                else:
                    groups.append(f"group_{i}")
            except:
                groups.append(f"group_{i}")
        
        # Extend with default groups if needed
        while len(groups) < len(dataset):
            groups.append(f"group_{len(groups)}")
        
        return groups[:len(dataset)]

    # Early Stopping and Learning Rate Scheduling Methods
    
    def create_early_stopping_config(self, 
                                   patience: int = 10,
                                   min_delta: float = 1e-4,
                                   mode: str = "min",
                                   monitor: str = "val_loss",
                                   restore_best_weights: bool = True,
                                   save_checkpoint: bool = True,
                                   checkpoint_path: str = "./checkpoints/best_model.pth",
                                   monitor_multiple: bool = False,
                                   monitors: List[str] = None,
                                   monitor_weights: List[float] = None,
                                   adaptive_patience: bool = False,
                                   patience_factor: float = 1.5,
                                   min_patience: int = 5,
                                   max_patience: int = 50,
                                   plateau_detection: bool = False,
                                   plateau_window: int = 5,
                                   plateau_threshold: float = 1e-3,
                                   overfitting_detection: bool = False,
                                   train_val_gap_threshold: float = 0.1,
                                   overfitting_patience: int = 5,
                                   verbose: bool = True) -> EarlyStoppingConfig:
        """Create early stopping configuration"""
        
        if monitors is None:
            monitors = ["val_loss", "val_accuracy"]
        if monitor_weights is None:
            monitor_weights = [1.0, 1.0]
        
        return EarlyStoppingConfig(
            enabled=True,
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            monitor=monitor,
            restore_best_weights=restore_best_weights,
            save_checkpoint=save_checkpoint,
            checkpoint_path=checkpoint_path,
            monitor_multiple=monitor_multiple,
            monitors=monitors,
            monitor_weights=monitor_weights,
            adaptive_patience=adaptive_patience,
            patience_factor=patience_factor,
            min_patience=min_patience,
            max_patience=max_patience,
            plateau_detection=plateau_detection,
            plateau_window=plateau_window,
            plateau_threshold=plateau_threshold,
            overfitting_detection=overfitting_detection,
            train_val_gap_threshold=train_val_gap_threshold,
            overfitting_patience=overfitting_patience,
            verbose=verbose
        )
    
    def create_lr_scheduler_config(self,
                                 scheduler_type: str = "cosine",
                                 initial_lr: float = 1e-3,
                                 min_lr: float = 1e-6,
                                 max_lr: float = 1e-2,
                                 step_size: int = 30,
                                 gamma: float = 0.1,
                                 T_max: int = 100,
                                 eta_min: float = 0.0,
                                 T_0: int = 10,
                                 T_mult: int = 2,
                                 mode: str = "min",
                                 factor: float = 0.5,
                                 patience: int = 5,
                                 threshold: float = 1e-4,
                                 threshold_mode: str = "rel",
                                 cooldown: int = 0,
                                 decay_rate: float = 0.95,
                                 milestones: List[int] = None,
                                 epochs: int = 100,
                                 steps_per_epoch: int = 100,
                                 pct_start: float = 0.3,
                                 anneal_strategy: str = "cos",
                                 cycle_momentum: bool = True,
                                 base_momentum: float = 0.85,
                                 max_momentum: float = 0.95,
                                 div_factor: float = 25.0,
                                 final_div_factor: float = 1e4,
                                 warmup_steps: int = 1000,
                                 warmup_start_lr: float = 1e-6,
                                 custom_lr_fn: Optional[Callable] = None,
                                 verbose: bool = True) -> LRSchedulerConfig:
        """Create learning rate scheduler configuration"""
        
        if milestones is None:
            milestones = [30, 60, 90]
        
        return LRSchedulerConfig(
            scheduler_type=scheduler_type,
            initial_lr=initial_lr,
            min_lr=min_lr,
            max_lr=max_lr,
            step_size=step_size,
            gamma=gamma,
            T_max=T_max,
            eta_min=eta_min,
            T_0=T_0,
            T_mult=T_mult,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            decay_rate=decay_rate,
            milestones=milestones,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            custom_lr_fn=custom_lr_fn,
            verbose=verbose
        )
    
    def create_training_optimizer(self,
                                model: nn.Module,
                                optimizer: optim.Optimizer,
                                early_stopping_config: Optional[EarlyStoppingConfig] = None,
                                lr_scheduler_config: Optional[LRSchedulerConfig] = None) -> TrainingOptimizer:
        """Create training optimizer with early stopping and LR scheduling"""
        
        if early_stopping_config is None:
            early_stopping_config = self.create_early_stopping_config()
        
        if lr_scheduler_config is None:
            lr_scheduler_config = self.create_lr_scheduler_config()
        
        return TrainingOptimizer(
            model=model,
            optimizer=optimizer,
            early_stopping_config=early_stopping_config,
            lr_scheduler_config=lr_scheduler_config
        )
    
    def train_with_early_stopping_lr_scheduling(self,
                                              model: nn.Module,
                                              train_loader: DataLoader,
                                              val_loader: DataLoader,
                                              criterion: nn.Module,
                                              optimizer: optim.Optimizer,
                                              early_stopping_config: Optional[EarlyStoppingConfig] = None,
                                              lr_scheduler_config: Optional[LRSchedulerConfig] = None,
                                              max_epochs: int = 100,
                                              device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Train model with early stopping and learning rate scheduling"""
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.to(device)
        
        # Create training optimizer
        trainer = self.create_training_optimizer(
            model=model,
            optimizer=optimizer,
            early_stopping_config=early_stopping_config,
            lr_scheduler_config=lr_scheduler_config
        )
        
        # Train
        summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=max_epochs)
        
        return summary
    
    def train_with_advanced_early_stopping(self,
                                         model: nn.Module,
                                         train_loader: DataLoader,
                                         val_loader: DataLoader,
                                         criterion: nn.Module,
                                         optimizer: optim.Optimizer,
                                         **kwargs) -> Dict[str, Any]:
        """Train with advanced early stopping strategies"""
        
        # Create advanced early stopping configuration
        early_stopping_config = self.create_early_stopping_config(
            patience=kwargs.get('patience', 15),
            min_delta=kwargs.get('min_delta', 1e-4),
            mode=kwargs.get('mode', 'min'),
            monitor=kwargs.get('monitor', 'val_loss'),
            restore_best_weights=True,
            save_checkpoint=True,
            checkpoint_path=kwargs.get('checkpoint_path', './checkpoints/best_model_advanced.pth'),
            monitor_multiple=kwargs.get('monitor_multiple', True),
            monitors=kwargs.get('monitors', ['val_loss', 'val_accuracy']),
            monitor_weights=kwargs.get('monitor_weights', [1.0, 0.5]),
            adaptive_patience=kwargs.get('adaptive_patience', True),
            patience_factor=kwargs.get('patience_factor', 1.2),
            min_patience=kwargs.get('min_patience', 5),
            max_patience=kwargs.get('max_patience', 30),
            plateau_detection=kwargs.get('plateau_detection', True),
            plateau_window=kwargs.get('plateau_window', 5),
            plateau_threshold=kwargs.get('plateau_threshold', 1e-3),
            overfitting_detection=kwargs.get('overfitting_detection', True),
            train_val_gap_threshold=kwargs.get('train_val_gap_threshold', 0.15),
            overfitting_patience=kwargs.get('overfitting_patience', 8),
            verbose=kwargs.get('verbose', True)
        )
        
        # Create LR scheduler configuration
        lr_scheduler_config = self.create_lr_scheduler_config(
            scheduler_type=kwargs.get('scheduler_type', 'warmup_cosine'),
            initial_lr=kwargs.get('initial_lr', 1e-3),
            warmup_steps=kwargs.get('warmup_steps', 1000),
            warmup_start_lr=kwargs.get('warmup_start_lr', 1e-6),
            T_max=kwargs.get('T_max', 10000),
            eta_min=kwargs.get('eta_min', 1e-6),
            verbose=kwargs.get('verbose', True)
        )
        
        return self.train_with_early_stopping_lr_scheduling(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            early_stopping_config=early_stopping_config,
            lr_scheduler_config=lr_scheduler_config,
            max_epochs=kwargs.get('max_epochs', 100),
            device=kwargs.get('device', None)
        )
    
    def train_with_onecycle_scheduler(self,
                                    model: nn.Module,
                                    train_loader: DataLoader,
                                    val_loader: DataLoader,
                                    criterion: nn.Module,
                                    optimizer: optim.Optimizer,
                                    **kwargs) -> Dict[str, Any]:
        """Train with OneCycle scheduler for fast convergence"""
        
        # Create OneCycle scheduler configuration
        lr_scheduler_config = self.create_lr_scheduler_config(
            scheduler_type="onecycle",
            initial_lr=kwargs.get('initial_lr', 1e-3),
            max_lr=kwargs.get('max_lr', 1e-2),
            epochs=kwargs.get('epochs', 50),
            steps_per_epoch=len(train_loader),
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
            cycle_momentum=kwargs.get('cycle_momentum', True),
            base_momentum=kwargs.get('base_momentum', 0.85),
            max_momentum=kwargs.get('max_momentum', 0.95),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 1e4)
        )
        
        # Create early stopping configuration for OneCycle
        early_stopping_config = self.create_early_stopping_config(
            patience=kwargs.get('patience', 20),  # More patience for OneCycle
            monitor=kwargs.get('monitor', 'val_loss'),
            mode=kwargs.get('mode', 'min'),
            restore_best_weights=True
        )
        
        return self.train_with_early_stopping_lr_scheduling(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            early_stopping_config=early_stopping_config,
            lr_scheduler_config=lr_scheduler_config,
            max_epochs=kwargs.get('max_epochs', 50),
            device=kwargs.get('device', None)
        )
    
    def compare_lr_schedulers(self,
                            model: nn.Module,
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            criterion: nn.Module,
                            optimizer: optim.Optimizer,
                            schedulers: List[str] = None,
                            max_epochs: int = 30,
                            device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Compare different learning rate schedulers"""
        
        if schedulers is None:
            schedulers = ["step", "cosine", "cosine_warm_restarts", "plateau", "exponential", "multistep"]
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        results = {}
        
        for scheduler_type in schedulers:
            logger.info(f"Testing {scheduler_type} scheduler...")
            
            # Create new model for each scheduler
            test_model = type(model)()
            test_model.load_state_dict(model.state_dict())
            test_optimizer = type(optimizer)(test_model.parameters(), **{k: v for k, v in optimizer.param_groups[0].items() if k != 'params'})
            
            # Create scheduler configuration
            lr_scheduler_config = self.create_lr_scheduler_config(
                scheduler_type=scheduler_type,
                initial_lr=1e-3,
                verbose=False
            )
            
            # Create early stopping configuration
            early_stopping_config = self.create_early_stopping_config(
                patience=20,
                monitor="val_loss",
                mode="min",
                verbose=False
            )
            
            try:
                # Train with this scheduler
                summary = self.train_with_early_stopping_lr_scheduling(
                    model=test_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=test_optimizer,
                    early_stopping_config=early_stopping_config,
                    lr_scheduler_config=lr_scheduler_config,
                    max_epochs=max_epochs,
                    device=device
                )
                
                results[scheduler_type] = {
                    'summary': summary,
                    'best_val_loss': summary['early_stopping']['best_score'],
                    'best_epoch': summary['early_stopping']['best_epoch']
                }
                
                logger.info(f"{scheduler_type} - Best val loss: {summary['early_stopping']['best_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error with {scheduler_type} scheduler: {e}")
                results[scheduler_type] = {'error': str(e)}
        
        return results
    
    def create_seo_optimized_training(self,
                                    model: nn.Module,
                                    train_loader: DataLoader,
                                    val_loader: DataLoader,
                                    criterion: nn.Module,
                                    optimizer: optim.Optimizer,
                                    **kwargs) -> Dict[str, Any]:
        """Create SEO-optimized training with early stopping and LR scheduling"""
        
        # SEO-specific early stopping configuration
        early_stopping_config = self.create_early_stopping_config(
            patience=kwargs.get('patience', 25),  # Longer patience for SEO
            min_delta=kwargs.get('min_delta', 1e-4),
            mode=kwargs.get('mode', 'min'),
            monitor=kwargs.get('monitor', 'val_loss'),
            restore_best_weights=True,
            save_checkpoint=True,
            checkpoint_path=kwargs.get('checkpoint_path', './checkpoints/best_seo_model.pth'),
            monitor_multiple=True,
            monitors=kwargs.get('monitors', ['val_loss', 'val_accuracy', 'val_ranking_score']),
            monitor_weights=kwargs.get('monitor_weights', [1.0, 0.3, 0.7]),  # Weight ranking score higher
            adaptive_patience=True,
            patience_factor=kwargs.get('patience_factor', 1.5),  # More adaptive for SEO
            min_patience=kwargs.get('min_patience', 10),
            max_patience=kwargs.get('max_patience', 100),
            plateau_detection=True,
            plateau_window=kwargs.get('plateau_window', 10),
            plateau_threshold=kwargs.get('plateau_threshold', 1e-3),
            overfitting_detection=True,
            train_val_gap_threshold=kwargs.get('train_val_gap_threshold', 0.1),
            overfitting_patience=kwargs.get('overfitting_patience', 10),
            verbose=kwargs.get('verbose', True)
        )
        
        # SEO-specific LR scheduler configuration
        lr_scheduler_config = self.create_lr_scheduler_config(
            scheduler_type=kwargs.get('scheduler_type', 'cosine'),
            initial_lr=kwargs.get('initial_lr', 1e-3),
            T_max=kwargs.get('T_max', 200),  # Longer training for SEO
            eta_min=kwargs.get('eta_min', 1e-7),  # Lower minimum LR
            verbose=kwargs.get('verbose', True)
        )
        
        return self.train_with_early_stopping_lr_scheduling(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            early_stopping_config=early_stopping_config,
            lr_scheduler_config=lr_scheduler_config,
            max_epochs=kwargs.get('max_epochs', 200),
            device=kwargs.get('device', None)
        )

# Example usage
async def main():
    """Example usage of the deep learning framework"""
    
    # Configuration
    config = TrainingConfig(
        model_type="classifier",
        model_name="bert-base-uncased",
        num_classes=2,
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=3,
        use_mixed_precision=True,
        early_stopping_patience=3
    )
    
    # Create framework
    framework = DeepLearningFramework(config)
    
    # Example: Create sample data (replace with real data)
    sample_data = [
        ProcessedData(
            input_ids=torch.randint(0, 1000, (512,)),
            attention_mask=torch.ones(512),
            labels=torch.tensor(0)
        ) for _ in range(100)
    ]
    
    # Split data
    train_size = int(0.8 * len(sample_data))
    val_size = int(0.1 * len(sample_data))
    
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:train_size + val_size]
    test_data = sample_data[train_size + val_size:]
    
    # Train model
    training_results = await framework.train_model(train_data, val_data)
    print(f"Training completed. Final validation loss: {training_results['val_losses'][-1]:.4f}")
    
    # Evaluate model
    evaluation_results = framework.evaluate_model(test_data)
    print(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
    
    # Get training summary
    summary = framework.get_training_summary()
    print(f"Training summary: {summary}")

match __name__:
    case "__main__":
    asyncio.run(main()) 