#!/usr/bin/env python3
"""
Transfer Learning System for Facebook Content Optimization v3.1
Cross-platform optimization and knowledge transfer capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import threading
import asyncio
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import hashlib
from datetime import datetime, timedelta
import warnings
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Import our existing components
from advanced_predictive_system import AdvancedPredictiveSystem, AdvancedPredictiveConfig


@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning system"""
    # Transfer learning parameters
    enable_transfer_learning: bool = True
    transfer_method: str = "fine_tuning"  # fine_tuning, feature_extraction, progressive_unfreezing
    freeze_layers: int = 0  # Number of layers to freeze
    learning_rate_multiplier: float = 0.1  # LR multiplier for transferred layers
    
    # Domain adaptation
    enable_domain_adaptation: bool = True
    adaptation_method: str = "adversarial"  # adversarial, coral, mmd
    domain_loss_weight: float = 0.1
    
    # Knowledge distillation
    enable_knowledge_distillation: bool = True
    distillation_temperature: float = 4.0
    distillation_weight: float = 0.5
    
    # Multi-task learning
    enable_multi_task: bool = True
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'viral_prediction': 1.0,
        'sentiment_analysis': 0.8,
        'engagement_forecasting': 0.9
    })
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    validation_frequency: int = 5
    early_stopping_patience: int = 10


class DomainAdaptationModule(nn.Module):
    """Domain adaptation module for cross-platform learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_domains: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()
        
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward pass with gradient reversal"""
        reversed_features = self.gradient_reversal(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        return domain_output


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for adversarial domain adaptation"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward pass with gradient reversal"""
        return GradientReversalFunction.apply(x, alpha)


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function for gradient reversal"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for model compression"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """Calculate knowledge distillation loss"""
        # Soft targets (teacher knowledge)
        soft_loss = self.kl_loss(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Hard targets (ground truth)
        hard_loss = self.ce_loss(student_output, target)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


class MultiTaskHead(nn.Module):
    """Multi-task learning head for multiple objectives"""
    
    def __init__(self, input_dim: int, task_configs: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.task_configs = task_configs
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            output_dim = config.get('output_dim', 1)
            hidden_dims = config.get('hidden_dims', [128, 64])
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            
            # Add activation if specified
            if config.get('activation') == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif config.get('activation') == 'softmax':
                layers.append(nn.Softmax(dim=1))
            
            self.task_heads[task_name] = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for all tasks"""
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(features)
        return outputs


class TransferLearningModel(nn.Module):
    """Main transfer learning model with domain adaptation"""
    
    def __init__(self, base_model: nn.Module, config: TransferLearningConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Extract feature dimensions
        if hasattr(base_model, 'classifier'):
            feature_dim = base_model.classifier.in_features
        else:
            # Estimate feature dimension
            feature_dim = 512  # Default for most models
        
        # Domain adaptation module
        if config.enable_domain_adaptation:
            self.domain_adaptation = DomainAdaptationModule(feature_dim)
        
        # Multi-task heads
        if config.enable_multi_task:
            task_configs = {
                'viral_prediction': {'output_dim': 1, 'activation': 'sigmoid'},
                'sentiment_analysis': {'output_dim': 7, 'activation': 'softmax'},
                'engagement_forecasting': {'output_dim': 1, 'activation': 'sigmoid'}
            }
            self.multi_task_head = MultiTaskHead(feature_dim, task_configs)
        
        # Freeze specified layers
        self._freeze_layers()
        
    def _freeze_layers(self):
        """Freeze specified number of layers"""
        if self.config.freeze_layers > 0:
            layers_to_freeze = list(self.base_model.children())[:self.config.freeze_layers]
            
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            
            logging.info(f"Frozen {self.config.freeze_layers} layers")
    
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        """Forward pass with feature extraction and task prediction"""
        # Extract features from base model
        features = self._extract_features(x)
        
        outputs = {'features': features}
        
        # Domain adaptation
        if self.config.enable_domain_adaptation and hasattr(self, 'domain_adaptation'):
            domain_output = self.domain_adaptation(features, alpha)
            outputs['domain'] = domain_output
        
        # Multi-task prediction
        if self.config.enable_multi_task and hasattr(self, 'multi_task_head'):
            task_outputs = self.multi_task_head(features)
            outputs.update(task_outputs)
        
        return outputs
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from base model"""
        # Remove classifier if exists
        if hasattr(self.base_model, 'classifier'):
            # Create a new model without classifier
            feature_extractor = copy.deepcopy(self.base_model)
            feature_extractor.classifier = nn.Identity()
            return feature_extractor(x)
        else:
            # Use the entire model as feature extractor
            return self.base_model(x)


class TransferLearningTrainer:
    """Trainer for transfer learning models"""
    
    def __init__(self, model: TransferLearningModel, config: TransferLearningConfig):
        self.model = model
        self.config = config
        self.logger = self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_performance = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Optimizers
        self.optimizers = self._setup_optimizers()
        
        # Loss functions
        self.loss_functions = self._setup_loss_functions()
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        
        self.logger.info("🚀 Transfer Learning Trainer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the trainer"""
        logger = logging.getLogger("TransferLearningTrainer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _setup_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Setup optimizers for different parameter groups"""
        optimizers = {}
        
        # Base model optimizer
        base_params = list(self.model.base_model.parameters())
        if self.config.freeze_layers > 0:
            # Only optimize unfrozen layers
            base_params = base_params[self.config.freeze_layers:]
        
        optimizers['base_model'] = torch.optim.Adam(
            base_params, 
            lr=0.001 * self.config.learning_rate_multiplier
        )
        
        # Task-specific optimizers
        if self.config.enable_multi_task and hasattr(self.model, 'multi_task_head'):
            optimizers['multi_task'] = torch.optim.Adam(
                self.model.multi_task_head.parameters(), 
                lr=0.001
            )
        
        # Domain adaptation optimizer
        if self.config.enable_domain_adaptation and hasattr(self.model, 'domain_adaptation'):
            optimizers['domain_adaptation'] = torch.optim.Adam(
                self.model.domain_adaptation.parameters(), 
                lr=0.001
            )
        
        return optimizers
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup loss functions for different tasks"""
        loss_functions = {}
        
        if self.config.enable_multi_task:
            loss_functions['viral_prediction'] = nn.BCELoss()
            loss_functions['sentiment_analysis'] = nn.CrossEntropyLoss()
            loss_functions['engagement_forecasting'] = nn.MSELoss()
        
        if self.config.enable_domain_adaptation:
            loss_functions['domain_classification'] = nn.CrossEntropyLoss()
        
        return loss_functions
    
    def train_epoch(self, source_loader: DataLoader, target_loader: DataLoader = None) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        
        # Combine source and target data if available
        if target_loader:
            combined_loader = self._combine_data_loaders(source_loader, target_loader)
        else:
            combined_loader = source_loader
        
        for batch_idx, batch_data in enumerate(combined_loader):
            # Prepare data
            if target_loader:
                source_data, target_data = batch_data
                source_input, source_target = source_data
                target_input, target_target = target_data
            else:
                source_input, source_target = batch_data
                target_input, target_target = None, None
            
            # Forward pass
            source_outputs = self.model(source_input)
            if target_input is not None:
                target_outputs = self.model(target_input, alpha=self._calculate_alpha())
            else:
                target_outputs = None
            
            # Calculate losses
            total_loss = 0.0
            
            # Task-specific losses
            if self.config.enable_multi_task:
                for task_name in self.config.task_weights.keys():
                    if task_name in source_outputs and task_name in source_target:
                        task_loss = self.loss_functions[task_name](
                            source_outputs[task_name], 
                            source_target[task_name]
                        )
                        weighted_loss = self.config.task_weights[task_name] * task_loss
                        total_loss += weighted_loss
                        epoch_losses[f'{task_name}_loss'] += weighted_loss.item()
            
            # Domain adaptation loss
            if self.config.enable_domain_adaptation and target_outputs is not None:
                if 'domain' in source_outputs and 'domain' in target_outputs:
                    # Source domain should be classified as source (0)
                    # Target domain should be classified as target (1)
                    source_domain_target = torch.zeros(source_input.size(0), dtype=torch.long)
                    target_domain_target = torch.ones(target_input.size(0), dtype=torch.long)
                    
                    domain_loss = (
                        self.loss_functions['domain_classification'](source_outputs['domain'], source_domain_target) +
                        self.loss_functions['domain_classification'](target_outputs['domain'], target_domain_target)
                    ) / 2
                    
                    weighted_domain_loss = self.config.domain_loss_weight * domain_loss
                    total_loss += weighted_domain_loss
                    epoch_losses['domain_loss'] += weighted_domain_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
            
            # Record metrics
            epoch_losses['total_loss'] += total_loss.item()
            
            # Calculate accuracy for classification tasks
            if self.config.enable_multi_task:
                for task_name in ['viral_prediction', 'sentiment_analysis']:
                    if task_name in source_outputs and task_name in source_target:
                        accuracy = self._calculate_accuracy(
                            source_outputs[task_name], 
                            source_target[task_name], 
                            task_name
                        )
                        epoch_metrics[f'{task_name}_accuracy'] += accuracy
        
        # Average losses and metrics
        num_batches = len(combined_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Update training history
        epoch_record = {
            'epoch': self.current_epoch,
            'losses': dict(epoch_losses),
            'metrics': dict(epoch_metrics),
            'timestamp': datetime.now().isoformat()
        }
        self.training_history.append(epoch_record)
        
        return dict(epoch_losses)
    
    def _combine_data_loaders(self, source_loader: DataLoader, target_loader: DataLoader) -> DataLoader:
        """Combine source and target data loaders"""
        # This is a simplified implementation
        # In practice, you might want more sophisticated data mixing strategies
        
        source_data = list(source_loader)
        target_data = list(target_loader)
        
        # Pad shorter loader with repeated data
        max_length = max(len(source_data), len(target_data))
        
        if len(source_data) < max_length:
            source_data.extend(source_data * (max_length // len(source_data)))
            source_data.extend(source_data[:max_length % len(source_data)])
        
        if len(target_data) < max_length:
            target_data.extend(target_data * (max_length // len(target_data)))
            target_data.extend(target_data[:max_length % len(target_data)])
        
        # Combine data
        combined_data = []
        for i in range(max_length):
            combined_data.append((source_data[i], target_data[i]))
        
        return combined_data
    
    def _calculate_alpha(self) -> float:
        """Calculate domain adaptation alpha parameter"""
        # Progressive domain adaptation
        progress = self.current_epoch / 100  # Assuming 100 epochs
        return min(1.0, progress * 2)
    
    def _calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, 
                           task_name: str) -> float:
        """Calculate accuracy for different tasks"""
        if task_name == 'viral_prediction':
            # Binary classification
            pred_labels = (predictions > 0.5).float()
            return (pred_labels == targets).float().mean().item()
        
        elif task_name == 'sentiment_analysis':
            # Multi-class classification
            pred_labels = torch.argmax(predictions, dim=1)
            return (pred_labels == targets).float().mean().item()
        
        else:
            return 0.0
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = defaultdict(float)
        val_metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, targets = batch_data
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate losses
                if self.config.enable_multi_task:
                    for task_name in self.config.task_weights.keys():
                        if task_name in outputs and task_name in targets:
                            task_loss = self.loss_functions[task_name](
                                outputs[task_name], 
                                targets[task_name]
                            )
                            weighted_loss = self.config.task_weights[task_name] * task_loss
                            val_losses[f'{task_name}_loss'] += weighted_loss.item()
                            
                            # Calculate accuracy
                            accuracy = self._calculate_accuracy(
                                outputs[task_name], 
                                targets[task_name], 
                                task_name
                            )
                            val_metrics[f'{task_name}_accuracy'] += accuracy
        
        # Average validation metrics
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return {**dict(val_losses), **dict(val_metrics)}
    
    def train(self, source_loader: DataLoader, target_loader: DataLoader = None,
              val_loader: DataLoader = None, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Main training loop"""
        self.logger.info(f"Starting transfer learning training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch(source_loader, target_loader)
            
            # Validation
            if val_loader and epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate(val_loader)
                
                # Record validation metrics
                for key, value in val_metrics.items():
                    self.performance_metrics[key].append(value)
                
                # Early stopping check
                if self._check_early_stopping(val_metrics):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}/{num_epochs}, "
                               f"Total Loss: {train_losses['total_loss']:.4f}")
        
        self.logger.info("Transfer learning training completed")
        return self.performance_metrics
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered"""
        if not self.config.enable_performance_tracking:
            return False
        
        # Use total loss for early stopping
        current_loss = sum(val_metrics.get(f'{task}_loss', 0) 
                          for task in self.config.task_weights.keys())
        
        if current_loss < self.best_performance:
            self.best_performance = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'best_performance': self.best_performance
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if Path(filepath).exists():
            checkpoint = torch.load(filepath)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            if 'performance_metrics' in checkpoint:
                self.performance_metrics = checkpoint['performance_metrics']
            
            if 'best_performance' in checkpoint:
                self.best_performance = checkpoint['best_performance']
            
            self.logger.info(f"Model loaded from {filepath}")


class TransferLearningSystem:
    """Main system orchestrating transfer learning capabilities"""
    
    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.model = None
        self.trainer = None
        self.performance_tracker = {}
        
        self.logger.info("🚀 Transfer Learning System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the system"""
        logger = logging.getLogger("TransferLearningSystem")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def setup_model(self, base_model: nn.Module):
        """Setup transfer learning model"""
        self.model = TransferLearningModel(base_model, self.config)
        self.trainer = TransferLearningTrainer(self.model, self.config)
        
        self.logger.info("Transfer learning model setup completed")
    
    def train_model(self, source_loader: DataLoader, target_loader: DataLoader = None,
                   val_loader: DataLoader = None, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Train the transfer learning model"""
        if not self.trainer:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        return self.trainer.train(source_loader, target_loader, val_loader, num_epochs)
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the trained model"""
        if not self.trainer:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        return self.trainer.validate(test_loader)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.trainer:
            self.trainer.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if self.trainer:
            self.trainer.load_model(filepath)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and performance metrics"""
        status = {
            'config': self.config,
            'model_initialized': self.model is not None,
            'trainer_initialized': self.trainer is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.trainer:
            status.update({
                'current_epoch': self.trainer.current_epoch,
                'best_performance': self.trainer.best_performance,
                'training_history_length': len(self.trainer.training_history),
                'performance_metrics': dict(self.trainer.performance_metrics)
            })
        
        return status


# Example usage and testing
if __name__ == "__main__":
    # Initialize transfer learning system
    config = TransferLearningConfig(
        transfer_method="fine_tuning",
        freeze_layers=2,
        enable_domain_adaptation=True,
        enable_multi_task=True
    )
    
    system = TransferLearningSystem(config)
    
    # Create a simple base model
    class SimpleBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU()
            )
            self.classifier = nn.Linear(256, 10)
            
        def forward(self, x):
            features = self.features(x)
            output = self.classifier(features)
            return output
    
    # Setup model
    base_model = SimpleBaseModel()
    system.setup_model(base_model)
    
    print("🚀 Transfer Learning System initialized successfully!")
    print("📊 System Status:", system.get_system_status())

