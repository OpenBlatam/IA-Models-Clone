"""
Loss Functions and Optimization Algorithms System
Comprehensive implementation of appropriate loss functions and optimization algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
    ReduceLROnPlateau, OneCycleLR, LambdaLR, MultiStepLR
)
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


@dataclass
class OptimizationConfig:
    """Configuration for optimization and loss functions."""
    
    # Model parameters
    input_size: int = 784
    hidden_size: int = 256
    output_size: int = 10
    num_layers: int = 3
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    num_epochs: int = 100
    
    # Loss function parameters
    loss_type: str = "cross_entropy"  # cross_entropy, mse, mae, huber, focal, dice, kl_div
    loss_alpha: float = 0.25  # For focal loss
    loss_gamma: float = 2.0   # For focal loss
    loss_smooth: float = 0.1  # For label smoothing
    
    # Optimizer parameters
    optimizer_type: str = "adam"  # sgd, adam, adamw, rmsprop, adagrad, adadelta
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    
    # Learning rate scheduler parameters
    scheduler_type: str = "cosine"  # step, exponential, cosine, plateau, onecycle, lambda
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    scheduler_warmup_steps: int = 1000


class CustomLossFunction(nn.Module, ABC):
    """Abstract base class for custom loss functions."""
    
    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss between predictions and targets."""
        pass


class FocalLoss(CustomLossFunction):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        # Apply softmax to get probabilities
        if predictions.dim() > 1:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = torch.sigmoid(predictions)
        
        # Get target probabilities
        if targets.dim() > 1:
            target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            target_probs = probs
        
        # Compute focal loss
        focal_weight = (1 - target_probs) ** self.gamma
        focal_loss = -self.alpha * focal_weight * torch.log(target_probs + 1e-8)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(CustomLossFunction):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss."""
        # Apply softmax for multi-class
        if predictions.dim() > 1 and predictions.size(1) > 1:
            predictions = F.softmax(predictions, dim=1)
        else:
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Compute dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return dice loss
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss * predictions.numel()
        else:
            return dice_loss


class LabelSmoothingLoss(CustomLossFunction):
    """Label Smoothing Loss for regularization."""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss."""
        num_classes = predictions.size(-1)
        
        # Create smoothed targets
        smoothed_targets = torch.zeros_like(predictions)
        smoothed_targets.fill_(self.smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute cross entropy with smoothed targets
        log_probs = F.log_softmax(predictions, dim=1)
        loss = -(smoothed_targets * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class KLDivergenceLoss(CustomLossFunction):
    """KL Divergence Loss for distribution matching."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        # Apply log softmax to predictions
        log_predictions = F.log_softmax(predictions, dim=1)
        
        # Apply softmax to targets
        soft_targets = F.softmax(targets, dim=1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(log_predictions, soft_targets, reduction='none').sum(dim=1)
        
        if self.reduction == 'mean':
            return kl_loss.mean()
        elif self.reduction == 'sum':
            return kl_loss.sum()
        else:
            return kl_loss


class LossFunctionFactory:
    """Factory for creating loss functions."""
    
    @staticmethod
    def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
        """Create a loss function based on type."""
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(**kwargs)
        elif loss_type == "mse":
            return nn.MSELoss(**kwargs)
        elif loss_type == "mae":
            return nn.L1Loss(**kwargs)
        elif loss_type == "huber":
            return nn.SmoothL1Loss(**kwargs)
        elif loss_type == "focal":
            return FocalLoss(**kwargs)
        elif loss_type == "dice":
            return DiceLoss(**kwargs)
        elif loss_type == "label_smoothing":
            return LabelSmoothingLoss(**kwargs)
        elif loss_type == "kl_divergence":
            return KLDivergenceLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create_optimizer(
        model: nn.Module, 
        optimizer_type: str, 
        learning_rate: float,
        **kwargs
    ) -> optim.Optimizer:
        """Create an optimizer based on type."""
        if optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0),
                nesterov=kwargs.get('nesterov', False)
            )
        elif optimizer_type == "adam":
            return optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0.0)
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0.01)
            )
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0.0),
                alpha=kwargs.get('alpha', 0.99),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0.0)
            )
        elif optimizer_type == "adagrad":
            return optim.Adagrad(
                model.parameters(),
                lr=learning_rate,
                weight_decay=kwargs.get('weight_decay', 0.0),
                eps=kwargs.get('eps', 1e-10)
            )
        elif optimizer_type == "adadelta":
            return optim.Adadelta(
                model.parameters(),
                lr=learning_rate,
                rho=kwargs.get('rho', 0.9),
                eps=kwargs.get('eps', 1e-6),
                weight_decay=kwargs.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str,
        **kwargs
    ) -> Any:
        """Create a learning rate scheduler based on type."""
        if scheduler_type == "step":
            return StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == "exponential":
            return ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 1e-7)
            )
        elif scheduler_type == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 1e-7)
            )
        elif scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                min_lr=kwargs.get('min_lr', 1e-7)
            )
        elif scheduler_type == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', 0.1),
                epochs=kwargs.get('epochs', 100),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100),
                pct_start=kwargs.get('pct_start', 0.3)
            )
        elif scheduler_type == "lambda":
            def lambda_func(epoch):
                return kwargs.get('lambda_factor', 0.95) ** epoch
            return LambdaLR(optimizer, lr_lambda=lambda_func)
        elif scheduler_type == "multistep":
            return MultiStepLR(
                optimizer,
                milestones=kwargs.get('milestones', [30, 60, 90]),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class AdvancedModel(nn.Module):
    """Advanced model for demonstrating loss functions and optimization."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(config.input_size, config.hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(config.num_layers - 1):
            self.layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
        
        # Output layer
        self.layers.append(nn.Linear(config.hidden_size, config.output_size))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        for layer in self.layers:
            x = layer(x)
        return x


class AdvancedTrainer:
    """Advanced trainer with comprehensive optimization features."""
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create loss function
        self.criterion = LossFunctionFactory.create_loss_function(
            config.loss_type,
            alpha=config.loss_alpha,
            gamma=config.loss_gamma,
            smoothing=config.loss_smooth
        )
        
        # Create optimizer
        self.optimizer = OptimizerFactory.create_optimizer(
            model,
            config.optimizer_type,
            config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            beta1=config.optimizer_beta1,
            beta2=config.optimizer_beta2,
            eps=config.optimizer_eps
        )
        
        # Create scheduler
        self.scheduler = SchedulerFactory.create_scheduler(
            self.optimizer,
            config.scheduler_type,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            min_lr=config.scheduler_min_lr,
            T_max=config.num_epochs,
            T_0=config.scheduler_warmup_steps
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logging.info(f"Trainer initialized with {config.optimizer_type} optimizer and {config.scheduler_type} scheduler")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update learning rate
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Record metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(
                    f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                    f"LR: {current_lr:.2e}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Update scheduler (for non-OneCycleLR schedulers)
        if not isinstance(self.scheduler, OneCycleLR):
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        return {'train_loss': avg_loss, 'num_batches': num_batches}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss, 'num_batches': num_batches}
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> Dict[str, List[float]]:
        """Complete training loop."""
        for epoch in range(self.config.num_epochs):
            logging.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            self.train_losses.append(train_metrics['train_loss'])
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            self.val_losses.append(val_metrics['val_loss'])
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            logging.info(
                f"Epoch {epoch + 1} completed. "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {current_lr:.2e}"
            )
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        logging.info(f"Checkpoint loaded from {filepath}")


class OptimizationAnalyzer:
    """Analyze optimization performance and convergence."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_convergence(self, trainer: AdvancedTrainer) -> Dict[str, Any]:
        """Analyze training convergence."""
        train_losses = trainer.train_losses
        val_losses = trainer.val_losses
        learning_rates = trainer.learning_rates
        
        if not train_losses or not val_losses:
            return {}
        
        # Calculate convergence metrics
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        best_val_loss = min(val_losses)
        best_epoch = val_losses.index(best_val_loss)
        
        # Calculate convergence rate
        if len(train_losses) > 1:
            convergence_rate = (train_losses[0] - final_train_loss) / len(train_losses)
        else:
            convergence_rate = 0.0
        
        # Calculate overfitting metric
        overfitting_metric = final_val_loss - final_train_loss
        
        # Calculate learning rate statistics
        lr_stats = {
            'initial_lr': learning_rates[0] if learning_rates else 0.0,
            'final_lr': learning_rates[-1] if learning_rates else 0.0,
            'lr_decay_factor': learning_rates[-1] / learning_rates[0] if learning_rates and learning_rates[0] > 0 else 1.0
        }
        
        self.analysis_results = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'convergence_rate': convergence_rate,
            'overfitting_metric': overfitting_metric,
            'learning_rate_stats': lr_stats
        }
        
        return self.analysis_results
    
    def plot_training_curves(self, trainer: AdvancedTrainer):
        """Plot training curves and learning rate."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(trainer.train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(trainer.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate over time
        axes[0, 1].plot(trainer.learning_rates, color='green')
        axes[0, 1].set_title('Learning Rate Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        if len(trainer.train_losses) == len(trainer.val_losses):
            loss_diff = [val - train for train, val in zip(trainer.train_losses, trainer.val_losses)]
            axes[1, 0].plot(loss_diff, color='orange')
            axes[1, 0].set_title('Overfitting Indicator (Val Loss - Train Loss)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Difference')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Loss ratio
        if len(trainer.train_losses) == len(trainer.val_losses):
            loss_ratio = [val / train if train > 0 else 1.0 for train, val in zip(trainer.train_losses, trainer.val_losses)]
            axes[1, 1].plot(loss_ratio, color='purple')
            axes[1, 1].set_title('Loss Ratio (Val Loss / Train Loss)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Ratio')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def print_analysis_summary(self):
        """Print analysis summary."""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_convergence first.")
            return
        
        print("\n=== Optimization Analysis Summary ===")
        print(f"Final Training Loss: {self.analysis_results['final_train_loss']:.4f}")
        print(f"Final Validation Loss: {self.analysis_results['final_val_loss']:.4f}")
        print(f"Best Validation Loss: {self.analysis_results['best_val_loss']:.4f} (Epoch {self.analysis_results['best_epoch']})")
        print(f"Convergence Rate: {self.analysis_results['convergence_rate']:.6f}")
        print(f"Overfitting Metric: {self.analysis_results['overfitting_metric']:.4f}")
        
        lr_stats = self.analysis_results['learning_rate_stats']
        print(f"Initial Learning Rate: {lr_stats['initial_lr']:.2e}")
        print(f"Final Learning Rate: {lr_stats['final_lr']:.2e}")
        print(f"Learning Rate Decay Factor: {lr_stats['lr_decay_factor']:.4f}")


class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration."""
    
    def __init__(self, num_samples: int, input_size: int, output_size: int, task_type: str = "classification"):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.task_type = task_type
        
        # Generate synthetic data
        self.inputs = torch.randn(num_samples, input_size)
        
        if task_type == "classification":
            # Generate random class labels
            self.targets = torch.randint(0, output_size, (num_samples,))
        elif task_type == "regression":
            # Generate continuous targets
            self.targets = torch.randn(num_samples, output_size)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = OptimizationConfig(
        input_size=784,
        hidden_size=256,
        output_size=10,
        num_layers=3,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=20,
        loss_type="cross_entropy",
        optimizer_type="adamw",
        scheduler_type="cosine_warm_restarts"
    )
    
    # Create model and trainer
    model = AdvancedModel(config)
    trainer = AdvancedTrainer(model, config)
    
    # Create synthetic datasets
    train_dataset = SyntheticDataset(1000, config.input_size, config.output_size, "classification")
    val_dataset = SyntheticDataset(200, config.input_size, config.output_size, "classification")
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train the model
    logging.info("Starting training...")
    training_history = trainer.train(train_dataloader, val_dataloader)
    
    # Analyze results
    analyzer = OptimizationAnalyzer()
    analysis_results = analyzer.analyze_convergence(trainer)
    
    # Print and visualize results
    analyzer.print_analysis_summary()
    analyzer.plot_training_curves(trainer)
    
    # Save checkpoint
    trainer.save_checkpoint("optimization_checkpoint.pt")
    
    logging.info("Training and optimization demonstration completed successfully!")





