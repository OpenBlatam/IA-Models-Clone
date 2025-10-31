from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from early_stopping_lr_scheduling import (
from model_training_evaluation import (
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Example: Early Stopping and Learning Rate Scheduling Framework
Demonstrates various early stopping strategies and learning rate scheduling algorithms
"""


# Import our early stopping and LR scheduling framework
    EarlyStoppingConfig, LRSchedulerConfig, TrainingMetrics,
    EarlyStopping, AdvancedLRScheduler, TrainingMonitor, TrainingOptimizer
)

# Import training framework for integration
    TrainingConfig, ModelTrainer, ModelEvaluator, EfficientDataLoader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEOSampleDataset(Dataset):
    """Sample SEO dataset for demonstration"""
    
    def __init__(self, num_samples=1000, num_features=768, num_classes=3) -> Any:
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self._generate_data()
    
    def _generate_data(self) -> Any:
        """Generate synthetic SEO data"""
        np.random.seed(42)
        
        # Generate features
        self.features = torch.randn(self.num_samples, self.num_features)
        
        # Generate labels with some class imbalance
        # Class 0: 60%, Class 1: 30%, Class 2: 10%
        class_weights = [0.6, 0.3, 0.1]
        self.labels = torch.tensor(
            np.random.choice(self.num_classes, self.num_samples, p=class_weights),
            dtype=torch.long
        )
        
        # Add some noise to make training challenging
        noise = torch.randn_like(self.features) * 0.1
        self.features += noise
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }

class SEOMultiTaskModel(nn.Module):
    """Multi-task SEO model for demonstration"""
    
    def __init__(self, input_size=768, hidden_size=512, num_classes=3, num_tasks=2) -> Any:
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classification_head = nn.Linear(hidden_size // 2, num_classes)
        
        # Regression head (for SEO scores)
        self.regression_head = nn.Linear(hidden_size // 2, num_tasks)
        
    def forward(self, x) -> Any:
        if isinstance(x, dict):
            x = x['features']
        
        shared_features = self.shared_layers(x)
        
        classification_output = self.classification_head(shared_features)
        regression_output = self.regression_head(shared_features)
        
        return {
            'classification': classification_output,
            'regression': regression_output
        }

def example_basic_early_stopping():
    """Example: Basic early stopping"""
    logger.info("=== Basic Early Stopping Example ===")
    
    # Create dataset
    dataset = SEOSampleDataset(1000)
    
    # Create model
    model = SEOMultiTaskModel(input_size=768, hidden_size=256, num_classes=3)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create early stopping configuration
    early_stopping_config = EarlyStoppingConfig(
        enabled=True,
        patience=10,
        min_delta=1e-4,
        mode="min",
        monitor="val_loss",
        restore_best_weights=True,
        verbose=True
    )
    
    # Create LR scheduler configuration
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type="cosine",
        initial_lr=1e-3,
        T_max=100,
        verbose=True
    )
    
    # Create training optimizer
    trainer = TrainingOptimizer(
        model=model,
        optimizer=optimizer,
        early_stopping_config=early_stopping_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=50)
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {summary['early_stopping']['best_score']:.4f}")
    logger.info(f"Best epoch: {summary['early_stopping']['best_epoch']}")
    
    return summary

def example_advanced_early_stopping():
    """Example: Advanced early stopping with multiple strategies"""
    logger.info("=== Advanced Early Stopping Example ===")
    
    # Create dataset
    dataset = SEOSampleDataset(1000)
    
    # Create model
    model = SEOMultiTaskModel(input_size=768, hidden_size=256, num_classes=3)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create advanced early stopping configuration
    early_stopping_config = EarlyStoppingConfig(
        enabled=True,
        patience=15,
        min_delta=1e-4,
        mode="min",
        monitor="val_loss",
        restore_best_weights=True,
        save_checkpoint=True,
        checkpoint_path="./checkpoints/best_model_advanced.pth",
        
        # Multiple metric monitoring
        monitor_multiple=True,
        monitors=["val_loss", "val_accuracy"],
        monitor_weights=[1.0, 0.5],
        
        # Adaptive patience
        adaptive_patience=True,
        patience_factor=1.2,
        min_patience=5,
        max_patience=30,
        
        # Plateau detection
        plateau_detection=True,
        plateau_window=5,
        plateau_threshold=1e-3,
        
        # Overfitting detection
        overfitting_detection=True,
        train_val_gap_threshold=0.15,
        overfitting_patience=8,
        
        verbose=True
    )
    
    # Create LR scheduler configuration
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type="plateau",
        initial_lr=1e-3,
        min_lr=1e-6,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        verbose=True
    )
    
    # Create training optimizer
    trainer = TrainingOptimizer(
        model=model,
        optimizer=optimizer,
        early_stopping_config=early_stopping_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=100)
    
    logger.info(f"Advanced training completed!")
    logger.info(f"Best validation loss: {summary['early_stopping']['best_score']:.4f}")
    logger.info(f"Best epoch: {summary['early_stopping']['best_epoch']}")
    
    return summary

def example_lr_scheduling_strategies():
    """Example: Different learning rate scheduling strategies"""
    logger.info("=== Learning Rate Scheduling Strategies Example ===")
    
    strategies = [
        {
            'name': 'Step LR',
            'config': LRSchedulerConfig(
                scheduler_type="step",
                initial_lr=1e-3,
                step_size=10,
                gamma=0.5,
                verbose=False
            )
        },
        {
            'name': 'Cosine Annealing',
            'config': LRSchedulerConfig(
                scheduler_type="cosine",
                initial_lr=1e-3,
                T_max=50,
                eta_min=1e-6,
                verbose=False
            )
        },
        {
            'name': 'Cosine Warm Restarts',
            'config': LRSchedulerConfig(
                scheduler_type="cosine_warm_restarts",
                initial_lr=1e-3,
                T_0=10,
                T_mult=2,
                eta_min=1e-6,
                verbose=False
            )
        },
        {
            'name': 'Reduce on Plateau',
            'config': LRSchedulerConfig(
                scheduler_type="plateau",
                initial_lr=1e-3,
                min_lr=1e-6,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=False
            )
        },
        {
            'name': 'Exponential',
            'config': LRSchedulerConfig(
                scheduler_type="exponential",
                initial_lr=1e-3,
                decay_rate=0.95,
                verbose=False
            )
        },
        {
            'name': 'Multi-Step',
            'config': LRSchedulerConfig(
                scheduler_type="multistep",
                initial_lr=1e-3,
                milestones=[15, 30, 45],
                gamma=0.5,
                verbose=False
            )
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Testing {strategy['name']}...")
        
        # Create model
        model = SEOMultiTaskModel(input_size=768, hidden_size=128, num_classes=3)
        optimizer = optim.Adam(model.parameters(), lr=strategy['config'].initial_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Create early stopping
        early_stopping_config = EarlyStoppingConfig(
            patience=20,
            monitor="val_loss",
            mode="min",
            verbose=False
        )
        
        # Create training optimizer
        trainer = TrainingOptimizer(
            model=model,
            optimizer=optimizer,
            early_stopping_config=early_stopping_config,
            lr_scheduler_config=strategy['config']
        )
        
        # Create data loaders
        dataset = SEOSampleDataset(500)  # Smaller dataset for faster testing
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=30)
        
        results[strategy['name']] = {
            'summary': summary,
            'lr_history': trainer.monitor.lr_scheduler.history
        }
        
        logger.info(f"{strategy['name']} - Best val loss: {summary['early_stopping']['best_score']:.4f}")
    
    # Plot LR schedules
    plt.figure(figsize=(15, 10))
    
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(2, 3, i + 1)
        epochs = [h['epoch'] for h in result['lr_history']]
        lrs = [h['learning_rate'] for h in result['lr_history']]
        plt.plot(epochs, lrs, label=name, linewidth=2)
        plt.title(f'{name} Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('lr_scheduling_strategies.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def example_onecycle_scheduler():
    """Example: OneCycle scheduler for fast training"""
    logger.info("=== OneCycle Scheduler Example ===")
    
    # Create dataset
    dataset = SEOSampleDataset(1000)
    
    # Create model
    model = SEOMultiTaskModel(input_size=768, hidden_size=256, num_classes=3)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create OneCycle scheduler configuration
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type="onecycle",
        initial_lr=1e-3,
        max_lr=1e-2,
        epochs=30,
        steps_per_epoch=25,  # len(train_loader)
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        verbose=True
    )
    
    # Create early stopping configuration
    early_stopping_config = EarlyStoppingConfig(
        patience=15,
        monitor="val_loss",
        mode="min",
        verbose=True
    )
    
    # Create training optimizer
    trainer = TrainingOptimizer(
        model=model,
        optimizer=optimizer,
        early_stopping_config=early_stopping_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=30)
    
    # Plot OneCycle schedule
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = [h['epoch'] for h in trainer.monitor.lr_scheduler.history]
    lrs = [h['learning_rate'] for h in trainer.monitor.lr_scheduler.history]
    plt.plot(epochs, lrs, 'b-', linewidth=2)
    plt.title('OneCycle Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs = [m.epoch for m in trainer.monitor.training_history]
    train_losses = [m.train_loss for m in trainer.monitor.training_history]
    val_losses = [m.val_loss for m in trainer.monitor.training_history]
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('onecycle_scheduler.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"OneCycle training completed!")
    logger.info(f"Best validation loss: {summary['early_stopping']['best_score']:.4f}")
    
    return summary

def example_warmup_cosine_scheduler():
    """Example: Warmup cosine scheduler"""
    logger.info("=== Warmup Cosine Scheduler Example ===")
    
    # Create dataset
    dataset = SEOSampleDataset(1000)
    
    # Create model
    model = SEOMultiTaskModel(input_size=768, hidden_size=256, num_classes=3)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create warmup cosine scheduler configuration
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type="warmup_cosine",
        initial_lr=1e-3,
        warmup_steps=5,
        warmup_start_lr=1e-6,
        T_max=30,
        eta_min=1e-6,
        verbose=True
    )
    
    # Create early stopping configuration
    early_stopping_config = EarlyStoppingConfig(
        patience=15,
        monitor="val_loss",
        mode="min",
        verbose=True
    )
    
    # Create training optimizer
    trainer = TrainingOptimizer(
        model=model,
        optimizer=optimizer,
        early_stopping_config=early_stopping_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=30)
    
    # Plot warmup cosine schedule
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = [h['epoch'] for h in trainer.monitor.lr_scheduler.history]
    lrs = [h['learning_rate'] for h in trainer.monitor.lr_scheduler.history]
    plt.plot(epochs, lrs, 'g-', linewidth=2)
    plt.title('Warmup Cosine Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs = [m.epoch for m in trainer.monitor.training_history]
    train_losses = [m.train_loss for m in trainer.monitor.training_history]
    val_losses = [m.val_loss for m in trainer.monitor.training_history]
    plt.plot(epochs, train_losses, 'g-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('warmup_cosine_scheduler.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Warmup cosine training completed!")
    logger.info(f"Best validation loss: {summary['early_stopping']['best_score']:.4f}")
    
    return summary

def example_integration_with_training_framework():
    """Example: Integration with the training framework"""
    logger.info("=== Integration with Training Framework Example ===")
    
    # Create dataset
    dataset = SEOSampleDataset(1000)
    
    # Create model
    model = SEOMultiTaskModel(input_size=768, hidden_size=256, num_classes=3)
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create training configuration
    training_config = TrainingConfig(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        use_mixed_precision=True,
        early_stopping_patience=15,
        early_stopping_min_delta=1e-4
    )
    
    # Create trainer
    trainer = ModelTrainer(training_config)
    
    # Train
    metrics = trainer.train()
    
    logger.info(f"Training framework integration completed!")
    logger.info(f"Best validation loss: {metrics.best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {metrics.best_val_accuracy:.4f}")
    
    return metrics

def example_custom_lr_function():
    """Example: Custom learning rate function"""
    logger.info("=== Custom Learning Rate Function Example ===")
    
    # Define custom LR function
    def custom_lr_fn(epoch) -> Any:
        """Custom learning rate function with warmup and cosine decay"""
        warmup_epochs = 5
        total_epochs = 30
        
        if epoch < warmup_epochs:
            # Linear warmup
            return 1e-6 + (1e-3 - 1e-6) * epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + np.cos(np.pi * progress))
    
    # Create dataset
    dataset = SEOSampleDataset(1000)
    
    # Create model
    model = SEOMultiTaskModel(input_size=768, hidden_size=256, num_classes=3)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create custom scheduler configuration
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type="custom",
        custom_lr_fn=custom_lr_fn,
        verbose=True
    )
    
    # Create early stopping configuration
    early_stopping_config = EarlyStoppingConfig(
        patience=15,
        monitor="val_loss",
        mode="min",
        verbose=True
    )
    
    # Create training optimizer
    trainer = TrainingOptimizer(
        model=model,
        optimizer=optimizer,
        early_stopping_config=early_stopping_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=30)
    
    # Plot custom LR schedule
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = [h['epoch'] for h in trainer.monitor.lr_scheduler.history]
    lrs = [h['learning_rate'] for h in trainer.monitor.lr_scheduler.history]
    plt.plot(epochs, lrs, 'purple', linewidth=2)
    plt.title('Custom Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs = [m.epoch for m in trainer.monitor.training_history]
    train_losses = [m.train_loss for m in trainer.monitor.training_history]
    val_losses = [m.val_loss for m in trainer.monitor.training_history]
    plt.plot(epochs, train_losses, 'purple', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'red', label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('custom_lr_schedule.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Custom LR function training completed!")
    logger.info(f"Best validation loss: {summary['early_stopping']['best_score']:.4f}")
    
    return summary

def main():
    """Run all examples"""
    logger.info("Starting Early Stopping and Learning Rate Scheduling Examples")
    
    try:
        # Basic early stopping
        basic_summary = example_basic_early_stopping()
        
        # Advanced early stopping
        advanced_summary = example_advanced_early_stopping()
        
        # LR scheduling strategies
        lr_strategies_results = example_lr_scheduling_strategies()
        
        # OneCycle scheduler
        onecycle_summary = example_onecycle_scheduler()
        
        # Warmup cosine scheduler
        warmup_cosine_summary = example_warmup_cosine_scheduler()
        
        # Custom LR function
        custom_lr_summary = example_custom_lr_function()
        
        # Integration with training framework
        framework_integration = example_integration_with_training_framework()
        
        logger.info("\n=== Summary ===")
        logger.info("All early stopping and LR scheduling examples completed successfully!")
        logger.info("Key features demonstrated:")
        logger.info("  ✓ Basic early stopping with patience")
        logger.info("  ✓ Advanced early stopping with multiple strategies")
        logger.info("  ✓ Multiple LR scheduling algorithms")
        logger.info("  ✓ OneCycle scheduler for fast training")
        logger.info("  ✓ Warmup cosine scheduler")
        logger.info("  ✓ Custom LR functions")
        logger.info("  ✓ Integration with training framework")
        
        # Print best results
        logger.info("\nBest Results:")
        logger.info(f"  Basic early stopping: {basic_summary['early_stopping']['best_score']:.4f}")
        logger.info(f"  Advanced early stopping: {advanced_summary['early_stopping']['best_score']:.4f}")
        logger.info(f"  OneCycle scheduler: {onecycle_summary['early_stopping']['best_score']:.4f}")
        logger.info(f"  Warmup cosine: {warmup_cosine_summary['early_stopping']['best_score']:.4f}")
        logger.info(f"  Custom LR function: {custom_lr_summary['early_stopping']['best_score']:.4f}")
        logger.info(f"  Training framework: {framework_integration.best_val_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

match __name__:
    case "__main__":
    main() 