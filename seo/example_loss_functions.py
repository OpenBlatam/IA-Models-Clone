from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Any, List
from loss_functions import (
from custom_models import (
from deep_learning_framework import CustomSEOModelTrainer, TrainingConfig
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Example Usage of Loss Functions and Optimization Algorithms
Demonstrates advanced loss functions and optimization strategies for SEO service
"""


# Import our custom modules
    LossFunctionManager, LossConfig, OptimizerConfig, SchedulerConfig,
    AdvancedOptimizer, AdvancedScheduler, SEOSpecificLoss, FocalLoss,
    LabelSmoothingLoss, RankingLoss, ContrastiveLoss, MultiTaskLoss,
    UncertaintyLoss, DiceLoss
)
    CustomSEOModel, CustomModelConfig, create_custom_model
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(batch_size: int = 16, seq_length: int = 128, vocab_size: int = 1000):
    """Create sample data for demonstration"""
    input_ids = torch.randint(0, vocab_size, (batch_size * 10, seq_length))
    attention_mask = torch.ones(batch_size * 10, seq_length)
    labels = torch.randint(0, 3, (batch_size * 10,))
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def demonstrate_loss_functions():
    """Demonstrate different loss functions"""
    logger.info("=== Demonstrating Loss Functions ===")
    
    # Create sample data
    batch_size = 8
    num_classes = 3
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test different loss functions
    loss_functions = [
        ("Cross Entropy", nn.CrossEntropyLoss()),
        ("Focal Loss", FocalLoss(alpha=1.0, gamma=2.0)),
        ("Label Smoothing", LabelSmoothingLoss(smoothing=0.1)),
        ("Dice Loss", DiceLoss()),
    ]
    
    for loss_name, loss_fn in loss_functions:
        try:
            loss = loss_fn(inputs, targets)
            logger.info(f"{loss_name}: {loss.item():.4f}")
        except Exception as e:
            logger.error(f"{loss_name} failed: {e}")

def demonstrate_ranking_loss():
    """Demonstrate ranking loss for SEO ranking tasks"""
    logger.info("\n=== Demonstrating Ranking Loss ===")
    
    # Create sample ranking data
    batch_size = 8
    scores = torch.randn(batch_size)
    labels = torch.tensor([3, 2, 1, 4, 2, 3, 1, 2])  # Ranking labels
    
    ranking_loss = RankingLoss(margin=1.0)
    loss = ranking_loss(scores, labels)
    
    logger.info(f"Ranking Loss: {loss.item():.4f}")
    logger.info(f"Scores: {scores.tolist()}")
    logger.info(f"Labels: {labels.tolist()}")

def demonstrate_contrastive_loss():
    """Demonstrate contrastive loss for similarity learning"""
    logger.info("\n=== Demonstrating Contrastive Loss ===")
    
    # Create sample embeddings and labels
    batch_size = 8
    embedding_dim = 128
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, 4, (batch_size,))  # 4 different classes
    
    contrastive_loss = ContrastiveLoss(margin=1.0, temperature=0.1)
    loss = contrastive_loss(embeddings, labels)
    
    logger.info(f"Contrastive Loss: {loss.item():.4f}")
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Labels: {labels.tolist()}")

def demonstrate_seo_specific_loss():
    """Demonstrate SEO-specific loss combining multiple objectives"""
    logger.info("\n=== Demonstrating SEO-Specific Loss ===")
    
    # Create sample outputs and targets for different SEO tasks
    batch_size = 8
    num_classes = 3
    
    outputs = {
        'classification': torch.randn(batch_size, num_classes),
        'ranking_scores': torch.randn(batch_size),
        'embeddings': torch.randn(batch_size, 128),
        'quality_scores': torch.randn(batch_size, 1)
    }
    
    targets = {
        'classification_targets': torch.randint(0, num_classes, (batch_size,)),
        'ranking_labels': torch.randint(1, 6, (batch_size,)),  # 1-5 ranking
        'similarity_labels': torch.randint(0, 4, (batch_size,)),  # 4 similarity classes
        'quality_targets': torch.rand(batch_size, 1)  # 0-1 quality scores
    }
    
    seo_loss = SEOSpecificLoss(
        classification_weight=1.0,
        ranking_weight=0.5,
        similarity_weight=0.3,
        content_quality_weight=0.2
    )
    
    loss = seo_loss(outputs, targets)
    logger.info(f"SEO-Specific Loss: {loss.item():.4f}")
    
    # Show individual components
    logger.info("Loss components:")
    logger.info(f"  Classification: {outputs['classification'].shape}")
    logger.info(f"  Ranking: {outputs['ranking_scores'].shape}")
    logger.info(f"  Similarity: {outputs['embeddings'].shape}")
    logger.info(f"  Quality: {outputs['quality_scores'].shape}")

def demonstrate_multi_task_loss():
    """Demonstrate multi-task loss for handling multiple SEO objectives"""
    logger.info("\n=== Demonstrating Multi-Task Loss ===")
    
    # Create sample data for multiple tasks
    batch_size = 8
    num_classes = 3
    
    outputs = {
        'content_classification': torch.randn(batch_size, num_classes),
        'sentiment_analysis': torch.randn(batch_size, 2),  # positive/negative
        'readability_score': torch.randn(batch_size, 1)
    }
    
    targets = {
        'content_classification': torch.randint(0, num_classes, (batch_size,)),
        'sentiment_analysis': torch.randint(0, 2, (batch_size,)),
        'readability_score': torch.rand(batch_size, 1)
    }
    
    # Create task-specific loss functions
    task_losses = {
        'content_classification': FocalLoss(alpha=1.0, gamma=2.0),
        'sentiment_analysis': nn.CrossEntropyLoss(),
        'readability_score': nn.MSELoss()
    }
    
    # Define task weights
    task_weights = {
        'content_classification': 1.0,
        'sentiment_analysis': 0.8,
        'readability_score': 0.5
    }
    
    multi_task_loss = MultiTaskLoss(task_weights, task_losses)
    loss = multi_task_loss(outputs, targets)
    
    logger.info(f"Multi-Task Loss: {loss.item():.4f}")
    logger.info(f"Task weights: {task_weights}")

def demonstrate_uncertainty_loss():
    """Demonstrate uncertainty-weighted loss for multi-task learning"""
    logger.info("\n=== Demonstrating Uncertainty Loss ===")
    
    # Create sample data for multiple tasks
    batch_size = 8
    num_tasks = 3
    
    outputs = [
        torch.randn(batch_size, 1),  # Task 1: regression
        torch.randn(batch_size, 1),  # Task 2: regression
        torch.randn(batch_size, 1)   # Task 3: regression
    ]
    
    targets = [
        torch.rand(batch_size, 1),  # Task 1 targets
        torch.rand(batch_size, 1),  # Task 2 targets
        torch.rand(batch_size, 1)   # Task 3 targets
    ]
    
    uncertainty_loss = UncertaintyLoss(num_tasks=num_tasks)
    loss = uncertainty_loss(outputs, targets)
    
    logger.info(f"Uncertainty Loss: {loss.item():.4f}")
    logger.info(f"Number of tasks: {num_tasks}")

def demonstrate_optimizers():
    """Demonstrate different optimizers"""
    logger.info("\n=== Demonstrating Optimizers ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Test different optimizers
    optimizer_configs = [
        ("Adam", OptimizerConfig(optimizer_type="adam", learning_rate=1e-3)),
        ("AdamW", OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3)),
        ("SGD", OptimizerConfig(optimizer_type="sgd", learning_rate=1e-2, momentum=0.9)),
        ("RMSprop", OptimizerConfig(optimizer_type="rmsprop", learning_rate=1e-3)),
        ("RAdam", OptimizerConfig(optimizer_type="radam", learning_rate=1e-3)),
    ]
    
    for opt_name, config in optimizer_configs:
        try:
            optimizer = AdvancedOptimizer.create_optimizer(model, config)
            logger.info(f"{opt_name} optimizer created successfully")
            
            # Test a forward pass
            inputs = torch.randn(8, 100)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, torch.randint(0, 10, (8,)))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.info(f"{opt_name} training step completed successfully")
            
        except Exception as e:
            logger.error(f"{opt_name} failed: {e}")

def demonstrate_schedulers():
    """Demonstrate different learning rate schedulers"""
    logger.info("\n=== Demonstrating Schedulers ===")
    
    # Create a simple model and optimizer
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test different schedulers
    scheduler_configs = [
        ("Step LR", SchedulerConfig(scheduler_type="step", step_size=10, gamma=0.1)),
        ("Cosine Annealing", SchedulerConfig(scheduler_type="cosine", T_max=100)),
        ("Cosine Warm Restarts", SchedulerConfig(scheduler_type="cosine_warm_restarts", T_0=10)),
        ("Reduce on Plateau", SchedulerConfig(scheduler_type="reduce_on_plateau", patience=5)),
        ("Exponential", SchedulerConfig(scheduler_type="exponential", gamma=0.95)),
    ]
    
    for sch_name, config in scheduler_configs:
        try:
            scheduler = AdvancedScheduler.create_scheduler(optimizer, config)
            logger.info(f"{sch_name} scheduler created successfully")
            
            # Test scheduler step
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"{sch_name} current LR: {current_lr:.6f}")
            
        except Exception as e:
            logger.error(f"{sch_name} failed: {e}")

def demonstrate_loss_function_manager():
    """Demonstrate loss function manager"""
    logger.info("\n=== Demonstrating Loss Function Manager ===")
    
    manager = LossFunctionManager()
    
    # Create different loss functions
    loss_configs = [
        ("Cross Entropy", LossConfig(loss_type="cross_entropy")),
        ("Focal Loss", LossConfig(loss_type="focal", alpha=1.0, gamma=2.0)),
        ("Label Smoothing", LossConfig(loss_type="label_smoothing", smoothing=0.1)),
        ("Ranking Loss", LossConfig(loss_type="ranking", margin=1.0)),
        ("Contrastive Loss", LossConfig(loss_type="contrastive", margin=1.0, temperature=0.1)),
        ("SEO Specific", LossConfig(loss_type="seo_specific")),
    ]
    
    for loss_name, config in loss_configs:
        try:
            loss_fn = manager.create_loss_function(config)
            logger.info(f"{loss_name} loss function created successfully")
            
            # Test the loss function
            batch_size = 8
            num_classes = 3
            inputs = torch.randn(batch_size, num_classes)
            targets = torch.randint(0, num_classes, (batch_size,))
            
            if loss_name == "SEO Specific":
                # For SEO specific loss, we need multiple outputs
                outputs = {
                    'classification': inputs,
                    'ranking_scores': torch.randn(batch_size),
                    'embeddings': torch.randn(batch_size, 128),
                    'quality_scores': torch.randn(batch_size, 1)
                }
                targets_dict = {
                    'classification_targets': targets,
                    'ranking_labels': torch.randint(1, 6, (batch_size,)),
                    'similarity_labels': torch.randint(0, 4, (batch_size,)),
                    'quality_targets': torch.rand(batch_size, 1)
                }
                loss = loss_fn(outputs, targets_dict)
            else:
                loss = loss_fn(inputs, targets)
            
            logger.info(f"{loss_name} loss value: {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"{loss_name} failed: {e}")

def demonstrate_custom_model_with_loss():
    """Demonstrate custom SEO model with advanced loss functions"""
    logger.info("\n=== Demonstrating Custom Model with Loss Functions ===")
    
    # Create custom model configuration
    config = CustomModelConfig(
        model_name="seo_model_with_loss",
        num_classes=3,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        dropout_rate=0.1,
        max_length=128,
        use_layer_norm=True,
        use_residual_connections=True,
        activation_function="gelu",
        initialization_method="orthogonal",
        gradient_checkpointing=False
    )
    
    # Create model
    model = create_custom_model(config)
    
    # Create sample data
    batch_size = 8
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    labels = torch.randint(0, 3, (batch_size,))
    
    # Test different loss functions
    loss_functions = [
        ("Cross Entropy", nn.CrossEntropyLoss()),
        ("Focal Loss", FocalLoss(alpha=1.0, gamma=2.0)),
        ("Label Smoothing", LabelSmoothingLoss(smoothing=0.1)),
    ]
    
    for loss_name, loss_fn in loss_functions:
        try:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
            
            loss = loss_fn(outputs, labels)
            logger.info(f"{loss_name} with custom model: {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"{loss_name} with custom model failed: {e}")

def demonstrate_training_with_loss_management():
    """Demonstrate training with comprehensive loss and optimization management"""
    logger.info("\n=== Demonstrating Training with Loss Management ===")
    
    # Create training configuration
    config = TrainingConfig(
        model_type="custom",
        model_name="seo_model_training",
        num_classes=3,
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=2,
        optimizer_type="adamw",
        scheduler_type="cosine",
        use_mixed_precision=True,
        use_gradient_checkpointing=False
    )
    
    # Create custom trainer
    trainer = CustomSEOModelTrainer(config)
    
    # Setup training (includes loss function creation)
    trainer.setup_training()
    
    # Create sample data
    dataloader = create_sample_data(batch_size=8, seq_length=128)
    
    # Train for a few epochs
    for epoch in range(2):
        logger.info(f"Training epoch {epoch+1}")
        results = trainer.train_epoch_with_monitoring(dataloader)
        
        logger.info(f"Epoch {epoch+1} results:")
        logger.info(f"  Loss: {results['loss']:.4f}")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        if 'gradient_stats' in results:
            logger.info(f"  Gradient stats: {results['gradient_stats']}")
    
    # Get comprehensive summary
    summary = trainer.get_autograd_summary()
    
    logger.info("=== Final Summary ===")
    logger.info(f"Model info: {summary['model_info']}")
    logger.info(f"Loss summary: {summary['loss_summary']}")

def main():
    """Main demonstration function"""
    logger.info("Starting Loss Functions and Optimization Demonstrations")
    
    try:
        # Demonstrate different loss functions
        demonstrate_loss_functions()
        
        # Demonstrate ranking loss
        demonstrate_ranking_loss()
        
        # Demonstrate contrastive loss
        demonstrate_contrastive_loss()
        
        # Demonstrate SEO-specific loss
        demonstrate_seo_specific_loss()
        
        # Demonstrate multi-task loss
        demonstrate_multi_task_loss()
        
        # Demonstrate uncertainty loss
        demonstrate_uncertainty_loss()
        
        # Demonstrate optimizers
        demonstrate_optimizers()
        
        # Demonstrate schedulers
        demonstrate_schedulers()
        
        # Demonstrate loss function manager
        demonstrate_loss_function_manager()
        
        # Demonstrate custom model with loss
        demonstrate_custom_model_with_loss()
        
        # Demonstrate training with loss management
        demonstrate_training_with_loss_management()
        
        logger.info("All loss functions and optimization demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise

match __name__:
    case "__main__":
    main() 