from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import sys
from core.gradient_management import (
from core.training_optimization import (
from typing import Any, List, Dict, Optional
"""
Gradient Management Example

Demonstrates comprehensive gradient clipping and NaN/Inf value handling
for stable training of email sequence models.
"""


# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

    GradientConfig,
    GradientManager,
    create_gradient_manager,
    safe_backward
)
    EarlyStoppingConfig,
    LRSchedulerConfig,
    GradientManagementConfig,
    TrainingOptimizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModel(nn.Module):
    """Test model that can generate problematic gradients for demonstration"""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 64, output_size: int = 1):
        
    """__init__ function."""
super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights to potentially cause issues
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights to demonstrate gradient issues"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize with large values to cause gradient explosion
                nn.init.normal_(layer.weight, mean=0.0, std=10.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x) -> Any:
        return self.layers(x)


class ProblematicModel(nn.Module):
    """Model designed to generate NaN/Inf gradients for testing"""
    
    def __init__(self, input_size: int = 50):
        
    """__init__ function."""
super().__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)
        
        # Initialize with problematic values
        self._initialize_problematic_weights()
    
    def _initialize_problematic_weights(self) -> Any:
        """Initialize weights to cause NaN/Inf issues"""
        # Set some weights to very large values
        with torch.no_grad():
            self.layer1.weight[0, 0] = float('inf')
            self.layer2.weight[0, 0] = float('nan')
    
    def forward(self, x) -> Any:
        # Add operations that can cause numerical issues
        x = self.layer1(x)
        x = torch.log(torch.abs(x) + 1e-8)  # Can cause issues with negative values
        x = self.layer2(x)
        x = torch.exp(x)  # Can cause overflow
        x = self.layer3(x)
        return x


async def demonstrate_basic_gradient_clipping():
    """Demonstrate basic gradient clipping"""
    
    logger.info("=== Basic Gradient Clipping Demo ===")
    
    # Create model and data
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create gradient manager
    config = GradientConfig(
        max_grad_norm=1.0,
        enable_gradient_clipping=True,
        enable_gradient_monitoring=True,
        verbose_logging=True
    )
    
    gradient_manager = GradientManager(config)
    
    # Generate some data
    x = torch.randn(32, 100)
    y = torch.randn(32, 1)
    
    # Training loop with gradient management
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass with gradient management
        step_info = gradient_manager.step(
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        logger.info(f"Step {step}: Loss={loss.item():.6f}, "
                   f"GradNorm={step_info['statistics']['total_norm']:.6f}, "
                   f"Clipped={step_info['clipping']['clipped']}")
        
        optimizer.step()
    
    # Get training summary
    summary = gradient_manager.get_training_summary()
    logger.info(f"Training Summary: {summary}")


async def demonstrate_nan_inf_handling():
    """Demonstrate NaN/Inf value handling"""
    
    logger.info("\n=== NaN/Inf Handling Demo ===")
    
    # Create problematic model
    model = ProblematicModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create gradient manager with NaN/Inf handling
    config = GradientConfig(
        enable_nan_inf_check=True,
        replace_nan_with=0.0,
        replace_inf_with=1e6,
        enable_gradient_monitoring=True,
        verbose_logging=True
    )
    
    gradient_manager = GradientManager(config)
    
    # Generate data
    x = torch.randn(16, 50)
    y = torch.randn(16, 1)
    
    # Training loop
    for step in range(5):
        optimizer.zero_grad()
        
        try:
            # Forward pass
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            # Backward pass with NaN/Inf handling
            step_info = gradient_manager.step(
                model=model,
                optimizer=optimizer,
                loss=loss
            )
            
            logger.info(f"Step {step}: Loss={loss.item():.6f}, "
                       f"NaN Count={step_info['nan_inf']['nan_count']}, "
                       f"Inf Count={step_info['nan_inf']['inf_count']}, "
                       f"Fixed={step_info['nan_inf']['fixed']}")
            
            optimizer.step()
            
        except Exception as e:
            logger.error(f"Error in step {step}: {e}")
            break
    
    # Get replacement summary
    replacement_summary = gradient_manager.nan_inf_handler.get_replacement_summary()
    logger.info(f"Replacement Summary: {replacement_summary}")


async def demonstrate_adaptive_clipping():
    """Demonstrate adaptive gradient clipping"""
    
    logger.info("\n=== Adaptive Gradient Clipping Demo ===")
    
    # Create model
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create gradient manager with adaptive clipping
    config = GradientConfig(
        adaptive_clipping=True,
        adaptive_window_size=5,
        adaptive_percentile=90.0,
        enable_gradient_monitoring=True,
        verbose_logging=True
    )
    
    gradient_manager = GradientManager(config)
    
    # Generate data
    x = torch.randn(32, 100)
    y = torch.randn(32, 1)
    
    # Training loop
    for step in range(15):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass with adaptive clipping
        step_info = gradient_manager.step(
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        logger.info(f"Step {step}: Loss={loss.item():.6f}, "
                   f"GradNorm={step_info['statistics']['total_norm']:.6f}, "
                   f"MaxNorm={step_info['clipping']['max_norm']:.6f}, "
                   f"Clipped={step_info['clipping']['clipped']}")
        
        optimizer.step()
    
    # Plot gradient statistics
    gradient_manager.plot_training_curves(save_path="adaptive_clipping_results.png")


async def demonstrate_gradient_health_monitoring():
    """Demonstrate gradient health monitoring"""
    
    logger.info("\n=== Gradient Health Monitoring Demo ===")
    
    # Create model
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create gradient manager with health monitoring
    config = GradientConfig(
        enable_gradient_monitoring=True,
        enable_gradient_clipping=True,
        max_grad_norm=2.0,
        verbose_logging=True
    )
    
    gradient_manager = GradientManager(config)
    
    # Generate data
    x = torch.randn(32, 100)
    y = torch.randn(32, 1)
    
    # Training loop with health monitoring
    for step in range(20):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass with health monitoring
        step_info = gradient_manager.step(
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        # Check health status
        health = step_info['health']
        if not health['healthy']:
            logger.warning(f"Step {step}: Gradient health issues detected!")
            for warning in health['warnings']:
                logger.warning(f"  - {warning}")
            for recommendation in health['recommendations']:
                logger.info(f"  Recommendation: {recommendation}")
        
        logger.info(f"Step {step}: Loss={loss.item():.6f}, "
                   f"Healthy={health['healthy']}, "
                   f"Warnings={len(health['warnings'])}")
        
        optimizer.step()
    
    # Get comprehensive health report
    summary = gradient_manager.get_training_summary()
    logger.info(f"Health Summary: {summary['health_issues']}")


async def demonstrate_integrated_training():
    """Demonstrate integrated training with all gradient management features"""
    
    logger.info("\n=== Integrated Training Demo ===")
    
    # Create configurations
    early_stopping_config = EarlyStoppingConfig(
        patience=5,
        min_delta=0.001,
        monitor="val_loss",
        mode="min"
    )
    
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type="cosine",
        initial_lr=0.01,
        min_lr=1e-6,
        T_max=50
    )
    
    gradient_config = GradientManagementConfig(
        enable_gradient_management=True,
        max_grad_norm=1.0,
        enable_nan_inf_check=True,
        enable_gradient_monitoring=True,
        verbose_logging=True
    )
    
    # Create model and optimizer
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create training optimizer
    training_optimizer = TrainingOptimizer(
        early_stopping_config=early_stopping_config,
        lr_scheduler_config=lr_scheduler_config,
        gradient_config=gradient_config,
        optimizer=optimizer
    )
    
    # Define training and validation functions
    async def train_epoch(model, epoch) -> Any:
        """Training function for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 10
        
        x = torch.randn(32, 100)
        y = torch.randn(32, 1)
        
        for batch in range(num_batches):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            # Backward pass with gradient management
            if training_optimizer.gradient_manager:
                step_info = training_optimizer.gradient_manager.step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss
                )
                
                if step_info and not step_info.get("skipped", False):
                    training_optimizer.gradient_history.append(step_info)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    async def validate_epoch(model, epoch) -> bool:
        """Validation function for one epoch"""
        model.eval()
        with torch.no_grad():
            x = torch.randn(32, 100)
            y = torch.randn(32, 1)
            output = model(x)
            loss = nn.MSELoss()(output, y)
            return loss.item()
    
    # Run training
    logger.info("Starting integrated training...")
    training_history = await training_optimizer.optimize_training(
        model=model,
        train_func=train_epoch,
        val_func=validate_epoch,
        max_epochs=20
    )
    
    # Get comprehensive report
    report = training_optimizer.get_optimization_report()
    logger.info(f"Training completed. Final report: {report}")
    
    # Plot results
    training_optimizer.plot_optimization_curves(save_path="integrated_training_results.png")
    
    return training_history


async def demonstrate_safe_backward_utility():
    """Demonstrate the safe_backward utility function"""
    
    logger.info("\n=== Safe Backward Utility Demo ===")
    
    # Create model and optimizer
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create gradient manager
    gradient_manager = create_gradient_manager(
        max_grad_norm=1.0,
        enable_monitoring=True,
        enable_nan_inf_check=True,
        verbose=True
    )
    
    # Generate data
    x = torch.randn(16, 100)
    y = torch.randn(16, 1)
    
    # Training loop using safe_backward
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # Use safe_backward utility
        step_info = safe_backward(
            loss=loss,
            model=model,
            optimizer=optimizer,
            gradient_manager=gradient_manager
        )
        
        logger.info(f"Step {step}: Loss={loss.item():.6f}, "
                   f"GradNorm={step_info['statistics']['total_norm']:.6f}, "
                   f"Healthy={step_info['health']['healthy']}")
        
        optimizer.step()
    
    # Get summary
    summary = gradient_manager.get_training_summary()
    logger.info(f"Safe Backward Summary: {summary}")


async def demonstrate_gradient_visualization():
    """Demonstrate gradient visualization capabilities"""
    
    logger.info("\n=== Gradient Visualization Demo ===")
    
    # Create model and gradient manager
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    config = GradientConfig(
        enable_gradient_monitoring=True,
        enable_gradient_clipping=True,
        max_grad_norm=1.0,
        save_gradient_plots=True
    )
    
    gradient_manager = GradientManager(config)
    
    # Generate data
    x = torch.randn(32, 100)
    y = torch.randn(32, 1)
    
    # Training loop to collect data
    for step in range(50):
        optimizer.zero_grad()
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        step_info = gradient_manager.step(
            model=model,
            optimizer=optimizer,
            loss=loss
        )
        
        optimizer.step()
    
    # Generate visualizations
    logger.info("Generating gradient visualizations...")
    
    # Plot gradient statistics
    gradient_manager.monitor.plot_gradient_statistics(save_path="gradient_statistics.png")
    
    # Plot training curves
    gradient_manager.plot_training_curves(save_path="gradient_training_curves.png")
    
    # Save training log
    gradient_manager.save_training_log("gradient_training_log.txt")
    
    logger.info("Visualizations saved to files")


async def main():
    """Run all gradient management demonstrations"""
    
    logger.info("Starting Gradient Management Demonstrations")
    logger.info("=" * 60)
    
    try:
        # Run individual demonstrations
        await demonstrate_basic_gradient_clipping()
        await demonstrate_nan_inf_handling()
        await demonstrate_adaptive_clipping()
        await demonstrate_gradient_health_monitoring()
        await demonstrate_safe_backward_utility()
        await demonstrate_gradient_visualization()
        
        # Run integrated training demonstration
        await demonstrate_integrated_training()
        
        logger.info("\n" + "=" * 60)
        logger.info("All gradient management demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


match __name__:
    case "__main__":
    asyncio.run(main()) 