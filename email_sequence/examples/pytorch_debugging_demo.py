from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from pathlib import Path
import json
from core.pytorch_debugging import (
from core.training_logger import create_training_logger
from core.enhanced_training_optimizer import create_enhanced_training_optimizer
        import traceback
from typing import Any, List, Dict, Optional
import logging
"""
PyTorch Debugging Tools Demonstration

Comprehensive demonstration of PyTorch's built-in debugging tools
including autograd.detect_anomaly(), profiler, and other debugging utilities.
"""


    PyTorchDebugger, create_pytorch_debugger, debug_training_session
)


class DebugTestModel(nn.Module):
    """Test model with potential debugging issues"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 2):
        
    """__init__ function."""
super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Normal layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # Layer that might cause issues
        self.problematic_layer = nn.Linear(hidden_size, hidden_size)
        
        # Initialize with potential issues
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights with potential issues for debugging"""
        
        # Normal initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        
        # Problematic initialization (very large weights)
        nn.init.normal_(self.problematic_layer.weight, mean=0, std=10.0)
        nn.init.zeros_(self.problematic_layer.bias)
    
    def forward(self, x) -> Any:
        """Forward pass with potential debugging issues"""
        
        # Normal forward pass
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        
        # Potentially problematic operation
        x = self.problematic_layer(x)
        
        # Check for NaN/Inf and handle
        if torch.isnan(x).any() or torch.isinf(x).any():
            # Replace NaN/Inf with zeros
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        x = torch.relu(self.linear3(x))
        
        return x


class GradientExplosionModel(nn.Module):
    """Model designed to demonstrate gradient explosion"""
    
    def __init__(self, input_size: int = 10, output_size: int = 2):
        
    """__init__ function."""
super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
        # Initialize with very large weights to cause gradient explosion
        nn.init.normal_(self.linear.weight, mean=0, std=100.0)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x) -> Any:
        return self.linear(x)


class MemoryLeakModel(nn.Module):
    """Model designed to demonstrate memory issues"""
    
    def __init__(self, input_size: int = 10, output_size: int = 2):
        
    """__init__ function."""
super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.cache = []  # This will cause memory leaks
    
    def forward(self, x) -> Any:
        # Store intermediate results (simulating memory leak)
        self.cache.append(x.detach().clone())
        
        # Keep only last 10 items to prevent infinite growth
        if len(self.cache) > 10:
            self.cache = self.cache[-10:]
        
        return self.linear(x)


def create_test_data(num_samples: int = 1000, input_size: int = 10, num_classes: int = 2):
    """Create test data for debugging"""
    
    # Generate random data
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Add some problematic data (NaN, Inf)
    if num_samples > 100:
        # Add some NaN values
        X[50:60, 0] = float('nan')
        # Add some Inf values
        X[60:70, 1] = float('inf')
    
    return X, y


def demonstrate_anomaly_detection():
    """Demonstrate autograd anomaly detection"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING AUTOGRAD ANOMALY DETECTION")
    print("="*60)
    
    # Create debugger
    debugger = create_pytorch_debugger(
        debug_mode=True,
        enable_anomaly_detection=True,
        enable_memory_tracking=True,
        enable_gradient_checking=True
    )
    
    # Create model that might cause issues
    model = GradientExplosionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create data
    X, y = create_test_data(100, 10, 2)
    
    print("Training with anomaly detection enabled...")
    
    try:
        with debugger.anomaly_detection():
            with debugger.memory_tracking():
                for epoch in range(5):
                    print(f"Epoch {epoch + 1}/5")
                    
                    # Forward pass
                    outputs = model(X)
                    loss = loss_fn(outputs, y)
                    
                    # Backward pass (this might trigger anomaly detection)
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Check gradients
                    gradient_norm = debugger.check_gradients(model, gradient_threshold=1.0)
                    print(f"  Gradient norm: {gradient_norm:.6f}")
                    
                    # Optimizer step
                    optimizer.step()
                    
                    print(f"  Loss: {loss.item():.6f}")
                    
    except Exception as e:
        print(f"Anomaly detected: {e}")
        print("This is expected behavior for demonstration purposes")
    
    # Save debug report
    debugger.save_debug_report("anomaly_detection_report.json")
    print("Anomaly detection demonstration completed")


def demonstrate_profiling():
    """Demonstrate PyTorch profiling"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING PYTORCH PROFILING")
    print("="*60)
    
    # Create debugger
    debugger = create_pytorch_debugger(
        debug_mode=True,
        enable_profiling=True,
        enable_memory_tracking=True
    )
    
    # Create model
    model = DebugTestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create data
    X, y = create_test_data(500, 10, 2)
    
    print("Training with profiling enabled...")
    
    with debugger.profiling(output_file="training_profile.json"):
        for epoch in range(3):
            print(f"Epoch {epoch + 1}/3")
            
            # Forward pass
            outputs = model(X)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  Loss: {loss.item():.6f}")
    
    print("Profiling demonstration completed")
    print("Check 'training_profile.json' for profiling results")


def demonstrate_memory_tracking():
    """Demonstrate memory tracking"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING MEMORY TRACKING")
    print("="*60)
    
    # Create debugger
    debugger = create_pytorch_debugger(
        debug_mode=True,
        enable_memory_tracking=True,
        enable_gradient_checking=True
    )
    
    # Create model with potential memory issues
    model = MemoryLeakModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create data
    X, y = create_test_data(200, 10, 2)
    
    print("Training with memory tracking enabled...")
    
    with debugger.memory_tracking():
        for epoch in range(5):
            print(f"Epoch {epoch + 1}/5")
            
            # Forward pass
            outputs = model(X)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  Loss: {loss.item():.6f}")
            
            # Check model parameters
            param_stats = debugger.check_model_parameters(model)
            print(f"  Total parameters: {param_stats.get('total_parameters', 0)}")
    
    print("Memory tracking demonstration completed")


def demonstrate_gradient_checking():
    """Demonstrate gradient checking"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING GRADIENT CHECKING")
    print("="*60)
    
    # Create debugger
    debugger = create_pytorch_debugger(
        debug_mode=True,
        enable_gradient_checking=True,
        enable_memory_tracking=True
    )
    
    # Create model
    model = DebugTestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create data
    X, y = create_test_data(100, 10, 2)
    
    print("Training with gradient checking enabled...")
    
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        
        # Forward pass
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        gradient_norm = debugger.check_gradients(model, gradient_threshold=1.0)
        print(f"  Gradient norm: {gradient_norm:.6f}")
        
        # Check model parameters
        param_stats = debugger.check_model_parameters(model)
        print(f"  NaN parameters: {param_stats.get('nan_parameters', 0)}")
        print(f"  Inf parameters: {param_stats.get('inf_parameters', 0)}")
        
        optimizer.step()
        print(f"  Loss: {loss.item():.6f}")
    
    print("Gradient checking demonstration completed")


def demonstrate_forward_backward_debugging():
    """Demonstrate forward and backward pass debugging"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING FORWARD/BACKWARD DEBUGGING")
    print("="*60)
    
    # Create debugger
    debugger = create_pytorch_debugger(
        debug_mode=True,
        enable_gradient_checking=True
    )
    
    # Create model
    model = DebugTestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create data
    X, y = create_test_data(50, 10, 2)
    
    print("Training with forward/backward debugging...")
    
    for batch in range(3):
        print(f"Batch {batch + 1}/3")
        
        # Debug forward pass
        outputs = debugger.debug_forward_pass(model, X, layer_hooks=True)
        
        # Calculate loss
        loss = loss_fn(outputs, y)
        
        # Debug backward pass
        debugger.debug_backward_pass(loss)
        
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Loss: {loss.item():.6f}")
    
    print("Forward/backward debugging demonstration completed")


async def demonstrate_training_session():
    """Demonstrate complete training session with debugging"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING COMPLETE TRAINING SESSION WITH DEBUGGING")
    print("="*60)
    
    # Create logger
    logger = create_training_logger(
        experiment_name="debugging_demo",
        log_dir="debug_logs",
        log_level="DEBUG"
    )
    
    # Create model
    model = DebugTestModel()
    
    # Create data
    X_train, y_train = create_test_data(500, 10, 2)
    X_val, y_val = create_test_data(100, 10, 2)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create training optimizer with debugging
    optimizer = create_enhanced_training_optimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="debugging_training_demo",
        debug_mode=True,
        enable_pytorch_debugging=True,
        max_epochs=3,
        learning_rate=0.001,
        gradient_clip=1.0
    )
    
    print("Starting training session with debugging...")
    
    try:
        results = await optimizer.train()
        print(f"Training completed: {results}")
        
        # Get training summary
        summary = optimizer.get_training_summary()
        print(f"Training summary: {json.dumps(summary, indent=2)}")
        
    except Exception as e:
        print(f"Training error: {e}")
        logger.log_error(e, "Training session", "demonstrate_training_session")
    
    finally:
        optimizer.cleanup()
        logger.cleanup()
    
    print("Training session demonstration completed")


def demonstrate_debug_summary():
    """Demonstrate debug summary and reporting"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING DEBUG SUMMARY AND REPORTING")
    print("="*60)
    
    # Create debugger
    debugger = create_pytorch_debugger(
        debug_mode=True,
        enable_anomaly_detection=True,
        enable_memory_tracking=True,
        enable_gradient_checking=True
    )
    
    # Create model and run some operations
    model = DebugTestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    X, y = create_test_data(100, 10, 2)
    
    # Run some training steps
    for epoch in range(2):
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        debugger.check_gradients(model)
        debugger.check_model_parameters(model)
        
        optimizer.step()
    
    # Get debug summary
    summary = debugger.get_debug_summary()
    print("Debug Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save debug report
    report_path = debugger.save_debug_report("comprehensive_debug_report.json")
    print(f"\nDebug report saved to: {report_path}")
    
    # Clear debug data
    debugger.clear_debug_data()
    print("Debug data cleared")


def main():
    """Run all debugging demonstrations"""
    
    print("PYTORCH DEBUGGING TOOLS DEMONSTRATION")
    print("="*60)
    print("This demonstration shows various PyTorch debugging tools:")
    print("1. Autograd anomaly detection")
    print("2. PyTorch profiling")
    print("3. Memory tracking")
    print("4. Gradient checking")
    print("5. Forward/backward pass debugging")
    print("6. Complete training session with debugging")
    print("7. Debug summary and reporting")
    print("="*60)
    
    try:
        # Run demonstrations
        demonstrate_anomaly_detection()
        demonstrate_profiling()
        demonstrate_memory_tracking()
        demonstrate_gradient_checking()
        demonstrate_forward_backward_debugging()
        
        # Run async training session
        asyncio.run(demonstrate_training_session())
        
        demonstrate_debug_summary()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- anomaly_detection_report.json")
        print("- training_profile.json")
        print("- comprehensive_debug_report.json")
        print("- debug_logs/ (directory with training logs)")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        traceback.print_exc()


match __name__:
    case "__main__":
    main() 