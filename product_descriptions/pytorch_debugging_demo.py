from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import structlog
from pytorch_debugging_optimization import (
from error_handling_debugging import ErrorHandlingDebuggingSystem
from training_logging_system import TrainingLogger, create_training_logger
from robust_operations import RobustOperations
                import shutil
from typing import Any, List, Dict, Optional
import logging
"""
PyTorch Debugging and Optimization Demo

This demo showcases comprehensive PyTorch debugging and optimization:
- autograd.detect_anomaly() for gradient debugging
- Memory profiling and leak detection
- Performance optimization with AMP and compilation
- Integration with robust operations and training logging
- Real-world cybersecurity ML scenarios
"""



# Add the current directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

    PyTorchDebugger,
    PyTorchOptimizer,
    DebugMode,
    OptimizationMode,
    debug_operation,
    optimize_model
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class CybersecurityModel(nn.Module):
    """Neural network for cybersecurity threat detection with potential issues."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_classes: int = 2):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Potential issue: Initialize weights with very large values
        self._initialize_weights()
        
    def _initialize_weights(self) -> Any:
        """Initialize weights with potential issues for debugging."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Potential issue: Very large initialization
                nn.init.normal_(m.weight, mean=0, std=10.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x) -> Any:
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class ProblematicModel(nn.Module):
    """Model with intentional issues for debugging demonstration."""
    
    def __init__(self, input_size: int = 10, num_classes: int = 2):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        
    def forward(self, x) -> Any:
        # Potential issue: Return NaN or Inf values
        if random.random() < 0.1:  # 10% chance of issue
            return torch.tensor([[float('nan'), float('inf')]])
        
        x = self.fc1(x)
        return F.softmax(x, dim=1)


class PyTorchDebuggingDemo:
    """Demo class showcasing PyTorch debugging and optimization."""
    
    def __init__(self) -> Any:
        self.config = {
            "log_dir": "demo_debug_logs",
            "log_level": "INFO",
            "enable_console": True,
            "enable_file": True,
            "enable_rich": True
        }
        
        # Initialize systems
        self.error_system = ErrorHandlingDebuggingSystem({
            "max_errors": 1000,
            "enable_persistence": True,
            "enable_profiling": True
        })
        
        self.training_logger = create_training_logger(self.config)
        
        self.robust_ops = RobustOperations({
            "max_errors": 1000,
            "enable_persistence": True,
            "enable_profiling": True
        })
        
        # Create debugger
        self.debugger = PyTorchDebugger(
            error_system=self.error_system,
            training_logger=self.training_logger,
            debug_mode=DebugMode.FULL_DEBUG
        )
        
        # Create optimizer
        self.optimizer = PyTorchOptimizer(self.debugger)
        
        # Create demo data directory
        self.demo_dir = Path("demo_data")
        self.demo_dir.mkdir(exist_ok=True)
        
        logger.info("PyTorchDebuggingDemo initialized", config=self.config)
    
    def create_demo_data(self) -> Tuple[DataLoader, DataLoader]:
        """Create demo cybersecurity data."""
        logger.info("Creating demo cybersecurity data")
        
        # Generate synthetic cybersecurity data
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        # Normal traffic features
        normal_data = np.random.normal(0, 1, (n_samples // 2, n_features))
        normal_labels = np.zeros(n_samples // 2)
        
        # Malicious traffic features (anomalies)
        malicious_data = np.random.normal(2, 1.5, (n_samples // 2, n_features))
        malicious_labels = np.ones(n_samples // 2)
        
        # Combine data
        X = np.vstack([normal_data, malicious_data])
        y = np.hstack([normal_labels, malicious_labels])
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Split into train and validation
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info("Demo data created", 
                   train_samples=len(train_dataset),
                   val_samples=len(val_dataset),
                   features=n_features)
        
        return train_loader, val_loader
    
    def demo_anomaly_detection(self) -> Any:
        """Demonstrate autograd.detect_anomaly() functionality."""
        logger.info("=== Anomaly Detection Demo ===")
        
        # Create problematic model
        model = ProblematicModel(input_size=10, num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create data
        data = torch.randn(16, 10)
        target = torch.randint(0, 2, (16,))
        
        # Test with anomaly detection enabled
        logger.info("Testing with anomaly detection enabled")
        
        try:
            with self.debugger.debug_context("anomaly_detection_test"):
                for i in range(5):
                    optimizer.zero_grad()
                    
                    # Forward pass (may produce NaN/Inf)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Backward pass (anomaly detection will catch issues)
                    loss.backward()
                    optimizer.step()
                    
                    logger.info(f"Step {i+1}: Loss = {loss.item():.4f}")
                    
        except Exception as e:
            logger.info(f"Anomaly detected: {str(e)}")
            
            # Log the error with debug context
            self.debugger.error_system.error_tracker.track_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.MODEL,
                context={"operation": "anomaly_detection_test"}
            )
    
    def demo_gradient_checking(self) -> Any:
        """Demonstrate gradient checking and analysis."""
        logger.info("=== Gradient Checking Demo ===")
        
        # Create model with potential gradient issues
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create data
        data = torch.randn(32, 10)
        target = torch.randint(0, 2, (32,))
        
        logger.info("Testing gradient checking")
        
        for epoch in range(3):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            gradient_info = self.debugger.check_gradients(model, loss)
            
            logger.info(f"Epoch {epoch+1}:")
            logger.info(f"  Loss: {loss.item():.4f}")
            logger.info(f"  Gradient norm: {gradient_info['gradient_norm']:.4f}")
            
            if gradient_info["gradient_anomalies"]:
                logger.warning(f"  Gradient anomalies: {gradient_info['gradient_anomalies']}")
            
            # Log gradient statistics
            if self.training_logger:
                self.training_logger.log_training_event(
                    "gradient_check",
                    f"Gradient check for epoch {epoch+1}",
                    level=LogLevel.DEBUG,
                    gradient_info=gradient_info
                )
            
            optimizer.step()
    
    def demo_memory_profiling(self) -> Any:
        """Demonstrate memory profiling and leak detection."""
        logger.info("=== Memory Profiling Demo ===")
        
        # Create model
        model = CybersecurityModel(input_size=10, hidden_size=64, num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create data
        data = torch.randn(64, 10)
        target = torch.randint(0, 2, (64,))
        
        logger.info("Testing memory profiling")
        
        # Simulate training with memory profiling
        for batch in range(10):
            # Profile memory before operation
            memory_before = self.debugger.profile_memory(f"before_batch_{batch}")
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Profile memory after operation
            memory_after = self.debugger.profile_memory(f"after_batch_{batch}")
            
            logger.info(f"Batch {batch+1}:")
            logger.info(f"  Loss: {loss.item():.4f}")
            logger.info(f"  CPU Memory: {memory_after['cpu_memory']:.1f}%")
            
            if memory_after["gpu_memory"]:
                logger.info(f"  GPU Memory: {memory_after['gpu_memory']:.1f}MB")
            
            if memory_after["memory_leaks"]:
                logger.warning(f"  Memory leaks: {memory_after['memory_leaks']}")
            
            # Log memory metrics
            if self.training_logger:
                self.training_logger.log_training_event(
                    "memory_profile",
                    f"Memory profile for batch {batch+1}",
                    level=LogLevel.DEBUG,
                    memory_before=memory_before,
                    memory_after=memory_after
                )
    
    def demo_performance_optimization(self) -> Any:
        """Demonstrate performance optimization techniques."""
        logger.info("=== Performance Optimization Demo ===")
        
        # Create model and data
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        train_loader, val_loader = self.create_demo_data()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Test different optimization modes
        optimization_modes = [
            OptimizationMode.NONE,
            OptimizationMode.AMP,
            OptimizationMode.COMPILATION,
            OptimizationMode.MEMORY_EFFICIENT,
            OptimizationMode.FULL_OPTIMIZATION
        ]
        
        for mode in optimization_modes:
            logger.info(f"Testing {mode.value} optimization")
            
            try:
                # Create fresh model for each test
                test_model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
                test_optimizer = optim.Adam(test_model.parameters(), lr=0.001)
                
                # Run optimization test
                result = self.optimizer.optimize_training_loop(
                    model=test_model,
                    dataloader=train_loader,
                    optimizer=test_optimizer,
                    criterion=criterion,
                    epochs=1,
                    optimization_mode=mode
                )
                
                logger.info(f"  {mode.value} results:")
                logger.info(f"    Total time: {result['total_time']:.2f}s")
                logger.info(f"    Training metrics: {len(result['training_metrics'])} batches")
                logger.info(f"    Model optimized: {result['model_optimized']}")
                
                if 'speedup_factor' in result['optimization_metrics']:
                    logger.info(f"    Speedup factor: {result['optimization_metrics']['speedup_factor']:.2f}x")
                
            except Exception as e:
                logger.error(f"  {mode.value} failed: {str(e)}")
                
                self.debugger.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.MODEL,
                    context={"optimization_mode": mode.value}
                )
    
    def demo_benchmark_optimizations(self) -> Any:
        """Demonstrate benchmarking of different optimization modes."""
        logger.info("=== Optimization Benchmarking Demo ===")
        
        # Create model and data
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        train_loader, val_loader = self.create_demo_data()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        logger.info("Running optimization benchmarks")
        
        try:
            benchmark_results = self.optimizer.benchmark_optimizations(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=1
            )
            
            # Analyze results
            logger.info("Benchmark Results:")
            for mode, result in benchmark_results.items():
                if "error" in result:
                    logger.error(f"  {mode}: {result['error']}")
                else:
                    logger.info(f"  {mode}:")
                    logger.info(f"    Time: {result['total_time']:.2f}s")
                    logger.info(f"    Batches: {len(result['training_metrics'])}")
                    logger.info(f"    Model optimized: {result['model_optimized']}")
                    
                    if 'speedup_factor' in result['optimization_metrics']:
                        logger.info(f"    Speedup: {result['optimization_metrics']['speedup_factor']:.2f}x")
            
            # Save benchmark results
            benchmark_file = self.demo_dir / "optimization_benchmark.json"
            with open(benchmark_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(benchmark_results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {benchmark_file}")
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {str(e)}")
    
    def demo_integration_with_robust_ops(self) -> Any:
        """Demonstrate integration with robust operations."""
        logger.info("=== Robust Operations Integration Demo ===")
        
        # Create model and data
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        train_loader, val_loader = self.create_demo_data()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        logger.info("Testing integration with robust operations")
        
        try:
            # Use robust operations for model inference
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 5:  # Test first 5 batches
                    break
                
                # Robust model inference with debugging
                with self.debugger.debug_context(f"robust_inference_{batch_idx}"):
                    result = self.robust_ops.model_inference.safe_inference(
                        model=model,
                        input_data=data,
                        device=torch.device('cpu'),
                        max_retries=2
                    )
                    
                    if result.success:
                        output = result.data
                        loss = criterion(output, target)
                        
                        logger.info(f"Batch {batch_idx+1}: Loss = {loss.item():.4f}")
                        
                        # Check gradients
                        loss.backward()
                        gradient_info = self.debugger.check_gradients(model, loss)
                        
                        if gradient_info["gradient_anomalies"]:
                            logger.warning(f"Gradient anomalies in batch {batch_idx+1}")
                        
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        logger.error(f"Model inference failed for batch {batch_idx+1}: {result.error_message}")
        
        except Exception as e:
            logger.error(f"Integration test failed: {str(e)}")
    
    def demo_decorator_usage(self) -> Any:
        """Demonstrate usage of debugging and optimization decorators."""
        logger.info("=== Decorator Usage Demo ===")
        
        # Create model and data
        model = CybersecurityModel(input_size=10, hidden_size=32, num_classes=2)
        data = torch.randn(32, 10)
        target = torch.randint(0, 2, (32,))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        @debug_operation(debug_mode=DebugMode.ANOMALY_DETECTION)
        def training_step(model, data, target, optimizer, criterion) -> Any:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        
        @optimize_model(optimization_mode=OptimizationMode.AMP)
        def optimized_training_step(model, data, target, optimizer, criterion) -> Any:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        
        logger.info("Testing debug decorator")
        try:
            loss = training_step(model, data, target, optimizer, criterion)
            logger.info(f"Debug training step completed: Loss = {loss.item():.4f}")
        except Exception as e:
            logger.info(f"Debug training step failed: {str(e)}")
        
        logger.info("Testing optimization decorator")
        try:
            loss = optimized_training_step(model, data, target, optimizer, criterion)
            logger.info(f"Optimized training step completed: Loss = {loss.item():.4f}")
        except Exception as e:
            logger.info(f"Optimized training step failed: {str(e)}")
    
    def demo_error_handling_and_recovery(self) -> Any:
        """Demonstrate error handling and recovery with debugging."""
        logger.info("=== Error Handling and Recovery Demo ===")
        
        # Create problematic model
        model = ProblematicModel(input_size=10, num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create data
        data = torch.randn(32, 10)
        target = torch.randint(0, 2, (32,))
        
        logger.info("Testing error handling and recovery")
        
        for batch in range(10):
            try:
                with self.debugger.debug_context(f"error_handling_batch_{batch}"):
                    optimizer.zero_grad()
                    
                    # Forward pass (may fail)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    logger.info(f"Batch {batch+1}: Success - Loss = {loss.item():.4f}")
                    
            except Exception as e:
                logger.warning(f"Batch {batch+1}: Failed - {str(e)}")
                
                # Log error with debug context
                self.debugger.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.MODEL,
                    context={
                        "operation": "error_handling_batch",
                        "batch": batch,
                        "debug_mode": self.debugger.debug_mode.value
                    }
                )
                
                # Continue with next batch
                continue
    
    def demo_performance_analysis(self) -> Any:
        """Demonstrate performance analysis and reporting."""
        logger.info("=== Performance Analysis Demo ===")
        
        # Get debug summary
        debug_summary = self.debugger.get_debug_summary()
        
        logger.info("Debug Summary:")
        logger.info(f"  Debug mode: {debug_summary['debug_mode']}")
        logger.info(f"  Total operations: {debug_summary['total_operations']}")
        logger.info(f"  Average execution time: {debug_summary['avg_execution_time']:.4f}s")
        logger.info(f"  Maximum execution time: {debug_summary['max_execution_time']:.4f}s")
        logger.info(f"  Average memory usage: {debug_summary['avg_memory_usage']:.2f}MB")
        logger.info(f"  Maximum memory usage: {debug_summary['max_memory_usage']:.2f}MB")
        
        if 'total_optimizations' in debug_summary:
            logger.info(f"  Total optimizations: {debug_summary['total_optimizations']}")
            logger.info(f"  Average optimization time: {debug_summary['avg_optimization_time']:.4f}s")
            logger.info(f"  Optimization modes used: {debug_summary['optimization_modes_used']}")
        
        # Save debug summary
        summary_file = self.demo_dir / "debug_summary.json"
        with open(summary_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(debug_summary, f, indent=2, default=str)
        
        logger.info(f"Debug summary saved to {summary_file}")
        
        # Generate performance report
        if self.debugger.debug_metrics:
            performance_data = []
            for metrics in self.debugger.debug_metrics:
                performance_data.append({
                    "execution_time": metrics.execution_time,
                    "memory_usage": metrics.memory_usage,
                    "gpu_memory": metrics.gpu_memory,
                    "timestamp": metrics.timestamp
                })
            
            # Create performance DataFrame
            df = pd.DataFrame(performance_data)
            
            # Save performance data
            performance_file = self.demo_dir / "performance_data.csv"
            df.to_csv(performance_file, index=False)
            
            logger.info(f"Performance data saved to {performance_file}")
            
            # Log performance statistics
            logger.info("Performance Statistics:")
            logger.info(f"  Execution time - Mean: {df['execution_time'].mean():.4f}s, Std: {df['execution_time'].std():.4f}s")
            logger.info(f"  Memory usage - Mean: {df['memory_usage'].mean():.2f}MB, Std: {df['memory_usage'].std():.2f}MB")
            
            if 'gpu_memory' in df.columns and df['gpu_memory'].notna().any():
                logger.info(f"  GPU memory - Mean: {df['gpu_memory'].mean():.2f}MB, Std: {df['gpu_memory'].std():.2f}MB")
    
    def run_comprehensive_demo(self) -> Any:
        """Run the complete PyTorch debugging and optimization demo."""
        logger.info("Starting Comprehensive PyTorch Debugging and Optimization Demo")
        
        try:
            # Run all demos
            self.demo_anomaly_detection()
            time.sleep(1)
            
            self.demo_gradient_checking()
            time.sleep(1)
            
            self.demo_memory_profiling()
            time.sleep(1)
            
            self.demo_performance_optimization()
            time.sleep(1)
            
            self.demo_benchmark_optimizations()
            time.sleep(1)
            
            self.demo_integration_with_robust_ops()
            time.sleep(1)
            
            self.demo_decorator_usage()
            time.sleep(1)
            
            self.demo_error_handling_and_recovery()
            time.sleep(1)
            
            self.demo_performance_analysis()
            
            logger.info("Comprehensive demo completed successfully")
            
        except Exception as e:
            logger.error("Demo failed", error=str(e))
        
        finally:
            # Cleanup
            self.debugger.cleanup()
            self.training_logger.cleanup()
            self.robust_ops.cleanup()
            
            # Clean up demo files
            try:
                shutil.rmtree(self.demo_dir)
                logger.info("Demo cleanup completed")
            except Exception as e:
                logger.warning("Demo cleanup failed", error=str(e))


async def main():
    """Main function to run the demo."""
    demo = PyTorchDebuggingDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 