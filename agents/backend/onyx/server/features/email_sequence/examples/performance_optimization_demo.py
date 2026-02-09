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

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
from pathlib import Path
import psutil
import GPUtil
from core.performance_optimizer import (
from core.optimized_training_optimizer import (
from core.training_logger import create_training_logger
        import traceback
from typing import Any, List, Dict, Optional
import logging
"""
Performance Optimization Demonstration

Comprehensive demonstration of performance optimization techniques
including memory optimization, computational efficiency, and training acceleration.
"""


    PerformanceOptimizer, PerformanceConfig, create_performance_optimizer,
    get_optimal_batch_size, benchmark_model_performance
)
    OptimizedTrainingOptimizer, create_optimized_training_optimizer,
    train_model_with_optimization
)


class PerformanceTestModel(nn.Module):
    """Test model for performance optimization demonstration"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 100, output_size: int = 2):
        
    """__init__ function."""
super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Multiple layers to test optimization
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x) -> Any:
        return self.layers(x)


class MemoryIntensiveModel(nn.Module):
    """Model designed to test memory optimization"""
    
    def __init__(self, input_size: int = 10, output_size: int = 2):
        
    """__init__ function."""
super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Large layers to consume memory
        self.large_layer1 = nn.Linear(input_size, 1000)
        self.large_layer2 = nn.Linear(1000, 1000)
        self.large_layer3 = nn.Linear(1000, 1000)
        self.output_layer = nn.Linear(1000, output_size)
        
        # Store intermediate activations (simulating memory usage)
        self.activations = []
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x) -> Any:
        # Store activations to simulate memory usage
        x1 = torch.relu(self.large_layer1(x))
        self.activations.append(x1.detach().clone())
        
        x2 = torch.relu(self.large_layer2(x1))
        self.activations.append(x2.detach().clone())
        
        x3 = torch.relu(self.large_layer3(x2))
        self.activations.append(x3.detach().clone())
        
        # Keep only last 5 activations to prevent memory explosion
        if len(self.activations) > 5:
            self.activations = self.activations[-5:]
        
        return self.output_layer(x3)


def create_test_data(num_samples: int = 1000, input_size: int = 10, num_classes: int = 2):
    """Create test data for performance testing"""
    
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


def demonstrate_memory_optimization():
    """Demonstrate memory optimization techniques"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING MEMORY OPTIMIZATION")
    print("="*60)
    
    # Create memory-intensive model
    model = MemoryIntensiveModel()
    
    # Create performance optimizer with memory optimizations
    optimizer = create_performance_optimizer(
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        enable_memory_efficient_attention=True,
        enable_compile=True
    )
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model)
    
    # Create test data
    X, y = create_test_data(500, 10, 2)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Benchmark before optimization
    print("Benchmarking before optimization...")
    before_benchmark = benchmark_model_performance(model, dataloader, num_iterations=50)
    print(f"Before optimization: {json.dumps(before_benchmark, indent=2)}")
    
    # Benchmark after optimization
    print("Benchmarking after optimization...")
    after_benchmark = benchmark_model_performance(optimized_model, dataloader, num_iterations=50)
    print(f"After optimization: {json.dumps(after_benchmark, indent=2)}")
    
    # Calculate improvement
    if before_benchmark and after_benchmark:
        throughput_improvement = (
            (after_benchmark["throughput"] - before_benchmark["throughput"]) / 
            before_benchmark["throughput"] * 100
        )
        print(f"Throughput improvement: {throughput_improvement:.2f}%")
    
    print("Memory optimization demonstration completed")


def demonstrate_computational_optimization():
    """Demonstrate computational optimization techniques"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING COMPUTATIONAL OPTIMIZATION")
    print("="*60)
    
    # Create test model
    model = PerformanceTestModel()
    
    # Create performance optimizer with computational optimizations
    optimizer = create_performance_optimizer(
        enable_compile=True,
        enable_torch_optimization=True,
        enable_cudnn_benchmark=True,
        enable_fused_optimizers=True
    )
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model)
    
    # Create test data
    X, y = create_test_data(1000, 10, 2)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Benchmark performance
    print("Benchmarking computational optimizations...")
    benchmark_results = benchmark_model_performance(optimized_model, dataloader, num_iterations=100)
    print(f"Computational optimization results: {json.dumps(benchmark_results, indent=2)}")
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128, 256]
    batch_performance = {}
    
    for batch_size in batch_sizes:
        try:
            test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            results = benchmark_model_performance(optimized_model, test_dataloader, num_iterations=50)
            batch_performance[batch_size] = results["throughput"]
            print(f"Batch size {batch_size}: {results['throughput']:.2f} samples/s")
        except RuntimeError as e:
            print(f"Batch size {batch_size}: Out of memory")
            break
    
    print("Computational optimization demonstration completed")


def demonstrate_data_loading_optimization():
    """Demonstrate data loading optimization"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING DATA LOADING OPTIMIZATION")
    print("="*60)
    
    # Create test data
    X, y = create_test_data(5000, 10, 2)
    dataset = TensorDataset(X, y)
    
    # Test different DataLoader configurations
    configs = [
        {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        {"num_workers": 2, "pin_memory": True, "persistent_workers": False},
        {"num_workers": 4, "pin_memory": True, "persistent_workers": True},
        {"num_workers": 8, "pin_memory": True, "persistent_workers": True}
    ]
    
    model = PerformanceTestModel()
    
    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}: {config}")
        
        # Create DataLoader with configuration
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            **config
        )
        
        # Benchmark performance
        start_time = time.time()
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= 50:  # Test 50 batches
                break
            total_samples += inputs.size(0)
        
        end_time = time.time()
        throughput = total_samples / (end_time - start_time)
        
        print(f"  Throughput: {throughput:.2f} samples/s")
        print(f"  Total time: {end_time - start_time:.2f}s")
    
    print("Data loading optimization demonstration completed")


def demonstrate_optimal_batch_size():
    """Demonstrate optimal batch size calculation"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING OPTIMAL BATCH SIZE CALCULATION")
    print("="*60)
    
    # Create test model
    model = PerformanceTestModel()
    
    # Calculate optimal batch size
    input_size = (10,)  # Input size for the model
    target_memory_usage = 0.8  # 80% memory usage target
    
    optimal_batch_size = get_optimal_batch_size(model, input_size, target_memory_usage)
    print(f"Optimal batch size: {optimal_batch_size}")
    
    # Test different batch sizes around the optimal
    test_batch_sizes = [optimal_batch_size // 2, optimal_batch_size, optimal_batch_size * 2]
    
    for batch_size in test_batch_sizes:
        try:
            # Create test data
            X, y = create_test_data(1000, 10, 2)
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Benchmark performance
            results = benchmark_model_performance(model, dataloader, num_iterations=20)
            print(f"Batch size {batch_size}: {results['throughput']:.2f} samples/s")
            
        except RuntimeError as e:
            print(f"Batch size {batch_size}: Out of memory")
    
    print("Optimal batch size demonstration completed")


async def demonstrate_optimized_training():
    """Demonstrate optimized training"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING OPTIMIZED TRAINING")
    print("="*60)
    
    # Create logger
    logger = create_training_logger(
        experiment_name="performance_optimization_demo",
        log_dir="performance_logs",
        log_level="INFO"
    )
    
    # Create model
    model = PerformanceTestModel()
    
    # Create data
    X_train, y_train = create_test_data(2000, 10, 2)
    X_val, y_val = create_test_data(500, 10, 2)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create performance configuration
    performance_config = PerformanceConfig(
        enable_mixed_precision=True,
        enable_compile=True,
        enable_torch_optimization=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        enable_gradient_accumulation=True,
        gradient_accumulation_steps=2
    )
    
    # Train with optimization
    print("Starting optimized training...")
    
    try:
        results = await train_model_with_optimization(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            experiment_name="performance_optimization_training",
            debug_mode=False,
            enable_pytorch_debugging=False,
            performance_config=performance_config,
            max_epochs=3,
            learning_rate=0.001
        )
        
        print(f"Training completed: {json.dumps(results, indent=2)}")
        
    except Exception as e:
        print(f"Training error: {e}")
        logger.log_error(e, "Optimized training", "demonstrate_optimized_training")
    
    finally:
        logger.cleanup()
    
    print("Optimized training demonstration completed")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING PERFORMANCE MONITORING")
    print("="*60)
    
    # Create performance optimizer with monitoring
    optimizer = create_performance_optimizer(
        enable_performance_monitoring=True,
        performance_log_interval=10
    )
    
    # Start monitoring
    optimizer.start_monitoring()
    
    # Create test model and data
    model = PerformanceTestModel()
    X, y = create_test_data(1000, 10, 2)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Simulate training with monitoring
    model.train()
    optimizer_model = optimizer.optimize_model(model)
    optimizer_optimizer = optimizer.optimize_optimizer(optim.Adam(model.parameters()))
    
    print("Running performance monitoring...")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= 50:  # Run 50 batches
            break
        
        start_time = time.time()
        
        # Forward pass
        outputs = optimizer_model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Backward pass
        optimizer_optimizer.zero_grad()
        loss.backward()
        optimizer_optimizer.step()
        
        # Record performance
        batch_time = time.time() - start_time
        optimizer.record_performance(inputs.size(0), batch_time)
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: {inputs.size(0) / batch_time:.2f} samples/s")
    
    # Get performance summary
    performance_summary = optimizer.get_performance_summary()
    print(f"\nPerformance summary: {json.dumps(performance_summary, indent=2)}")
    
    print("Performance monitoring demonstration completed")


def demonstrate_resource_usage():
    """Demonstrate resource usage monitoring"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING RESOURCE USAGE MONITORING")
    print("="*60)
    
    # Monitor CPU and memory usage
    print("System resource usage:")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # Test memory usage with different models
    models = [
        ("Small Model", PerformanceTestModel(input_size=10, hidden_size=50)),
        ("Medium Model", PerformanceTestModel(input_size=10, hidden_size=200)),
        ("Large Model", PerformanceTestModel(input_size=10, hidden_size=500))
    ]
    
    for name, model in models:
        print(f"\nTesting {name}:")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Test memory usage
        try:
            # Create test input
            test_input = torch.randn(100, 10)
            if torch.cuda.is_available():
                test_input = test_input.cuda()
            
            # Forward pass
            with torch.no_grad():
                output = model(test_input)
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU Memory Used: {memory_allocated:.2f} GB")
            
        except RuntimeError as e:
            print(f"  Error: {e}")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Resource usage monitoring demonstration completed")


def demonstrate_optimization_recommendations():
    """Demonstrate optimization recommendations"""
    
    print("\n" + "="*60)
    print("DEMONSTRATING OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    # Create performance optimizer
    optimizer = create_performance_optimizer(
        enable_mixed_precision=False,  # Disable to trigger recommendation
        enable_compile=False,          # Disable to trigger recommendation
        num_workers=1                  # Low number to trigger recommendation
    )
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    
    print("Optimization recommendations:")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")
    
    # Test with optimized settings
    print("\nTesting with optimized settings...")
    optimized_optimizer = create_performance_optimizer(
        enable_mixed_precision=True,
        enable_compile=True,
        num_workers=4
    )
    
    optimized_recommendations = optimized_optimizer.get_optimization_recommendations()
    
    if optimized_recommendations:
        print("Remaining recommendations:")
        for i, recommendation in enumerate(optimized_recommendations, 1):
            print(f"{i}. {recommendation}")
    else:
        print("No additional recommendations - optimizations are well configured!")
    
    print("Optimization recommendations demonstration completed")


async def main():
    """Run all performance optimization demonstrations"""
    
    print("PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    print("This demonstration shows various performance optimization techniques:")
    print("1. Memory optimization")
    print("2. Computational optimization")
    print("3. Data loading optimization")
    print("4. Optimal batch size calculation")
    print("5. Optimized training")
    print("6. Performance monitoring")
    print("7. Resource usage monitoring")
    print("8. Optimization recommendations")
    print("="*60)
    
    try:
        # Run demonstrations
        demonstrate_memory_optimization()
        demonstrate_computational_optimization()
        demonstrate_data_loading_optimization()
        demonstrate_optimal_batch_size()
        
        # Run async demonstrations
        await demonstrate_optimized_training()
        
        demonstrate_performance_monitoring()
        demonstrate_resource_usage()
        demonstrate_optimization_recommendations()
        
        print("\n" + "="*60)
        print("ALL PERFORMANCE OPTIMIZATION DEMONSTRATIONS COMPLETED!")
        print("="*60)
        print("\nGenerated files:")
        print("- performance_logs/ (directory with training logs)")
        print("- Performance optimization reports")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        traceback.print_exc()


match __name__:
    case "__main__":
    asyncio.run(main()) 