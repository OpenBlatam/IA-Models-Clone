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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
from typing import Dict, List, Any
import logging
import psutil # Added for system info
from code_profiling_system import (
from typing import Any, List, Dict, Optional
import asyncio
"""
🚀 Code Profiling Example
=========================

This example demonstrates comprehensive code profiling for identifying and optimizing
bottlenecks in data loading, preprocessing, and other operations.
"""


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the profiling system
    CodeProfiler, ProfilingConfig, DataLoadingProfiler, PreprocessingProfiler,
    profile_function, profile_data_loading, profile_preprocessing
)


class SampleNeuralNetwork(nn.Module):
    """Sample neural network for profiling demonstration."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_classes: int = 2):
        
    """__init__ function."""
super(SampleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x) -> Any:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SlowDataLoader:
    """Intentionally slow data loader for profiling demonstration."""
    
    def __init__(self, num_samples: int = 1000, batch_size: int = 32):
        
    """__init__ function."""
self.num_samples = num_samples
        self.batch_size = batch_size
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, 2, (num_samples,))
        
    def __iter__(self) -> Any:
        for i in range(0, self.num_samples, self.batch_size):
            # Simulate slow data loading
            time.sleep(0.01)  # 10ms delay
            batch_data = self.data[i:i+self.batch_size]
            batch_labels = self.labels[i:i+self.batch_size]
            yield batch_data, batch_labels


def demonstrate_basic_profiling():
    """Demonstrate basic function profiling."""
    print("=" * 60)
    print("🔧 BASIC FUNCTION PROFILING DEMONSTRATION")
    print("=" * 60)
    
    # Create profiler
    config = ProfilingConfig(
        enabled=True,
        profile_memory=True,
        profile_cpu=True,
        profile_gpu=True,
        cpu_threshold_ms=50.0,  # Lower threshold for demonstration
        memory_threshold_mb=50.0  # Lower threshold for demonstration
    )
    
    profiler = CodeProfiler(config)
    
    # Define functions to profile
    @profiler.profile_function
    def fast_function():
        """Fast function that should not trigger bottlenecks."""
        time.sleep(0.01)  # 10ms
        return "fast"
    
    @profiler.profile_function
    def slow_function():
        """Slow function that should trigger bottlenecks."""
        time.sleep(0.2)  # 200ms - should trigger CPU bottleneck
        return "slow"
    
    @profiler.profile_function
    def memory_intensive_function():
        """Memory intensive function that should trigger memory bottleneck."""
        # Allocate large tensor
        large_tensor = torch.randn(1000, 1000)  # ~8MB
        time.sleep(0.05)
        return large_tensor.sum().item()
    
    print("🚀 Running basic profiling...")
    
    # Run functions
    fast_function()
    slow_function()
    memory_intensive_function()
    
    # Get results
    report = profiler.get_performance_report()
    
    print(f"✅ Basic profiling completed")
    print(f"Functions profiled: {report['overall_statistics']['total_functions_profiled']}")
    print(f"Bottlenecks found: {report['bottlenecks']['bottlenecks_found']}")
    
    if report['bottlenecks']['top_bottlenecks']:
        print("\n🔍 Top bottlenecks:")
        for bottleneck in report['bottlenecks']['top_bottlenecks']:
            print(f"  - {bottleneck['function']}: {bottleneck['execution_time_ms']:.1f}ms")
    
    return report


def demonstrate_data_loading_profiling():
    """Demonstrate data loading profiling."""
    print("\n" + "=" * 60)
    print("📊 DATA LOADING PROFILING DEMONSTRATION")
    print("=" * 60)
    
    # Create profiler
    config = ProfilingConfig(
        enabled=True,
        profile_memory=True,
        profile_cpu=True,
        profile_io=True,
        cpu_threshold_ms=100.0,
        memory_threshold_mb=100.0
    )
    
    profiler = CodeProfiler(config)
    data_profiler = DataLoadingProfiler(profiler)
    
    # Create slow data loader
    slow_loader = SlowDataLoader(num_samples=500, batch_size=32)
    
    print("🚀 Profiling data loading...")
    
    # Profile the data loader
    dataloader_results = data_profiler.profile_dataloader(slow_loader, num_batches=5)
    
    print(f"✅ Data loading profiling completed")
    print(f"Average batch time: {dataloader_results['avg_batch_time_ms']:.2f}ms")
    print(f"Average memory usage: {dataloader_results['avg_memory_usage_mb']:.2f}MB")
    
    # Profile with PyTorch DataLoader for comparison
    @profiler.profile_data_loading
    def create_pytorch_dataloader():
        """Create and use PyTorch DataLoader."""
        X = torch.randn(500, 10)
        y = torch.randint(0, 2, (500,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Iterate through a few batches
        for i, (data, labels) in enumerate(dataloader):
            if i >= 5:
                break
            time.sleep(0.01)  # Simulate processing
        
        return "completed"
    
    pytorch_result = create_pytorch_dataloader()
    
    # Get overall results
    report = profiler.get_performance_report()
    
    return {
        'dataloader_results': dataloader_results,
        'overall_report': report
    }


def demonstrate_preprocessing_profiling():
    """Demonstrate preprocessing profiling."""
    print("\n" + "=" * 60)
    print("🔄 PREPROCESSING PROFILING DEMONSTRATION")
    print("=" * 60)
    
    # Create profiler
    config = ProfilingConfig(
        enabled=True,
        profile_memory=True,
        profile_cpu=True,
        profile_gpu=True,
        cpu_threshold_ms=50.0,
        memory_threshold_mb=50.0
    )
    
    profiler = CodeProfiler(config)
    preprocessing_profiler = PreprocessingProfiler(profiler)
    
    # Define preprocessing functions
    @profiler.profile_preprocessing
    def normalize_data(data) -> Any:
        """Normalize data - should be fast."""
        return (data - data.mean()) / data.std()
    
    @profiler.profile_preprocessing
    def augment_data(data) -> Any:
        """Data augmentation - should be slower."""
        # Simulate data augmentation
        time.sleep(0.05)
        augmented = data + torch.randn_like(data) * 0.1
        return augmented
    
    @profiler.profile_preprocessing
    def complex_preprocessing(data) -> Any:
        """Complex preprocessing - should be memory intensive."""
        # Multiple operations
        result = data
        for i in range(10):
            result = torch.relu(result)
            result = torch.dropout(result, p=0.1, training=True)
            result = result + torch.randn_like(result) * 0.01
        
        return result
    
    print("🚀 Profiling preprocessing pipeline...")
    
    # Create sample data
    sample_data = torch.randn(100, 10)
    
    # Profile preprocessing pipeline
    preprocessing_funcs = [normalize_data, augment_data, complex_preprocessing]
    pipeline_results = preprocessing_profiler.profile_preprocessing_pipeline(
        preprocessing_funcs, sample_data
    )
    
    print(f"✅ Preprocessing profiling completed")
    print(f"Pipeline steps: {len(pipeline_results)}")
    
    for step_name, step_result in pipeline_results.items():
        print(f"  - {step_result['function_name']}: {step_result['execution_time_ms']:.2f}ms, "
              f"{step_result['memory_usage_mb']:.2f}MB")
    
    # Get overall results
    report = profiler.get_performance_report()
    
    return {
        'pipeline_results': pipeline_results,
        'overall_report': report
    }


def demonstrate_model_training_profiling():
    """Demonstrate model training profiling."""
    print("\n" + "=" * 60)
    print("🧠 MODEL TRAINING PROFILING DEMONSTRATION")
    print("=" * 60)
    
    # Create profiler
    config = ProfilingConfig(
        enabled=True,
        profile_memory=True,
        profile_cpu=True,
        profile_gpu=True,
        cpu_threshold_ms=100.0,
        memory_threshold_mb=100.0
    )
    
    profiler = CodeProfiler(config)
    
    # Create model and data
    model = SampleNeuralNetwork()
    if torch.cuda.is_available():
        model = model.cuda()
    
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("🚀 Profiling model training...")
    
    # Profile training loop
    with profiler.profile_context("Training_Loop"):
        model.train()
        
        for epoch in range(3):
            with profiler.profile_context(f"Epoch_{epoch}"):
                # Forward pass
                with profiler.profile_context("Forward_Pass"):
                    outputs = model(X)
                    loss = criterion(outputs, y)
                
                # Backward pass
                with profiler.profile_context("Backward_Pass"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print(f"✅ Model training profiling completed")
    
    # Get results
    report = profiler.get_performance_report()
    
    return report


def demonstrate_bottleneck_analysis():
    """Demonstrate bottleneck analysis and optimization suggestions."""
    print("\n" + "=" * 60)
    print("🔍 BOTTLENECK ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create profiler with strict thresholds
    config = ProfilingConfig(
        enabled=True,
        profile_memory=True,
        profile_cpu=True,
        profile_gpu=True,
        cpu_threshold_ms=10.0,  # Very strict threshold
        memory_threshold_mb=10.0,  # Very strict threshold
        gpu_memory_threshold_mb=50.0
    )
    
    profiler = CodeProfiler(config)
    
    # Define functions that will trigger bottlenecks
    @profiler.profile_function
    def cpu_intensive_function():
        """CPU intensive function."""
        # Simulate CPU intensive work
        result = 0
        for i in range(1000000):
            result += i * i
        return result
    
    @profiler.profile_function
    def memory_intensive_function():
        """Memory intensive function."""
        # Allocate multiple large tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000)  # ~8MB each
            tensors.append(tensor)
        
        # Process tensors
        result = sum(t.sum() for t in tensors)
        
        # Clean up
        del tensors
        return result
    
    @profiler.profile_function
    def io_intensive_function():
        """I/O intensive function."""
        # Simulate I/O operations
        for i in range(100):
            # Simulate file read/write
            time.sleep(0.001)
        
        return "io_completed"
    
    print("🚀 Running bottleneck analysis...")
    
    # Run functions
    cpu_intensive_function()
    memory_intensive_function()
    io_intensive_function()
    
    # Get bottleneck analysis
    bottlenecks = profiler.get_bottlenecks_summary()
    
    print(f"✅ Bottleneck analysis completed")
    print(f"Total functions profiled: {bottlenecks['total_functions_profiled']}")
    print(f"Bottlenecks found: {bottlenecks['bottlenecks_found']}")
    
    if bottlenecks['bottleneck_types']:
        print("\n🔍 Bottleneck types:")
        for btype, count in bottlenecks['bottleneck_types'].items():
            print(f"  - {btype}: {count} occurrences")
    
    if bottlenecks['top_bottlenecks']:
        print("\n🚨 Top bottlenecks:")
        for bottleneck in bottlenecks['top_bottlenecks']:
            print(f"  - {bottleneck['function']}")
            print(f"    Execution time: {bottleneck['execution_time_ms']:.1f}ms")
            print(f"    Memory usage: {bottleneck['memory_usage_mb']:.1f}MB")
            print(f"    Bottleneck type: {bottleneck['bottleneck_type']}")
            print(f"    Suggestions: {bottleneck['suggestions']}")
            print()
    
    if bottlenecks['optimization_recommendations']:
        print("💡 Optimization recommendations:")
        for recommendation in bottlenecks['optimization_recommendations']:
            print(f"  - {recommendation}")
    
    return bottlenecks


def demonstrate_profiling_export():
    """Demonstrate profiling results export."""
    print("\n" + "=" * 60)
    print("💾 PROFILING RESULTS EXPORT DEMONSTRATION")
    print("=" * 60)
    
    # Create profiler
    config = ProfilingConfig(
        enabled=True,
        save_profiles=True,
        export_format="json",
        output_dir="profiling_exports"
    )
    
    profiler = CodeProfiler(config)
    
    # Run some profiling
    @profiler.profile_function
    def export_test_function():
        
    """export_test_function function."""
time.sleep(0.1)
        return "export_test"
    
    export_test_function()
    
    print("🚀 Exporting profiling results...")
    
    # Export results
    export_path = profiler.export_results("profiling_demo")
    
    print(f"✅ Profiling results exported to: {export_path}")
    
    # Load and display exported results
    try:
        with open(export_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            exported_data = json.load(f)
        
        print(f"📊 Exported data summary:")
        print(f"  - Functions profiled: {len(exported_data['results'])}")
        print(f"  - Bottlenecks found: {exported_data['summary']['bottlenecks_found']}")
        print(f"  - Export format: {config.export_format}")
        
    except Exception as e:
        print(f"❌ Failed to load exported data: {e}")
    
    return export_path


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between optimized and unoptimized code."""
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Create profiler
    config = ProfilingConfig(
        enabled=True,
        profile_memory=True,
        profile_cpu=True,
        profile_gpu=True
    )
    
    profiler = CodeProfiler(config)
    
    # Unoptimized function
    @profiler.profile_function
    def unoptimized_function(data) -> Any:
        """Unoptimized function with loops."""
        result = torch.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                result[i, j] = data[i, j] * 2 + 1
        return result
    
    # Optimized function
    @profiler.profile_function
    def optimized_function(data) -> Any:
        """Optimized function with vectorized operations."""
        return data * 2 + 1
    
    # Create test data
    test_data = torch.randn(1000, 1000)
    
    print("🚀 Comparing optimized vs unoptimized performance...")
    
    # Profile both functions
    unoptimized_result = unoptimized_function(test_data)
    optimized_result = optimized_function(test_data)
    
    # Verify results are the same
    assert torch.allclose(unoptimized_result, optimized_result), "Results should be identical"
    
    # Get performance comparison
    report = profiler.get_performance_report()
    
    print(f"✅ Performance comparison completed")
    
    # Find the two functions in results
    unopt_result = None
    opt_result = None
    
    for result in profiler.results.values():
        if "unoptimized" in result.function_name:
            unopt_result = result
        elif "optimized" in result.function_name:
            opt_result = result
    
    if unopt_result and opt_result:
        speedup = unopt_result.execution_time / opt_result.execution_time
        memory_improvement = unopt_result.memory_usage / opt_result.memory_usage
        
        print(f"📊 Performance comparison:")
        print(f"  - Unoptimized: {unopt_result.execution_time:.1f}ms, {unopt_result.memory_usage:.1f}MB")
        print(f"  - Optimized: {opt_result.execution_time:.1f}ms, {opt_result.memory_usage:.1f}MB")
        print(f"  - Speedup: {speedup:.1f}x")
        print(f"  - Memory improvement: {memory_improvement:.1f}x")
    
    return report


def main():
    """Run all profiling demonstrations."""
    print("🚀 CODE PROFILING DEMONSTRATION SUITE")
    print("=" * 80)
    
    # Check system capabilities
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"System Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    print("\n" + "=" * 80)
    
    # Run demonstrations
    demonstrations = [
        ("Basic Function Profiling", demonstrate_basic_profiling),
        ("Data Loading Profiling", demonstrate_data_loading_profiling),
        ("Preprocessing Profiling", demonstrate_preprocessing_profiling),
        ("Model Training Profiling", demonstrate_model_training_profiling),
        ("Bottleneck Analysis", demonstrate_bottleneck_analysis),
        ("Profiling Export", demonstrate_profiling_export),
        ("Performance Comparison", demonstrate_performance_comparison),
    ]
    
    results = {}
    
    for name, demo_func in demonstrations:
        try:
            print(f"\n🎯 Running: {name}")
            result = demo_func()
            results[name] = {"success": True, "result": result}
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {"success": False, "error": str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 PROFILING DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    print(f"Successful Demonstrations: {successful}/{total}")
    
    for name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {name}")
    
    print(f"\n🎉 Code profiling demonstration completed!")
    print(f"   Success Rate: {successful/total*100:.1f}%")
    
    # Final recommendations
    print(f"\n💡 Key Takeaways:")
    print(f"   - Use @profile_function decorator for function-level profiling")
    print(f"   - Use profile_context() for code block profiling")
    print(f"   - Monitor bottlenecks with automatic threshold detection")
    print(f"   - Export results for detailed analysis")
    print(f"   - Compare optimized vs unoptimized implementations")
    
    return results


match __name__:
    case "__main__":
    main() 