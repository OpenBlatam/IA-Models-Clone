#!/usr/bin/env python3
"""
Quick Start Code Profiling for Video-OpusClip

Easy-to-use script for getting started with code profiling
to identify and optimize bottlenecks in the Video-OpusClip system.
"""

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import argparse
from pathlib import Path
import json
import os

# Import profiling modules
from code_profiler import (
    VideoOpusClipProfiler,
    ProfilerConfig,
    create_profiler_config
)

def create_sample_model(input_size=784, hidden_size=512, num_classes=10):
    """Create a sample model for demonstration."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size // 2, num_classes)
    )

def create_sample_dataset(num_samples=1000, input_size=784, num_classes=10):
    """Create a sample dataset for demonstration."""
    torch.manual_seed(42)
    data = torch.randn(num_samples, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(data, labels)

def quick_start_basic_profiling():
    """Basic profiling setup and usage."""
    print("=== Quick Start: Basic Profiling ===")
    
    # Create profiler
    config = create_profiler_config("basic")
    profiler = VideoOpusClipProfiler(config)
    
    # Define functions to profile
    @profiler.profile_function
    def fast_function():
        """Fast function."""
        return sum(range(1000))
    
    @profiler.profile_function
    def slow_function():
        """Slow function."""
        time.sleep(0.1)
        return sum(range(10000))
    
    # Start profiling
    profiler.start_profiling()
    
    # Run functions
    for i in range(10):
        fast_function()
        slow_function()
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Basic Profiling Results:")
    print(f"Total functions profiled: {report['performance']['total_functions_profiled']}")
    print(f"Slow functions: {len(bottlenecks['slow_functions'])}")
    
    # Print function details
    for func_name, profile_data in report['performance']['profiles'].items():
        print(f"  {func_name}: {profile_data['execution_time']:.4f}s")
    
    return profiler

def quick_start_data_loading_profiling():
    """Data loading profiling."""
    print("\n=== Quick Start: Data Loading Profiling ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Create dataset and data loader
    dataset = create_sample_dataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Profile data loader
    profiled_loader = profiler.profile_data_loader(train_loader)
    
    # Start profiling
    profiler.start_profiling()
    
    # Test data loading
    print("Testing data loading...")
    for batch_idx, (inputs, targets) in enumerate(profiled_loader):
        if batch_idx >= 5:  # Test first 5 batches
            break
        
        # Simulate processing
        time.sleep(0.01)
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    loader_stats = profiled_loader.get_stats()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Data Loading Results:")
    print(f"Average batch time: {loader_stats['avg_batch_time']:.4f}s")
    print(f"Max batch time: {loader_stats['max_batch_time']:.4f}s")
    print(f"Total batches: {loader_stats['total_batches']}")
    
    print(f"Data loading bottlenecks: {len(bottlenecks['slow_data_loading'])}")
    
    return profiler

def quick_start_memory_profiling():
    """Memory profiling."""
    print("\n=== Quick Start: Memory Profiling ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Start profiling
    profiler.start_profiling()
    
    # Test memory usage
    with profiler.memory_profiler.memory_context("data_creation"):
        large_data = np.random.randn(1000, 1000)
        print(f"Created data: {large_data.nbytes / 1024 / 1024:.1f}MB")
    
    with profiler.memory_profiler.memory_context("model_creation"):
        model = create_sample_model()
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    with profiler.memory_profiler.memory_context("model_inference"):
        with torch.no_grad():
            inputs = torch.randn(100, 784)
            outputs = model(inputs)
        print("Model inference completed")
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    memory_report = profiler.memory_profiler.get_memory_report()
    
    print("Memory Profiling Results:")
    print(f"Total traces: {memory_report['total_traces']}")
    
    stats = memory_report['memory_statistics']
    print(f"Total memory delta: {stats['total_memory_delta'] / 1024 / 1024:.1f}MB")
    print(f"Average memory delta: {stats['avg_memory_delta'] / 1024 / 1024:.1f}MB")
    
    return profiler

def quick_start_gpu_profiling():
    """GPU profiling."""
    print("\n=== Quick Start: GPU Profiling ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU profiling")
        return None
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Start profiling
    profiler.start_profiling()
    
    # Test GPU operations
    with profiler.gpu_profiler.cuda_context("model_creation"):
        model = create_sample_model().cuda()
        print(f"Created GPU model")
    
    with profiler.gpu_profiler.cuda_context("data_transfer"):
        inputs = torch.randn(100, 784).cuda()
        print(f"Transferred data to GPU")
    
    with profiler.gpu_profiler.cuda_context("model_inference"):
        with torch.no_grad():
            outputs = model(inputs)
        print("GPU model inference completed")
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    gpu_report = profiler.gpu_profiler.get_gpu_report()
    
    print("GPU Profiling Results:")
    print(f"Total events: {gpu_report['total_events']}")
    
    stats = gpu_report['gpu_statistics']
    print(f"Total GPU time: {stats['total_time_ms']:.2f}ms")
    print(f"Average GPU time: {stats['avg_time_ms']:.2f}ms")
    
    return profiler

def quick_start_comprehensive_profiling():
    """Comprehensive profiling."""
    print("\n=== Quick Start: Comprehensive Profiling ===")
    
    # Create comprehensive profiler
    config = create_profiler_config("comprehensive")
    profiler = VideoOpusClipProfiler(config)
    
    # Create model and data
    model = create_sample_model()
    dataset = create_sample_dataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Profile data loader
    profiled_loader = profiler.profile_data_loader(train_loader)
    
    # Start profiling
    profiler.start_profiling()
    
    # Test complete pipeline
    print("Testing complete pipeline...")
    
    with profiler.profiling_context("training_session"):
        for epoch in range(2):
            with profiler.profiling_context(f"epoch_{epoch}"):
                for batch_idx, (inputs, targets) in enumerate(profiled_loader):
                    if batch_idx >= 3:  # Test first 3 batches per epoch
                        break
                    
                    with profiler.profiling_context(f"batch_{batch_idx}"):
                        # Forward pass
                        outputs = model(inputs)
                        
                        # Simulate training step
                        time.sleep(0.01)
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get comprehensive results
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Comprehensive Profiling Results:")
    print(f"Session duration: {report['session_info']['duration']:.2f}s")
    print(f"Total functions profiled: {report['performance']['total_functions_profiled']}")
    
    print(f"\nBottlenecks Found:")
    print(f"  Slow functions: {len(bottlenecks['slow_functions'])}")
    print(f"  Memory intensive functions: {len(bottlenecks['memory_intensive_functions'])}")
    print(f"  Slow data loading: {len(bottlenecks['slow_data_loading'])}")
    print(f"  Slow preprocessing: {len(bottlenecks['slow_preprocessing'])}")
    print(f"  Slow GPU operations: {len(bottlenecks['slow_gpu_operations'])}")
    
    print(f"\nRecommendations:")
    for recommendation in bottlenecks['recommendations']:
        print(f"  - {recommendation}")
    
    # Save report
    report_path = profiler.save_comprehensive_report("quick_start_profile.json")
    print(f"\nReport saved to: {report_path}")
    
    return profiler

def quick_start_bottleneck_analysis():
    """Bottleneck analysis and optimization suggestions."""
    print("\n=== Quick Start: Bottleneck Analysis ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Define functions with known bottlenecks
    @profiler.profile_function
    def inefficient_function():
        """Function with inefficient implementation."""
        result = 0
        for i in range(10000):
            result += i * 2
        return result
    
    @profiler.profile_function
    def efficient_function():
        """Function with efficient implementation."""
        return sum(i * 2 for i in range(10000))
    
    @profiler.profile_function
    def memory_heavy_function():
        """Function that uses a lot of memory."""
        arrays = []
        for i in range(10):
            array = np.random.randn(500, 500)
            arrays.append(array)
        
        result = sum(np.sum(arr) for arr in arrays)
        return result
    
    # Start profiling
    profiler.start_profiling()
    
    # Test functions
    print("Testing functions...")
    inefficient_result = inefficient_function()
    efficient_result = efficient_function()
    memory_result = memory_heavy_function()
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Analyze bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    report = profiler.get_comprehensive_report()
    
    print("Bottleneck Analysis Results:")
    print(f"Total functions profiled: {report['performance']['total_functions_profiled']}")
    
    # Compare performance
    profiles = report['performance']['profiles']
    
    if 'inefficient_function' in profiles and 'efficient_function' in profiles:
        inefficient_time = profiles['inefficient_function']['execution_time']
        efficient_time = profiles['efficient_function']['execution_time']
        speedup = inefficient_time / efficient_time if efficient_time > 0 else float('inf')
        
        print(f"\nPerformance Comparison:")
        print(f"  Inefficient: {inefficient_time:.4f}s")
        print(f"  Efficient: {efficient_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    print(f"\nIdentified Bottlenecks:")
    for category, items in bottlenecks.items():
        if items and category != 'recommendations':
            print(f"  {category}: {len(items)} items")
    
    print(f"\nOptimization Recommendations:")
    for recommendation in bottlenecks['recommendations']:
        print(f"  - {recommendation}")
    
    return profiler

def quick_start_performance_monitoring():
    """Performance monitoring setup."""
    print("\n=== Quick Start: Performance Monitoring ===")
    
    # Create monitoring profiler
    config = ProfilerConfig(
        enabled=True,
        profile_level="basic",
        save_profiles=True,
        detailed_reports=False
    )
    profiler = VideoOpusClipProfiler(config)
    
    # Define monitoring functions
    @profiler.profile_function
    def monitored_function():
        """Function to monitor."""
        time.sleep(0.05)
        return "result"
    
    # Start monitoring
    profiler.start_profiling()
    
    # Simulate monitoring over time
    print("Monitoring performance over time...")
    for i in range(20):
        result = monitored_function()
        
        # Log every 5 calls
        if i % 5 == 0:
            print(f"  Call {i}: completed")
    
    # Stop monitoring
    profiler.stop_profiling()
    
    # Get monitoring results
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Performance Monitoring Results:")
    print(f"Monitoring duration: {report['session_info']['duration']:.2f}s")
    print(f"Functions monitored: {report['performance']['total_functions_profiled']}")
    print(f"Performance issues: {len(bottlenecks['slow_functions'])}")
    
    # Save monitoring report
    report_path = profiler.save_comprehensive_report("monitoring_report.json")
    print(f"Monitoring report saved to: {report_path}")
    
    return profiler

def main():
    """Main function for quick start script."""
    parser = argparse.ArgumentParser(description="Quick Start Code Profiling")
    parser.add_argument(
        "--mode",
        choices=["basic", "data_loading", "memory", "gpu", "comprehensive", "bottlenecks", "monitoring", "all"],
        default="basic",
        help="Profiling mode to run"
    )
    parser.add_argument(
        "--level",
        choices=["basic", "detailed", "comprehensive"],
        default="detailed",
        help="Profiling level"
    )
    parser.add_argument(
        "--save-reports",
        action="store_true",
        help="Save profiling reports to files"
    )
    
    args = parser.parse_args()
    
    print("Code Profiling Quick Start for Video-OpusClip")
    print("=" * 50)
    
    # Check system information
    print(f"System Information:")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print()
    
    try:
        if args.mode == "basic" or args.mode == "all":
            quick_start_basic_profiling()
        
        if args.mode == "data_loading" or args.mode == "all":
            quick_start_data_loading_profiling()
        
        if args.mode == "memory" or args.mode == "all":
            quick_start_memory_profiling()
        
        if args.mode == "gpu" or args.mode == "all":
            quick_start_gpu_profiling()
        
        if args.mode == "comprehensive" or args.mode == "all":
            quick_start_comprehensive_profiling()
        
        if args.mode == "bottlenecks" or args.mode == "all":
            quick_start_bottleneck_analysis()
        
        if args.mode == "monitoring" or args.mode == "all":
            quick_start_performance_monitoring()
        
        print("\n" + "=" * 50)
        print("Quick start completed successfully!")
        
        if args.mode == "all":
            print("\nAll modes completed. Check the output above for results.")
        
        # Summary
        print("\nSummary:")
        print("- Basic profiling: Function timing and performance")
        print("- Data loading profiling: Dataset and DataLoader optimization")
        print("- Memory profiling: Memory usage and leaks detection")
        print("- GPU profiling: CUDA operations and memory")
        print("- Comprehensive profiling: Full system analysis")
        print("- Bottleneck analysis: Performance comparison and optimization")
        print("- Performance monitoring: Continuous monitoring setup")
        
    except Exception as e:
        print(f"\nError during quick start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 