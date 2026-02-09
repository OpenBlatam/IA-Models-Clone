"""
Performance Optimization Examples for Video-OpusClip

Practical examples demonstrating various performance optimization techniques:
- Smart caching strategies
- Memory optimization patterns
- GPU optimization techniques
- Load balancing implementations
- Adaptive performance tuning
- Real-world optimization scenarios
"""

import asyncio
import time
import psutil
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import structlog
from pathlib import Path
import json
import pickle

# Import performance optimization components
from performance_optimizer import (
    PerformanceOptimizer, OptimizationConfig, SmartCache, MemoryOptimizer,
    GPUOptimizer, LoadBalancer, AdaptivePerformanceTuner,
    optimize_performance, cache_result
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: BASIC PERFORMANCE OPTIMIZATION SETUP
# =============================================================================

def example_basic_optimization_setup():
    """Basic performance optimization setup example."""
    
    print("🚀 Example 1: Basic Performance Optimization Setup")
    print("=" * 60)
    
    # Create optimization configuration
    config = OptimizationConfig(
        enable_smart_caching=True,
        enable_memory_optimization=True,
        enable_gpu_optimization=True,
        enable_load_balancing=True,
        enable_adaptive_tuning=True,
        max_workers=4,
        cache_size_limit=500,
        memory_cleanup_threshold=80.0
    )
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(config)
    
    print("✅ Performance optimizer initialized")
    print(f"📊 Configuration: {config.__dict__}")
    
    # Example operation with optimization
    @optimize_performance("data_processing")
    def process_data(data: List[float]) -> float:
        # Simulate data processing
        time.sleep(0.1)
        return sum(data) / len(data)
    
    # Execute with optimization
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result, metrics = process_data(test_data)
    
    print(f"📈 Processing result: {result}")
    print(f"⏱️  Processing time: {metrics.processing_time:.3f}s")
    print(f"💾 Memory usage: {metrics.memory_usage:.1f}%")
    print(f"🎯 Optimization score: {metrics.optimization_score:.2f}")
    
    return optimizer

# =============================================================================
# EXAMPLE 2: SMART CACHING STRATEGIES
# =============================================================================

def example_smart_caching_strategies():
    """Demonstrate smart caching strategies."""
    
    print("\n🚀 Example 2: Smart Caching Strategies")
    print("=" * 60)
    
    # Initialize cache with optimization config
    config = OptimizationConfig(
        enable_smart_caching=True,
        cache_size_limit=100,
        cache_ttl=3600,
        enable_predictive_caching=True
    )
    
    cache = SmartCache(config)
    
    # Simulate different types of data with different caching strategies
    data_types = {
        "user_profile": {"ttl": 3600, "frequency": "high"},
        "session_data": {"ttl": 300, "frequency": "medium"},
        "static_content": {"ttl": 86400, "frequency": "low"},
        "temporary_data": {"ttl": 60, "frequency": "very_high"}
    }
    
    print("📦 Caching different data types...")
    
    # Cache different types of data
    for data_type, properties in data_types.items():
        # Simulate data
        data = {
            "type": data_type,
            "content": f"data_for_{data_type}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache with appropriate TTL
        cache.set(data_type, data, ttl=properties["ttl"])
        print(f"  Cached {data_type} with TTL {properties['ttl']}s")
    
    # Simulate access patterns
    print("\n🔄 Simulating access patterns...")
    
    # Simulate frequent access to temporary data
    for i in range(10):
        cache.get("temporary_data")
        time.sleep(0.01)
    
    # Simulate occasional access to static content
    for i in range(3):
        cache.get("static_content")
        time.sleep(0.1)
    
    # Simulate regular access to user profile
    for i in range(5):
        cache.get("user_profile")
        time.sleep(0.05)
    
    # Get cache statistics
    stats = cache.get_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Predictions: {stats['predictions']}")
    
    # Demonstrate cache optimization
    print("\n🔧 Optimizing cache...")
    cache._optimize_cache()
    
    # Show final statistics
    final_stats = cache.get_stats()
    print(f"  Final cache size: {final_stats['cache_size']}")
    print(f"  Final hit rate: {final_stats['hit_rate']:.2%}")
    
    return cache

# =============================================================================
# EXAMPLE 3: MEMORY OPTIMIZATION PATTERNS
# =============================================================================

def example_memory_optimization_patterns():
    """Demonstrate memory optimization patterns."""
    
    print("\n🚀 Example 3: Memory Optimization Patterns")
    print("=" * 60)
    
    # Initialize memory optimizer
    config = OptimizationConfig(
        enable_memory_optimization=True,
        memory_cleanup_threshold=70.0,
        enable_garbage_collection=True
    )
    
    memory_optimizer = MemoryOptimizer(config)
    
    print("✅ Memory optimizer initialized")
    
    # Simulate memory-intensive operations
    print("\n📊 Simulating memory-intensive operations...")
    
    # Create large tensors
    large_tensors = []
    for i in range(5):
        # Create large tensor
        tensor = torch.randn(1000, 1000)
        memory_optimizer.register_tensor(tensor, f"large_tensor_{i}")
        large_tensors.append(tensor)
        
        # Get memory stats
        stats = memory_optimizer.get_memory_stats()
        print(f"  Created tensor {i}: Memory usage {stats['memory_percent']:.1f}%")
        
        # Force optimization if memory usage is high
        if stats['memory_percent'] > 80:
            print(f"  🧹 High memory usage detected, optimizing...")
            optimization_result = memory_optimizer.optimize_memory()
            print(f"  💾 Memory saved: {optimization_result['memory_saved_mb']:.2f} MB")
    
    # Simulate memory cleanup
    print("\n🧹 Cleaning up memory...")
    
    # Remove some tensors
    for i in range(3):
        del large_tensors[i]
    
    # Force garbage collection
    gc.collect()
    
    # Optimize memory
    optimization_result = memory_optimizer.optimize_memory()
    print(f"  Memory optimization completed:")
    print(f"    Memory saved: {optimization_result['memory_saved_mb']:.2f} MB")
    print(f"    Optimization time: {optimization_result['optimization_time']:.3f}s")
    print(f"    Optimizations applied: {optimization_result['optimizations_applied']}")
    
    # Get final memory statistics
    final_stats = memory_optimizer.get_memory_stats()
    print(f"\n📊 Final Memory Statistics:")
    print(f"  Total memory: {final_stats['total_memory_gb']:.2f} GB")
    print(f"  Used memory: {final_stats['used_memory_gb']:.2f} GB")
    print(f"  Memory usage: {final_stats['memory_percent']:.1f}%")
    print(f"  Tracked tensors: {final_stats['tensor_count']}")
    print(f"  Optimization count: {final_stats['optimization_count']}")
    
    return memory_optimizer

# =============================================================================
# EXAMPLE 4: GPU OPTIMIZATION TECHNIQUES
# =============================================================================

def example_gpu_optimization_techniques():
    """Demonstrate GPU optimization techniques."""
    
    print("\n🚀 Example 4: GPU Optimization Techniques")
    print("=" * 60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ GPU not available, skipping GPU optimization example")
        return None
    
    # Initialize GPU optimizer
    config = OptimizationConfig(
        enable_gpu_optimization=True,
        mixed_precision=True,
        enable_cuda_graphs=True,
        tensor_memory_fraction=0.8
    )
    
    gpu_optimizer = GPUOptimizer(config)
    
    print("✅ GPU optimizer initialized")
    
    # Get initial GPU stats
    initial_stats = gpu_optimizer.get_gpu_stats()
    print(f"\n📊 Initial GPU Statistics:")
    print(f"  Device: {initial_stats['device_name']}")
    print(f"  Memory allocated: {initial_stats['memory_allocated_gb']:.2f} GB")
    print(f"  Memory reserved: {initial_stats['memory_reserved_gb']:.2f} GB")
    
    # Simulate GPU-intensive operations
    print("\n🔥 Simulating GPU-intensive operations...")
    
    # Create large GPU tensors
    gpu_tensors = []
    for i in range(3):
        # Create large tensor on GPU
        tensor = torch.randn(2000, 2000, device='cuda')
        gpu_tensors.append(tensor)
        
        # Perform some operations
        result = torch.matmul(tensor, tensor.T)
        
        # Get GPU stats
        stats = gpu_optimizer.get_gpu_stats()
        print(f"  Created tensor {i}: GPU memory {stats['memory_allocated_gb']:.2f} GB")
    
    # Demonstrate mixed precision
    print("\n🎯 Demonstrating mixed precision...")
    
    context = gpu_optimizer.create_optimized_context()
    if context and context["autocast"]:
        with context["autocast"]:
            # Mixed precision operations
            input_data = torch.randn(1000, 1000, device='cuda')
            result = torch.matmul(input_data, input_data.T)
            print("  Mixed precision operation completed")
    else:
        print("  Mixed precision not available")
    
    # Optimize GPU memory
    print("\n🧹 Optimizing GPU memory...")
    optimization_result = gpu_optimizer.optimize_gpu_memory()
    print(f"  GPU memory optimization completed:")
    print(f"    Memory saved: {optimization_result['memory_saved_mb']:.2f} MB")
    print(f"    Optimization time: {optimization_result['optimization_time']:.3f}s")
    
    # Get final GPU stats
    final_stats = gpu_optimizer.get_gpu_stats()
    print(f"\n📊 Final GPU Statistics:")
    print(f"  Memory allocated: {final_stats['memory_allocated_gb']:.2f} GB")
    print(f"  Memory reserved: {final_stats['memory_reserved_gb']:.2f} GB")
    print(f"  Max memory allocated: {final_stats['max_memory_allocated_gb']:.2f} GB")
    
    # Clean up
    del gpu_tensors
    torch.cuda.empty_cache()
    
    return gpu_optimizer

# =============================================================================
# EXAMPLE 5: LOAD BALANCING IMPLEMENTATIONS
# =============================================================================

def example_load_balancing_implementations():
    """Demonstrate load balancing implementations."""
    
    print("\n🚀 Example 5: Load Balancing Implementations")
    print("=" * 60)
    
    # Initialize load balancer
    config = OptimizationConfig(
        enable_load_balancing=True,
        max_workers=4,
        load_balancing_strategy="adaptive"
    )
    
    load_balancer = LoadBalancer(config)
    
    print("✅ Load balancer initialized")
    print(f"  Workers: {len(load_balancer.workers)}")
    print(f"  Strategy: {config.load_balancing_strategy}")
    
    # Simulate different load balancing strategies
    strategies = ["round_robin", "least_loaded", "adaptive"]
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy} strategy...")
        
        # Update strategy
        config.load_balancing_strategy = strategy
        load_balancer = LoadBalancer(config)
        
        # Simulate workload distribution
        workload_sizes = [10, 5, 15, 8, 12, 6, 20, 3]
        
        worker_assignments = []
        for i, workload_size in enumerate(workload_sizes):
            worker_id = load_balancer.get_next_worker()
            worker_assignments.append(worker_id)
            
            # Simulate processing time based on workload
            processing_time = workload_size * 0.1
            
            # Update worker statistics
            load_balancer.update_worker_stats(
                worker_id=worker_id,
                load=workload_size / 20.0,  # Normalize load
                response_time=processing_time,
                error_rate=0.01 if i % 10 == 0 else 0.0  # Occasional errors
            )
            
            print(f"  Workload {i} (size {workload_size}): assigned to worker {worker_id}")
        
        # Show worker statistics
        print(f"  Worker statistics for {strategy}:")
        for worker_id, stats in load_balancer.worker_stats.items():
            print(f"    Worker {worker_id}: load={stats['load']:.2f}, "
                  f"response_time={stats['response_time']:.3f}s, "
                  f"error_rate={stats['error_rate']:.2%}")
    
    # Demonstrate adaptive load balancing
    print(f"\n🎯 Adaptive load balancing analysis:")
    
    # Simulate varying worker performance
    for worker_id in range(len(load_balancer.workers)):
        # Simulate different performance characteristics
        if worker_id == 0:
            # Fast worker
            load_balancer.update_worker_stats(worker_id, 0.3, 0.1, 0.0)
        elif worker_id == 1:
            # Slow worker
            load_balancer.update_worker_stats(worker_id, 0.8, 0.5, 0.05)
        else:
            # Normal worker
            load_balancer.update_worker_stats(worker_id, 0.5, 0.2, 0.01)
    
    # Test adaptive assignment
    print("  Testing adaptive assignment with varying worker performance:")
    for i in range(10):
        worker_id = load_balancer.get_next_worker()
        print(f"    Request {i}: assigned to worker {worker_id}")
    
    return load_balancer

# =============================================================================
# EXAMPLE 6: ADAPTIVE PERFORMANCE TUNING
# =============================================================================

def example_adaptive_performance_tuning():
    """Demonstrate adaptive performance tuning."""
    
    print("\n🚀 Example 6: Adaptive Performance Tuning")
    print("=" * 60)
    
    # Initialize adaptive tuner
    config = OptimizationConfig(
        enable_adaptive_tuning=True,
        tuning_interval=5,  # Short interval for demo
        performance_history_size=50
    )
    
    adaptive_tuner = AdaptivePerformanceTuner(config)
    
    print("✅ Adaptive performance tuner initialized")
    
    # Show available strategies
    print(f"\n📋 Available optimization strategies:")
    for strategy in adaptive_tuner.optimization_strategies:
        print(f"  {strategy['name']}: {strategy['description']}")
        print(f"    Conditions: {strategy['conditions']}")
        print(f"    Actions: {strategy['actions']}")
        print()
    
    # Simulate performance monitoring
    print("📊 Simulating performance monitoring...")
    
    # Simulate different performance scenarios
    scenarios = [
        {"name": "Normal Performance", "cpu": 50, "memory": 60, "cache_hit": 0.8, "throughput": 100},
        {"name": "High CPU Usage", "cpu": 85, "memory": 70, "cache_hit": 0.6, "throughput": 80},
        {"name": "High Memory Usage", "cpu": 60, "memory": 85, "cache_hit": 0.7, "throughput": 90},
        {"name": "Low Cache Hit Rate", "cpu": 55, "memory": 65, "cache_hit": 0.4, "throughput": 70},
        {"name": "Optimal Performance", "cpu": 40, "memory": 50, "cache_hit": 0.9, "throughput": 120}
    ]
    
    for scenario in scenarios:
        print(f"\n🔄 Simulating {scenario['name']}...")
        
        # Create performance metrics
        from performance_optimizer import OptimizationMetrics
        
        metrics = OptimizationMetrics(
            processing_time=1.0 / scenario['throughput'],
            throughput=scenario['throughput'],
            cpu_usage=scenario['cpu'],
            memory_usage=scenario['memory'],
            cache_hit_rate=scenario['cache_hit'],
            optimization_score=0.8 if scenario['cache_hit'] > 0.8 else 0.6
        )
        
        # Record performance
        adaptive_tuner.record_performance(metrics)
        
        # Check if strategy changed
        if adaptive_tuner.current_strategy:
            print(f"  Active strategy: {adaptive_tuner.current_strategy['name']}")
        else:
            print("  No active strategy")
        
        # Wait for tuning interval
        time.sleep(1)
    
    # Show performance history
    print(f"\n📈 Performance History Analysis:")
    history = adaptive_tuner.performance_history
    
    if history:
        recent_metrics = list(history)[-10:]
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        print(f"  Average CPU usage: {avg_cpu:.1f}%")
        print(f"  Average memory usage: {avg_memory:.1f}%")
        print(f"  Average throughput: {avg_throughput:.1f}")
        print(f"  Average cache hit rate: {avg_cache_hit:.2%}")
    
    return adaptive_tuner

# =============================================================================
# EXAMPLE 7: REAL-WORLD VIDEO PROCESSING OPTIMIZATION
# =============================================================================

def example_video_processing_optimization():
    """Demonstrate real-world video processing optimization."""
    
    print("\n🚀 Example 7: Real-World Video Processing Optimization")
    print("=" * 60)
    
    # Initialize comprehensive optimizer
    config = OptimizationConfig(
        enable_smart_caching=True,
        enable_memory_optimization=True,
        enable_gpu_optimization=True,
        enable_load_balancing=True,
        enable_adaptive_tuning=True,
        max_workers=6,
        cache_size_limit=1000,
        memory_cleanup_threshold=75.0,
        mixed_precision=True
    )
    
    optimizer = PerformanceOptimizer(config)
    
    print("✅ Video processing optimizer initialized")
    
    # Simulate video processing pipeline
    class VideoProcessor:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.cache = optimizer.smart_cache
            self.memory_optimizer = optimizer.memory_optimizer
            self.gpu_optimizer = optimizer.gpu_optimizer
        
        @optimize_performance("video_loading")
        @cache_result(ttl=3600)
        def load_video(self, video_path: str):
            # Simulate video loading
            time.sleep(0.2)
            return {"path": video_path, "frames": 1000, "resolution": "1920x1080"}
        
        @optimize_performance("frame_extraction")
        def extract_frames(self, video_data: dict, frame_indices: List[int]):
            # Simulate frame extraction
            frames = []
            for idx in frame_indices:
                # Simulate frame processing
                frame = torch.randn(3, 1080, 1920)  # RGB frame
                self.memory_optimizer.register_tensor(frame, f"frame_{idx}")
                frames.append(frame)
                time.sleep(0.01)
            
            return frames
        
        @optimize_performance("frame_processing")
        def process_frames(self, frames: List[torch.Tensor]):
            # Simulate frame processing
            processed_frames = []
            
            for i, frame in enumerate(frames):
                # Simulate processing operations
                processed = torch.nn.functional.interpolate(
                    frame.unsqueeze(0), scale_factor=0.5
                ).squeeze(0)
                
                processed_frames.append(processed)
                time.sleep(0.02)
            
            return processed_frames
        
        @optimize_performance("video_encoding")
        @cache_result(ttl=1800)
        def encode_video(self, frames: List[torch.Tensor], output_path: str):
            # Simulate video encoding
            time.sleep(0.5)
            return {"output_path": output_path, "size_mb": 50.0}
    
    # Create video processor
    processor = VideoProcessor(optimizer)
    
    # Process multiple videos
    video_paths = [f"video_{i}.mp4" for i in range(5)]
    
    print("\n🎬 Processing videos with optimization...")
    
    for i, video_path in enumerate(video_paths):
        print(f"\n📹 Processing {video_path}...")
        
        # Load video
        video_data, load_metrics = processor.load_video(video_path)
        print(f"  Loaded video: {load_metrics.processing_time:.3f}s")
        
        # Extract frames
        frame_indices = list(range(0, 100, 10))  # Every 10th frame
        frames, extract_metrics = processor.extract_frames(video_data, frame_indices)
        print(f"  Extracted {len(frames)} frames: {extract_metrics.processing_time:.3f}s")
        
        # Process frames
        processed_frames, process_metrics = processor.process_frames(frames)
        print(f"  Processed frames: {process_metrics.processing_time:.3f}s")
        
        # Encode video
        output_path = f"processed_{video_path}"
        result, encode_metrics = processor.encode_video(processed_frames, output_path)
        print(f"  Encoded video: {encode_metrics.processing_time:.3f}s")
        
        # Clean up
        del frames, processed_frames
        
        # Force optimization if needed
        if i % 2 == 0:  # Every other video
            optimizer.force_optimization()
    
    # Get comprehensive optimization report
    report = optimizer.get_optimization_report()
    
    print(f"\n📊 Video Processing Optimization Report:")
    print(f"  Cache hit rate: {report['cache_performance']['hit_rate']:.2%}")
    print(f"  Memory usage: {report['memory_stats']['memory_percent']:.1f}%")
    print(f"  GPU memory: {report['gpu_stats'].get('memory_allocated_gb', 0):.2f} GB")
    print(f"  Average processing time: {report['performance_summary']['avg_processing_time']:.3f}s")
    print(f"  Optimization score: {report['performance_summary']['avg_optimization_score']:.2f}")
    
    return processor, optimizer

# =============================================================================
# EXAMPLE 8: BATCH PROCESSING WITH OPTIMIZATION
# =============================================================================

def example_batch_processing_optimization():
    """Demonstrate batch processing with optimization."""
    
    print("\n🚀 Example 8: Batch Processing with Optimization")
    print("=" * 60)
    
    # Initialize optimizer for batch processing
    config = OptimizationConfig(
        enable_smart_caching=True,
        enable_memory_optimization=True,
        enable_load_balancing=True,
        max_workers=8,
        chunk_size=50,
        enable_async_processing=True
    )
    
    optimizer = PerformanceOptimizer(config)
    
    print("✅ Batch processing optimizer initialized")
    
    # Simulate batch processing function
    @optimize_performance("batch_processing")
    def process_batch(batch_data: List[Dict], batch_id: int):
        # Simulate batch processing
        results = []
        
        for item in batch_data:
            # Simulate item processing
            processed_item = {
                "id": item["id"],
                "processed": True,
                "timestamp": datetime.now().isoformat(),
                "batch_id": batch_id
            }
            results.append(processed_item)
            time.sleep(0.01)  # Simulate processing time
        
        return results
    
    # Create sample batch data
    total_items = 1000
    batch_size = 100
    batches = []
    
    for i in range(0, total_items, batch_size):
        batch = [{"id": j} for j in range(i, min(i + batch_size, total_items))]
        batches.append(batch)
    
    print(f"📦 Created {len(batches)} batches with {batch_size} items each")
    
    # Process batches with load balancing
    print("\n🔄 Processing batches with load balancing...")
    
    all_results = []
    start_time = time.perf_counter()
    
    for batch_id, batch in enumerate(batches):
        # Get next available worker
        worker_id = optimizer.load_balancer.get_next_worker()
        
        # Process batch
        results, metrics = process_batch(batch, batch_id)
        all_results.extend(results)
        
        # Update worker statistics
        optimizer.load_balancer.update_worker_stats(
            worker_id=worker_id,
            load=len(batch) / batch_size,
            response_time=metrics.processing_time,
            error_rate=0.0
        )
        
        print(f"  Batch {batch_id}: {len(results)} items processed in {metrics.processing_time:.3f}s "
              f"(worker {worker_id})")
        
        # Force optimization every 5 batches
        if batch_id % 5 == 0:
            optimizer.force_optimization()
    
    total_time = time.perf_counter() - start_time
    
    print(f"\n📊 Batch Processing Results:")
    print(f"  Total items processed: {len(all_results)}")
    print(f"  Total processing time: {total_time:.3f}s")
    print(f"  Average time per item: {total_time / len(all_results):.3f}s")
    print(f"  Throughput: {len(all_results) / total_time:.1f} items/second")
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    print(f"  Cache hit rate: {report['cache_performance']['hit_rate']:.2%}")
    print(f"  Memory usage: {report['memory_stats']['memory_percent']:.1f}%")
    print(f"  Optimization score: {report['performance_summary']['avg_optimization_score']:.2f}")
    
    return all_results, optimizer

# =============================================================================
# EXAMPLE 9: PERFORMANCE MONITORING AND ANALYSIS
# =============================================================================

def example_performance_monitoring_analysis():
    """Demonstrate performance monitoring and analysis."""
    
    print("\n🚀 Example 9: Performance Monitoring and Analysis")
    print("=" * 60)
    
    # Initialize optimizer with monitoring
    config = OptimizationConfig(
        enable_performance_monitoring=True,
        monitoring_interval=0.5,
        performance_history_size=200
    )
    
    optimizer = PerformanceOptimizer(config)
    
    print("✅ Performance monitoring initialized")
    
    # Simulate performance monitoring
    print("\n📊 Simulating performance monitoring...")
    
    # Create monitoring function
    def monitor_performance():
        metrics = optimizer._collect_metrics()
        return {
            "timestamp": datetime.now(),
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "gpu_usage": metrics.gpu_usage,
            "cache_hit_rate": metrics.cache_hit_rate,
            "optimization_score": metrics.optimization_score
        }
    
    # Simulate workload with monitoring
    monitoring_data = []
    
    for i in range(20):
        # Simulate some work
        time.sleep(0.1)
        
        # Record performance
        performance_data = monitor_performance()
        monitoring_data.append(performance_data)
        
        # Force optimization occasionally
        if i % 5 == 0:
            optimizer.force_optimization()
        
        print(f"  Step {i}: CPU={performance_data['cpu_usage']:.1f}%, "
              f"Memory={performance_data['memory_usage']:.1f}%, "
              f"Cache={performance_data['cache_hit_rate']:.2%}")
    
    # Analyze performance trends
    print(f"\n📈 Performance Analysis:")
    
    if monitoring_data:
        # Calculate trends
        cpu_values = [d['cpu_usage'] for d in monitoring_data]
        memory_values = [d['memory_usage'] for d in monitoring_data]
        cache_values = [d['cache_hit_rate'] for d in monitoring_data]
        
        # Calculate statistics
        cpu_avg = np.mean(cpu_values)
        cpu_std = np.std(cpu_values)
        memory_avg = np.mean(memory_values)
        cache_avg = np.mean(cache_values)
        
        print(f"  CPU Usage: {cpu_avg:.1f}% ± {cpu_std:.1f}%")
        print(f"  Memory Usage: {memory_avg:.1f}%")
        print(f"  Cache Hit Rate: {cache_avg:.2%}")
        
        # Identify trends
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        
        print(f"  CPU Trend: {'↗️ Increasing' if cpu_trend > 0.1 else '↘️ Decreasing' if cpu_trend < -0.1 else '→ Stable'}")
        print(f"  Memory Trend: {'↗️ Increasing' if memory_trend > 0.1 else '↘️ Decreasing' if memory_trend < -0.1 else '→ Stable'}")
        
        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        
        if cpu_avg > 80:
            print("  ⚠️  High CPU usage detected - consider reducing workload or adding workers")
        
        if memory_avg > 85:
            print("  ⚠️  High memory usage detected - consider memory optimization")
        
        if cache_avg < 0.7:
            print("  ⚠️  Low cache hit rate - consider increasing cache size or TTL")
        
        if cpu_trend > 0.5:
            print("  ⚠️  CPU usage trending upward - monitor for potential bottlenecks")
    
    # Get final optimization report
    report = optimizer.get_optimization_report()
    print(f"\n📊 Final Optimization Report:")
    print(f"  Total optimizations: {report['performance_summary']['total_optimizations']}")
    print(f"  Average optimization score: {report['performance_summary']['avg_optimization_score']:.2f}")
    
    return monitoring_data, optimizer

# =============================================================================
# EXAMPLE 10: COMPREHENSIVE OPTIMIZATION INTEGRATION
# =============================================================================

def example_comprehensive_optimization_integration():
    """Demonstrate comprehensive optimization integration."""
    
    print("\n🚀 Example 10: Comprehensive Optimization Integration")
    print("=" * 60)
    
    # Create comprehensive configuration
    config = OptimizationConfig(
        # Enable all optimizations
        enable_smart_caching=True,
        enable_memory_optimization=True,
        enable_gpu_optimization=True,
        enable_load_balancing=True,
        enable_adaptive_tuning=True,
        enable_performance_monitoring=True,
        
        # Resource limits
        max_workers=8,
        max_cpu_usage=85.0,
        max_memory_usage=80.0,
        max_gpu_usage=90.0,
        
        # Caching
        cache_size_limit=2000,
        cache_ttl=7200,
        enable_predictive_caching=True,
        
        # Memory
        memory_cleanup_threshold=75.0,
        enable_garbage_collection=True,
        tensor_memory_fraction=0.8,
        
        # GPU
        mixed_precision=True,
        enable_cuda_graphs=True,
        
        # Load balancing
        load_balancing_strategy="adaptive",
        
        # Adaptive tuning
        tuning_interval=60,
        performance_history_size=500,
        
        # Monitoring
        monitoring_interval=1.0
    )
    
    # Initialize comprehensive optimizer
    optimizer = PerformanceOptimizer(config)
    
    print("✅ Comprehensive optimizer initialized")
    print(f"📋 Configuration: {config.__dict__}")
    
    # Simulate comprehensive workload
    print("\n🔄 Simulating comprehensive workload...")
    
    # Define workload types
    workload_types = [
        {"name": "Data Processing", "cpu_intensive": True, "memory_intensive": False},
        {"name": "Video Encoding", "cpu_intensive": True, "memory_intensive": True},
        {"name": "Model Inference", "cpu_intensive": False, "memory_intensive": True},
        {"name": "File I/O", "cpu_intensive": False, "memory_intensive": False}
    ]
    
    # Simulate different workloads
    for workload in workload_types:
        print(f"\n📊 Simulating {workload['name']}...")
        
        @optimize_performance(workload['name'].lower().replace(' ', '_'))
        def simulate_workload(workload_type: str, duration: float):
            start_time = time.perf_counter()
            
            # Simulate CPU-intensive work
            if workload_type['cpu_intensive']:
                for i in range(int(duration * 1000)):
                    _ = i ** 2
            
            # Simulate memory-intensive work
            if workload_type['memory_intensive']:
                large_data = [i for i in range(int(duration * 10000))]
                del large_data
            
            # Simulate I/O work
            if not workload_type['cpu_intensive'] and not workload_type['memory_intensive']:
                time.sleep(duration)
            
            return {"workload": workload_type['name'], "duration": duration}
        
        # Execute workload
        result, metrics = simulate_workload(workload, 0.5)
        
        print(f"  Completed in {metrics.processing_time:.3f}s")
        print(f"  CPU usage: {metrics.cpu_usage:.1f}%")
        print(f"  Memory usage: {metrics.memory_usage:.1f}%")
        print(f"  Optimization score: {metrics.optimization_score:.2f}")
    
    # Force comprehensive optimization
    print("\n🔧 Forcing comprehensive optimization...")
    optimization_result = optimizer.force_optimization()
    
    print("Optimization results:")
    for component, result in optimization_result.items():
        print(f"  {component}: {result}")
    
    # Get comprehensive report
    report = optimizer.get_optimization_report()
    
    print(f"\n📊 Comprehensive Optimization Report:")
    print(f"Cache Performance:")
    print(f"  Hit rate: {report['cache_performance']['hit_rate']:.2%}")
    print(f"  Cache size: {report['cache_performance']['cache_size']}")
    print(f"  Predictions: {report['cache_performance']['predictions']}")
    
    print(f"\nMemory Statistics:")
    print(f"  Memory usage: {report['memory_stats']['memory_percent']:.1f}%")
    print(f"  Used memory: {report['memory_stats']['used_memory_gb']:.2f} GB")
    print(f"  Optimization count: {report['memory_stats']['optimization_count']}")
    
    if 'error' not in report['gpu_stats']:
        print(f"\nGPU Statistics:")
        print(f"  Memory allocated: {report['gpu_stats']['memory_allocated_gb']:.2f} GB")
        print(f"  Memory reserved: {report['gpu_stats']['memory_reserved_gb']:.2f} GB")
    
    print(f"\nLoad Balancer:")
    print(f"  Workers: {report['load_balancer_stats']['worker_count']}")
    print(f"  Strategy: {report['load_balancer_stats']['strategy']}")
    
    print(f"\nPerformance Summary:")
    print(f"  Average processing time: {report['performance_summary']['avg_processing_time']:.3f}s")
    print(f"  Average throughput: {report['performance_summary']['avg_throughput']:.1f}")
    print(f"  Average optimization score: {report['performance_summary']['avg_optimization_score']:.2f}")
    print(f"  Total optimizations: {report['performance_summary']['total_optimizations']}")
    
    if report['current_strategy']:
        print(f"\nCurrent Strategy: {report['current_strategy']['name']}")
        print(f"Description: {report['current_strategy']['description']}")
    
    return optimizer

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all performance optimization examples."""
    
    print("🚀 Performance Optimization Examples for Video-OpusClip")
    print("=" * 80)
    
    examples = [
        ("Basic Optimization Setup", example_basic_optimization_setup),
        ("Smart Caching Strategies", example_smart_caching_strategies),
        ("Memory Optimization Patterns", example_memory_optimization_patterns),
        ("GPU Optimization Techniques", example_gpu_optimization_techniques),
        ("Load Balancing Implementations", example_load_balancing_implementations),
        ("Adaptive Performance Tuning", example_adaptive_performance_tuning),
        ("Video Processing Optimization", example_video_processing_optimization),
        ("Batch Processing Optimization", example_batch_processing_optimization),
        ("Performance Monitoring Analysis", example_performance_monitoring_analysis),
        ("Comprehensive Optimization Integration", example_comprehensive_optimization_integration)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    print(f"\n{'='*80}")
    print("🎉 Performance Optimization Examples Summary")
    print(f"Successful examples: {sum(1 for r in results.values() if r is not None)}/{len(examples)}")
    
    for name, result in results.items():
        status = "✅ PASS" if result is not None else "❌ FAIL"
        print(f"  {name}: {status}")
    
    return results

def run_specific_example(example_name: str):
    """Run a specific example by name."""
    
    example_map = {
        "basic": example_basic_optimization_setup,
        "caching": example_smart_caching_strategies,
        "memory": example_memory_optimization_patterns,
        "gpu": example_gpu_optimization_techniques,
        "load_balancing": example_load_balancing_implementations,
        "adaptive": example_adaptive_performance_tuning,
        "video": example_video_processing_optimization,
        "batch": example_batch_processing_optimization,
        "monitoring": example_performance_monitoring_analysis,
        "comprehensive": example_comprehensive_optimization_integration
    }
    
    if example_name in example_map:
        print(f"🚀 Running {example_name} example...")
        return example_map[example_name]()
    else:
        print(f"❌ Example '{example_name}' not found. Available examples: {list(example_map.keys())}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1]
        run_specific_example(example_name)
    else:
        # Run all examples
        run_all_examples() 