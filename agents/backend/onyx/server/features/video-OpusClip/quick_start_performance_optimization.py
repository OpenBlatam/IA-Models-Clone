#!/usr/bin/env python3
"""
Quick Start Performance Optimization for Video-OpusClip

This script demonstrates how to quickly enable and use performance optimization
features in the Video-OpusClip system.
"""

import time
import psutil
import threading
from typing import Dict, List, Any
import structlog

# Setup logging
structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso")])
logger = structlog.get_logger()

def quick_start_basic_optimization():
    """Quick start example for basic performance optimization."""
    
    print("🚀 Quick Start: Basic Performance Optimization")
    print("=" * 50)
    
    try:
        # Import performance optimization components
        from performance_optimizer import PerformanceOptimizer, OptimizationConfig
        
        # Create basic configuration
        config = OptimizationConfig(
            enable_smart_caching=True,
            enable_memory_optimization=True,
            enable_gpu_optimization=True,
            max_workers=4,
            cache_size_limit=500
        )
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer(config)
        
        print("✅ Performance optimizer initialized")
        
        # Simple optimized function
        @optimizer.optimize_operation("simple_processing")
        def simple_processing(data: List[float]) -> float:
            # Simulate processing
            time.sleep(0.1)
            return sum(data) / len(data)
        
        # Test optimization
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result, metrics = simple_processing(test_data)
        
        print(f"📊 Processing result: {result}")
        print(f"⏱️  Processing time: {metrics.processing_time:.3f}s")
        print(f"💾 Memory usage: {metrics.memory_usage:.1f}%")
        print(f"🎯 Optimization score: {metrics.optimization_score:.2f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Performance optimization tools not available: {e}")
        print("Please ensure performance_optimizer.py is in your Python path")
        return False
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return False

def quick_start_caching():
    """Quick start example for smart caching."""
    
    print("\n🚀 Quick Start: Smart Caching")
    print("=" * 50)
    
    try:
        from performance_optimizer import SmartCache, OptimizationConfig
        
        # Initialize cache
        config = OptimizationConfig(
            enable_smart_caching=True,
            cache_size_limit=100,
            enable_predictive_caching=True
        )
        cache = SmartCache(config)
        
        print("✅ Smart cache initialized")
        
        # Cache some data
        cache.set("user_data", {"name": "John", "age": 30}, ttl=3600)
        cache.set("session_data", {"session_id": "12345"}, ttl=300)
        cache.set("static_data", {"version": "1.0.0"}, ttl=86400)
        
        print("📦 Data cached successfully")
        
        # Access cached data
        user_data = cache.get("user_data")
        session_data = cache.get("session_data")
        static_data = cache.get("static_data")
        
        print(f"👤 User data: {user_data}")
        print(f"🔐 Session data: {session_data}")
        print(f"📋 Static data: {static_data}")
        
        # Get cache statistics
        stats = cache.get_stats()
        print(f"📊 Cache hit rate: {stats['hit_rate']:.2%}")
        print(f"📦 Cache size: {stats['cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during caching: {e}")
        return False

def quick_start_memory_optimization():
    """Quick start example for memory optimization."""
    
    print("\n🚀 Quick Start: Memory Optimization")
    print("=" * 50)
    
    try:
        from performance_optimizer import MemoryOptimizer, OptimizationConfig
        
        # Initialize memory optimizer
        config = OptimizationConfig(
            enable_memory_optimization=True,
            memory_cleanup_threshold=70.0
        )
        memory_optimizer = MemoryOptimizer(config)
        
        print("✅ Memory optimizer initialized")
        
        # Get initial memory stats
        initial_stats = memory_optimizer.get_memory_stats()
        print(f"📊 Initial memory usage: {initial_stats['memory_percent']:.1f}%")
        
        # Simulate memory-intensive operations
        large_lists = []
        for i in range(5):
            large_list = [i for i in range(100000)]
            large_lists.append(large_list)
            
            stats = memory_optimizer.get_memory_stats()
            print(f"  Created list {i}: Memory usage {stats['memory_percent']:.1f}%")
        
        # Force memory optimization
        print("\n🧹 Optimizing memory...")
        optimization_result = memory_optimizer.optimize_memory()
        print(f"💾 Memory saved: {optimization_result['memory_saved_mb']:.2f} MB")
        
        # Clean up
        del large_lists
        
        # Final memory stats
        final_stats = memory_optimizer.get_memory_stats()
        print(f"📊 Final memory usage: {final_stats['memory_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during memory optimization: {e}")
        return False

def quick_start_gpu_optimization():
    """Quick start example for GPU optimization."""
    
    print("\n🚀 Quick Start: GPU Optimization")
    print("=" * 50)
    
    try:
        import torch
        from performance_optimizer import GPUOptimizer, OptimizationConfig
        
        if not torch.cuda.is_available():
            print("❌ GPU not available, skipping GPU optimization example")
            return False
        
        # Initialize GPU optimizer
        config = OptimizationConfig(
            enable_gpu_optimization=True,
            mixed_precision=True
        )
        gpu_optimizer = GPUOptimizer(config)
        
        print("✅ GPU optimizer initialized")
        
        # Get GPU stats
        stats = gpu_optimizer.get_gpu_stats()
        print(f"📊 GPU: {stats['device_name']}")
        print(f"💾 GPU memory: {stats['memory_allocated_gb']:.2f} GB")
        
        # Create GPU tensors
        tensors = []
        for i in range(3):
            tensor = torch.randn(1000, 1000, device='cuda')
            tensors.append(tensor)
            
            stats = gpu_optimizer.get_gpu_stats()
            print(f"  Created tensor {i}: GPU memory {stats['memory_allocated_gb']:.2f} GB")
        
        # Optimize GPU memory
        print("\n🧹 Optimizing GPU memory...")
        optimization_result = gpu_optimizer.optimize_gpu_memory()
        print(f"💾 GPU memory saved: {optimization_result['memory_saved_mb']:.2f} MB")
        
        # Clean up
        del tensors
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during GPU optimization: {e}")
        return False

def quick_start_load_balancing():
    """Quick start example for load balancing."""
    
    print("\n🚀 Quick Start: Load Balancing")
    print("=" * 50)
    
    try:
        from performance_optimizer import LoadBalancer, OptimizationConfig
        
        # Initialize load balancer
        config = OptimizationConfig(
            enable_load_balancing=True,
            max_workers=4,
            load_balancing_strategy="adaptive"
        )
        load_balancer = LoadBalancer(config)
        
        print("✅ Load balancer initialized")
        print(f"👥 Workers: {len(load_balancer.workers)}")
        print(f"⚖️  Strategy: {config.load_balancing_strategy}")
        
        # Simulate workload distribution
        workloads = [10, 5, 15, 8, 12, 6, 20, 3]
        
        print("\n🔄 Distributing workloads...")
        for i, workload_size in enumerate(workloads):
            worker_id = load_balancer.get_next_worker()
            
            # Simulate processing time
            processing_time = workload_size * 0.1
            
            # Update worker stats
            load_balancer.update_worker_stats(
                worker_id=worker_id,
                load=workload_size / 20.0,
                response_time=processing_time,
                error_rate=0.0
            )
            
            print(f"  Workload {i} (size {workload_size}): worker {worker_id}")
        
        # Show worker statistics
        print(f"\n📊 Worker Statistics:")
        for worker_id, stats in load_balancer.worker_stats.items():
            print(f"  Worker {worker_id}: load={stats['load']:.2f}, "
                  f"response_time={stats['response_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during load balancing: {e}")
        return False

def quick_start_adaptive_tuning():
    """Quick start example for adaptive performance tuning."""
    
    print("\n🚀 Quick Start: Adaptive Performance Tuning")
    print("=" * 50)
    
    try:
        from performance_optimizer import AdaptivePerformanceTuner, OptimizationConfig, OptimizationMetrics
        
        # Initialize adaptive tuner
        config = OptimizationConfig(
            enable_adaptive_tuning=True,
            tuning_interval=5,
            performance_history_size=50
        )
        adaptive_tuner = AdaptivePerformanceTuner(config)
        
        print("✅ Adaptive performance tuner initialized")
        
        # Show available strategies
        print(f"\n📋 Available strategies:")
        for strategy in adaptive_tuner.optimization_strategies:
            print(f"  {strategy['name']}: {strategy['description']}")
        
        # Simulate performance monitoring
        print(f"\n📊 Simulating performance monitoring...")
        
        scenarios = [
            {"cpu": 50, "memory": 60, "cache_hit": 0.8},
            {"cpu": 85, "memory": 70, "cache_hit": 0.6},
            {"cpu": 60, "memory": 85, "cache_hit": 0.7},
            {"cpu": 40, "memory": 50, "cache_hit": 0.9}
        ]
        
        for i, scenario in enumerate(scenarios):
            metrics = OptimizationMetrics(
                processing_time=0.1,
                throughput=100.0,
                cpu_usage=scenario["cpu"],
                memory_usage=scenario["memory"],
                cache_hit_rate=scenario["cache_hit"]
            )
            
            adaptive_tuner.record_performance(metrics)
            
            if adaptive_tuner.current_strategy:
                print(f"  Scenario {i}: {adaptive_tuner.current_strategy['name']}")
            else:
                print(f"  Scenario {i}: No active strategy")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during adaptive tuning: {e}")
        return False

def quick_start_video_processing():
    """Quick start example for video processing optimization."""
    
    print("\n🚀 Quick Start: Video Processing Optimization")
    print("=" * 50)
    
    try:
        from performance_optimizer import PerformanceOptimizer, OptimizationConfig
        
        # Initialize optimizer for video processing
        config = OptimizationConfig(
            enable_smart_caching=True,
            enable_memory_optimization=True,
            enable_gpu_optimization=True,
            enable_load_balancing=True,
            max_workers=4,
            cache_size_limit=1000
        )
        
        optimizer = PerformanceOptimizer(config)
        
        print("✅ Video processing optimizer initialized")
        
        # Simulate video processing pipeline
        def process_video(video_path: str, quality: str = "high"):
            # Simulate video processing
            time.sleep(0.2)
            return {"processed": True, "path": video_path, "quality": quality}
        
        # Process videos with optimization
        video_paths = [f"video_{i}.mp4" for i in range(5)]
        
        print("\n🎬 Processing videos...")
        for video_path in video_paths:
            result, metrics = optimizer.optimize_operation("video_processing", process_video, video_path, "high")
            print(f"  {video_path}: {metrics.processing_time:.3f}s")
        
        # Get optimization report
        report = optimizer.get_optimization_report()
        print(f"\n📊 Optimization Report:")
        print(f"  Cache hit rate: {report['cache_performance']['hit_rate']:.2%}")
        print(f"  Memory usage: {report['memory_stats']['memory_percent']:.1f}%")
        print(f"  Optimization score: {report['performance_summary']['avg_optimization_score']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        return False

def quick_start_batch_processing():
    """Quick start example for batch processing optimization."""
    
    print("\n🚀 Quick Start: Batch Processing Optimization")
    print("=" * 50)
    
    try:
        from performance_optimizer import PerformanceOptimizer, OptimizationConfig
        
        # Initialize optimizer for batch processing
        config = OptimizationConfig(
            enable_smart_caching=True,
            enable_load_balancing=True,
            max_workers=6,
            chunk_size=50
        )
        
        optimizer = PerformanceOptimizer(config)
        
        print("✅ Batch processing optimizer initialized")
        
        # Simulate batch processing
        def process_batch(batch_data: list, batch_id: int):
            # Simulate batch processing
            results = []
            for item in batch_data:
                processed_item = {"id": item, "processed": True, "batch_id": batch_id}
                results.append(processed_item)
                time.sleep(0.01)
            return results
        
        # Create sample batches
        total_items = 200
        batch_size = 50
        batches = []
        
        for i in range(0, total_items, batch_size):
            batch = list(range(i, min(i + batch_size, total_items)))
            batches.append(batch)
        
        print(f"📦 Created {len(batches)} batches with {batch_size} items each")
        
        # Process batches
        print("\n🔄 Processing batches...")
        start_time = time.perf_counter()
        
        all_results = []
        for batch_id, batch in enumerate(batches):
            results, metrics = optimizer.optimize_operation("batch_processing", process_batch, batch, batch_id)
            all_results.extend(results)
            print(f"  Batch {batch_id}: {len(results)} items in {metrics.processing_time:.3f}s")
        
        total_time = time.perf_counter() - start_time
        
        print(f"\n📊 Batch Processing Results:")
        print(f"  Total items: {len(all_results)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {len(all_results) / total_time:.1f} items/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        return False

def quick_start_comprehensive_optimization():
    """Quick start example for comprehensive optimization."""
    
    print("\n🚀 Quick Start: Comprehensive Optimization")
    print("=" * 50)
    
    try:
        from performance_optimizer import PerformanceOptimizer, OptimizationConfig
        
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
            max_workers=6,
            max_cpu_usage=85.0,
            max_memory_usage=80.0,
            
            # Caching
            cache_size_limit=1000,
            enable_predictive_caching=True,
            
            # Memory
            memory_cleanup_threshold=75.0,
            
            # Load balancing
            load_balancing_strategy="adaptive",
            
            # Adaptive tuning
            tuning_interval=30,
            performance_history_size=100
        )
        
        # Initialize comprehensive optimizer
        optimizer = PerformanceOptimizer(config)
        
        print("✅ Comprehensive optimizer initialized")
        
        # Simulate comprehensive workload
        def comprehensive_workload(workload_type: str, duration: float):
            start_time = time.perf_counter()
            
            # Simulate different types of work
            if "cpu" in workload_type:
                for i in range(int(duration * 1000)):
                    _ = i ** 2
            
            if "memory" in workload_type:
                large_data = [i for i in range(int(duration * 5000))]
                del large_data
            
            if "io" in workload_type:
                time.sleep(duration)
            
            return {"workload": workload_type, "duration": duration}
        
        # Execute different workloads
        workloads = [
            ("cpu_intensive", 0.3),
            ("memory_intensive", 0.2),
            ("io_intensive", 0.1),
            ("mixed_workload", 0.4)
        ]
        
        print("\n🔄 Executing comprehensive workloads...")
        for workload_type, duration in workloads:
            result, metrics = optimizer.optimize_operation("comprehensive", comprehensive_workload, workload_type, duration)
            print(f"  {workload_type}: {metrics.processing_time:.3f}s, "
                  f"CPU={metrics.cpu_usage:.1f}%, Memory={metrics.memory_usage:.1f}%")
        
        # Force optimization
        print("\n🔧 Forcing optimization...")
        optimization_result = optimizer.force_optimization()
        print("Optimization completed")
        
        # Get comprehensive report
        report = optimizer.get_optimization_report()
        
        print(f"\n📊 Comprehensive Report:")
        print(f"  Cache hit rate: {report['cache_performance']['hit_rate']:.2%}")
        print(f"  Memory usage: {report['memory_stats']['memory_percent']:.1f}%")
        print(f"  Workers: {report['load_balancer_stats']['worker_count']}")
        print(f"  Optimization score: {report['performance_summary']['avg_optimization_score']:.2f}")
        print(f"  Total optimizations: {report['performance_summary']['total_optimizations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during comprehensive optimization: {e}")
        return False

def main():
    """Run all quick start examples."""
    
    print("🚀 Performance Optimization Quick Start for Video-OpusClip")
    print("=" * 70)
    
    examples = [
        ("Basic Optimization", quick_start_basic_optimization),
        ("Smart Caching", quick_start_caching),
        ("Memory Optimization", quick_start_memory_optimization),
        ("GPU Optimization", quick_start_gpu_optimization),
        ("Load Balancing", quick_start_load_balancing),
        ("Adaptive Tuning", quick_start_adaptive_tuning),
        ("Video Processing", quick_start_video_processing),
        ("Batch Processing", quick_start_batch_processing),
        ("Comprehensive Optimization", quick_start_comprehensive_optimization)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = example_func()
            results[name] = result
            if result:
                print(f"✅ {name} completed successfully")
            else:
                print(f"❌ {name} failed")
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results[name] = False
    
    print(f"\n{'='*70}")
    print("🎉 Performance Optimization Quick Start Summary")
    print(f"Successful examples: {sum(1 for r in results.values() if r)}/{len(examples)}")
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n📚 For detailed usage, see PERFORMANCE_OPTIMIZATION_GUIDE.md")
    print(f"🔧 For examples, see performance_optimization_examples.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1].lower()
        
        if example_name == "basic":
            quick_start_basic_optimization()
        elif example_name == "caching":
            quick_start_caching()
        elif example_name == "memory":
            quick_start_memory_optimization()
        elif example_name == "gpu":
            quick_start_gpu_optimization()
        elif example_name == "load_balancing":
            quick_start_load_balancing()
        elif example_name == "adaptive":
            quick_start_adaptive_tuning()
        elif example_name == "video":
            quick_start_video_processing()
        elif example_name == "batch":
            quick_start_batch_processing()
        elif example_name == "comprehensive":
            quick_start_comprehensive_optimization()
        else:
            print(f"❌ Unknown example: {example_name}")
            print("Available examples: basic, caching, memory, gpu, load_balancing, adaptive, video, batch, comprehensive")
    else:
        # Run all examples
        main() 