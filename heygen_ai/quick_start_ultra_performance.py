#!/usr/bin/env python3
"""
Quick Start Ultra Performance Optimizer for HeyGen AI

This script demonstrates how to use the ultra-performance optimizer
for maximum speed and efficiency. Perfect for getting started quickly!
"""

import logging
import sys
import time
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
try:
    import torch
    import torch.nn as nn
    import numpy as np
    
    # Import our ultra performance optimizer
    from ultra_performance_optimizer import (
        UltraPerformanceOptimizer, UltraPerformanceConfig,
        create_ultra_performance_optimizer,
        create_maximum_performance_config,
        create_balanced_performance_config,
        create_memory_efficient_config
    )
    
    MODULES_AVAILABLE = True
    logger.info("✅ Ultra Performance Optimizer imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Could not import Ultra Performance Optimizer: {e}")
    MODULES_AVAILABLE = False


def create_sample_model() -> nn.Module:
    """Create a sample model for demonstration."""
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    # Initialize weights properly
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    return model


def demonstrate_basic_optimization():
    """Demonstrate basic model optimization."""
    logger.info("🚀 Demonstrating Basic Model Optimization...")
    
    try:
        # Create sample model
        model = create_sample_model()
        logger.info(f"✅ Sample model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create maximum performance optimizer
        config = create_maximum_performance_config()
        optimizer = create_ultra_performance_optimizer(**config.__dict__)
        
        # Generate sample input
        sample_input = torch.randn(32, 512)
        
        # Benchmark original model
        logger.info("📊 Benchmarking original model...")
        original_benchmark = optimizer.performance_profiler.benchmark_model(
            model, sample_input, num_runs=20, warmup_runs=5
        )
        
        if original_benchmark["status"] == "success":
            original_result = original_benchmark["benchmark_result"]
            logger.info(f"  Original model: {original_result['avg_inference_time_ms']:.2f}ms, "
                       f"Throughput: {original_result['throughput_samples_per_sec']:.1f} samples/sec")
        
        # Optimize the model
        logger.info("⚡ Applying ultra-performance optimizations...")
        optimized_model = optimizer.optimize_model(model, "sample_model")
        
        # Benchmark optimized model
        logger.info("📊 Benchmarking optimized model...")
        optimized_benchmark = optimizer.performance_profiler.benchmark_model(
            optimized_model, sample_input, num_runs=20, warmup_runs=5
        )
        
        if optimized_benchmark["status"] == "success":
            optimized_result = optimized_benchmark["benchmark_result"]
            logger.info(f"  Optimized model: {optimized_result['avg_inference_time_ms']:.2f}ms, "
                       f"Throughput: {optimized_result['throughput_samples_per_sec']:.1f} samples/sec")
            
            # Calculate improvements
            if original_benchmark["status"] == "success":
                speedup = original_result["avg_inference_time_ms"] / optimized_result["avg_inference_time_ms"]
                throughput_improvement = optimized_result["throughput_samples_per_sec"] / original_result["throughput_samples_per_sec"]
                
                logger.info(f"🚀 Performance Improvements:")
                logger.info(f"  Speedup: {speedup:.2f}x")
                logger.info(f"  Throughput improvement: {throughput_improvement:.2f}x")
        
        # Get optimization summary
        summary = optimizer.get_optimization_summary()
        logger.info(f"📋 Optimization Summary: {summary['total_models_optimized']} models optimized")
        
        # Cleanup
        optimizer.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic optimization demonstration failed: {e}")
        return False


def demonstrate_configuration_comparison():
    """Demonstrate different optimization configurations."""
    logger.info("⚙️ Demonstrating Configuration Comparison...")
    
    try:
        # Create sample model
        model = create_sample_model()
        sample_input = torch.randn(16, 512)
        
        # Test different configurations
        configs = {
            "Maximum Performance": create_maximum_performance_config(),
            "Balanced": create_balanced_performance_config(),
            "Memory Efficient": create_memory_efficient_config()
        }
        
        results = {}
        
        for config_name, config in configs.items():
            logger.info(f"🔧 Testing {config_name} configuration...")
            
            try:
                # Create optimizer with this configuration
                optimizer = create_ultra_performance_optimizer(**config.__dict__)
                
                # Benchmark original model
                original_benchmark = optimizer.performance_profiler.benchmark_model(
                    model, sample_input, num_runs=15, warmup_runs=3
                )
                
                # Optimize model
                optimized_model = optimizer.optimize_model(model, f"test_model_{config_name}")
                
                # Benchmark optimized model
                optimized_benchmark = optimizer.performance_profiler.benchmark_model(
                    optimized_model, sample_input, num_runs=15, warmup_runs=3
                )
                
                # Calculate improvements
                if (original_benchmark["status"] == "success" and 
                    optimized_benchmark["status"] == "success"):
                    
                    original_result = original_benchmark["benchmark_result"]
                    optimized_result = optimized_benchmark["benchmark_result"]
                    
                    speedup = original_result["avg_inference_time_ms"] / optimized_result["avg_inference_time_ms"]
                    throughput_improvement = optimized_result["throughput_samples_per_sec"] / original_result["throughput_samples_per_sec"]
                    
                    results[config_name] = {
                        "speedup": speedup,
                        "throughput_improvement": throughput_improvement,
                        "memory_usage": optimized_result["avg_memory_delta_mb"]
                    }
                    
                    logger.info(f"  {config_name}: {speedup:.2f}x speedup, {throughput_improvement:.2f}x throughput")
                
                # Cleanup
                optimizer.cleanup()
                
            except Exception as e:
                logger.warning(f"Failed to test {config_name}: {e}")
                results[config_name] = {"error": str(e)}
        
        # Print comparison summary
        logger.info("\n📊 Configuration Comparison Summary:")
        for config_name, result in results.items():
            if "error" not in result:
                logger.info(f"  {config_name}: {result['speedup']:.2f}x speedup, "
                           f"{result['throughput_improvement']:.2f}x throughput, "
                           f"{result['memory_usage']:+.2f}MB memory")
            else:
                logger.info(f"  {config_name}: Failed - {result['error']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration comparison failed: {e}")
        return False


def demonstrate_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    logger.info("💾 Demonstrating Memory Optimization...")
    
    try:
        # Create a larger model to demonstrate memory optimization
        large_model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Initialize weights
        for module in large_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        logger.info(f"✅ Large model created with {sum(p.numel() for p in large_model.parameters()):,} parameters")
        
        # Create memory-efficient optimizer
        config = create_memory_efficient_config()
        optimizer = create_ultra_performance_optimizer(**config.__dict__)
        
        # Get initial memory stats
        initial_memory = optimizer.memory_optimizer.get_memory_stats()
        logger.info(f"📊 Initial memory usage: {initial_memory.get('allocated_mb', 0):.2f}MB")
        
        # Optimize memory usage
        memory_result = optimizer.memory_optimizer.optimize_memory_usage()
        logger.info(f"🔧 Memory optimization: {memory_result['status']}")
        
        if memory_result['status'] == 'success':
            memory_stats = memory_result['memory_stats']
            logger.info(f"  Current memory: {memory_stats['current_memory_mb']:.2f}MB")
            logger.info(f"  Max memory: {memory_stats['max_memory_mb']:.2f}MB")
            logger.info(f"  Usage: {memory_stats['memory_usage_percent']:.1f}%")
        
        # Test with sample input
        sample_input = torch.randn(8, 1024)
        
        # Benchmark with memory profiling
        logger.info("📊 Benchmarking with memory profiling...")
        benchmark_result = optimizer.performance_profiler.benchmark_model(
            large_model, sample_input, num_runs=10, warmup_runs=2
        )
        
        if benchmark_result["status"] == "success":
            result = benchmark_result["benchmark_result"]
            logger.info(f"  Inference time: {result['avg_inference_time_ms']:.2f}ms")
            logger.info(f"  Memory delta: {result['avg_memory_delta_mb']:+.2f}MB")
            logger.info(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
        
        # Get final memory stats
        final_memory = optimizer.memory_optimizer.get_memory_stats()
        logger.info(f"📊 Final memory usage: {final_memory.get('allocated_mb', 0):.2f}MB")
        
        # Cleanup
        optimizer.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory optimization demonstration failed: {e}")
        return False


def demonstrate_dynamic_batching():
    """Demonstrate dynamic batch size optimization."""
    logger.info("🔄 Demonstrating Dynamic Batch Size Optimization...")
    
    try:
        # Create sample model
        model = create_sample_model()
        
        # Create maximum performance optimizer
        config = create_maximum_performance_config()
        optimizer = create_ultra_performance_optimizer(**config.__dict__)
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"📦 Testing batch size: {batch_size}")
            
            try:
                # Create input with this batch size
                sample_input = torch.randn(batch_size, 512)
                
                # Benchmark this batch size
                benchmark_result = optimizer.performance_profiler.benchmark_model(
                    model, sample_input, num_runs=10, warmup_runs=2
                )
                
                if benchmark_result["status"] == "success":
                    result = benchmark_result["benchmark_result"]
                    throughput = result["throughput_samples_per_sec"]
                    
                    results[batch_size] = {
                        "throughput": throughput,
                        "inference_time": result["avg_inference_time_ms"],
                        "memory_delta": result["avg_memory_delta_mb"]
                    }
                    
                    logger.info(f"  Batch {batch_size}: {throughput:.1f} samples/sec, "
                               f"{result['avg_inference_time_ms']:.2f}ms, "
                               f"{result['avg_memory_delta_mb']:+.2f}MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info(f"  Batch {batch_size}: OOM - stopping batch size testing")
                    break
                else:
                    logger.warning(f"  Batch {batch_size}: Error - {e}")
        
        # Find optimal batch size
        if results:
            optimal_batch_size = max(results.keys(), key=lambda x: results[x]["throughput"])
            optimal_throughput = results[optimal_batch_size]["throughput"]
            
            logger.info(f"🚀 Optimal batch size: {optimal_batch_size} with {optimal_throughput:.1f} samples/sec")
            
            # Test dynamic batch optimization
            logger.info("🔄 Testing dynamic batch optimization...")
            current_batch_size = 8
            optimized_batch_size = optimizer.batch_optimizer.optimize_batch_size(
                model, torch.randn(current_batch_size, 512), current_batch_size
            )
            
            logger.info(f"  Dynamic optimization: {current_batch_size} -> {optimized_batch_size}")
        
        # Cleanup
        optimizer.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dynamic batching demonstration failed: {e}")
        return False


def main():
    """Main demonstration function."""
    if not MODULES_AVAILABLE:
        logger.error("❌ Ultra Performance Optimizer not available. Please install dependencies.")
        return
    
    logger.info("🚀 HeyGen AI - Ultra Performance Optimizer Quick Start")
    logger.info("=" * 60)
    
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            logger.warning("⚠️ CUDA not available - some optimizations will be limited")
        
        # Run demonstrations
        demonstrations = [
            ("Basic Model Optimization", demonstrate_basic_optimization),
            ("Configuration Comparison", demonstrate_configuration_comparison),
            ("Memory Optimization", demonstrate_memory_optimization),
            ("Dynamic Batch Optimization", demonstrate_dynamic_batching)
        ]
        
        successful_demos = 0
        total_demos = len(demonstrations)
        
        for demo_name, demo_func in demonstrations:
            logger.info(f"\n{'='*20} {demo_name} {'='*20}")
            try:
                if demo_func():
                    successful_demos += 1
                    logger.info(f"✅ {demo_name} completed successfully")
                else:
                    logger.error(f"❌ {demo_name} failed")
            except Exception as e:
                logger.error(f"❌ {demo_name} failed with exception: {e}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Quick Start Summary: {successful_demos}/{total_demos} demonstrations successful")
        
        if successful_demos == total_demos:
            logger.info("🚀 All demonstrations completed successfully!")
            logger.info("💡 You're ready to use the Ultra Performance Optimizer!")
        else:
            logger.warning(f"⚠️ {total_demos - successful_demos} demonstrations failed")
            logger.info("🔧 Check the logs above for error details")
        
        logger.info("\n📚 Next Steps:")
        logger.info("   1. Run 'python ultra_performance_benchmark.py' for comprehensive benchmarking")
        logger.info("   2. Integrate the optimizer into your existing models")
        logger.info("   3. Experiment with different optimization configurations")
        logger.info("   4. Monitor performance improvements in your applications")
        
    except Exception as e:
        logger.error(f"❌ Quick start failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

