#!/usr/bin/env python3
"""
Ultra Performance Benchmark for HeyGen AI

This script demonstrates the ultra-performance optimization capabilities:
- Comprehensive model benchmarking
- Performance comparison between original and optimized models
- Memory usage analysis
- Throughput optimization
- Real-time performance monitoring
"""

import asyncio
import logging
import sys
import time
import os
from pathlib import Path
import warnings
import json
from typing import Dict, List, Any, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ultra_performance_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

# Import required modules
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    
    # Import our enhanced modules
    from enhanced_transformer_models import (
        TransformerManager, TransformerConfig,
        create_transformer_manager, create_gpt2_config
    )
    from enhanced_diffusion_models import (
        DiffusionPipelineManager, DiffusionConfig,
        create_diffusion_manager, create_stable_diffusion_config
    )
    from ultra_performance_optimizer import (
        UltraPerformanceOptimizer, UltraPerformanceConfig,
        create_ultra_performance_optimizer,
        create_maximum_performance_config,
        create_balanced_performance_config,
        create_memory_efficient_config
    )
    
    MODULES_AVAILABLE = True
    logger.info("✅ All required modules imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Could not import required modules: {e}")
    MODULES_AVAILABLE = False


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.logger = logger
        self.benchmark_results = {}
        self.optimization_results = {}
        
        # Initialize performance configurations
        self.performance_configs = {
            "maximum": create_maximum_performance_config(),
            "balanced": create_balanced_performance_config(),
            "memory_efficient": create_memory_efficient_config()
        }
        
        # Initialize optimizers
        self.optimizers = {}
        for name, config in self.performance_configs.items():
            self.optimizers[name] = create_ultra_performance_optimizer(**config.__dict__)
        
        logger.info("🚀 Performance Benchmark Suite initialized")
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarking."""
        logger.info("🚀 Starting comprehensive performance benchmark...")
        
        try:
            benchmark_start = time.time()
            
            # 1. Benchmark transformer models
            transformer_results = await self._benchmark_transformer_models()
            
            # 2. Benchmark diffusion models
            diffusion_results = await self._benchmark_diffusion_models()
            
            # 3. Benchmark optimization impact
            optimization_results = await self._benchmark_optimization_impact()
            
            # 4. Generate performance report
            performance_report = self._generate_performance_report(
                transformer_results, diffusion_results, optimization_results
            )
            
            benchmark_duration = time.time() - benchmark_start
            
            final_results = {
                "benchmark_duration_seconds": benchmark_duration,
                "transformer_results": transformer_results,
                "diffusion_results": diffusion_results,
                "optimization_results": optimization_results,
                "performance_report": performance_report,
                "timestamp": time.time()
            }
            
            self.benchmark_results = final_results
            
            logger.info(f"✅ Comprehensive benchmark completed in {benchmark_duration:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive benchmark failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _benchmark_transformer_models(self) -> Dict[str, Any]:
        """Benchmark transformer model performance."""
        logger.info("📊 Benchmarking transformer models...")
        
        try:
            results = {}
            
            # Test different transformer configurations
            configs = [
                ("GPT-2 Small", create_gpt2_config()),
                ("GPT-2 Medium", create_gpt2_config(hidden_size=1024, num_attention_heads=16)),
                ("BERT Base", create_gpt2_config(model_type="encoder", model_name="bert-base-uncased"))
            ]
            
            for config_name, config in configs:
                logger.info(f"Testing {config_name}...")
                
                try:
                    # Create model manager
                    manager = create_transformer_manager(config)
                    
                    # Generate sample input
                    sample_input = torch.randint(0, 50257, (1, 128), dtype=torch.long)
                    
                    # Benchmark original model
                    original_benchmark = self.optimizers["maximum"].performance_profiler.benchmark_model(
                        manager.model, sample_input, num_runs=20, warmup_runs=5
                    )
                    
                    # Optimize model
                    optimized_model = self.optimizers["maximum"].optimize_model(
                        manager.model, f"{config_name}_optimized"
                    )
                    
                    # Benchmark optimized model
                    optimized_benchmark = self.optimizers["maximum"].performance_profiler.benchmark_model(
                        optimized_model, sample_input, num_runs=20, warmup_runs=5
                    )
                    
                    # Calculate improvements
                    if (original_benchmark["status"] == "success" and 
                        optimized_benchmark["status"] == "success"):
                        
                        original_result = original_benchmark["benchmark_result"]
                        optimized_result = optimized_benchmark["benchmark_result"]
                        
                        speedup = original_result["avg_inference_time_ms"] / optimized_result["avg_inference_time_ms"]
                        throughput_improvement = optimized_result["throughput_samples_per_sec"] / original_result["throughput_samples_per_sec"]
                        
                        results[config_name] = {
                            "original_performance": original_result,
                            "optimized_performance": optimized_result,
                            "speedup": speedup,
                            "throughput_improvement": throughput_improvement,
                            "memory_improvement": optimized_result["avg_memory_delta_mb"] - original_result["avg_memory_delta_mb"]
                        }
                        
                        logger.info(f"  {config_name}: {speedup:.2f}x speedup, {throughput_improvement:.2f}x throughput")
                    
                except Exception as e:
                    logger.warning(f"Failed to benchmark {config_name}: {e}")
                    results[config_name] = {"status": "error", "message": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Transformer benchmarking failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _benchmark_diffusion_models(self) -> Dict[str, Any]:
        """Benchmark diffusion model performance."""
        logger.info("🎨 Benchmarking diffusion models...")
        
        try:
            results = {}
            
            # Test different diffusion configurations
            configs = [
                ("Stable Diffusion v1.5", create_stable_diffusion_config()),
                ("Stable Diffusion XL", create_stable_diffusion_xl_config())
            ]
            
            for config_name, config in configs:
                logger.info(f"Testing {config_name}...")
                
                try:
                    # Create diffusion manager
                    manager = create_diffusion_manager(config)
                    
                    # Generate sample prompt
                    sample_prompt = "A beautiful landscape with mountains and lakes, high quality, detailed"
                    
                    # Benchmark generation time
                    start_time = time.time()
                    start_memory = self._get_memory_usage()
                    
                    # Generate image
                    image = manager.generate_image(sample_prompt)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    generation_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    
                    results[config_name] = {
                        "generation_time_seconds": generation_time,
                        "memory_usage_mb": memory_usage,
                        "prompt": sample_prompt,
                        "image_generated": image is not None
                    }
                    
                    logger.info(f"  {config_name}: {generation_time:.2f}s, {memory_usage:+.2f}MB")
                    
                except Exception as e:
                    logger.warning(f"Failed to benchmark {config_name}: {e}")
                    results[config_name] = {"status": "error", "message": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Diffusion benchmarking failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _benchmark_optimization_impact(self) -> Dict[str, Any]:
        """Benchmark the impact of different optimization configurations."""
        logger.info("⚡ Benchmarking optimization configurations...")
        
        try:
            results = {}
            
            # Create a simple test model
            test_model = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            
            # Generate sample input
            sample_input = torch.randn(32, 512)
            
            # Test each optimization configuration
            for config_name, optimizer in self.optimizers.items():
                logger.info(f"Testing {config_name} optimization...")
                
                try:
                    # Benchmark original model
                    original_benchmark = optimizer.performance_profiler.benchmark_model(
                        test_model, sample_input, num_runs=30, warmup_runs=5
                    )
                    
                    # Optimize model
                    optimized_model = optimizer.optimize_model(test_model, f"test_model_{config_name}")
                    
                    # Benchmark optimized model
                    optimized_benchmark = optimizer.performance_profiler.benchmark_model(
                        optimized_model, sample_input, num_runs=30, warmup_runs=5
                    )
                    
                    # Calculate improvements
                    if (original_benchmark["status"] == "success" and 
                        optimized_benchmark["status"] == "success"):
                        
                        original_result = original_benchmark["benchmark_result"]
                        optimized_result = optimized_benchmark["benchmark_result"]
                        
                        speedup = original_result["avg_inference_time_ms"] / optimized_result["avg_inference_time_ms"]
                        throughput_improvement = optimized_result["throughput_samples_per_sec"] / original_result["throughput_samples_per_sec"]
                        
                        results[config_name] = {
                            "original_performance": original_result,
                            "optimized_performance": optimized_result,
                            "speedup": speedup,
                            "throughput_improvement": throughput_improvement,
                            "memory_improvement": optimized_result["avg_memory_delta_mb"] - original_result["avg_memory_delta_mb"],
                            "optimization_config": optimizer.config.__dict__
                        }
                        
                        logger.info(f"  {config_name}: {speedup:.2f}x speedup, {throughput_improvement:.2f}x throughput")
                    
                except Exception as e:
                    logger.warning(f"Failed to benchmark {config_name} optimization: {e}")
                    results[config_name] = {"status": "error", "message": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization benchmarking failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_performance_report(self, transformer_results: Dict, 
                                   diffusion_results: Dict, 
                                   optimization_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Calculate overall statistics
            total_speedup = 0
            total_throughput_improvement = 0
            successful_benchmarks = 0
            
            # Process transformer results
            for result in transformer_results.values():
                if isinstance(result, dict) and "speedup" in result:
                    total_speedup += result["speedup"]
                    total_throughput_improvement += result["throughput_improvement"]
                    successful_benchmarks += 1
            
            # Process optimization results
            for result in optimization_results.values():
                if isinstance(result, dict) and "speedup" in result:
                    total_speedup += result["speedup"]
                    total_throughput_improvement += result["throughput_improvement"]
                    successful_benchmarks += 1
            
            # Calculate averages
            avg_speedup = total_speedup / successful_benchmarks if successful_benchmarks > 0 else 0
            avg_throughput_improvement = total_throughput_improvement / successful_benchmarks if successful_benchmarks > 0 else 0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                avg_speedup, avg_throughput_improvement, transformer_results, optimization_results
            )
            
            report = {
                "summary": {
                    "total_benchmarks": successful_benchmarks,
                    "average_speedup": avg_speedup,
                    "average_throughput_improvement": avg_throughput_improvement,
                    "overall_performance_rating": self._calculate_performance_rating(avg_speedup)
                },
                "recommendations": recommendations,
                "detailed_results": {
                    "transformer_models": transformer_results,
                    "diffusion_models": diffusion_results,
                    "optimization_configs": optimization_results
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_recommendations(self, avg_speedup: float, avg_throughput: float,
                                 transformer_results: Dict, optimization_results: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if avg_speedup < 1.5:
            recommendations.append("Consider enabling torch.compile with max-autotune mode for better performance")
            recommendations.append("Enable Flash Attention 2.0 if available for your hardware")
            recommendations.append("Use maximum performance configuration for inference workloads")
        
        if avg_speedup < 2.0:
            recommendations.append("Enable Triton kernels for custom CUDA operations")
            recommendations.append("Consider model quantization (INT8/FP16) for faster inference")
            recommendations.append("Optimize batch sizes dynamically for maximum throughput")
        
        if avg_speedup >= 2.0:
            recommendations.append("Excellent performance achieved! Consider enabling cudagraphs for even better performance")
            recommendations.append("Monitor memory usage and consider gradient checkpointing for training")
        
        # Model-specific recommendations
        for model_name, result in transformer_results.items():
            if isinstance(result, dict) and "speedup" in result:
                if result["speedup"] < 1.3:
                    recommendations.append(f"Consider optimizing {model_name} with LoRA fine-tuning")
        
        return recommendations
    
    def _calculate_performance_rating(self, avg_speedup: float) -> str:
        """Calculate overall performance rating."""
        if avg_speedup >= 3.0:
            return "🚀 EXCEPTIONAL"
        elif avg_speedup >= 2.0:
            return "⚡ EXCELLENT"
        elif avg_speedup >= 1.5:
            return "🔥 GOOD"
        elif avg_speedup >= 1.2:
            return "✅ SATISFACTORY"
        else:
            return "⚠️ NEEDS IMPROVEMENT"
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    
    def save_results(self, filename: str = "ultra_performance_benchmark_results.json"):
        """Save benchmark results to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2, default=str)
            logger.info(f"✅ Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def print_summary(self):
        """Print benchmark summary to console."""
        if not self.benchmark_results:
            logger.warning("No benchmark results available")
            return
        
        try:
            report = self.benchmark_results.get("performance_report", {})
            summary = report.get("summary", {})
            
            print("\n" + "="*80)
            print("🚀 ULTRA PERFORMANCE BENCHMARK RESULTS")
            print("="*80)
            
            print(f"📊 Overall Performance Rating: {summary.get('overall_performance_rating', 'N/A')}")
            print(f"⚡ Average Speedup: {summary.get('average_speedup', 0):.2f}x")
            print(f"🚀 Average Throughput Improvement: {summary.get('average_throughput_improvement', 0):.2f}x")
            print(f"🔢 Total Benchmarks: {summary.get('total_benchmarks', 0)}")
            
            print("\n📋 Key Recommendations:")
            recommendations = report.get("recommendations", [])
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Failed to print summary: {e}")


async def main():
    """Main benchmark execution."""
    if not MODULES_AVAILABLE:
        logger.error("❌ Required modules not available. Please install dependencies.")
        return
    
    try:
        logger.info("🚀 Starting Ultra Performance Benchmark Suite...")
        
        # Initialize benchmark suite
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Run comprehensive benchmark
        results = await benchmark_suite.run_comprehensive_benchmark()
        
        if results.get("status") == "error":
            logger.error(f"Benchmark failed: {results['message']}")
            return
        
        # Print summary
        benchmark_suite.print_summary()
        
        # Save results
        benchmark_suite.save_results()
        
        logger.info("✅ Ultra Performance Benchmark completed successfully!")
        
        # Cleanup
        for optimizer in benchmark_suite.optimizers.values():
            optimizer.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Benchmark execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the benchmark
    asyncio.run(main())

