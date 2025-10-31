"""
Performance Benchmarking Suite for HeyGen AI Enterprise

This module provides comprehensive performance benchmarking capabilities:
- Multi-dimensional performance metrics
- Automated benchmarking workflows
- Performance comparison and analysis
- Benchmark result visualization
- Performance regression detection
- Cross-platform benchmarking
- Real-time performance tracking
"""

import logging
import os
import time
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from collections import defaultdict, deque
import statistics

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""
    
    # Benchmark settings
    num_runs: int = 100
    warmup_runs: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    input_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    
    # Performance metrics
    enable_timing: bool = True
    enable_memory_profiling: bool = True
    enable_gpu_profiling: bool = True
    enable_throughput_measurement: bool = True
    
    # Analysis settings
    enable_statistical_analysis: bool = True
    enable_performance_regression: bool = True
    enable_comparison_analysis: bool = True
    
    # Output settings
    save_results: bool = True
    results_directory: str = "benchmark_results"
    enable_visualization: bool = True


class PerformanceMetrics:
    """Comprehensive performance metrics collection."""
    
    def __init__(self):
        self.timestamp = time.time()
        self.timing_metrics = {}
        self.memory_metrics = {}
        self.gpu_metrics = {}
        self.throughput_metrics = {}
        self.custom_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "timing_metrics": self.timing_metrics,
            "memory_metrics": self.memory_metrics,
            "gpu_metrics": self.gpu_metrics,
            "throughput_metrics": self.throughput_metrics,
            "custom_metrics": self.custom_metrics
        }


class ModelBenchmarker:
    """Benchmarks individual models with comprehensive metrics."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmark_history = []
        
    def benchmark_model(self, model: nn.Module, input_tensor: torch.Tensor,
                       device: torch.device) -> Dict[str, Any]:
        """Comprehensive model benchmarking."""
        try:
            model.eval()
            model = model.to(device)
            
            # Warm up
            self._warmup_model(model, input_tensor, device)
            
            # Benchmark timing
            timing_metrics = self._benchmark_timing(model, input_tensor, device)
            
            # Benchmark memory
            memory_metrics = self._benchmark_memory(model, input_tensor, device)
            
            # Benchmark GPU (if available)
            gpu_metrics = self._benchmark_gpu(model, input_tensor, device)
            
            # Benchmark throughput
            throughput_metrics = self._benchmark_throughput(model, input_tensor, device)
            
            # Compile results
            results = {
                "timestamp": time.time(),
                "model_info": self._extract_model_info(model),
                "input_info": self._extract_input_info(input_tensor),
                "device_info": str(device),
                "timing_metrics": timing_metrics,
                "memory_metrics": memory_metrics,
                "gpu_metrics": gpu_metrics,
                "throughput_metrics": throughput_metrics
            }
            
            # Store in history
            self.benchmark_history.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Model benchmarking failed: {e}")
            return {}
    
    def _warmup_model(self, model: nn.Module, input_tensor: torch.Tensor, device: torch.device):
        """Warm up the model for accurate benchmarking."""
        try:
            model.eval()
            with torch.no_grad():
                for _ in range(self.config.warmup_runs):
                    _ = model(input_tensor)
                    
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _benchmark_timing(self, model: nn.Module, input_tensor: torch.Tensor,
                         device: torch.device) -> Dict[str, Any]:
        """Benchmark model timing performance."""
        try:
            if not self.config.enable_timing:
                return {}
            
            model.eval()
            times = []
            
            with torch.no_grad():
                for _ in range(self.config.num_runs):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    _ = model(input_tensor)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            p95_time = np.percentile(times, 95)
            p99_time = np.percentile(times, 99)
            
            return {
                "average_time_ms": avg_time,
                "std_time_ms": std_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "p95_time_ms": p95_time,
                "p99_time_ms": p99_time,
                "total_runs": len(times),
                "raw_times": times
            }
            
        except Exception as e:
            logger.warning(f"Timing benchmark failed: {e}")
            return {}
    
    def _benchmark_memory(self, model: nn.Module, input_tensor: torch.Tensor,
                          device: torch.device) -> Dict[str, Any]:
        """Benchmark model memory usage."""
        try:
            if not self.config.enable_memory_profiling:
                return {}
            
            model.eval()
            memory_usage = []
            
            with torch.no_grad():
                for _ in range(self.config.num_runs):
                    # Clear cache before measurement
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Measure memory before
                    if device.type == 'cuda':
                        memory_before = torch.cuda.memory_allocated(device)
                    else:
                        memory_before = self._get_cpu_memory_usage()
                    
                    # Run inference
                    _ = model(input_tensor)
                    
                    # Measure memory after
                    if device.type == 'cuda':
                        memory_after = torch.cuda.memory_allocated(device)
                        memory_used = (memory_after - memory_before) / (1024**2)  # MB
                    else:
                        memory_after = self._get_cpu_memory_usage()
                        memory_used = (memory_after - memory_before) / (1024**2)  # MB
                    
                    memory_usage.append(memory_used)
            
            # Calculate statistics
            avg_memory = statistics.mean(memory_usage)
            std_memory = statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
            max_memory = max(memory_usage)
            
            return {
                "average_memory_mb": avg_memory,
                "std_memory_mb": std_memory,
                "max_memory_mb": max_memory,
                "memory_usage_samples": memory_usage
            }
            
        except Exception as e:
            logger.warning(f"Memory benchmark failed: {e}")
            return {}
    
    def _benchmark_gpu(self, model: nn.Module, input_tensor: torch.Tensor,
                       device: torch.device) -> Dict[str, Any]:
        """Benchmark GPU-specific metrics."""
        try:
            if not self.config.enable_gpu_profiling or device.type != 'cuda':
                return {}
            
            model.eval()
            gpu_metrics = []
            
            with torch.no_grad():
                for _ in range(self.config.num_runs):
                    # Get GPU metrics before
                    gpu_before = self._get_gpu_metrics(device)
                    
                    # Run inference
                    _ = model(input_tensor)
                    
                    # Get GPU metrics after
                    gpu_after = self._get_gpu_metrics(device)
                    
                    # Calculate differences
                    gpu_diff = {
                        "utilization_change": gpu_after.get("utilization", 0) - gpu_before.get("utilization", 0),
                        "memory_change_mb": gpu_after.get("memory_used", 0) - gpu_before.get("memory_used", 0),
                        "temperature_change": gpu_after.get("temperature", 0) - gpu_before.get("temperature", 0)
                    }
                    
                    gpu_metrics.append(gpu_diff)
            
            # Calculate averages
            avg_utilization_change = statistics.mean([m["utilization_change"] for m in gpu_metrics])
            avg_memory_change = statistics.mean([m["memory_change_mb"] for m in gpu_metrics])
            avg_temperature_change = statistics.mean([m["temperature_change"] for m in gpu_metrics])
            
            return {
                "average_utilization_change": avg_utilization_change,
                "average_memory_change_mb": avg_memory_change,
                "average_temperature_change": avg_temperature_change,
                "gpu_metrics_samples": gpu_metrics
            }
            
        except Exception as e:
            logger.warning(f"GPU benchmark failed: {e}")
            return {}
    
    def _benchmark_throughput(self, model: nn.Module, input_tensor: torch.Tensor,
                             device: torch.device) -> Dict[str, Any]:
        """Benchmark model throughput."""
        try:
            if not self.config.enable_throughput_measurement:
                return {}
            
            model.eval()
            batch_size = input_tensor.size(0)
            
            # Measure time for multiple batches
            total_time = 0
            total_samples = 0
            
            with torch.no_grad():
                for _ in range(self.config.num_runs):
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    _ = model(input_tensor)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    total_time += (end_time - start_time)
                    total_samples += batch_size
            
            # Calculate throughput
            throughput_samples_per_second = total_samples / total_time
            throughput_batches_per_second = self.config.num_runs / total_time
            
            return {
                "throughput_samples_per_second": throughput_samples_per_second,
                "throughput_batches_per_second": throughput_batches_per_second,
                "total_samples_processed": total_samples,
                "total_time_seconds": total_time,
                "batch_size": batch_size
            }
            
        except Exception as e:
            logger.warning(f"Throughput benchmark failed: {e}")
            return {}
    
    def _extract_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model information for benchmarking."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Count different layer types
            layer_counts = defaultdict(int)
            for module in model.modules():
                layer_counts[type(module).__name__] += 1
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": (total_params * 4) / (1024**2),  # Assuming float32
                "layer_counts": dict(layer_counts),
                "model_class": type(model).__name__
            }
            
        except Exception as e:
            logger.warning(f"Model info extraction failed: {e}")
            return {}
    
    def _extract_input_info(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Extract input tensor information."""
        try:
            return {
                "shape": list(input_tensor.shape),
                "dtype": str(input_tensor.dtype),
                "device": str(input_tensor.device),
                "requires_grad": input_tensor.requires_grad,
                "memory_size_mb": input_tensor.numel() * input_tensor.element_size() / (1024**2)
            }
            
        except Exception as e:
            logger.warning(f"Input info extraction failed: {e}")
            return {}
    
    def _get_cpu_memory_usage(self) -> int:
        """Get current CPU memory usage."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0
    
    def _get_gpu_metrics(self, device: torch.device) -> Dict[str, Any]:
        """Get GPU metrics."""
        try:
            if device.type != 'cuda':
                return {}
            
            # Basic GPU metrics
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            
            return {
                "memory_allocated": memory_allocated,
                "memory_reserved": memory_reserved,
                "memory_used": memory_allocated,
                "utilization": 0,  # Would need pynvml for actual utilization
                "temperature": 0    # Would need pynvml for actual temperature
            }
            
        except Exception as e:
            logger.warning(f"GPU metrics collection failed: {e}")
            return {}


class PerformanceAnalyzer:
    """Analyzes benchmark results and provides insights."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.analysis_history = []
        
    def analyze_benchmark_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results for insights."""
        try:
            if not results:
                return {"error": "No results to analyze"}
            
            analysis = {
                "summary": self._generate_summary(results),
                "performance_trends": self._analyze_performance_trends(results),
                "statistical_analysis": self._perform_statistical_analysis(results),
                "recommendations": self._generate_recommendations(results)
            }
            
            # Store analysis
            self.analysis_history.append({
                "timestamp": time.time(),
                "analysis": analysis,
                "results_count": len(results)
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Benchmark analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        try:
            if not results:
                return {}
            
            # Extract key metrics
            timing_metrics = [r.get("timing_metrics", {}) for r in results]
            memory_metrics = [r.get("memory_metrics", {}) for r in results]
            throughput_metrics = [r.get("throughput_metrics", {}) for r in results]
            
            # Calculate averages
            avg_inference_time = statistics.mean([
                m.get("average_time_ms", 0) for m in timing_metrics if m
            ])
            
            avg_memory_usage = statistics.mean([
                m.get("average_memory_mb", 0) for m in memory_metrics if m
            ])
            
            avg_throughput = statistics.mean([
                m.get("throughput_samples_per_second", 0) for m in throughput_metrics if m
            ])
            
            return {
                "total_benchmarks": len(results),
                "average_inference_time_ms": avg_inference_time,
                "average_memory_usage_mb": avg_memory_usage,
                "average_throughput_samples_per_second": avg_throughput,
                "benchmark_timestamp_range": {
                    "start": min(r.get("timestamp", 0) for r in results),
                    "end": max(r.get("timestamp", 0) for r in results)
                }
            }
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return {}
    
    def _analyze_performance_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            if len(results) < 3:
                return {"error": "Insufficient data for trend analysis"}
            
            # Sort by timestamp
            sorted_results = sorted(results, key=lambda x: x.get("timestamp", 0))
            
            # Extract timing trends
            timing_data = []
            for result in sorted_results:
                timing = result.get("timing_metrics", {})
                if timing:
                    timing_data.append({
                        "timestamp": result.get("timestamp", 0),
                        "avg_time": timing.get("average_time_ms", 0)
                    })
            
            # Calculate trend
            if len(timing_data) >= 2:
                x = [d["timestamp"] for d in timing_data]
                y = [d["avg_time"] for d in timing_data]
                
                # Simple linear trend
                if len(x) >= 2:
                    slope = np.polyfit(x, y, 1)[0]
                    trend = "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
                else:
                    trend = "insufficient_data"
            else:
                trend = "insufficient_data"
            
            return {
                "performance_trend": trend,
                "data_points": len(timing_data),
                "trend_analysis_available": len(timing_data) >= 2
            }
            
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    def _perform_statistical_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on benchmark results."""
        try:
            if not self.config.enable_statistical_analysis:
                return {"enabled": False}
            
            # Extract all timing data
            all_times = []
            for result in results:
                timing = result.get("timing_metrics", {})
                if timing and "raw_times" in timing:
                    all_times.extend(timing["raw_times"])
            
            if not all_times:
                return {"error": "No timing data available"}
            
            # Statistical measures
            mean_time = statistics.mean(all_times)
            median_time = statistics.median(all_times)
            std_time = statistics.stdev(all_times) if len(all_times) > 1 else 0
            
            # Percentiles
            p95 = np.percentile(all_times, 95)
            p99 = np.percentile(all_times, 99)
            
            # Outlier detection (simple IQR method)
            q1 = np.percentile(all_times, 25)
            q3 = np.percentile(all_times, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [t for t in all_times if t < lower_bound or t > upper_bound]
            
            return {
                "enabled": True,
                "total_samples": len(all_times),
                "mean_time_ms": mean_time,
                "median_time_ms": median_time,
                "std_time_ms": std_time,
                "p95_time_ms": p95,
                "p99_time_ms": p99,
                "outlier_count": len(outliers),
                "outlier_percentage": (len(outliers) / len(all_times)) * 100
            }
            
        except Exception as e:
            logger.warning(f"Statistical analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations."""
        try:
            recommendations = []
            
            if not results:
                return ["No benchmark results available for analysis"]
            
            # Analyze timing performance
            timing_metrics = [r.get("timing_metrics", {}) for r in results if r.get("timing_metrics")]
            if timing_metrics:
                avg_time = statistics.mean([m.get("average_time_ms", 0) for m in timing_metrics])
                
                if avg_time > 100:  # More than 100ms
                    recommendations.append("High inference time detected. Consider model optimization or quantization.")
                elif avg_time > 50:  # More than 50ms
                    recommendations.append("Moderate inference time. Consider batch processing or model compression.")
            
            # Analyze memory usage
            memory_metrics = [r.get("memory_metrics", {}) for r in results if r.get("memory_metrics")]
            if memory_metrics:
                avg_memory = statistics.mean([m.get("average_memory_mb", 0) for m in memory_metrics])
                
                if avg_memory > 1000:  # More than 1GB
                    recommendations.append("High memory usage detected. Consider model pruning or memory optimization.")
                elif avg_memory > 500:  # More than 500MB
                    recommendations.append("Moderate memory usage. Consider batch size reduction or gradient checkpointing.")
            
            # Analyze throughput
            throughput_metrics = [r.get("throughput_metrics", {}) for r in results if r.get("throughput_metrics")]
            if throughput_metrics:
                avg_throughput = statistics.mean([m.get("throughput_samples_per_second", 0) for m in throughput_metrics])
                
                if avg_throughput < 10:  # Less than 10 samples/second
                    recommendations.append("Low throughput detected. Consider model optimization or hardware upgrade.")
                elif avg_throughput < 50:  # Less than 50 samples/second
                    recommendations.append("Moderate throughput. Consider batch processing or parallel inference.")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Performance is within acceptable ranges. Consider monitoring for regressions.")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to error"]


class PerformanceBenchmarkingSuite:
    """Main benchmarking suite orchestrating all components."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.benchmarking_suite")
        
        # Initialize components
        self.benchmarker = ModelBenchmarker(config)
        self.analyzer = PerformanceAnalyzer(config)
        
        # Benchmark state
        self.benchmark_results = []
        self.comparison_results = {}
        
        # Create results directory
        if config.save_results:
            os.makedirs(config.results_directory, exist_ok=True)
    
    def run_comprehensive_benchmark(self, model: nn.Module, device: torch.device,
                                  input_sizes: Optional[List[int]] = None,
                                  batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run comprehensive benchmarking across multiple configurations."""
        try:
            self.logger.info("🚀 Starting Comprehensive Performance Benchmarking...")
            
            if input_sizes is None:
                input_sizes = self.config.input_sizes
            
            if batch_sizes is None:
                batch_sizes = self.config.batch_sizes
            
            all_results = []
            
            # Benchmark across different configurations
            for input_size in input_sizes:
                for batch_size in batch_sizes:
                    self.logger.info(f"🔍 Benchmarking: input_size={input_size}, batch_size={batch_size}")
                    
                    # Create input tensor
                    input_tensor = torch.randn(batch_size, input_size, device=device)
                    
                    # Run benchmark
                    result = self.benchmarker.benchmark_model(model, input_tensor, device)
                    
                    if result:
                        result["configuration"] = {
                            "input_size": input_size,
                            "batch_size": batch_size
                        }
                        all_results.append(result)
                        
                        # Save individual result
                        if self.config.save_results:
                            self._save_benchmark_result(result, input_size, batch_size)
            
            # Store results
            self.benchmark_results.extend(all_results)
            
            # Analyze results
            analysis = self.analyzer.analyze_benchmark_results(all_results)
            
            # Save comprehensive results
            if self.config.save_results:
                self._save_comprehensive_results(all_results, analysis)
            
            self.logger.info(f"✅ Comprehensive benchmarking completed. {len(all_results)} benchmarks run.")
            
            return {
                "benchmark_results": all_results,
                "analysis": analysis,
                "total_benchmarks": len(all_results)
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive benchmarking failed: {e}")
            return {"error": str(e)}
    
    def compare_models(self, models: Dict[str, nn.Module], device: torch.device,
                      input_size: int = 512, batch_size: int = 8) -> Dict[str, Any]:
        """Compare performance across multiple models."""
        try:
            self.logger.info(f"🔍 Comparing {len(models)} models...")
            
            comparison_results = {}
            
            for model_name, model in models.items():
                self.logger.info(f"Benchmarking model: {model_name}")
                
                # Create input tensor
                input_tensor = torch.randn(batch_size, input_size, device=device)
                
                # Run benchmark
                result = self.benchmarker.benchmark_model(model, input_tensor, device)
                
                if result:
                    comparison_results[model_name] = result
            
            # Store comparison results
            self.comparison_results = comparison_results
            
            # Generate comparison analysis
            comparison_analysis = self._analyze_model_comparison(comparison_results)
            
            # Save comparison results
            if self.config.save_results:
                self._save_comparison_results(comparison_results, comparison_analysis)
            
            self.logger.info("✅ Model comparison completed successfully.")
            
            return {
                "comparison_results": comparison_results,
                "comparison_analysis": comparison_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            return {"error": str(e)}
    
    def _analyze_model_comparison(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model comparison results."""
        try:
            if not comparison_results:
                return {"error": "No comparison results available"}
            
            analysis = {
                "model_count": len(comparison_results),
                "performance_ranking": [],
                "best_performing_model": None,
                "performance_differences": {}
            }
            
            # Extract key metrics for comparison
            model_metrics = {}
            for model_name, result in comparison_results.items():
                timing = result.get("timing_metrics", {})
                memory = result.get("memory_metrics", {})
                throughput = result.get("throughput_metrics", {})
                
                model_metrics[model_name] = {
                    "inference_time_ms": timing.get("average_time_ms", float('inf')),
                    "memory_usage_mb": memory.get("average_memory_mb", float('inf')),
                    "throughput_samples_per_second": throughput.get("throughput_samples_per_second", 0)
                }
            
            # Rank models by inference time (lower is better)
            sorted_models = sorted(
                model_metrics.items(),
                key=lambda x: x[1]["inference_time_ms"]
            )
            
            analysis["performance_ranking"] = [model_name for model_name, _ in sorted_models]
            analysis["best_performing_model"] = sorted_models[0][0] if sorted_models else None
            
            # Calculate performance differences
            if len(sorted_models) >= 2:
                best_time = sorted_models[0][1]["inference_time_ms"]
                for model_name, metrics in sorted_models[1:]:
                    time_diff = ((metrics["inference_time_ms"] - best_time) / best_time) * 100
                    analysis["performance_differences"][model_name] = {
                        "time_difference_percent": time_diff,
                        "slower_than_best": time_diff > 0
                    }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Model comparison analysis failed: {e}")
            return {"error": str(e)}
    
    def _save_benchmark_result(self, result: Dict[str, Any], input_size: int, batch_size: int):
        """Save individual benchmark result."""
        try:
            filename = f"benchmark_{input_size}_{batch_size}_{int(time.time())}.json"
            filepath = os.path.join(self.config.results_directory, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save benchmark result: {e}")
    
    def _save_comprehensive_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save comprehensive benchmark results."""
        try:
            filename = f"comprehensive_benchmark_{int(time.time())}.json"
            filepath = os.path.join(self.config.results_directory, filename)
            
            data = {
                "timestamp": time.time(),
                "benchmark_results": results,
                "analysis": analysis,
                "configuration": {
                    "num_runs": self.config.num_runs,
                    "warmup_runs": self.config.warmup_runs,
                    "batch_sizes": self.config.batch_sizes,
                    "input_sizes": self.config.input_sizes
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save comprehensive results: {e}")
    
    def _save_comparison_results(self, comparison_results: Dict[str, Any], analysis: Dict[str, Any]):
        """Save model comparison results."""
        try:
            filename = f"model_comparison_{int(time.time())}.json"
            filepath = os.path.join(self.config.results_directory, filename)
            
            data = {
                "timestamp": time.time(),
                "comparison_results": comparison_results,
                "comparison_analysis": analysis
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save comparison results: {e}")
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get comprehensive benchmark summary."""
        return {
            "total_benchmarks": len(self.benchmark_results),
            "total_comparisons": len(self.comparison_results),
            "results_directory": self.config.results_directory if self.config.save_results else None,
            "configuration": {
                "num_runs": self.config.num_runs,
                "warmup_runs": self.config.warmup_runs,
                "batch_sizes": self.config.batch_sizes,
                "input_sizes": self.config.input_sizes
            }
        }


# Factory functions
def create_benchmarking_suite(config: Optional[BenchmarkConfig] = None) -> PerformanceBenchmarkingSuite:
    """Create a performance benchmarking suite."""
    if config is None:
        config = BenchmarkConfig()
    
    return PerformanceBenchmarkingSuite(config)


def create_comprehensive_benchmark_config() -> BenchmarkConfig:
    """Create comprehensive benchmarking configuration."""
    return BenchmarkConfig(
        num_runs=100,
        warmup_runs=10,
        batch_sizes=[1, 4, 8, 16, 32],
        input_sizes=[64, 128, 256, 512, 1024],
        enable_timing=True,
        enable_memory_profiling=True,
        enable_gpu_profiling=True,
        enable_throughput_measurement=True,
        save_results=True,
        enable_visualization=True
    )


def create_quick_benchmark_config() -> BenchmarkConfig:
    """Create quick benchmarking configuration."""
    return BenchmarkConfig(
        num_runs=20,
        warmup_runs=5,
        batch_sizes=[1, 8],
        input_sizes=[256, 512],
        enable_timing=True,
        enable_memory_profiling=True,
        enable_gpu_profiling=False,
        enable_throughput_measurement=True,
        save_results=False,
        enable_visualization=False
    )


if __name__ == "__main__":
    # Test the benchmarking suite
    config = create_comprehensive_benchmark_config()
    suite = create_benchmarking_suite(config)
    
    print(f"Performance Benchmarking Suite created with config: {config}")
    print(f"Suite ready for comprehensive benchmarking!")
