"""
Benchmarking System
Comprehensive benchmarking and performance evaluation for TruthGPT
"""

import torch
import torch.nn as nn
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import psutil
import gc
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    # Benchmark settings
    num_runs: int = 5
    warmup_runs: int = 2
    timeout_seconds: int = 300
    
    # Performance metrics
    measure_memory: bool = True
    measure_cpu: bool = True
    measure_gpu: bool = True
    measure_latency: bool = True
    measure_throughput: bool = True
    
    # Test data
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    
    # Output settings
    save_results: bool = True
    output_dir: str = "benchmark_results"
    verbose: bool = True

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    name: str
    metric: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    start_time: float
    end_time: float
    total_duration: float
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

class PerformanceProfiler:
    """Advanced performance profiler for neural networks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.monitoring = False
        self.monitor_thread = None
        
    def start_profiling(self):
        """Start performance profiling"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_profiling(self):
        """Stop performance profiling"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                if self.config.measure_cpu:
                    cpu_percent = psutil.cpu_percent()
                    self.metrics['cpu_percent'].append(cpu_percent)
                
                if self.config.measure_memory:
                    memory = psutil.virtual_memory()
                    self.metrics['memory_percent'].append(memory.percent)
                    self.metrics['memory_used_mb'].append(memory.used / 1024 / 1024)
                
                if self.config.measure_gpu and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    self.metrics['gpu_memory_mb'].append(gpu_memory)
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                logger.error(f"Profiling error: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        return summary

class ModelBenchmarker:
    """Benchmarker for neural network models"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.profiler = PerformanceProfiler(config)
        
    def benchmark_model(self, 
                       model: nn.Module, 
                       model_name: str,
                       test_data: Dict[str, torch.Tensor]) -> BenchmarkSuite:
        """Benchmark a neural network model"""
        logger.info(f"🧪 Starting benchmark for {model_name}")
        
        suite = BenchmarkSuite(
            suite_name=f"{model_name}_benchmark",
            start_time=time.time()
        )
        
        try:
            # Warmup runs
            if self.config.warmup_runs > 0:
                logger.info(f"🔥 Running {self.config.warmup_runs} warmup runs...")
                self._run_warmup(model, test_data)
            
            # Main benchmark runs
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    if self._should_skip_test(batch_size, seq_len):
                        continue
                    
                    logger.info(f"📊 Benchmarking batch_size={batch_size}, seq_len={seq_len}")
                    
                    # Prepare test data
                    test_input = self._prepare_test_data(test_data, batch_size, seq_len)
                    
                    # Run benchmark
                    results = self._run_benchmark(model, test_input, batch_size, seq_len)
                    
                    # Add results to suite
                    for result in results:
                        suite.results.append(result)
            
            # Generate summary
            suite.end_time = time.time()
            suite.total_duration = suite.end_time - suite.start_time
            suite.summary = self._generate_summary(suite.results)
            
            # Save results if requested
            if self.config.save_results:
                self._save_results(suite)
            
            logger.info(f"✅ Benchmark completed for {model_name}")
            return suite
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            suite.end_time = time.time()
            suite.total_duration = suite.end_time - suite.start_time
            return suite
    
    def _run_warmup(self, model: nn.Module, test_data: Dict[str, torch.Tensor]):
        """Run warmup iterations"""
        model.eval()
        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                # Use smallest batch size and sequence length for warmup
                test_input = self._prepare_test_data(test_data, 1, 64)
                _ = model(test_input)
    
    def _run_benchmark(self, 
                      model: nn.Module, 
                      test_input: torch.Tensor,
                      batch_size: int, 
                      seq_len: int) -> List[BenchmarkResult]:
        """Run benchmark for specific configuration"""
        results = []
        
        # Start profiling
        self.profiler.start_profiling()
        
        # Run multiple iterations
        latencies = []
        for run in range(self.config.num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                output = model(test_input)
            
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
        
        # Stop profiling
        self.profiler.stop_profiling()
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Add latency results
        results.append(BenchmarkResult(
            name="latency",
            metric="average_latency",
            value=avg_latency,
            unit="seconds",
            timestamp=time.time(),
            metadata={
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "std_latency": std_latency,
                "min_latency": min_latency,
                "max_latency": max_latency
            }
        ))
        
        # Add throughput results
        throughput = batch_size / avg_latency
        results.append(BenchmarkResult(
            name="throughput",
            metric="samples_per_second",
            value=throughput,
            unit="samples/sec",
            timestamp=time.time(),
            metadata={
                "batch_size": batch_size,
                "sequence_length": seq_len
            }
        ))
        
        # Add memory results
        profiler_summary = self.profiler.get_metrics_summary()
        for metric_name, stats in profiler_summary.items():
            results.append(BenchmarkResult(
                name="memory",
                metric=metric_name,
                value=stats['mean'],
                unit="percent" if "percent" in metric_name else "MB",
                timestamp=time.time(),
                metadata={
                    "batch_size": batch_size,
                    "sequence_length": seq_len,
                    "max": stats['max'],
                    "min": stats['min'],
                    "std": stats['std']
                }
            ))
        
        return results
    
    def _prepare_test_data(self, 
                          test_data: Dict[str, torch.Tensor], 
                          batch_size: int, 
                          seq_len: int) -> torch.Tensor:
        """Prepare test data for benchmarking"""
        # Get a sample from test data
        sample = next(iter(test_data.values()))
        
        # Reshape to desired batch size and sequence length
        if len(sample.shape) == 2:  # (seq_len, features)
            test_input = sample[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        elif len(sample.shape) == 1:  # (features,)
            test_input = sample.unsqueeze(0).repeat(batch_size, 1)
        else:
            # For other shapes, just repeat the sample
            test_input = sample.unsqueeze(0).repeat(batch_size, *([1] * (len(sample.shape))))
        
        return test_input
    
    def _should_skip_test(self, batch_size: int, seq_len: int) -> bool:
        """Determine if a test should be skipped"""
        # Skip very large configurations that might cause OOM
        if batch_size * seq_len > 1000000:  # 1M tokens
            return True
        return False
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results"""
        summary = {
            "total_tests": len(results),
            "metrics": {},
            "performance_ranking": {}
        }
        
        # Group results by metric type
        metric_groups = defaultdict(list)
        for result in results:
            metric_groups[result.metric].append(result.value)
        
        # Calculate statistics for each metric
        for metric, values in metric_groups.items():
            summary["metrics"][metric] = {
                "mean": np.mean(values),
                "max": np.max(values),
                "min": np.min(values),
                "std": np.std(values),
                "count": len(values)
            }
        
        return summary
    
    def _save_results(self, suite: BenchmarkSuite):
        """Save benchmark results to file"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"{suite.suite_name}_{int(suite.start_time)}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in suite.results:
            serializable_results.append({
                "name": result.name,
                "metric": result.metric,
                "value": result.value,
                "unit": result.unit,
                "timestamp": result.timestamp,
                "metadata": result.metadata
            })
        
        data = {
            "suite_name": suite.suite_name,
            "start_time": suite.start_time,
            "end_time": suite.end_time,
            "total_duration": suite.total_duration,
            "results": serializable_results,
            "summary": suite.summary
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Benchmark results saved to {results_file}")

class ComparativeBenchmarker:
    """Benchmarker for comparing multiple models"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmarker = ModelBenchmarker(config)
    
    def compare_models(self, 
                      models: Dict[str, nn.Module], 
                      test_data: Dict[str, torch.Tensor]) -> Dict[str, BenchmarkSuite]:
        """Compare multiple models"""
        logger.info(f"🔬 Starting comparative benchmark for {len(models)} models")
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"📊 Benchmarking {model_name}...")
            suite = self.benchmarker.benchmark_model(model, model_name, test_data)
            results[model_name] = suite
        
        # Generate comparative analysis
        self._generate_comparative_analysis(results)
        
        return results
    
    def _generate_comparative_analysis(self, results: Dict[str, BenchmarkSuite]):
        """Generate comparative analysis of benchmark results"""
        logger.info("📈 Generating comparative analysis...")
        
        # Find best performing model for each metric
        best_models = {}
        
        # Get all unique metrics
        all_metrics = set()
        for suite in results.values():
            for result in suite.results:
                all_metrics.add(result.metric)
        
        # Find best model for each metric
        for metric in all_metrics:
            best_model = None
            best_value = float('inf') if "latency" in metric else 0
            
            for model_name, suite in results.items():
                metric_values = [r.value for r in suite.results if r.metric == metric]
                if metric_values:
                    avg_value = np.mean(metric_values)
                    
                    if "latency" in metric or "memory" in metric:
                        # Lower is better
                        if avg_value < best_value:
                            best_value = avg_value
                            best_model = model_name
                    else:
                        # Higher is better
                        if avg_value > best_value:
                            best_value = avg_value
                            best_model = model_name
            
            if best_model:
                best_models[metric] = {
                    "model": best_model,
                    "value": best_value
                }
        
        # Log results
        logger.info("🏆 Best performing models:")
        for metric, info in best_models.items():
            logger.info(f"  {metric}: {info['model']} ({info['value']:.3f})")

class BenchmarkRunner:
    """Main benchmark runner"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmarker = ModelBenchmarker(config)
        self.comparative_benchmarker = ComparativeBenchmarker(config)
    
    def run_single_model_benchmark(self, 
                                 model: nn.Module, 
                                 model_name: str,
                                 test_data: Dict[str, torch.Tensor]) -> BenchmarkSuite:
        """Run benchmark for a single model"""
        return self.benchmarker.benchmark_model(model, model_name, test_data)
    
    def run_comparative_benchmark(self, 
                                 models: Dict[str, nn.Module], 
                                 test_data: Dict[str, torch.Tensor]) -> Dict[str, BenchmarkSuite]:
        """Run comparative benchmark for multiple models"""
        return self.comparative_benchmarker.compare_models(models, test_data)
    
    def run_optimization_benchmark(self, 
                                  model: nn.Module, 
                                  model_name: str,
                                  test_data: Dict[str, torch.Tensor],
                                  optimization_levels: List[str]) -> Dict[str, BenchmarkSuite]:
        """Run benchmark comparing different optimization levels"""
        from .optimization import OptimizationEngine, OptimizationConfig, OptimizationLevel
        
        results = {}
        
        for level_name in optimization_levels:
            try:
                level = OptimizationLevel(level_name)
                config = OptimizationConfig(level=level)
                optimizer = OptimizationEngine(config)
                
                # Optimize model
                optimized_model = optimizer.optimize_model(model)
                
                # Benchmark optimized model
                optimized_name = f"{model_name}_{level_name}"
                suite = self.benchmarker.benchmark_model(optimized_model, optimized_name, test_data)
                results[optimized_name] = suite
                
            except Exception as e:
                logger.error(f"Failed to benchmark {level_name}: {e}")
        
        return results

