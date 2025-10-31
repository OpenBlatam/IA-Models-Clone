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
import time
import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import json
from collections import defaultdict
import threading
    import psutil
    import memray
import numpy as np
import random
from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ultra_performance_optimizers import (
        import traceback
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
🚀 ADVANCED BENCHMARK SYSTEM - ULTRA PERFORMANCE 2024
======================================================

Sistema de benchmarking avanzado para medir rendimiento de optimizaciones:
✅ Benchmarks multi-método (Ray, Polars, GPU, Arrow)
✅ Profiling de memoria con Memray
✅ Análisis de rendimiento por método
✅ Comparativas de escalabilidad
✅ Reportes detallados con gráficos
✅ Stress testing con cargas variables
✅ Medición de latencia y throughput
"""


# Performance monitoring
try:
    MEMORY_PROFILING = True
except ImportError:
    MEMORY_PROFILING = False

# Data generation

# Plotting (opcional)
try:
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Import our ultra performance system
try:
        UltraPerformanceManager, 
        UltraPerformanceConfig,
        create_ultra_performance_manager
    )
    ULTRA_SYSTEM_AVAILABLE = True
except ImportError:
    ULTRA_SYSTEM_AVAILABLE = False
    logging.warning("Ultra performance system not available")

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    
    # Test sizes
    small_dataset: int = 100
    medium_dataset: int = 1000
    large_dataset: int = 10000
    xl_dataset: int = 50000
    
    # Test iterations
    warmup_runs: int = 3
    benchmark_runs: int = 5
    
    # Stress test
    max_concurrent: int = 20
    stress_duration: int = 60  # seconds
    
    # Memory tracking
    enable_memory_profiling: bool = MEMORY_PROFILING
    memory_sampling_interval: float = 0.1
    
    # Methods to test
    test_methods: List[str] = None
    
    def __post_init__(self) -> Any:
        if self.test_methods is None:
            self.test_methods = ["polars", "gpu", "ray", "arrow", "fallback"]

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    method: str
    dataset_size: int
    processing_time: float
    videos_per_second: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    success: bool
    error: Optional[str] = None
    
    # Additional metrics
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    
    config: BenchmarkConfig
    results: List[BenchmarkResult]
    system_info: Dict[str, Any]
    timestamp: str
    total_duration: float
    
    def get_results_by_method(self, method: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.method == method]
    
    def get_results_by_size(self, size: int) -> List[BenchmarkResult]:
        return [r for r in self.results if r.dataset_size == size]
    
    def get_best_method_by_size(self) -> Dict[int, str]:
        """Get best performing method for each dataset size."""
        best_methods = {}
        
        for size in [self.config.small_dataset, self.config.medium_dataset, 
                    self.config.large_dataset, self.config.xl_dataset]:
            
            size_results = self.get_results_by_size(size)
            if size_results:
                best_result = max(size_results, key=lambda r: r.videos_per_second if r.success else 0)
                best_methods[size] = best_result.method
        
        return best_methods

# =============================================================================
# MEMORY PROFILER
# =============================================================================

class MemoryProfiler:
    """Advanced memory profiling with real-time monitoring."""
    
    def __init__(self, sampling_interval: float = 0.1):
        
    """__init__ function."""
self.sampling_interval = sampling_interval
        self.process = psutil.Process()
        self.monitoring = False
        self.memory_samples = []
        self._monitor_thread = None
    
    def start_monitoring(self) -> Any:
        """Start memory monitoring in background thread."""
        self.monitoring = True
        self.memory_samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return memory statistics."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        if not self.memory_samples:
            return {
                'peak_memory_mb': 0.0,
                'avg_memory_mb': 0.0,
                'min_memory_mb': 0.0
            }
        
        memory_mb = [sample / 1024 / 1024 for sample in self.memory_samples]
        
        return {
            'peak_memory_mb': max(memory_mb),
            'avg_memory_mb': sum(memory_mb) / len(memory_mb),
            'min_memory_mb': min(memory_mb)
        }
    
    def _monitor_memory(self) -> Any:
        """Background memory monitoring."""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                self.memory_samples.append(memory_info.rss)
                time.sleep(self.sampling_interval)
            except:
                break

# =============================================================================
# ADVANCED BENCHMARK RUNNER
# =============================================================================

class AdvancedBenchmarkRunner:
    """Advanced benchmark runner with comprehensive testing."""
    
    def __init__(self, config: BenchmarkConfig = None):
        
    """__init__ function."""
self.config = config or BenchmarkConfig()
        self.memory_profiler = MemoryProfiler(self.config.memory_sampling_interval)
        self.results = []
        
        # System info
        self.system_info = self._collect_system_info()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmark context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_test_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate realistic test data for benchmarking."""
        test_data = []
        
        for i in range(size):
            # Realistic video data distribution
            duration = random.choices(
                [10, 15, 30, 45, 60, 90, 120],
                weights=[30, 25, 20, 10, 8, 4, 3]
            )[0] + random.uniform(-5, 5)
            
            faces_count = random.choices(
                [0, 1, 2, 3, 4, 5],
                weights=[20, 40, 20, 10, 7, 3]
            )[0]
            
            visual_quality = np.random.normal(6.5, 1.5)
            visual_quality = max(1.0, min(10.0, visual_quality))
            
            test_data.append({
                'id': f'video_{i}',
                'title': f'Test Video {i}',
                'duration': max(5, duration),
                'faces_count': faces_count,
                'visual_quality': visual_quality,
                'timestamp': datetime.now().isoformat()
            })
        
        return test_data
    
    async def run_single_benchmark(
        self, 
        method: str, 
        dataset_size: int, 
        manager: UltraPerformanceManager
    ) -> BenchmarkResult:
        """Run a single benchmark test."""
        
        self.logger.info(f"🧪 Benchmarking {method.upper()} with {dataset_size} videos")
        
        # Generate test data
        test_data = self.generate_test_data(dataset_size)
        
        # Warmup runs
        for _ in range(self.config.warmup_runs):
            try:
                await manager.process_videos_ultra_performance(test_data[:10], method=method)
                gc.collect()  # Force garbage collection
            except:
                pass
        
        # Actual benchmark
        latencies = []
        
        # Start monitoring
        start_cpu = psutil.cpu_percent(interval=None)
        self.memory_profiler.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Run benchmark iterations
            for _ in range(self.config.benchmark_runs):
                iteration_start = time.time()
                
                result = await manager.process_videos_ultra_performance(
                    test_data, method=method
                )
                
                iteration_time = time.time() - iteration_start
                latencies.append(iteration_time)
                
                if not result.get('success', False):
                    raise Exception(result.get('error', 'Unknown error'))
            
            processing_time = time.time() - start_time
            
            # Stop monitoring
            memory_stats = self.memory_profiler.stop_monitoring()
            end_cpu = psutil.cpu_percent(interval=None)
            
            # Calculate metrics
            videos_per_second = (dataset_size * self.config.benchmark_runs) / processing_time
            
            # Latency percentiles
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            
            return BenchmarkResult(
                method=method,
                dataset_size=dataset_size,
                processing_time=processing_time / self.config.benchmark_runs,
                videos_per_second=videos_per_second / self.config.benchmark_runs,
                memory_usage_mb=memory_stats['avg_memory_mb'],
                peak_memory_mb=memory_stats['peak_memory_mb'],
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                success=True,
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99
            )
            
        except Exception as e:
            self.memory_profiler.stop_monitoring()
            
            return BenchmarkResult(
                method=method,
                dataset_size=dataset_size,
                processing_time=0.0,
                videos_per_second=0.0,
                memory_usage_mb=0.0,
                peak_memory_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error=str(e)
            )
    
    async def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run comprehensive benchmark across all methods and sizes."""
        
        self.logger.info("🚀 Starting Comprehensive Benchmark Suite")
        suite_start = time.time()
        
        if not ULTRA_SYSTEM_AVAILABLE:
            raise ImportError("Ultra performance system not available")
        
        # Initialize ultra performance manager
        manager = await create_ultra_performance_manager("production")
        
        # Test sizes
        test_sizes = [
            self.config.small_dataset,
            self.config.medium_dataset,
            self.config.large_dataset,
            self.config.xl_dataset
        ]
        
        # Run benchmarks
        all_results = []
        
        for size in test_sizes:
            self.logger.info(f"📊 Testing dataset size: {size}")
            
            for method in self.config.test_methods:
                try:
                    result = await self.run_single_benchmark(method, size, manager)
                    all_results.append(result)
                    
                    if result.success:
                        self.logger.info(
                            f"✅ {method.upper()}: {result.videos_per_second:.1f} videos/sec, "
                            f"{result.peak_memory_mb:.1f} MB peak memory"
                        )
                    else:
                        self.logger.warning(f"❌ {method.upper()}: Failed - {result.error}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {method.upper()}: Exception - {e}")
                    
                # Brief pause between tests
                await asyncio.sleep(1)
        
        # Cleanup
        await manager.cleanup()
        
        suite_duration = time.time() - suite_start
        
        return BenchmarkSuite(
            config=self.config,
            results=all_results,
            system_info=self.system_info,
            timestamp=datetime.now().isoformat(),
            total_duration=suite_duration
        )
    
    async def run_stress_test(self, method: str = "auto") -> Dict[str, Any]:
        """Run stress test with increasing load."""
        
        self.logger.info(f"🔥 Starting Stress Test with {method.upper()}")
        
        manager = await create_ultra_performance_manager("production")
        
        stress_results = {
            'method': method,
            'max_concurrent': self.config.max_concurrent,
            'duration': self.config.stress_duration,
            'results': []
        }
        
        # Generate test data
        test_data = self.generate_test_data(1000)
        
        start_time = time.time()
        
        while time.time() - start_time < self.config.stress_duration:
            concurrent_tasks = []
            
            # Gradually increase load
            current_concurrent = min(
                self.config.max_concurrent,
                int((time.time() - start_time) / 10) + 1
            )
            
            for _ in range(current_concurrent):
                task = manager.process_videos_ultra_performance(test_data, method=method)
                concurrent_tasks.append(task)
            
            # Run concurrent tasks
            iteration_start = time.time()
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            iteration_time = time.time() - iteration_start
            
            # Calculate success rate
            successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
            success_rate = successful / len(results)
            
            stress_results['results'].append({
                'timestamp': time.time() - start_time,
                'concurrent_tasks': current_concurrent,
                'success_rate': success_rate,
                'avg_processing_time': iteration_time / len(results),
                'memory_usage': psutil.virtual_memory().percent
            })
            
            self.logger.info(
                f"🔥 Stress: {current_concurrent} concurrent, "
                f"{success_rate*100:.1f}% success rate"
            )
        
        await manager.cleanup()
        return stress_results

# =============================================================================
# BENCHMARK REPORTER
# =============================================================================

class BenchmarkReporter:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, suite: BenchmarkSuite):
        
    """__init__ function."""
self.suite = suite
    
    def generate_text_report(self) -> str:
        """Generate detailed text report."""
        report = []
        report.append("🚀 ULTRA PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {self.suite.timestamp}")
        report.append(f"Total Duration: {self.suite.total_duration:.2f}s")
        report.append("")
        
        # System info
        report.append("💻 SYSTEM INFORMATION")
        report.append("-" * 30)
        for key, value in self.suite.system_info.items():
            report.append(f"{key}: {value}")
        report.append("")
        
        # Results by method
        methods = set(r.method for r in self.suite.results)
        
        for method in sorted(methods):
            method_results = self.suite.get_results_by_method(method)
            successful_results = [r for r in method_results if r.success]
            
            report.append(f"📊 {method.upper()} RESULTS")
            report.append("-" * 30)
            
            if successful_results:
                avg_vps = sum(r.videos_per_second for r in successful_results) / len(successful_results)
                avg_memory = sum(r.peak_memory_mb for r in successful_results) / len(successful_results)
                
                report.append(f"Success Rate: {len(successful_results)}/{len(method_results)} ({len(successful_results)/len(method_results)*100:.1f}%)")
                report.append(f"Avg Videos/Second: {avg_vps:.1f}")
                report.append(f"Avg Peak Memory: {avg_memory:.1f} MB")
                
                # Performance by dataset size
                for result in successful_results:
                    report.append(f"  {result.dataset_size:>6} videos: {result.videos_per_second:>8.1f} v/s, {result.peak_memory_mb:>6.1f} MB")
            else:
                report.append("❌ All tests failed")
                
            report.append("")
        
        # Best methods
        best_methods = self.suite.get_best_method_by_size()
        report.append("🏆 BEST METHODS BY DATASET SIZE")
        report.append("-" * 30)
        for size, method in best_methods.items():
            report.append(f"{size:>6} videos: {method.upper()}")
        
        return "\n".join(report)
    
    def save_json_report(self, filepath: str):
        """Save detailed JSON report."""
        report_data = {
            'suite': {
                'config': asdict(self.suite.config),
                'system_info': self.suite.system_info,
                'timestamp': self.suite.timestamp,
                'total_duration': self.suite.total_duration
            },
            'results': [r.to_dict() for r in self.suite.results],
            'summary': {
                'best_methods': self.suite.get_best_method_by_size(),
                'total_tests': len(self.suite.results),
                'successful_tests': len([r for r in self.suite.results if r.success])
            }
        }
        
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report_data, f, indent=2)
    
    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """Generate performance comparison plots."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available for plotting")
            return
        
        # Prepare data for plotting
        successful_results = [r for r in self.suite.results if r.success]
        
        if not successful_results:
            print("No successful results to plot")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Videos per second by method and size
        methods = sorted(set(r.method for r in successful_results))
        sizes = sorted(set(r.dataset_size for r in successful_results))
        
        for method in methods:
            method_results = [r for r in successful_results if r.method == method]
            method_sizes = [r.dataset_size for r in method_results]
            method_vps = [r.videos_per_second for r in method_results]
            ax1.plot(method_sizes, method_vps, marker='o', label=method.upper())
        
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Videos per Second')
        ax1.set_title('Performance by Dataset Size')
        ax1.legend()
        ax1.set_xscale('log')
        
        # Memory usage comparison
        for method in methods:
            method_results = [r for r in successful_results if r.method == method]
            method_sizes = [r.dataset_size for r in method_results]
            method_memory = [r.peak_memory_mb for r in method_results]
            ax2.plot(method_sizes, method_memory, marker='s', label=method.upper())
        
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Memory Usage by Dataset Size')
        ax2.legend()
        ax2.set_xscale('log')
        
        # Latency percentiles
        latency_data = defaultdict(list)
        for result in successful_results:
            latency_data[result.method].extend([result.latency_p50, result.latency_p95, result.latency_p99])
        
        for i, method in enumerate(methods):
            if method in latency_data:
                ax3.boxplot(latency_data[method], positions=[i], labels=[method.upper()])
        
        ax3.set_ylabel('Latency (seconds)')
        ax3.set_title('Latency Distribution by Method')
        
        # Success rate by method
        method_success_rates = {}
        for method in methods:
            method_results = self.suite.get_results_by_method(method)
            success_rate = len([r for r in method_results if r.success]) / len(method_results)
            method_success_rates[method] = success_rate * 100
        
        ax4.bar(method_success_rates.keys(), method_success_rates.values())
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Success Rate by Method')
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance plots saved to {save_path}")
        else:
            plt.show()

# =============================================================================
# MAIN DEMO
# =============================================================================

async def run_full_benchmark_demo():
    """Run complete benchmark demonstration."""
    
    print("🚀 Starting Advanced Benchmark System Demo")
    print("=" * 50)
    
    # Configure benchmark
    config = BenchmarkConfig(
        small_dataset=50,
        medium_dataset=500,
        large_dataset=2000,
        xl_dataset=5000,
        warmup_runs=2,
        benchmark_runs=3,
        test_methods=["polars", "gpu", "ray", "fallback"]
    )
    
    # Create benchmark runner
    runner = AdvancedBenchmarkRunner(config)
    
    try:
        # Run comprehensive benchmark
        print("📊 Running comprehensive benchmark...")
        suite = await runner.run_comprehensive_benchmark()
        
        # Generate report
        reporter = BenchmarkReporter(suite)
        
        # Print text report
        print("\n" + reporter.generate_text_report())
        
        # Save JSON report
        json_path = "benchmark_results.json"
        reporter.save_json_report(json_path)
        print(f"\n📄 Detailed results saved to {json_path}")
        
        # Generate plots if available
        if PLOTTING_AVAILABLE:
            plot_path = "benchmark_plots.png"
            reporter.plot_performance_comparison(plot_path)
        
        # Run stress test
        print("\n🔥 Running stress test...")
        stress_results = await runner.run_stress_test("auto")
        print(f"Stress test completed: {len(stress_results['results'])} iterations")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()

match __name__:
    case "__main__":
    asyncio.run(run_full_benchmark_demo()) 