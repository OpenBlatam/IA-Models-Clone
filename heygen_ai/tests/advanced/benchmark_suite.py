"""
Advanced benchmarking suite for HeyGen AI system.
Comprehensive performance testing with detailed metrics and analysis.
"""

import time
import asyncio
import statistics
import psutil
import gc
import json
import threading
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import memory_profiler
import cProfile
import pstats
import io

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    iterations: int = 100
    warmup_iterations: int = 10
    timeout: float = 300.0
    memory_profiling: bool = True
    cpu_profiling: bool = True
    parallel_execution: bool = False
    max_workers: int = 4
    output_dir: str = "benchmark_reports"
    generate_plots: bool = True
    generate_reports: bool = True

@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    name: str
    function_name: str
    iterations: int
    total_duration: float
    average_duration: float
    min_duration: float
    max_duration: float
    median_duration: float
    std_deviation: float
    throughput: float  # operations per second
    memory_usage: Dict[str, float]
    cpu_usage: Dict[str, float]
    percentiles: Dict[str, float]
    error_rate: float
    success_count: int
    failure_count: int
    timestamp: datetime = field(default_factory=datetime.now)

class MemoryProfiler:
    """Advanced memory profiling for benchmarks."""
    
    def __init__(self):
        self.initial_memory = 0
        self.peak_memory = 0
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_profiling(self):
        """Start memory profiling."""
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.memory_samples = []
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
    
    def stop_profiling(self):
        """Stop memory profiling."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_memory(self):
        """Monitor memory usage."""
        while self.monitoring:
            try:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_samples.append(current_memory)
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.01)  # Sample every 10ms
            except Exception:
                break
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.memory_samples:
            return {}
        
        return {
            "initial_mb": self.initial_memory,
            "peak_mb": self.peak_memory,
            "final_mb": self.memory_samples[-1] if self.memory_samples else self.initial_memory,
            "average_mb": statistics.mean(self.memory_samples),
            "std_deviation_mb": statistics.stdev(self.memory_samples) if len(self.memory_samples) > 1 else 0,
            "memory_growth_mb": (self.memory_samples[-1] if self.memory_samples else self.initial_memory) - self.initial_memory
        }

class CPUProfiler:
    """Advanced CPU profiling for benchmarks."""
    
    def __init__(self):
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_profiling(self):
        """Start CPU profiling."""
        self.cpu_samples = []
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_cpu, daemon=True)
        self.monitor_thread.start()
    
    def stop_profiling(self):
        """Stop CPU profiling."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_cpu(self):
        """Monitor CPU usage."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.01)
                self.cpu_samples.append(cpu_percent)
                time.sleep(0.01)
            except Exception:
                break
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU usage statistics."""
        if not self.cpu_samples:
            return {}
        
        return {
            "average_percent": statistics.mean(self.cpu_samples),
            "max_percent": max(self.cpu_samples),
            "min_percent": min(self.cpu_samples),
            "std_deviation_percent": statistics.stdev(self.cpu_samples) if len(self.cpu_samples) > 1 else 0
        }

class AdvancedBenchmarkSuite:
    """Advanced benchmarking suite with comprehensive analysis."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.memory_profiler = MemoryProfiler()
        self.cpu_profiler = CPUProfiler()
        self.benchmark_functions: Dict[str, Callable] = {}
    
    def register_benchmark(self, name: str, function: Callable):
        """Register a function for benchmarking."""
        self.benchmark_functions[name] = function
    
    def run_benchmark(self, name: str, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        if name not in self.benchmark_functions:
            raise ValueError(f"Benchmark '{name}' not registered")
        
        function = self.benchmark_functions[name]
        print(f"🔄 Running benchmark: {name}")
        
        # Start profiling
        if self.config.memory_profiling:
            self.memory_profiler.start_profiling()
        if self.config.cpu_profiling:
            self.cpu_profiler.start_profiling()
        
        # Warmup iterations
        print(f"   Warming up with {self.config.warmup_iterations} iterations...")
        for _ in range(self.config.warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(function):
                    asyncio.run(function(*args, **kwargs))
                else:
                    function(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Force garbage collection
        gc.collect()
        
        # Main benchmark iterations
        durations = []
        success_count = 0
        failure_count = 0
        
        print(f"   Running {self.config.iterations} benchmark iterations...")
        
        for i in range(self.config.iterations):
            start_time = time.perf_counter()
            
            try:
                if asyncio.iscoroutinefunction(function):
                    asyncio.run(function(*args, **kwargs))
                else:
                    function(*args, **kwargs)
                
                duration = time.perf_counter() - start_time
                durations.append(duration)
                success_count += 1
                
            except Exception as e:
                failure_count += 1
                print(f"   Warning: Iteration {i+1} failed: {e}")
        
        # Stop profiling
        if self.config.memory_profiling:
            self.memory_profiler.stop_profiling()
        if self.config.cpu_profiling:
            self.cpu_profiler.stop_profiling()
        
        # Calculate statistics
        if not durations:
            raise RuntimeError("All benchmark iterations failed")
        
        total_duration = sum(durations)
        average_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        median_duration = statistics.median(durations)
        std_deviation = statistics.stdev(durations) if len(durations) > 1 else 0
        throughput = len(durations) / total_duration if total_duration > 0 else 0
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        percentiles = {
            "p50": sorted_durations[int(len(sorted_durations) * 0.5)],
            "p90": sorted_durations[int(len(sorted_durations) * 0.9)],
            "p95": sorted_durations[int(len(sorted_durations) * 0.95)],
            "p99": sorted_durations[int(len(sorted_durations) * 0.99)]
        }
        
        # Get profiling data
        memory_stats = self.memory_profiler.get_memory_stats() if self.config.memory_profiling else {}
        cpu_stats = self.cpu_profiler.get_cpu_stats() if self.config.cpu_profiling else {}
        
        # Create result
        result = BenchmarkResult(
            name=name,
            function_name=function.__name__,
            iterations=len(durations),
            total_duration=total_duration,
            average_duration=average_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            median_duration=median_duration,
            std_deviation=std_deviation,
            throughput=throughput,
            memory_usage=memory_stats,
            cpu_usage=cpu_stats,
            percentiles=percentiles,
            error_rate=failure_count / (success_count + failure_count) * 100,
            success_count=success_count,
            failure_count=failure_count
        )
        
        self.results.append(result)
        self._print_benchmark_result(result)
        
        return result
    
    def run_all_benchmarks(self, *args, **kwargs) -> List[BenchmarkResult]:
        """Run all registered benchmarks."""
        print("🚀 Starting Advanced Benchmark Suite")
        print("=" * 60)
        
        all_results = []
        
        for name in self.benchmark_functions:
            try:
                result = self.run_benchmark(name, *args, **kwargs)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Benchmark '{name}' failed: {e}")
        
        self._generate_comprehensive_report()
        return all_results
    
    def _print_benchmark_result(self, result: BenchmarkResult):
        """Print individual benchmark result."""
        print(f"   ✅ {result.name} completed")
        print(f"      Duration: {result.average_duration*1000:.2f}ms avg, {result.min_duration*1000:.2f}ms min, {result.max_duration*1000:.2f}ms max")
        print(f"      Throughput: {result.throughput:.2f} ops/sec")
        print(f"      Success Rate: {100-result.error_rate:.1f}%")
        
        if result.memory_usage:
            print(f"      Memory: {result.memory_usage.get('peak_mb', 0):.1f}MB peak, {result.memory_usage.get('memory_growth_mb', 0):.1f}MB growth")
        
        if result.cpu_usage:
            print(f"      CPU: {result.cpu_usage.get('average_percent', 0):.1f}% avg, {result.cpu_usage.get('max_percent', 0):.1f}% max")
        
        print()
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive benchmark report."""
        if not self.results:
            return
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE BENCHMARK REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_benchmarks = len(self.results)
        total_iterations = sum(r.iterations for r in self.results)
        total_duration = sum(r.total_duration for r in self.results)
        
        print(f"📈 Summary:")
        print(f"   Benchmarks: {total_benchmarks}")
        print(f"   Total Iterations: {total_iterations}")
        print(f"   Total Duration: {total_duration:.2f}s")
        print(f"   Average Throughput: {total_iterations/total_duration:.2f} ops/sec")
        
        # Performance ranking
        print(f"\n🏆 Performance Ranking (by throughput):")
        sorted_results = sorted(self.results, key=lambda r: r.throughput, reverse=True)
        for i, result in enumerate(sorted_results, 1):
            print(f"   {i}. {result.name}: {result.throughput:.2f} ops/sec")
        
        # Memory usage ranking
        memory_results = [r for r in self.results if r.memory_usage]
        if memory_results:
            print(f"\n🧠 Memory Usage Ranking (by peak memory):")
            sorted_memory = sorted(memory_results, key=lambda r: r.memory_usage.get('peak_mb', 0), reverse=True)
            for i, result in enumerate(sorted_memory, 1):
                peak_mb = result.memory_usage.get('peak_mb', 0)
                print(f"   {i}. {result.name}: {peak_mb:.1f}MB peak")
        
        # Error analysis
        error_results = [r for r in self.results if r.error_rate > 0]
        if error_results:
            print(f"\n⚠️  Error Analysis:")
            for result in error_results:
                print(f"   {result.name}: {result.error_rate:.1f}% error rate ({result.failure_count} failures)")
        
        print("=" * 60)
        
        # Save detailed report
        if self.config.generate_reports:
            self._save_detailed_report()
        
        # Generate plots
        if self.config.generate_plots:
            self._generate_plots()
    
    def _save_detailed_report(self):
        """Save detailed benchmark report to file."""
        report_dir = Path(self.config.output_dir)
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"benchmark_report_{timestamp}.json"
        
        # Convert results to serializable format
        report_data = {
            "config": {
                "iterations": self.config.iterations,
                "warmup_iterations": self.config.warmup_iterations,
                "timeout": self.config.timeout,
                "memory_profiling": self.config.memory_profiling,
                "cpu_profiling": self.config.cpu_profiling
            },
            "summary": {
                "total_benchmarks": len(self.results),
                "total_iterations": sum(r.iterations for r in self.results),
                "total_duration": sum(r.total_duration for r in self.results),
                "average_throughput": sum(r.throughput for r in self.results) / len(self.results) if self.results else 0
            },
            "results": []
        }
        
        for result in self.results:
            result_data = {
                "name": result.name,
                "function_name": result.function_name,
                "iterations": result.iterations,
                "total_duration": result.total_duration,
                "average_duration": result.average_duration,
                "min_duration": result.min_duration,
                "max_duration": result.max_duration,
                "median_duration": result.median_duration,
                "std_deviation": result.std_deviation,
                "throughput": result.throughput,
                "memory_usage": result.memory_usage,
                "cpu_usage": result.cpu_usage,
                "percentiles": result.percentiles,
                "error_rate": result.error_rate,
                "success_count": result.success_count,
                "failure_count": result.failure_count,
                "timestamp": result.timestamp.isoformat()
            }
            report_data["results"].append(result_data)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"💾 Detailed report saved to: {report_file}")
    
    def _generate_plots(self):
        """Generate performance plots."""
        if not self.results:
            return
        
        try:
            report_dir = Path(self.config.output_dir)
            report_dir.mkdir(exist_ok=True)
            
            # Throughput comparison
            plt.figure(figsize=(12, 8))
            names = [r.name for r in self.results]
            throughputs = [r.throughput for r in self.results]
            
            plt.subplot(2, 2, 1)
            plt.bar(names, throughputs)
            plt.title('Throughput Comparison')
            plt.ylabel('Operations per Second')
            plt.xticks(rotation=45)
            
            # Duration distribution
            plt.subplot(2, 2, 2)
            durations = [r.average_duration * 1000 for r in self.results]  # Convert to ms
            plt.bar(names, durations)
            plt.title('Average Duration Comparison')
            plt.ylabel('Duration (ms)')
            plt.xticks(rotation=45)
            
            # Memory usage
            memory_results = [r for r in self.results if r.memory_usage]
            if memory_results:
                plt.subplot(2, 2, 3)
                memory_names = [r.name for r in memory_results]
                memory_peaks = [r.memory_usage.get('peak_mb', 0) for r in memory_results]
                plt.bar(memory_names, memory_peaks)
                plt.title('Peak Memory Usage')
                plt.ylabel('Memory (MB)')
                plt.xticks(rotation=45)
            
            # Error rates
            plt.subplot(2, 2, 4)
            error_rates = [r.error_rate for r in self.results]
            plt.bar(names, error_rates)
            plt.title('Error Rates')
            plt.ylabel('Error Rate (%)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = report_dir / f"benchmark_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance plots saved to: {plot_file}")
            
        except ImportError:
            print("⚠️  matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")

# Example benchmark functions
def cpu_intensive_task(n: int = 1000000):
    """CPU intensive task for benchmarking."""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def memory_intensive_task(size: int = 1000000):
    """Memory intensive task for benchmarking."""
    data = [i for i in range(size)]
    return sum(data)

def io_intensive_task():
    """IO intensive task for benchmarking."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test data" * 1000)
        temp_file = f.name
    
    with open(temp_file, 'r') as f:
        content = f.read()
    
    Path(temp_file).unlink()
    return len(content)

async def async_task():
    """Async task for benchmarking."""
    await asyncio.sleep(0.01)
    return "async_result"

def demo_benchmark_suite():
    """Demonstrate advanced benchmark suite."""
    print("🔄 Advanced Benchmark Suite Demo")
    print("=" * 40)
    
    # Create benchmark suite
    config = BenchmarkConfig(
        iterations=50,
        warmup_iterations=5,
        memory_profiling=True,
        cpu_profiling=True,
        generate_plots=True
    )
    
    suite = AdvancedBenchmarkSuite(config)
    
    # Register benchmarks
    suite.register_benchmark("CPU Intensive", cpu_intensive_task)
    suite.register_benchmark("Memory Intensive", memory_intensive_task)
    suite.register_benchmark("IO Intensive", io_intensive_task)
    suite.register_benchmark("Async Task", async_task)
    
    # Run all benchmarks
    results = suite.run_all_benchmarks()
    
    return results

if __name__ == "__main__":
    # Run demo
    demo_benchmark_suite()
