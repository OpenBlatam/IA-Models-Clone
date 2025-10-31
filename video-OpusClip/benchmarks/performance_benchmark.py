"""
Performance Benchmark

Comprehensive benchmark to measure the performance of the parallel processing system.
"""

import time
import asyncio
import multiprocessing as mp
from typing import List, Dict, Any
import statistics
import structlog

from ..models.video_models import VideoClipRequest
from ..processors.video_processor import create_high_performance_processor
from ..processors.viral_processor import create_high_performance_viral_processor
from ..utils.parallel_utils import (
    BackendType,
    ParallelConfig,
    setup_async_loop,
    estimate_processing_time
)

logger = structlog.get_logger()

# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================

class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    
    def __init__(self):
        self.sample_sizes = [10, 50, 100, 500, 1000]
        self.backends = [
            BackendType.THREAD,
            BackendType.PROCESS,
            BackendType.JOBLIB,
            BackendType.DASK,
            BackendType.ASYNC
        ]
        self.iterations = 3
        self.warmup_iterations = 2

# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def generate_benchmark_data(size: int) -> List[VideoClipRequest]:
    """Generate benchmark data of specified size."""
    base_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
        "https://www.youtube.com/watch?v=ZZ5LpwO-An4",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0"
    ]
    
    requests = []
    for i in range(size):
        url = base_urls[i % len(base_urls)]
        requests.append(VideoClipRequest(
            youtube_url=f"{url}&t={i}",
            language="en" if i % 2 == 0 else "es",
            max_clip_length=60 + (i % 30),
            min_clip_length=15 + (i % 10)
        ))
    
    return requests

def measure_performance(func, *args, **kwargs) -> Dict[str, Any]:
    """Measure performance of a function."""
    # Warmup
    for _ in range(BenchmarkConfig().warmup_iterations):
        try:
            func(*args, **kwargs)
        except:
            pass
    
    # Actual measurement
    times = []
    for _ in range(BenchmarkConfig().iterations):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            times.append(duration)
        except Exception as e:
            logger.warning(f"Benchmark iteration failed: {e}")
            times.append(float('inf'))
    
    return {
        'min_time': min(times),
        'max_time': max(times),
        'avg_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
        'success_rate': len([t for t in times if t != float('inf')]) / len(times)
    }

# =============================================================================
# BENCHMARK TESTS
# =============================================================================

class VideoProcessingBenchmark:
    """Comprehensive benchmark for video processing performance."""
    
    def __init__(self):
        self.config = BenchmarkConfig()
        self.video_processor = create_high_performance_processor()
        self.viral_processor = create_high_performance_viral_processor()
        setup_async_loop()
    
    def benchmark_backend_performance(self) -> Dict[str, Dict]:
        """Benchmark performance across different backends."""
        print("🔄 Benchmarking backend performance...")
        
        results = {}
        
        for size in self.config.sample_sizes:
            print(f"  Testing with {size} items...")
            data = generate_benchmark_data(size)
            size_results = {}
            
            for backend in self.config.backends:
                print(f"    Backend: {backend.value}")
                
                if backend == BackendType.ASYNC:
                    def async_wrapper():
                        return asyncio.run(self.video_processor.process_batch_async(data))
                    performance = measure_performance(async_wrapper)
                else:
                    def sync_wrapper():
                        return self.video_processor.process_batch_parallel(data, backend=backend)
                    performance = measure_performance(sync_wrapper)
                
                size_results[backend.value] = performance
            
            results[f"size_{size}"] = size_results
        
        return results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability with different data sizes."""
        print("📈 Benchmarking scalability...")
        
        scalability_results = {}
        
        for size in self.config.sample_sizes:
            data = generate_benchmark_data(size)
            
            # Auto backend
            performance = measure_performance(
                self.video_processor.process_batch_parallel, data
            )
            
            scalability_results[f"size_{size}"] = {
                'items_per_second': size / performance['avg_time'],
                'performance': performance
            }
        
        return scalability_results
    
    def benchmark_viral_processing(self) -> Dict[str, Any]:
        """Benchmark viral content processing."""
        print("🔥 Benchmarking viral processing...")
        
        viral_results = {}
        
        for size in [5, 10, 20, 50]:  # Smaller sizes for viral processing
            data = generate_benchmark_data(size)
            
            performance = measure_performance(
                self.viral_processor.process_batch_parallel,
                data,
                n_variants=5
            )
            
            viral_results[f"size_{size}"] = {
                'total_variants': size * 5,
                'variants_per_second': (size * 5) / performance['avg_time'],
                'performance': performance
            }
        
        return viral_results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("💾 Benchmarking memory usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_results = {}
        
        for size in [100, 500, 1000]:
            data = generate_benchmark_data(size)
            
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process data
            start_time = time.perf_counter()
            results = self.video_processor.process_batch_parallel(data)
            duration = time.perf_counter() - start_time
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            memory_results[f"size_{size}"] = {
                'memory_used_mb': memory_used,
                'memory_per_item_mb': memory_used / size,
                'processing_time': duration,
                'items_per_second': size / duration
            }
        
        return memory_results
    
    def benchmark_error_handling(self) -> Dict[str, Any]:
        """Benchmark error handling and recovery."""
        print("🛡️ Benchmarking error handling...")
        
        # Create data with some invalid requests
        valid_data = generate_benchmark_data(50)
        invalid_data = [
            VideoClipRequest(
                youtube_url="invalid_url",
                max_clip_length=0,  # Invalid
                min_clip_length=100  # Invalid
            )
        ] * 10
        
        mixed_data = valid_data + invalid_data
        
        error_results = {}
        
        for backend in self.config.backends:
            print(f"  Testing error handling with {backend.value}")
            
            try:
                if backend == BackendType.ASYNC:
                    def async_error_wrapper():
                        return asyncio.run(self.video_processor.process_batch_async(mixed_data))
                    performance = measure_performance(async_error_wrapper)
                else:
                    def sync_error_wrapper():
                        return self.video_processor.process_batch_parallel(mixed_data, backend=backend)
                    performance = measure_performance(sync_error_wrapper)
                
                error_results[backend.value] = {
                    'success_rate': performance['success_rate'],
                    'avg_time': performance['avg_time']
                }
                
            except Exception as e:
                error_results[backend.value] = {
                    'success_rate': 0.0,
                    'error': str(e)
                }
        
        return error_results

# =============================================================================
# BENCHMARK REPORTING
# =============================================================================

class BenchmarkReporter:
    """Generate comprehensive benchmark reports."""
    
    @staticmethod
    def print_backend_comparison(results: Dict[str, Dict]):
        """Print backend comparison results."""
        print("\n" + "="*60)
        print("BACKEND PERFORMANCE COMPARISON")
        print("="*60)
        
        for size_key, backends in results.items():
            size = size_key.replace("size_", "")
            print(f"\n📊 Data Size: {size} items")
            print("-" * 40)
            
            # Sort by average time
            sorted_backends = sorted(
                backends.items(),
                key=lambda x: x[1]['avg_time'] if x[1]['avg_time'] != float('inf') else float('inf')
            )
            
            for backend, metrics in sorted_backends:
                if metrics['avg_time'] == float('inf'):
                    print(f"  {backend:12} | ❌ Failed")
                else:
                    items_per_sec = int(size) / metrics['avg_time']
                    print(f"  {backend:12} | {metrics['avg_time']:.3f}s | {items_per_sec:.1f} items/s")
    
    @staticmethod
    def print_scalability_results(results: Dict[str, Any]):
        """Print scalability results."""
        print("\n" + "="*60)
        print("SCALABILITY ANALYSIS")
        print("="*60)
        
        sizes = []
        items_per_second = []
        
        for size_key, data in results.items():
            size = int(size_key.replace("size_", ""))
            sizes.append(size)
            items_per_second.append(data['items_per_second'])
            print(f"  {size:4} items | {data['items_per_second']:.1f} items/s")
        
        # Calculate scaling factor
        if len(items_per_second) > 1:
            scaling_factor = items_per_second[-1] / items_per_second[0]
            print(f"\n📈 Scaling factor: {scaling_factor:.2f}x")
    
    @staticmethod
    def print_viral_results(results: Dict[str, Any]):
        """Print viral processing results."""
        print("\n" + "="*60)
        print("VIRAL PROCESSING PERFORMANCE")
        print("="*60)
        
        for size_key, data in results.items():
            size = size_key.replace("size_", "")
            print(f"  {size:4} videos | {data['total_variants']:3} variants | {data['variants_per_second']:.1f} variants/s")
    
    @staticmethod
    def print_memory_results(results: Dict[str, Any]):
        """Print memory usage results."""
        print("\n" + "="*60)
        print("MEMORY USAGE ANALYSIS")
        print("="*60)
        
        for size_key, data in results.items():
            size = size_key.replace("size_", "")
            print(f"  {size:4} items | {data['memory_used_mb']:.1f} MB | {data['memory_per_item_mb']:.3f} MB/item | {data['items_per_second']:.1f} items/s")
    
    @staticmethod
    def print_error_handling_results(results: Dict[str, Any]):
        """Print error handling results."""
        print("\n" + "="*60)
        print("ERROR HANDLING ANALYSIS")
        print("="*60)
        
        for backend, data in results.items():
            if 'error' in data:
                print(f"  {backend:12} | ❌ {data['error']}")
            else:
                print(f"  {backend:12} | ✅ {data['success_rate']:.1%} success rate | {data['avg_time']:.3f}s")

# =============================================================================
# MAIN BENCHMARK EXECUTION
# =============================================================================

def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark."""
    print("🚀 Video Processing Performance Benchmark")
    print("=" * 60)
    
    benchmark = VideoProcessingBenchmark()
    reporter = BenchmarkReporter()
    
    try:
        # Run all benchmarks
        print("\n1️⃣ Backend Performance Benchmark")
        backend_results = benchmark.benchmark_backend_performance()
        reporter.print_backend_comparison(backend_results)
        
        print("\n2️⃣ Scalability Benchmark")
        scalability_results = benchmark.benchmark_scalability()
        reporter.print_scalability_results(scalability_results)
        
        print("\n3️⃣ Viral Processing Benchmark")
        viral_results = benchmark.benchmark_viral_processing()
        reporter.print_viral_results(viral_results)
        
        print("\n4️⃣ Memory Usage Benchmark")
        memory_results = benchmark.benchmark_memory_usage()
        reporter.print_memory_results(memory_results)
        
        print("\n5️⃣ Error Handling Benchmark")
        error_results = benchmark.benchmark_error_handling()
        reporter.print_error_handling_results(error_results)
        
        print("\n" + "="*60)
        print("🎉 Benchmark completed successfully!")
        print("="*60)
        
        return {
            'backend_performance': backend_results,
            'scalability': scalability_results,
            'viral_processing': viral_results,
            'memory_usage': memory_results,
            'error_handling': error_results
        }
        
    except Exception as e:
        logger.error("Benchmark execution failed", error=str(e))
        print(f"❌ Benchmark failed: {e}")
        return None

if __name__ == "__main__":
    run_comprehensive_benchmark() 