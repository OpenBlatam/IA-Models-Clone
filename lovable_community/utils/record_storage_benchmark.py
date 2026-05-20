"""
Record Storage Benchmark

Performance benchmarking for RecordStorage operations.
"""

import time
import statistics
from pathlib import Path
import tempfile
import shutil

from .record_storage import RecordStorage


class Benchmark:
    """Benchmark RecordStorage operations."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results = {}
    
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def time_operation(self, operation, iterations=100):
        """Time an operation multiple times."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return times
    
    def benchmark_write(self, record_count=100):
        """Benchmark write operations."""
        file_path = self.temp_dir / "bench_write.json"
        storage = RecordStorage(str(file_path))
        
        records = [{"id": str(i), "data": f"record_{i}"} for i in range(record_count)]
        
        times = self.time_operation(lambda: storage.write(records), iterations=50)
        
        return {
            "operation": "write",
            "record_count": record_count,
            "iterations": 50,
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_read(self, record_count=100):
        """Benchmark read operations."""
        file_path = self.temp_dir / "bench_read.json"
        storage = RecordStorage(str(file_path))
        
        records = [{"id": str(i), "data": f"record_{i}"} for i in range(record_count)]
        storage.write(records)
        
        times = self.time_operation(lambda: storage.read(), iterations=100)
        
        return {
            "operation": "read",
            "record_count": record_count,
            "iterations": 100,
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_update(self, record_count=100):
        """Benchmark update operations."""
        file_path = self.temp_dir / "bench_update.json"
        storage = RecordStorage(str(file_path))
        
        records = [{"id": str(i), "data": f"record_{i}"} for i in range(record_count)]
        storage.write(records)
        
        times = self.time_operation(
            lambda: storage.update("50", {"updated": True}),
            iterations=100
        )
        
        return {
            "operation": "update",
            "record_count": record_count,
            "iterations": 100,
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_add(self, initial_count=100):
        """Benchmark add operations."""
        file_path = self.temp_dir / "bench_add.json"
        storage = RecordStorage(str(file_path))
        
        records = [{"id": str(i), "data": f"record_{i}"} for i in range(initial_count)]
        storage.write(records)
        
        counter = initial_count
        times = []
        for _ in range(50):
            start = time.perf_counter()
            storage.add({"id": str(counter), "data": f"record_{counter}"})
            end = time.perf_counter()
            times.append((end - start) * 1000)
            counter += 1
        
        return {
            "operation": "add",
            "initial_count": initial_count,
            "iterations": 50,
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_get(self, record_count=100):
        """Benchmark get operations."""
        file_path = self.temp_dir / "bench_get.json"
        storage = RecordStorage(str(file_path))
        
        records = [{"id": str(i), "data": f"record_{i}"} for i in range(record_count)]
        storage.write(records)
        
        times = self.time_operation(
            lambda: storage.get("50"),
            iterations=100
        )
        
        return {
            "operation": "get",
            "record_count": record_count,
            "iterations": 100,
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("\n" + "=" * 70)
        print("  Record Storage - Performance Benchmark")
        print("=" * 70 + "\n")
        
        print("Running benchmarks...\n")
        
        results = []
        
        print("1. Write benchmark (100 records)...")
        results.append(self.benchmark_write(100))
        
        print("2. Read benchmark (100 records)...")
        results.append(self.benchmark_read(100))
        
        print("3. Update benchmark (100 records)...")
        results.append(self.benchmark_update(100))
        
        print("4. Add benchmark (starting with 100 records)...")
        results.append(self.benchmark_add(100))
        
        print("5. Get benchmark (100 records)...")
        results.append(self.benchmark_get(100))
        
        print("\n" + "=" * 70)
        print("  Benchmark Results")
        print("=" * 70 + "\n")
        
        for result in results:
            print(f"Operation: {result['operation']}")
            print(f"  Records: {result.get('record_count', result.get('initial_count', 'N/A'))}")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Mean: {result['mean_ms']:.3f} ms")
            print(f"  Median: {result['median_ms']:.3f} ms")
            print(f"  Min: {result['min_ms']:.3f} ms")
            print(f"  Max: {result['max_ms']:.3f} ms")
            print(f"  Std Dev: {result['stdev_ms']:.3f} ms")
            print()
        
        print("=" * 70)
        print("  Summary")
        print("=" * 70)
        
        total_mean = sum(r['mean_ms'] for r in results)
        print(f"Total mean time: {total_mean:.3f} ms")
        print(f"Average per operation: {total_mean / len(results):.3f} ms")
        
        return results


def main():
    """Run benchmarks."""
    benchmark = Benchmark()
    try:
        results = benchmark.run_all_benchmarks()
        print("\n✅ Benchmark complete!")
        return 0
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    exit(main())


