import time
import statistics
from record_storage import RecordStorage
from typing import List, Dict, Any


def generate_test_records(count: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": str(i),
            "name": f"User_{i}",
            "age": 20 + (i % 50),
            "city": f"City_{i % 10}",
            "department": f"Dept_{i % 5}",
            "status": "active" if i % 2 == 0 else "inactive"
        }
        for i in range(count)
    ]


def benchmark_write(storage: RecordStorage, records: List[Dict[str, Any]], iterations: int = 10) -> Dict[str, float]:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        storage.write(records)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0
    }


def benchmark_read(storage: RecordStorage, iterations: int = 100) -> Dict[str, float]:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        storage.read()
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0
    }


def benchmark_update(storage: RecordStorage, record_count: int, iterations: int = 50) -> Dict[str, float]:
    times = []
    for i in range(iterations):
        record_id = str(i % record_count)
        updates = {"updated": True, "iteration": i}
        start = time.perf_counter()
        storage.update(record_id, updates)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0
    }


def run_benchmark_suite():
    print("="*60)
    print("RecordStorage Performance Benchmark")
    print("="*60)
    
    test_sizes = [10, 100, 1000, 5000]
    
    for size in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {size} records")
        print(f"{'='*60}")
        
        storage = RecordStorage(f"benchmark_{size}.json")
        test_records = generate_test_records(size)
        
        print(f"\n📝 Write Performance ({size} records):")
        write_stats = benchmark_write(storage, test_records, iterations=5)
        print(f"   Mean:   {write_stats['mean']*1000:.2f} ms")
        print(f"   Median: {write_stats['median']*1000:.2f} ms")
        print(f"   Min:    {write_stats['min']*1000:.2f} ms")
        print(f"   Max:    {write_stats['max']*1000:.2f} ms")
        
        print(f"\n📖 Read Performance ({size} records):")
        read_stats = benchmark_read(storage, iterations=50)
        print(f"   Mean:   {read_stats['mean']*1000:.2f} ms")
        print(f"   Median: {read_stats['median']*1000:.2f} ms")
        print(f"   Min:    {read_stats['min']*1000:.2f} ms")
        print(f"   Max:    {read_stats['max']*1000:.2f} ms")
        
        print(f"\n✏️  Update Performance ({size} records):")
        update_stats = benchmark_update(storage, size, iterations=20)
        print(f"   Mean:   {update_stats['mean']*1000:.2f} ms")
        print(f"   Median: {update_stats['median']*1000:.2f} ms")
        print(f"   Min:    {update_stats['min']*1000:.2f} ms")
        print(f"   Max:    {update_stats['max']*1000:.2f} ms")
        
        records_per_second_read = size / read_stats['mean']
        print(f"\n📊 Throughput:")
        print(f"   Read: {records_per_second_read:,.0f} records/second")
    
    print(f"\n{'='*60}")
    print("✅ Benchmark completed!")
    print(f"{'='*60}")


def compare_context_manager_vs_manual():
    print("\n" + "="*60)
    print("Context Manager vs Manual File Handling")
    print("="*60)
    
    import json
    from pathlib import Path
    
    test_file = Path("comparison_test.json")
    test_data = {"records": generate_test_records(100)}
    
    iterations = 100
    
    print("\n⏱️  Testing Context Manager (with statement):")
    times_with = []
    for _ in range(iterations):
        start = time.perf_counter()
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        end = time.perf_counter()
        times_with.append(end - start)
    
    print(f"   Mean: {statistics.mean(times_with)*1000:.2f} ms")
    print(f"   Safe: ✅ Files always closed")
    
    print("\n⏱️  Testing Manual File Handling (open/close):")
    times_manual = []
    for _ in range(iterations):
        start = time.perf_counter()
        f = open(test_file, 'w', encoding='utf-8')
        json.dump(test_data, f)
        f.close()
        end = time.perf_counter()
        times_manual.append(end - start)
    
    print(f"   Mean: {statistics.mean(times_manual)*1000:.2f} ms")
    print(f"   Safe: ❌ Files may not close on exceptions")
    
    if test_file.exists():
        test_file.unlink()
    
    print("\n💡 Recommendation: Use context managers for safety!")


if __name__ == "__main__":
    run_benchmark_suite()
    compare_context_manager_vs_manual()


