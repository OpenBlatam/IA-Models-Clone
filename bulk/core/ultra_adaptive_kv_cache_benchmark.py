#!/usr/bin/env python3
"""
Benchmarking Tool for Ultra-Adaptive K/V Cache Engine
Tests performance under various workloads and configurations
"""

import asyncio
import time
import statistics
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
from dataclasses import dataclass
import logging

try:
    from ultra_adaptive_kv_cache_engine import (
        UltraAdaptiveKVCacheEngine,
        AdaptiveConfig,
        AdaptiveMode,
        TruthGPTIntegration
    )
except ImportError:
    print("Warning: Ultra-Adaptive K/V Cache Engine not available")
    print("Benchmark will run with mocks")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    total_requests: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    error_rate: float
    memory_usage: float
    cache_hit_rate: float


class BenchmarkSuite:
    """Comprehensive benchmark suite for the engine."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine):
        self.engine = engine
        self.results: List[BenchmarkResult] = []
    
    async def generate_test_requests(self, count: int, sequence_length: int = 100) -> List[Dict[str, Any]]:
        """Generate test requests."""
        requests = []
        for i in range(count):
            requests.append({
                'text': f'Test request {i}: ' + 'x' * sequence_length,
                'max_length': 50,
                'temperature': 0.7,
                'session_id': f'session_{i % 10}'  # Reuse some sessions
            })
        return requests
    
    async def benchmark_single_request(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark single request processing."""
        logger.info(f"Benchmarking single requests ({iterations} iterations)...")
        
        requests = await self.generate_test_requests(iterations)
        latencies = []
        errors = 0
        
        start_time = time.time()
        
        for request in requests:
            req_start = time.time()
            result = await self.engine.process_request(request)
            req_time = time.time() - req_start
            
            if result['success']:
                latencies.append(req_time)
            else:
                errors += 1
        
        total_time = time.time() - start_time
        
        if latencies:
            latencies.sort()
            return BenchmarkResult(
                name="single_request",
                total_requests=iterations,
                total_time=total_time,
                avg_latency=statistics.mean(latencies),
                p50_latency=latencies[int(len(latencies) * 0.50)],
                p95_latency=latencies[int(len(latencies) * 0.95)],
                p99_latency=latencies[int(len(latencies) * 0.99)],
                throughput=iterations / total_time,
                error_rate=errors / iterations,
                memory_usage=self.engine._get_current_memory_usage(),
                cache_hit_rate=self.engine.performance_metrics.get('cache_hit_rate', 0.0)
            )
        else:
            raise Exception("All requests failed")
    
    async def benchmark_batch_processing(self, batch_sizes: List[int], requests_per_batch: int = 10) -> List[BenchmarkResult]:
        """Benchmark batch processing with different batch sizes."""
        logger.info(f"Benchmarking batch processing (batch sizes: {batch_sizes})...")
        
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")
            
            latencies = []
            total_requests = 0
            
            for _ in range(requests_per_batch):
                requests = await self.generate_test_requests(batch_size)
                
                batch_start = time.time()
                batch_results = await self.engine.process_batch(requests)
                batch_time = time.time() - batch_start
                
                latencies.append(batch_time / batch_size)  # Per-request latency
                total_requests += len(batch_results)
            
            if latencies:
                latencies.sort()
                result = BenchmarkResult(
                    name=f"batch_{batch_size}",
                    total_requests=total_requests,
                    total_time=sum(latencies) * batch_size,
                    avg_latency=statistics.mean(latencies),
                    p50_latency=latencies[int(len(latencies) * 0.50)],
                    p95_latency=latencies[int(len(latencies) * 0.95)],
                    p99_latency=latencies[int(len(latencies) * 0.99)],
                    throughput=total_requests / (sum(latencies) * batch_size),
                    error_rate=0.0,
                    memory_usage=self.engine._get_current_memory_usage(),
                    cache_hit_rate=0.0
                )
                results.append(result)
        
        return results
    
    async def benchmark_concurrent_load(self, concurrent_requests: int = 50, total_requests: int = 500) -> BenchmarkResult:
        """Benchmark under concurrent load."""
        logger.info(f"Benchmarking concurrent load ({concurrent_requests} concurrent, {total_requests} total)...")
        
        requests = await self.generate_test_requests(total_requests)
        latencies = []
        errors = 0
        
        start_time = time.time()
        
        # Process in chunks of concurrent requests
        for i in range(0, total_requests, concurrent_requests):
            chunk = requests[i:i + concurrent_requests]
            
            chunk_start = time.time()
            tasks = [self.engine.process_request(req) for req in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            chunk_time = time.time() - chunk_start
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                elif result['success']:
                    latencies.append(chunk_time / len(chunk))
                else:
                    errors += 1
        
        total_time = time.time() - start_time
        
        if latencies:
            latencies.sort()
            return BenchmarkResult(
                name=f"concurrent_{concurrent_requests}",
                total_requests=total_requests,
                total_time=total_time,
                avg_latency=statistics.mean(latencies),
                p50_latency=latencies[int(len(latencies) * 0.50)],
                p95_latency=latencies[int(len(latencies) * 0.95)],
                p99_latency=latencies[int(len(latencies) * 0.99)],
                throughput=total_requests / total_time,
                error_rate=errors / total_requests,
                memory_usage=self.engine._get_current_memory_usage(),
                cache_hit_rate=self.engine.performance_metrics.get('cache_hit_rate', 0.0)
            )
        else:
            raise Exception("All requests failed")
    
    async def benchmark_cache_performance(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark cache hit performance."""
        logger.info(f"Benchmarking cache performance ({iterations} iterations)...")
        
        # Create requests with session reuse
        requests = []
        for i in range(iterations):
            requests.append({
                'text': f'Cached request {i}',
                'max_length': 50,
                'temperature': 0.7,
                'session_id': f'session_{i % 5}'  # Reuse 5 sessions
            })
        
        latencies = []
        start_time = time.time()
        
        for request in requests:
            req_start = time.time()
            result = await self.engine.process_request(request)
            req_time = time.time() - req_start
            
            if result['success']:
                latencies.append(req_time)
        
        total_time = time.time() - start_time
        
        if latencies:
            latencies.sort()
            return BenchmarkResult(
                name="cache_performance",
                total_requests=iterations,
                total_time=total_time,
                avg_latency=statistics.mean(latencies),
                p50_latency=latencies[int(len(latencies) * 0.50)],
                p95_latency=latencies[int(len(latencies) * 0.95)],
                p99_latency=latencies[int(len(latencies) * 0.99)],
                throughput=iterations / total_time,
                error_rate=0.0,
                memory_usage=self.engine._get_current_memory_usage(),
                cache_hit_rate=self.engine.performance_metrics.get('cache_hit_rate', 0.0)
            )
        else:
            raise Exception("All requests failed")
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\n{result.name.upper()}")
            print(f"  Total Requests: {result.total_requests}")
            print(f"  Total Time: {result.total_time:.3f}s")
            print(f"  Average Latency: {result.avg_latency*1000:.2f}ms")
            print(f"  P50 Latency: {result.p50_latency*1000:.2f}ms")
            print(f"  P95 Latency: {result.p95_latency*1000:.2f}ms")
            print(f"  P99 Latency: {result.p99_latency*1000:.2f}ms")
            print(f"  Throughput: {result.throughput:.2f} req/s")
            print(f"  Error Rate: {result.error_rate*100:.2f}%")
            print(f"  Memory Usage: {result.memory_usage*100:.2f}%")
            print(f"  Cache Hit Rate: {result.cache_hit_rate*100:.2f}%")
    
    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save results to JSON file."""
        data = []
        for result in results:
            data.append({
                'name': result.name,
                'total_requests': result.total_requests,
                'total_time': result.total_time,
                'avg_latency': result.avg_latency,
                'p50_latency': result.p50_latency,
                'p95_latency': result.p95_latency,
                'p99_latency': result.p99_latency,
                'throughput': result.throughput,
                'error_rate': result.error_rate,
                'memory_usage': result.memory_usage,
                'cache_hit_rate': result.cache_hit_rate
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")


async def run_full_benchmark_suite(config: AdaptiveConfig = None):
    """Run full benchmark suite."""
    if config is None:
        config = AdaptiveConfig(
            enable_metrics=True,
            enable_cache_persistence=False,  # Disable for benchmark
            enable_checkpointing=False,
            num_workers=4
        )
    
    # Create engine
    engine = TruthGPTIntegration.create_engine_for_truthgpt()
    
    suite = BenchmarkSuite(engine)
    results = []
    
    try:
        # Run benchmarks
        logger.info("Starting benchmark suite...")
        
        single_result = await suite.benchmark_single_request(iterations=100)
        results.append(single_result)
        
        batch_results = await suite.benchmark_batch_processing(batch_sizes=[1, 5, 10, 20], requests_per_batch=10)
        results.extend(batch_results)
        
        concurrent_result = await suite.benchmark_concurrent_load(concurrent_requests=20, total_requests=200)
        results.append(concurrent_result)
        
        cache_result = await suite.benchmark_cache_performance(iterations=100)
        results.append(cache_result)
        
        # Print and save results
        suite.print_results(results)
        
        output_file = f"benchmark_results_{int(time.time())}.json"
        suite.save_results(results, output_file)
        
        # Print engine stats
        stats = engine.get_performance_stats()
        print("\n" + "="*80)
        print("ENGINE PERFORMANCE STATS")
        print("="*80)
        print(json.dumps(stats, indent=2, default=str))
        
    finally:
        engine.shutdown()
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Ultra-Adaptive K/V Cache Engine")
    parser.add_argument("--mode", choices=["full", "single", "batch", "concurrent", "cache"], 
                       default="full", help="Benchmark mode")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    config = AdaptiveConfig(
        enable_metrics=True,
        enable_cache_persistence=False,
        enable_checkpointing=False,
        num_workers=4
    )
    
    engine = TruthGPTIntegration.create_engine_for_truthgpt()
    suite = BenchmarkSuite(engine)
    results = []
    
    try:
        if args.mode == "full":
            results = asyncio.run(run_full_benchmark_suite(config))
        elif args.mode == "single":
            result = asyncio.run(suite.benchmark_single_request(args.iterations))
            results.append(result)
            suite.print_results(results)
        elif args.mode == "batch":
            batch_results = asyncio.run(suite.benchmark_batch_processing([1, 5, 10, 20], args.iterations))
            results.extend(batch_results)
            suite.print_results(results)
        elif args.mode == "concurrent":
            result = asyncio.run(suite.benchmark_concurrent_load(20, args.iterations))
            results.append(result)
            suite.print_results(results)
        elif args.mode == "cache":
            result = asyncio.run(suite.benchmark_cache_performance(args.iterations))
            results.append(result)
            suite.print_results(results)
        
        if args.output:
            suite.save_results(results, args.output)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

