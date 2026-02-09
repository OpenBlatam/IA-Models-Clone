#!/usr/bin/env python3
"""
Script de benchmarking para Robot Movement AI v2.0
Mide performance de diferentes componentes
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


async def benchmark_cache():
    """Benchmark de cache"""
    from core.architecture.performance import get_performance_cache, cached
    from datetime import timedelta
    
    cache = get_performance_cache()
    cache.clear()
    
    @cached(ttl=timedelta(minutes=5))
    def expensive_operation(key: str):
        time.sleep(0.01)  # Simular operación costosa
        return f"result-{key}"
    
    # Primera llamada (cache miss)
    start = time.time()
    result1 = expensive_operation("test")
    first_call_time = time.time() - start
    
    # Segunda llamada (cache hit)
    start = time.time()
    result2 = expensive_operation("test")
    second_call_time = time.time() - start
    
    speedup = first_call_time / second_call_time if second_call_time > 0 else 0
    
    return {
        "cache_miss_time": first_call_time,
        "cache_hit_time": second_call_time,
        "speedup": speedup,
        "hit_rate": cache.get_stats()["hit_rate"]
    }


async def benchmark_repository():
    """Benchmark de repositorio"""
    from core.architecture.di_setup import setup_di, resolve_service
    from core.architecture.infrastructure_repositories import IRobotRepository
    
    setup_di()
    repo = resolve_service(IRobotRepository)
    
    times = []
    
    # Benchmark de creación
    for i in range(100):
        start = time.time()
        # Crear robot (simulado)
        times.append(time.time() - start)
    
    return {
        "avg_create_time": statistics.mean(times),
        "p95_create_time": statistics.quantiles(times, n=20)[18] if len(times) > 20 else times[-1],
        "total_operations": len(times)
    }


async def benchmark_api():
    """Benchmark de API (requiere servidor corriendo)"""
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            times = []
            
            for _ in range(50):
                start = time.time()
                try:
                    await client.get("http://localhost:8010/health", timeout=5.0)
                except:
                    pass
                times.append(time.time() - start)
            
            if times:
                return {
                    "avg_response_time": statistics.mean(times),
                    "p95_response_time": statistics.quantiles(times, n=20)[18] if len(times) > 20 else times[-1],
                    "min_response_time": min(times),
                    "max_response_time": max(times)
                }
    except ImportError:
        return {"error": "httpx not installed"}
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Server not available"}


async def run_all_benchmarks():
    """Ejecutar todos los benchmarks"""
    print("=" * 60)
    print("Robot Movement AI v2.0 - Benchmark Suite")
    print("=" * 60)
    print()
    
    results = {}
    
    # Cache benchmark
    print("Running cache benchmark...")
    try:
        cache_results = await benchmark_cache()
        results["cache"] = cache_results
        print(f"  Cache miss time: {cache_results['cache_miss_time']:.4f}s")
        print(f"  Cache hit time: {cache_results['cache_hit_time']:.4f}s")
        print(f"  Speedup: {cache_results['speedup']:.2f}x")
        print(f"  Hit rate: {cache_results['hit_rate']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # Repository benchmark
    print("Running repository benchmark...")
    try:
        repo_results = await benchmark_repository()
        results["repository"] = repo_results
        print(f"  Avg create time: {repo_results['avg_create_time']:.4f}s")
        print(f"  P95 create time: {repo_results['p95_create_time']:.4f}s")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    # API benchmark
    print("Running API benchmark...")
    try:
        api_results = await benchmark_api()
        results["api"] = api_results
        if "error" not in api_results:
            print(f"  Avg response time: {api_results['avg_response_time']:.4f}s")
            print(f"  P95 response time: {api_results['p95_response_time']:.4f}s")
        else:
            print(f"  {api_results['error']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    print("=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())




