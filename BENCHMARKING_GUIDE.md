# 📊 Guía de Benchmarking - Blatam Academy Features

## 📋 Tabla de Contenidos

- [Setup de Benchmarking](#setup-de-benchmarking)
- [Benchmarks de Latencia](#benchmarks-de-latencia)
- [Benchmarks de Throughput](#benchmarks-de-throughput)
- [Benchmarks de Memoria](#benchmarks-de-memoria)
- [Benchmarks de Cache](#benchmarks-de-cache)
- [Benchmarks Comparativos](#benchmarks-comparativos)
- [Análisis de Resultados](#análisis-de-resultados)

## 🔧 Setup de Benchmarking

### Instalación de Herramientas

```bash
pip install pytest pytest-benchmark pytest-asyncio
pip install memory_profiler line_profiler
pip install matplotlib pandas numpy
```

### Script Base

```python
# benchmarks/base_benchmark.py
import time
import asyncio
import statistics
from typing import List, Dict, Any
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

class BenchmarkRunner:
    """Runner base para benchmarks."""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.engine = UltraAdaptiveKVCacheEngine(config)
        self.results = []
    
    async def run(self, num_iterations: int = 1000):
        """Ejecutar benchmark."""
        raise NotImplementedError
    
    def get_results(self) -> Dict[str, Any]:
        """Obtener resultados."""
        return {
            "iterations": len(self.results),
            "mean": statistics.mean(self.results),
            "median": statistics.median(self.results),
            "stdev": statistics.stdev(self.results) if len(self.results) > 1 else 0,
            "min": min(self.results),
            "max": max(self.results),
            "p50": self._percentile(50),
            "p95": self._percentile(95),
            "p99": self._percentile(99)
        }
    
    def _percentile(self, p: float) -> float:
        """Calcular percentil."""
        sorted_results = sorted(self.results)
        index = int(len(sorted_results) * p / 100)
        return sorted_results[index]
```

## ⏱️ Benchmarks de Latencia

### Latencia Individual

```python
# benchmarks/latency_benchmark.py
import asyncio
import time
from benchmarks.base_benchmark import BenchmarkRunner

class LatencyBenchmark(BenchmarkRunner):
    """Benchmark de latencia individual."""
    
    async def run(self, num_iterations: int = 1000):
        """Ejecutar benchmark de latencia."""
        self.results = []
        
        for i in range(num_iterations):
            request = {
                'text': f'Benchmark query {i}',
                'priority': 1
            }
            
            start = time.time()
            result = await self.engine.process_request(request)
            latency_ms = (time.time() - start) * 1000
            
            self.results.append(latency_ms)
        
        return self.get_results()

# Ejecutar
async def main():
    config = KVCacheConfig(max_tokens=8192)
    benchmark = LatencyBenchmark(config)
    results = await benchmark.run(num_iterations=1000)
    
    print("Latency Benchmark Results:")
    print(f"  Mean: {results['mean']:.2f}ms")
    print(f"  P50: {results['p50']:.2f}ms")
    print(f"  P95: {results['p95']:.2f}ms")
    print(f"  P99: {results['p99']:.2f}ms")

asyncio.run(main())
```

### Latencia con Cache

```python
class CacheLatencyBenchmark(BenchmarkRunner):
    """Benchmark de latencia con cache."""
    
    async def run(self, num_iterations: int = 1000):
        """Primera pasada (cache miss) y segunda (cache hit)."""
        # Primera pasada - cache miss
        miss_results = []
        for i in range(num_iterations):
            request = {'text': f'Query {i}', 'priority': 1}
            start = time.time()
            await self.engine.process_request(request)
            miss_results.append((time.time() - start) * 1000)
        
        # Segunda pasada - cache hit
        hit_results = []
        for i in range(num_iterations):
            request = {'text': f'Query {i}', 'priority': 1}
            start = time.time()
            await self.engine.process_request(request)
            hit_results.append((time.time() - start) * 1000)
        
        return {
            "cache_miss": {
                "mean": statistics.mean(miss_results),
                "p95": self._percentile_of_list(miss_results, 95)
            },
            "cache_hit": {
                "mean": statistics.mean(hit_results),
                "p95": self._percentile_of_list(hit_results, 95)
            },
            "improvement": {
                "mean": (statistics.mean(miss_results) - statistics.mean(hit_results)) / statistics.mean(miss_results) * 100,
                "p95": (self._percentile_of_list(miss_results, 95) - self._percentile_of_list(hit_results, 95)) / self._percentile_of_list(miss_results, 95) * 100
            }
        }
```

## 🚀 Benchmarks de Throughput

### Throughput Concurrente

```python
class ThroughputBenchmark(BenchmarkRunner):
    """Benchmark de throughput."""
    
    async def run(self, num_requests: int = 10000, concurrency: int = 100):
        """Ejecutar benchmark de throughput."""
        requests = [
            {'text': f'Query {i}', 'priority': 1}
            for i in range(num_requests)
        ]
        
        start = time.time()
        
        # Procesar en batches concurrentes
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(request):
            async with semaphore:
                return await self.engine.process_request(request)
        
        tasks = [process_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start
        throughput = num_requests / duration
        
        return {
            "total_requests": num_requests,
            "concurrency": concurrency,
            "duration_seconds": duration,
            "throughput_req_per_sec": throughput,
            "avg_latency_ms": duration / num_requests * 1000
        }

# Ejecutar con diferentes niveles de concurrencia
async def benchmark_concurrency():
    config = KVCacheConfig(max_tokens=16384)
    
    for concurrency in [10, 50, 100, 200, 500]:
        benchmark = ThroughputBenchmark(config)
        results = await benchmark.run(
            num_requests=10000,
            concurrency=concurrency
        )
        print(f"Concurrency {concurrency}: {results['throughput_req_per_sec']:.2f} req/s")
```

### Throughput Batch

```python
class BatchThroughputBenchmark(BenchmarkRunner):
    """Benchmark de throughput en batch."""
    
    async def run(self, num_requests: int = 10000, batch_size: int = 50):
        """Ejecutar benchmark batch."""
        requests = [
            {'text': f'Query {i}', 'priority': 1}
            for i in range(num_requests)
        ]
        
        start = time.time()
        
        # Procesar en batches
        num_batches = (num_requests + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch = requests[i * batch_size:(i + 1) * batch_size]
            await self.engine.process_batch_optimized(batch, batch_size=batch_size)
        
        duration = time.time() - start
        throughput = num_requests / duration
        
        return {
            "total_requests": num_requests,
            "batch_size": batch_size,
            "duration_seconds": duration,
            "throughput_req_per_sec": throughput
        }
```

## 💾 Benchmarks de Memoria

### Memory Usage

```python
import psutil
import os

class MemoryBenchmark:
    """Benchmark de uso de memoria."""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.engine = UltraAdaptiveKVCacheEngine(config)
        self.process = psutil.Process(os.getpid())
    
    async def run(self, num_requests: int = 1000):
        """Ejecutar benchmark de memoria."""
        # Memoria inicial
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Procesar requests
        for i in range(num_requests):
            await self.engine.process_request({
                'text': f'Query {i}',
                'priority': 1
            })
        
        # Memoria final
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Memoria peak
        peak_memory = self.process.memory_info().peak_wss / 1024 / 1024  # MB
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": final_memory - initial_memory,
            "memory_per_request_kb": (final_memory - initial_memory) * 1024 / num_requests
        }
```

### GPU Memory

```python
import torch

class GPUMemoryBenchmark:
    """Benchmark de memoria GPU."""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.engine = UltraAdaptiveKVCacheEngine(config)
    
    async def run(self, num_requests: int = 1000):
        """Ejecutar benchmark de memoria GPU."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # Memoria inicial
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Procesar requests
        for i in range(num_requests):
            await self.engine.process_request({
                'text': f'Query {i}',
                'priority': 1
            })
        
        # Memoria final
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": final_memory - initial_memory,
            "memory_per_request_kb": (final_memory - initial_memory) * 1024 / num_requests
        }
```

## 📊 Benchmarks de Cache

### Cache Hit Rate

```python
class CacheHitRateBenchmark:
    """Benchmark de cache hit rate."""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.engine = UltraAdaptiveKVCacheEngine(config)
    
    async def run(self, num_requests: int = 10000, repeat_ratio: float = 0.5):
        """Ejecutar benchmark de hit rate."""
        # Generar requests con repetición
        unique_requests = [
            {'text': f'Query {i}', 'priority': 1}
            for i in range(int(num_requests * (1 - repeat_ratio)))
        ]
        
        repeated_requests = unique_requests[:int(num_requests * repeat_ratio)]
        all_requests = unique_requests + repeated_requests
        
        # Procesar todos los requests
        for request in all_requests:
            await self.engine.process_request(request)
        
        # Obtener estadísticas
        stats = self.engine.get_stats()
        
        return {
            "total_requests": num_requests,
            "unique_requests": len(unique_requests),
            "repeated_requests": len(repeated_requests),
            "cache_hits": stats['cache_hits'],
            "cache_misses": stats['cache_misses'],
            "hit_rate": stats['hit_rate'],
            "expected_hit_rate": repeat_ratio
        }
```

### Estrategias Comparativas

```python
from bulk.core.ultra_adaptive_kv_cache_engine import CacheStrategy

async def benchmark_strategies():
    """Comparar diferentes estrategias."""
    strategies = [
        CacheStrategy.LRU,
        CacheStrategy.LFU,
        CacheStrategy.ADAPTIVE
    ]
    
    results = {}
    
    for strategy in strategies:
        config = KVCacheConfig(
            max_tokens=8192,
            cache_strategy=strategy
        )
        
        benchmark = CacheHitRateBenchmark(config)
        result = await benchmark.run(
            num_requests=10000,
            repeat_ratio=0.6
        )
        
        results[strategy.value] = result
    
    return results
```

## 🔄 Benchmarks Comparativos

### Comparación de Configuraciones

```python
async def benchmark_configurations():
    """Comparar diferentes configuraciones."""
    configurations = [
        {
            "name": "Baseline",
            "config": KVCacheConfig(max_tokens=4096)
        },
        {
            "name": "Compressed",
            "config": KVCacheConfig(
                max_tokens=4096,
                use_compression=True,
                compression_ratio=0.3
            )
        },
        {
            "name": "Quantized",
            "config": KVCacheConfig(
                max_tokens=4096,
                use_quantization=True,
                quantization_bits=8
            )
        },
        {
            "name": "Large",
            "config": KVCacheConfig(max_tokens=16384)
        }
    ]
    
    results = {}
    
    for config_info in configurations:
        benchmark = LatencyBenchmark(config_info["config"])
        latency_result = await benchmark.run(num_iterations=1000)
        
        memory_benchmark = MemoryBenchmark(config_info["config"])
        memory_result = await memory_benchmark.run(num_requests=1000)
        
        results[config_info["name"]] = {
            "latency": latency_result,
            "memory": memory_result
        }
    
    return results
```

## 📈 Análisis de Resultados

### Visualización

```python
import matplotlib.pyplot as plt
import pandas as pd

def visualize_results(results: Dict[str, Any]):
    """Visualizar resultados de benchmark."""
    # Latencia
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # P50 Latency
    configs = list(results.keys())
    p50_values = [results[c]["latency"]["p50"] for c in configs]
    axes[0, 0].bar(configs, p50_values)
    axes[0, 0].set_title("P50 Latency")
    axes[0, 0].set_ylabel("ms")
    
    # P95 Latency
    p95_values = [results[c]["latency"]["p95"] for c in configs]
    axes[0, 1].bar(configs, p95_values)
    axes[0, 1].set_title("P95 Latency")
    axes[0, 1].set_ylabel("ms")
    
    # Memory Usage
    memory_values = [results[c]["memory"]["final_memory_mb"] for c in configs]
    axes[1, 0].bar(configs, memory_values)
    axes[1, 0].set_title("Memory Usage")
    axes[1, 0].set_ylabel("MB")
    
    # Throughput
    throughput_values = [results[c].get("throughput", 0) for c in configs]
    axes[1, 1].bar(configs, throughput_values)
    axes[1, 1].set_title("Throughput")
    axes[1, 1].set_ylabel("req/s")
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()

def export_to_csv(results: Dict[str, Any], filename: str = "benchmark_results.csv"):
    """Exportar resultados a CSV."""
    data = []
    for config_name, config_results in results.items():
        row = {"configuration": config_name}
        row.update({
            f"latency_{k}": v for k, v in config_results["latency"].items()
        })
        row.update({
            f"memory_{k}": v for k, v in config_results["memory"].items()
        })
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename}")
```

### Reporte HTML

```python
def generate_html_report(results: Dict[str, Any], filename: str = "benchmark_report.html"):
    """Generar reporte HTML."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Benchmark Report</h1>
        <h2>Configuration Comparison</h2>
        <table>
            <tr>
                <th>Configuration</th>
                <th>P50 Latency (ms)</th>
                <th>P95 Latency (ms)</th>
                <th>Memory (MB)</th>
                <th>Throughput (req/s)</th>
            </tr>
    """
    
    for config_name, config_results in results.items():
        latency = config_results["latency"]
        memory = config_results["memory"]
        throughput = config_results.get("throughput", 0)
        
        html += f"""
            <tr>
                <td>{config_name}</td>
                <td>{latency['p50']:.2f}</td>
                <td>{latency['p95']:.2f}</td>
                <td>{memory['final_memory_mb']:.2f}</td>
                <td>{throughput:.2f}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {filename}")
```

## 🎯 Script Completo de Benchmarking

```python
# benchmarks/run_all_benchmarks.py
import asyncio
import json
from datetime import datetime
from benchmarks.latency_benchmark import LatencyBenchmark
from benchmarks.throughput_benchmark import ThroughputBenchmark
from benchmarks.memory_benchmark import MemoryBenchmark
from benchmarks.cache_hit_rate_benchmark import CacheHitRateBenchmark

async def run_all_benchmarks():
    """Ejecutar todos los benchmarks."""
    config = KVCacheConfig(max_tokens=8192)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_tokens": config.max_tokens,
            "strategy": config.cache_strategy.value
        },
        "benchmarks": {}
    }
    
    # Latency
    print("Running latency benchmark...")
    latency_benchmark = LatencyBenchmark(config)
    results["benchmarks"]["latency"] = await latency_benchmark.run(1000)
    
    # Throughput
    print("Running throughput benchmark...")
    throughput_benchmark = ThroughputBenchmark(config)
    results["benchmarks"]["throughput"] = await throughput_benchmark.run(10000, 100)
    
    # Memory
    print("Running memory benchmark...")
    memory_benchmark = MemoryBenchmark(config)
    results["benchmarks"]["memory"] = await memory_benchmark.run(1000)
    
    # Cache Hit Rate
    print("Running cache hit rate benchmark...")
    hit_rate_benchmark = CacheHitRateBenchmark(config)
    results["benchmarks"]["cache_hit_rate"] = await hit_rate_benchmark.run(10000, 0.6)
    
    # Guardar resultados
    with open(f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = asyncio.run(run_all_benchmarks())
    print("\nBenchmark Results:")
    print(json.dumps(results, indent=2))
```

---

**Más información:**
- [Performance Tuning](PERFORMANCE_TUNING.md)
- [Production Ready](bulk/PRODUCTION_READY.md)
- [Advanced Troubleshooting](bulk/ADVANCED_TROUBLESHOOTING.md)

