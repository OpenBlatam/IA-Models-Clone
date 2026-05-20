from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import statistics
import psutil
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from test_simple import SimplifiedBlogAnalyzer
from typing import Any, List, Dict, Optional
import logging
"""
⚡ ADVANCED PERFORMANCE TESTS - Blog Model
==========================================

Tests de performance avanzado, benchmarking y optimización
para el sistema de análisis de contenido de blog.
"""



class PerformanceBenchmark:
    """Clase para realizar benchmarks de performance."""
    
    def __init__(self) -> Any:
        self.results = []
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{psutil.Process().exe}",
            'timestamp': time.time()
        }
    
    def add_result(self, test_name: str, metrics: Dict[str, Any]):
        """Añadir resultado de benchmark."""
        self.results.append({
            'test_name': test_name,
            'timestamp': time.time(),
            **metrics
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de todos los benchmarks."""
        if not self.results:
            return {'error': 'No benchmark results'}
        
        # Calcular estadísticas agregadas
        processing_times = [r.get('processing_time_ms', 0) for r in self.results]
        throughputs = [r.get('throughput_ops_per_second', 0) for r in self.results]
        
        return {
            'system_info': self.system_info,
            'total_tests': len(self.results),
            'processing_time_stats': {
                'mean': statistics.mean(processing_times),
                'median': statistics.median(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'stdev': statistics.stdev(processing_times) if len(processing_times) > 1 else 0
            },
            'throughput_stats': {
                'mean': statistics.mean(throughputs),
                'median': statistics.median(throughputs),
                'min': min(throughputs),
                'max': max(throughputs)
            },
            'results': self.results
        }


class TestAdvancedPerformance:
    """Tests de performance avanzado."""
    
    def __init__(self) -> Any:
        self.benchmark = PerformanceBenchmark()
    
    def test_latency_consistency(self) -> Any:
        """Test consistencia de latencia en múltiples ejecuciones."""
        print("⚡ Testing latency consistency...")
        
        analyzer = SimplifiedBlogAnalyzer()
        test_content = "Este es un artículo excelente sobre inteligencia artificial y marketing digital."
        
        latencies = []
        
        # Ejecutar 100 análisis individuales
        for i in range(100):
            start_time = time.perf_counter()
            sentiment = analyzer.analyze_sentiment(test_content)
            processing_time = (time.perf_counter() - start_time) * 1000
            latencies.append(processing_time)
        
        # Estadísticas de latencia
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        stdev_latency = statistics.stdev(latencies)
        
        # Verificar consistencia
        assert mean_latency < 2.0, f"Mean latency too high: {mean_latency:.3f}ms"
        assert p95_latency < 5.0, f"P95 latency too high: {p95_latency:.3f}ms"
        assert p99_latency < 10.0, f"P99 latency too high: {p99_latency:.3f}ms"
        assert stdev_latency < 2.0, f"Latency too inconsistent: {stdev_latency:.3f}ms"
        
        self.benchmark.add_result('latency_consistency', {
            'iterations': 100,
            'mean_latency_ms': mean_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'stdev_latency_ms': stdev_latency
        })
        
        print(f"✅ Latency consistency test passed!")
        print(f"   Mean: {mean_latency:.3f}ms, P95: {p95_latency:.3f}ms, P99: {p99_latency:.3f}ms")
    
    def test_throughput_scaling(self) -> Any:
        """Test escalabilidad del throughput con diferentes tamaños de lote."""
        print("📈 Testing throughput scaling...")
        
        analyzer = SimplifiedBlogAnalyzer()
        batch_sizes = [1, 5, 10, 25, 50, 100, 250, 500]
        
        scaling_results = []
        
        for batch_size in batch_sizes:
            # Crear lote de prueba
            test_batch = [
                f"Artículo excelente número {i} sobre tecnologías emergentes."
                for i in range(batch_size)
            ]
            
            start_time = time.perf_counter()
            
            # Procesar lote
            for content in test_batch:
                sentiment = analyzer.analyze_sentiment(content)
                quality = analyzer.analyze_quality(content)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            throughput = batch_size / (processing_time / 1000)
            
            scaling_results.append({
                'batch_size': batch_size,
                'processing_time_ms': processing_time,
                'throughput_ops_per_second': throughput,
                'latency_per_item_ms': processing_time / batch_size
            })
        
        # Verificar escalabilidad
        max_throughput = max(r['throughput_ops_per_second'] for r in scaling_results)
        assert max_throughput > 1000, f"Max throughput too low: {max_throughput:.0f} ops/s"
        
        # Verificar que el throughput mejora con lotes más grandes
        small_batch_throughput = scaling_results[0]['throughput_ops_per_second']  # batch_size=1
        large_batch_throughput = scaling_results[-1]['throughput_ops_per_second']  # batch_size=500
        
        assert large_batch_throughput > small_batch_throughput, "Throughput should improve with larger batches"
        
        self.benchmark.add_result('throughput_scaling', {
            'batch_sizes_tested': batch_sizes,
            'max_throughput_ops_per_second': max_throughput,
            'scaling_results': scaling_results
        })
        
        print(f"✅ Throughput scaling test passed!")
        print(f"   Max throughput: {max_throughput:.0f} ops/s")
        print(f"   Scaling factor: {large_batch_throughput/small_batch_throughput:.1f}x")
    
    def test_memory_efficiency(self) -> Any:
        """Test eficiencia de memoria con diferentes cargas de trabajo."""
        print("💾 Testing memory efficiency...")
        
        process = psutil.Process(os.getpid())
        memory_baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        analyzer = SimplifiedBlogAnalyzer()
        
        # Test diferentes tipos de carga
        memory_tests = [
            ("small_batch", 50, "Contenido corto."),
            ("medium_batch", 200, "Contenido de tamaño medio con más palabras y estructura."),
            ("large_batch", 500, "Contenido extenso con múltiples párrafos, estructura compleja y gran cantidad de palabras para análisis detallado."),
            ("huge_batch", 1000, "Artículo muy largo con información detallada, múltiples secciones, ejemplos prácticos, casos de uso específicos y contenido comprehensivo.")
        ]
        
        memory_results = []
        
        for test_name, batch_size, content_template in memory_tests:
            # Medir memoria antes
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Crear lote de prueba
            test_batch = [content_template for _ in range(batch_size)]
            
            start_time = time.perf_counter()
            
            # Procesar lote
            for content in test_batch:
                sentiment = analyzer.analyze_sentiment(content)
                quality = analyzer.analyze_quality(content)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Medir memoria después
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            memory_results.append({
                'test_name': test_name,
                'batch_size': batch_size,
                'content_length': len(content_template),
                'memory_used_mb': memory_used,
                'memory_per_item_kb': (memory_used * 1024) / batch_size,
                'processing_time_ms': processing_time
            })
        
        # Verificar eficiencia de memoria
        max_memory_per_item = max(r['memory_per_item_kb'] for r in memory_results)
        assert max_memory_per_item < 100, f"Memory usage per item too high: {max_memory_per_item:.1f}KB"
        
        total_memory_used = sum(r['memory_used_mb'] for r in memory_results)
        assert total_memory_used < 200, f"Total memory usage too high: {total_memory_used:.1f}MB"
        
        self.benchmark.add_result('memory_efficiency', {
            'memory_tests': memory_results,
            'max_memory_per_item_kb': max_memory_per_item,
            'total_memory_used_mb': total_memory_used
        })
        
        print(f"✅ Memory efficiency test passed!")
        print(f"   Max memory per item: {max_memory_per_item:.1f}KB")
        print(f"   Total memory used: {total_memory_used:.1f}MB")
    
    def test_cache_performance(self) -> Any:
        """Test performance del sistema de cache."""
        print("🗂️ Testing cache performance...")
        
        analyzer = SimplifiedBlogAnalyzer()
        
        # Test contenido para cache
        test_contents = [
            "Contenido único número 1 para testing de cache.",
            "Contenido único número 2 para testing de cache.",
            "Contenido único número 3 para testing de cache.",
        ]
        
        # Primera pasada (cache miss)
        first_pass_times = []
        for content in test_contents:
            start_time = time.perf_counter()
            sentiment = analyzer.analyze_sentiment(content)
            processing_time = (time.perf_counter() - start_time) * 1000
            first_pass_times.append(processing_time)
        
        # Segunda pasada (cache hit esperado para cache más sofisticado)
        second_pass_times = []
        for content in test_contents:
            start_time = time.perf_counter()
            sentiment = analyzer.analyze_sentiment(content)
            processing_time = (time.perf_counter() - start_time) * 1000
            second_pass_times.append(processing_time)
        
        # Análisis de performance del cache
        avg_first_pass = statistics.mean(first_pass_times)
        avg_second_pass = statistics.mean(second_pass_times)
        
        cache_speedup = avg_first_pass / avg_second_pass if avg_second_pass > 0 else 1.0
        
        self.benchmark.add_result('cache_performance', {
            'first_pass_avg_ms': avg_first_pass,
            'second_pass_avg_ms': avg_second_pass,
            'cache_speedup_factor': cache_speedup,
            'cache_efficiency': (cache_speedup - 1) * 100  # % improvement
        })
        
        print(f"✅ Cache performance test passed!")
        print(f"   First pass avg: {avg_first_pass:.3f}ms")
        print(f"   Second pass avg: {avg_second_pass:.3f}ms")
        print(f"   Cache speedup: {cache_speedup:.1f}x")
    
    async def test_concurrent_performance(self) -> Any:
        """Test performance con procesamiento concurrente."""
        print("🔄 Testing concurrent performance...")
        
        analyzer = SimplifiedBlogAnalyzer()
        
        # Test contenido
        test_content = "Artículo excelente sobre performance y concurrencia en sistemas de IA."
        
        # Test secuencial
        sequential_start = time.perf_counter()
        for _ in range(100):
            result = await analyzer.analyze_blog_content(test_content)
        sequential_time = (time.perf_counter() - sequential_start) * 1000
        
        # Test concurrente
        concurrent_start = time.perf_counter()
        
        tasks = []
        for _ in range(100):
            task = analyzer.analyze_blog_content(test_content)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        concurrent_time = (time.perf_counter() - concurrent_start) * 1000
        
        # Análisis de concurrencia
        concurrency_speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1.0
        
        # Verificar que todos los resultados son consistentes
        first_result = results[0]
        all_consistent = all(
            r.sentiment_score == first_result.sentiment_score and
            r.quality_score == first_result.quality_score
            for r in results
        )
        
        assert all_consistent, "Concurrent results should be consistent"
        
        self.benchmark.add_result('concurrent_performance', {
            'sequential_time_ms': sequential_time,
            'concurrent_time_ms': concurrent_time,
            'concurrency_speedup': concurrency_speedup,
            'operations': 100,
            'results_consistent': all_consistent
        })
        
        print(f"✅ Concurrent performance test passed!")
        print(f"   Sequential: {sequential_time:.2f}ms")
        print(f"   Concurrent: {concurrent_time:.2f}ms")
        print(f"   Speedup: {concurrency_speedup:.1f}x")
    
    def test_cpu_utilization(self) -> Any:
        """Test utilización de CPU bajo diferentes cargas."""
        print("💻 Testing CPU utilization...")
        
        analyzer = SimplifiedBlogAnalyzer()
        
        # Monitorear CPU durante procesamiento intensivo
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        # Carga intensiva
        large_batch = [
            f"Artículo número {i} con contenido extenso para análisis intensivo de CPU."
            for i in range(1000)
        ]
        
        start_time = time.perf_counter()
        
        for content in large_batch:
            sentiment = analyzer.analyze_sentiment(content)
            quality = analyzer.analyze_quality(content)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        cpu_utilization_increase = cpu_percent_after - cpu_percent_before
        throughput = len(large_batch) / (processing_time / 1000)
        
        self.benchmark.add_result('cpu_utilization', {
            'batch_size': len(large_batch),
            'processing_time_ms': processing_time,
            'throughput_ops_per_second': throughput,
            'cpu_before_percent': cpu_percent_before,
            'cpu_after_percent': cpu_percent_after,
            'cpu_utilization_increase': cpu_utilization_increase
        })
        
        print(f"✅ CPU utilization test passed!")
        print(f"   Processed {len(large_batch)} items in {processing_time:.0f}ms")
        print(f"   Throughput: {throughput:.0f} ops/s")
        print(f"   CPU utilization increase: {cpu_utilization_increase:.1f}%")


def run_comprehensive_benchmark():
    """Ejecutar benchmark comprehensivo completo."""
    print("🏆 COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    test_suite = TestAdvancedPerformance()
    
    # Ejecutar todos los tests de performance
    test_suite.test_latency_consistency()
    test_suite.test_throughput_scaling()
    test_suite.test_memory_efficiency()
    test_suite.test_cache_performance()
    test_suite.test_cpu_utilization()
    
    return test_suite.benchmark


async def run_async_performance_tests():
    """Ejecutar tests de performance async."""
    print("\n🔄 ASYNC PERFORMANCE TESTS")
    print("=" * 30)
    
    test_suite = TestAdvancedPerformance()
    await test_suite.test_concurrent_performance()
    
    return test_suite.benchmark


def generate_performance_report(benchmark: PerformanceBenchmark):
    """Generar reporte detallado de performance."""
    summary = benchmark.get_summary()
    
    print("\n📊 PERFORMANCE REPORT")
    print("=" * 25)
    
    print(f"\n🖥️  System Info:")
    print(f"   CPU cores: {summary['system_info']['cpu_count']}")
    print(f"   Memory: {summary['system_info']['memory_total_gb']:.1f}GB")
    
    print(f"\n⚡ Processing Time Statistics:")
    stats = summary['processing_time_stats']
    print(f"   Mean: {stats['mean']:.3f}ms")
    print(f"   Median: {stats['median']:.3f}ms")
    print(f"   Min: {stats['min']:.3f}ms")
    print(f"   Max: {stats['max']:.3f}ms")
    print(f"   Std Dev: {stats['stdev']:.3f}ms")
    
    print(f"\n📈 Throughput Statistics:")
    throughput = summary['throughput_stats']
    print(f"   Mean: {throughput['mean']:.0f} ops/s")
    print(f"   Median: {throughput['median']:.0f} ops/s")
    print(f"   Min: {throughput['min']:.0f} ops/s")
    print(f"   Max: {throughput['max']:.0f} ops/s")
    
    print(f"\n✅ Total tests completed: {summary['total_tests']}")


async def main():
    """Ejecutar suite completo de tests de performance avanzado."""
    # Tests síncronos
    benchmark = run_comprehensive_benchmark()
    
    # Tests asíncronos
    async_benchmark = await run_async_performance_tests()
    
    # Combinar resultados
    benchmark.results.extend(async_benchmark.results)
    
    # Generar reporte
    generate_performance_report(benchmark)
    
    print("\n🎉 ALL ADVANCED PERFORMANCE TESTS COMPLETED!")
    print("🚀 System performance validated successfully!")


match __name__:
    case "__main__":
    asyncio.run(main()) 