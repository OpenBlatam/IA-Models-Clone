"""
Benchmark de rendimiento de la API BUL
Mide y compara el rendimiento de diferentes endpoints
"""

import requests
import time
import statistics
from typing import Dict, Any, List
from datetime import datetime
import json

BASE_URL = "http://localhost:8000"

class Benchmark:
    """Benchmark de rendimiento."""
    
    def __init__(self):
        self.results: Dict[str, List[float]] = {}
    
    def benchmark_endpoint(self, name: str, method: str, url: str, 
                         data: Dict = None, iterations: int = 10) -> Dict[str, Any]:
        """Ejecuta benchmark de un endpoint."""
        print(f"🔍 Benchmarking {name}...")
        
        times = []
        errors = 0
        
        for i in range(iterations):
            try:
                start = time.time()
                
                if method == "GET":
                    response = requests.get(url, timeout=10)
                elif method == "POST":
                    response = requests.post(url, json=data, timeout=10)
                else:
                    continue
                
                elapsed = (time.time() - start) * 1000  # ms
                times.append(elapsed)
                
                if response.status_code >= 400:
                    errors += 1
                
            except Exception as e:
                errors += 1
                print(f"  ⚠ Error en iteración {i+1}: {e}")
        
        if not times:
            return {"error": "No se pudo completar el benchmark"}
        
        result = {
            "name": name,
            "iterations": iterations,
            "errors": errors,
            "success_rate": ((iterations - errors) / iterations * 100) if iterations > 0 else 0,
            "times_ms": times,
            "avg_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "p95_ms": self.percentile(times, 95),
            "p99_ms": self.percentile(times, 99),
            "requests_per_second": 1000 / statistics.mean(times) if statistics.mean(times) > 0 else 0
        }
        
        self.results[name] = times
        
        print(f"  ✓ Completado: {result['avg_ms']:.2f}ms promedio")
        return result
    
    def percentile(self, data: List[float], percentile: int) -> float:
        """Calcula percentil."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def run_full_benchmark(self, iterations: int = 10):
        """Ejecuta benchmark completo de todos los endpoints."""
        print(f"\n{'='*70}")
        print(f"  🚀 BENCHMARK DE RENDIMIENTO - API BUL")
        print(f"{'='*70}\n")
        
        benchmarks = []
        
        # Health check
        benchmarks.append(self.benchmark_endpoint(
            "Health Check",
            "GET",
            f"{BASE_URL}/api/health",
            iterations=iterations
        ))
        
        # Stats
        benchmarks.append(self.benchmark_endpoint(
            "Stats",
            "GET",
            f"{BASE_URL}/api/stats",
            iterations=iterations
        ))
        
        # List tasks
        benchmarks.append(self.benchmark_endpoint(
            "List Tasks",
            "GET",
            f"{BASE_URL}/api/tasks?limit=10",
            iterations=iterations
        ))
        
        # List documents
        benchmarks.append(self.benchmark_endpoint(
            "List Documents",
            "GET",
            f"{BASE_URL}/api/documents?limit=10",
            iterations=iterations
        ))
        
        # Generate document (solo request, sin esperar)
        benchmarks.append(self.benchmark_endpoint(
            "Generate Document (Request)",
            "POST",
            f"{BASE_URL}/api/documents/generate",
            data={"query": "Test benchmark query", "priority": 1},
            iterations=iterations
        ))
        
        return benchmarks
    
    def print_results(self, benchmarks: List[Dict[str, Any]]):
        """Imprime resultados del benchmark."""
        print(f"\n{'='*70}")
        print(f"  RESULTADOS DEL BENCHMARK")
        print(f"{'='*70}\n")
        
        print(f"{'Endpoint':<30} {'Avg (ms)':<12} {'P95 (ms)':<12} {'RPS':<10} {'Success %':<10}")
        print("-" * 70)
        
        for bench in benchmarks:
            if "error" in bench:
                continue
            
            name = bench["name"][:28]
            avg = f"{bench['avg_ms']:.2f}"
            p95 = f"{bench['p95_ms']:.2f}"
            rps = f"{bench['requests_per_second']:.2f}"
            success = f"{bench['success_rate']:.1f}%"
            
            print(f"{name:<30} {avg:<12} {p95:<12} {rps:<10} {success:<10}")
        
        print("\n" + "="*70)
    
    def export_results(self, benchmarks: List[Dict[str, Any]], filename: str = "benchmark_results.json"):
        """Exporta resultados a JSON."""
        report = {
            "benchmark_date": datetime.now().isoformat(),
            "iterations": benchmarks[0].get("iterations", 0) if benchmarks else 0,
            "results": benchmarks
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados exportados a {filename}")
        return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark de API BUL")
    parser.add_argument("--iterations", type=int, default=10, help="Número de iteraciones por endpoint")
    
    args = parser.parse_args()
    
    benchmark = Benchmark()
    results = benchmark.run_full_benchmark(iterations=args.iterations)
    benchmark.print_results(results)
    benchmark.export_results(results)
































