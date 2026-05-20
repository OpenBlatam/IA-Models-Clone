"""
MOEA Performance Optimizer - Optimizador de performance
=======================================================
Analiza y optimiza el performance del sistema MOEA
"""
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import statistics


class MOEAPerformanceOptimizer:
    """Optimizador de performance MOEA"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.benchmarks: List[Dict] = []
    
    def benchmark_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict] = None,
        iterations: int = 10
    ) -> Dict:
        """Hacer benchmark de un endpoint"""
        print(f"⚡ Benchmarking {method} {endpoint} ({iterations} iteraciones)...")
        
        times = []
        errors = 0
        
        for i in range(iterations):
            try:
                start = time.time()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                elif method == "POST":
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        timeout=30
                    )
                else:
                    continue
                
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    times.append(elapsed)
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                continue
        
        if not times:
            return {
                "endpoint": endpoint,
                "method": method,
                "status": "failed",
                "errors": errors
            }
        
        benchmark = {
            "endpoint": endpoint,
            "method": method,
            "iterations": iterations,
            "successful": len(times),
            "errors": errors,
            "times": {
                "min": min(times),
                "max": max(times),
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "p95": self._percentile(times, 95),
                "p99": self._percentile(times, 99)
            },
            "throughput": len(times) / sum(times) if times else 0
        }
        
        self.benchmarks.append(benchmark)
        
        print(f"   ✅ Promedio: {benchmark['times']['mean']:.3f}s")
        print(f"   📊 P95: {benchmark['times']['p95']:.3f}s")
        print(f"   🔄 Throughput: {benchmark['throughput']:.2f} req/s")
        
        return benchmark
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcular percentil"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def benchmark_all_endpoints(self) -> Dict:
        """Hacer benchmark de todos los endpoints principales"""
        print("🚀 Iniciando benchmark completo del sistema...\n")
        
        endpoints = [
            ("/health", "GET", None),
            ("/api/v1/stats", "GET", None),
            ("/api/v1/queue", "GET", None),
            ("/docs", "GET", None),
        ]
        
        results = {}
        for endpoint, method, payload in endpoints:
            result = self.benchmark_endpoint(endpoint, method, payload)
            results[endpoint] = result
            time.sleep(0.5)  # Pausa entre requests
        
        return results
    
    def analyze_bottlenecks(self, benchmarks: Dict) -> List[Dict]:
        """Analizar cuellos de botella"""
        bottlenecks = []
        
        for endpoint, benchmark in benchmarks.items():
            if benchmark.get("status") == "failed":
                bottlenecks.append({
                    "endpoint": endpoint,
                    "issue": "endpoint_failed",
                    "severity": "high",
                    "message": f"Endpoint {endpoint} falló en todas las iteraciones"
                })
                continue
            
            times = benchmark.get("times", {})
            mean_time = times.get("mean", 0)
            p95_time = times.get("p95", 0)
            
            if mean_time > 1.0:
                bottlenecks.append({
                    "endpoint": endpoint,
                    "issue": "slow_response",
                    "severity": "medium",
                    "message": f"Tiempo de respuesta promedio alto: {mean_time:.3f}s"
                })
            
            if p95_time > 2.0:
                bottlenecks.append({
                    "endpoint": endpoint,
                    "issue": "high_p95",
                    "severity": "high",
                    "message": f"P95 muy alto: {p95_time:.3f}s"
                })
            
            error_rate = benchmark.get("errors", 0) / benchmark.get("iterations", 1)
            if error_rate > 0.1:
                bottlenecks.append({
                    "endpoint": endpoint,
                    "issue": "high_error_rate",
                    "severity": "high",
                    "message": f"Tasa de errores alta: {error_rate*100:.1f}%"
                })
        
        return bottlenecks
    
    def generate_recommendations(self, benchmarks: Dict, bottlenecks: List[Dict]) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        if any(b["severity"] == "high" for b in bottlenecks):
            recommendations.append("🔴 Revisar endpoints con alta severidad inmediatamente")
        
        if any(b["issue"] == "slow_response" for b in bottlenecks):
            recommendations.append("⚡ Considerar optimización de queries o cache para endpoints lentos")
        
        if any(b["issue"] == "high_error_rate" for b in bottlenecks):
            recommendations.append("🛡️ Revisar manejo de errores y validación de requests")
        
        # Análisis de throughput
        avg_throughput = statistics.mean([
            b.get("throughput", 0) for b in benchmarks.values()
            if b.get("status") != "failed"
        ]) if benchmarks else 0
        
        if avg_throughput < 10:
            recommendations.append("📈 Considerar escalado horizontal o optimización de código")
        
        return recommendations
    
    def generate_report(self, output_file: str = "moea_performance_report.json") -> str:
        """Generar reporte de performance"""
        benchmarks = self.benchmark_all_endpoints()
        bottlenecks = self.analyze_bottlenecks(benchmarks)
        recommendations = self.generate_recommendations(benchmarks, bottlenecks)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "base_url": self.base_url,
            "benchmarks": benchmarks,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "summary": {
                "total_endpoints": len(benchmarks),
                "failed_endpoints": sum(1 for b in benchmarks.values() if b.get("status") == "failed"),
                "total_bottlenecks": len(bottlenecks),
                "high_severity": sum(1 for b in bottlenecks if b["severity"] == "high")
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Performance Optimizer")
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='URL base de la API'
    )
    parser.add_argument(
        '--endpoint',
        help='Endpoint específico a benchmarkear'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Número de iteraciones'
    )
    parser.add_argument(
        '--report',
        help='Generar reporte completo'
    )
    
    args = parser.parse_args()
    
    optimizer = MOEAPerformanceOptimizer(args.url)
    
    if args.endpoint:
        optimizer.benchmark_endpoint(args.endpoint, iterations=args.iterations)
    elif args.report:
        report_file = optimizer.generate_report(args.report)
        print(f"\n✅ Reporte generado: {report_file}")
    else:
        optimizer.benchmark_all_endpoints()


if __name__ == "__main__":
    main()

