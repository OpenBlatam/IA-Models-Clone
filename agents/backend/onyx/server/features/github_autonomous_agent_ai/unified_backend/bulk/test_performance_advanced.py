"""
Pruebas de Performance Avanzadas
Incluye stress testing, memory profiling, y más
"""

import requests
import time
import statistics
import concurrent.futures
from typing import Dict, Any, List
from datetime import datetime
import json

BASE_URL = "http://localhost:8000"

class AdvancedPerformanceTest:
    """Pruebas de performance avanzadas."""
    
    def stress_test(self, endpoint: str, method: str = "GET", 
                   concurrent_requests: int = 10, total_requests: int = 100,
                   data: Dict = None) -> Dict[str, Any]:
        """Stress test de un endpoint."""
        print(f"\n🔥 Stress Test: {method} {endpoint}")
        print(f"   Requests concurrentes: {concurrent_requests}")
        print(f"   Total de requests: {total_requests}\n")
        
        times = []
        errors = 0
        status_codes = {}
        
        def make_request():
            try:
                start = time.time()
                if method == "GET":
                    response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
                elif method == "POST":
                    response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=10)
                else:
                    return None
                
                elapsed = (time.time() - start) * 1000
                status = response.status_code
                status_codes[status] = status_codes.get(status, 0) + 1
                
                if status >= 400:
                    errors += 1
                
                return elapsed
            except Exception as e:
                errors += 1
                return None
        
        # Ejecutar requests en batches concurrentes
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            for i in range(total_requests):
                future = executor.submit(make_request)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    times.append(result)
        
        total_time = time.time() - start_time
        
        if not times:
            return {"error": "No se completaron requests exitosos"}
        
        result = {
            "endpoint": endpoint,
            "method": method,
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
            "successful_requests": len(times),
            "failed_requests": errors,
            "success_rate": (len(times) / total_requests * 100) if total_requests > 0 else 0,
            "total_time_seconds": total_time,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "avg_response_time_ms": statistics.mean(times),
            "median_response_time_ms": statistics.median(times),
            "min_response_time_ms": min(times),
            "max_response_time_ms": max(times),
            "p95_ms": self.percentile(times, 95),
            "p99_ms": self.percentile(times, 99),
            "status_codes": status_codes
        }
        
        return result
    
    def percentile(self, data: List[float], percentile: int) -> float:
        """Calcula percentil."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def run_stress_tests(self):
        """Ejecuta stress tests de múltiples endpoints."""
        print("\n" + "="*70)
        print("  🔥 STRESS TESTS - API BUL")
        print("="*70)
        
        results = []
        
        # Health check stress test
        results.append(self.stress_test(
            "/api/health",
            "GET",
            concurrent_requests=20,
            total_requests=200
        ))
        
        # Stats stress test
        results.append(self.stress_test(
            "/api/stats",
            "GET",
            concurrent_requests=15,
            total_requests=150
        ))
        
        # Generate document stress test (solo request)
        results.append(self.stress_test(
            "/api/documents/generate",
            "POST",
            concurrent_requests=10,
            total_requests=50,
            data={"query": "Test stress query", "priority": 1}
        ))
        
        return results
    
    def print_results(self, results: List[Dict[str, Any]]):
        """Imprime resultados de stress tests."""
        print(f"\n{'='*70}")
        print(f"  RESULTADOS DE STRESS TESTS")
        print(f"{'='*70}\n")
        
        print(f"{'Endpoint':<30} {'RPS':<10} {'Avg (ms)':<12} {'P95 (ms)':<12} {'Success %':<12}")
        print("-" * 70)
        
        for result in results:
            if "error" in result:
                continue
            
            name = result["endpoint"][:28]
            rps = f"{result['requests_per_second']:.2f}"
            avg = f"{result['avg_response_time_ms']:.2f}"
            p95 = f"{result['p95_ms']:.2f}"
            success = f"{result['success_rate']:.1f}%"
            
            print(f"{name:<30} {rps:<10} {avg:<12} {p95:<12} {success:<12}")
        
        print("\n" + "="*70)
    
    def export_results(self, results: List[Dict[str, Any]], filename: str = "stress_test_results.json"):
        """Exporta resultados."""
        report = {
            "test_date": datetime.now().isoformat(),
            "test_type": "stress_test",
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados exportados a {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stress Tests de API")
    parser.add_argument("--concurrent", type=int, default=10, help="Requests concurrentes")
    parser.add_argument("--total", type=int, default=100, help="Total de requests")
    
    args = parser.parse_args()
    
    tester = AdvancedPerformanceTest()
    results = tester.run_stress_tests()
    tester.print_results(results)
    tester.export_results(results)
































