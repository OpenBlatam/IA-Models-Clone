"""
MOEA Benchmark Script
====================
Script para hacer benchmarks de los algoritmos MOEA
"""
import requests
import time
import json
from typing import Dict, List, Any
from datetime import datetime

API_BASE = "http://localhost:8000"


class MOEABenchmark:
    """Benchmark de algoritmos MOEA"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.results = []
    
    def benchmark_algorithm(
        self,
        algorithm: str,
        problem: str,
        population_size: int = 100,
        generations: int = 50,
        runs: int = 3
    ) -> Dict[str, Any]:
        """Hacer benchmark de un algoritmo"""
        print(f"\n🔬 Benchmarking {algorithm.upper()} on {problem}")
        print("-" * 60)
        
        problem_config = {
            "name": problem,
            "variables": 10 if "DTLZ" in problem else 5
        }
        
        algorithm_config = {
            "algorithm": algorithm,
            "population_size": population_size,
            "generations": generations
        }
        
        run_times = []
        hypervolumes = []
        igds = []
        gds = []
        
        for run in range(1, runs + 1):
            print(f"   Run {run}/{runs}...", end=" ", flush=True)
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/moea/optimize",
                    json={
                        "problem": problem_config,
                        "algorithm": algorithm_config
                    },
                    timeout=300
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    metrics = result.get('metrics', {})
                    
                    run_times.append(elapsed)
                    hypervolumes.append(metrics.get('hypervolume', 0))
                    igds.append(metrics.get('igd', 0))
                    gds.append(metrics.get('gd', 0))
                    
                    print(f"✅ ({elapsed:.2f}s, HV: {metrics.get('hypervolume', 0):.6f})")
                else:
                    print(f"❌ Error {response.status_code}")
            except Exception as e:
                print(f"❌ Exception: {e}")
        
        # Calcular estadísticas
        if run_times:
            stats = {
                "algorithm": algorithm,
                "problem": problem,
                "population_size": population_size,
                "generations": generations,
                "runs": runs,
                "avg_time": sum(run_times) / len(run_times),
                "min_time": min(run_times),
                "max_time": max(run_times),
                "avg_hypervolume": sum(hypervolumes) / len(hypervolumes),
                "best_hypervolume": max(hypervolumes),
                "avg_igd": sum(igds) / len(igds),
                "best_igd": min(igds),
                "avg_gd": sum(gds) / len(gds),
                "best_gd": min(gds),
            }
            
            print(f"   ⏱️  Tiempo promedio: {stats['avg_time']:.2f}s")
            print(f"   📊 Hypervolume promedio: {stats['avg_hypervolume']:.6f}")
            print(f"   📊 IGD promedio: {stats['avg_igd']:.6f}")
            
            return stats
        else:
            return None
    
    def compare_algorithms(
        self,
        algorithms: List[str],
        problems: List[str],
        population_size: int = 100,
        generations: int = 50,
        runs: int = 3
    ):
        """Comparar múltiples algoritmos"""
        print("=" * 60)
        print("MOEA Algorithm Benchmark")
        print("=" * 60)
        print(f"\nAlgoritmos: {', '.join(algorithms)}")
        print(f"Problemas: {', '.join(problems)}")
        print(f"Población: {population_size}, Generaciones: {generations}")
        print(f"Runs por test: {runs}")
        print()
        
        all_results = []
        
        for problem in problems:
            for algorithm in algorithms:
                result = self.benchmark_algorithm(
                    algorithm=algorithm,
                    problem=problem,
                    population_size=population_size,
                    generations=generations,
                    runs=runs
                )
                if result:
                    all_results.append(result)
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moea_benchmark_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Resultados guardados en: {filename}")
        
        # Mostrar resumen
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: List[Dict]):
        """Imprimir resumen de resultados"""
        print("\n" + "=" * 60)
        print("📊 Benchmark Summary")
        print("=" * 60)
        
        # Agrupar por problema
        problems = {}
        for result in results:
            problem = result['problem']
            if problem not in problems:
                problems[problem] = []
            problems[problem].append(result)
        
        for problem, problem_results in problems.items():
            print(f"\n📋 {problem}:")
            print(f"{'Algorithm':<12} {'Avg Time':<12} {'Avg HV':<15} {'Best HV':<15}")
            print("-" * 60)
            
            # Ordenar por hypervolume
            problem_results.sort(key=lambda x: x['avg_hypervolume'], reverse=True)
            
            for result in problem_results:
                print(f"{result['algorithm']:<12} "
                      f"{result['avg_time']:>10.2f}s "
                      f"{result['avg_hypervolume']:>14.6f} "
                      f"{result['best_hypervolume']:>14.6f}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Benchmark")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["nsga2", "nsga3", "moead", "spea2"],
        help="Algorithms to benchmark"
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["ZDT1", "ZDT2"],
        help="Problems to test"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per test"
    )
    parser.add_argument(
        "--url",
        default=API_BASE,
        help="API base URL"
    )
    
    args = parser.parse_args()
    
    # Verificar servidor
    try:
        response = requests.get(f"{args.url}/health", timeout=2)
        if response.status_code != 200:
            print(f"❌ Servidor no disponible en {args.url}")
            return 1
    except:
        print(f"❌ No se puede conectar a {args.url}")
        print("   Asegúrate de que el backend esté corriendo")
        return 1
    
    benchmark = MOEABenchmark(args.url)
    benchmark.compare_algorithms(
        algorithms=args.algorithms,
        problems=args.problems,
        population_size=args.population,
        generations=args.generations,
        runs=args.runs
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

