"""
MOEA Project - Example Usage
=============================

Ejemplos de cómo usar el proyecto MOEA una vez generado.
"""

import requests
import json
from typing import Dict, Any, List

API_BASE_URL = "http://localhost:8000"


def example_nsga2_optimization():
    """Ejemplo: Optimización con NSGA-II"""
    print("=" * 60)
    print("Example 1: NSGA-II Optimization")
    print("=" * 60)
    
    # Configuración del problema
    problem = {
        "name": "ZDT1",
        "objectives": [
            {
                "name": "f1",
                "type": "minimize",
                "function": "x[0]"
            },
            {
                "name": "f2",
                "type": "minimize",
                "function": "g(x) * (1 - sqrt(f1/g(x)))"
            }
        ],
        "variables": [
            {"name": "x1", "min": 0, "max": 1},
            {"name": "x2", "min": 0, "max": 1},
            {"name": "x3", "min": 0, "max": 1},
            {"name": "x4", "min": 0, "max": 1},
            {"name": "x5", "min": 0, "max": 1}
        ],
        "constraints": []
    }
    
    # Configuración del algoritmo
    algorithm_config = {
        "algorithm": "nsga2",
        "population_size": 100,
        "generations": 50,
        "mutation_rate": 0.1,
        "crossover_rate": 0.9,
        "mutation_distribution_index": 20,
        "crossover_distribution_index": 20
    }
    
    # Ejecutar optimización
    response = requests.post(
        f"{API_BASE_URL}/api/v1/moea/optimize",
        json={
            "problem": problem,
            "algorithm": algorithm_config
        },
        timeout=300
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Optimization completed!")
        print(f"   Pareto solutions: {len(result.get('pareto_front', []))}")
        print(f"   Hypervolume: {result.get('metrics', {}).get('hypervolume', 'N/A')}")
        print(f"   IGD: {result.get('metrics', {}).get('igd', 'N/A')}")
        print(f"   GD: {result.get('metrics', {}).get('gd', 'N/A')}")
        return result
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        return None


def example_compare_algorithms():
    """Ejemplo: Comparar múltiples algoritmos"""
    print("\n" + "=" * 60)
    print("Example 2: Compare Multiple Algorithms")
    print("=" * 60)
    
    problem = {
        "name": "DTLZ2",
        "objectives": 3,
        "variables": 12
    }
    
    algorithms = ["nsga2", "nsga3", "moead", "spea2"]
    results = {}
    
    for algo in algorithms:
        print(f"\nRunning {algo.upper()}...")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/moea/optimize",
            json={
                "problem": problem,
                "algorithm": {
                    "algorithm": algo,
                    "population_size": 100,
                    "generations": 50
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            results[algo] = {
                "hypervolume": result.get('metrics', {}).get('hypervolume'),
                "igd": result.get('metrics', {}).get('igd'),
                "gd": result.get('metrics', {}).get('gd'),
                "pareto_size": len(result.get('pareto_front', []))
            }
            print(f"   ✅ {algo} completed")
        else:
            print(f"   ❌ {algo} failed: {response.status_code}")
    
    # Mostrar comparación
    print("\n" + "=" * 60)
    print("Comparison Results:")
    print("=" * 60)
    print(f"{'Algorithm':<10} {'Hypervolume':<15} {'IGD':<15} {'GD':<15} {'Pareto Size':<12}")
    print("-" * 60)
    for algo, metrics in results.items():
        print(f"{algo:<10} {metrics['hypervolume']:<15.6f} {metrics['igd']:<15.6f} "
              f"{metrics['gd']:<15.6f} {metrics['pareto_size']:<12}")
    
    return results


def example_batch_optimization():
    """Ejemplo: Optimización en batch"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Optimization")
    print("=" * 60)
    
    problems = [
        {"name": "ZDT1", "variables": 5},
        {"name": "ZDT2", "variables": 5},
        {"name": "ZDT3", "variables": 5}
    ]
    
    batch_request = {
        "problems": problems,
        "algorithm": {
            "algorithm": "nsga2",
            "population_size": 100,
            "generations": 50
        }
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/moea/batch",
        json=batch_request,
        timeout=600
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"\n✅ Batch optimization completed!")
        print(f"   Problems solved: {len(results.get('results', []))}")
        for i, result in enumerate(results.get('results', [])):
            print(f"   Problem {i+1}: {result.get('problem_name')} - "
                  f"HV: {result.get('metrics', {}).get('hypervolume', 'N/A')}")
        return results
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)
        return None


def example_get_pareto_front():
    """Ejemplo: Obtener frente de Pareto"""
    print("\n" + "=" * 60)
    print("Example 4: Get Pareto Front")
    print("=" * 60)
    
    # Primero ejecutar optimización
    optimization_result = example_nsga2_optimization()
    
    if optimization_result and 'project_id' in optimization_result:
        project_id = optimization_result['project_id']
        
        # Obtener frente de Pareto
        response = requests.get(
            f"{API_BASE_URL}/api/v1/moea/pareto/{project_id}"
        )
        
        if response.status_code == 200:
            pareto_data = response.json()
            print(f"\n✅ Pareto front retrieved!")
            print(f"   Solutions: {len(pareto_data.get('solutions', []))}")
            print(f"   Objectives: {pareto_data.get('objectives', [])}")
            return pareto_data
        else:
            print(f"\n❌ Error: {response.status_code}")
            return None
    
    return None


def example_export_results():
    """Ejemplo: Exportar resultados"""
    print("\n" + "=" * 60)
    print("Example 5: Export Results")
    print("=" * 60)
    
    # Ejecutar optimización primero
    result = example_nsga2_optimization()
    
    if result and 'project_id' in result:
        project_id = result['project_id']
        
        # Exportar a JSON
        response = requests.get(
            f"{API_BASE_URL}/api/v1/moea/export/{project_id}",
            params={"format": "json"}
        )
        
        if response.status_code == 200:
            print(f"\n✅ Results exported!")
            print(f"   Format: JSON")
            print(f"   Size: {len(response.content)} bytes")
            
            # Guardar archivo
            with open("moea_results.json", "wb") as f:
                f.write(response.content)
            print(f"   Saved to: moea_results.json")
            return True
    
    return False


def example_real_time_optimization():
    """Ejemplo: Optimización en tiempo real con WebSocket"""
    print("\n" + "=" * 60)
    print("Example 6: Real-time Optimization (WebSocket)")
    print("=" * 60)
    print("Note: This requires WebSocket client implementation")
    print("See frontend code for WebSocket usage examples")
    
    # Esta función muestra cómo se usaría WebSocket
    # La implementación real estaría en el frontend
    print("\nWebSocket endpoint: ws://localhost:8000/ws/moea/{project_id}")
    print("Subscribe to events:")
    print("  - generation.update")
    print("  - optimization.complete")
    print("  - metrics.update")


def main():
    """Ejecutar todos los ejemplos"""
    print("\n" + "=" * 60)
    print("MOEA Project - Example Usage")
    print("=" * 60)
    print("\nMake sure the backend server is running on http://localhost:8000")
    print("Start server: cd backend && uvicorn app.main:app --reload")
    print()
    
    try:
        # Verificar que el servidor esté corriendo
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Server is not running!")
            print("   Start the backend server first.")
            return
    except:
        print("❌ Cannot connect to server!")
        print("   Make sure the backend is running on http://localhost:8000")
        return
    
    print("✅ Server is running!\n")
    
    # Ejecutar ejemplos
    examples = [
        ("NSGA-II Optimization", example_nsga2_optimization),
        ("Compare Algorithms", example_compare_algorithms),
        ("Batch Optimization", example_batch_optimization),
        ("Get Pareto Front", example_get_pareto_front),
        ("Export Results", example_export_results),
        ("Real-time Optimization", example_real_time_optimization),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

