"""
Routing Testing Optimizations
==============================

Optimizaciones para testing y validación.
Incluye: Test fixtures, Mock data generation, Performance testing, etc.
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable
import threading

logger = logging.getLogger(__name__)

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestDataGenerator:
    """Generador de datos de prueba."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Inicializar generador de datos.
        
        Args:
            seed: Semilla para reproducibilidad
        """
        if seed is not None:
            random.seed(seed)
    
    def generate_nodes(self, count: int, bounds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Generar nodos de prueba.
        
        Args:
            count: Número de nodos
            bounds: Límites espaciales {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}
        
        Returns:
            Lista de nodos
        """
        if bounds is None:
            bounds = {'x_min': 0, 'x_max': 100, 'y_min': 0, 'y_max': 100, 'z_min': 0, 'z_max': 10}
        
        nodes = []
        for i in range(count):
            node = {
                'name': f'node_{i}',
                'position': {
                    'x': random.uniform(bounds['x_min'], bounds['x_max']),
                    'y': random.uniform(bounds['y_min'], bounds['y_max']),
                    'z': random.uniform(bounds['z_min'], bounds['z_max'])
                },
                'capacity': random.uniform(1.0, 10.0),
                'cost': random.uniform(0.1, 5.0)
            }
            nodes.append(node)
        
        return nodes
    
    def generate_edges(self, node_ids: List[str], density: float = 0.3) -> List[Dict[str, Any]]:
        """
        Generar aristas de prueba.
        
        Args:
            node_ids: Lista de IDs de nodos
            density: Densidad de conexiones (0.0-1.0)
        
        Returns:
            Lista de aristas
        """
        edges = []
        num_edges = int(len(node_ids) * (len(node_ids) - 1) * density)
        
        for _ in range(num_edges):
            from_node = random.choice(node_ids)
            to_node = random.choice(node_ids)
            
            if from_node != to_node:
                edge = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'distance': random.uniform(1.0, 100.0),
                    'time': random.uniform(0.1, 10.0),
                    'cost': random.uniform(0.1, 5.0),
                    'capacity': random.uniform(1.0, 10.0)
                }
                edges.append(edge)
        
        return edges


class PerformanceTester:
    """Tester de rendimiento."""
    
    def __init__(self):
        """Inicializar tester de rendimiento."""
        self.results: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def benchmark(self, name: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Ejecutar benchmark.
        
        Args:
            name: Nombre del benchmark
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Resultados del benchmark
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        benchmark_result = {
            'name': name,
            'duration': duration,
            'memory_delta': memory_delta,
            'success': success,
            'error': error,
            'timestamp': time.time()
        }
        
        with self.lock:
            self.results.append(benchmark_result)
        
        return benchmark_result
    
    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria (simplificado)."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # MB
        except:
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de benchmarks."""
        with self.lock:
            if not self.results:
                return {}
            
            successful = [r for r in self.results if r['success']]
            failed = [r for r in self.results if not r['success']]
            
            durations = [r['duration'] for r in successful]
            
            return {
                'total_benchmarks': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'avg_duration': sum(durations) / len(durations) if durations else 0.0,
                'min_duration': min(durations) if durations else 0.0,
                'max_duration': max(durations) if durations else 0.0,
                'total_duration': sum(durations)
            }


class TestOptimizer:
    """Optimizador completo de testing."""
    
    def __init__(self):
        """Inicializar optimizador de testing."""
        self.data_generator = TestDataGenerator()
        self.performance_tester = PerformanceTester()
    
    def generate_test_graph(self, num_nodes: int = 10, density: float = 0.3) -> Dict[str, Any]:
        """
        Generar grafo de prueba.
        
        Args:
            num_nodes: Número de nodos
            density: Densidad de conexiones
        
        Returns:
            Diccionario con nodes y edges
        """
        nodes_data = self.data_generator.generate_nodes(num_nodes)
        node_ids = [f"node_{i}" for i in range(num_nodes)]
        edges_data = self.data_generator.generate_edges(node_ids, density)
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'node_ids': node_ids
        }
    
    def benchmark_route_finding(self, router, start: str, end: str, iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark de búsqueda de rutas.
        
        Args:
            router: Router a probar
            start: Nodo de inicio
            end: Nodo de destino
            iterations: Número de iteraciones
        
        Returns:
            Resultados del benchmark
        """
        def find_route():
            return router.find_route(start, end)
        
        results = []
        for i in range(iterations):
            result = self.performance_tester.benchmark(
                f"route_finding_{i}",
                find_route
            )
            results.append(result)
        
        return self.performance_tester.get_statistics()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'performance_stats': self.performance_tester.get_statistics()
        }

