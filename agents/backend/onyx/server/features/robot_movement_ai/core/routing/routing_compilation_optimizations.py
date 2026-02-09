"""
Routing Compilation Optimizations
==================================

Optimizaciones de compilación para máximo rendimiento.
Incluye: Numba JIT, Cython, Caching compilado, etc.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import functools
import time

logger = logging.getLogger(__name__)

try:
    import numba
    from numba import jit, types, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None
    prange = None
    logger.warning("Numba not available, JIT compilation disabled")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class NumbaJITOptimizer:
    """Optimizador usando Numba JIT para funciones críticas."""
    
    def __init__(self, nopython: bool = True, parallel: bool = True):
        """
        Inicializar optimizador Numba.
        
        Args:
            nopython: Modo nopython (más rápido, sin Python)
            parallel: Habilitar paralelización automática
        """
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba not available")
        
        self.nopython = nopython
        self.parallel = parallel
        self.compiled_functions: Dict[str, Callable] = {}
    
    def compile_function(self, func: Callable, name: str = None) -> Callable:
        """
        Compilar función con Numba JIT.
        
        Args:
            func: Función a compilar
            name: Nombre de la función
        
        Returns:
            Función compilada
        """
        if not NUMBA_AVAILABLE:
            return func
        
        name = name or func.__name__
        
        if name in self.compiled_functions:
            return self.compiled_functions[name]
        
        try:
            compiled = jit(nopython=self.nopython, parallel=self.parallel)(func)
            self.compiled_functions[name] = compiled
            logger.info(f"Function {name} compiled with Numba JIT")
            return compiled
        except Exception as e:
            logger.warning(f"Failed to compile {name} with Numba: {e}")
            return func


# Funciones críticas optimizadas con Numba
if NUMBA_AVAILABLE and NUMPY_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def fast_distance_calculation(positions: np.ndarray) -> np.ndarray:
        """
        Calcular distancias entre posiciones (vectorizado y compilado).
        
        Args:
            positions: Array de posiciones (N, 3)
        
        Returns:
            Matriz de distancias (N, N)
        """
        n = positions.shape[0]
        distances = np.zeros((n, n))
        
        for i in prange(n):
            for j in prange(n):
                if i != j:
                    diff = positions[i] - positions[j]
                    distances[i, j] = np.sqrt(np.sum(diff ** 2))
        
        return distances
    
    @jit(nopython=True)
    def fast_path_cost(path: np.ndarray, cost_matrix: np.ndarray) -> float:
        """
        Calcular costo de un path rápidamente.
        
        Args:
            path: Array de índices de nodos
            cost_matrix: Matriz de costos
        
        Returns:
            Costo total
        """
        total_cost = 0.0
        for i in range(len(path) - 1):
            from_idx = int(path[i])
            to_idx = int(path[i + 1])
            total_cost += cost_matrix[from_idx, to_idx]
        return total_cost
    
    @jit(nopython=True, parallel=True)
    def fast_batch_distance(positions1: np.ndarray, positions2: np.ndarray) -> np.ndarray:
        """
        Calcular distancias entre dos sets de posiciones en batch.
        
        Args:
            positions1: Primer set (N, 3)
            positions2: Segundo set (M, 3)
        
        Returns:
            Matriz de distancias (N, M)
        """
        n = positions1.shape[0]
        m = positions2.shape[0]
        distances = np.zeros((n, m))
        
        for i in prange(n):
            for j in prange(m):
                diff = positions1[i] - positions2[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))
        
        return distances
else:
    # Fallback sin Numba
    def fast_distance_calculation(positions):
        """Fallback sin Numba."""
        n = len(positions)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = positions[i] - positions[j]
                    distances[i, j] = np.sqrt(np.sum(diff ** 2))
        return distances
    
    def fast_path_cost(path, cost_matrix):
        """Fallback sin Numba."""
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += cost_matrix[path[i], path[i + 1]]
        return total_cost
    
    def fast_batch_distance(positions1, positions2):
        """Fallback sin Numba."""
        n = len(positions1)
        m = len(positions2)
        distances = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                diff = positions1[i] - positions2[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))
        return distances


class CompiledRouteCalculator:
    """Calculadora de rutas con funciones compiladas."""
    
    def __init__(self):
        """Inicializar calculadora compilada."""
        self.jit_optimizer = NumbaJITOptimizer() if NUMBA_AVAILABLE else None
    
    def calculate_distances_matrix(self, nodes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calcular matriz de distancias entre nodos.
        
        Args:
            nodes: Lista de nodos con posiciones
        
        Returns:
            Matriz de distancias
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy not available")
        
        positions = np.array([
            [node.get('position', {}).get('x', 0.0),
             node.get('position', {}).get('y', 0.0),
             node.get('position', {}).get('z', 0.0)]
            for node in nodes
        ])
        
        return fast_distance_calculation(positions)
    
    def calculate_path_cost(self, path: List[int], cost_matrix: np.ndarray) -> float:
        """
        Calcular costo de un path.
        
        Args:
            path: Lista de índices de nodos
            cost_matrix: Matriz de costos
        
        Returns:
            Costo total
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy not available")
        
        path_array = np.array(path, dtype=np.int32)
        return fast_path_cost(path_array, cost_matrix)
    
    def batch_calculate_distances(
        self,
        positions1: List[Dict[str, float]],
        positions2: List[Dict[str, float]]
    ) -> np.ndarray:
        """
        Calcular distancias entre dos sets de posiciones en batch.
        
        Args:
            positions1: Primer set de posiciones
            positions2: Segundo set de posiciones
        
        Returns:
            Matriz de distancias
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy not available")
        
        arr1 = np.array([
            [p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)]
            for p in positions1
        ])
        arr2 = np.array([
            [p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)]
            for p in positions2
        ])
        
        return fast_batch_distance(arr1, arr2)


class FunctionCache:
    """Cache para funciones compiladas."""
    
    def __init__(self, max_size: int = 100):
        """
        Inicializar cache de funciones.
        
        Args:
            max_size: Tamaño máximo del cache
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener función del cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Guardar función en cache."""
        if len(self.cache) >= self.max_size:
            # Evictar menos usado
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Limpiar cache."""
        self.cache.clear()
        self.access_times.clear()


class CompilationOptimizer:
    """Optimizador completo de compilación."""
    
    def __init__(self):
        """Inicializar optimizador de compilación."""
        self.jit_optimizer = NumbaJITOptimizer() if NUMBA_AVAILABLE else None
        self.route_calculator = CompiledRouteCalculator() if NUMBA_AVAILABLE and NUMPY_AVAILABLE else None
        self.function_cache = FunctionCache()
    
    def optimize_route_calculation(self):
        """Optimizar cálculos de rutas."""
        if self.route_calculator:
            logger.info("Route calculation optimized with compiled functions")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'numba_available': NUMBA_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'compiled_functions': len(self.jit_optimizer.compiled_functions) if self.jit_optimizer else 0,
            'cache_size': len(self.function_cache.cache)
        }

