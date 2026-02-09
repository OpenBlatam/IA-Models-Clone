"""
Tests de optimización y mejoras de performance
"""

import pytest
import time
from unittest.mock import Mock
import functools


class TestMemoization:
    """Tests de memoización"""
    
    def test_memoization_basic(self):
        """Test básico de memoización"""
        cache = {}
        
        def memoized(func):
            @functools.wraps(func)
            def wrapper(*args):
                key = str(args)
                if key not in cache:
                    cache[key] = func(*args)
                return cache[key]
            return wrapper
        
        call_count = [0]
        
        @memoized
        def expensive_operation(n):
            call_count[0] += 1
            return n * 2
        
        result1 = expensive_operation(5)
        result2 = expensive_operation(5)
        
        assert result1 == 10
        assert result2 == 10
        assert call_count[0] == 1  # Solo se llamó una vez
    
    def test_memoization_with_different_args(self):
        """Test de memoización con diferentes argumentos"""
        cache = {}
        
        def memoized(func):
            @functools.wraps(func)
            def wrapper(*args):
                key = str(args)
                if key not in cache:
                    cache[key] = func(*args)
                return cache[key]
            return wrapper
        
        @memoized
        def add(a, b):
            return a + b
        
        assert add(1, 2) == 3
        assert add(1, 2) == 3  # Debe usar caché
        assert add(2, 3) == 5  # Nuevo cálculo


class TestLazyLoading:
    """Tests de lazy loading"""
    
    def test_lazy_evaluation(self):
        """Test de evaluación perezosa"""
        class LazyValue:
            def __init__(self, compute_func):
                self._compute = compute_func
                self._value = None
                self._computed = False
            
            @property
            def value(self):
                if not self._computed:
                    self._value = self._compute()
                    self._computed = True
                return self._value
        
        call_count = [0]
        
        def expensive_computation():
            call_count[0] += 1
            return "computed_value"
        
        lazy = LazyValue(expensive_computation)
        
        # No debe calcular hasta acceder
        assert call_count[0] == 0
        
        # Acceder al valor
        value = lazy.value
        assert value == "computed_value"
        assert call_count[0] == 1
        
        # Acceder de nuevo (no debe recalcular)
        value2 = lazy.value
        assert value2 == "computed_value"
        assert call_count[0] == 1


class TestDataStructuresOptimization:
    """Tests de optimización de estructuras de datos"""
    
    def test_set_lookup_performance(self):
        """Test de performance de lookup en set vs list"""
        large_list = list(range(10000))
        large_set = set(range(10000))
        
        # Lookup en list (O(n))
        start = time.time()
        for _ in range(1000):
            _ = 5000 in large_list
        list_time = time.time() - start
        
        # Lookup en set (O(1))
        start = time.time()
        for _ in range(1000):
            _ = 5000 in large_set
        set_time = time.time() - start
        
        # Set debe ser más rápido
        assert set_time < list_time
    
    def test_dict_comprehension_vs_loop(self):
        """Test de dict comprehension vs loop"""
        items = list(range(1000))
        
        # Loop tradicional
        start = time.time()
        result1 = {}
        for item in items:
            result1[item] = item * 2
        loop_time = time.time() - start
        
        # Dict comprehension
        start = time.time()
        result2 = {item: item * 2 for item in items}
        comp_time = time.time() - start
        
        assert result1 == result2
        # Comprehension generalmente es más rápido
        assert comp_time <= loop_time * 1.5


class TestAlgorithmOptimization:
    """Tests de optimización de algoritmos"""
    
    def test_early_exit_optimization(self):
        """Test de optimización con early exit"""
        def find_item_early_exit(items, target):
            for item in items:
                if item == target:
                    return True
            return False
        
        def find_item_no_early_exit(items, target):
            found = False
            for item in items:
                if item == target:
                    found = True
            return found
        
        items = list(range(10000))
        target = 100
        
        # Early exit debe ser más rápido
        start = time.time()
        result1 = find_item_early_exit(items, target)
        early_time = time.time() - start
        
        start = time.time()
        result2 = find_item_no_early_exit(items, target)
        no_early_time = time.time() - start
        
        assert result1 == result2
        assert early_time < no_early_time
    
    def test_batch_processing_optimization(self):
        """Test de optimización de procesamiento en lote"""
        def process_one_by_one(items):
            results = []
            for item in items:
                results.append(item * 2)
            return results
        
        def process_batch(items, batch_size=100):
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                results.extend([item * 2 for item in batch])
            return results
        
        items = list(range(1000))
        
        result1 = process_one_by_one(items)
        result2 = process_batch(items)
        
        assert result1 == result2


class TestMemoryOptimization:
    """Tests de optimización de memoria"""
    
    def test_generator_vs_list(self):
        """Test de generador vs lista para ahorrar memoria"""
        def list_approach(n):
            return [i * 2 for i in range(n)]
        
        def generator_approach(n):
            return (i * 2 for i in range(n))
        
        n = 1000000
        
        # Lista consume más memoria
        list_result = list_approach(1000)  # Limitado para test
        assert len(list_result) == 1000
        
        # Generador es más eficiente en memoria
        gen_result = generator_approach(n)
        first = next(gen_result)
        assert first == 0
    
    def test_del_unused_variables(self):
        """Test de eliminación de variables no usadas"""
        def process_with_cleanup(data):
            # Procesar datos
            result = [item * 2 for item in data]
            
            # Limpiar variables grandes
            del data
            
            return result
        
        large_data = list(range(10000))
        result = process_with_cleanup(large_data)
        
        assert len(result) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

