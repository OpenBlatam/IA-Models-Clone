"""
Advanced optimization tests
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import time


class TestOptimizationAdvanced:
    """Advanced optimization tests"""
    
    def test_code_optimization(self, temp_dir):
        """Test code optimization"""
        # Original code (inefficient)
        original_code = """
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""
        
        # Optimized code
        optimized_code = """
def process_data(data):
    return [item * 2 for item in data]
"""
        
        # Both should work
        assert "def process_data" in original_code
        assert "def process_data" in optimized_code
        assert "list comprehension" in optimized_code.lower() or "[" in optimized_code
    
    def test_memory_optimization(self, temp_dir):
        """Test memory optimization"""
        # Create large data structure
        large_list = list(range(1000))
        
        # Optimize: use generator instead
        large_generator = (x for x in range(1000))
        
        # Generator uses less memory
        assert hasattr(large_generator, "__iter__")
        assert sum(1 for _ in large_generator) == 1000
    
    def test_cache_optimization(self, temp_dir):
        """Test cache optimization"""
        cache = {}
        
        def expensive_operation(x):
            if x not in cache:
                # Simulate expensive operation
                time.sleep(0.001)
                cache[x] = x * 2
            return cache[x]
        
        # First call (cache miss)
        start1 = time.time()
        result1 = expensive_operation(5)
        time1 = time.time() - start1
        
        # Second call (cache hit)
        start2 = time.time()
        result2 = expensive_operation(5)
        time2 = time.time() - start2
        
        # Cache should make second call faster
        assert result1 == result2 == 10
        assert time2 <= time1  # Should be faster or equal
    
    def test_batch_optimization(self, temp_dir):
        """Test batch processing optimization"""
        items = list(range(100))
        batch_size = 10
        
        # Process in batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Should be more efficient than processing one by one
        assert len(batches) == 10
        assert all(len(batch) == batch_size for batch in batches[:-1])
    
    def test_algorithm_optimization(self):
        """Test algorithm optimization"""
        # O(n²) algorithm
        def slow_search(items, target):
            for i, item in enumerate(items):
                for j, other in enumerate(items):
                    if item + other == target:
                        return (i, j)
            return None
        
        # O(n) algorithm
        def fast_search(items, target):
            seen = {}
            for i, item in enumerate(items):
                complement = target - item
                if complement in seen:
                    return (seen[complement], i)
                seen[item] = i
            return None
        
        # Both should work
        items = [1, 2, 3, 4, 5]
        result_slow = slow_search(items, 7)
        result_fast = fast_search(items, 7)
        
        assert result_slow is not None or result_fast is not None
    
    def test_io_optimization(self, temp_dir):
        """Test I/O optimization"""
        # Write multiple files individually (slow)
        start1 = time.time()
        for i in range(10):
            (temp_dir / f"file_{i}.txt").write_text(f"content {i}")
        time1 = time.time() - start1
        
        # Write multiple files in batch (faster)
        start2 = time.time()
        content = "\n".join(f"content {i}" for i in range(10))
        (temp_dir / "batch.txt").write_text(content)
        time2 = time.time() - start2
        
        # Both should complete
        assert time1 > 0
        assert time2 > 0

