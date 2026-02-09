"""
Tests refactorizados para helper decorators
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import time
from unittest.mock import Mock

from core.helpers.decorators import (
    timer,
    memoize,
    singleton,
    deprecated
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestTimerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para timer decorator"""
    
    def test_timer_decorator(self):
        """Test de decorador timer"""
        @timer
        def slow_function():
            time.sleep(0.01)
            return "result"
        
        result = slow_function()
        
        assert result == "result"
        assert hasattr(slow_function, '__wrapped__')


class TestMemoizeRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para memoize decorator"""
    
    def test_memoize_decorator(self):
        """Test de decorador memoize"""
        call_count = 0
        
        @memoize
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_function(5)
        result2 = expensive_function(5)
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Solo se llamó una vez
    
    @pytest.mark.parametrize("args1,args2,expected_calls", [
        ((5,), (5,), 1),
        ((5,), (10,), 2),
        ((5, 6), (5, 6), 1),
        ((5, 6), (5, 7), 2)
    ])
    def test_memoize_different_args(self, args1, args2, expected_calls):
        """Test de memoize con diferentes argumentos"""
        call_count = 0
        
        @memoize
        def expensive_function(*args):
            nonlocal call_count
            call_count += 1
            return sum(args) * 2
        
        expensive_function(*args1)
        expensive_function(*args2)
        
        assert call_count == expected_calls


class TestSingletonRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para singleton decorator"""
    
    def test_singleton_decorator(self):
        """Test de decorador singleton"""
        @singleton
        class TestClass:
            def __init__(self):
                self.value = 42
        
        instance1 = TestClass()
        instance2 = TestClass()
        
        assert instance1 is instance2
        assert instance1.value == 42


class TestDeprecatedRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para deprecated decorator"""
    
    def test_deprecated_decorator(self):
        """Test de decorador deprecated"""
        @deprecated
        def old_function():
            return "result"
        
        result = old_function()
        
        assert result == "result"
        assert hasattr(old_function, '__wrapped__')



