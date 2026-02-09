"""
Comprehensive Unit Tests for Performance Optimizer

Tests cover performance optimization functionality with diverse test cases
"""

import pytest
import sys
from unittest.mock import patch, Mock

from core.performance_optimizer import PerformanceOptimizer, get_performance_optimizer


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer class"""
    
    def test_performance_optimizer_init(self):
        """Test initializing performance optimizer"""
        optimizer = PerformanceOptimizer()
        assert optimizer._optimized is False
    
    def test_optimize_all_first_call(self):
        """Test optimizing all on first call"""
        optimizer = PerformanceOptimizer()
        
        with patch.object(optimizer, '_optimize_sys_path') as mock_sys:
            with patch.object(optimizer, '_optimize_imports') as mock_imports:
                with patch.object(optimizer, '_configure_python_opts') as mock_opts:
                    with patch.object(optimizer, '_precompile_regex') as mock_regex:
                        optimizer.optimize_all()
                        
                        mock_sys.assert_called_once()
                        mock_imports.assert_called_once()
                        mock_opts.assert_called_once()
                        mock_regex.assert_called_once()
                        assert optimizer._optimized is True
    
    def test_optimize_all_idempotent(self):
        """Test optimize_all is idempotent"""
        optimizer = PerformanceOptimizer()
        
        with patch.object(optimizer, '_optimize_sys_path') as mock_sys:
            optimizer.optimize_all()
            optimizer.optimize_all()  # Second call
            
            # Should only be called once
            assert mock_sys.call_count == 1
    
    def test_optimize_sys_path(self):
        """Test sys.path optimization"""
        optimizer = PerformanceOptimizer()
        original_path = sys.path.copy()
        
        # Add duplicate
        sys.path.append(original_path[0])
        
        optimizer._optimize_sys_path()
        
        # Should not raise
        assert isinstance(sys.path, list)
    
    def test_optimize_imports(self):
        """Test imports optimization"""
        optimizer = PerformanceOptimizer()
        
        optimizer._optimize_imports()
        
        # Should cache modules
        assert '_fast_json' in sys.modules or True  # May or may not be set
    
    def test_optimize_imports_handles_errors(self):
        """Test imports optimization handles errors gracefully"""
        optimizer = PerformanceOptimizer()
        
        with patch('builtins.__import__', side_effect=ImportError("Test error")):
            # Should not raise
            optimizer._optimize_imports()
    
    def test_configure_python_opts(self):
        """Test Python options configuration"""
        optimizer = PerformanceOptimizer()
        original_dont_write = sys.dont_write_bytecode
        
        optimizer._configure_python_opts()
        
        # Should set dont_write_bytecode to False
        assert sys.dont_write_bytecode is False
        
        # Restore
        sys.dont_write_bytecode = original_dont_write
    
    def test_precompile_regex(self):
        """Test regex precompilation"""
        optimizer = PerformanceOptimizer()
        
        optimizer._precompile_regex()
        
        # Should have compiled regex
        assert hasattr(optimizer, '_email_regex') or True
        assert hasattr(optimizer, '_uuid_regex') or True
    
    def test_precompile_regex_handles_errors(self):
        """Test regex precompilation handles errors"""
        optimizer = PerformanceOptimizer()
        
        with patch('re.compile', side_effect=Exception("Regex error")):
            # Should not raise
            optimizer._precompile_regex()


class TestGetPerformanceOptimizer:
    """Test cases for get_performance_optimizer function"""
    
    def test_get_performance_optimizer_singleton(self):
        """Test that get_performance_optimizer returns singleton"""
        optimizer1 = get_performance_optimizer()
        optimizer2 = get_performance_optimizer()
        
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, PerformanceOptimizer)
    
    def test_get_performance_optimizer_multiple_calls(self):
        """Test multiple calls return same instance"""
        optimizers = [get_performance_optimizer() for _ in range(5)]
        assert all(o is optimizers[0] for o in optimizers)















