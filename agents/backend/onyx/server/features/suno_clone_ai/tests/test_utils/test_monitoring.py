"""
Comprehensive Unit Tests for Monitoring

Tests cover monitoring and metrics functionality with diverse test cases
"""

import pytest
from utils.monitoring import GenerationMetrics, SystemMonitor


class TestGenerationMetrics:
    """Test cases for GenerationMetrics class"""
    
    def test_generation_metrics_init(self):
        """Test initializing generation metrics"""
        metrics = GenerationMetrics()
        assert len(metrics.generation_times) == 0
        assert metrics.total_generations == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
    
    def test_record_generation_basic(self):
        """Test recording basic generation"""
        metrics = GenerationMetrics()
        metrics.record_generation(duration=1.5, text_length=30)
        
        assert metrics.total_generations == 1
        assert len(metrics.generation_times) == 1
        assert metrics.generation_times[0] == 1.5
    
    def test_record_generation_with_cache_hit(self):
        """Test recording generation with cache hit"""
        metrics = GenerationMetrics()
        metrics.record_generation(duration=0.1, text_length=20, cache_hit=True)
        
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 0
    
    def test_record_generation_with_cache_miss(self):
        """Test recording generation with cache miss"""
        metrics = GenerationMetrics()
        metrics.record_generation(duration=2.0, text_length=50, cache_hit=False)
        
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 1
    
    def test_record_generation_categorizes_by_length(self):
        """Test generation categorization by text length"""
        metrics = GenerationMetrics()
        
        # Short text
        metrics.record_generation(duration=1.0, text_length=10)
        assert metrics.generation_counts["short"] == 1
        
        # Medium text
        metrics.record_generation(duration=1.0, text_length=30)
        assert metrics.generation_counts["medium"] == 1
        
        # Long text
        metrics.record_generation(duration=1.0, text_length=60)
        assert metrics.generation_counts["long"] == 1
    
    def test_record_error(self):
        """Test recording error"""
        metrics = GenerationMetrics()
        metrics.record_error("timeout")
        metrics.record_error("timeout")
        metrics.record_error("validation")
        
        assert metrics.error_counts["timeout"] == 2
        assert metrics.error_counts["validation"] == 1
    
    def test_get_stats_empty(self):
        """Test getting stats with no data"""
        metrics = GenerationMetrics()
        stats = metrics.get_stats()
        
        assert stats["total_generations"] == 0
        assert stats["avg_generation_time"] == 0
        assert stats["cache_hit_rate"] == 0
    
    def test_get_stats_with_data(self):
        """Test getting stats with data"""
        metrics = GenerationMetrics()
        metrics.record_generation(duration=1.0, text_length=20, cache_hit=True)
        metrics.record_generation(duration=2.0, text_length=30, cache_hit=False)
        
        stats = metrics.get_stats()
        
        assert stats["total_generations"] == 2
        assert stats["avg_generation_time_seconds"] == 1.5
        assert stats["min_generation_time_seconds"] == 1.0
        assert stats["max_generation_time_seconds"] == 2.0
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["cache_hit_rate_percent"] == 50.0
    
    def test_get_stats_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation"""
        metrics = GenerationMetrics()
        
        # 3 hits, 2 misses
        for _ in range(3):
            metrics.record_generation(1.0, 20, cache_hit=True)
        for _ in range(2):
            metrics.record_generation(1.0, 20, cache_hit=False)
        
        stats = metrics.get_stats()
        assert stats["cache_hit_rate_percent"] == 60.0
    
    def test_generation_times_deque_limit(self):
        """Test generation times deque has max length"""
        metrics = GenerationMetrics()
        
        # Add more than maxlen
        for i in range(1500):
            metrics.record_generation(duration=1.0, text_length=20)
        
        # Should be limited to maxlen (1000)
        assert len(metrics.generation_times) == 1000


class TestSystemMonitor:
    """Test cases for SystemMonitor class"""
    
    def test_system_monitor_init(self):
        """Test initializing system monitor"""
        monitor = SystemMonitor()
        assert monitor.metrics is not None
        assert isinstance(monitor.metrics, GenerationMetrics)
    
    def test_get_system_info(self):
        """Test getting system information"""
        monitor = SystemMonitor()
        info = monitor.get_system_info()
        
        assert isinstance(info, dict)
        assert "metrics" in info
    
    def test_get_system_info_includes_metrics(self):
        """Test system info includes metrics"""
        monitor = SystemMonitor()
        monitor.metrics.record_generation(duration=1.0, text_length=20)
        
        info = monitor.get_system_info()
        
        assert "metrics" in info
        assert info["metrics"]["total_generations"] == 1















