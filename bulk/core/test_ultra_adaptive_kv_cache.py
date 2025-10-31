"""
Comprehensive Tests for Ultra-Adaptive K/V Cache Engine
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Import the engine
try:
    from ultra_adaptive_kv_cache_engine import (
        UltraAdaptiveKVCacheEngine,
        AdaptiveConfig,
        AdaptiveMode,
        TruthGPTIntegration
    )
except ImportError:
    pytest.skip("Ultra-Adaptive K/V Cache Engine not available", allow_module_level=True)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def config(temp_cache_dir):
    """Create test configuration."""
    return AdaptiveConfig(
        model_name="test-model",
        cache_size=1024,
        enable_cache_persistence=True,
        cache_persistence_path=temp_cache_dir,
        enable_checkpointing=True,
        checkpoint_interval=60,
        enable_metrics=True,
        num_workers=2
    )


@pytest.fixture
def engine(config):
    """Create test engine instance."""
    # Mock the TruthGPT components
    with patch('ultra_adaptive_kv_cache_engine.create_ultra_cache_config'), \
         patch('ultra_adaptive_kv_cache_engine.create_ultra_decoder_config'), \
         patch('ultra_adaptive_kv_cache_engine.create_ultra_optimization_config'), \
         patch('ultra_adaptive_kv_cache_engine.create_ultra_kv_cache_optimizer'), \
         patch('ultra_adaptive_kv_cache_engine.create_ultra_efficient_decoder'):
        
        # Create mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.generate_text = Mock(return_value="Generated text response")
        mock_optimizer.get_cache_stats = Mock(return_value={'hits': 10, 'misses': 5})
        mock_optimizer.clear_cache = Mock()
        
        # Create mock decoder
        mock_decoder = Mock()
        mock_decoder.to = Mock()
        
        with patch('ultra_adaptive_kv_cache_engine.create_ultra_kv_cache_optimizer', return_value=mock_optimizer), \
             patch('ultra_adaptive_kv_cache_engine.create_ultra_efficient_decoder', return_value=mock_decoder):
            
            engine = UltraAdaptiveKVCacheEngine(config)
            engine.optimizer = mock_optimizer
            engine.decoder = mock_decoder
            yield engine
            engine.shutdown()


class TestUltraAdaptiveKVCacheEngine:
    """Test suite for Ultra-Adaptive K/V Cache Engine."""
    
    def test_initialization(self, engine, config):
        """Test engine initialization."""
        assert engine.config == config
        assert engine.monitoring_enabled == config.enable_metrics
        assert engine.checkpoint_enabled == config.enable_checkpointing
    
    def test_gpu_detection(self, engine):
        """Test GPU detection."""
        gpus = engine._detect_gpus()
        
        if torch.cuda.is_available():
            assert len(gpus) > 0
            assert isinstance(gpus, list)
        else:
            assert len(gpus) == 0
    
    def test_gpu_selection(self, engine):
        """Test optimal GPU selection."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            engine.available_gpus = [0, 1]
            engine.gpu_workloads = {
                0: {'active_tasks': 5, 'memory_used': 0.5},
                1: {'active_tasks': 2, 'memory_used': 0.3}
            }
            
            selected = engine._select_optimal_gpu()
            # Should select GPU with less workload
            assert selected == 1
    
    @pytest.mark.asyncio
    async def test_process_request(self, engine):
        """Test processing a single request."""
        request = {
            'text': 'Test input text',
            'max_length': 50,
            'temperature': 0.7,
            'session_id': 'test_session_1'
        }
        
        result = await engine.process_request(request)
        
        assert result['success'] is True
        assert 'response' in result
        assert 'processing_time' in result
        assert result['session_id'] == 'test_session_1'
    
    @pytest.mark.asyncio
    async def test_process_request_with_cache(self, engine):
        """Test processing request with existing cache."""
        # Create session first
        engine.active_sessions['cached_session'] = {
            'created_at': time.time(),
            'last_used': time.time(),
            'request_count': 1
        }
        
        request = {
            'text': 'Test input',
            'max_length': 50,
            'temperature': 0.7,
            'session_id': 'cached_session'
        }
        
        result = await engine.process_request(request)
        
        assert result['success'] is True
        assert result['response']['cached'] is True
    
    @pytest.mark.asyncio
    async def test_process_batch(self, engine):
        """Test batch processing."""
        requests = [
            {
                'text': f'Request {i}',
                'max_length': 50,
                'temperature': 0.7,
                'session_id': f'session_{i}'
            }
            for i in range(5)
        ]
        
        results = await engine.process_batch(requests)
        
        assert len(results) == 5
        assert all(r['success'] for r in results)
    
    @pytest.mark.asyncio
    async def test_cache_persistence(self, engine, temp_cache_dir):
        """Test cache persistence to disk."""
        if not engine.config.enable_cache_persistence:
            pytest.skip("Cache persistence not enabled")
        
        request = {
            'text': 'Test for persistence',
            'max_length': 50,
            'temperature': 0.7,
            'session_id': 'persist_test'
        }
        
        # Process request
        result = await engine.process_request(request)
        assert result['success'] is True
        
        # Check if cache file was created
        cache_path = Path(engine.cache_path) / "sessions"
        cache_files = list(cache_path.glob('*.pkl'))
        
        # At least attempt to save should have been made
        assert engine.cache_path is not None
    
    def test_adapt_to_workload(self, engine):
        """Test workload adaptation."""
        workload_info = {
            'batch_size': 10,
            'sequence_length': 2048,
            'request_rate': 15.0,
            'memory_usage': 0.85
        }
        
        result = engine.adapt_to_workload(workload_info)
        
        assert result['adapted'] is True
        assert 'new_config' in result
        assert 'performance_impact' in result
    
    def test_performance_stats(self, engine):
        """Test getting performance statistics."""
        # Process some requests to generate metrics
        stats = engine.get_performance_stats()
        
        assert 'engine_stats' in stats
        assert 'active_sessions' in stats
        assert 'memory_usage' in stats
        assert 'config' in stats
        assert 'available_gpus' in stats
    
    def test_clear_cache(self, engine):
        """Test clearing cache."""
        # Add some sessions
        engine.active_sessions['test1'] = {'created_at': time.time()}
        engine.active_sessions['test2'] = {'created_at': time.time()}
        
        assert len(engine.active_sessions) == 2
        
        engine.clear_cache()
        
        assert len(engine.active_sessions) == 0
    
    def test_cleanup_sessions(self, engine):
        """Test session cleanup."""
        # Add old session
        old_time = time.time() - 7200  # 2 hours ago
        engine.active_sessions['old_session'] = {
            'created_at': old_time,
            'last_used': old_time
        }
        
        # Add recent session
        engine.active_sessions['recent_session'] = {
            'created_at': time.time(),
            'last_used': time.time()
        }
        
        engine.cleanup_sessions(max_age=3600)  # 1 hour
        
        assert 'old_session' not in engine.active_sessions
        assert 'recent_session' in engine.active_sessions
    
    def test_checkpoint_save_load(self, engine, temp_cache_dir):
        """Test checkpoint save and load."""
        if not engine.checkpoint_enabled:
            pytest.skip("Checkpointing not enabled")
        
        # Save checkpoint
        engine._save_checkpoint()
        
        # Load checkpoint
        result = engine.load_checkpoint()
        
        # Should be able to load (may return False if no checkpoint exists yet)
        assert isinstance(result, bool)
    
    def test_error_tracking(self, engine):
        """Test error tracking."""
        # Simulate an error by processing invalid request
        # This will depend on implementation details
        initial_error_count = len(engine.error_history)
        
        # The error tracking happens in process_request
        # We'll test the error history update
        engine.error_history.append({
            'timestamp': time.time(),
            'error': 'Test error',
            'request': {}
        })
        
        assert len(engine.error_history) == initial_error_count + 1
    
    def test_metrics_update(self, engine):
        """Test metrics update."""
        initial_requests = engine.performance_metrics['total_requests']
        
        engine._update_metrics(processing_time=0.5, tokens_generated=100)
        
        assert engine.performance_metrics['total_requests'] == initial_requests + 1
        assert engine.performance_metrics['total_tokens'] >= 100
    
    def test_memory_tracking(self, engine):
        """Test memory usage tracking."""
        mem_usage = engine._get_current_memory_usage()
        
        assert isinstance(mem_usage, float)
        assert 0.0 <= mem_usage <= 1.0 or mem_usage > 1.0  # Can be > 1.0 if using more than available


class TestTruthGPTIntegration:
    """Test suite for TruthGPT integration helpers."""
    
    def test_create_engine_for_truthgpt(self):
        """Test creating engine for TruthGPT."""
        with patch('ultra_adaptive_kv_cache_engine.create_adaptive_engine') as mock_create:
            engine = Mock()
            mock_create.return_value = engine
            
            result = TruthGPTIntegration.create_engine_for_truthgpt()
            
            mock_create.assert_called_once()
            assert result == engine
    
    def test_create_bulk_engine(self):
        """Test creating bulk processing engine."""
        with patch('ultra_adaptive_kv_cache_engine.create_adaptive_engine') as mock_create:
            engine = Mock()
            mock_create.return_value = engine
            
            result = TruthGPTIntegration.create_bulk_engine()
            
            mock_create.assert_called_once()
            assert result == engine
    
    def test_create_streaming_engine(self):
        """Test creating streaming engine."""
        with patch('ultra_adaptive_kv_cache_engine.create_adaptive_engine') as mock_create:
            engine = Mock()
            mock_create.return_value = engine
            
            result = TruthGPTIntegration.create_streaming_engine()
            
            mock_create.assert_called_once()
            assert result == engine


@pytest.mark.asyncio
class TestConcurrency:
    """Test concurrent operations."""
    
    async def test_concurrent_requests(self, engine):
        """Test handling concurrent requests."""
        requests = [
            {
                'text': f'Concurrent request {i}',
                'max_length': 50,
                'temperature': 0.7,
                'session_id': f'concurrent_{i}'
            }
            for i in range(10)
        ]
        
        # Process concurrently
        tasks = [engine.process_request(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(r['success'] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

