"""
Parallel Processing Tests

Comprehensive test suite for the parallel processing system.
"""

import pytest
import asyncio
import time
from typing import List
from unittest.mock import Mock, patch, AsyncMock

from ..models.video_models import VideoClipRequest, VideoClipResponse
from ..models.viral_models import ViralVideoBatchResponse
from ..processors.video_processor import VideoClipProcessor, create_high_performance_processor
from ..processors.viral_processor import ViralVideoProcessor, create_high_performance_viral_processor
from ..utils.parallel_utils import (
    HybridParallelProcessor,
    parallel_map,
    BackendType,
    ParallelConfig,
    setup_async_loop
)

# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def generate_test_requests(count: int = 10) -> List[VideoClipRequest]:
    """Generate test video requests."""
    return [
        VideoClipRequest(
            youtube_url=f"https://youtube.com/watch?v=test{i}",
            language="en" if i % 2 == 0 else "es",
            max_clip_length=60 + (i % 30),
            min_clip_length=15 + (i % 10)
        )
        for i in range(count)
    ]

def generate_mock_responses(count: int = 10) -> List[VideoClipResponse]:
    """Generate mock video responses."""
    return [
        VideoClipResponse(
            success=True,
            clip_id=f"clip_{i}",
            duration=30 + (i % 20),
            language="en" if i % 2 == 0 else "es"
        )
        for i in range(count)
    ]

# =============================================================================
# UNIT TESTS
# =============================================================================

class TestVideoClipProcessor:
    """Test VideoClipProcessor functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = VideoClipProcessor()
        self.test_requests = generate_test_requests(5)
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'process')
        assert hasattr(self.processor, 'process_batch_parallel')
    
    def test_single_request_processing(self):
        """Test processing a single request."""
        request = self.test_requests[0]
        
        with patch.object(self.processor, 'process', return_value=VideoClipResponse(
            success=True, clip_id="test_clip", duration=30, language="en"
        )):
            result = self.processor.process(request)
            
            assert result.success is True
            assert result.clip_id == "test_clip"
            assert result.duration == 30
            assert result.language == "en"
    
    def test_batch_validation(self):
        """Test batch validation."""
        # Valid requests
        valid_results = self.processor.validate_batch(self.test_requests)
        assert len(valid_results) == len(self.test_requests)
        assert all(valid_results)
        
        # Invalid requests
        invalid_requests = [
            VideoClipRequest(
                youtube_url="invalid_url",
                max_clip_length=0,  # Invalid
                min_clip_length=100  # Invalid
            )
        ]
        
        invalid_results = self.processor.validate_batch(invalid_requests)
        assert len(invalid_results) == len(invalid_requests)
        assert not any(invalid_results)
    
    def test_batch_processing_sequential(self):
        """Test batch processing in sequential mode."""
        mock_responses = generate_mock_responses(len(self.test_requests))
        
        with patch.object(self.processor, 'process', side_effect=mock_responses):
            results = self.processor.process_batch_sequential(self.test_requests)
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)
    
    def test_batch_processing_parallel_thread(self):
        """Test batch processing with threading backend."""
        mock_responses = generate_mock_responses(len(self.test_requests))
        
        with patch.object(self.processor, 'process', side_effect=mock_responses):
            results = self.processor.process_batch_parallel(
                self.test_requests,
                backend=BackendType.THREAD
            )
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)
    
    def test_batch_processing_parallel_process(self):
        """Test batch processing with multiprocessing backend."""
        mock_responses = generate_mock_responses(len(self.test_requests))
        
        with patch.object(self.processor, 'process', side_effect=mock_responses):
            results = self.processor.process_batch_parallel(
                self.test_requests,
                backend=BackendType.PROCESS
            )
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)
    
    def test_batch_processing_parallel_joblib(self):
        """Test batch processing with joblib backend."""
        mock_responses = generate_mock_responses(len(self.test_requests))
        
        with patch.object(self.processor, 'process', side_effect=mock_responses):
            results = self.processor.process_batch_parallel(
                self.test_requests,
                backend=BackendType.JOBLIB
            )
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)
    
    def test_error_handling(self):
        """Test error handling in batch processing."""
        def mock_process_with_error(request):
            if request.youtube_url == "https://youtube.com/watch?v=test0":
                raise Exception("Processing error")
            return VideoClipResponse(success=True, clip_id="success", duration=30, language="en")
        
        with patch.object(self.processor, 'process', side_effect=mock_process_with_error):
            results = self.processor.process_batch_parallel(self.test_requests)
            
            assert len(results) == len(self.test_requests)
            # First request should have error, others should succeed
            assert not results[0].success
            assert all(result.success for result in results[1:])

class TestViralVideoProcessor:
    """Test ViralVideoProcessor functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ViralVideoProcessor()
        self.test_requests = generate_test_requests(3)
    
    def test_viral_processor_initialization(self):
        """Test viral processor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'process_viral')
        assert hasattr(self.processor, 'process_batch_parallel')
    
    def test_viral_batch_processing(self):
        """Test viral batch processing."""
        mock_responses = [
            ViralVideoBatchResponse(
                success=True,
                original_clip_id=f"clip_{i}",
                variants=[
                    Mock(viral_score=0.8 + (i * 0.1), title=f"Variant {i}_{j}")
                    for j in range(3)
                ]
            )
            for i in range(len(self.test_requests))
        ]
        
        with patch.object(self.processor, 'process_viral', side_effect=mock_responses):
            results = self.processor.process_batch_parallel(
                self.test_requests,
                n_variants=3
            )
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)
            assert all(len(result.variants) == 3 for result in results)
    
    def test_viral_variant_generation(self):
        """Test viral variant generation."""
        mock_response = ViralVideoBatchResponse(
            success=True,
            original_clip_id="test_clip",
            variants=[
                Mock(viral_score=0.9, title="High Score Variant"),
                Mock(viral_score=0.7, title="Medium Score Variant"),
                Mock(viral_score=0.5, title="Low Score Variant")
            ]
        )
        
        with patch.object(self.processor, 'process_viral', return_value=mock_response):
            result = self.processor.process_viral(
                self.test_requests[0],
                n_variants=3,
                audience_profile={'age': '18-35'}
            )
            
            assert result.success is True
            assert len(result.variants) == 3
            assert all(hasattr(variant, 'viral_score') for variant in result.variants)

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestParallelProcessingIntegration:
    """Integration tests for parallel processing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_requests = generate_test_requests(20)
    
    def test_high_performance_processor_creation(self):
        """Test high performance processor creation."""
        processor = create_high_performance_processor()
        assert processor is not None
        assert hasattr(processor, 'process_batch_parallel')
    
    def test_high_performance_viral_processor_creation(self):
        """Test high performance viral processor creation."""
        processor = create_high_performance_viral_processor()
        assert processor is not None
        assert hasattr(processor, 'process_batch_parallel')
    
    def test_auto_backend_selection(self):
        """Test automatic backend selection."""
        processor = create_high_performance_processor()
        
        # Mock the process method to avoid actual processing
        with patch.object(processor, 'process', return_value=VideoClipResponse(
            success=True, clip_id="test", duration=30, language="en"
        )):
            results = processor.process_batch_parallel(self.test_requests)
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)
    
    def test_backend_performance_comparison(self):
        """Test performance comparison between backends."""
        processor = create_high_performance_processor()
        
        backends = [BackendType.THREAD, BackendType.PROCESS, BackendType.JOBLIB]
        performance_results = {}
        
        # Mock the process method
        with patch.object(processor, 'process', return_value=VideoClipResponse(
            success=True, clip_id="test", duration=30, language="en"
        )):
            for backend in backends:
                start_time = time.perf_counter()
                results = processor.process_batch_parallel(
                    self.test_requests[:5],  # Smaller batch for testing
                    backend=backend
                )
                duration = time.perf_counter() - start_time
                
                performance_results[backend] = {
                    'duration': duration,
                    'success_count': sum(1 for r in results if r.success)
                }
        
        # Verify all backends completed successfully
        for backend, results in performance_results.items():
            assert results['success_count'] == 5
            assert results['duration'] > 0

# =============================================================================
# ASYNC TESTS
# =============================================================================

class TestAsyncProcessing:
    """Test async processing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_requests = generate_test_requests(10)
        setup_async_loop()
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test async batch processing."""
        processor = create_high_performance_processor()
        
        # Mock the async process method
        async def mock_async_process(request):
            await asyncio.sleep(0.01)  # Simulate async work
            return VideoClipResponse(
                success=True,
                clip_id=f"async_{request.youtube_url}",
                duration=30,
                language=request.language
            )
        
        with patch.object(processor, 'process_async', side_effect=mock_async_process):
            results = await processor.process_batch_async(self.test_requests)
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)
    
    @pytest.mark.asyncio
    async def test_async_viral_processing(self):
        """Test async viral processing."""
        processor = create_high_performance_viral_processor()
        
        # Mock the async viral process method
        async def mock_async_viral_process(request, n_variants=3):
            await asyncio.sleep(0.01)  # Simulate async work
            return ViralVideoBatchResponse(
                success=True,
                original_clip_id=f"viral_{request.youtube_url}",
                variants=[
                    Mock(viral_score=0.8 + (i * 0.1), title=f"Async Variant {i}")
                    for i in range(n_variants)
                ]
            )
        
        with patch.object(processor, 'process_viral_async', side_effect=mock_async_viral_process):
            results = await processor.process_batch_async(
                self.test_requests,
                n_variants=3
            )
            
            assert len(results) == len(self.test_requests)
            assert all(result.success for result in results)

# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestParallelUtils:
    """Test parallel processing utilities."""
    
    def test_parallel_map_function(self):
        """Test parallel_map utility function."""
        def test_function(x):
            return x * 2
        
        data = list(range(10))
        
        # Test with different backends
        for backend in [BackendType.THREAD, BackendType.PROCESS]:
            results = parallel_map(test_function, data, backend=backend)
            assert len(results) == len(data)
            assert results == [x * 2 for x in data]
    
    def test_hybrid_parallel_processor(self):
        """Test HybridParallelProcessor."""
        processor = HybridParallelProcessor()
        
        def test_function(x):
            return x ** 2
        
        data = list(range(10))
        results = processor.map(test_function, data)
        
        assert len(results) == len(data)
        assert results == [x ** 2 for x in data]
    
    def test_parallel_config(self):
        """Test ParallelConfig dataclass."""
        config = ParallelConfig(
            max_workers=16,
            chunk_size=500,
            timeout=60.0,
            backend=BackendType.PROCESS
        )
        
        assert config.max_workers == 16
        assert config.chunk_size == 500
        assert config.timeout == 60.0
        assert config.backend == BackendType.PROCESS

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for parallel processing."""
    
    def test_processing_speed_comparison(self):
        """Test processing speed comparison between backends."""
        processor = create_high_performance_processor()
        test_requests = generate_test_requests(50)
        
        # Mock fast processing
        with patch.object(processor, 'process', return_value=VideoClipResponse(
            success=True, clip_id="test", duration=30, language="en"
        )):
            # Test sequential processing
            start_time = time.perf_counter()
            sequential_results = processor.process_batch_sequential(test_requests)
            sequential_time = time.perf_counter() - start_time
            
            # Test parallel processing
            start_time = time.perf_counter()
            parallel_results = processor.process_batch_parallel(test_requests)
            parallel_time = time.perf_counter() - start_time
            
            # Parallel should be faster (or at least not slower)
            assert parallel_time <= sequential_time * 1.5  # Allow some overhead
            assert len(sequential_results) == len(parallel_results)
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        processor = create_high_performance_processor()
        test_requests = generate_test_requests(100)
        
        # Mock processing
        with patch.object(processor, 'process', return_value=VideoClipResponse(
            success=True, clip_id="test", duration=30, language="en"
        )):
            memory_before = process.memory_info().rss
            
            results = processor.process_batch_parallel(test_requests)
            
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (less than 100MB for 100 items)
            assert memory_increase < 100 * 1024 * 1024  # 100MB
            assert len(results) == len(test_requests)

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in parallel processing."""
    
    def test_partial_failure_handling(self):
        """Test handling of partial failures."""
        processor = create_high_performance_processor()
        test_requests = generate_test_requests(10)
        
        def mock_process_with_failures(request):
            if "test0" in request.youtube_url:
                raise Exception("Simulated failure")
            return VideoClipResponse(
                success=True,
                clip_id=request.youtube_url,
                duration=30,
                language=request.language
            )
        
        with patch.object(processor, 'process', side_effect=mock_process_with_failures):
            results = processor.process_batch_parallel(test_requests)
            
            # Should handle failures gracefully
            assert len(results) == len(test_requests)
            # First request should fail, others should succeed
            assert not results[0].success
            assert all(result.success for result in results[1:])
    
    def test_timeout_handling(self):
        """Test timeout handling."""
        processor = create_high_performance_processor()
        test_requests = generate_test_requests(5)
        
        def mock_slow_process(request):
            time.sleep(2)  # Simulate slow processing
            return VideoClipResponse(
                success=True,
                clip_id=request.youtube_url,
                duration=30,
                language=request.language
            )
        
        with patch.object(processor, 'process', side_effect=mock_slow_process):
            # Should handle timeouts gracefully
            results = processor.process_batch_parallel(
                test_requests,
                timeout=1.0  # 1 second timeout
            )
            
            assert len(results) == len(test_requests)

# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test configuration options."""
    
    def test_custom_configuration(self):
        """Test custom processor configuration."""
        config = ParallelConfig(
            max_workers=4,
            chunk_size=50,
            timeout=10.0,
            backend=BackendType.THREAD
        )
        
        processor = VideoClipProcessor(config)
        assert processor.config.max_workers == 4
        assert processor.config.chunk_size == 50
        assert processor.config.timeout == 10.0
        assert processor.config.backend == BackendType.THREAD
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            ParallelConfig(max_workers=0)
        
        with pytest.raises(ValueError):
            ParallelConfig(chunk_size=0)
        
        with pytest.raises(ValueError):
            ParallelConfig(timeout=-1.0)

if __name__ == "__main__":
    pytest.main([__file__]) 