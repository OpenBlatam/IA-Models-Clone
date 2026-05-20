"""
Tests for Batch Processing
Tests for batch operations and processing
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List

from core.infrastructure.batch_processor import BatchProcessor


class TestBatchProcessor:
    """Tests for BatchProcessor"""
    
    @pytest.fixture
    def batch_processor(self):
        """Create batch processor"""
        return BatchProcessor(batch_size=10)
    
    @pytest.mark.asyncio
    async def test_process_batch(self, batch_processor):
        """Test processing a batch"""
        items = list(range(20))
        
        async def process_item(item):
            return item * 2
        
        results = await batch_processor.process_batch(items, process_item)
        
        assert len(results) == 20
        assert all(r == i * 2 for i, r in zip(items, results))
    
    @pytest.mark.asyncio
    async def test_process_batch_with_chunking(self, batch_processor):
        """Test batch processing with chunking"""
        items = list(range(25))  # More than batch_size
        
        processed_chunks = []
        
        async def process_item(item):
            return item * 2
        
        results = await batch_processor.process_batch(items, process_item)
        
        assert len(results) == 25
        # Should process in chunks of batch_size
    
    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, batch_processor):
        """Test batch processing with some errors"""
        items = list(range(10))
        
        async def process_item(item):
            if item == 5:
                raise Exception("Processing error")
            return item * 2
        
        # Should handle errors gracefully
        results = await batch_processor.process_batch(
            items, 
            process_item,
            continue_on_error=True
        )
        
        # Should have results for successful items
        assert len(results) < 10  # Some failed
    
    @pytest.mark.asyncio
    async def test_process_batch_empty(self, batch_processor):
        """Test processing empty batch"""
        items = []
        
        async def process_item(item):
            return item * 2
        
        results = await batch_processor.process_batch(items, process_item)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_concurrent(self, batch_processor):
        """Test concurrent batch processing"""
        items = list(range(50))
        
        async def process_item(item):
            await asyncio.sleep(0.01)  # Simulate async work
            return item * 2
        
        start_time = asyncio.get_event_loop().time()
        results = await batch_processor.process_batch(items, process_item)
        duration = asyncio.get_event_loop().time() - start_time
        
        assert len(results) == 50
        # Should be faster than sequential (though exact timing depends on implementation)
        assert duration < 1.0


class TestBatchAnalysis:
    """Tests for batch analysis operations"""
    
    @pytest.mark.asyncio
    async def test_batch_analyze_images(self):
        """Test batch image analysis"""
        from core.infrastructure.batch_processor import BatchProcessor
        
        processor = BatchProcessor(batch_size=5)
        
        images = [b"image_data"] * 10
        
        async def analyze_image(image_data):
            # Mock analysis
            return {"score": 75.0}
        
        results = await processor.process_batch(images, analyze_image)
        
        assert len(results) == 10
        assert all(r["score"] == 75.0 for r in results)
    
    @pytest.mark.asyncio
    async def test_batch_save_analyses(self):
        """Test batch saving analyses"""
        from core.infrastructure.batch_processor import BatchProcessor
        
        processor = BatchProcessor(batch_size=10)
        
        analyses = [
            {"id": f"analysis-{i}", "user_id": "user-123"}
            for i in range(20)
        ]
        
        saved_count = 0
        
        async def save_analysis(analysis):
            nonlocal saved_count
            saved_count += 1
            return True
        
        results = await processor.process_batch(analyses, save_analysis)
        
        assert len(results) == 20
        assert saved_count == 20



