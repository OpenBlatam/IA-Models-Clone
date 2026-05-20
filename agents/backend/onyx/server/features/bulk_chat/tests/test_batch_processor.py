"""
Tests for Batch Processor
==========================
"""

import pytest
import asyncio
from ..core.batch_processor import BatchProcessor, BatchStrategy


@pytest.fixture
def batch_processor():
    """Create batch processor for testing."""
    return BatchProcessor(max_batch_size=10, batch_timeout=5.0)


@pytest.mark.asyncio
async def test_add_to_batch(batch_processor):
    """Test adding item to batch."""
    item_id = batch_processor.add_to_batch(
        batch_id="test_batch",
        item={"data": "test"},
        priority=1
    )
    
    assert item_id is not None
    assert "test_batch" in batch_processor.batches


@pytest.mark.asyncio
async def test_process_batch(batch_processor):
    """Test processing a batch."""
    async def process_func(items):
        return [f"processed_{item['data']}" for item in items]
    
    batch_processor.add_to_batch("test_batch", {"data": "item1"})
    batch_processor.add_to_batch("test_batch", {"data": "item2"})
    
    result = await batch_processor.process_batch(
        "test_batch",
        process_func,
        strategy=BatchStrategy.SIZE_BASED
    )
    
    assert result is not None
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_get_batch_status(batch_processor):
    """Test getting batch status."""
    batch_processor.add_to_batch("test_batch", {"data": "test"})
    
    status = batch_processor.get_batch_status("test_batch")
    
    assert status is not None
    assert "batch_id" in status or "size" in status


@pytest.mark.asyncio
async def test_get_batch_processor_summary(batch_processor):
    """Test getting batch processor summary."""
    batch_processor.add_to_batch("batch1", {"data": "test"})
    batch_processor.add_to_batch("batch2", {"data": "test"})
    
    summary = batch_processor.get_batch_processor_summary()
    
    assert summary is not None
    assert "total_batches" in summary or "total_items" in summary


