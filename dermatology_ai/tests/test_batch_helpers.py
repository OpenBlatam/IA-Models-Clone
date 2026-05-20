"""
Batch Processing Testing Helpers
Specialized helpers for batch processing testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio


class BatchTestHelpers:
    """Helpers for batch processing testing"""
    
    @staticmethod
    def create_mock_batch_processor(
        batch_size: int = 10,
        processed_batches: Optional[List[List[Any]]] = None
    ) -> Mock:
        """Create mock batch processor"""
        batches = processed_batches or []
        processor = Mock()
        processor.batch_size = batch_size
        
        async def process_batch_side_effect(items: List[Any]):
            batches.append(items)
            return {"processed": len(items), "batch_id": f"batch-{len(batches)}"}
        
        processor.process_batch = AsyncMock(side_effect=process_batch_side_effect)
        processor.batches = batches
        return processor
    
    @staticmethod
    def assert_batch_processed(
        processor: Mock,
        expected_batch_size: Optional[int] = None
    ):
        """Assert batch was processed"""
        assert processor.process_batch.called, "Batch was not processed"
        
        if hasattr(processor, "batches") and expected_batch_size:
            assert len(processor.batches) > 0, "No batches were processed"
            assert len(processor.batches[0]) == expected_batch_size, \
                f"Batch size {len(processor.batches[0])} does not match expected {expected_batch_size}"


class ChunkHelpers:
    """Helpers for chunking testing"""
    
    @staticmethod
    def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Split list into chunks"""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    @staticmethod
    def assert_chunks_valid(
        chunks: List[List[Any]],
        expected_chunk_size: int,
        total_items: Optional[int] = None
    ):
        """Assert chunks are valid"""
        for chunk in chunks:
            assert len(chunk) <= expected_chunk_size, \
                f"Chunk size {len(chunk)} exceeds expected {expected_chunk_size}"
        
        if total_items:
            total_in_chunks = sum(len(chunk) for chunk in chunks)
            assert total_in_chunks == total_items, \
                f"Total items in chunks {total_in_chunks} does not match expected {total_items}"


class ParallelBatchHelpers:
    """Helpers for parallel batch processing testing"""
    
    @staticmethod
    async def process_batches_parallel(
        batches: List[List[Any]],
        processor: Callable,
        max_concurrent: int = 5
    ) -> List[Any]:
        """Process batches in parallel"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(batch):
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(batch)
                return processor(batch)
        
        return await asyncio.gather(*[process_with_limit(batch) for batch in batches])
    
    @staticmethod
    def assert_parallel_processing_complete(
        results: List[Any],
        expected_count: int
    ):
        """Assert parallel processing completed"""
        assert len(results) == expected_count, \
            f"Processed {len(results)} batches, expected {expected_count}"


# Convenience exports
create_mock_batch_processor = BatchTestHelpers.create_mock_batch_processor
assert_batch_processed = BatchTestHelpers.assert_batch_processed

chunk_list = ChunkHelpers.chunk_list
assert_chunks_valid = ChunkHelpers.assert_chunks_valid

process_batches_parallel = ParallelBatchHelpers.process_batches_parallel
assert_parallel_processing_complete = ParallelBatchHelpers.assert_parallel_processing_complete



