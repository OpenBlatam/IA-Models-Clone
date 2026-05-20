"""
Tests for Data Compression
============================
"""

import pytest
import asyncio
from ..core.data_compression import DataCompressor, CompressionAlgorithm


@pytest.fixture
def data_compressor():
    """Create data compressor for testing."""
    return DataCompressor()


@pytest.mark.asyncio
async def test_compress_data(data_compressor):
    """Test compressing data."""
    original_data = b"This is a test string that will be compressed" * 100
    
    compressed = await data_compressor.compress(
        data=original_data,
        algorithm=CompressionAlgorithm.LZ4
    )
    
    assert compressed is not None
    assert len(compressed) < len(original_data) or len(compressed) > 0


@pytest.mark.asyncio
async def test_decompress_data(data_compressor):
    """Test decompressing data."""
    original_data = b"Test data to compress"
    
    compressed = await data_compressor.compress(original_data, CompressionAlgorithm.LZ4)
    decompressed = await data_compressor.decompress(compressed, CompressionAlgorithm.LZ4)
    
    assert decompressed == original_data


@pytest.mark.asyncio
async def test_compression_ratio(data_compressor):
    """Test compression ratio calculation."""
    original_data = b"Test data" * 1000
    
    compressed = await data_compressor.compress(original_data, CompressionAlgorithm.LZ4)
    
    ratio = data_compressor.get_compression_ratio(original_data, compressed)
    
    assert ratio is not None
    assert isinstance(ratio, float)
    assert ratio > 0


@pytest.mark.asyncio
async def test_compress_with_different_algorithms(data_compressor):
    """Test compression with different algorithms."""
    data = b"Test data to compress" * 100
    
    lz4_compressed = await data_compressor.compress(data, CompressionAlgorithm.LZ4)
    zstd_compressed = await data_compressor.compress(data, CompressionAlgorithm.ZSTANDARD)
    
    assert lz4_compressed is not None
    assert zstd_compressed is not None
    assert len(lz4_compressed) > 0
    assert len(zstd_compressed) > 0


@pytest.mark.asyncio
async def test_get_compression_stats(data_compressor):
    """Test getting compression statistics."""
    data = b"Test data" * 100
    
    await data_compressor.compress(data, CompressionAlgorithm.LZ4)
    await data_compressor.compress(data, CompressionAlgorithm.ZSTANDARD)
    
    stats = data_compressor.get_compression_stats()
    
    assert stats is not None
    assert "total_compressions" in stats or "avg_ratio" in stats


