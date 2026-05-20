"""
Compression Testing Helpers
Specialized helpers for compression and decompression testing
"""

from typing import Any, Optional
from unittest.mock import Mock
import gzip
import zlib


class CompressionTestHelpers:
    """Helpers for compression testing"""
    
    @staticmethod
    def create_mock_compressor(
        compression_ratio: float = 0.5
    ) -> Mock:
        """Create mock compressor"""
        compressor = Mock()
        compressor.compression_ratio = compression_ratio
        
        def compress_side_effect(data: bytes):
            # Simulate compression
            compressed_size = int(len(data) * compression_ratio)
            return b"compressed_" + data[:compressed_size]
        
        def decompress_side_effect(compressed_data: bytes):
            # Simulate decompression
            if compressed_data.startswith(b"compressed_"):
                return compressed_data[11:]  # Remove prefix
            return compressed_data
        
        compressor.compress = Mock(side_effect=compress_side_effect)
        compressor.decompress = Mock(side_effect=decompress_side_effect)
        return compressor
    
    @staticmethod
    def assert_compression_valid(
        original: bytes,
        compressed: bytes,
        min_ratio: float = 0.1
    ):
        """Assert compression is valid"""
        assert compressed is not None, "Compressed data is None"
        assert len(compressed) > 0, "Compressed data is empty"
        
        ratio = len(compressed) / len(original) if len(original) > 0 else 1.0
        assert ratio >= min_ratio, \
            f"Compression ratio {ratio} is below minimum {min_ratio}"


class GzipHelpers:
    """Helpers for gzip compression testing"""
    
    @staticmethod
    def compress_gzip(data: bytes) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data)
    
    @staticmethod
    def decompress_gzip(compressed_data: bytes) -> bytes:
        """Decompress gzip data"""
        return gzip.decompress(compressed_data)
    
    @staticmethod
    def assert_gzip_valid(
        original: bytes,
        compressed: bytes
    ):
        """Assert gzip compression/decompression is valid"""
        decompressed = GzipHelpers.decompress_gzip(compressed)
        assert decompressed == original, \
            "Decompressed data does not match original"


class ZlibHelpers:
    """Helpers for zlib compression testing"""
    
    @staticmethod
    def compress_zlib(data: bytes) -> bytes:
        """Compress data using zlib"""
        return zlib.compress(data)
    
    @staticmethod
    def decompress_zlib(compressed_data: bytes) -> bytes:
        """Decompress zlib data"""
        return zlib.decompress(compressed_data)
    
    @staticmethod
    def assert_zlib_valid(
        original: bytes,
        compressed: bytes
    ):
        """Assert zlib compression/decompression is valid"""
        decompressed = ZlibHelpers.decompress_zlib(compressed)
        assert decompressed == original, \
            "Decompressed data does not match original"


# Convenience exports
create_mock_compressor = CompressionTestHelpers.create_mock_compressor
assert_compression_valid = CompressionTestHelpers.assert_compression_valid

compress_gzip = GzipHelpers.compress_gzip
decompress_gzip = GzipHelpers.decompress_gzip
assert_gzip_valid = GzipHelpers.assert_gzip_valid

compress_zlib = ZlibHelpers.compress_zlib
decompress_zlib = ZlibHelpers.decompress_zlib
assert_zlib_valid = ZlibHelpers.assert_zlib_valid



