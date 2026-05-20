"""
Advanced compression system v2 for KV cache.

This module provides additional compression techniques and optimizations.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import zlib
import gzip
import bz2
import lzma
try:
    import snappy
    SNAPPY_AVAILABLE = True
except ImportError:
    SNAPPY_AVAILABLE = False


class CompressionAlgorithm(Enum):
    """Compression algorithms."""
    ZLIB = "zlib"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"
    SNAPPY = "snappy"
    LZ4 = "lz4"  # Would need lz4 library
    AUTO = "auto"  # Automatically select best


@dataclass
class CompressionResult:
    """Result of compression."""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: CompressionAlgorithm
    compression_time: float


@dataclass
class CompressionStats:
    """Compression statistics."""
    total_compressions: int
    total_decompressions: int
    total_bytes_saved: int
    average_compression_ratio: float
    compression_time_total: float
    decompression_time_total: float


class AdvancedCompressor:
    """Advanced compression system."""
    
    def __init__(
        self,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO,
        compression_level: int = 6
    ):
        self.algorithm = algorithm
        self.compression_level = compression_level
        self._stats = CompressionStats(
            total_compressions=0,
            total_decompressions=0,
            total_bytes_saved=0,
            average_compression_ratio=0.0,
            compression_time_total=0.0,
            decompression_time_total=0.0
        )
        self._lock = threading.Lock()
        
    def compress(self, data: bytes) -> CompressionResult:
        """Compress data."""
        start_time = time.time()
        original_size = len(data)
        
        if self.algorithm == CompressionAlgorithm.AUTO:
            # Try different algorithms and pick best
            result = self._auto_select_algorithm(data)
        else:
            compressed_data = self._compress_with_algorithm(data, self.algorithm)
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 0.0
            
            result = CompressionResult(
                compressed_data=compressed_data,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                algorithm=self.algorithm,
                compression_time=time.time() - start_time
            )
            
        # Update stats
        with self._lock:
            self._stats.total_compressions += 1
            self._stats.total_bytes_saved += (original_size - result.compressed_size)
            # Update average compression ratio
            total_ratio = self._stats.average_compression_ratio * (self._stats.total_compressions - 1)
            self._stats.average_compression_ratio = (total_ratio + result.compression_ratio) / self._stats.total_compressions
            self._stats.compression_time_total += result.compression_time
            
        return result
        
    def decompress(self, compressed_data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data."""
        start_time = time.time()
        
        data = self._decompress_with_algorithm(compressed_data, algorithm)
        
        decompression_time = time.time() - start_time
        
        with self._lock:
            self._stats.total_decompressions += 1
            self._stats.decompression_time_total += decompression_time
            
        return data
        
    def _compress_with_algorithm(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Compress with specific algorithm."""
        if algorithm == CompressionAlgorithm.ZLIB:
            return zlib.compress(data, self.compression_level)
        elif algorithm == CompressionAlgorithm.GZIP:
            return gzip.compress(data, compresslevel=self.compression_level)
        elif algorithm == CompressionAlgorithm.BZ2:
            return bz2.compress(data, compresslevel=self.compression_level)
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.compress(data, preset=self.compression_level)
        elif algorithm == CompressionAlgorithm.SNAPPY:
            if SNAPPY_AVAILABLE:
                return snappy.compress(data)
            else:
                raise ValueError("Snappy not available")
        else:
            # Default to zlib
            return zlib.compress(data, self.compression_level)
            
    def _decompress_with_algorithm(self, compressed_data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress with specific algorithm."""
        if algorithm == CompressionAlgorithm.ZLIB:
            return zlib.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.GZIP:
            return gzip.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.BZ2:
            return bz2.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.SNAPPY:
            if SNAPPY_AVAILABLE:
                return snappy.decompress(compressed_data)
            else:
                raise ValueError("Snappy not available")
        else:
            # Default to zlib
            return zlib.decompress(compressed_data)
            
    def _auto_select_algorithm(self, data: bytes) -> CompressionResult:
        """Automatically select best compression algorithm."""
        algorithms = [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.BZ2,
            CompressionAlgorithm.LZMA
        ]
        
        if SNAPPY_AVAILABLE:
            algorithms.append(CompressionAlgorithm.SNAPPY)
            
        best_result = None
        best_ratio = 1.0
        
        for algorithm in algorithms:
            try:
                compressed = self._compress_with_algorithm(data, algorithm)
                ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_result = CompressionResult(
                        compressed_data=compressed,
                        original_size=len(data),
                        compressed_size=len(compressed),
                        compression_ratio=ratio,
                        algorithm=algorithm,
                        compression_time=0.0  # Simplified
                    )
            except Exception:
                continue
                
        if best_result is None:
            # Fallback to no compression
            best_result = CompressionResult(
                compressed_data=data,
                original_size=len(data),
                compressed_size=len(data),
                compression_ratio=1.0,
                algorithm=CompressionAlgorithm.ZLIB,
                compression_time=0.0
            )
            
        return best_result
        
    def get_stats(self) -> CompressionStats:
        """Get compression statistics."""
        with self._lock:
            return CompressionStats(
                total_compressions=self._stats.total_compressions,
                total_decompressions=self._stats.total_decompressions,
                total_bytes_saved=self._stats.total_bytes_saved,
                average_compression_ratio=self._stats.average_compression_ratio,
                compression_time_total=self._stats.compression_time_total,
                decompression_time_total=self._stats.decompression_time_total
            )


class CompressedCache:
    """Cache wrapper with compression."""
    
    def __init__(
        self,
        cache: Any,
        compressor: Optional[AdvancedCompressor] = None,
        compress_threshold: int = 1024  # Only compress if > 1KB
    ):
        self.cache = cache
        self.compressor = compressor or AdvancedCompressor()
        self.compress_threshold = compress_threshold
        
    def get(self, key: str) -> Any:
        """Get and decompress value."""
        value = self.cache.get(key)
        if value is None:
            return None
            
        # Check if value is compressed (has metadata)
        if isinstance(value, dict) and '_compressed' in value:
            compressed_data = value['data']
            algorithm = CompressionAlgorithm(value['algorithm'])
            decompressed = self.compressor.decompress(compressed_data, algorithm)
            return decompressed.decode('utf-8') if isinstance(decompressed, bytes) else decompressed
        else:
            return value
            
    def put(self, key: str, value: Any) -> bool:
        """Compress and put value."""
        # Convert value to bytes if needed
        if isinstance(value, str):
            data_bytes = value.encode('utf-8')
        elif isinstance(value, bytes):
            data_bytes = value
        else:
            data_bytes = str(value).encode('utf-8')
            
        # Only compress if above threshold
        if len(data_bytes) > self.compress_threshold:
            result = self.compressor.compress(data_bytes)
            
            # Store with metadata
            compressed_value = {
                '_compressed': True,
                'data': result.compressed_data,
                'algorithm': result.algorithm.value,
                'original_size': result.original_size
            }
            
            return self.cache.put(key, compressed_value)
        else:
            return self.cache.put(key, value)
            
    def delete(self, key: str) -> bool:
        """Delete value."""
        return self.cache.delete(key)
        
    def get_compression_stats(self) -> CompressionStats:
        """Get compression statistics."""
        return self.compressor.get_stats()














