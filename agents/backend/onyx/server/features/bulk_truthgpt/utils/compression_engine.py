"""
Compression Engine
=================

Advanced compression system for optimizing storage and network usage.
"""

import asyncio
import logging
import time
import gzip
import bz2
import lzma
import zlib
import pickle
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class CompressionAlgorithm(str, Enum):
    """Compression algorithms."""
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    BROTLI = "brotli"

class CompressionLevel(str, Enum):
    """Compression levels."""
    FAST = "fast"
    BALANCED = "balanced"
    MAXIMUM = "maximum"

@dataclass
class CompressionResult:
    """Compression result."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: CompressionAlgorithm
    level: CompressionLevel
    compression_time: float
    decompression_time: float

@dataclass
class CompressionConfig:
    """Compression configuration."""
    default_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    default_level: CompressionLevel = CompressionLevel.BALANCED
    enable_auto_selection: bool = True
    min_compression_ratio: float = 0.1
    max_compression_time: float = 5.0
    enable_parallel: bool = True
    max_workers: int = 4

class CompressionEngine:
    """
    Advanced compression engine.
    
    Features:
    - Multiple compression algorithms
    - Automatic algorithm selection
    - Parallel compression
    - Compression ratio optimization
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.algorithms = {
            CompressionAlgorithm.GZIP: self._compress_gzip,
            CompressionAlgorithm.BZIP2: self._compress_bzip2,
            CompressionAlgorithm.LZMA: self._compress_lzma,
            CompressionAlgorithm.ZLIB: self._compress_zlib,
        }
        self.decompressors = {
            CompressionAlgorithm.GZIP: self._decompress_gzip,
            CompressionAlgorithm.BZIP2: self._decompress_bzip2,
            CompressionAlgorithm.LZMA: self._decompress_lzma,
            CompressionAlgorithm.ZLIB: self._decompress_zlib,
        }
        self.levels = {
            CompressionLevel.FAST: 1,
            CompressionLevel.BALANCED: 6,
            CompressionLevel.MAXIMUM: 9
        }
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_time_saved': 0.0,
            'average_ratio': 0.0,
            'algorithm_usage': {alg.value: 0 for alg in CompressionAlgorithm}
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
    async def initialize(self):
        """Initialize compression engine."""
        logger.info("Initializing Compression Engine...")
        
        try:
            # Test all algorithms
            await self._test_algorithms()
            
            logger.info("Compression Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Compression Engine: {str(e)}")
            raise
    
    async def _test_algorithms(self):
        """Test all compression algorithms."""
        test_data = b"Test data for compression algorithm testing. " * 100
        
        for algorithm in CompressionAlgorithm:
            try:
                # Test compression
                compressed = await self.compress(test_data, algorithm, CompressionLevel.BALANCED)
                
                # Test decompression
                decompressed = await self.decompress(compressed, algorithm)
                
                if decompressed == test_data:
                    logger.info(f"Algorithm {algorithm.value} working correctly")
                else:
                    logger.warning(f"Algorithm {algorithm.value} failed test")
                    
            except Exception as e:
                logger.warning(f"Algorithm {algorithm.value} not available: {str(e)}")
    
    async def compress(self, 
                     data: Union[bytes, str, Any], 
                     algorithm: Optional[CompressionAlgorithm] = None,
                     level: Optional[CompressionLevel] = None) -> bytes:
        """Compress data using specified algorithm."""
        try:
            start_time = time.time()
            
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, (dict, list)):
                data_bytes = json.dumps(data).encode('utf-8')
            else:
                data_bytes = pickle.dumps(data)
            
            # Select algorithm
            if algorithm is None:
                algorithm = self._select_best_algorithm(data_bytes)
            
            # Select level
            if level is None:
                level = self.config.default_level
            
            # Compress data
            if self.config.enable_parallel and len(data_bytes) > 1024 * 1024:  # 1MB
                compressed = await self._compress_parallel(data_bytes, algorithm, level)
            else:
                compressed = await self._compress_sync(data_bytes, algorithm, level)
            
            # Calculate metrics
            compression_time = time.time() - start_time
            compression_ratio = len(compressed) / len(data_bytes)
            
            # Update stats
            self.stats['total_compressions'] += 1
            self.stats['algorithm_usage'][algorithm.value] += 1
            self.stats['total_time_saved'] += (len(data_bytes) - len(compressed))
            
            # Calculate average ratio
            if self.stats['total_compressions'] > 0:
                self.stats['average_ratio'] = (
                    (self.stats['average_ratio'] * (self.stats['total_compressions'] - 1) + compression_ratio) /
                    self.stats['total_compressions']
                )
            
            logger.debug(f"Compressed {len(data_bytes)} bytes to {len(compressed)} bytes "
                        f"({compression_ratio:.2f} ratio) in {compression_time:.3f}s")
            
            return compressed
            
        except Exception as e:
            logger.error(f"Failed to compress data: {str(e)}")
            raise
    
    async def decompress(self, 
                        compressed_data: bytes, 
                        algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data using specified algorithm."""
        try:
            start_time = time.time()
            
            # Decompress data
            if algorithm in self.decompressors:
                decompressed = await self.decompressors[algorithm](compressed_data)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Update stats
            self.stats['total_decompressions'] += 1
            
            decompression_time = time.time() - start_time
            logger.debug(f"Decompressed {len(compressed_data)} bytes to {len(decompressed)} bytes "
                        f"in {decompression_time:.3f}s")
            
            return decompressed
            
        except Exception as e:
            logger.error(f"Failed to decompress data: {str(e)}")
            raise
    
    def _select_best_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """Select best compression algorithm for data."""
        if not self.config.enable_auto_selection:
            return self.config.default_algorithm
        
        # Simple heuristic based on data characteristics
        if len(data) < 1024:  # Small data
            return CompressionAlgorithm.ZLIB
        elif len(data) < 1024 * 1024:  # Medium data
            return CompressionAlgorithm.GZIP
        else:  # Large data
            return CompressionAlgorithm.LZMA
    
    async def _compress_sync(self, 
                           data: bytes, 
                           algorithm: CompressionAlgorithm, 
                           level: CompressionLevel) -> bytes:
        """Synchronous compression."""
        compression_level = self.levels[level]
        
        if algorithm == CompressionAlgorithm.GZIP:
            return await self._compress_gzip(data, compression_level)
        elif algorithm == CompressionAlgorithm.BZIP2:
            return await self._compress_bzip2(data, compression_level)
        elif algorithm == CompressionAlgorithm.LZMA:
            return await self._compress_lzma(data, compression_level)
        elif algorithm == CompressionAlgorithm.ZLIB:
            return await self._compress_zlib(data, compression_level)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    async def _compress_parallel(self, 
                               data: bytes, 
                               algorithm: CompressionAlgorithm, 
                               level: CompressionLevel) -> bytes:
        """Parallel compression for large data."""
        # Split data into chunks
        chunk_size = len(data) // self.config.max_workers
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Compress chunks in parallel
        loop = asyncio.get_event_loop()
        tasks = []
        
        for chunk in chunks:
            task = loop.run_in_executor(
                self.thread_pool,
                self._compress_chunk,
                chunk,
                algorithm,
                level
            )
            tasks.append(task)
        
        compressed_chunks = await asyncio.gather(*tasks)
        
        # Combine compressed chunks
        result = b''.join(compressed_chunks)
        
        return result
    
    def _compress_chunk(self, chunk: bytes, algorithm: CompressionAlgorithm, level: CompressionLevel) -> bytes:
        """Compress a single chunk."""
        compression_level = self.levels[level]
        
        if algorithm == CompressionAlgorithm.GZIP:
            return gzip.compress(chunk, compresslevel=compression_level)
        elif algorithm == CompressionAlgorithm.BZIP2:
            return bz2.compress(chunk, compresslevel=compression_level)
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.compress(chunk, preset=compression_level)
        elif algorithm == CompressionAlgorithm.ZLIB:
            return zlib.compress(chunk, level=compression_level)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    async def _compress_gzip(self, data: bytes, level: int = 6) -> bytes:
        """Compress using GZIP."""
        return gzip.compress(data, compresslevel=level)
    
    async def _compress_bzip2(self, data: bytes, level: int = 6) -> bytes:
        """Compress using BZIP2."""
        return bz2.compress(data, compresslevel=level)
    
    async def _compress_lzma(self, data: bytes, level: int = 6) -> bytes:
        """Compress using LZMA."""
        return lzma.compress(data, preset=level)
    
    async def _compress_zlib(self, data: bytes, level: int = 6) -> bytes:
        """Compress using ZLIB."""
        return zlib.compress(data, level=level)
    
    async def _decompress_gzip(self, data: bytes) -> bytes:
        """Decompress using GZIP."""
        return gzip.decompress(data)
    
    async def _decompress_bzip2(self, data: bytes) -> bytes:
        """Decompress using BZIP2."""
        return bz2.decompress(data)
    
    async def _decompress_lzma(self, data: bytes) -> bytes:
        """Decompress using LZMA."""
        return lzma.decompress(data)
    
    async def _decompress_zlib(self, data: bytes) -> bytes:
        """Decompress using ZLIB."""
        return zlib.decompress(data)
    
    async def compress_file(self, 
                           file_path: str, 
                           output_path: Optional[str] = None,
                           algorithm: Optional[CompressionAlgorithm] = None) -> str:
        """Compress a file."""
        try:
            # Read file
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Compress data
            compressed = await self.compress(data, algorithm)
            
            # Write compressed file
            if output_path is None:
                output_path = f"{file_path}.{algorithm.value if algorithm else 'gz'}"
            
            with open(output_path, 'wb') as f:
                f.write(compressed)
            
            logger.info(f"Compressed file {file_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to compress file {file_path}: {str(e)}")
            raise
    
    async def decompress_file(self, 
                             file_path: str, 
                             output_path: Optional[str] = None,
                             algorithm: Optional[CompressionAlgorithm] = None) -> str:
        """Decompress a file."""
        try:
            # Read compressed file
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            # Detect algorithm from file extension if not specified
            if algorithm is None:
                if file_path.endswith('.gz'):
                    algorithm = CompressionAlgorithm.GZIP
                elif file_path.endswith('.bz2'):
                    algorithm = CompressionAlgorithm.BZIP2
                elif file_path.endswith('.xz'):
                    algorithm = CompressionAlgorithm.LZMA
                else:
                    algorithm = CompressionAlgorithm.GZIP
            
            # Decompress data
            decompressed = await self.decompress(compressed_data, algorithm)
            
            # Write decompressed file
            if output_path is None:
                output_path = file_path.rsplit('.', 1)[0]
            
            with open(output_path, 'wb') as f:
                f.write(decompressed)
            
            logger.info(f"Decompressed file {file_path} to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to decompress file {file_path}: {str(e)}")
            raise
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            'total_compressions': self.stats['total_compressions'],
            'total_decompressions': self.stats['total_decompressions'],
            'total_time_saved': self.stats['total_time_saved'],
            'average_ratio': self.stats['average_ratio'],
            'algorithm_usage': self.stats['algorithm_usage'],
            'config': {
                'default_algorithm': self.config.default_algorithm.value,
                'default_level': self.config.default_level.value,
                'auto_selection': self.config.enable_auto_selection,
                'parallel_enabled': self.config.enable_parallel,
                'max_workers': self.config.max_workers
            }
        }
    
    async def benchmark_algorithms(self, test_data: bytes) -> Dict[str, CompressionResult]:
        """Benchmark all compression algorithms."""
        results = {}
        
        for algorithm in CompressionAlgorithm:
            try:
                # Test compression
                start_time = time.time()
                compressed = await self.compress(test_data, algorithm, CompressionLevel.BALANCED)
                compression_time = time.time() - start_time
                
                # Test decompression
                start_time = time.time()
                decompressed = await self.decompress(compressed, algorithm)
                decompression_time = time.time() - start_time
                
                # Verify correctness
                if decompressed == test_data:
                    results[algorithm.value] = CompressionResult(
                        original_size=len(test_data),
                        compressed_size=len(compressed),
                        compression_ratio=len(compressed) / len(test_data),
                        algorithm=algorithm,
                        level=CompressionLevel.BALANCED,
                        compression_time=compression_time,
                        decompression_time=decompression_time
                    )
                else:
                    logger.warning(f"Algorithm {algorithm.value} failed verification")
                    
            except Exception as e:
                logger.warning(f"Algorithm {algorithm.value} benchmark failed: {str(e)}")
        
        return results
    
    async def cleanup(self):
        """Cleanup compression engine."""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("Compression Engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Compression Engine: {str(e)}")

# Global compression engine
compression_engine = CompressionEngine()

# Decorators for compression
def compress_data(algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP, 
                 level: CompressionLevel = CompressionLevel.BALANCED):
    """Decorator to compress function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            return await compression_engine.compress(result, algorithm, level)
        
        return wrapper
    return decorator

def decompress_data(algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP):
    """Decorator to decompress function arguments."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Decompress first argument if it's bytes
            if args and isinstance(args[0], bytes):
                decompressed = await compression_engine.decompress(args[0], algorithm)
                args = (decompressed,) + args[1:]
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator











