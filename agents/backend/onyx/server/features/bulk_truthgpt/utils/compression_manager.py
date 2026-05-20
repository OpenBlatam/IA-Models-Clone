"""
Compression Manager
===================

Advanced compression with multiple algorithms and automatic selection.
"""

import gzip
import lzma
import bz2
import zlib
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import time

logger = logging.getLogger(__name__)

class CompressionAlgorithm(str, Enum):
    """Compression algorithms."""
    GZIP = "gzip"
    LZMA = "lzma"
    BZ2 = "bz2"
    ZLIB = "zlib"
    NONE = "none"

class CompressionManager:
    """Advanced compression manager with algorithm selection."""
    
    def __init__(self):
        self.compression_stats: Dict[str, Dict[str, Any]] = {}
        self.algorithm_performance: Dict[str, List[float]] = {}
    
    def compress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
        level: int = 6
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data with specified algorithm."""
        start_time = time.time()
        original_size = len(data)
        
        try:
            if algorithm == CompressionAlgorithm.GZIP:
                compressed = gzip.compress(data, compresslevel=level)
            elif algorithm == CompressionAlgorithm.LZMA:
                compressed = lzma.compress(data, preset=level)
            elif algorithm == CompressionAlgorithm.BZ2:
                compressed = bz2.compress(data, compresslevel=level)
            elif algorithm == CompressionAlgorithm.ZLIB:
                compressed = zlib.compress(data, level=level)
            else:
                compressed = data
            
            compression_time = time.time() - start_time
            compressed_size = len(compressed)
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            stats = {
                "algorithm": algorithm.value,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": round(ratio, 2),
                "compression_time_ms": round(compression_time * 1000, 2),
                "speed_mb_per_sec": round((original_size / (1024 * 1024)) / compression_time, 2) if compression_time > 0 else 0
            }
            
            # Track performance
            if algorithm.value not in self.algorithm_performance:
                self.algorithm_performance[algorithm.value] = []
            self.algorithm_performance[algorithm.value].append(compression_time)
            
            return compressed, stats
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    def decompress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    ) -> bytes:
        """Decompress data."""
        try:
            if algorithm == CompressionAlgorithm.GZIP:
                return gzip.decompress(data)
            elif algorithm == CompressionAlgorithm.LZMA:
                return lzma.decompress(data)
            elif algorithm == CompressionAlgorithm.BZ2:
                return bz2.decompress(data)
            elif algorithm == CompressionAlgorithm.ZLIB:
                return zlib.decompress(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    def select_best_algorithm(self, data: bytes, test_size: int = 1024) -> CompressionAlgorithm:
        """Select best algorithm based on performance."""
        if len(data) < test_size:
            test_data = data
        else:
            test_data = data[:test_size]
        
        best_algorithm = CompressionAlgorithm.GZIP
        best_ratio = 0
        
        for algo in [CompressionAlgorithm.GZIP, CompressionAlgorithm.LZMA, CompressionAlgorithm.BZ2, CompressionAlgorithm.ZLIB]:
            try:
                compressed, stats = self.compress(test_data, algorithm=algo)
                if stats["compression_ratio"] > best_ratio:
                    best_ratio = stats["compression_ratio"]
                    best_algorithm = algo
            except:
                continue
        
        return best_algorithm
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        algo_stats = {}
        for algo, times in self.algorithm_performance.items():
            if times:
                algo_stats[algo] = {
                    "count": len(times),
                    "avg_time_ms": round(sum(times) / len(times) * 1000, 2),
                    "min_time_ms": round(min(times) * 1000, 2),
                    "max_time_ms": round(max(times) * 1000, 2)
                }
        
        return {
            "algorithms": algo_stats,
            "total_compressions": sum(len(times) for times in self.algorithm_performance.values())
        }

# Global instance
compression_manager = CompressionManager()
































