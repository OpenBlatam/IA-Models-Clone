"""
Data Compression - Compresión de Datos
=======================================

Sistema avanzado de compresión de datos con múltiples algoritmos y análisis de eficiencia.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import gzip
import zlib
import bz2
import lzma

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Algoritmo de compresión."""
    GZIP = "gzip"
    ZLIB = "zlib"
    BZ2 = "bz2"
    LZMA = "lzma"
    NONE = "none"


@dataclass
class CompressionResult:
    """Resultado de compresión."""
    algorithm: CompressionAlgorithm
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataCompression:
    """Sistema de compresión de datos."""
    
    def __init__(self):
        self.compression_history: List[CompressionResult] = []
        self.algorithm_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_compressions": 0,
            "total_bytes_saved": 0,
            "avg_ratio": 0.0,
        })
        self._lock = asyncio.Lock()
    
    def compress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
    ) -> bytes:
        """Comprimir datos."""
        start_time = datetime.now()
        
        if algorithm == CompressionAlgorithm.GZIP:
            compressed = gzip.compress(data)
        elif algorithm == CompressionAlgorithm.ZLIB:
            compressed = zlib.compress(data)
        elif algorithm == CompressionAlgorithm.BZ2:
            compressed = bz2.compress(data)
        elif algorithm == CompressionAlgorithm.LZMA:
            compressed = lzma.compress(data)
        else:
            compressed = data
        
        compression_time = (datetime.now() - start_time).total_seconds()
        
        original_size = len(data)
        compressed_size = len(compressed)
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        result = CompressionResult(
            algorithm=algorithm,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
        )
        
        async def update_stats():
            async with self._lock:
                self.compression_history.append(result)
                if len(self.compression_history) > 10000:
                    self.compression_history.pop(0)
                
                stats = self.algorithm_stats[algorithm.value]
                stats["total_compressions"] += 1
                stats["total_bytes_saved"] += (original_size - compressed_size)
                
                # Actualizar ratio promedio
                ratios = [r.compression_ratio for r in self.compression_history if r.algorithm == algorithm]
                if ratios:
                    stats["avg_ratio"] = sum(ratios) / len(ratios)
        
        asyncio.create_task(update_stats())
        
        logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes using {algorithm.value}")
        return compressed
    
    def decompress(
        self,
        compressed_data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
    ) -> bytes:
        """Descomprimir datos."""
        start_time = datetime.now()
        
        if algorithm == CompressionAlgorithm.GZIP:
            data = gzip.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.ZLIB:
            data = zlib.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.BZ2:
            data = bz2.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.LZMA:
            data = lzma.decompress(compressed_data)
        else:
            data = compressed_data
        
        decompression_time = (datetime.now() - start_time).total_seconds()
        
        logger.debug(f"Decompressed {len(compressed_data)} bytes to {len(data)} bytes using {algorithm.value}")
        return data
    
    def find_best_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """Encontrar mejor algoritmo para datos."""
        best_algorithm = CompressionAlgorithm.NONE
        best_ratio = 1.0
        
        for algorithm in [CompressionAlgorithm.GZIP, CompressionAlgorithm.ZLIB, CompressionAlgorithm.BZ2, CompressionAlgorithm.LZMA]:
            try:
                compressed = self.compress(data, algorithm)
                ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_algorithm = algorithm
            except Exception as e:
                logger.warning(f"Error compressing with {algorithm.value}: {e}")
        
        return best_algorithm
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de compresión."""
        total_original = sum(r.original_size for r in self.compression_history)
        total_compressed = sum(r.compressed_size for r in self.compression_history)
        total_saved = total_original - total_compressed
        
        return {
            "total_compressions": len(self.compression_history),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "total_bytes_saved": total_saved,
            "overall_compression_ratio": total_compressed / total_original if total_original > 0 else 0.0,
            "algorithm_stats": dict(self.algorithm_stats),
        }















