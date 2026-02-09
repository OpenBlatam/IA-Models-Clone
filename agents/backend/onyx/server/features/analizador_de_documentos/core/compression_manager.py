"""
Compression Manager for Document Analyzer
==========================================

Advanced compression and decompression utilities.
"""

import logging
import gzip
import zlib
import bz2
import lzma
from typing import Optional, Union, bytes
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class CompressionAlgorithm(Enum):
    """Compression algorithms"""
    GZIP = "gzip"
    ZLIB = "zlib"
    BZ2 = "bz2"
    LZMA = "lzma"
    NONE = "none"

class CompressionManager:
    """Advanced compression manager"""
    
    def __init__(self):
        logger.info("CompressionManager initialized")
    
    def compress(
        self,
        data: Union[str, bytes],
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    ) -> bytes:
        """Compress data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == CompressionAlgorithm.GZIP:
            return gzip.compress(data)
        elif algorithm == CompressionAlgorithm.ZLIB:
            return zlib.compress(data)
        elif algorithm == CompressionAlgorithm.BZ2:
            return bz2.compress(data)
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.compress(data)
        elif algorithm == CompressionAlgorithm.NONE:
            return data
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def decompress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    ) -> bytes:
        """Decompress data"""
        if algorithm == CompressionAlgorithm.GZIP:
            return gzip.decompress(data)
        elif algorithm == CompressionAlgorithm.ZLIB:
            return zlib.decompress(data)
        elif algorithm == CompressionAlgorithm.BZ2:
            return bz2.decompress(data)
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.decompress(data)
        elif algorithm == CompressionAlgorithm.NONE:
            return data
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def compress_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    ) -> Path:
        """Compress a file"""
        input_path = Path(input_path)
        
        if output_path is None:
            if algorithm == CompressionAlgorithm.GZIP:
                output_path = input_path.with_suffix(input_path.suffix + '.gz')
            elif algorithm == CompressionAlgorithm.BZ2:
                output_path = input_path.with_suffix(input_path.suffix + '.bz2')
            elif algorithm == CompressionAlgorithm.LZMA:
                output_path = input_path.with_suffix(input_path.suffix + '.xz')
            else:
                output_path = input_path.with_suffix(input_path.suffix + '.compressed')
        else:
            output_path = Path(output_path)
        
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        
        compressed = self.compress(data, algorithm)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(compressed)
        
        logger.info(f"Compressed {input_path} to {output_path} ({len(data)} -> {len(compressed)} bytes)")
        
        return output_path
    
    def decompress_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    ) -> Path:
        """Decompress a file"""
        input_path = Path(input_path)
        
        if output_path is None:
            # Remove compression extension
            output_path = input_path.with_suffix('')
            if output_path.suffix in ['.gz', '.bz2', '.xz']:
                output_path = output_path.with_suffix('')
        else:
            output_path = Path(output_path)
        
        with open(input_path, 'rb') as f_in:
            compressed = f_in.read()
        
        decompressed = self.decompress(compressed, algorithm)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(decompressed)
        
        logger.info(f"Decompressed {input_path} to {output_path}")
        
        return output_path
    
    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio"""
        if len(original) == 0:
            return 0.0
        return len(compressed) / len(original)

# Global instance
compression_manager = CompressionManager()

