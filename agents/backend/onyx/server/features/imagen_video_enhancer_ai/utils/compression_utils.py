"""
Compression Utilities
=====================

Advanced compression utilities.
"""

import gzip
import zlib
import bz2
import lzma
import logging
from typing import Optional, Union, bytes
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression type."""
    GZIP = "gzip"
    ZLIB = "zlib"
    BZ2 = "bz2"
    LZMA = "lzma"
    NONE = "none"


class CompressionUtils:
    """Compression utility functions."""
    
    @staticmethod
    def compress(data: bytes, compression_type: CompressionType = CompressionType.GZIP) -> bytes:
        """
        Compress data.
        
        Args:
            data: Data to compress
            compression_type: Compression type
            
        Returns:
            Compressed data
        """
        if compression_type == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression_type == CompressionType.ZLIB:
            return zlib.compress(data)
        elif compression_type == CompressionType.BZ2:
            return bz2.compress(data)
        elif compression_type == CompressionType.LZMA:
            return lzma.compress(data)
        else:
            return data
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType = CompressionType.GZIP) -> bytes:
        """
        Decompress data.
        
        Args:
            data: Compressed data
            compression_type: Compression type
            
        Returns:
            Decompressed data
        """
        if compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression_type == CompressionType.BZ2:
            return bz2.decompress(data)
        elif compression_type == CompressionType.LZMA:
            return lzma.decompress(data)
        else:
            return data
    
    @staticmethod
    def compress_file(input_path: str, output_path: str, compression_type: CompressionType = CompressionType.GZIP):
        """
        Compress file.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            compression_type: Compression type
        """
        with open(input_path, "rb") as f_in:
            data = f_in.read()
        
        compressed = CompressionUtils.compress(data, compression_type)
        
        with open(output_path, "wb") as f_out:
            f_out.write(compressed)
        
        logger.info(f"Compressed {input_path} to {output_path}")
    
    @staticmethod
    def decompress_file(input_path: str, output_path: str, compression_type: CompressionType = CompressionType.GZIP):
        """
        Decompress file.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            compression_type: Compression type
        """
        with open(input_path, "rb") as f_in:
            data = f_in.read()
        
        decompressed = CompressionUtils.decompress(data, compression_type)
        
        with open(output_path, "wb") as f_out:
            f_out.write(decompressed)
        
        logger.info(f"Decompressed {input_path} to {output_path}")
    
    @staticmethod
    def get_compression_ratio(original: bytes, compressed: bytes) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original: Original data size
            compressed: Compressed data size
            
        Returns:
            Compression ratio
        """
        if len(original) == 0:
            return 0.0
        return len(compressed) / len(original)




