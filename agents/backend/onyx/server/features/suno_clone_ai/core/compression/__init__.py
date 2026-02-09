"""
Compression Module

Provides:
- Data compression utilities
- Model compression
- Compression algorithms
"""

from .compressor import (
    Compressor,
    compress_data,
    decompress_data,
    compress_model
)

from .algorithms import (
    gzip_compress,
    gzip_decompress,
    lz4_compress,
    lz4_decompress
)

__all__ = [
    # Compression
    "Compressor",
    "compress_data",
    "decompress_data",
    "compress_model",
    # Algorithms
    "gzip_compress",
    "gzip_decompress",
    "lz4_compress",
    "lz4_decompress"
]



