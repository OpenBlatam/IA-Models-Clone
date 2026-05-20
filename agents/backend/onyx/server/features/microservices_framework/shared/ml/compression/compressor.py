"""
Compression Utilities
Compression and decompression for models and data.
"""

import pickle
import gzip
import zlib
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class Compressor:
    """Compress and decompress data."""
    
    @staticmethod
    def compress_pickle(obj: Any, level: int = 6) -> bytes:
        """Compress object using pickle and gzip."""
        pickled = pickle.dumps(obj)
        compressed = gzip.compress(pickled, compresslevel=level)
        return compressed
    
    @staticmethod
    def decompress_pickle(data: bytes) -> Any:
        """Decompress object from gzip and pickle."""
        decompressed = gzip.decompress(data)
        obj = pickle.loads(decompressed)
        return obj
    
    @staticmethod
    def compress_string(text: str, encoding: str = "utf-8") -> bytes:
        """Compress string."""
        return zlib.compress(text.encode(encoding))
    
    @staticmethod
    def decompress_string(data: bytes, encoding: str = "utf-8") -> str:
        """Decompress string."""
        return zlib.decompress(data).decode(encoding)
    
    @staticmethod
    def compress_bytes(data: bytes, level: int = 6) -> bytes:
        """Compress bytes."""
        return zlib.compress(data, level=level)
    
    @staticmethod
    def decompress_bytes(data: bytes) -> bytes:
        """Decompress bytes."""
        return zlib.decompress(data)


class ModelCompressor:
    """Compress model weights and state."""
    
    @staticmethod
    def compress_state_dict(state_dict: dict, level: int = 6) -> bytes:
        """Compress model state dict."""
        return Compressor.compress_pickle(state_dict, level=level)
    
    @staticmethod
    def decompress_state_dict(data: bytes) -> dict:
        """Decompress model state dict."""
        return Compressor.decompress_pickle(data)
    
    @staticmethod
    def get_compression_ratio(original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio."""
        if len(original) == 0:
            return 0.0
        return len(compressed) / len(original)



