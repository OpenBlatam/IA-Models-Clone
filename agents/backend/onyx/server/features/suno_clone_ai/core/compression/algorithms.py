"""
Compression Algorithms

Individual compression algorithm functions.
"""

import logging
import gzip
import pickle
from typing import Any

logger = logging.getLogger(__name__)

# Try to import lz4
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logger.warning("lz4 not available")


def gzip_compress(data: Any) -> bytes:
    """
    Compress data with gzip.
    
    Args:
        data: Data to compress
        
    Returns:
        Compressed bytes
    """
    serialized = pickle.dumps(data)
    return gzip.compress(serialized)


def gzip_decompress(compressed_data: bytes) -> Any:
    """
    Decompress gzip data.
    
    Args:
        compressed_data: Compressed bytes
        
    Returns:
        Decompressed data
    """
    serialized = gzip.decompress(compressed_data)
    return pickle.loads(serialized)


def lz4_compress(data: Any) -> bytes:
    """
    Compress data with lz4.
    
    Args:
        data: Data to compress
        
    Returns:
        Compressed bytes
    """
    if not LZ4_AVAILABLE:
        raise ImportError("lz4 not available")
    
    serialized = pickle.dumps(data)
    return lz4.frame.compress(serialized)


def lz4_decompress(compressed_data: bytes) -> Any:
    """
    Decompress lz4 data.
    
    Args:
        compressed_data: Compressed bytes
        
    Returns:
        Decompressed data
    """
    if not LZ4_AVAILABLE:
        raise ImportError("lz4 not available")
    
    serialized = lz4.frame.decompress(compressed_data)
    return pickle.loads(serialized)



