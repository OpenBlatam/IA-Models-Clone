"""
Compressor

Utilities for data and model compression.
"""

import logging
import gzip
import pickle
from typing import Any, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Try to import lz4
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logger.warning("lz4 not available")


class Compressor:
    """Compress and decompress data."""
    
    def __init__(self, algorithm: str = "gzip"):
        """
        Initialize compressor.
        
        Args:
            algorithm: Compression algorithm ('gzip', 'lz4')
        """
        self.algorithm = algorithm
    
    def compress(
        self,
        data: Any,
        file_path: Optional[str] = None
    ) -> bytes:
        """
        Compress data.
        
        Args:
            data: Data to compress
            file_path: Optional file path to save
            
        Returns:
            Compressed bytes
        """
        # Serialize data
        serialized = pickle.dumps(data)
        
        # Compress
        if self.algorithm == "gzip":
            compressed = gzip.compress(serialized)
        elif self.algorithm == "lz4" and LZ4_AVAILABLE:
            compressed = lz4.frame.compress(serialized)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Save if path provided
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(compressed)
            logger.info(f"Compressed data saved: {file_path}")
        
        return compressed
    
    def decompress(
        self,
        compressed_data: bytes,
        file_path: Optional[str] = None
    ) -> Any:
        """
        Decompress data.
        
        Args:
            compressed_data: Compressed bytes (or None if file_path provided)
            file_path: Optional file path to load from
            
        Returns:
            Decompressed data
        """
        # Load from file if provided
        if file_path:
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
        
        # Decompress
        if self.algorithm == "gzip":
            serialized = gzip.decompress(compressed_data)
        elif self.algorithm == "lz4" and LZ4_AVAILABLE:
            serialized = lz4.frame.decompress(compressed_data)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Deserialize
        data = pickle.loads(serialized)
        
        return data
    
    def compress_model(
        self,
        model: torch.nn.Module,
        file_path: str
    ) -> str:
        """
        Compress model state dict.
        
        Args:
            model: Model to compress
            file_path: Output file path
            
        Returns:
            File path
        """
        state_dict = model.state_dict()
        compressed = self.compress(state_dict, file_path)
        logger.info(f"Model compressed: {file_path}")
        return file_path
    
    def decompress_model(
        self,
        model: torch.nn.Module,
        file_path: str
    ) -> torch.nn.Module:
        """
        Decompress and load model state dict.
        
        Args:
            model: Model to load into
            file_path: Compressed file path
            
        Returns:
            Model with loaded state
        """
        state_dict = self.decompress(None, file_path)
        model.load_state_dict(state_dict)
        logger.info(f"Model decompressed: {file_path}")
        return model


def compress_data(
    data: Any,
    file_path: Optional[str] = None,
    algorithm: str = "gzip"
) -> bytes:
    """Compress data."""
    compressor = Compressor(algorithm)
    return compressor.compress(data, file_path)


def decompress_data(
    compressed_data: Optional[bytes] = None,
    file_path: Optional[str] = None,
    algorithm: str = "gzip"
) -> Any:
    """Decompress data."""
    compressor = Compressor(algorithm)
    return compressor.decompress(compressed_data, file_path)


def compress_model(
    model: torch.nn.Module,
    file_path: str,
    algorithm: str = "gzip"
) -> str:
    """Compress model."""
    compressor = Compressor(algorithm)
    return compressor.compress_model(model, file_path)



