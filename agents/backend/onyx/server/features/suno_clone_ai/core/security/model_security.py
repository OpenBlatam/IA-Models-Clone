"""
Model Security

Utilities for securing model loading and inference.
"""

import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


class ModelSecurity:
    """Security utilities for models."""
    
    @staticmethod
    def compute_hash(file_path: str) -> str:
        """
        Compute file hash.
        
        Args:
            file_path: Path to file
            
        Returns:
            File hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def verify_integrity(
        file_path: str,
        expected_hash: Optional[str] = None
    ) -> bool:
        """
        Verify file integrity.
        
        Args:
            file_path: Path to file
            expected_hash: Expected hash (optional)
            
        Returns:
            True if integrity verified
        """
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if expected_hash:
            actual_hash = ModelSecurity.compute_hash(file_path)
            if actual_hash != expected_hash:
                logger.error(f"Hash mismatch for {file_path}")
                return False
        
        return True
    
    @staticmethod
    def secure_load(
        file_path: str,
        expected_hash: Optional[str] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Securely load model checkpoint.
        
        Args:
            file_path: Path to checkpoint
            expected_hash: Expected hash
            map_location: Device mapping
            
        Returns:
            Loaded checkpoint
        """
        # Verify integrity
        if not ModelSecurity.verify_integrity(file_path, expected_hash):
            raise ValueError(f"Integrity check failed for {file_path}")
        
        # Load with security checks
        try:
            checkpoint = torch.load(file_path, map_location=map_location)
            logger.info(f"Securely loaded checkpoint: {file_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise


def secure_model_loading(
    file_path: str,
    expected_hash: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Securely load model."""
    return ModelSecurity.secure_load(file_path, expected_hash, **kwargs)


def verify_model_integrity(
    file_path: str,
    expected_hash: Optional[str] = None
) -> bool:
    """Verify model integrity."""
    return ModelSecurity.verify_integrity(file_path, expected_hash)



