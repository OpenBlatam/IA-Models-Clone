"""
Security Utilities
Security and safety checks
"""

import torch
import hashlib
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SecurityChecker:
    """
    Security and safety checks
    """
    
    @staticmethod
    def verify_checkpoint(
        checkpoint_path: Path,
        expected_hash: Optional[str] = None,
    ) -> bool:
        """
        Verify checkpoint file integrity
        
        Args:
            checkpoint_path: Path to checkpoint
            expected_hash: Expected hash (optional)
            
        Returns:
            True if valid
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        # Compute hash
        sha256_hash = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        file_hash = sha256_hash.hexdigest()
        
        if expected_hash and file_hash != expected_hash:
            logger.error(f"Hash mismatch for {checkpoint_path}")
            return False
        
        # Try to load
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' not in checkpoint and 'state_dict' not in checkpoint:
                logger.warning(f"Checkpoint may be invalid: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for security
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        return filename
    
    @staticmethod
    def check_model_safety(model: torch.nn.Module) -> bool:
        """
        Check model for safety issues
        
        Args:
            model: Model to check
            
        Returns:
            True if safe
        """
        # Check for NaN/Inf in parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                logger.warning(f"NaN found in parameter: {name}")
                return False
            if torch.isinf(param).any():
                logger.warning(f"Inf found in parameter: {name}")
                return False
        
        return True



