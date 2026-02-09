"""
Model Security
===============
Security utilities for models
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import hashlib
import structlog
from pathlib import Path
import json

logger = structlog.get_logger()


class ModelSecurity:
    """
    Model security manager
    """
    
    def __init__(self):
        """Initialize security manager"""
        logger.info("ModelSecurity initialized")
    
    def compute_model_hash(
        self,
        model: nn.Module,
        algorithm: str = "sha256"
    ) -> str:
        """
        Compute hash of model weights
        
        Args:
            model: Model
            algorithm: Hash algorithm
            
        Returns:
            Hash string
        """
        try:
            # Get model state dict
            state_dict = model.state_dict()
            
            # Serialize to bytes
            import pickle
            model_bytes = pickle.dumps(state_dict)
            
            # Compute hash
            if algorithm == "sha256":
                hash_obj = hashlib.sha256(model_bytes)
            elif algorithm == "md5":
                hash_obj = hashlib.md5(model_bytes)
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error("Error computing model hash", error=str(e))
            raise
    
    def verify_model_integrity(
        self,
        model: nn.Module,
        expected_hash: str,
        algorithm: str = "sha256"
    ) -> bool:
        """
        Verify model integrity
        
        Args:
            model: Model to verify
            expected_hash: Expected hash
            algorithm: Hash algorithm
            
        Returns:
            True if integrity verified
        """
        try:
            actual_hash = self.compute_model_hash(model, algorithm)
            return actual_hash == expected_hash
        except Exception as e:
            logger.error("Error verifying model integrity", error=str(e))
            return False
    
    def sign_model(
        self,
        model: nn.Module,
        signature_key: str
    ) -> str:
        """
        Sign model with key
        
        Args:
            model: Model
            signature_key: Signature key
            
        Returns:
            Signature
        """
        try:
            model_hash = self.compute_model_hash(model)
            signature = hashlib.sha256(f"{model_hash}{signature_key}".encode()).hexdigest()
            return signature
        except Exception as e:
            logger.error("Error signing model", error=str(e))
            raise
    
    def validate_model_signature(
        self,
        model: nn.Module,
        signature: str,
        signature_key: str
    ) -> bool:
        """
        Validate model signature
        
        Args:
            model: Model
            signature: Expected signature
            signature_key: Signature key
            
        Returns:
            True if signature valid
        """
        try:
            expected_signature = self.sign_model(model, signature_key)
            return expected_signature == signature
        except Exception as e:
            logger.error("Error validating signature", error=str(e))
            return False


class ModelSanitizer:
    """
    Model sanitization utilities
    """
    
    def __init__(self):
        """Initialize sanitizer"""
        logger.info("ModelSanitizer initialized")
    
    def sanitize_model_weights(
        self,
        model: nn.Module,
        remove_nan: bool = True,
        remove_inf: bool = True,
        clip_values: Optional[tuple] = None
    ) -> nn.Module:
        """
        Sanitize model weights
        
        Args:
            model: Model
            remove_nan: Remove NaN values
            remove_inf: Remove Inf values
            clip_values: Clip values to range (min, max)
            
        Returns:
            Sanitized model
        """
        try:
            with torch.no_grad():
                for param in model.parameters():
                    if remove_nan:
                        param.data = torch.where(
                            torch.isnan(param.data),
                            torch.zeros_like(param.data),
                            param.data
                        )
                    
                    if remove_inf:
                        param.data = torch.where(
                            torch.isinf(param.data),
                            torch.zeros_like(param.data),
                            param.data
                        )
                    
                    if clip_values is not None:
                        param.data = torch.clamp(param.data, clip_values[0], clip_values[1])
            
            logger.info("Model weights sanitized")
            return model
        except Exception as e:
            logger.error("Error sanitizing model weights", error=str(e))
            raise


# Global instances
model_security = ModelSecurity()
model_sanitizer = ModelSanitizer()




