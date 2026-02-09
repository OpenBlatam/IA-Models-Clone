"""
Modular Checkpoint Validator
Validates checkpoint integrity and completeness
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class CheckpointValidator:
    """Validates checkpoint integrity"""
    
    REQUIRED_KEYS = ["model_state_dict", "epoch"]
    OPTIONAL_KEYS = ["optimizer_state_dict", "scheduler_state_dict", "metrics", "metadata"]
    
    def validate(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Validate checkpoint
        
        Args:
            checkpoint: Checkpoint dictionary
        
        Returns:
            True if valid, False otherwise
        """
        # Check required keys
        for key in self.REQUIRED_KEYS:
            if key not in checkpoint:
                logger.error(f"Missing required key in checkpoint: {key}")
                return False
        
        # Validate model state dict
        if not isinstance(checkpoint["model_state_dict"], dict):
            logger.error("model_state_dict must be a dictionary")
            return False
        
        # Validate epoch
        if not isinstance(checkpoint["epoch"], int):
            logger.error("epoch must be an integer")
            return False
        
        # Check for NaN/Inf in model weights
        if self._check_model_weights(checkpoint["model_state_dict"]):
            logger.warning("NaN/Inf detected in model weights")
            return False
        
        return True
    
    def _check_model_weights(self, state_dict: Dict[str, Any]) -> bool:
        """Check for NaN/Inf in model weights"""
        if not TORCH_AVAILABLE:
            return False
        
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    logger.warning(f"NaN/Inf in weight: {key}")
                    return True
        
        return False
    
    def get_checkpoint_info(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about checkpoint
        
        Args:
            checkpoint: Checkpoint dictionary
        
        Returns:
            Dictionary with checkpoint information
        """
        info = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "has_optimizer": "optimizer_state_dict" in checkpoint,
            "has_scheduler": "scheduler_state_dict" in checkpoint,
            "has_metrics": "metrics" in checkpoint,
            "has_metadata": "metadata" in checkpoint,
            "num_model_params": len(checkpoint.get("model_state_dict", {})),
            "timestamp": checkpoint.get("timestamp", "unknown")
        }
        
        if "metrics" in checkpoint:
            info["metrics"] = checkpoint["metrics"]
        
        return info



