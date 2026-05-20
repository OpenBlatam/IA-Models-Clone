"""
Model Monitor
Monitor model statistics and health
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ModelMonitor:
    """Monitor model statistics"""
    
    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
    
    def get_model_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get model statistics"""
        if not TORCH_AVAILABLE:
            return {}
        
        total_params = 0
        trainable_params = 0
        layer_stats = {}
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            
            layer_stats[name] = {
                "num_params": num_params,
                "shape": list(param.shape),
                "requires_grad": param.requires_grad,
                "mean": param.mean().item() if param.numel() > 0 else 0.0,
                "std": param.std().item() if param.numel() > 0 else 0.0,
                "min": param.min().item() if param.numel() > 0 else 0.0,
                "max": param.max().item() if param.numel() > 0 else 0.0
            }
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "layers": layer_stats
        }
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Get model size in MB"""
        if not TORCH_AVAILABLE:
            return {}
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            "model_size_mb": size_all_mb,
            "param_size_mb": param_size / 1024 / 1024,
            "buffer_size_mb": buffer_size / 1024 / 1024
        }
    
    def take_snapshot(self, model: nn.Module, step: int = 0) -> Dict[str, Any]:
        """Take model snapshot"""
        snapshot = {
            "step": step,
            "stats": self.get_model_stats(model),
            "size": self.get_model_size(model)
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all snapshots"""
        return self.snapshots.copy()
    
    def clear_snapshots(self):
        """Clear snapshots"""
        self.snapshots.clear()



