"""
Inference Debugger
Debugging utilities for inference
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class InferenceDebugger:
    """Debugging utilities for inference"""
    
    def __init__(self, track_activations: bool = False):
        self.track_activations = track_activations
        self.activation_hooks = []
        self.activations = {}
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks to track activations"""
        if not self.track_activations:
            return
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = {
                        "shape": list(output.shape),
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                        "has_nan": torch.isnan(output).any().item(),
                        "has_inf": torch.isinf(output).any().item()
                    }
            return hook
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hook = module.register_forward_hook(make_hook(name))
                self.activation_hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks.clear()
    
    def check_output(
        self,
        output: torch.Tensor,
        name: str = "output"
    ) -> Dict[str, Any]:
        """
        Check output statistics
        
        Args:
            output: Model output
            name: Output name
        
        Returns:
            Dictionary of output statistics
        """
        if not TORCH_AVAILABLE:
            return {}
        
        stats = {
            "name": name,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "mean": output.mean().item() if output.numel() > 0 else 0.0,
            "std": output.std().item() if output.numel() > 0 else 0.0,
            "min": output.min().item() if output.numel() > 0 else 0.0,
            "max": output.max().item() if output.numel() > 0 else 0.0,
            "has_nan": torch.isnan(output).any().item(),
            "has_inf": torch.isinf(output).any().item()
        }
        
        if stats["has_nan"]:
            logger.warning(f"NaN detected in {name}")
        if stats["has_inf"]:
            logger.warning(f"Inf detected in {name}")
        
        return stats
    
    def get_activations(self) -> Dict[str, Dict[str, Any]]:
        """Get tracked activations"""
        return self.activations.copy()
    
    def clear_activations(self):
        """Clear activation history"""
        self.activations.clear()



