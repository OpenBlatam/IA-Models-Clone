"""
Gradient Debugger
Specialized debugging for gradients
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class GradientDebugger:
    """Specialized gradient debugging"""
    
    def __init__(self):
        self.gradient_history: List[Dict[str, Any]] = []
    
    def analyze_gradients(
        self,
        model: nn.Module,
        step: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze gradient flow
        
        Args:
            model: Model to analyze
            step: Current step
        
        Returns:
            Gradient analysis
        """
        if not TORCH_AVAILABLE:
            return {}
        
        analysis = {
            "step": step,
            "layers": {},
            "total_norm": 0.0,
            "vanishing_gradients": [],
            "exploding_gradients": []
        }
        
        total_norm_sq = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_norm_sq += grad_norm ** 2
                
                analysis["layers"][name] = {
                    "norm": grad_norm,
                    "mean": param.grad.mean().item(),
                    "std": param.grad.std().item(),
                    "max": param.grad.abs().max().item(),
                    "min": param.grad.abs().min().item()
                }
                
                # Check for vanishing gradients
                if grad_norm < 1e-6:
                    analysis["vanishing_gradients"].append(name)
                
                # Check for exploding gradients
                if grad_norm > 100.0:
                    analysis["exploding_gradients"].append(name)
        
        analysis["total_norm"] = (total_norm_sq ** 0.5)
        
        # Log warnings
        if analysis["vanishing_gradients"]:
            logger.warning(f"Vanishing gradients in: {analysis['vanishing_gradients']}")
        if analysis["exploding_gradients"]:
            logger.warning(f"Exploding gradients in: {analysis['exploding_gradients']}")
        
        self.gradient_history.append(analysis)
        return analysis
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get gradient history"""
        return self.gradient_history.copy()
    
    def clear_history(self):
        """Clear gradient history"""
        self.gradient_history.clear()



