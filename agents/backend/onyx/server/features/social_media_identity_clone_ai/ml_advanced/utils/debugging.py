"""
Herramientas de debugging para deep learning
"""

import torch
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DebuggingTools:
    """Herramientas de debugging"""
    
    @staticmethod
    @contextmanager
    def detect_anomalies(enabled: bool = True):
        """Detecta anomalías en autograd"""
        if enabled and torch.cuda.is_available():
            torch.autograd.set_detect_anomaly(True)
            try:
                yield
            finally:
                torch.autograd.set_detect_anomaly(False)
        else:
            yield
    
    @staticmethod
    def check_gradients(model: torch.nn.Module) -> Dict[str, Any]:
        """Verifica gradientes del modelo"""
        grad_info = {
            "has_gradients": False,
            "grad_norms": {},
            "zero_grads": [],
            "inf_grads": [],
            "nan_grads": []
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_info["has_gradients"] = True
                grad_norm = param.grad.norm().item()
                grad_info["grad_norms"][name] = grad_norm
                
                if grad_norm == 0:
                    grad_info["zero_grads"].append(name)
                elif torch.isinf(param.grad).any():
                    grad_info["inf_grads"].append(name)
                elif torch.isnan(param.grad).any():
                    grad_info["nan_grads"].append(name)
        
        return grad_info
    
    @staticmethod
    def check_weights(model: torch.nn.Module) -> Dict[str, Any]:
        """Verifica pesos del modelo"""
        weight_info = {
            "has_nan": False,
            "has_inf": False,
            "nan_layers": [],
            "inf_layers": [],
            "weight_stats": {}
        }
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                weight_info["has_nan"] = True
                weight_info["nan_layers"].append(name)
            
            if torch.isinf(param).any():
                weight_info["has_inf"] = True
                weight_info["inf_layers"].append(name)
            
            weight_info["weight_stats"][name] = {
                "mean": float(param.mean().item()),
                "std": float(param.std().item()),
                "min": float(param.min().item()),
                "max": float(param.max().item())
            }
        
        return weight_info
    
    @staticmethod
    def log_model_info(model: torch.nn.Module):
        """Log información del modelo"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Memoria
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"GPU Memory: {memory_mb:.2f} MB")


@contextmanager
def enable_debugging(enabled: bool = True):
    """Context manager para debugging"""
    if enabled:
        with DebuggingTools.detect_anomalies():
            yield
    else:
        yield




