"""
Advanced Debugging Tools
========================
Debugging utilities for deep learning models
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
from contextlib import contextmanager
import numpy as np

logger = structlog.get_logger()


class ModelDebugger:
    """
    Advanced debugging tools for model training and inference
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize debugger
        
        Args:
            enabled: Enable debugging
        """
        self.enabled = enabled
        self.debug_info = {}
        
        logger.info("ModelDebugger initialized", enabled=enabled)
    
    @contextmanager
    def detect_anomalies(self):
        """
        Context manager for detecting anomalies in gradients
        
        Usage:
            with debugger.detect_anomalies():
                loss.backward()
        """
        if self.enabled:
            torch.autograd.set_detect_anomaly(True)
            try:
                yield
            finally:
                torch.autograd.set_detect_anomaly(False)
        else:
            yield
    
    def check_gradients(
        self,
        model: nn.Module,
        check_nan: bool = True,
        check_inf: bool = True,
        check_exploding: bool = True,
        max_norm: float = 10.0
    ) -> Dict[str, Any]:
        """
        Check gradients for common issues
        
        Args:
            model: Model to check
            check_nan: Check for NaN values
            check_inf: Check for Inf values
            check_exploding: Check for exploding gradients
            max_norm: Maximum gradient norm
            
        Returns:
            Gradient statistics
        """
        stats = {
            "has_nan": False,
            "has_inf": False,
            "has_exploding": False,
            "grad_norms": {},
            "total_params": 0,
            "params_with_grad": 0
        }
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats["params_with_grad"] += 1
                
                # Check NaN
                if check_nan and torch.isnan(grad).any():
                    stats["has_nan"] = True
                    logger.warning(f"NaN gradient detected in {name}")
                
                # Check Inf
                if check_inf and torch.isinf(grad).any():
                    stats["has_inf"] = True
                    logger.warning(f"Inf gradient detected in {name}")
                
                # Calculate norm
                param_norm = grad.norm(2)
                total_norm += param_norm.item() ** 2
                stats["grad_norms"][name] = param_norm.item()
                
                param_count += grad.numel()
            
            stats["total_params"] += param.numel()
        
        total_norm = total_norm ** (1. / 2)
        stats["total_grad_norm"] = total_norm
        
        # Check exploding gradients
        if check_exploding and total_norm > max_norm:
            stats["has_exploding"] = True
            logger.warning(f"Exploding gradient detected: norm={total_norm:.2f}")
        
        return stats
    
    def check_weights(
        self,
        model: nn.Module,
        check_nan: bool = True,
        check_inf: bool = True
    ) -> Dict[str, Any]:
        """
        Check model weights for issues
        
        Args:
            model: Model to check
            check_nan: Check for NaN values
            check_inf: Check for Inf values
            
        Returns:
            Weight statistics
        """
        stats = {
            "has_nan": False,
            "has_inf": False,
            "weight_stats": {}
        }
        
        for name, param in model.named_parameters():
            weight = param.data
            
            if check_nan and torch.isnan(weight).any():
                stats["has_nan"] = True
                logger.warning(f"NaN weight detected in {name}")
            
            if check_inf and torch.isinf(weight).any():
                stats["has_inf"] = True
                logger.warning(f"Inf weight detected in {name}")
            
            stats["weight_stats"][name] = {
                "mean": float(weight.mean().item()),
                "std": float(weight.std().item()),
                "min": float(weight.min().item()),
                "max": float(weight.max().item())
            }
        
        return stats
    
    def log_training_step(
        self,
        step: int,
        loss: torch.Tensor,
        learning_rate: float,
        model: nn.Module
    ) -> None:
        """
        Log detailed training step information
        
        Args:
            step: Training step
            loss: Loss value
            learning_rate: Current learning rate
            model: Model
        """
        if not self.enabled:
            return
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Invalid loss at step {step}: {loss.item()}")
        
        # Check gradients
        grad_stats = self.check_gradients(model)
        
        # Store debug info
        self.debug_info[f"step_{step}"] = {
            "loss": float(loss.item()),
            "learning_rate": learning_rate,
            "gradient_stats": grad_stats
        }
        
        # Log warnings
        if grad_stats["has_nan"] or grad_stats["has_inf"] or grad_stats["has_exploding"]:
            logger.warning(
                f"Gradient issues at step {step}",
                **grad_stats
            )


class DataDebugger:
    """Debugging tools for data loading"""
    
    @staticmethod
    def validate_batch(
        batch: Dict[str, Any],
        expected_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Validate data batch
        
        Args:
            batch: Data batch
            expected_keys: Expected keys in batch
            
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "missing_keys": [],
            "invalid_shapes": [],
            "has_nan": False,
            "has_inf": False
        }
        
        # Check for expected keys
        for key in expected_keys:
            if key not in batch:
                results["missing_keys"].append(key)
                results["valid"] = False
        
        # Check tensor values
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    results["has_nan"] = True
                    results["valid"] = False
                    logger.warning(f"NaN values in batch key: {key}")
                
                if torch.isinf(value).any():
                    results["has_inf"] = True
                    results["valid"] = False
                    logger.warning(f"Inf values in batch key: {key}")
        
        return results


# Global debugger instance
model_debugger = ModelDebugger()




