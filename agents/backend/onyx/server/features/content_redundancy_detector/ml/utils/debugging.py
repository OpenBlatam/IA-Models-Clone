"""
Debugging Utilities
PyTorch debugging tools and utilities
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Debugger:
    """
    Debugging utilities for PyTorch models
    """
    
    @staticmethod
    @contextmanager
    def detect_anomaly():
        """
        Context manager for detecting anomalies in autograd
        
        Usage:
            with Debugger.detect_anomaly():
                loss.backward()
        """
        torch.autograd.set_detect_anomaly(True)
        try:
            yield
        finally:
            torch.autograd.set_detect_anomaly(False)
    
    @staticmethod
    def check_gradients(
        model: nn.Module,
        check_nan: bool = True,
        check_inf: bool = True,
        check_zero: bool = False,
    ) -> Dict[str, Any]:
        """
        Check model gradients for issues
        
        Args:
            model: Model to check
            check_nan: Check for NaN gradients
            check_inf: Check for Inf gradients
            check_zero: Check for zero gradients
            
        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            'total_params': 0,
            'params_with_grad': 0,
            'nan_grads': [],
            'inf_grads': [],
            'zero_grads': [],
            'grad_norms': [],
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                stats['params_with_grad'] += 1
                grad = param.grad.data
                
                if check_nan and torch.isnan(grad).any():
                    stats['nan_grads'].append(name)
                
                if check_inf and torch.isinf(grad).any():
                    stats['inf_grads'].append(name)
                
                if check_zero and (grad == 0).all():
                    stats['zero_grads'].append(name)
                
                # Calculate gradient norm
                grad_norm = grad.norm().item()
                stats['grad_norms'].append({
                    'name': name,
                    'norm': grad_norm
                })
            
            stats['total_params'] += 1
        
        return stats
    
    @staticmethod
    def check_weights(
        model: nn.Module,
        check_nan: bool = True,
        check_inf: bool = True,
    ) -> Dict[str, Any]:
        """
        Check model weights for issues
        
        Args:
            model: Model to check
            check_nan: Check for NaN weights
            check_inf: Check for Inf weights
            
        Returns:
            Dictionary with weight statistics
        """
        stats = {
            'total_params': 0,
            'nan_weights': [],
            'inf_weights': [],
            'weight_stats': [],
        }
        
        for name, param in model.named_parameters():
            stats['total_params'] += 1
            weight = param.data
            
            if check_nan and torch.isnan(weight).any():
                stats['nan_weights'].append(name)
            
            if check_inf and torch.isinf(weight).any():
                stats['inf_weights'].append(name)
            
            stats['weight_stats'].append({
                'name': name,
                'mean': float(weight.mean().item()),
                'std': float(weight.std().item()),
                'min': float(weight.min().item()),
                'max': float(weight.max().item()),
            })
        
        return stats
    
    @staticmethod
    def log_model_info(model: nn.Module) -> None:
        """
        Log detailed model information
        
        Args:
            model: Model to log
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Log layer information
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    logger.debug(f"  {name}: {num_params:,} parameters")


class TrainingDebugger:
    """
    Debugging utilities for training
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize training debugger
        
        Args:
            enabled: Whether debugging is enabled
        """
        self.enabled = enabled
        self.debugger = Debugger()
    
    def check_training_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        """
        Check training step for issues
        
        Args:
            model: Model
            loss: Loss value
            optimizer: Optimizer
            
        Returns:
            Dictionary with debug information
        """
        if not self.enabled:
            return {}
        
        debug_info = {}
        
        # Check loss
        if torch.isnan(loss):
            debug_info['loss_nan'] = True
            logger.warning("Loss is NaN!")
        
        if torch.isinf(loss):
            debug_info['loss_inf'] = True
            logger.warning("Loss is Inf!")
        
        # Check gradients
        grad_stats = self.debugger.check_gradients(model)
        if grad_stats['nan_grads']:
            debug_info['nan_gradients'] = grad_stats['nan_grads']
            logger.warning(f"NaN gradients in: {grad_stats['nan_grads']}")
        
        if grad_stats['inf_grads']:
            debug_info['inf_gradients'] = grad_stats['inf_grads']
            logger.warning(f"Inf gradients in: {grad_stats['inf_grads']}")
        
        # Check learning rate
        current_lr = optimizer.param_groups[0]['lr']
        debug_info['learning_rate'] = current_lr
        
        return debug_info



