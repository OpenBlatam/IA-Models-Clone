from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import json
import pickle
from enum import Enum
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from typing import Any, List, Dict, Optional
import asyncio

# Import our centralized logging configuration
from logging_config import (
    get_logger, log_training_step, log_numerical_issue, 
    log_system_event, log_error_with_context, log_performance_metrics
)

# Import performance optimization system
try:
    from performance_optimization import (
        PerformanceConfig, create_performance_optimizer, 
        OptimizationLevel, MemoryFormat
    )
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False
    print("Warning: Performance optimization system not available")
#!/usr/bin/env python3
"""
Gradient Clipping and NaN/Inf Value Handling System
Comprehensive gradient clipping and numerical stability implementation.
"""

warnings.filterwarnings('ignore')


class ClippingType(Enum):
    """Types of gradient clipping."""
    NORM = "norm"  # L2 norm clipping
    VALUE = "value"  # Value clipping
    GLOBAL_NORM = "global_norm"  # Global norm clipping
    ADAPTIVE = "adaptive"  # Adaptive clipping
    LAYER_WISE = "layer_wise"  # Layer-wise clipping
    PERCENTILE = "percentile"  # Percentile-based clipping
    EXPONENTIAL = "exponential"  # Exponential moving average clipping


class NaNHandlingType(Enum):
    """Types of NaN/Inf handling."""
    DETECT = "detect"  # Detect and log
    REPLACE = "replace"  # Replace with safe values
    SKIP = "skip"  # Skip update
    RESTORE = "restore"  # Restore from checkpoint
    GRADIENT_ZEROING = "gradient_zeroing"  # Zero gradients
    ADAPTIVE = "adaptive"  # Adaptive handling based on severity
    GRADIENT_SCALING = "gradient_scaling"  # Scale gradients instead of zeroing


@dataclass
class GradientClippingConfig:
    """Configuration for gradient clipping."""
    # Clipping type
    clipping_type: ClippingType = ClippingType.NORM
    
    # Clipping parameters
    max_norm: float = 1.0
    max_value: float = 1.0
    clip_ratio: float = 0.1
    
    # Adaptive clipping
    adaptive_threshold: float = 0.1
    adaptive_factor: float = 2.0
    
    # Layer-wise clipping
    layer_wise_enabled: bool = False
    layer_norm_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Percentile clipping
    percentile_enabled: bool = False
    percentile_threshold: float = 95.0
    
    # Exponential clipping
    exponential_enabled: bool = False
    exponential_alpha: float = 0.9
    exponential_min_threshold: float = 0.1
    
    # Monitoring
    monitor_clipping: bool = True
    log_clipping_stats: bool = True
    save_clipping_history: bool = True
    clipping_history_file: str = "gradient_clipping_history.json"
    
    # Advanced settings
    clip_grad_norm: bool = True
    clip_grad_value: bool = False
    use_global_norm: bool = False
    
    # Performance
    efficient_clipping: bool = True
    parallel_clipping: bool = False


@dataclass
class NaNHandlingConfig:
    """Configuration for NaN/Inf handling."""
    # Handling type
    handling_type: NaNHandlingType = NaNHandlingType.DETECT
    
    # Detection settings
    detect_nan: bool = True
    detect_inf: bool = True
    detect_overflow: bool = True
    
    # Replacement values
    nan_replacement: float = 0.0
    inf_replacement: float = 1e6
    overflow_replacement: float = 1e6
    
    # Thresholds
    nan_threshold: float = 1e-6
    inf_threshold: float = 1e6
    overflow_threshold: float = 1e6
    
    # Monitoring
    monitor_nan: bool = True
    log_nan_stats: bool = True
    save_nan_history: bool = True
    nan_history_file: str = "nan_handling_history.json"
    
    # Advanced settings
    gradient_zeroing: bool = True
    parameter_checking: bool = True
    loss_checking: bool = True
    
    # Performance
    efficient_detection: bool = True
    batch_detection: bool = True


@dataclass
class PyTorchDebuggingConfig:
    """Configuration for PyTorch built-in debugging tools."""
    # Autograd anomaly detection
    enable_autograd_anomaly: bool = False
    autograd_anomaly_mode: str = "detect"  # "detect" or "raise"
    
    # Gradient checking
    enable_grad_check: bool = False
    grad_check_numerical: bool = True
    grad_check_sparse_numerical: bool = True
    
    # Memory debugging
    enable_memory_debugging: bool = False
    memory_tracking: bool = False
    memory_profiling: bool = False
    
    # CUDA debugging
    enable_cuda_debugging: bool = False
    cuda_synchronize: bool = False
    cuda_memory_fraction: float = 1.0
    
    # Performance debugging
    enable_performance_debugging: bool = False
    profile_autograd: bool = False
    profile_memory: bool = False
    
    # Debugging levels
    debug_level: str = "info"  # "info", "warning", "error"
    verbose_logging: bool = False
    
    # Safety settings
    max_debug_iterations: int = 1000
    debug_timeout: float = 300.0  # 5 minutes
    
    # Output settings
    save_debug_info: bool = True
    debug_output_dir: str = "debug_output"
    debug_file_prefix: str = "pytorch_debug"


@dataclass
class PerformanceOptimizationConfig:
    """Configuration for performance optimization integration."""
    # Enable performance optimization
    enable_performance_optimization: bool = True
    
    # Performance optimization level
    optimization_level: str = "advanced"  # "basic", "advanced", "ultra"
    
    # Multi-GPU training settings
    enable_multi_gpu: bool = True
    multi_gpu_mode: str = "auto"  # "none", "dataparallel", "distributed", "hybrid", "auto"
    multi_gpu_device_ids: Optional[List[int]] = None  # None = all available GPUs
    multi_gpu_backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    multi_gpu_find_unused_parameters: bool = False
    multi_gpu_bucket_cap_mb: int = 25
    multi_gpu_static_graph: bool = True
    
    # Integration settings
    integrate_with_stability: bool = True
    optimize_training_wrapper: bool = True
    enable_mixed_precision: bool = True
    enable_model_compilation: bool = True
    enable_memory_optimization: bool = True
    
    # Performance monitoring
    monitor_performance: bool = True
    log_performance_metrics: bool = True
    save_performance_history: bool = True


class GradientClipper:
    """Advanced gradient clipping implementation."""
    
    def __init__(self, config: GradientClippingConfig):
        """__init__ function."""
        self.config = config
        self.logger = self._setup_logging()
        
        # Clipping history
        self.clipping_history = {
            'steps': [],
            'clipping_ratios': [],
            'gradient_norms': [],
            'clipped_norms': [],
            'clipping_types': []
        }
        
        # Current step
        self.current_step = 0
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging using centralized logging configuration."""
        return get_logger('numerical_stability')
    
    def clip_gradients(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Clip gradients based on configuration."""
        start_time = time.time()
        self.current_step += 1
        
        # Log clipping operation start
        self.logger.info(f"Starting gradient clipping step {self.current_step} with type: {self.config.clipping_type.value}")
        
        try:
            result = None
            
            if self.config.clipping_type == ClippingType.NORM:
                result = self._clip_grad_norm(model, optimizer)
            
            elif self.config.clipping_type == ClippingType.VALUE:
                result = self._clip_grad_value(model, optimizer)
            
            elif self.config.clipping_type == ClippingType.GLOBAL_NORM:
                result = self._clip_global_norm(model, optimizer)
            
            elif self.config.clipping_type == ClippingType.ADAPTIVE:
                result = self._clip_adaptive(model, optimizer)
            
            elif self.config.clipping_type == ClippingType.LAYER_WISE:
                result = self._clip_layer_wise(model, optimizer)
            
            elif self.config.clipping_type == ClippingType.PERCENTILE:
                result = self._clip_percentile(model, optimizer)
            
            elif self.config.clipping_type == ClippingType.EXPONENTIAL:
                result = self._clip_exponential(model, optimizer)
            
            else:
                raise ValueError(f"Unknown clipping type: {self.config.clipping_type}")
            
            # Log successful clipping
            duration = time.time() - start_time
            self.logger.info(f"Gradient clipping step {self.current_step} completed successfully in {duration:.4f}s")
            
            # Log clipping performance metrics
            log_performance_metrics(
                self.logger,
                metrics={
                    "clipping_type": self.config.clipping_type.value,
                    "clipping_ratio": result.get('clipping_ratio', 0.0),
                    "gradient_norm": result.get('gradient_norm', 0.0),
                    "clipped_norm": result.get('clipped_norm', 0.0)
                },
                operation=f"gradient_clipping_{self.current_step}",
                duration=duration
            )
            
            return result
            
        except Exception as e:
            # Log error with context
            log_error_with_context(
                self.logger,
                error=e,
                operation=f"gradient_clipping_{self.current_step}",
                context={
                    "step": self.current_step,
                    "clipping_type": self.config.clipping_type.value,
                    "model_parameters": sum(p.numel() for p in model.parameters())
                },
                recovery_attempted=False
            )
            
            # Log system event for failed clipping
            log_system_event(
                self.logger,
                event_type="gradient_clipping_failed",
                description=f"Gradient clipping step {self.current_step} failed",
                details={"error": str(e), "clipping_type": self.config.clipping_type.value},
                level="error"
            )
            
            # Re-raise the exception
            raise
    
    def _clip_grad_norm(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Clip gradients by norm."""
        # Get all gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.view(-1))
        
        if not gradients:
            return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
        
        # Concatenate gradients
        all_gradients = torch.cat(gradients)
        
        # Calculate norm
        total_norm = all_gradients.norm(2)
        
        # Clip if necessary
        clip_coef = self.config.max_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        # Apply clipping
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Update history
        self._update_clipping_history(total_norm.item(), total_norm.item() * clip_coef, 'norm')
        
        return {
            'clipping_ratio': 1.0 - clip_coef,
            'gradient_norm': total_norm.item(),
            'clipped_norm': total_norm.item() * clip_coef
        }
    
    def _clip_grad_value(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Clip gradients by value."""
        total_norm = 0.0
        clipped_norm = 0.0
        clipped_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                # Calculate norm before clipping
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Clip gradients
                param.grad.data.clamp_(-self.config.max_value, self.config.max_value)
                
                # Calculate norm after clipping
                clipped_param_norm = param.grad.data.norm(2)
                clipped_norm += clipped_param_norm.item() ** 2
                
                if param_norm.item() > self.config.max_value:
                    clipped_count += 1
        
        total_norm = math.sqrt(total_norm)
        clipped_norm = math.sqrt(clipped_norm)
        
        # Update history
        self._update_clipping_history(total_norm, clipped_norm, 'value')
        
        return {
            'clipping_ratio': clipped_count / len(list(model.parameters())),
            'gradient_norm': total_norm,
            'clipped_norm': clipped_norm
        }
    
    def _clip_global_norm(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Clip gradients using global norm."""
        # Get all gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.view(-1))
        
        if not gradients:
            return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
        
        # Concatenate gradients
        all_gradients = torch.cat(gradients)
        
        # Calculate global norm
        global_norm = all_gradients.norm(2)
        
        # Clip if necessary
        clip_coef = self.config.max_norm / (global_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        # Apply clipping
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Update history
        self._update_clipping_history(global_norm.item(), global_norm.item() * clip_coef, 'global_norm')
        
        return {
            'clipping_ratio': 1.0 - clip_coef,
            'gradient_norm': global_norm.item(),
            'clipped_norm': global_norm.item() * clip_coef
        }
    
    def _clip_adaptive(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Adaptive gradient clipping."""
        # Get all gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.view(-1))
        
        if not gradients:
            return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
        
        # Concatenate gradients
        all_gradients = torch.cat(gradients)
        
        # Calculate norm
        total_norm = all_gradients.norm(2)
        
        # Adaptive threshold
        adaptive_threshold = self.config.adaptive_threshold
        if len(self.clipping_history['gradient_norms']) > 0:
            avg_norm = np.mean(self.clipping_history['gradient_norms'][-10:])
            adaptive_threshold = avg_norm * self.config.adaptive_factor
        
        # Clip if necessary
        clip_coef = adaptive_threshold / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        # Apply clipping
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Update history
        self._update_clipping_history(total_norm.item(), total_norm.item() * clip_coef, 'adaptive')
        
        return {
            'clipping_ratio': 1.0 - clip_coef,
            'gradient_norm': total_norm.item(),
            'clipped_norm': total_norm.item() * clip_coef,
            'adaptive_threshold': adaptive_threshold
        }
    
    def _clip_layer_wise(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Clip gradients layer by layer."""
        total_norm = 0.0
        clipped_norm = 0.0
        clipping_ratios = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Get layer-specific threshold or use default
                layer_threshold = self.config.layer_norm_thresholds.get(name, self.config.max_norm)
                
                # Calculate parameter norm
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Clip if necessary
                if param_norm.item() > layer_threshold:
                    clip_coef = layer_threshold / (param_norm.item() + 1e-6)
                    param.grad.data.mul_(clip_coef)
                    clipping_ratios.append(1.0 - clip_coef)
                else:
                    clipping_ratios.append(0.0)
                
                # Calculate clipped norm
                clipped_param_norm = param.grad.data.norm(2)
                clipped_norm += clipped_param_norm.item() ** 2
        
        total_norm = math.sqrt(total_norm)
        clipped_norm = math.sqrt(clipped_norm)
        avg_clipping_ratio = np.mean(clipping_ratios) if clipping_ratios else 0.0
        
        # Update history
        self._update_clipping_history(total_norm, clipped_norm, 'layer_wise')
        
        return {
            'clipping_ratio': avg_clipping_ratio,
            'gradient_norm': total_norm,
            'clipped_norm': clipped_norm,
            'layer_clipping_ratios': clipping_ratios
        }
    
    def _clip_percentile(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Clip gradients based on percentile threshold."""
        # Collect all gradient values
        all_gradients = []
        for param in model.parameters():
            if param.grad is not None:
                all_gradients.extend(param.grad.data.abs().view(-1).tolist())
        
        if not all_gradients:
            return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
        
        # Calculate percentile threshold
        threshold = np.percentile(all_gradients, self.config.percentile_threshold)
        
        # Clip gradients above threshold
        clipped_count = 0
        total_norm = 0.0
        clipped_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                # Calculate norm before clipping
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Clip gradients above threshold
                param.grad.data.clamp_(-threshold, threshold)
                
                # Calculate norm after clipping
                clipped_param_norm = param.grad.data.norm(2)
                clipped_norm += clipped_param_norm.item() ** 2
                
                if param_norm.item() > threshold:
                    clipped_count += 1
        
        total_norm = math.sqrt(total_norm)
        clipped_norm = math.sqrt(clipped_norm)
        clipping_ratio = clipped_count / len(list(model.parameters()))
        
        # Update history
        self._update_clipping_history(total_norm, clipped_norm, 'percentile')
        
        return {
            'clipping_ratio': clipping_ratio,
            'gradient_norm': total_norm,
            'clipped_norm': clipped_norm,
            'percentile_threshold': threshold
        }
    
    def _clip_exponential(self, model: nn.Module, optimizer: Optimizer) -> Dict[str, float]:
        """Clip gradients using exponential moving average of thresholds."""
        # Initialize exponential threshold if not exists
        if not hasattr(self, 'exp_threshold'):
            self.exp_threshold = self.config.max_norm
        
        # Get all gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.view(-1))
        
        if not gradients:
            return {'clipping_ratio': 0.0, 'gradient_norm': 0.0, 'clipped_norm': 0.0}
        
        # Concatenate gradients
        all_gradients = torch.cat(gradients)
        
        # Calculate norm
        total_norm = all_gradients.norm(2)
        
        # Update exponential threshold
        self.exp_threshold = (self.config.exponential_alpha * self.exp_threshold + 
                             (1 - self.config.exponential_alpha) * total_norm.item())
        
        # Ensure minimum threshold
        self.exp_threshold = max(self.exp_threshold, self.config.exponential_min_threshold)
        
        # Clip if necessary
        clip_coef = self.exp_threshold / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        # Apply clipping
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Update history
        self._update_clipping_history(total_norm.item(), total_norm.item() * clip_coef, 'exponential')
        
        return {
            'clipping_ratio': 1.0 - clip_coef,
            'gradient_norm': total_norm.item(),
            'clipped_norm': total_norm.item() * clip_coef,
            'exp_threshold': self.exp_threshold
        }
    
    def _update_clipping_history(self, gradient_norm: float, clipped_norm: float, clipping_type: str):
        """Update clipping history."""
        self.clipping_history['steps'].append(self.current_step)
        self.clipping_history['gradient_norms'].append(gradient_norm)
        self.clipping_history['clipped_norms'].append(clipped_norm)
        self.clipping_history['clipping_types'].append(clipping_type)
        
        # Calculate clipping ratio
        if gradient_norm > 0:
            clipping_ratio = 1.0 - (clipped_norm / gradient_norm)
        else:
            clipping_ratio = 0.0
        
        self.clipping_history['clipping_ratios'].append(clipping_ratio)
        
        # Log if enabled
        if self.config.log_clipping_stats:
            self.logger.info(f"Step {self.current_step}: Gradient norm = {gradient_norm:.6f}, "
                           f"Clipped norm = {clipped_norm:.6f}, Clipping ratio = {clipping_ratio:.4f}")
    
    def save_history(self) -> Any:
        """Save clipping history."""
        if self.config.save_clipping_history:
            history_data = {
                'history': self.clipping_history,
                'config': self.config,
                'current_step': self.current_step
            }
            
            with open(self.config.clipping_history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            self.logger.info(f"Clipping history saved to {self.config.clipping_history_file}")
    
    def plot_clipping_history(self, save_path: Optional[str] = None):
        """Plot gradient clipping history."""
        if not self.clipping_history['steps']:
            self.logger.warning("No clipping history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot gradient norms
        axes[0, 0].plot(self.clipping_history['steps'], self.clipping_history['gradient_norms'], 
                        label='Gradient Norm', alpha=0.7)
        axes[0, 0].plot(self.clipping_history['steps'], self.clipping_history['clipped_norms'], 
                        label='Clipped Norm', alpha=0.7)
        axes[0, 0].set_title('Gradient Norms Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Norm')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_yscale('log')
        
        # Plot clipping ratios
        axes[0, 1].plot(self.clipping_history['steps'], self.clipping_history['clipping_ratios'], 
                        label='Clipping Ratio', color='red')
        axes[0, 1].set_title('Clipping Ratios Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Clipping Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot clipping ratio distribution
        axes[1, 0].hist(self.clipping_history['clipping_ratios'], bins=30, alpha=0.7, 
                        edgecolor='black', color='red')
        axes[1, 0].set_title('Clipping Ratio Distribution')
        axes[1, 0].set_xlabel('Clipping Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Plot gradient norm distribution
        axes[1, 1].hist(self.clipping_history['gradient_norms'], bins=30, alpha=0.7, 
                        edgecolor='black', color='blue')
        axes[1, 1].set_title('Gradient Norm Distribution')
        axes[1, 1].set_xlabel('Gradient Norm')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Clipping history plot saved to {save_path}")
        
        plt.show()


class NaNHandler:
    """Advanced NaN/Inf value handling."""
    
    def __init__(self, config: NaNHandlingConfig):
        """__init__ function."""
        self.config = config
        self.logger = self._setup_logging()
        
        # NaN history
        self.nan_history = {
            'steps': [],
            'nan_counts': [],
            'inf_counts': [],
            'overflow_counts': [],
            'handling_actions': []
        }
        
        # Current step
        self.current_step = 0
        
        # Statistics
        self.total_nan_detected = 0
        self.total_inf_detected = 0
        self.total_overflow_detected = 0
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging using centralized logging configuration."""
        return get_logger('numerical_stability')
    
    def check_and_handle(self, model: nn.Module, loss: torch.Tensor, 
                         optimizer: Optimizer) -> Dict[str, Any]:
        """Check for NaN/Inf values and handle them."""
        start_time = time.time()
        self.current_step += 1
        
        # Log NaN checking operation start
        self.logger.info(f"Starting NaN/Inf checking step {self.current_step}")
        
        try:
            nan_detected = False
            inf_detected = False
            overflow_detected = False
            handling_action = "none"
            
            # Check loss
            if self.config.loss_checking:
                loss_nan, loss_inf, loss_overflow = self._check_tensor(loss, "loss")
                nan_detected |= loss_nan
                inf_detected |= loss_inf
                overflow_detected |= loss_overflow
                
                if loss_nan or loss_inf or loss_overflow:
                    self.logger.warning(f"Step {self.current_step}: Loss tensor has numerical issues - "
                                      f"NaN={loss_nan}, Inf={loss_inf}, Overflow={loss_overflow}")
            
            # Check parameters
            if self.config.parameter_checking:
                param_nan, param_inf, param_overflow = self._check_parameters(model)
                nan_detected |= param_nan
                inf_detected |= param_inf
                overflow_detected |= param_overflow
                
                if param_nan or param_inf or param_overflow:
                    self.logger.warning(f"Step {self.current_step}: Model parameters have numerical issues - "
                                      f"NaN={param_nan}, Inf={param_inf}, Overflow={param_overflow}")
            
            # Check gradients
            grad_nan, grad_inf, grad_overflow = self._check_gradients(model)
            nan_detected |= grad_nan
            inf_detected |= grad_inf
            overflow_detected |= grad_overflow
            
            if grad_nan or grad_inf or grad_overflow:
                self.logger.warning(f"Step {self.current_step}: Gradients have numerical issues - "
                                  f"NaN={grad_nan}, Inf={grad_inf}, Overflow={grad_overflow}")
            
            # Handle if detected
            if nan_detected or inf_detected or overflow_detected:
                handling_action = self._handle_numerical_issues(model, optimizer, 
                                                              nan_detected, inf_detected, overflow_detected)
                
                # Log handling action
                self.logger.info(f"Step {self.current_step}: Applied handling action: {handling_action}")
            
            # Update statistics
            if nan_detected:
                self.total_nan_detected += 1
            if inf_detected:
                self.total_inf_detected += 1
            if overflow_detected:
                self.total_overflow_detected += 1
            
            # Update history
            self._update_nan_history(nan_detected, inf_detected, overflow_detected, handling_action)
            
            # Log comprehensive results
            duration = time.time() - start_time
            result = {
                'nan_detected': nan_detected,
                'inf_detected': inf_detected,
                'overflow_detected': overflow_detected,
                'handling_action': handling_action,
                'total_nan': self.total_nan_detected,
                'total_inf': self.total_inf_detected,
                'total_overflow': self.total_overflow_detected
            }
            
            # Log performance metrics
            log_performance_metrics(
                self.logger,
                metrics={
                    "nan_detected": nan_detected,
                    "inf_detected": inf_detected,
                    "overflow_detected": overflow_detected,
                    "handling_action": handling_action,
                    "total_nan": self.total_nan_detected,
                    "total_inf": self.total_inf_detected,
                    "total_overflow": self.total_overflow_detected
                },
                operation=f"nan_checking_{self.current_step}",
                duration=duration
            )
            
            # Log system event
            if nan_detected or inf_detected or overflow_detected:
                log_system_event(
                    self.logger,
                    event_type="numerical_issues_detected",
                    description=f"Numerical issues detected in step {self.current_step}",
                    details={
                        "nan_detected": nan_detected,
                        "inf_detected": inf_detected,
                        "overflow_detected": overflow_detected,
                        "handling_action": handling_action,
                        "duration_seconds": duration
                    },
                    level="warning"
                )
            else:
                log_system_event(
                    self.logger,
                    event_type="numerical_check_passed",
                    description=f"Numerical check passed for step {self.current_step}",
                    details={"duration_seconds": duration}
                )
            
            self.logger.info(f"NaN/Inf checking step {self.current_step} completed successfully in {duration:.4f}s")
            return result
            
        except Exception as e:
            # Log error with context
            log_error_with_context(
                self.logger,
                error=e,
                operation=f"nan_checking_{self.current_step}",
                context={
                    "step": self.current_step,
                    "loss_value": loss.item() if hasattr(loss, 'item') else str(loss),
                    "model_parameters": sum(p.numel() for p in model.parameters())
                },
                recovery_attempted=False
            )
            
            # Log system event for failed checking
            log_system_event(
                self.logger,
                event_type="nan_checking_failed",
                description=f"NaN/Inf checking step {self.current_step} failed",
                details={"error": str(e)},
                level="error"
            )
            
            # Re-raise the exception
            raise
    
    def _check_tensor(self, tensor: torch.Tensor, name: str) -> Tuple[bool, bool, bool]:
        """Check tensor for NaN/Inf values."""
        nan_detected = torch.isnan(tensor).any().item()
        inf_detected = torch.isinf(tensor).any().item()
        overflow_detected = (tensor.abs() > self.config.overflow_threshold).any().item()
        
        if nan_detected or inf_detected or overflow_detected:
            if self.config.log_nan_stats:
                self.logger.warning(f"Step {self.current_step}: {name} has "
                                  f"NaN={nan_detected}, Inf={inf_detected}, Overflow={overflow_detected}")
        
        return nan_detected, inf_detected, overflow_detected
    
    def _check_parameters(self, model: nn.Module) -> Tuple[bool, bool, bool]:
        """Check model parameters for NaN/Inf values."""
        nan_detected = False
        inf_detected = False
        overflow_detected = False
        
        for name, param in model.named_parameters():
            param_nan, param_inf, param_overflow = self._check_tensor(param.data, f"parameter {name}")
            nan_detected |= param_nan
            inf_detected |= param_inf
            overflow_detected |= param_overflow
        
        return nan_detected, inf_detected, overflow_detected
    
    def _check_gradients(self, model: nn.Module) -> Tuple[bool, bool, bool]:
        """Check model gradients for NaN/Inf values."""
        nan_detected = False
        inf_detected = False
        overflow_detected = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_nan, grad_inf, grad_overflow = self._check_tensor(param.grad.data, f"gradient {name}")
                nan_detected |= grad_nan
                inf_detected |= grad_inf
                overflow_detected |= grad_overflow
        
        return nan_detected, inf_detected, overflow_detected
    
    def _handle_numerical_issues(self, model: nn.Module, optimizer: Optimizer,
                                nan_detected: bool, inf_detected: bool, 
                                overflow_detected: bool) -> str:
        """Handle numerical issues based on configuration."""
        if self.config.handling_type == NaNHandlingType.DETECT:
            return "detected"
        
        elif self.config.handling_type == NaNHandlingType.REPLACE:
            self._replace_numerical_issues(model)
            return "replaced"
        
        elif self.config.handling_type == NaNHandlingType.SKIP:
            optimizer.zero_grad()
            return "skipped"
        
        elif self.config.handling_type == NaNHandlingType.RESTORE:
            # This would require checkpoint management
            self.logger.warning("Restore handling requires checkpoint management")
            return "restore_failed"
        
        elif self.config.handling_type == NaNHandlingType.GRADIENT_ZEROING:
            if self.config.gradient_zeroing:
                self._zero_gradients(model)
            return "gradients_zeroed"
        
        elif self.config.handling_type == NaNHandlingType.ADAPTIVE:
            return self._handle_adaptive(model, optimizer, nan_detected, inf_detected, overflow_detected)
        
        elif self.config.handling_type == NaNHandlingType.GRADIENT_SCALING:
            return self._handle_gradient_scaling(model, optimizer, nan_detected, inf_detected, overflow_detected)
        
        else:
            return "unknown"
    
    def _replace_numerical_issues(self, model: nn.Module):
        """Replace NaN/Inf values with safe values."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Replace NaN gradients
                if torch.isnan(param.grad.data).any():
                    param.grad.data = torch.where(torch.isnan(param.grad.data),
                                                torch.tensor(self.config.nan_replacement),
                                                param.grad.data)
                
                # Replace Inf gradients
                if torch.isinf(param.grad.data).any():
                    param.grad.data = torch.where(torch.isinf(param.grad.data),
                                                torch.tensor(self.config.inf_replacement),
                                                param.grad.data)
                
                # Replace overflow gradients
                overflow_mask = param.grad.data.abs() > self.config.overflow_threshold
                if overflow_mask.any():
                    param.grad.data = torch.where(overflow_mask,
                                                torch.tensor(self.config.overflow_replacement),
                                                param.grad.data)
    
    def _zero_gradients(self, model: nn.Module):
        """Zero gradients with numerical issues."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Zero gradients with NaN/Inf
                nan_inf_mask = torch.isnan(param.grad.data) | torch.isinf(param.grad.data)
                if nan_inf_mask.any():
                    param.grad.data[nan_inf_mask] = 0.0
    
    def _handle_adaptive(self, model: nn.Module, optimizer: Optimizer,
                         nan_detected: bool, inf_detected: bool, overflow_detected: bool) -> str:
        """Adaptive handling based on severity of numerical issues."""
        # Calculate severity score
        severity = 0.0
        if nan_detected:
            severity += 1.0  # NaN is most severe
        if inf_detected:
            severity += 0.7  # Inf is moderately severe
        if overflow_detected:
            severity += 0.3  # Overflow is least severe
        
        # Adaptive handling based on severity
        if severity >= 1.5:  # High severity
            self._zero_gradients(model)
            return "adaptive_zeroed_high"
        elif severity >= 0.8:  # Medium severity
            self._replace_numerical_issues(model)
            return "adaptive_replaced_medium"
        else:  # Low severity
            # Just scale down problematic gradients
            self._scale_problematic_gradients(model, scale_factor=0.5)
            return "adaptive_scaled_low"
    
    def _handle_gradient_scaling(self, model: nn.Module, optimizer: Optimizer,
                                nan_detected: bool, inf_detected: bool, overflow_detected: bool) -> str:
        """Scale gradients instead of zeroing them."""
        if nan_detected or inf_detected or overflow_detected:
            # Calculate appropriate scale factor based on issue type
            if nan_detected:
                scale_factor = 0.1  # Very aggressive scaling for NaN
            elif inf_detected:
                scale_factor = 0.3  # Moderate scaling for Inf
            else:
                scale_factor = 0.7  # Light scaling for overflow
            
            self._scale_problematic_gradients(model, scale_factor)
            return f"scaled_{scale_factor}"
        
        return "no_scaling_needed"
    
    def _scale_problematic_gradients(self, model: nn.Module, scale_factor: float):
        """Scale gradients that have numerical issues."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Scale gradients with NaN/Inf/Overflow
                nan_mask = torch.isnan(param.grad.data)
                inf_mask = torch.isinf(param.grad.data)
                overflow_mask = param.grad.data.abs() > self.config.overflow_threshold
                
                # Apply scaling to problematic gradients
                if nan_mask.any():
                    param.grad.data = torch.where(nan_mask,
                                                param.grad.data * scale_factor,
                                                param.grad.data)
                
                if inf_mask.any():
                    param.grad.data = torch.where(inf_mask,
                                                param.grad.data * scale_factor,
                                                param.grad.data)
                
                if overflow_mask.any():
                    param.grad.data = torch.where(overflow_mask,
                                                param.grad.data * scale_factor,
                                                param.grad.data)
    
    def _update_nan_history(self, nan_detected: bool, inf_detected: bool, 
                           overflow_detected: bool, handling_action: str):
        """Update NaN handling history."""
        self.nan_history['steps'].append(self.current_step)
        self.nan_history['nan_counts'].append(1 if nan_detected else 0)
        self.nan_history['inf_counts'].append(1 if inf_detected else 0)
        self.nan_history['overflow_counts'].append(1 if overflow_detected else 0)
        self.nan_history['handling_actions'].append(handling_action)
    
    def save_history(self) -> Any:
        """Save NaN handling history."""
        if self.config.save_nan_history:
            history_data = {
                'history': self.nan_history,
                'config': self.config,
                'current_step': self.current_step,
                'total_nan': self.total_nan_detected,
                'total_inf': self.total_inf_detected,
                'total_overflow': self.total_overflow_detected
            }
            
            with open(self.config.nan_history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            self.logger.info(f"NaN handling history saved to {self.config.nan_history_file}")
    
    def plot_nan_history(self, save_path: Optional[str] = None):
        """Plot NaN handling history."""
        if not self.nan_history['steps']:
            self.logger.warning("No NaN history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot NaN/Inf/Overflow counts over time
        axes[0, 0].plot(self.nan_history['steps'], self.nan_history['nan_counts'], 
                        label='NaN', alpha=0.7, color='red')
        axes[0, 0].plot(self.nan_history['steps'], self.nan_history['inf_counts'], 
                        label='Inf', alpha=0.7, color='orange')
        axes[0, 0].plot(self.nan_history['steps'], self.nan_history['overflow_counts'], 
                        label='Overflow', alpha=0.7, color='yellow')
        axes[0, 0].set_title('Numerical Issues Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot cumulative counts
        cumulative_nan = np.cumsum(self.nan_history['nan_counts'])
        cumulative_inf = np.cumsum(self.nan_history['inf_counts'])
        cumulative_overflow = np.cumsum(self.nan_history['overflow_counts'])
        
        axes[0, 1].plot(self.nan_history['steps'], cumulative_nan, 
                        label='Cumulative NaN', color='red')
        axes[0, 1].plot(self.nan_history['steps'], cumulative_inf, 
                        label='Cumulative Inf', color='orange')
        axes[0, 1].plot(self.nan_history['steps'], cumulative_overflow, 
                        label='Cumulative Overflow', color='yellow')
        axes[0, 1].set_title('Cumulative Numerical Issues')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cumulative Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot handling actions distribution
        action_counts = {}
        for action in self.nan_history['handling_actions']:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        if action_counts:
            actions = list(action_counts.keys())
            counts = list(action_counts.values())
            axes[1, 0].bar(actions, counts, alpha=0.7)
            axes[1, 0].set_title('Handling Actions Distribution')
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True)
        
        # Plot numerical issues distribution
        total_issues = np.array(self.nan_history['nan_counts']) + \
                      np.array(self.nan_history['inf_counts']) + \
                      np.array(self.nan_history['overflow_counts'])
        
        axes[1, 1].hist(total_issues, bins=range(int(max(total_issues)) + 2), 
                        alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Total Numerical Issues Distribution')
        axes[1, 1].set_xlabel('Total Issues per Step')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"NaN history plot saved to {save_path}")
        
        plt.show()


class PyTorchDebuggingManager:
    """Manager for PyTorch built-in debugging tools."""
    
    def __init__(self, config: PyTorchDebuggingConfig):
        """Initialize the debugging manager."""
        self.config = config
        self.logger = get_logger('pytorch_debugging')
        
        # Debug state
        self.debug_enabled = False
        self.anomaly_detector_enabled = False
        self.grad_check_enabled = False
        self.memory_debug_enabled = False
        
        # Debug history
        self.debug_history = []
        self.anomaly_history = []
        self.grad_check_history = []
        self.memory_history = []
        
        # Performance tracking
        self.debug_start_time = None
        self.debug_iteration_count = 0
        
        # Setup debugging if enabled
        if config.enable_autograd_anomaly or config.enable_grad_check or config.enable_memory_debugging:
            self._setup_debugging()
    
    def _setup_debugging(self):
        """Setup PyTorch debugging tools."""
        try:
            # Create debug output directory
            if self.config.save_debug_info:
                debug_dir = Path(self.config.debug_output_dir)
                debug_dir.mkdir(exist_ok=True)
                self.logger.info(f"Debug output directory created: {debug_dir}")
            
            # Enable autograd anomaly detection
            if self.config.enable_autograd_anomaly:
                self._enable_autograd_anomaly()
            
            # Enable gradient checking
            if self.config.enable_grad_check:
                self._enable_grad_check()
            
            # Enable memory debugging
            if self.config.enable_memory_debugging:
                self._enable_memory_debugging()
            
            # Enable CUDA debugging
            if self.config.enable_cuda_debugging:
                self._enable_cuda_debugging()
            
            self.debug_enabled = True
            self.logger.info("PyTorch debugging tools initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup PyTorch debugging: {e}")
            self.debug_enabled = False
    
    def _enable_autograd_anomaly(self):
        """Enable autograd anomaly detection."""
        try:
            if self.config.autograd_anomaly_mode == "detect":
                torch.autograd.set_detect_anomaly(True)
                self.anomaly_detector_enabled = True
                self.logger.info("Autograd anomaly detection enabled (detect mode)")
            elif self.config.autograd_anomaly_mode == "raise":
                torch.autograd.set_detect_anomaly(True)
                self.anomaly_detector_enabled = True
                self.logger.info("Autograd anomaly detection enabled (raise mode)")
        except Exception as e:
            self.logger.error(f"Failed to enable autograd anomaly detection: {e}")
    
    def _enable_grad_check(self):
        """Enable gradient checking."""
        try:
            if self.config.grad_check_numerical:
                torch.autograd.gradcheck.enable()
                self.grad_check_enabled = True
                self.logger.info("Gradient numerical checking enabled")
            
            if self.config.grad_check_sparse_numerical:
                torch.autograd.gradcheck.enable_sparse_numerical()
                self.logger.info("Sparse gradient numerical checking enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable gradient checking: {e}")
    
    def _enable_memory_debugging(self):
        """Enable memory debugging."""
        try:
            if self.config.memory_tracking:
                torch.cuda.empty_cache()
                self.memory_debug_enabled = True
                self.logger.info("CUDA memory debugging enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable memory debugging: {e}")
    
    def _enable_cuda_debugging(self):
        """Enable CUDA debugging."""
        try:
            if torch.cuda.is_available():
                if self.config.cuda_synchronize:
                    torch.cuda.synchronize()
                    self.logger.info("CUDA synchronization enabled")
                
                if self.config.cuda_memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(self.config.cuda_memory_fraction)
                    self.logger.info(f"CUDA memory fraction set to {self.config.cuda_memory_fraction}")
            else:
                self.logger.warning("CUDA not available, skipping CUDA debugging setup")
        except Exception as e:
            self.logger.error(f"Failed to enable CUDA debugging: {e}")
    
    def start_debug_session(self, session_name: str = None):
        """Start a new debugging session."""
        if not self.debug_enabled:
            self.logger.warning("Debugging not enabled, cannot start session")
            return
        
        try:
            self.debug_start_time = time.time()
            self.debug_iteration_count = 0
            
            session_name = session_name or f"debug_session_{int(time.time())}"
            
            self.logger.info(f"Started debugging session: {session_name}")
            
            # Log system event
            log_system_event(
                self.logger,
                event_type="debug_session_started",
                description=f"PyTorch debugging session started: {session_name}",
                details={"session_name": session_name, "config": self.config.__dict__}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start debug session: {e}")
    
    def check_debug_status(self, model: nn.Module, loss: torch.Tensor = None, 
                          optimizer: Optimizer = None) -> Dict[str, Any]:
        """Check debugging status and perform diagnostics."""
        if not self.debug_enabled:
            return {"debug_enabled": False}
        
        try:
            self.debug_iteration_count += 1
            
            # Check timeout
            if self.debug_start_time and (time.time() - self.debug_start_time) > self.config.debug_timeout:
                self.logger.warning("Debug session timeout reached")
                return {"debug_enabled": True, "timeout_reached": True}
            
            # Check iteration limit
            if self.debug_iteration_count > self.config.max_debug_iterations:
                self.logger.warning("Maximum debug iterations reached")
                return {"debug_enabled": True, "max_iterations_reached": True}
            
            debug_info = {
                "debug_enabled": True,
                "iteration": self.debug_iteration_count,
                "session_duration": time.time() - self.debug_start_time if self.debug_start_time else 0,
                "anomaly_detector": self.anomaly_detector_enabled,
                "grad_check": self.grad_check_enabled,
                "memory_debug": self.memory_debug_enabled
            }
            
            # Check for anomalies if enabled
            if self.anomaly_detector_enabled:
                anomaly_info = self._check_for_anomalies(model, loss, optimizer)
                debug_info.update(anomaly_info)
            
            # Check gradients if enabled
            if self.grad_check_enabled:
                grad_info = self._check_gradients(model)
                debug_info.update(grad_info)
            
            # Check memory if enabled
            if self.memory_debug_enabled:
                memory_info = self._check_memory_usage()
                debug_info.update(memory_info)
            
            # Update debug history
            self.debug_history.append(debug_info)
            
            # Log debug status
            if self.config.verbose_logging:
                self.logger.debug(f"Debug status check {self.debug_iteration_count}: {debug_info}")
            
            return debug_info
            
        except Exception as e:
            self.logger.error(f"Error during debug status check: {e}")
            return {"debug_enabled": True, "error": str(e)}
    
    def _check_for_anomalies(self, model: nn.Module, loss: torch.Tensor = None, 
                            optimizer: Optimizer = None) -> Dict[str, Any]:
        """Check for autograd anomalies."""
        try:
            anomaly_info = {
                "anomalies_detected": False,
                "anomaly_details": []
            }
            
            # Check model parameters for NaN/Inf
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        anomaly_info["anomalies_detected"] = True
                        anomaly_info["anomaly_details"].append(f"NaN gradient in {name}")
                    
                    if torch.isinf(param.grad).any():
                        anomaly_info["anomalies_detected"] = True
                        anomaly_info["anomaly_details"].append(f"Inf gradient in {name}")
            
            # Check loss for anomalies
            if loss is not None:
                if torch.isnan(loss):
                    anomaly_info["anomalies_detected"] = True
                    anomaly_info["anomaly_details"].append("NaN loss")
                
                if torch.isinf(loss):
                    anomaly_info["anomalies_detected"] = True
                    anomaly_info["anomaly_details"].append("Inf loss")
            
            # Update anomaly history
            if anomaly_info["anomalies_detected"]:
                self.anomaly_history.append({
                    "iteration": self.debug_iteration_count,
                    "timestamp": time.time(),
                    "details": anomaly_info["anomaly_details"]
                })
                
                # Log anomaly detection
                log_numerical_issue(
                    self.logger,
                    issue_type="autograd_anomaly",
                    severity="high",
                    location=f"debug_iteration_{self.debug_iteration_count}",
                    details={"anomalies": anomaly_info["anomaly_details"]},
                    recovery_action="debug_investigation"
                )
            
            return anomaly_info
            
        except Exception as e:
            self.logger.error(f"Error checking for anomalies: {e}")
            return {"anomalies_detected": False, "error": str(e)}
    
    def _check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Check gradient properties."""
        try:
            grad_info = {
                "total_parameters": 0,
                "parameters_with_grad": 0,
                "gradient_norms": [],
                "gradient_stats": {}
            }
            
            total_norm = 0.0
            grad_norms = []
            
            for param in model.parameters():
                grad_info["total_parameters"] += 1
                
                if param.grad is not None:
                    grad_info["parameters_with_grad"] += 1
                    
                    # Calculate gradient norm
                    param_norm = param.grad.data.norm(2).item()
                    grad_norms.append(param_norm)
                    total_norm += param_norm ** 2
            
            if grad_norms:
                grad_info["gradient_norms"] = grad_norms
                grad_info["gradient_stats"] = {
                    "mean_norm": np.mean(grad_norms),
                    "std_norm": np.std(grad_norms),
                    "min_norm": np.min(grad_norms),
                    "max_norm": np.max(grad_norms),
                    "total_norm": np.sqrt(total_norm)
                }
            
            # Update gradient check history
            self.grad_check_history.append({
                "iteration": self.debug_iteration_count,
                "timestamp": time.time(),
                "gradient_stats": grad_info["gradient_stats"]
            })
            
            return grad_info
            
        except Exception as e:
            self.logger.error(f"Error checking gradients: {e}")
            return {"error": str(e)}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage and CUDA memory."""
        try:
            memory_info = {
                "cpu_memory": {},
                "cuda_memory": {}
            }
            
            # CPU memory info
            import psutil
            process = psutil.Process()
            memory_info["cpu_memory"] = {
                "rss": process.memory_info().rss / 1024 / 1024,  # MB
                "vms": process.memory_info().vms / 1024 / 1024,  # MB
                "percent": process.memory_percent()
            }
            
            # CUDA memory info
            if torch.cuda.is_available():
                memory_info["cuda_memory"] = {
                    "allocated": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                    "cached": torch.cuda.memory_reserved() / 1024 / 1024,  # MB
                    "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024,  # MB
                    "device_count": torch.cuda.device_count()
                }
                
                # Check for memory issues
                if memory_info["cuda_memory"]["allocated"] > 1000:  # > 1GB
                    self.logger.warning(f"High CUDA memory usage: {memory_info['cuda_memory']['allocated']:.2f} MB")
            
            # Update memory history
            self.memory_history.append({
                "iteration": self.debug_iteration_count,
                "timestamp": time.time(),
                "memory_info": memory_info
            })
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")
            return {"error": str(e)}
    
    def stop_debug_session(self):
        """Stop the current debugging session."""
        if not self.debug_enabled:
            return
        
        try:
            # Disable debugging tools
            if self.anomaly_detector_enabled:
                torch.autograd.set_detect_anomaly(False)
                self.anomaly_detector_enabled = False
            
            if self.grad_check_enabled:
                torch.autograd.gradcheck.disable()
                self.grad_check_enabled = False
            
            # Save debug information
            if self.config.save_debug_info and self.debug_history:
                self._save_debug_info()
            
            # Log session completion
            session_duration = time.time() - self.debug_start_time if self.debug_start_time else 0
            
            self.logger.info(f"Debug session stopped. Duration: {session_duration:.2f}s, Iterations: {self.debug_iteration_count}")
            
            # Log system event
            log_system_event(
                self.logger,
                event_type="debug_session_stopped",
                description="PyTorch debugging session stopped",
                details={
                    "session_duration": session_duration,
                    "total_iterations": self.debug_iteration_count,
                    "anomalies_detected": len(self.anomaly_history),
                    "debug_history_size": len(self.debug_history)
                }
            )
            
            # Reset state
            self.debug_start_time = None
            self.debug_iteration_count = 0
            
        except Exception as e:
            self.logger.error(f"Error stopping debug session: {e}")
    
    def _save_debug_info(self):
        """Save debug information to files."""
        try:
            debug_dir = Path(self.config.debug_output_dir)
            timestamp = int(time.time())
            
            # Save debug history
            debug_file = debug_dir / f"{self.config.debug_file_prefix}_history_{timestamp}.json"
            with open(debug_file, 'w') as f:
                json.dump(self.debug_history, f, indent=2, default=str)
            
            # Save anomaly history
            if self.anomaly_history:
                anomaly_file = debug_dir / f"{self.config.debug_file_prefix}_anomalies_{timestamp}.json"
                with open(anomaly_file, 'w') as f:
                    json.dump(self.anomaly_history, f, indent=2, default=str)
            
            # Save gradient check history
            if self.grad_check_history:
                grad_file = debug_dir / f"{self.config.debug_file_prefix}_gradients_{timestamp}.json"
                with open(grad_file, 'w') as f:
                    json.dump(self.grad_check_history, f, indent=2, default=str)
            
            # Save memory history
            if self.memory_history:
                memory_file = debug_dir / f"{self.config.debug_file_prefix}_memory_{timestamp}.json"
                with open(memory_file, 'w') as f:
                    json.dump(self.memory_history, f, indent=2, default=str)
            
            self.logger.info(f"Debug information saved to {debug_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save debug information: {e}")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get a summary of debugging information."""
        if not self.debug_enabled:
            return {"debug_enabled": False}
        
        try:
            summary = {
                "debug_enabled": True,
                "session_active": self.debug_start_time is not None,
                "total_iterations": self.debug_iteration_count,
                "anomalies_detected": len(self.anomaly_history),
                "gradient_checks": len(self.grad_check_history),
                "memory_checks": len(self.memory_history),
                "config": self.config.__dict__
            }
            
            if self.debug_start_time:
                summary["session_duration"] = time.time() - self.debug_start_time
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting debug summary: {e}")
            return {"debug_enabled": True, "error": str(e)}


class NumericalStabilityManager:
    """Comprehensive numerical stability manager."""
    
    def __init__(self, clipping_config: GradientClippingConfig, 
                 nan_config: NaNHandlingConfig,
                 debug_config: PyTorchDebuggingConfig = None,
                 performance_config: PerformanceOptimizationConfig = None):
        """__init__ function."""
        self.clipping_config = clipping_config
        self.nan_config = nan_config
        self.performance_config = performance_config or PerformanceOptimizationConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.gradient_clipper = GradientClipper(clipping_config)
        self.nan_handler = NaNHandler(nan_config)
        
        # Initialize PyTorch debugging manager if config provided
        self.debug_manager = None
        if debug_config is not None:
            self.debug_manager = PyTorchDebuggingManager(debug_config)
            self.logger.info("PyTorch debugging manager initialized")
        
        # Initialize performance optimization if available
        self.performance_optimizer = None
        if (self.performance_config.enable_performance_optimization and 
            PERFORMANCE_OPTIMIZATION_AVAILABLE):
            self._setup_performance_optimization()
        
        # Training state
        self.current_step = 0
        self.stability_history = {
            'steps': [],
            'clipping_stats': [],
            'nan_stats': [],
            'stability_scores': [],
            'debug_info': [],
            'performance_metrics': []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging using centralized logging configuration."""
        return get_logger('numerical_stability')
    
    def _setup_performance_optimization(self):
        """Setup performance optimization system."""
        try:
            # Create performance configuration based on level
            if self.performance_config.optimization_level == "basic":
                opt_level = OptimizationLevel.BASIC
            elif self.performance_config.optimization_level == "ultra":
                opt_level = OptimizationLevel.ULTRA
            else:
                opt_level = OptimizationLevel.ADVANCED
            
            # Create multi-GPU configuration
            multi_gpu_config = MultiGPUConfig(
                mode=MultiGPUMode(self.performance_config.multi_gpu_mode),
                device_ids=self.performance_config.multi_gpu_device_ids,
                backend=self.performance_config.multi_gpu_backend,
                find_unused_parameters=self.performance_config.multi_gpu_find_unused_parameters,
                bucket_cap_mb=self.performance_config.multi_gpu_bucket_cap_mb,
                static_graph=self.performance_config.multi_gpu_static_graph
            )
            
            perf_config = PerformanceConfig(
                optimization_level=opt_level,
                enable_mixed_precision=self.performance_config.enable_mixed_precision,
                enable_compile=self.performance_config.enable_model_compilation,
                enable_gradient_checkpointing=True,  # Always enable for stability
                enable_memory_format_optimization=self.performance_config.enable_memory_optimization,
                multi_gpu_config=multi_gpu_config
            )
            
            self.performance_optimizer = create_performance_optimizer(perf_config)
            self.logger.info(f"Performance optimization initialized at {opt_level.value} level")
            
        except Exception as e:
            self.logger.warning(f"Performance optimization setup failed: {e}")
            self.performance_optimizer = None
    
    def start_debug_session(self, session_name: str = None):
        """Start a PyTorch debugging session if enabled."""
        if self.debug_manager is not None:
            self.debug_manager.start_debug_session(session_name)
            self.logger.info(f"Debug session started: {session_name}")
        else:
            self.logger.warning("Debug manager not initialized, cannot start session")
    
    def stop_debug_session(self):
        """Stop the current PyTorch debugging session if active."""
        if self.debug_manager is not None:
            self.debug_manager.stop_debug_session()
            self.logger.info("Debug session stopped")
        else:
            self.logger.warning("Debug manager not initialized, cannot stop session")
    
    def step(self, model: nn.Module, loss: torch.Tensor, 
             optimizer: Optimizer) -> Dict[str, Any]:
        """Perform one step with numerical stability checks."""
        start_time = time.time()
        self.current_step += 1
        
        # Log step start
        self.logger.info(f"Starting stability step {self.current_step}")
        
        try:
            # Start debugging session if this is the first step
            if self.debug_manager is not None and self.current_step == 1:
                self.start_debug_session(f"training_session_{int(time.time())}")
            
            # Check debugging status if enabled
            debug_info = {}
            if self.debug_manager is not None:
                debug_info = self.debug_manager.check_debug_status(model, loss, optimizer)
                
                # Log debug information
                if debug_info.get('anomalies_detected', False):
                    self.logger.warning(f"Debug anomalies detected in step {self.current_step}: {debug_info.get('anomaly_details', [])}")
                
                # Update debug history
                self.stability_history['debug_info'].append(debug_info)
            
            # Apply performance optimizations if enabled
            performance_metrics = {}
            if self.performance_optimizer is not None:
                try:
                    # Monitor memory usage
                    memory_info = self.performance_optimizer.memory_manager.monitor_memory(self.current_step)
                    
                    # Record performance metrics
                    self.performance_optimizer.performance_monitor.record_metrics(
                        self.current_step, 
                        {
                            'loss': loss.item() if hasattr(loss, 'item') else float(loss),
                            'step_time': 0,  # Will be updated later
                            'memory_usage_percent': memory_info.get('system_percent', 0),
                            'gpu_memory_mb': memory_info.get('gpu_allocated_mb', 0)
                        }
                    )
                    
                    performance_metrics = {
                        'memory_usage_percent': memory_info.get('system_percent', 0),
                        'gpu_memory_mb': memory_info.get('gpu_allocated_mb', 0),
                        'optimization_level': self.performance_config.optimization_level
                    }
                    
                    # Store in stability history
                    self.stability_history['performance_metrics'].append(performance_metrics)
                    
                except Exception as e:
                    self.logger.warning(f"Performance optimization monitoring failed: {e}")
            
            # Check for NaN/Inf values
            nan_stats = self.nan_handler.check_and_handle(model, loss, optimizer)
            
            # Apply gradient clipping
            clipping_stats = self.gradient_clipper.clip_gradients(model, optimizer)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(nan_stats, clipping_stats)
            
            # Update history
            self._update_stability_history(nan_stats, clipping_stats, stability_score, debug_info)
            
            # Comprehensive logging
            duration = time.time() - start_time
            
            # Log training step with stability metrics
            log_training_step(
                self.logger,
                step=self.current_step,
                epoch=0,  # Will be updated by training loop
                loss=loss.item() if hasattr(loss, 'item') else float(loss),
                stability_score=stability_score,
                gradient_norm=clipping_stats.get('gradient_norm', 0.0),
                clipping_ratio=clipping_stats.get('clipping_ratio', 0.0)
            )
            
            # Log numerical issues if detected
            if nan_stats['nan_detected'] or nan_stats['inf_detected'] or nan_stats['overflow_detected']:
                issue_type = []
                if nan_stats['nan_detected']:
                    issue_type.append("NaN")
                if nan_stats['inf_detected']:
                    issue_type.append("Inf")
                if nan_stats['overflow_detected']:
                    issue_type.append("Overflow")
                
                severity = "high" if len(issue_type) > 1 else "medium"
                
                log_numerical_issue(
                    self.logger,
                    issue_type=", ".join(issue_type),
                    severity=severity,
                    location=f"step_{self.current_step}",
                    details={
                        "nan_count": nan_stats.get('nan_count', 0),
                        "inf_count": nan_stats.get('inf_count', 0),
                        "overflow_count": nan_stats.get('overflow_count', 0),
                        "loss_value": loss.item() if hasattr(loss, 'item') else str(loss)
                    },
                    recovery_action="automatic_handling"
                )
            
            # Log performance metrics
            log_performance_metrics(
                self.logger,
                metrics={
                    "stability_score": stability_score,
                    "clipping_ratio": clipping_stats.get('clipping_ratio', 0.0),
                    "gradient_norm": clipping_stats.get('gradient_norm', 0.0),
                    "nan_detected": nan_stats['nan_detected'],
                    "inf_detected": nan_stats['inf_detected'],
                    "overflow_detected": nan_stats['overflow_detected'],
                    "debug_enabled": self.debug_manager is not None,
                    "debug_anomalies": debug_info.get('anomalies_detected', False)
                },
                operation=f"stability_step_{self.current_step}",
                duration=duration
            )
            
            # Log system event for successful step
            log_system_event(
                self.logger,
                event_type="stability_step_completed",
                description=f"Stability step {self.current_step} completed successfully",
                details={
                    "stability_score": stability_score,
                    "clipping_ratio": clipping_stats.get('clipping_ratio', 0.0),
                    "debug_anomalies": debug_info.get('anomalies_detected', False)
                }
            )
            
            return {
                'nan_stats': nan_stats,
                'clipping_stats': clipping_stats,
                'stability_score': stability_score,
                'debug_info': debug_info
            }
            
        except Exception as e:
            # Log error with context
            log_error_with_context(
                self.logger,
                error=e,
                operation=f"stability_step_{self.current_step}",
                context={
                    "step": self.current_step,
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "debug_enabled": self.debug_manager is not None
                },
                recovery_attempted=False
            )
            
            # Log system event for failed step
            log_system_event(
                self.logger,
                event_type="stability_step_failed",
                description=f"Stability step {self.current_step} failed",
                details={"error": str(e)},
                level="error"
            )
            
            # Re-raise the exception
            raise
    
    def _calculate_stability_score(self, nan_stats: Dict[str, Any], 
                                  clipping_stats: Dict[str, float]) -> float:
        """Calculate numerical stability score."""
        score = 1.0
        
        # Penalize for numerical issues
        if nan_stats['nan_detected']:
            score -= 0.3
        if nan_stats['inf_detected']:
            score -= 0.2
        if nan_stats['overflow_detected']:
            score -= 0.1
        
        # Penalize for excessive clipping
        if 'clipping_ratio' in clipping_stats:
            score -= clipping_stats['clipping_ratio'] * 0.1
        
        return max(0.0, score)
    
    def _update_stability_history(self, nan_stats: Dict[str, Any], 
                                 clipping_stats: Dict[str, float], 
                                 stability_score: float,
                                 debug_info: Dict[str, Any]):
        """Update stability history."""
        self.stability_history['steps'].append(self.current_step)
        self.stability_history['nan_stats'].append(nan_stats)
        self.stability_history['clipping_stats'].append(clipping_stats)
        self.stability_history['stability_scores'].append(stability_score)
        self.stability_history['debug_info'].append(debug_info)
    
    def save_histories(self) -> Any:
        """Save all histories."""
        self.gradient_clipper.save_history()
        self.nan_handler.save_history()
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get a summary of debugging information if available."""
        if self.debug_manager is not None:
            return self.debug_manager.get_debug_summary()
        else:
            return {"debug_enabled": False, "message": "Debug manager not initialized"}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary if available."""
        if self.performance_optimizer is not None:
            return self.performance_optimizer.get_optimization_status()
        else:
            return {"performance_optimization": False, "message": "Performance optimizer not initialized"}
    
    def wrap_model_for_multi_gpu(self, model: nn.Module) -> nn.Module:
        """Wrap model for multi-GPU training if available."""
        if self.performance_optimizer is not None:
            try:
                wrapped_model = self.performance_optimizer.wrap_model_for_multi_gpu(model)
                self.logger.info("Model wrapped for multi-GPU training")
                return wrapped_model
            except Exception as e:
                self.logger.warning(f"Multi-GPU model wrapping failed: {e}")
                return model
        else:
            return model
    
    def get_multi_gpu_status(self) -> Dict[str, Any]:
        """Get multi-GPU training status if available."""
        if self.performance_optimizer is not None:
            return self.performance_optimizer.get_multi_gpu_status()
        else:
            return {"multi_gpu": False, "message": "Performance optimizer not initialized"}
    
    def get_optimal_batch_size(self, base_batch_size: int) -> int:
        """Get optimal batch size for multi-GPU training if available."""
        if self.performance_optimizer is not None:
            return self.performance_optimizer.get_optimal_batch_size(base_batch_size)
        else:
            return base_batch_size
    
    def synchronize_gpus(self):
        """Synchronize all GPUs if multi-GPU training is enabled."""
        if self.performance_optimizer is not None:
            self.performance_optimizer.synchronize_gpus()
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply performance optimizations to the model if available."""
        if self.performance_optimizer is not None:
            try:
                optimized_model = self.performance_optimizer.model_optimizer.optimize_model(model)
                self.logger.info("Model performance optimizations applied")
                return optimized_model
            except Exception as e:
                self.logger.warning(f"Model performance optimization failed: {e}")
                return model
        else:
            return model
    
    def plot_stability_history(self, save_path: Optional[str] = None):
        """Plot numerical stability history."""
        if not self.stability_history['steps']:
            self.logger.warning("No stability history to plot")
            return
        
        # Create subplots - add one more if debug info is available
        debug_available = any(info.get('debug_enabled', False) for info in self.stability_history['debug_info'])
        num_plots = 3 if debug_available else 2
        fig, axes = plt.subplots(2, num_plots, figsize=(15, 10))
        
        # Plot stability scores
        axes[0, 0].plot(self.stability_history['steps'], self.stability_history['stability_scores'], 
                        label='Stability Score', color='green')
        axes[0, 0].set_title('Numerical Stability Score Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Stability Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot clipping ratios
        clipping_ratios = [stats.get('clipping_ratio', 0.0) for stats in self.stability_history['clipping_stats']]
        axes[0, 1].plot(self.stability_history['steps'], clipping_ratios, 
                        label='Clipping Ratio', color='blue')
        axes[0, 1].set_title('Gradient Clipping Ratio Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Clipping Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot debug anomalies if available
        if debug_available:
            debug_anomalies = [info.get('anomalies_detected', False) for info in self.stability_history['debug_info']]
            anomaly_steps = [step for step, anomaly in zip(self.stability_history['steps'], debug_anomalies) if anomaly]
            anomaly_values = [1] * len(anomaly_steps)
            
            axes[0, 2].scatter(anomaly_steps, anomaly_values, color='red', s=100, alpha=0.7, label='Debug Anomalies')
            axes[0, 2].set_title('PyTorch Debug Anomalies')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Anomaly Detected')
            axes[0, 2].set_ylim(0, 2)
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Plot numerical issues
        nan_counts = [stats['nan_detected'] for stats in self.stability_history['nan_stats']]
        inf_counts = [stats['inf_detected'] for stats in self.stability_history['nan_stats']]
        overflow_counts = [stats['overflow_detected'] for stats in self.stability_history['nan_stats']]
        
        axes[1, 0].plot(self.stability_history['steps'], nan_counts, 
                        label='NaN', color='red', alpha=0.7)
        axes[1, 0].plot(self.stability_history['steps'], inf_counts, 
                        label='Inf', color='orange', alpha=0.7)
        axes[1, 0].plot(self.stability_history['steps'], overflow_counts, 
                        label='Overflow', color='yellow', alpha=0.7)
        axes[1, 0].set_title('Numerical Issues Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot stability score distribution
        axes[1, 1].hist(self.stability_history['stability_scores'], bins=20, alpha=0.7, 
                        edgecolor='black', color='green')
        axes[1, 1].set_title('Stability Score Distribution')
        axes[1, 1].set_xlabel('Stability Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        # Plot debug information if available
        if debug_available:
            # Plot gradient statistics over time
            grad_norms = []
            for info in self.stability_history['debug_info']:
                if info.get('gradient_stats', {}).get('total_norm'):
                    grad_norms.append(info['gradient_stats']['total_norm'])
                else:
                    grad_norms.append(0.0)
            
            axes[1, 2].plot(self.stability_history['steps'], grad_norms, 
                            label='Total Gradient Norm', color='purple', alpha=0.7)
            axes[1, 2].set_title('Gradient Norms Over Time (Debug)')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Gradient Norm')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Stability history plot saved to {save_path}")
        
        plt.show()


def demonstrate_gradient_clipping_nan_handling():
    """Demonstrate gradient clipping and NaN handling capabilities."""
    print("Gradient Clipping and NaN Handling Demonstration")
    print("=" * 50)
    
    # Create sample model
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x) -> Any:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create configurations for different clipping types
    clipping_configs = {
        'norm': GradientClippingConfig(
            clipping_type=ClippingType.NORM,
            max_norm=1.0,
            monitor_clipping=True,
            log_clipping_stats=True
        ),
        'layer_wise': GradientClippingConfig(
            clipping_type=ClippingType.LAYER_WISE,
            max_norm=1.0,
            layer_wise_enabled=True,
            layer_norm_thresholds={'fc1.weight': 0.8, 'fc2.weight': 1.2, 'fc3.weight': 0.6},
            monitor_clipping=True,
            log_clipping_stats=True
        ),
        'percentile': GradientClippingConfig(
            clipping_type=ClippingType.PERCENTILE,
            percentile_enabled=True,
            percentile_threshold=90.0,
            monitor_clipping=True,
            log_clipping_stats=True
        ),
        'exponential': GradientClippingConfig(
            clipping_type=ClippingType.EXPONENTIAL,
            exponential_enabled=True,
            exponential_alpha=0.95,
            exponential_min_threshold=0.5,
            monitor_clipping=True,
            log_clipping_stats=True
        )
    }
    
    # Create configurations for different NaN handling types
    nan_configs = {
        'detect': NaNHandlingConfig(
            handling_type=NaNHandlingType.DETECT,
            detect_nan=True,
            detect_inf=True,
            detect_overflow=True,
            monitor_nan=True,
            log_nan_stats=True
        ),
        'adaptive': NaNHandlingConfig(
            handling_type=NaNHandlingType.ADAPTIVE,
            detect_nan=True,
            detect_inf=True,
            detect_overflow=True,
            monitor_nan=True,
            log_nan_stats=True
        )
    }
    
    # Create PyTorch debugging configurations
    debug_configs = {
        'basic': PyTorchDebuggingConfig(
            enable_autograd_anomaly=True,
            autograd_anomaly_mode="detect",
            enable_grad_check=False,
            enable_memory_debugging=False,
            debug_level="info",
            verbose_logging=False
        ),
        'comprehensive': PyTorchDebuggingConfig(
            enable_autograd_anomaly=True,
            autograd_anomaly_mode="raise",
            enable_grad_check=True,
            grad_check_numerical=True,
            grad_check_sparse_numerical=True,
            enable_memory_debugging=True,
            memory_tracking=True,
            enable_cuda_debugging=True,
            cuda_synchronize=True,
            debug_level="warning",
            verbose_logging=True,
            save_debug_info=True,
            debug_output_dir="debug_output"
        ),
        'performance': PyTorchDebuggingConfig(
            enable_autograd_anomaly=False,
            enable_grad_check=False,
            enable_memory_debugging=True,
            memory_profiling=True,
            enable_performance_debugging=True,
            profile_autograd=True,
            profile_memory=True,
            debug_level="info",
            verbose_logging=False,
            save_debug_info=True
        )
    }
    
    print("\nPyTorch Debugging Configurations:")
    print("-" * 30)
    for name, config in debug_configs.items():
        print(f"{name}: {config.enable_autograd_anomaly}, {config.enable_grad_check}, {config.enable_memory_debugging}")
    
    # Test different configurations
    results = {}
    
    for config_name, clipping_config in clipping_configs.items():
        for nan_name, nan_config in nan_configs.items():
            for debug_name, debug_config in debug_configs.items():
                full_config_name = f"{config_name}_{nan_name}_{debug_name}"
                print(f"\nTesting configuration: {full_config_name}")
                print("-" * 50)
                
                # Create stability manager with debugging
                stability_manager = NumericalStabilityManager(
                    clipping_config, 
                    nan_config,
                    debug_config
                )
                
                # Create model and optimizer
                model = SimpleModel()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                
                # Test scenarios
                scenarios = [
                    {
                        'name': 'Normal training',
                        'input_data': torch.randn(5, 10),
                        'target': torch.randn(5, 1),
                        'introduce_nan': False
                    },
                    {
                        'name': 'Training with NaN introduction',
                        'input_data': torch.randn(5, 10),
                        'target': torch.randn(5, 1),
                        'introduce_nan': True
                    },
                    {
                        'name': 'Large gradient test',
                        'input_data': torch.randn(5, 10) * 100,  # Large input
                        'target': torch.randn(5, 1) * 100,      # Large target
                        'introduce_nan': False
                    }
                ]
                
                config_results = {}
                
                for i, scenario in enumerate(scenarios):
                    print(f"  Testing scenario {i+1}: {scenario['name']}")
                    
                    try:
                        # Forward pass
                        output = model(scenario['input_data'])
                        loss = nn.MSELoss()(output, scenario['target'])
                        
                        # Introduce NaN if requested
                        if scenario['introduce_nan']:
                            # Introduce NaN in gradients
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data[0, 0] = float('nan')
                        
                        # Backward pass
                        loss.backward()
                        
                        # Apply numerical stability measures
                        stability_result = stability_manager.step(model, loss, optimizer)
                        
                        # Optimizer step
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        print(f"    Loss: {loss.item():.6f}")
                        print(f"    Stability Score: {stability_result['stability_score']:.4f}")
                        print(f"    Clipping Ratio: {stability_result['clipping_stats'].get('clipping_ratio', 0.0):.4f}")
                        print(f"    NaN Detected: {stability_result['nan_stats']['nan_detected']}")
                        print(f"    Inf Detected: {stability_result['nan_stats']['inf_detected']}")
                        
                        # Show debug information if available
                        if stability_result.get('debug_info', {}).get('debug_enabled', False):
                            debug_info = stability_result['debug_info']
                            print(f"    Debug Info:")
                            print(f"      Anomalies: {debug_info.get('anomalies_detected', False)}")
                            if debug_info.get('gradient_stats'):
                                grad_stats = debug_info['gradient_stats']
                                print(f"      Total Gradient Norm: {grad_stats.get('total_norm', 0.0):.4f}")
                                print(f"      Mean Gradient Norm: {grad_stats.get('mean_norm', 0.0):.4f}")
                        
                        config_results[f"scenario_{i}"] = {
                            'scenario': scenario,
                            'stability_result': stability_result,
                            'success': True
                        }
                        
                    except Exception as e:
                        print(f"    Error: {e}")
                        config_results[f"scenario_{i}"] = {
                            'scenario': scenario,
                            'error': str(e),
                            'success': False
                        }
                    
                    # Stop debug session if active
                    if stability_manager.debug_manager is not None:
                        stability_manager.stop_debug_session()
                    
                    # Get debug summary
                    debug_summary = stability_manager.get_debug_summary()
                    if debug_summary.get('debug_enabled', False):
                        print(f"\n  Debug Summary:")
                        print(f"    Session Active: {debug_summary.get('session_active', False)}")
                        print(f"    Total Iterations: {debug_summary.get('total_iterations', 0)}")
                        print(f"    Anomalies Detected: {debug_summary.get('anomalies_detected', 0)}")
                        print(f"    Gradient Checks: {debug_summary.get('gradient_checks', 0)}")
                        print(f"    Memory Checks: {debug_summary.get('memory_checks', 0)}")
                    
                    # Save and plot results for this configuration
                    stability_manager.save_histories()
                    stability_manager.plot_stability_history(f"stability_history_{full_config_name}.png")
                    
                    results[full_config_name] = {
                        'clipping_config': clipping_config,
                        'nan_config': nan_config,
                        'debug_config': debug_config,
                        'scenarios': config_results,
                        'debug_summary': debug_summary
                    }
    
    return results


def create_training_wrapper(clipping_config: GradientClippingConfig, 
                           nan_config: NaNHandlingConfig,
                           debug_config: PyTorchDebuggingConfig = None,
                           performance_config: PerformanceOptimizationConfig = None):
    """Create a training wrapper that automatically applies numerical stability measures and performance optimization."""
    
    class TrainingWrapper:
        def __init__(self):
            self.stability_manager = NumericalStabilityManager(
                clipping_config, 
                nan_config,
                debug_config,
                performance_config
            )
            self.step_count = 0
            
            # Start debug session if debugging is enabled
            if debug_config is not None:
                self.stability_manager.start_debug_session(f"training_wrapper_{int(time.time())}")
        
        def __call__(self, model: nn.Module, loss: torch.Tensor, 
                     optimizer: Optimizer, **kwargs):
            """Wrapper for training step with automatic stability measures and performance optimization."""
            self.step_count += 1
            
            # Apply numerical stability measures
            stability_result = self.stability_manager.step(model, loss, optimizer)
            
            # Log stability information
            if stability_result['nan_stats']['nan_detected'] or \
               stability_result['nan_stats']['inf_detected'] or \
               stability_result['nan_stats']['overflow_detected']:
                print(f"Step {self.step_count}: Numerical issues detected - "
                      f"NaN: {stability_result['nan_stats']['nan_detected']}, "
                      f"Inf: {stability_result['nan_stats']['inf_detected']}, "
                      f"Overflow: {stability_result['nan_stats']['overflow_detected']}")
            
            # Log debug information if available
            if stability_result.get('debug_info', {}).get('debug_enabled', False):
                debug_info = stability_result['debug_info']
                if debug_info.get('anomalies_detected', False):
                    print(f"Step {self.step_count}: Debug anomalies detected - "
                          f"{debug_info.get('anomaly_details', [])}")
            
            # Log performance metrics if available
            if stability_result.get('performance_metrics'):
                perf_metrics = stability_result['performance_metrics']
                print(f"Step {self.step_count}: Performance - "
                      f"Memory: {perf_metrics.get('memory_usage_percent', 0):.1f}%, "
                      f"GPU: {perf_metrics.get('gpu_memory_mb', 0):.1f}MB")
            
            return stability_result
        
        def save_histories(self):
            """Save all stability histories."""
            self.stability_manager.save_histories()
        
        def plot_histories(self, save_path: Optional[str] = None):
            """Plot stability histories."""
            self.stability_manager.plot_stability_history(save_path)
        
        def get_debug_summary(self):
            """Get debugging summary if available."""
            return self.stability_manager.get_debug_summary()
        
        def get_performance_summary(self):
            """Get performance optimization summary if available."""
            return self.stability_manager.get_performance_summary()
        
        def stop_debug_session(self):
            """Stop debugging session if active."""
            if self.stability_manager.debug_manager is not None:
                self.stability_manager.stop_debug_session()
        
        def __del__(self):
            """Cleanup when wrapper is destroyed."""
            try:
                self.stop_debug_session()
            except:
                pass
    
    return TrainingWrapper()


if __name__ == "__main__":
    # Demonstrate gradient clipping and NaN handling
    results = demonstrate_gradient_clipping_nan_handling()
    print("\nGradient clipping and NaN handling demonstration completed!")
    
    # Example of using the training wrapper with debugging and performance optimization
    print("\n" + "=" * 50)
    print("Training Wrapper with PyTorch Debugging and Performance Optimization Example")
    print("=" * 50)
    
    # Create debugging configuration
    debug_config = PyTorchDebuggingConfig(
        enable_autograd_anomaly=True,
        autograd_anomaly_mode="detect",
        enable_grad_check=True,
        enable_memory_debugging=True,
        debug_level="info",
        verbose_logging=True,
        save_debug_info=True,
        debug_output_dir="debug_output"
    )
    
    # Create performance optimization configuration
    performance_config = PerformanceOptimizationConfig(
        enable_performance_optimization=True,
        optimization_level="advanced",
        integrate_with_stability=True,
        enable_mixed_precision=True,
        enable_model_compilation=True,
        enable_memory_optimization=True,
        enable_multi_gpu=True,
        multi_gpu_mode="auto",
        multi_gpu_backend="nccl",
        multi_gpu_find_unused_parameters=False,
        multi_gpu_bucket_cap_mb=25,
        multi_gpu_static_graph=True
    )
    
    # Create a training wrapper with debugging and performance optimization
    wrapper = create_training_wrapper(
        clipping_configs['adaptive'],
        nan_configs['adaptive'],
        debug_config,
        performance_config
    )
    
    print("Training wrapper with PyTorch debugging created successfully!")
    print("Use it in your training loop like this:")
    print("stability_result = wrapper(model, loss, optimizer)")
    print("wrapper.save_histories()  # Save at the end of training")
    print("wrapper.plot_histories()  # Plot results")
    print("debug_summary = wrapper.get_debug_summary()  # Get debug info")
    print("wrapper.stop_debug_session()  # Stop debugging when done")
    
    # Demonstrate debugging features
    print("\n" + "=" * 50)
    print("PyTorch Debugging Features Demonstration")
    print("=" * 50)
    
    # Create a simple model for demonstration
    class DebugDemoModel(nn.Module):
        def __init__(self):
            super(DebugDemoModel, self).__init__()
            self.fc = nn.Linear(5, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    model = DebugDemoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    print("Running training steps with debugging enabled...")
    
    # Run a few training steps to demonstrate debugging
    for step in range(3):
        try:
            # Generate data
            x = torch.randn(3, 5)
            y = torch.randn(3, 1)
            
            # Forward pass
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            # Backward pass
            loss.backward()
            
            # Apply stability measures (this will trigger debugging)
            result = wrapper(model, loss, optimizer)
            
            print(f"Step {step + 1}: Loss={loss.item():.4f}, "
                  f"Stability={result['stability_score']:.4f}")
            
            # Show debug info
            if result.get('debug_info', {}).get('debug_enabled', False):
                debug_info = result['debug_info']
                if debug_info.get('gradient_stats'):
                    grad_stats = debug_info['gradient_stats']
                    print(f"  Gradient Norm: {grad_stats.get('total_norm', 0.0):.4f}")
                    print(f"  Parameters with Grad: {grad_stats.get('parameters_with_grad', 0)}")
            
            optimizer.step()
            optimizer.zero_grad()
            
        except Exception as e:
            print(f"Step {step + 1}: Error - {e}")
    
    # Get final debug summary
    debug_summary = wrapper.get_debug_summary()
    if debug_summary.get('debug_enabled', False):
        print(f"\nFinal Debug Summary:")
        print(f"  Session Active: {debug_summary.get('session_active', False)}")
        print(f"  Total Iterations: {debug_summary.get('total_iterations', 0)}")
        print(f"  Anomalies Detected: {debug_summary.get('anomalies_detected', 0)}")
        print(f"  Gradient Checks: {debug_summary.get('gradient_checks', 0)}")
        print(f"  Memory Checks: {debug_summary.get('memory_checks', 0)}")
    
    # Stop debugging session
    wrapper.stop_debug_session()
    print("\nDebugging session stopped. Check 'debug_output' directory for detailed logs.")
    
    # Demonstrate multi-GPU capabilities
    print("\n" + "=" * 50)
    print("Multi-GPU Training Features Demonstration")
    print("=" * 50)
    
    # Get multi-GPU status
    multi_gpu_status = wrapper.get_multi_gpu_status()
    print(f"Multi-GPU Status: {multi_gpu_status.get('current_mode', 'none')}")
    print(f"Device Count: {multi_gpu_status.get('device_count', 0)}")
    
    if multi_gpu_status.get('device_count', 0) > 1:
        print("Multi-GPU training is available!")
        
        # Show optimal batch size
        optimal_batch_size = wrapper.get_optimal_batch_size(32)
        print(f"Optimal batch size for multi-GPU: {optimal_batch_size} (base: 32)")
        
        # Demonstrate model wrapping
        print("\nDemonstrating model wrapping for multi-GPU...")
        wrapped_model = wrapper.wrap_model_for_multi_gpu(model)
        if wrapped_model != model:
            print("✅ Model successfully wrapped for multi-GPU training")
            print(f"   Model device: {next(wrapped_model.parameters()).device}")
        else:
            print("ℹ️  Model not wrapped (single GPU or wrapping failed)")
        
        # Show GPU memory stats
        gpu_memory = multi_gpu_status.get('gpu_memory', [])
        if gpu_memory:
            print("\nGPU Memory Status:")
            for gpu_info in gpu_memory:
                print(f"   GPU {gpu_info['device']}: "
                      f"Allocated: {gpu_info['allocated_mb']:.1f}MB, "
                      f"Free: {gpu_info['free_mb']:.1f}MB")
    else:
        print("Single GPU or no CUDA devices available")
    
    print("\n✅ Multi-GPU demonstration completed!")