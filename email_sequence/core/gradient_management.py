from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, List, Dict, Optional
import asyncio
"""
Gradient Management System

Comprehensive gradient clipping and NaN/Inf value handling for stable training
of email sequence models with monitoring and debugging capabilities.
"""



logger = logging.getLogger(__name__)


@dataclass
class GradientConfig:
    """Configuration for gradient management"""
    
    # Gradient clipping
    enable_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    clip_type: str = "norm"  # "norm", "value", "adaptive"
    
    # NaN/Inf handling
    enable_nan_inf_check: bool = True
    nan_inf_threshold: float = 1e-6
    replace_nan_with: float = 0.0
    replace_inf_with: float = 1e6
    
    # Monitoring
    enable_gradient_monitoring: bool = True
    log_gradient_stats: bool = True
    save_gradient_plots: bool = True
    
    # Adaptive clipping
    adaptive_clipping: bool = False
    adaptive_window_size: int = 100
    adaptive_percentile: float = 95.0
    
    # Debugging
    debug_mode: bool = False
    verbose_logging: bool = False
    
    # Performance
    check_frequency: int = 1  # Check every N steps
    max_gradient_history: int = 1000


class GradientMonitor:
    """Monitor gradient statistics and detect issues"""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.gradient_history = defaultdict(lambda: deque(maxlen=config.max_gradient_history))
        self.nan_inf_counts = defaultdict(int)
        self.clipping_counts = defaultdict(int)
        self.step_count = 0
        
        # Statistics tracking
        self.grad_norms = deque(maxlen=config.max_gradient_history)
        self.grad_means = deque(maxlen=config.max_gradient_history)
        self.grad_stds = deque(maxlen=config.max_gradient_history)
        self.grad_maxs = deque(maxlen=config.max_gradient_history)
        self.grad_mins = deque(maxlen=config.max_gradient_history)
        
        logger.info("Gradient Monitor initialized")
    
    def update_statistics(self, gradients: List[torch.Tensor]) -> Dict[str, float]:
        """Update gradient statistics"""
        
        if not gradients:
            return {}
        
        # Calculate overall gradient norm
        total_norm = torch.norm(torch.stack([torch.norm(grad.detach()) for grad in gradients if grad is not None]))
        
        # Calculate statistics across all gradients
        all_gradients = torch.cat([grad.detach().flatten() for grad in gradients if grad is not None])
        
        stats = {
            "total_norm": total_norm.item(),
            "mean": all_gradients.mean().item(),
            "std": all_gradients.std().item(),
            "max": all_gradients.max().item(),
            "min": all_gradients.min().item(),
            "nan_count": torch.isnan(all_gradients).sum().item(),
            "inf_count": torch.isinf(all_gradients).sum().item(),
            "zero_count": (all_gradients == 0).sum().item(),
            "total_params": all_gradients.numel()
        }
        
        # Store in history
        self.grad_norms.append(stats["total_norm"])
        self.grad_means.append(stats["mean"])
        self.grad_stds.append(stats["std"])
        self.grad_maxs.append(stats["max"])
        self.grad_mins.append(stats["min"])
        
        self.step_count += 1
        
        return stats
    
    def check_gradient_health(self, gradients: List[torch.Tensor]) -> Dict[str, Any]:
        """Check gradient health and detect issues"""
        
        health_report = {
            "healthy": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        if not gradients:
            health_report["warnings"].append("No gradients provided")
            return health_report
        
        # Check for NaN/Inf values
        nan_count = 0
        inf_count = 0
        
        for i, grad in enumerate(gradients):
            if grad is None:
                continue
            
            grad_nan = torch.isnan(grad).sum().item()
            grad_inf = torch.isinf(grad).sum().item()
            
            nan_count += grad_nan
            inf_count += grad_inf
            
            if grad_nan > 0:
                health_report["warnings"].append(f"NaN values detected in gradient {i}: {grad_nan}")
                self.nan_inf_counts[f"grad_{i}_nan"] += 1
            
            if grad_inf > 0:
                health_report["warnings"].append(f"Inf values detected in gradient {i}: {grad_inf}")
                self.nan_inf_counts[f"grad_{i}_inf"] += 1
        
        # Check gradient norms
        total_norm = torch.norm(torch.stack([torch.norm(grad.detach()) for grad in gradients if grad is not None]))
        
        if total_norm > 10.0:
            health_report["warnings"].append(f"Large gradient norm: {total_norm:.4f}")
            health_report["recommendations"].append("Consider reducing learning rate or increasing gradient clipping")
        
        if total_norm < 1e-8:
            health_report["warnings"].append(f"Very small gradient norm: {total_norm:.4f}")
            health_report["recommendations"].append("Check if model is stuck or learning rate is too small")
        
        # Check for gradient explosion
        if len(self.grad_norms) > 10:
            recent_norms = list(self.grad_norms)[-10:]
            norm_increase = recent_norms[-1] / (np.mean(recent_norms[:-1]) + 1e-8)
            
            if norm_increase > 10.0:
                health_report["warnings"].append(f"Gradient explosion detected: {norm_increase:.2f}x increase")
                health_report["recommendations"].append("Reduce learning rate or increase gradient clipping")
        
        # Check for vanishing gradients
        if total_norm < 1e-6:
            health_report["warnings"].append("Vanishing gradients detected")
            health_report["recommendations"].append("Check model architecture or increase learning rate")
        
        # Determine overall health
        if health_report["errors"]:
            health_report["healthy"] = False
        elif health_report["warnings"]:
            health_report["healthy"] = False
        
        return health_report
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of gradient statistics"""
        
        if not self.grad_norms:
            return {"message": "No gradient statistics available"}
        
        summary = {
            "total_steps": self.step_count,
            "gradient_norms": {
                "mean": np.mean(self.grad_norms),
                "std": np.std(self.grad_norms),
                "min": np.min(self.grad_norms),
                "max": np.max(self.grad_norms),
                "median": np.median(self.grad_norms)
            },
            "gradient_means": {
                "mean": np.mean(self.grad_means),
                "std": np.std(self.grad_means)
            },
            "gradient_stds": {
                "mean": np.mean(self.grad_stds),
                "std": np.std(self.grad_stds)
            },
            "nan_inf_counts": dict(self.nan_inf_counts),
            "clipping_counts": dict(self.clipping_counts)
        }
        
        return summary
    
    def plot_gradient_statistics(self, save_path: Optional[str] = None):
        """Plot gradient statistics"""
        
        if not self.grad_norms:
            logger.warning("No gradient statistics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot gradient norms over time
        axes[0, 0].plot(list(self.grad_norms), 'b-', linewidth=1)
        axes[0, 0].set_title("Gradient Norms Over Time")
        axes[0, 0].set_xlabel("Training Step")
        axes[0, 0].set_ylabel("Gradient Norm")
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot gradient distribution
        all_norms = list(self.grad_norms)
        axes[0, 1].hist(all_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title("Gradient Norm Distribution")
        axes[0, 1].set_xlabel("Gradient Norm")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot gradient means and stds
        steps = range(len(self.grad_means))
        axes[1, 0].plot(steps, list(self.grad_means), 'g-', label='Mean', linewidth=1)
        axes[1, 0].plot(steps, list(self.grad_stds), 'r-', label='Std', linewidth=1)
        axes[1, 0].set_title("Gradient Statistics Over Time")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot max and min gradients
        axes[1, 1].plot(steps, list(self.grad_maxs), 'orange', label='Max', linewidth=1)
        axes[1, 1].plot(steps, list(self.grad_mins), 'purple', label='Min', linewidth=1)
        axes[1, 1].set_title("Gradient Extremes Over Time")
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class GradientClipper:
    """Advanced gradient clipping with multiple strategies"""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.clip_history = deque(maxlen=config.max_gradient_history)
        self.adaptive_thresholds = deque(maxlen=config.adaptive_window_size)
        
        logger.info(f"Gradient Clipper initialized with {config.clip_type} clipping")
    
    def clip_gradients(
        self,
        model: nn.Module,
        max_norm: Optional[float] = None
    ) -> Dict[str, Any]:
        """Clip gradients using specified method"""
        
        if not self.config.enable_gradient_clipping:
            return {"clipped": False, "total_norm": 0.0}
        
        # Get gradients
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        
        if not gradients:
            return {"clipped": False, "total_norm": 0.0}
        
        # Calculate total norm
        total_norm = torch.norm(torch.stack([torch.norm(grad.detach()) for grad in gradients]))
        
        # Determine clipping threshold
        if max_norm is None:
            max_norm = self.config.max_grad_norm
        
        if self.config.adaptive_clipping:
            max_norm = self._get_adaptive_threshold(total_norm.item())
        
        # Perform clipping
        clip_info = {
            "clipped": False,
            "total_norm": total_norm.item(),
            "max_norm": max_norm,
            "clip_ratio": 1.0
        }
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            
            for grad in gradients:
                grad.data.mul_(clip_coef)
            
            clip_info["clipped"] = True
            clip_info["clip_ratio"] = clip_coef.item()
            
            if self.config.verbose_logging:
                logger.info(f"Gradients clipped: norm {total_norm:.4f} -> {max_norm:.4f}")
        
        # Store clipping information
        self.clip_history.append(clip_info)
        
        return clip_info
    
    def clip_gradients_by_value(
        self,
        model: nn.Module,
        max_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """Clip gradients by value"""
        
        if not self.config.enable_gradient_clipping:
            return {"clipped": False, "clipped_count": 0}
        
        if max_value is None:
            max_value = self.config.max_grad_norm
        
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        
        clip_info = {
            "clipped": False,
            "clipped_count": 0,
            "max_value": max_value
        }
        
        for grad in gradients:
            if grad is not None:
                clipped = torch.clamp(grad, -max_value, max_value)
                if not torch.equal(grad, clipped):
                    grad.data.copy_(clipped)
                    clip_info["clipped"] = True
                    clip_info["clipped_count"] += 1
        
        return clip_info
    
    def _get_adaptive_threshold(self, current_norm: float) -> float:
        """Get adaptive clipping threshold based on recent history"""
        
        self.adaptive_thresholds.append(current_norm)
        
        if len(self.adaptive_thresholds) < 10:
            return self.config.max_grad_norm
        
        # Calculate percentile-based threshold
        recent_norms = list(self.adaptive_thresholds)
        threshold = np.percentile(recent_norms, self.config.adaptive_percentile)
        
        # Ensure threshold is within reasonable bounds
        threshold = max(0.1, min(threshold, 10.0))
        
        return threshold


class NaNInfHandler:
    """Handle NaN and Inf values in gradients and model parameters"""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.nan_inf_history = deque(maxlen=config.max_gradient_history)
        self.replacement_counts = defaultdict(int)
        
        logger.info("NaN/Inf Handler initialized")
    
    def check_and_fix_gradients(
        self,
        model: nn.Module,
        replace_values: bool = True
    ) -> Dict[str, Any]:
        """Check and fix NaN/Inf values in gradients"""
        
        if not self.config.enable_nan_inf_check:
            return {"nan_count": 0, "inf_count": 0, "fixed": False}
        
        nan_count = 0
        inf_count = 0
        fixed_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Count NaN/Inf values
            param_nan = torch.isnan(grad).sum().item()
            param_inf = torch.isinf(grad).sum().item()
            
            nan_count += param_nan
            inf_count += param_inf
            
            # Fix NaN/Inf values if requested
            if replace_values and (param_nan > 0 or param_inf > 0):
                fixed = self._replace_nan_inf_values(grad)
                if fixed:
                    fixed_count += 1
                    self.replacement_counts[name] += 1
                    
                    if self.config.verbose_logging:
                        logger.warning(f"Fixed NaN/Inf in {name}: {param_nan} NaN, {param_inf} Inf")
        
        result = {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "fixed": fixed_count > 0,
            "fixed_count": fixed_count
        }
        
        self.nan_inf_history.append(result)
        
        return result
    
    def check_and_fix_parameters(
        self,
        model: nn.Module,
        replace_values: bool = True
    ) -> Dict[str, Any]:
        """Check and fix NaN/Inf values in model parameters"""
        
        if not self.config.enable_nan_inf_check:
            return {"nan_count": 0, "inf_count": 0, "fixed": False}
        
        nan_count = 0
        inf_count = 0
        fixed_count = 0
        
        for name, param in model.named_parameters():
            # Count NaN/Inf values
            param_nan = torch.isnan(param.data).sum().item()
            param_inf = torch.isinf(param.data).sum().item()
            
            nan_count += param_nan
            inf_count += param_inf
            
            # Fix NaN/Inf values if requested
            if replace_values and (param_nan > 0 or param_inf > 0):
                fixed = self._replace_nan_inf_values(param.data)
                if fixed:
                    fixed_count += 1
                    self.replacement_counts[f"{name}_param"] += 1
                    
                    if self.config.verbose_logging:
                        logger.warning(f"Fixed NaN/Inf in parameter {name}: {param_nan} NaN, {param_inf} Inf")
        
        result = {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "fixed": fixed_count > 0,
            "fixed_count": fixed_count
        }
        
        return result
    
    def _replace_nan_inf_values(self, tensor: torch.Tensor) -> bool:
        """Replace NaN and Inf values in tensor"""
        
        original_tensor = tensor.clone()
        
        # Replace NaN values
        if torch.isnan(tensor).any():
            tensor[torch.isnan(tensor)] = self.config.replace_nan_with
        
        # Replace Inf values
        if torch.isinf(tensor).any():
            tensor[torch.isinf(tensor)] = self.config.replace_inf_with
        
        # Check if any changes were made
        return not torch.equal(original_tensor, tensor)
    
    def get_replacement_summary(self) -> Dict[str, Any]:
        """Get summary of NaN/Inf replacements"""
        
        return {
            "total_replacements": sum(self.replacement_counts.values()),
            "replacement_counts": dict(self.replacement_counts),
            "history_summary": {
                "total_checks": len(self.nan_inf_history),
                "total_nan_found": sum(check["nan_count"] for check in self.nan_inf_history),
                "total_inf_found": sum(check["inf_count"] for check in self.nan_inf_history),
                "total_fixed": sum(check["fixed_count"] for check in self.nan_inf_history)
            }
        }


class GradientManager:
    """Unified gradient management system"""
    
    def __init__(self, config: GradientConfig):
        
    """__init__ function."""
self.config = config
        self.monitor = GradientMonitor(config)
        self.clipper = GradientClipper(config)
        self.nan_inf_handler = NaNInfHandler(config)
        
        # Training state
        self.current_step = 0
        self.training_history = []
        
        logger.info("Gradient Manager initialized")
    
    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        backward: bool = True
    ) -> Dict[str, Any]:
        """Perform a complete gradient management step"""
        
        self.current_step += 1
        
        # Skip if not checking this step
        if self.current_step % self.config.check_frequency != 0:
            return {"step": self.current_step, "skipped": True}
        
        step_info = {
            "step": self.current_step,
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "timestamp": time.time()
        }
        
        # Backward pass if requested
        if backward and loss.requires_grad:
            loss.backward()
        
        # Get gradients
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        
        # Check gradient health
        health_report = self.monitor.check_gradient_health(gradients)
        step_info["health"] = health_report
        
        # Handle NaN/Inf values
        nan_inf_info = self.nan_inf_handler.check_and_fix_gradients(model)
        step_info["nan_inf"] = nan_inf_info
        
        # Clip gradients
        clip_info = self.clipper.clip_gradients(model)
        step_info["clipping"] = clip_info
        
        # Update statistics
        stats = self.monitor.update_statistics(gradients)
        step_info["statistics"] = stats
        
        # Log if verbose
        if self.config.verbose_logging:
            self._log_step_info(step_info)
        
        # Store in history
        self.training_history.append(step_info)
        
        return step_info
    
    def _log_step_info(self, step_info: Dict[str, Any]):
        """Log step information"""
        
        step = step_info["step"]
        loss = step_info["loss"]
        total_norm = step_info["statistics"]["total_norm"]
        
        logger.info(f"Step {step}: Loss={loss:.6f}, GradNorm={total_norm:.6f}")
        
        if not step_info["health"]["healthy"]:
            logger.warning(f"Step {step}: Gradient health issues detected")
            for warning in step_info["health"]["warnings"]:
                logger.warning(f"  - {warning}")
        
        if step_info["nan_inf"]["nan_count"] > 0 or step_info["nan_inf"]["inf_count"] > 0:
            logger.warning(f"Step {step}: NaN={step_info['nan_inf']['nan_count']}, "
                         f"Inf={step_info['nan_inf']['inf_count']}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        
        if not self.training_history:
            return {"message": "No training history available"}
        
        # Extract key metrics
        losses = [step["loss"] for step in self.training_history]
        grad_norms = [step["statistics"]["total_norm"] for step in self.training_history]
        
        # Calculate statistics
        summary = {
            "total_steps": len(self.training_history),
            "loss_statistics": {
                "mean": np.mean(losses),
                "std": np.std(losses),
                "min": np.min(losses),
                "max": np.max(losses),
                "final": losses[-1] if losses else 0.0
            },
            "gradient_statistics": self.monitor.get_statistics_summary(),
            "nan_inf_summary": self.nan_inf_handler.get_replacement_summary(),
            "health_issues": {
                "unhealthy_steps": sum(1 for step in self.training_history 
                                     if not step["health"]["healthy"]),
                "total_warnings": sum(len(step["health"]["warnings"]) 
                                    for step in self.training_history),
                "total_errors": sum(len(step["health"]["errors"]) 
                                  for step in self.training_history)
            }
        }
        
        return summary
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves including gradients"""
        
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        steps = [step["step"] for step in self.training_history]
        losses = [step["loss"] for step in self.training_history]
        grad_norms = [step["statistics"]["total_norm"] for step in self.training_history]
        nan_counts = [step["nan_inf"]["nan_count"] for step in self.training_history]
        inf_counts = [step["nan_inf"]["inf_count"] for step in self.training_history]
        
        # Plot loss
        axes[0, 0].plot(steps, losses, 'b-', linewidth=1)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot gradient norms
        axes[0, 1].plot(steps, grad_norms, 'r-', linewidth=1)
        axes[0, 1].set_title("Gradient Norms")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Gradient Norm")
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot NaN/Inf counts
        axes[1, 0].plot(steps, nan_counts, 'orange', label='NaN', linewidth=1)
        axes[1, 0].plot(steps, inf_counts, 'purple', label='Inf', linewidth=1)
        axes[1, 0].set_title("NaN/Inf Counts")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot clipping ratio
        clip_ratios = [step["clipping"]["clip_ratio"] for step in self.training_history]
        axes[1, 1].plot(steps, clip_ratios, 'g-', linewidth=1)
        axes[1, 1].set_title("Gradient Clipping Ratio")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Clip Ratio")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def save_training_log(self, file_path: str):
        """Save training log to file"""
        
        summary = self.get_training_summary()
        
        with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("Gradient Management Training Log\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("=" * 50 + "\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write(f"Total Steps: {summary['total_steps']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Final Loss: {summary['loss_statistics']['final']:.6f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Average Loss: {summary['loss_statistics']['mean']:.6f}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"Loss Std: {summary['loss_statistics']['std']:.6f}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("Health Issues:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"  Unhealthy Steps: {summary['health_issues']['unhealthy_steps']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"  Total Warnings: {summary['health_issues']['total_warnings']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"  Total Errors: {summary['health_issues']['total_errors']}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            f.write("NaN/Inf Summary:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            nan_summary = summary['nan_inf_summary']
            f.write(f"  Total Replacements: {nan_summary['total_replacements']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"  Total NaN Found: {nan_summary['history_summary']['total_nan_found']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(f"  Total Inf Found: {nan_summary['history_summary']['total_inf_found']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Training log saved to {file_path}")


# Utility functions for easy integration
def create_gradient_manager(
    max_grad_norm: float = 1.0,
    enable_monitoring: bool = True,
    enable_nan_inf_check: bool = True,
    verbose: bool = False
) -> GradientManager:
    """Create a gradient manager with common settings"""
    
    config = GradientConfig(
        max_grad_norm=max_grad_norm,
        enable_gradient_monitoring=enable_monitoring,
        enable_nan_inf_check=enable_nan_inf_check,
        verbose_logging=verbose
    )
    
    return GradientManager(config)


def safe_backward(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_manager: Optional[GradientManager] = None,
    **kwargs
) -> Dict[str, Any]:
    """Safe backward pass with gradient management"""
    
    if gradient_manager is None:
        gradient_manager = create_gradient_manager()
    
    return gradient_manager.step(model, optimizer, loss, backward=True, **kwargs) 