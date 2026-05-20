from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
from torch.autograd import detect_anomaly, gradcheck
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from enum import Enum
import time
import traceback
from typing import Any, List, Dict, Optional
import asyncio
"""
PyTorch debugging utilities for ads generation features.
Provides advanced debugging tools for tensor analysis, gradient monitoring, and anomaly detection.
"""

logger = logging.getLogger(__name__)

class DebugLevel(Enum):
    """Debug levels for PyTorch debugging."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXTREME = "extreme"

@dataclass
class TensorDebugInfo:
    """Debug information for a tensor."""
    name: str
    shape: List[int]
    dtype: str
    device: str
    requires_grad: bool
    has_nan: bool
    has_inf: bool
    has_neg_inf: bool
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    std_value: Optional[float]
    num_elements: int
    memory_usage: Optional[float] = None

@dataclass
class GradientDebugInfo:
    """Debug information for gradients."""
    param_name: str
    grad_norm: float
    has_nan: bool
    has_inf: bool
    grad_mean: float
    grad_std: float
    grad_min: float
    grad_max: float

class PyTorchDebugger:
    """Advanced PyTorch debugging utilities."""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.BASIC):
        """Initialize PyTorch debugger."""
        self.debug_level = debug_level
        self.autograd_context = None
        self.debug_history = []
        self.anomaly_count = 0
        
    @contextmanager
    def autograd_anomaly_detection(self) -> Any:
        """Context manager for autograd anomaly detection."""
        try:
            self.autograd_context = detect_anomaly()
            self.autograd_context.__enter__()
            logger.info("PyTorch autograd anomaly detection enabled")
            yield
        except Exception as e:
            logger.error(f"Autograd anomaly detected: {e}")
            self.anomaly_count += 1
            raise
        finally:
            if self.autograd_context:
                self.autograd_context.__exit__(None, None, None)
                self.autograd_context = None
                logger.info("PyTorch autograd anomaly detection disabled")
    
    def analyze_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> TensorDebugInfo:
        """Analyze a tensor for anomalies and statistics."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # Basic tensor info
        debug_info = TensorDebugInfo(
            name=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            requires_grad=tensor.requires_grad,
            has_nan=torch.isnan(tensor).any().item(),
            has_inf=torch.isinf(tensor).any().item(),
            has_neg_inf=torch.isneginf(tensor).any().item(),
            min_value=tensor.min().item() if tensor.numel() > 0 else None,
            max_value=tensor.max().item() if tensor.numel() > 0 else None,
            mean_value=tensor.mean().item() if tensor.numel() > 0 else None,
            std_value=tensor.std().item() if tensor.numel() > 0 else None,
            num_elements=tensor.numel()
        )
        
        # Calculate memory usage
        if tensor.numel() > 0:
            debug_info.memory_usage = tensor.element_size() * tensor.numel() / (1024 * 1024)  # MB
        
        # Log anomalies
        if debug_info.has_nan or debug_info.has_inf or debug_info.has_neg_inf:
            logger.warning(f"Tensor anomalies detected in {name}: {debug_info}")
            self.debug_history.append({
                "type": "tensor_anomaly",
                "tensor_name": name,
                "info": debug_info,
                "timestamp": time.time()
            })
        
        return debug_info
    
    def analyze_model_parameters(self, model: nn.Module) -> Dict[str, TensorDebugInfo]:
        """Analyze all parameters in a model."""
        param_analysis = {}
        
        for name, param in model.named_parameters():
            param_analysis[name] = self.analyze_tensor(param, name)
        
        return param_analysis
    
    def analyze_gradients(self, model: nn.Module) -> Dict[str, GradientDebugInfo]:
        """Analyze gradients of all parameters in a model."""
        gradient_analysis = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                gradient_analysis[name] = GradientDebugInfo(
                    param_name=name,
                    grad_norm=grad.norm().item(),
                    has_nan=torch.isnan(grad).any().item(),
                    has_inf=torch.isinf(grad).any().item(),
                    grad_mean=grad.mean().item(),
                    grad_std=grad.std().item(),
                    grad_min=grad.min().item(),
                    grad_max=grad.max().item()
                )
                
                # Log gradient anomalies
                if gradient_analysis[name].has_nan or gradient_analysis[name].has_inf:
                    logger.warning(f"Gradient anomalies detected in {name}: {gradient_analysis[name]}")
                    self.debug_history.append({
                        "type": "gradient_anomaly",
                        "param_name": name,
                        "info": gradient_analysis[name],
                        "timestamp": time.time()
                    })
        
        return gradient_analysis
    
    def check_gradient_explosion(self, model: nn.Module, threshold: float = 10.0) -> List[str]:
        """Check for gradient explosion in model parameters."""
        exploded_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > threshold:
                    exploded_params.append(name)
                    logger.warning(f"Gradient explosion detected in {name}: norm = {grad_norm}")
        
        return exploded_params
    
    def check_gradient_vanishing(self, model: nn.Module, threshold: float = 1e-6) -> List[str]:
        """Check for gradient vanishing in model parameters."""
        vanished_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < threshold:
                    vanished_params.append(name)
                    logger.warning(f"Gradient vanishing detected in {name}: norm = {grad_norm}")
        
        return vanished_params
    
    def monitor_training_step(self, model: nn.Module, loss: torch.Tensor, step: int):
        """Monitor a training step for anomalies."""
        # Analyze loss
        loss_info = self.analyze_tensor(loss, f"loss_step_{step}")
        
        # Analyze gradients after backward pass
        gradient_info = self.analyze_gradients(model)
        
        # Check for gradient explosion/vanishing
        exploded_params = self.check_gradient_explosion(model)
        vanished_params = self.check_gradient_vanishing(model)
        
        # Log summary
        if loss_info.has_nan or loss_info.has_inf:
            logger.error(f"Loss anomaly at step {step}: {loss_info}")
        
        if exploded_params or vanished_params:
            logger.warning(f"Gradient issues at step {step}: exploded={exploded_params}, vanished={vanished_params}")
        
        # Store debug info
        self.debug_history.append({
            "type": "training_step",
            "step": step,
            "loss_info": loss_info,
            "gradient_info": gradient_info,
            "exploded_params": exploded_params,
            "vanished_params": vanished_params,
            "timestamp": time.time()
        })
    
    def validate_model_outputs(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validate model outputs for anomalies."""
        model.eval()
        
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                
                validation_results = {
                    "success": True,
                    "outputs_analyzed": {}
                }
                
                # Analyze outputs
                if hasattr(outputs, 'logits'):
                    validation_results["outputs_analyzed"]["logits"] = self.analyze_tensor(
                        outputs.logits, "model_output_logits"
                    )
                
                if hasattr(outputs, 'loss'):
                    validation_results["outputs_analyzed"]["loss"] = self.analyze_tensor(
                        outputs.loss, "model_output_loss"
                    )
                
                # Check for anomalies in outputs
                for name, info in validation_results["outputs_analyzed"].items():
                    if info.has_nan or info.has_inf:
                        validation_results["success"] = False
                        logger.error(f"Model output anomaly in {name}: {info}")
                
                return validation_results
                
            except Exception as e:
                logger.error(f"Model validation failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
    
    def profile_memory_usage(self, model: nn.Module) -> Dict[str, float]:
        """Profile memory usage of model parameters."""
        memory_profile = {}
        total_memory = 0
        
        for name, param in model.named_parameters():
            param_memory = param.element_size() * param.numel() / (1024 * 1024)  # MB
            memory_profile[name] = param_memory
            total_memory += param_memory
        
        memory_profile["total"] = total_memory
        
        # Log memory usage
        logger.info(f"Model memory usage: {total_memory:.2f} MB")
        
        return memory_profile
    
    def check_model_consistency(self, model: nn.Module) -> Dict[str, Any]:
        """Check model consistency and integrity."""
        consistency_report = {
            "valid": True,
            "issues": [],
            "parameter_count": 0,
            "trainable_parameters": 0
        }
        
        for name, param in model.named_parameters():
            consistency_report["parameter_count"] += 1
            
            if param.requires_grad:
                consistency_report["trainable_parameters"] += 1
            
            # Check parameter consistency
            param_info = self.analyze_tensor(param, name)
            
            if param_info.has_nan or param_info.has_inf:
                consistency_report["valid"] = False
                consistency_report["issues"].append({
                    "type": "parameter_anomaly",
                    "parameter": name,
                    "info": param_info
                })
        
        return consistency_report
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get a summary of all debug information."""
        return {
            "total_anomalies": self.anomaly_count,
            "debug_history_length": len(self.debug_history),
            "recent_anomalies": [
                entry for entry in self.debug_history[-10:] 
                if "anomaly" in entry.get("type", "")
            ],
            "debug_level": self.debug_level.value
        }
    
    def clear_debug_history(self) -> Any:
        """Clear debug history."""
        self.debug_history.clear()
        self.anomaly_count = 0
        logger.info("Debug history cleared")

class DiffusionModelDebugger(PyTorchDebugger):
    """Specialized debugger for diffusion models."""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.BASIC):
        
    """__init__ function."""
super().__init__(debug_level)
    
    def analyze_diffusion_pipeline(self, pipeline) -> Dict[str, Any]:
        """Analyze diffusion pipeline components."""
        pipeline_analysis = {
            "components": {},
            "total_parameters": 0,
            "memory_usage": 0
        }
        
        # Analyze UNet
        if hasattr(pipeline, 'unet'):
            pipeline_analysis["components"]["unet"] = self.analyze_model_parameters(pipeline.unet)
            pipeline_analysis["total_parameters"] += sum(p.numel() for p in pipeline.unet.parameters())
        
        # Analyze text encoder
        if hasattr(pipeline, 'text_encoder'):
            pipeline_analysis["components"]["text_encoder"] = self.analyze_model_parameters(pipeline.text_encoder)
            pipeline_analysis["total_parameters"] += sum(p.numel() for p in pipeline.text_encoder.parameters())
        
        # Analyze VAE
        if hasattr(pipeline, 'vae'):
            pipeline_analysis["components"]["vae"] = self.analyze_model_parameters(pipeline.vae)
            pipeline_analysis["total_parameters"] += sum(p.numel() for p in pipeline.vae.parameters())
        
        # Calculate total memory usage
        for component_name, component_params in pipeline_analysis["components"].items():
            for param_name, param_info in component_params.items():
                if param_info.memory_usage:
                    pipeline_analysis["memory_usage"] += param_info.memory_usage
        
        return pipeline_analysis
    
    def monitor_diffusion_step(self, pipeline, step: int, latents: torch.Tensor = None):
        """Monitor a diffusion step for anomalies."""
        step_info = {
            "step": step,
            "timestamp": time.time(),
            "latents_analysis": None,
            "pipeline_analysis": None
        }
        
        # Analyze latents if provided
        if latents is not None:
            step_info["latents_analysis"] = self.analyze_tensor(latents, f"latents_step_{step}")
        
        # Analyze pipeline components
        step_info["pipeline_analysis"] = self.analyze_diffusion_pipeline(pipeline)
        
        # Check for anomalies
        has_anomalies = False
        if step_info["latents_analysis"]:
            if step_info["latents_analysis"].has_nan or step_info["latents_analysis"].has_inf:
                has_anomalies = True
                logger.warning(f"Latent anomalies detected at step {step}")
        
        step_info["has_anomalies"] = has_anomalies
        
        # Store in debug history
        self.debug_history.append(step_info)
        
        return step_info

class TrainingDebugger(PyTorchDebugger):
    """Specialized debugger for training processes."""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.BASIC):
        
    """__init__ function."""
super().__init__(debug_level)
        self.training_stats = {
            "total_steps": 0,
            "anomaly_steps": 0,
            "gradient_explosions": 0,
            "gradient_vanishing": 0,
            "loss_anomalies": 0
        }
    
    def monitor_training_step(self, model: nn.Module, loss: torch.Tensor, step: int):
        """Enhanced training step monitoring."""
        self.training_stats["total_steps"] += 1
        
        # Analyze loss
        loss_info = self.analyze_tensor(loss, f"loss_step_{step}")
        if loss_info.has_nan or loss_info.has_inf:
            self.training_stats["loss_anomalies"] += 1
            self.training_stats["anomaly_steps"] += 1
        
        # Analyze gradients
        gradient_info = self.analyze_gradients(model)
        
        # Check for gradient issues
        exploded_params = self.check_gradient_explosion(model)
        vanished_params = self.check_gradient_vanishing(model)
        
        if exploded_params:
            self.training_stats["gradient_explosions"] += 1
            self.training_stats["anomaly_steps"] += 1
        
        if vanished_params:
            self.training_stats["gradient_vanishing"] += 1
            self.training_stats["anomaly_steps"] += 1
        
        # Store detailed info
        step_info = {
            "type": "training_step",
            "step": step,
            "loss_info": loss_info,
            "gradient_info": gradient_info,
            "exploded_params": exploded_params,
            "vanished_params": vanished_params,
            "timestamp": time.time()
        }
        
        self.debug_history.append(step_info)
        
        return step_info
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training-specific debug summary."""
        base_summary = self.get_debug_summary()
        base_summary.update(self.training_stats)
        
        # Calculate anomaly rates
        if base_summary["total_steps"] > 0:
            base_summary["anomaly_rate"] = base_summary["anomaly_steps"] / base_summary["total_steps"]
            base_summary["loss_anomaly_rate"] = base_summary["loss_anomalies"] / base_summary["total_steps"]
            base_summary["gradient_explosion_rate"] = base_summary["gradient_explosions"] / base_summary["total_steps"]
            base_summary["gradient_vanishing_rate"] = base_summary["gradient_vanishing"] / base_summary["total_steps"]
        
        return base_summary

# Utility functions
def enable_autograd_debugging():
    """Enable PyTorch autograd debugging globally."""
    torch.autograd.set_detect_anomaly(True)
    logger.info("PyTorch autograd debugging enabled globally")

def disable_autograd_debugging():
    """Disable PyTorch autograd debugging globally."""
    torch.autograd.set_detect_anomaly(False)
    logger.info("PyTorch autograd debugging disabled globally")

@contextmanager
def debug_context(debug_level: DebugLevel = DebugLevel.BASIC):
    """Context manager for PyTorch debugging."""
    debugger = PyTorchDebugger(debug_level)
    try:
        yield debugger
    finally:
        summary = debugger.get_debug_summary()
        logger.info(f"Debug session summary: {summary}") 