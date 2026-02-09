"""
PyTorch Debugging Tools for Video-OpusClip

Comprehensive PyTorch debugging utilities including:
- autograd.detect_anomaly() integration
- Gradient debugging and analysis
- Memory debugging for tensors
- Model debugging and inspection
- Training debugging with detailed monitoring
- CUDA debugging and optimization
- Performance profiling with PyTorch tools
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import gc
import psutil
import os
import json
import traceback
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings
import structlog

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PyTorchDebugConfig:
    """Configuration for PyTorch debugging tools."""
    # Autograd debugging
    enable_autograd_anomaly: bool = False
    autograd_anomaly_mode: str = "detect"  # "detect", "warn", "raise"
    
    # Gradient debugging
    enable_gradient_debugging: bool = False
    gradient_norm_threshold: float = 1e6
    gradient_clip_threshold: float = 1.0
    check_gradient_flow: bool = True
    
    # Memory debugging
    enable_memory_debugging: bool = False
    track_tensor_memory: bool = True
    memory_snapshot_frequency: int = 100
    
    # Model debugging
    enable_model_debugging: bool = False
    check_model_parameters: bool = True
    validate_model_inputs: bool = True
    
    # Training debugging
    enable_training_debugging: bool = False
    debug_loss_computation: bool = True
    debug_optimizer_steps: bool = True
    
    # CUDA debugging
    enable_cuda_debugging: bool = False
    cuda_memory_fraction: float = 0.9
    cuda_synchronize: bool = False
    
    # Profiling
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_cpu: bool = True
    profile_cuda: bool = True
    
    # Logging
    debug_log_level: str = "INFO"
    save_debug_reports: bool = True
    debug_output_dir: str = "debug_reports"

# =============================================================================
# AUTOGRAD ANOMALY DETECTION
# =============================================================================

class AutogradAnomalyDetector:
    """Enhanced autograd anomaly detection with detailed reporting."""
    
    def __init__(self, config: PyTorchDebugConfig):
        self.config = config
        self.anomaly_history = []
        self.is_enabled = False
        
    @contextmanager
    def detect_anomaly(self, mode: str = None):
        """Context manager for autograd anomaly detection."""
        if mode is None:
            mode = self.config.autograd_anomaly_mode
            
        if not self.config.enable_autograd_anomaly:
            yield
            return
            
        try:
            # Enable anomaly detection
            autograd.set_detect_anomaly(True)
            self.is_enabled = True
            logger.info(f"Autograd anomaly detection enabled with mode: {mode}")
            
            yield
            
        except Exception as e:
            # Capture anomaly information
            anomaly_info = {
                "timestamp": time.time(),
                "mode": mode,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "grad_fn": self._get_grad_fn_info(),
                "memory_usage": self._get_memory_usage()
            }
            
            self.anomaly_history.append(anomaly_info)
            logger.error(f"Autograd anomaly detected: {e}")
            
            if mode == "raise":
                raise
            elif mode == "warn":
                warnings.warn(f"Autograd anomaly: {e}")
                
        finally:
            # Disable anomaly detection
            autograd.set_detect_anomaly(False)
            self.is_enabled = False
            
    def _get_grad_fn_info(self) -> Dict[str, Any]:
        """Get information about the current grad_fn."""
        try:
            # This is a simplified version - in practice, you'd need to
            # capture the actual grad_fn from the exception context
            return {
                "grad_fn_type": "unknown",
                "requires_grad": True
            }
        except:
            return {"error": "Could not extract grad_fn info"}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        return {
            "cpu_memory": psutil.virtual_memory().percent,
            "gpu_memory": self._get_gpu_memory_usage() if torch.cuda.is_available() else None
        }
    
    def _get_gpu_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get GPU memory usage."""
        if not torch.cuda.is_available():
            return None
            
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Generate anomaly detection report."""
        return {
            "total_anomalies": len(self.anomaly_history),
            "anomalies": self.anomaly_history,
            "is_enabled": self.is_enabled,
            "config": {
                "enable_autograd_anomaly": self.config.enable_autograd_anomaly,
                "autograd_anomaly_mode": self.config.autograd_anomaly_mode
            }
        }

# =============================================================================
# GRADIENT DEBUGGING
# =============================================================================

class GradientDebugger:
    """Comprehensive gradient debugging and analysis."""
    
    def __init__(self, config: PyTorchDebugConfig):
        self.config = config
        self.gradient_history = []
        self.gradient_stats = {
            "total_gradients": 0,
            "nan_gradients": 0,
            "inf_gradients": 0,
            "large_gradients": 0
        }
        
    def check_gradients(self, model: nn.Module, step: int = 0) -> Dict[str, Any]:
        """Check gradients for anomalies."""
        if not self.config.enable_gradient_debugging:
            return {}
            
        gradient_info = {
            "step": step,
            "timestamp": time.time(),
            "parameters": {},
            "anomalies": [],
            "statistics": {}
        }
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                
                param_info = {
                    "norm": grad_norm,
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "min": grad.min().item(),
                    "max": grad.max().item(),
                    "shape": list(grad.shape)
                }
                
                # Check for anomalies
                anomalies = []
                if torch.isnan(grad).any():
                    anomalies.append("nan_gradients")
                    self.gradient_stats["nan_gradients"] += 1
                    
                if torch.isinf(grad).any():
                    anomalies.append("inf_gradients")
                    self.gradient_stats["inf_gradients"] += 1
                    
                if grad_norm > self.config.gradient_norm_threshold:
                    anomalies.append("large_gradients")
                    self.gradient_stats["large_gradients"] += 1
                
                param_info["anomalies"] = anomalies
                gradient_info["parameters"][name] = param_info
                
                total_norm += grad_norm ** 2
                param_count += 1
        
        # Calculate total gradient norm
        if param_count > 0:
            total_norm = total_norm ** 0.5
            gradient_info["statistics"]["total_norm"] = total_norm
            gradient_info["statistics"]["param_count"] = param_count
            
            # Check if total norm is too large
            if total_norm > self.config.gradient_norm_threshold:
                gradient_info["anomalies"].append("large_total_norm")
        
        self.gradient_history.append(gradient_info)
        self.gradient_stats["total_gradients"] += 1
        
        return gradient_info
    
    def analyze_gradient_flow(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze gradient flow through the model."""
        if not self.config.check_gradient_flow:
            return {}
            
        flow_analysis = {
            "layers": {},
            "bottlenecks": [],
            "vanishing_gradients": [],
            "exploding_gradients": []
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.grad is not None:
                    grad_norm = module.weight.grad.norm().item()
                    weight_norm = module.weight.norm().item()
                    
                    layer_info = {
                        "grad_norm": grad_norm,
                        "weight_norm": weight_norm,
                        "grad_weight_ratio": grad_norm / (weight_norm + 1e-8)
                    }
                    
                    flow_analysis["layers"][name] = layer_info
                    
                    # Detect vanishing gradients
                    if grad_norm < 1e-6:
                        flow_analysis["vanishing_gradients"].append(name)
                    
                    # Detect exploding gradients
                    if grad_norm > 10.0:
                        flow_analysis["exploding_gradients"].append(name)
        
        return flow_analysis
    
    def clip_gradients(self, model: nn.Module, max_norm: float = None) -> float:
        """Clip gradients and return total norm."""
        if max_norm is None:
            max_norm = self.config.gradient_clip_threshold
            
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm
        )
        
        if self.config.enable_gradient_debugging:
            logger.info(f"Gradients clipped. Total norm: {total_norm:.6f}")
        
        return total_norm
    
    def get_gradient_report(self) -> Dict[str, Any]:
        """Generate gradient debugging report."""
        return {
            "gradient_stats": self.gradient_stats,
            "recent_gradients": self.gradient_history[-10:] if self.gradient_history else [],
            "config": {
                "enable_gradient_debugging": self.config.enable_gradient_debugging,
                "gradient_norm_threshold": self.config.gradient_norm_threshold,
                "gradient_clip_threshold": self.config.gradient_clip_threshold
            }
        }

# =============================================================================
# MEMORY DEBUGGING
# =============================================================================

class PyTorchMemoryDebugger:
    """PyTorch-specific memory debugging and analysis."""
    
    def __init__(self, config: PyTorchDebugConfig):
        self.config = config
        self.memory_snapshots = []
        self.tensor_tracker = {}
        
    def take_memory_snapshot(self, label: str = None) -> Dict[str, Any]:
        """Take a comprehensive memory snapshot."""
        if not self.config.enable_memory_debugging:
            return {}
            
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "cpu_memory": self._get_cpu_memory_info(),
            "gpu_memory": self._get_gpu_memory_info() if torch.cuda.is_available() else None,
            "tensor_memory": self._get_tensor_memory_info() if self.config.track_tensor_memory else None
        }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def _get_cpu_memory_info(self) -> Dict[str, Any]:
        """Get CPU memory information."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / 1024**3,  # GB
            "available": memory.available / 1024**3,  # GB
            "used": memory.used / 1024**3,  # GB
            "percent": memory.percent
        }
    
    def _get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return None
            
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
        }
    
    def _get_tensor_memory_info(self) -> Dict[str, Any]:
        """Get information about tracked tensors."""
        tensor_info = {
            "total_tensors": len(self.tensor_tracker),
            "total_memory": 0.0,
            "tensors_by_device": {},
            "tensors_by_dtype": {}
        }
        
        for tensor_id, info in self.tensor_tracker.items():
            if info["tensor"] is not None:
                memory_size = info["tensor"].element_size() * info["tensor"].nelement()
                tensor_info["total_memory"] += memory_size
                
                device = str(info["tensor"].device)
                dtype = str(info["tensor"].dtype)
                
                if device not in tensor_info["tensors_by_device"]:
                    tensor_info["tensors_by_device"][device] = {"count": 0, "memory": 0.0}
                if dtype not in tensor_info["tensors_by_dtype"]:
                    tensor_info["tensors_by_dtype"][dtype] = {"count": 0, "memory": 0.0}
                
                tensor_info["tensors_by_device"][device]["count"] += 1
                tensor_info["tensors_by_device"][device]["memory"] += memory_size
                tensor_info["tensors_by_dtype"][dtype]["count"] += 1
                tensor_info["tensors_by_dtype"][dtype]["memory"] += memory_size
        
        return tensor_info
    
    def track_tensor(self, tensor: torch.Tensor, name: str = None):
        """Track a tensor for memory debugging."""
        if not self.config.track_tensor_memory:
            return
            
        tensor_id = id(tensor)
        self.tensor_tracker[tensor_id] = {
            "tensor": tensor,
            "name": name,
            "created_at": time.time(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad
        }
    
    def untrack_tensor(self, tensor: torch.Tensor):
        """Stop tracking a tensor."""
        if not self.config.track_tensor_memory:
            return
            
        tensor_id = id(tensor)
        if tensor_id in self.tensor_tracker:
            del self.tensor_tracker[tensor_id]
    
    def clear_memory(self):
        """Clear memory and garbage collect."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if self.config.enable_memory_debugging:
            logger.info("Memory cleared and garbage collected")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory debugging report."""
        return {
            "snapshots": self.memory_snapshots,
            "tensor_tracker": self.tensor_tracker,
            "config": {
                "enable_memory_debugging": self.config.enable_memory_debugging,
                "track_tensor_memory": self.config.track_tensor_memory
            }
        }

# =============================================================================
# MODEL DEBUGGING
# =============================================================================

class ModelDebugger:
    """Model debugging and inspection tools."""
    
    def __init__(self, config: PyTorchDebugConfig):
        self.config = config
        self.model_inspections = []
        
    def inspect_model(self, model: nn.Module, input_shape: Tuple[int, ...] = None) -> Dict[str, Any]:
        """Comprehensive model inspection."""
        if not self.config.enable_model_debugging:
            return {}
            
        inspection = {
            "timestamp": time.time(),
            "model_info": self._get_model_info(model),
            "parameter_info": self._get_parameter_info(model),
            "layer_info": self._get_layer_info(model),
            "input_validation": self._validate_model_inputs(model, input_shape) if input_shape else None
        }
        
        self.model_inspections.append(inspection)
        return inspection
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get basic model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_type": type(model).__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "modules_count": len(list(model.modules())),
            "device": next(model.parameters()).device if list(model.parameters()) else "cpu"
        }
    
    def _get_parameter_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get detailed parameter information."""
        param_info = {
            "parameters": {},
            "statistics": {
                "total_params": 0,
                "zero_params": 0,
                "nan_params": 0,
                "inf_params": 0
            }
        }
        
        for name, param in model.named_parameters():
            param_data = param.data
            
            info = {
                "shape": list(param_data.shape),
                "dtype": str(param_data.dtype),
                "device": str(param_data.device),
                "requires_grad": param.requires_grad,
                "norm": param_data.norm().item(),
                "mean": param_data.mean().item(),
                "std": param_data.std().item(),
                "min": param_data.min().item(),
                "max": param_data.max().item()
            }
            
            param_info["parameters"][name] = info
            param_info["statistics"]["total_params"] += param_data.numel()
            
            # Check for anomalies
            if (param_data == 0).all():
                param_info["statistics"]["zero_params"] += 1
            if torch.isnan(param_data).any():
                param_info["statistics"]["nan_params"] += 1
            if torch.isinf(param_data).any():
                param_info["statistics"]["inf_params"] += 1
        
        return param_info
    
    def _get_layer_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about model layers."""
        layer_info = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_info[name] = {
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "trainable": any(p.requires_grad for p in module.parameters())
                }
        
        return layer_info
    
    def _validate_model_inputs(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Validate model inputs."""
        if not self.config.validate_model_inputs:
            return {}
            
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            return {
                "input_shape": input_shape,
                "output_shape": list(output.shape) if hasattr(output, 'shape') else str(type(output)),
                "forward_pass_successful": True
            }
            
        except Exception as e:
            return {
                "input_shape": input_shape,
                "forward_pass_successful": False,
                "error": str(e)
            }
    
    def get_model_report(self) -> Dict[str, Any]:
        """Generate model debugging report."""
        return {
            "inspections": self.model_inspections,
            "config": {
                "enable_model_debugging": self.config.enable_model_debugging,
                "check_model_parameters": self.config.check_model_parameters,
                "validate_model_inputs": self.config.validate_model_inputs
            }
        }

# =============================================================================
# TRAINING DEBUGGING
# =============================================================================

class TrainingDebugger:
    """Training-specific debugging tools."""
    
    def __init__(self, config: PyTorchDebugConfig):
        self.config = config
        self.training_debug_info = []
        
    def debug_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        step: int = 0
    ) -> Dict[str, Any]:
        """Debug a single training step."""
        if not self.config.enable_training_debugging:
            return {}
            
        debug_info = {
            "step": step,
            "timestamp": time.time(),
            "inputs": self._analyze_tensor(inputs, "inputs"),
            "targets": self._analyze_tensor(targets, "targets"),
            "loss_computation": self._debug_loss_computation(model, loss_fn, inputs, targets) if self.config.debug_loss_computation else None,
            "optimizer_state": self._debug_optimizer_state(optimizer) if self.config.debug_optimizer_steps else None
        }
        
        self.training_debug_info.append(debug_info)
        return debug_info
    
    def _analyze_tensor(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Analyze tensor properties."""
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "norm": tensor.norm().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item()
        }
    
    def _debug_loss_computation(self, model: nn.Module, loss_fn: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Debug loss computation."""
        try:
            with torch.enable_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                return {
                    "outputs": self._analyze_tensor(outputs, "outputs"),
                    "loss_value": loss.item(),
                    "loss_grad_fn": str(loss.grad_fn) if loss.grad_fn else None,
                    "computation_successful": True
                }
        except Exception as e:
            return {
                "computation_successful": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _debug_optimizer_state(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Debug optimizer state."""
        state_info = {
            "optimizer_type": type(optimizer).__name__,
            "param_groups": []
        }
        
        for i, param_group in enumerate(optimizer.param_groups):
            group_info = {
                "group_index": i,
                "lr": param_group.get('lr', None),
                "weight_decay": param_group.get('weight_decay', None),
                "momentum": param_group.get('momentum', None),
                "params_count": len(param_group['params'])
            }
            state_info["param_groups"].append(group_info)
        
        return state_info
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generate training debugging report."""
        return {
            "training_debug_info": self.training_debug_info,
            "config": {
                "enable_training_debugging": self.config.enable_training_debugging,
                "debug_loss_computation": self.config.debug_loss_computation,
                "debug_optimizer_steps": self.config.debug_optimizer_steps
            }
        }

# =============================================================================
# CUDA DEBUGGING
# =============================================================================

class CUDADebugger:
    """CUDA-specific debugging tools."""
    
    def __init__(self, config: PyTorchDebugConfig):
        self.config = config
        self.cuda_debug_info = []
        
    def debug_cuda_operations(self, operation_name: str = None) -> Dict[str, Any]:
        """Debug CUDA operations."""
        if not self.config.enable_cuda_debugging or not torch.cuda.is_available():
            return {}
            
        cuda_info = {
            "timestamp": time.time(),
            "operation": operation_name,
            "device_info": self._get_device_info(),
            "memory_info": self._get_cuda_memory_info(),
            "synchronization": self._check_synchronization()
        }
        
        self.cuda_debug_info.append(cuda_info)
        return cuda_info
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        device = torch.cuda.current_device()
        return {
            "current_device": device,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": torch.cuda.get_device_capability(device),
            "total_memory": torch.cuda.get_device_properties(device).total_memory / 1024**3,  # GB
            "device_count": torch.cuda.device_count()
        }
    
    def _get_cuda_memory_info(self) -> Dict[str, Any]:
        """Get detailed CUDA memory information."""
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
            "max_cached": torch.cuda.max_memory_reserved() / 1024**3,  # GB
            "memory_fraction": torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
        }
    
    def _check_synchronization(self) -> Dict[str, Any]:
        """Check CUDA synchronization status."""
        if self.config.cuda_synchronize:
            torch.cuda.synchronize()
            return {"synchronized": True, "synchronization_time": time.time()}
        else:
            return {"synchronized": False}
    
    def get_cuda_report(self) -> Dict[str, Any]:
        """Generate CUDA debugging report."""
        return {
            "cuda_debug_info": self.cuda_debug_info,
            "config": {
                "enable_cuda_debugging": self.config.enable_cuda_debugging,
                "cuda_memory_fraction": self.config.cuda_memory_fraction,
                "cuda_synchronize": self.config.cuda_synchronize
            }
        }

# =============================================================================
# PROFILING
# =============================================================================

class PyTorchProfiler:
    """PyTorch profiling with detailed analysis."""
    
    def __init__(self, config: PyTorchDebugConfig):
        self.config = config
        self.profiler_results = []
        
    @contextmanager
    def profile_operation(self, operation_name: str = "operation"):
        """Context manager for profiling operations."""
        if not self.config.enable_profiling:
            yield
            return
            
        try:
            with profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU if self.config.profile_cpu else None,
                    profiler.ProfilerActivity.CUDA if self.config.profile_cuda else None
                ],
                record_shapes=self.config.profile_memory,
                with_stack=True
            ) as prof:
                yield prof
                
                # Save profiling results
                result = {
                    "operation_name": operation_name,
                    "timestamp": time.time(),
                    "key_averages": prof.key_averages().table(sort_by="cuda_time_total", row_limit=10),
                    "events": prof.function_events
                }
                
                self.profiler_results.append(result)
                
        except Exception as e:
            logger.error(f"Profiling error: {e}")
            yield None
    
    def get_profiler_report(self) -> Dict[str, Any]:
        """Generate profiling report."""
        return {
            "profiler_results": self.profiler_results,
            "config": {
                "enable_profiling": self.config.enable_profiling,
                "profile_memory": self.config.profile_memory,
                "profile_cpu": self.config.profile_cpu,
                "profile_cuda": self.config.profile_cuda
            }
        }

# =============================================================================
# MAIN DEBUG MANAGER
# =============================================================================

class PyTorchDebugManager:
    """Main PyTorch debugging manager that coordinates all debugging tools."""
    
    def __init__(self, config: Optional[PyTorchDebugConfig] = None):
        self.config = config or PyTorchDebugConfig()
        
        # Initialize all debugging components
        self.anomaly_detector = AutogradAnomalyDetector(self.config)
        self.gradient_debugger = GradientDebugger(self.config)
        self.memory_debugger = PyTorchMemoryDebugger(self.config)
        self.model_debugger = ModelDebugger(self.config)
        self.training_debugger = TrainingDebugger(self.config)
        self.cuda_debugger = CUDADebugger(self.config)
        self.profiler = PyTorchProfiler(self.config)
        
        # Setup output directory
        if self.config.save_debug_reports:
            self.output_dir = Path(self.config.debug_output_dir)
            self.output_dir.mkdir(exist_ok=True)
        
        logger.info("PyTorch Debug Manager initialized")
    
    def enable_all_debugging(self):
        """Enable all debugging features."""
        self.config.enable_autograd_anomaly = True
        self.config.enable_gradient_debugging = True
        self.config.enable_memory_debugging = True
        self.config.enable_model_debugging = True
        self.config.enable_training_debugging = True
        self.config.enable_cuda_debugging = True
        self.config.enable_profiling = True
        
        logger.info("All PyTorch debugging features enabled")
    
    def disable_all_debugging(self):
        """Disable all debugging features."""
        self.config.enable_autograd_anomaly = False
        self.config.enable_gradient_debugging = False
        self.config.enable_memory_debugging = False
        self.config.enable_model_debugging = False
        self.config.enable_training_debugging = False
        self.config.enable_cuda_debugging = False
        self.config.enable_profiling = False
        
        logger.info("All PyTorch debugging features disabled")
    
    def debug_training_loop(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        num_epochs: int = 1
    ):
        """Debug a complete training loop."""
        logger.info("Starting training loop debugging")
        
        # Initial model inspection
        model_inspection = self.model_debugger.inspect_model(model)
        
        # Memory snapshot before training
        memory_snapshot = self.memory_debugger.take_memory_snapshot("before_training")
        
        try:
            for epoch in range(num_epochs):
                logger.info(f"Debugging epoch {epoch + 1}/{num_epochs}")
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # CUDA debugging
                    cuda_info = self.cuda_debugger.debug_cuda_operations(f"batch_{batch_idx}")
                    
                    # Training step debugging
                    with self.anomaly_detector.detect_anomaly():
                        training_debug = self.training_debugger.debug_training_step(
                            model, optimizer, loss_fn, inputs, targets, batch_idx
                        )
                    
                    # Gradient debugging
                    optimizer.zero_grad()
                    loss = loss_fn(model(inputs), targets)
                    loss.backward()
                    
                    gradient_info = self.gradient_debugger.check_gradients(model, batch_idx)
                    
                    # Memory snapshot periodically
                    if batch_idx % self.config.memory_snapshot_frequency == 0:
                        self.memory_debugger.take_memory_snapshot(f"epoch_{epoch}_batch_{batch_idx}")
                    
                    optimizer.step()
                    
                    # Break after a few batches for debugging
                    if batch_idx >= 5:  # Debug first 5 batches
                        break
                
                # Memory cleanup
                self.memory_debugger.clear_memory()
        
        except Exception as e:
            logger.error(f"Training debugging error: {e}")
            raise
        
        # Final memory snapshot
        final_memory_snapshot = self.memory_debugger.take_memory_snapshot("after_training")
        
        logger.info("Training loop debugging completed")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive debugging report."""
        report = {
            "timestamp": time.time(),
            "config": self.config.__dict__,
            "anomaly_detection": self.anomaly_detector.get_anomaly_report(),
            "gradient_debugging": self.gradient_debugger.get_gradient_report(),
            "memory_debugging": self.memory_debugger.get_memory_report(),
            "model_debugging": self.model_debugger.get_model_report(),
            "training_debugging": self.training_debugger.get_training_report(),
            "cuda_debugging": self.cuda_debugger.get_cuda_report(),
            "profiling": self.profiler.get_profiler_report()
        }
        
        # Save report if enabled
        if self.config.save_debug_reports:
            report_path = self.output_dir / f"debug_report_{int(time.time())}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Debug report saved to {report_path}")
        
        return report
    
    def get_debug_status(self) -> Dict[str, bool]:
        """Get current debugging status."""
        return {
            "autograd_anomaly": self.config.enable_autograd_anomaly,
            "gradient_debugging": self.config.enable_gradient_debugging,
            "memory_debugging": self.config.enable_memory_debugging,
            "model_debugging": self.config.enable_model_debugging,
            "training_debugging": self.config.enable_training_debugging,
            "cuda_debugging": self.config.enable_cuda_debugging,
            "profiling": self.config.enable_profiling
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def enable_pytorch_debugging(config: Optional[PyTorchDebugConfig] = None) -> PyTorchDebugManager:
    """Convenience function to enable PyTorch debugging."""
    debug_manager = PyTorchDebugManager(config)
    debug_manager.enable_all_debugging()
    return debug_manager

def debug_tensor_operations(tensor: torch.Tensor, operation_name: str = "tensor_operation"):
    """Debug a single tensor operation."""
    print(f"🔍 Debugging {operation_name}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires grad: {tensor.requires_grad}")
    print(f"  Norm: {tensor.norm().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf: {torch.isinf(tensor).any().item()}")

def debug_model_forward(model: nn.Module, inputs: torch.Tensor, layer_name: str = None):
    """Debug model forward pass."""
    print(f"🔍 Debugging model forward pass")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Input device: {inputs.device}")
    
    try:
        with torch.no_grad():
            outputs = model(inputs)
        print(f"  Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
        print(f"  Forward pass successful: ✅")
    except Exception as e:
        print(f"  Forward pass failed: ❌")
        print(f"  Error: {e}")

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_pytorch_debugging():
    """Example usage of PyTorch debugging tools."""
    
    # Initialize debug manager
    config = PyTorchDebugConfig(
        enable_autograd_anomaly=True,
        enable_gradient_debugging=True,
        enable_memory_debugging=True,
        enable_model_debugging=True,
        enable_training_debugging=True,
        enable_cuda_debugging=True,
        enable_profiling=True
    )
    
    debug_manager = PyTorchDebugManager(config)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Debug model
    model_inspection = debug_manager.model_debugger.inspect_model(model)
    print("Model inspection completed")
    
    # Debug training loop
    train_loader = torch.utils.data.DataLoader(
        [(torch.randn(10), torch.randn(1)) for _ in range(100)],
        batch_size=32
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    debug_manager.debug_training_loop(model, train_loader, optimizer, loss_fn, num_epochs=1)
    
    # Generate report
    report = debug_manager.generate_comprehensive_report()
    print("Debug report generated")
    
    return debug_manager, report

if __name__ == "__main__":
    # Run example
    debug_manager, report = example_pytorch_debugging()
    print("PyTorch debugging example completed") 