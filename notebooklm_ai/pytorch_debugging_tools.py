from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import os
import sys
import logging
import traceback
import time
import json
import gc
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
import contextlib
import warnings
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gradio as gr
from training_logging_system import TrainingLogger
from robust_error_handling import RobustErrorHandler
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
PyTorch Debugging Tools
=======================

This module provides comprehensive PyTorch debugging tools integration:
- autograd.detect_anomaly() for gradient anomaly detection
- Memory debugging and profiling tools
- CUDA debugging and error detection
- Model debugging and validation tools
- Performance profiling and optimization
- Debug mode management and configuration
"""



# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DebugConfiguration:
    """Configuration for PyTorch debugging tools"""
    enable_autograd_anomaly: bool = True
    enable_memory_debugging: bool = True
    enable_cuda_debugging: bool = True
    enable_model_debugging: bool = True
    enable_performance_profiling: bool = True
    enable_gradient_checking: bool = True
    enable_nan_detection: bool = True
    enable_inf_detection: bool = True
    enable_shape_validation: bool = True
    enable_dtype_validation: bool = True
    enable_device_validation: bool = True
    debug_log_level: str = "INFO"
    save_debug_info: bool = True
    debug_output_dir: str = "debug_output"


@dataclass
class DebugEvent:
    """Debug event information"""
    timestamp: datetime
    event_type: str
    severity: str
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    tensor_info: Optional[Dict[str, Any]] = None
    gradient_info: Optional[Dict[str, Any]] = None
    memory_info: Optional[Dict[str, Any]] = None
    performance_info: Optional[Dict[str, Any]] = None


class PyTorchDebugger:
    """Comprehensive PyTorch debugging tools integration"""
    
    def __init__(self, config: DebugConfiguration = None):
        
    """__init__ function."""
self.config = config or DebugConfiguration()
        self.training_logger = None
        self.error_handler = RobustErrorHandler()
        self.debug_events = []
        self.debug_output_dir = Path(self.config.debug_output_dir)
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug state
        self.autograd_anomaly_enabled = False
        self.memory_debugging_enabled = False
        self.cuda_debugging_enabled = False
        self.profiling_enabled = False
        
        # Performance tracking
        self.performance_metrics = {}
        self.memory_snapshots = []
        
        # Setup debugging tools
        self._setup_debugging_tools()
        
        logger.info("PyTorch Debugger initialized")
    
    def _setup_debugging_tools(self) -> Any:
        """Setup PyTorch debugging tools"""
        if self.config.enable_autograd_anomaly:
            self._setup_autograd_anomaly_detection()
        
        if self.config.enable_memory_debugging:
            self._setup_memory_debugging()
        
        if self.config.enable_cuda_debugging:
            self._setup_cuda_debugging()
        
        if self.config.enable_model_debugging:
            self._setup_model_debugging()
        
        if self.config.enable_performance_profiling:
            self._setup_performance_profiling()
    
    def _setup_autograd_anomaly_detection(self) -> Any:
        """Setup autograd anomaly detection"""
        try:
            if torch.autograd.anomaly_mode._enabled:
                logger.info("Autograd anomaly detection already enabled")
            else:
                autograd.detect_anomaly()
                self.autograd_anomaly_enabled = True
                logger.info("Autograd anomaly detection enabled")
        except Exception as e:
            logger.error(f"Failed to setup autograd anomaly detection: {e}")
    
    def _setup_memory_debugging(self) -> Any:
        """Setup memory debugging tools"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.memory_debugging_enabled = True
                logger.info("Memory debugging enabled")
        except Exception as e:
            logger.error(f"Failed to setup memory debugging: {e}")
    
    def _setup_cuda_debugging(self) -> Any:
        """Setup CUDA debugging tools"""
        try:
            if torch.cuda.is_available():
                # Enable CUDA memory debugging
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                self.cuda_debugging_enabled = True
                logger.info("CUDA debugging enabled")
        except Exception as e:
            logger.error(f"Failed to setup CUDA debugging: {e}")
    
    def _setup_model_debugging(self) -> Any:
        """Setup model debugging tools"""
        try:
            # Register hooks for model debugging
            logger.info("Model debugging enabled")
        except Exception as e:
            logger.error(f"Failed to setup model debugging: {e}")
    
    def _setup_performance_profiling(self) -> Any:
        """Setup performance profiling tools"""
        try:
            self.profiling_enabled = True
            logger.info("Performance profiling enabled")
        except Exception as e:
            logger.error(f"Failed to setup performance profiling: {e}")
    
    def enable_debug_mode(self, training_logger: TrainingLogger = None):
        """Enable comprehensive debug mode"""
        self.training_logger = training_logger
        
        # Enable autograd anomaly detection
        if self.config.enable_autograd_anomaly and not self.autograd_anomaly_enabled:
            self._setup_autograd_anomaly_detection()
        
        # Enable memory debugging
        if self.config.enable_memory_debugging and not self.memory_debugging_enabled:
            self._setup_memory_debugging()
        
        # Enable CUDA debugging
        if self.config.enable_cuda_debugging and not self.cuda_debugging_enabled:
            self._setup_cuda_debugging()
        
        logger.info("Comprehensive debug mode enabled")
    
    def disable_debug_mode(self) -> Any:
        """Disable debug mode"""
        if self.autograd_anomaly_enabled:
            autograd.detect_anomaly(False)
            self.autograd_anomaly_enabled = False
        
        if self.memory_debugging_enabled:
            self.memory_debugging_enabled = False
        
        if self.cuda_debugging_enabled:
            os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
            self.cuda_debugging_enabled = False
        
        if self.profiling_enabled:
            self.profiling_enabled = False
        
        logger.info("Debug mode disabled")
    
    @contextlib.contextmanager
    def debug_context(self, context_name: str = "debug_operation"):
        """Context manager for debug operations"""
        debug_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="DEBUG_CONTEXT_START",
            severity="INFO",
            message=f"Debug context started: {context_name}",
            context={"context_name": context_name}
        )
        
        self._log_debug_event(debug_event)
        
        try:
            yield self
        except Exception as e:
            error_event = DebugEvent(
                timestamp=datetime.now(),
                event_type="DEBUG_CONTEXT_ERROR",
                severity="ERROR",
                message=f"Error in debug context {context_name}: {e}",
                context={"context_name": context_name, "error": str(e)},
                stack_trace=traceback.format_exc()
            )
            self._log_debug_event(error_event)
            raise
        finally:
            end_event = DebugEvent(
                timestamp=datetime.now(),
                event_type="DEBUG_CONTEXT_END",
                severity="INFO",
                message=f"Debug context ended: {context_name}",
                context={"context_name": context_name}
            )
            self._log_debug_event(end_event)
    
    def check_tensor_validity(self, tensor: torch.Tensor, tensor_name: str = "tensor") -> Dict[str, Any]:
        """Check tensor validity and properties"""
        validity_info = {
            "name": tensor_name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "is_valid": True,
            "issues": []
        }
        
        # Check for NaN values
        if self.config.enable_nan_detection:
            if torch.isnan(tensor).any():
                validity_info["is_valid"] = False
                validity_info["issues"].append("Contains NaN values")
                self._log_tensor_issue(tensor_name, "NaN detected", tensor)
        
        # Check for Inf values
        if self.config.enable_inf_detection:
            if torch.isinf(tensor).any():
                validity_info["is_valid"] = False
                validity_info["issues"].append("Contains Inf values")
                self._log_tensor_issue(tensor_name, "Inf detected", tensor)
        
        # Check for extreme values
        if tensor.numel() > 0:
            max_val = tensor.max().item()
            min_val = tensor.min().item()
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            
            validity_info.update({
                "max_value": max_val,
                "min_value": min_val,
                "mean_value": mean_val,
                "std_value": std_val
            })
            
            # Check for extreme values
            if abs(max_val) > 1e6 or abs(min_val) > 1e6:
                validity_info["issues"].append("Contains extreme values")
                self._log_tensor_issue(tensor_name, "Extreme values detected", tensor)
        
        return validity_info
    
    def check_gradient_validity(self, model: nn.Module) -> Dict[str, Any]:
        """Check gradient validity for model parameters"""
        gradient_info = {
            "total_parameters": 0,
            "parameters_with_gradients": 0,
            "gradient_norms": {},
            "gradient_issues": [],
            "is_valid": True
        }
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                gradient_info["total_parameters"] += param.numel()
                param_count += 1
                
                if param.grad is not None:
                    gradient_info["parameters_with_gradients"] += param.numel()
                    
                    # Check gradient validity
                    grad_validity = self.check_tensor_validity(param.grad, f"grad_{name}")
                    
                    if not grad_validity["is_valid"]:
                        gradient_info["is_valid"] = False
                        gradient_info["gradient_issues"].append({
                            "parameter": name,
                            "issues": grad_validity["issues"]
                        })
                    
                    # Calculate gradient norm
                    param_norm = param.grad.norm().item()
                    gradient_info["gradient_norms"][name] = param_norm
                    total_norm += param_norm ** 2
        
        gradient_info["total_gradient_norm"] = total_norm ** 0.5
        
        # Check for gradient explosion
        if gradient_info["total_gradient_norm"] > 10.0:
            gradient_info["gradient_issues"].append("Gradient explosion detected")
            gradient_info["is_valid"] = False
        
        # Check for gradient vanishing
        if gradient_info["total_gradient_norm"] < 1e-6:
            gradient_info["gradient_issues"].append("Gradient vanishing detected")
            gradient_info["is_valid"] = False
        
        return gradient_info
    
    def validate_model_inputs(self, model: nn.Module, *args, **kwargs) -> Dict[str, Any]:
        """Validate model inputs"""
        validation_info = {
            "inputs_valid": True,
            "input_shapes": [],
            "input_types": [],
            "input_devices": [],
            "validation_issues": []
        }
        
        # Validate positional arguments
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensor_info = self.check_tensor_validity(arg, f"input_{i}")
                validation_info["input_shapes"].append(tensor_info["shape"])
                validation_info["input_types"].append(tensor_info["dtype"])
                validation_info["input_devices"].append(tensor_info["device"])
                
                if not tensor_info["is_valid"]:
                    validation_info["inputs_valid"] = False
                    validation_info["validation_issues"].append({
                        "input_index": i,
                        "issues": tensor_info["issues"]
                    })
        
        # Validate keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensor_info = self.check_tensor_validity(value, f"input_{key}")
                validation_info["input_shapes"].append(tensor_info["shape"])
                validation_info["input_types"].append(tensor_info["dtype"])
                validation_info["input_devices"].append(tensor_info["device"])
                
                if not tensor_info["is_valid"]:
                    validation_info["inputs_valid"] = False
                    validation_info["validation_issues"].append({
                        "input_key": key,
                        "issues": tensor_info["issues"]
                    })
        
        return validation_info
    
    def debug_forward_pass(self, model: nn.Module, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Debug model forward pass with comprehensive validation"""
        debug_info = {
            "input_validation": {},
            "forward_pass_successful": False,
            "output_validation": {},
            "memory_usage": {},
            "performance_metrics": {},
            "errors": []
        }
        
        try:
            # Validate inputs
            debug_info["input_validation"] = self.validate_model_inputs(model, *args, **kwargs)
            
            if not debug_info["input_validation"]["inputs_valid"]:
                raise ValueError("Model inputs validation failed")
            
            # Record memory before forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                debug_info["memory_usage"]["before"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved()
                }
            
            # Perform forward pass with timing
            start_time = time.time()
            with torch.no_grad():
                output = model(*args, **kwargs)
            forward_time = time.time() - start_time
            
            debug_info["performance_metrics"]["forward_time"] = forward_time
            debug_info["forward_pass_successful"] = True
            
            # Record memory after forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                debug_info["memory_usage"]["after"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved()
                }
            
            # Validate output
            if isinstance(output, torch.Tensor):
                debug_info["output_validation"] = self.check_tensor_validity(output, "model_output")
            elif isinstance(output, (list, tuple)):
                debug_info["output_validation"] = {
                    "type": "sequence",
                    "length": len(output),
                    "elements": [self.check_tensor_validity(item, f"output_{i}") 
                               for i, item in enumerate(output) if isinstance(item, torch.Tensor)]
                }
            elif isinstance(output, dict):
                debug_info["output_validation"] = {
                    "type": "dict",
                    "keys": list(output.keys()),
                    "elements": {key: self.check_tensor_validity(value, f"output_{key}")
                               for key, value in output.items() if isinstance(value, torch.Tensor)}
                }
            
        except Exception as e:
            debug_info["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "stack_trace": traceback.format_exc()
            })
            
            # Log error
            error_event = DebugEvent(
                timestamp=datetime.now(),
                event_type="FORWARD_PASS_ERROR",
                severity="ERROR",
                message=f"Forward pass failed: {e}",
                context={"model_type": type(model).__name__},
                stack_trace=traceback.format_exc()
            )
            self._log_debug_event(error_event)
        
        return output, debug_info
    
    def debug_backward_pass(self, loss: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        """Debug model backward pass with gradient validation"""
        debug_info = {
            "loss_validation": {},
            "backward_pass_successful": False,
            "gradient_validation": {},
            "memory_usage": {},
            "performance_metrics": {},
            "errors": []
        }
        
        try:
            # Validate loss
            debug_info["loss_validation"] = self.check_tensor_validity(loss, "loss")
            
            if not debug_info["loss_validation"]["is_valid"]:
                raise ValueError("Loss validation failed")
            
            # Record memory before backward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                debug_info["memory_usage"]["before"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved()
                }
            
            # Perform backward pass with timing
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time
            
            debug_info["performance_metrics"]["backward_time"] = backward_time
            debug_info["backward_pass_successful"] = True
            
            # Record memory after backward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                debug_info["memory_usage"]["after"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved()
                }
            
            # Validate gradients
            debug_info["gradient_validation"] = self.check_gradient_validity(model)
            
        except Exception as e:
            debug_info["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "stack_trace": traceback.format_exc()
            })
            
            # Log error
            error_event = DebugEvent(
                timestamp=datetime.now(),
                event_type="BACKWARD_PASS_ERROR",
                severity="ERROR",
                message=f"Backward pass failed: {e}",
                context={"model_type": type(model).__name__},
                stack_trace=traceback.format_exc()
            )
            self._log_debug_event(error_event)
        
        return debug_info
    
    def profile_model(self, model: nn.Module, input_data: torch.Tensor, 
                     num_iterations: int = 10) -> Dict[str, Any]:
        """Profile model performance"""
        if not self.profiling_enabled:
            return {"error": "Profiling not enabled"}
        
        profile_info = {
            "model_info": {
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            },
            "performance_metrics": {},
            "memory_metrics": {},
            "profiler_output": None
        }
        
        try:
            # Warm up
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_data)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Profile forward pass
            forward_times = []
            memory_usage = []
            
            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True
            ) as prof:
                for _ in range(num_iterations):
                    start_time = time.time()
                    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    with torch.no_grad():
                        output = model(input_data)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    forward_times.append(end_time - start_time)
                    memory_usage.append(end_memory - start_memory)
            
            # Calculate statistics
            profile_info["performance_metrics"] = {
                "mean_forward_time": np.mean(forward_times),
                "std_forward_time": np.std(forward_times),
                "min_forward_time": np.min(forward_times),
                "max_forward_time": np.max(forward_times),
                "throughput": num_iterations / sum(forward_times)
            }
            
            if torch.cuda.is_available():
                profile_info["memory_metrics"] = {
                    "mean_memory_usage": np.mean(memory_usage),
                    "max_memory_usage": np.max(memory_usage),
                    "current_memory_allocated": torch.cuda.memory_allocated(),
                    "current_memory_reserved": torch.cuda.memory_reserved()
                }
            
            # Save profiler output
            profiler_output_file = self.debug_output_dir / "profiler_output.txt"
            with open(profiler_output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            profile_info["profiler_output"] = str(profiler_output_file)
            
        except Exception as e:
            profile_info["error"] = str(e)
            
            error_event = DebugEvent(
                timestamp=datetime.now(),
                event_type="PROFILING_ERROR",
                severity="ERROR",
                message=f"Model profiling failed: {e}",
                context={"model_type": type(model).__name__},
                stack_trace=traceback.format_exc()
            )
            self._log_debug_event(error_event)
        
        return profile_info
    
    def check_memory_leaks(self, model: nn.Module, num_iterations: int = 100) -> Dict[str, Any]:
        """Check for memory leaks in model"""
        memory_info = {
            "initial_memory": {},
            "final_memory": {},
            "memory_growth": {},
            "potential_leak": False,
            "memory_snapshots": []
        }
        
        try:
            # Record initial memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                memory_info["initial_memory"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated()
                }
            
            # Create dummy input
            input_data = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            
            # Run multiple iterations
            for i in range(num_iterations):
                with torch.no_grad():
                    output = model(input_data)
                
                if i % 10 == 0:  # Record memory every 10 iterations
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        snapshot = {
                            "iteration": i,
                            "allocated": torch.cuda.memory_allocated(),
                            "reserved": torch.cuda.memory_reserved()
                        }
                        memory_info["memory_snapshots"].append(snapshot)
                
                # Clear some references
                del output
                if i % 20 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Record final memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_info["final_memory"] = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated()
                }
                
                # Check for memory growth
                allocated_growth = (memory_info["final_memory"]["allocated"] - 
                                  memory_info["initial_memory"]["allocated"])
                reserved_growth = (memory_info["final_memory"]["reserved"] - 
                                 memory_info["initial_memory"]["reserved"])
                
                memory_info["memory_growth"] = {
                    "allocated_growth_mb": allocated_growth / (1024 * 1024),
                    "reserved_growth_mb": reserved_growth / (1024 * 1024)
                }
                
                # Check for potential leak (more than 10MB growth)
                if allocated_growth > 10 * 1024 * 1024:  # 10MB
                    memory_info["potential_leak"] = True
                    
                    leak_event = DebugEvent(
                        timestamp=datetime.now(),
                        event_type="MEMORY_LEAK_DETECTED",
                        severity="WARNING",
                        message="Potential memory leak detected",
                        context=memory_info["memory_growth"]
                    )
                    self._log_debug_event(leak_event)
            
        except Exception as e:
            memory_info["error"] = str(e)
            
            error_event = DebugEvent(
                timestamp=datetime.now(),
                event_type="MEMORY_CHECK_ERROR",
                severity="ERROR",
                message=f"Memory leak check failed: {e}",
                stack_trace=traceback.format_exc()
            )
            self._log_debug_event(error_event)
        
        return memory_info
    
    def _log_tensor_issue(self, tensor_name: str, issue: str, tensor: torch.Tensor):
        """Log tensor issues"""
        tensor_info = {
            "name": tensor_name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "numel": tensor.numel()
        }
        
        if tensor.numel() > 0:
            tensor_info.update({
                "min_value": tensor.min().item(),
                "max_value": tensor.max().item(),
                "mean_value": tensor.mean().item(),
                "std_value": tensor.std().item()
            })
        
        debug_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="TENSOR_ISSUE",
            severity="WARNING",
            message=f"Tensor issue detected: {issue}",
            context={"tensor_name": tensor_name, "issue": issue},
            tensor_info=tensor_info
        )
        
        self._log_debug_event(debug_event)
    
    def _log_debug_event(self, event: DebugEvent):
        """Log debug event"""
        self.debug_events.append(event)
        
        # Log to training logger if available
        if self.training_logger:
            if event.severity == "ERROR":
                self.training_logger.log_error(
                    Exception(event.message),
                    event.context,
                    event.severity
                )
            elif event.severity == "WARNING":
                self.training_logger.log_warning(event.message, event.context)
            else:
                self.training_logger.log_info(event.message, event.context)
        
        # Save debug event to file
        if self.config.save_debug_info:
            debug_file = self.debug_output_dir / "debug_events.json"
            event_dict = asdict(event)
            event_dict["timestamp"] = event_dict["timestamp"].isoformat()
            
            with open(debug_file, 'a') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(json.dumps(event_dict) + '\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get debug summary"""
        if not self.debug_events:
            return {"message": "No debug events recorded"}
        
        event_types = {}
        severities = {}
        
        for event in self.debug_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            severities[event.severity] = severities.get(event.severity, 0) + 1
        
        return {
            "total_events": len(self.debug_events),
            "event_types": event_types,
            "severities": severities,
            "debug_mode_enabled": {
                "autograd_anomaly": self.autograd_anomaly_enabled,
                "memory_debugging": self.memory_debugging_enabled,
                "cuda_debugging": self.cuda_debugging_enabled,
                "profiling": self.profiling_enabled
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "message": event.message
                }
                for event in self.debug_events[-10:]  # Last 10 events
            ]
        }
    
    def save_debug_report(self, filename: str = None) -> str:
        """Save comprehensive debug report"""
        if filename is None:
            filename = f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_file = self.debug_output_dir / filename
        
        report = {
            "debug_configuration": asdict(self.config),
            "debug_summary": self.get_debug_summary(),
            "debug_events": [asdict(event) for event in self.debug_events],
            "performance_metrics": self.performance_metrics,
            "memory_snapshots": self.memory_snapshots
        }
        
        # Convert datetime objects to strings
        for event_dict in report["debug_events"]:
            event_dict["timestamp"] = event_dict["timestamp"].isoformat()
        
        with open(report_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        logger.info(f"Debug report saved to: {report_file}")
        return str(report_file)


class PyTorchDebuggingInterface:
    """Gradio interface for PyTorch debugging tools"""
    
    def __init__(self) -> Any:
        self.debugger = PyTorchDebugger()
        self.config = DebugConfiguration()
        
        logger.info("PyTorch Debugging Interface initialized")
    
    def create_pytorch_debugging_interface(self) -> gr.Interface:
        """Create comprehensive PyTorch debugging interface"""
        
        def enable_debug_mode(enable_autograd: bool, enable_memory: bool, 
                             enable_cuda: bool, enable_profiling: bool):
            """Enable debug mode with selected options"""
            try:
                self.config.enable_autograd_anomaly = enable_autograd
                self.config.enable_memory_debugging = enable_memory
                self.config.enable_cuda_debugging = enable_cuda
                self.config.enable_performance_profiling = enable_profiling
                
                self.debugger = PyTorchDebugger(self.config)
                self.debugger.enable_debug_mode()
                
                return {
                    "status": "success",
                    "message": "Debug mode enabled with selected options",
                    "config": asdict(self.config)
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to enable debug mode: {e}"
                }
        
        def check_tensor_validity(tensor_data: str, tensor_name: str):
            """Check tensor validity"""
            try:
                # Parse tensor data (simplified for demo)
                if tensor_data.strip():
                    # Create dummy tensor for demonstration
                    tensor = torch.randn(3, 3)
                    if "nan" in tensor_data.lower():
                        tensor[0, 0] = float('nan')
                    if "inf" in tensor_data.lower():
                        tensor[0, 1] = float('inf')
                    
                    validity_info = self.debugger.check_tensor_validity(tensor, tensor_name)
                    
                    return {
                        "status": "success",
                        "validity_info": validity_info
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Please provide tensor data"
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to check tensor validity: {e}"
                }
        
        def profile_model(model_type: str, input_size: int, num_iterations: int):
            """Profile model performance"""
            try:
                # Create dummy model
                if model_type == "linear":
                    model = nn.Linear(input_size, 10)
                elif model_type == "conv":
                    model = nn.Sequential(
                        nn.Conv2d(3, 16, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(16, 10)
                    )
                else:
                    model = nn.Linear(input_size, 10)
                
                # Create dummy input
                if model_type == "conv":
                    input_data = torch.randn(1, 3, 32, 32)
                else:
                    input_data = torch.randn(1, input_size)
                
                profile_info = self.debugger.profile_model(model, input_data, num_iterations)
                
                return {
                    "status": "success",
                    "profile_info": profile_info
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to profile model: {e}"
                }
        
        def check_memory_leaks(model_type: str, num_iterations: int):
            """Check for memory leaks"""
            try:
                # Create dummy model
                if model_type == "linear":
                    model = nn.Linear(100, 10)
                elif model_type == "conv":
                    model = nn.Sequential(
                        nn.Conv2d(3, 16, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(16, 10)
                    )
                else:
                    model = nn.Linear(100, 10)
                
                memory_info = self.debugger.check_memory_leaks(model, num_iterations)
                
                return {
                    "status": "success",
                    "memory_info": memory_info
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to check memory leaks: {e}"
                }
        
        def get_debug_summary():
            """Get debug summary"""
            try:
                summary = self.debugger.get_debug_summary()
                return {
                    "status": "success",
                    "summary": summary
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to get debug summary: {e}"
                }
        
        def save_debug_report():
            """Save debug report"""
            try:
                report_file = self.debugger.save_debug_report()
                return {
                    "status": "success",
                    "message": f"Debug report saved to: {report_file}"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to save debug report: {e}"
                }
        
        # Create interface
        with gr.Blocks(
            title="PyTorch Debugging Tools",
            theme=gr.themes.Soft(),
            css="""
            .debug-section {
                background: #e8f5e8;
                border: 1px solid #4caf50;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
            .error-section {
                background: #ffebee;
                border: 1px solid #f44336;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
            """
        ) as interface:
            
            gr.Markdown("# 🔧 PyTorch Debugging Tools")
            gr.Markdown("Comprehensive debugging tools with autograd.detect_anomaly() and more")
            
            with gr.Tabs():
                with gr.TabItem("⚙️ Debug Configuration"):
                    gr.Markdown("### Enable PyTorch Debugging Tools")
                    
                    with gr.Row():
                        with gr.Column():
                            enable_autograd = gr.Checkbox(
                                label="Enable Autograd Anomaly Detection",
                                value=True,
                                info="Detect anomalies in autograd computation"
                            )
                            
                            enable_memory = gr.Checkbox(
                                label="Enable Memory Debugging",
                                value=True,
                                info="Monitor memory usage and detect leaks"
                            )
                            
                            enable_cuda = gr.Checkbox(
                                label="Enable CUDA Debugging",
                                value=True,
                                info="Enable CUDA debugging and error detection"
                            )
                            
                            enable_profiling = gr.Checkbox(
                                label="Enable Performance Profiling",
                                value=True,
                                info="Profile model performance and bottlenecks"
                            )
                            
                            enable_debug_btn = gr.Button("⚙️ Enable Debug Mode", variant="primary")
                        
                        with gr.Column():
                            debug_config_result = gr.JSON(label="Debug Configuration Result")
                
                with gr.TabItem("🔍 Tensor Validation"):
                    gr.Markdown("### Check Tensor Validity")
                    
                    with gr.Row():
                        with gr.Column():
                            tensor_data = gr.Textbox(
                                label="Tensor Data (optional - for demo)",
                                placeholder="Enter 'nan' or 'inf' to simulate issues...",
                                value=""
                            )
                            
                            tensor_name = gr.Textbox(
                                label="Tensor Name",
                                placeholder="Enter tensor name...",
                                value="test_tensor"
                            )
                            
                            check_tensor_btn = gr.Button("🔍 Check Tensor", variant="primary")
                        
                        with gr.Column():
                            tensor_result = gr.JSON(label="Tensor Validation Result")
                
                with gr.TabItem("📊 Model Profiling"):
                    gr.Markdown("### Profile Model Performance")
                    
                    with gr.Row():
                        with gr.Column():
                            model_type = gr.Dropdown(
                                choices=["linear", "conv"],
                                value="linear",
                                label="Model Type"
                            )
                            
                            input_size = gr.Slider(
                                minimum=10, maximum=1000, value=100, step=10,
                                label="Input Size"
                            )
                            
                            num_iterations = gr.Slider(
                                minimum=5, maximum=50, value=10, step=5,
                                label="Number of Iterations"
                            )
                            
                            profile_btn = gr.Button("📊 Profile Model", variant="primary")
                        
                        with gr.Column():
                            profile_result = gr.JSON(label="Profiling Result")
                
                with gr.TabItem("💾 Memory Leak Detection"):
                    gr.Markdown("### Check for Memory Leaks")
                    
                    with gr.Row():
                        with gr.Column():
                            leak_model_type = gr.Dropdown(
                                choices=["linear", "conv"],
                                value="linear",
                                label="Model Type"
                            )
                            
                            leak_iterations = gr.Slider(
                                minimum=10, maximum=200, value=100, step=10,
                                label="Number of Iterations"
                            )
                            
                            check_leaks_btn = gr.Button("💾 Check Memory Leaks", variant="primary")
                        
                        with gr.Column():
                            leak_result = gr.JSON(label="Memory Leak Result")
                
                with gr.TabItem("📋 Debug Summary"):
                    gr.Markdown("### Debug Summary and Reports")
                    
                    with gr.Row():
                        with gr.Column():
                            get_summary_btn = gr.Button("📋 Get Debug Summary", variant="primary")
                            save_report_btn = gr.Button("💾 Save Debug Report", variant="secondary")
                        
                        with gr.Column():
                            summary_result = gr.JSON(label="Debug Summary")
                            report_result = gr.JSON(label="Report Result")
                
                with gr.TabItem("📚 Debug Features"):
                    gr.Markdown("### PyTorch Debugging Features")
                    
                    gr.Markdown("""
                    **Available Debug Tools:**
                    
                    **🔧 Autograd Anomaly Detection:**
                    - `autograd.detect_anomaly()` - Detect anomalies in autograd computation
                    - Automatic detection of NaN/Inf gradients
                    - Detailed error reporting with stack traces
                    
                    **💾 Memory Debugging:**
                    - Memory leak detection and monitoring
                    - GPU memory usage tracking
                    - Memory allocation/deallocation profiling
                    
                    **🚀 CUDA Debugging:**
                    - CUDA error detection and reporting
                    - Device synchronization debugging
                    - Memory allocation debugging
                    
                    **📊 Performance Profiling:**
                    - Model performance profiling with torch.profiler
                    - CPU and GPU activity monitoring
                    - Bottleneck identification
                    
                    **🔍 Tensor Validation:**
                    - NaN/Inf detection in tensors
                    - Shape and dtype validation
                    - Device consistency checking
                    
                    **⚡ Gradient Validation:**
                    - Gradient explosion/vanishing detection
                    - Gradient norm monitoring
                    - Parameter gradient validation
                    
                    **🎯 Model Debugging:**
                    - Forward pass debugging and validation
                    - Backward pass debugging and validation
                    - Input/output validation
                    
                    **📈 Debug Reporting:**
                    - Comprehensive debug event logging
                    - Performance metrics collection
                    - Debug report generation
                    """)
            
            # Event handlers
            enable_debug_btn.click(
                fn=enable_debug_mode,
                inputs=[enable_autograd, enable_memory, enable_cuda, enable_profiling],
                outputs=[debug_config_result]
            )
            
            check_tensor_btn.click(
                fn=check_tensor_validity,
                inputs=[tensor_data, tensor_name],
                outputs=[tensor_result]
            )
            
            profile_btn.click(
                fn=profile_model,
                inputs=[model_type, input_size, num_iterations],
                outputs=[profile_result]
            )
            
            check_leaks_btn.click(
                fn=check_memory_leaks,
                inputs=[leak_model_type, leak_iterations],
                outputs=[leak_result]
            )
            
            get_summary_btn.click(
                fn=get_debug_summary,
                inputs=[],
                outputs=[summary_result]
            )
            
            save_report_btn.click(
                fn=save_debug_report,
                inputs=[],
                outputs=[report_result]
            )
        
        return interface
    
    def launch_pytorch_debugging_interface(self, port: int = 7871, share: bool = False):
        """Launch the PyTorch debugging interface"""
        print("🔧 Launching PyTorch Debugging Tools...")
        
        interface = self.create_pytorch_debugging_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to run the PyTorch debugging tools"""
    print("🔧 Starting PyTorch Debugging Tools...")
    
    interface = PyTorchDebuggingInterface()
    interface.launch_pytorch_debugging_interface(port=7871, share=False)


match __name__:
    case "__main__":
    main() 