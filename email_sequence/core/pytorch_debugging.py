from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.autograd as autograd
import torch.profiler as profiler
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, ContextManager
import time
import traceback
import warnings
import gc
import psutil
import os
from pathlib import Path
from contextlib import contextmanager
import json
from datetime import datetime
import numpy as np
from core.training_logger import TrainingLogger, TrainingEventType, LogLevel
from core.error_handling import ErrorHandler, ModelError
    import torch.nn as nn
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
PyTorch Debugging Tools Integration

Comprehensive integration of PyTorch's built-in debugging tools
including autograd.detect_anomaly(), profiler, and other debugging utilities
for enhanced error detection and performance analysis.
"""




class PyTorchDebugger:
    """Comprehensive PyTorch debugging system"""
    
    def __init__(
        self,
        logger: Optional[TrainingLogger] = None,
        debug_mode: bool = False,
        enable_anomaly_detection: bool = False,
        enable_profiling: bool = False,
        enable_memory_tracking: bool = False,
        enable_gradient_checking: bool = False,
        log_dir: str = "debug_logs"
    ):
        """Initialize the PyTorch debugger"""
        
        self.logger = logger
        self.debug_mode = debug_mode
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_profiling = enable_profiling
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_gradient_checking = enable_gradient_checking
        
        # Create debug log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug state
        self.anomaly_detected = False
        self.profiler_active = False
        self.memory_snapshots = []
        self.gradient_violations = []
        self.debug_events = []
        
        # Performance tracking
        self.forward_times = []
        self.backward_times = []
        self.memory_usage = []
        
        # Error tracking
        self.nan_detections = 0
        self.inf_detections = 0
        self.gradient_explosions = 0
        self.gradient_vanishing = 0
        
        if self.logger:
            self.logger.log_info("PyTorch debugger initialized")
    
    def _log_debug_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """Log debug event"""
        
        if self.logger:
            self.logger._log_event(
                TrainingEventType.ERROR if "error" in event_type.lower() else TrainingEventType.WARNING,
                LogLevel.DEBUG if self.debug_mode else LogLevel.INFO,
                f"[DEBUG] {message}",
                data or {}
            )
        
        # Store debug event
        debug_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        self.debug_events.append(debug_event)
    
    @contextmanager
    def anomaly_detection(self, enabled: bool = None):
        """Context manager for autograd anomaly detection"""
        
        if enabled is None:
            enabled = self.enable_anomaly_detection
        
        if not enabled:
            yield
            return
        
        try:
            # Enable anomaly detection
            autograd.set_detect_anomaly(True)
            self._log_debug_event("anomaly_detection_start", "Autograd anomaly detection enabled")
            
            yield
            
        except Exception as e:
            self.anomaly_detected = True
            self._log_debug_event(
                "anomaly_detected",
                f"Autograd anomaly detected: {str(e)}",
                {
                    "error": str(e),
                    "stack_trace": traceback.format_exc(),
                    "anomaly_type": type(e).__name__
                }
            )
            raise
            
        finally:
            # Disable anomaly detection
            autograd.set_detect_anomaly(False)
            self._log_debug_event("anomaly_detection_end", "Autograd anomaly detection disabled")
    
    @contextmanager
    def profiling(self, enabled: bool = None, output_file: str = None):
        """Context manager for PyTorch profiling"""
        
        if enabled is None:
            enabled = self.enable_profiling
        
        if not enabled:
            yield
            return
        
        if output_file is None:
            output_file = self.log_dir / f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Start profiling
            with profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                use_cuda=torch.cuda.is_available()
            ) as prof:
                self.profiler_active = True
                self._log_debug_event("profiling_start", "PyTorch profiling started")
                
                yield
                
                # Export profiling results
                prof.export_chrome_trace(str(output_file))
                self._log_debug_event(
                    "profiling_end",
                    f"PyTorch profiling completed - Results saved to {output_file}",
                    {"output_file": str(output_file)}
                )
                
        except Exception as e:
            self._log_debug_event(
                "profiling_error",
                f"Profiling error: {str(e)}",
                {"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
        finally:
            self.profiler_active = False
    
    @contextmanager
    def memory_tracking(self, enabled: bool = None):
        """Context manager for memory tracking"""
        
        if enabled is None:
            enabled = self.enable_memory_tracking
        
        if not enabled:
            yield
            return
        
        try:
            # Take initial memory snapshot
            initial_memory = self._get_memory_snapshot()
            self._log_debug_event("memory_tracking_start", "Memory tracking started", initial_memory)
            
            yield
            
            # Take final memory snapshot
            final_memory = self._get_memory_snapshot()
            
            # Calculate memory differences
            memory_diff = self._calculate_memory_diff(initial_memory, final_memory)
            
            self._log_debug_event(
                "memory_tracking_end",
                "Memory tracking completed",
                {
                    "initial_memory": initial_memory,
                    "final_memory": final_memory,
                    "memory_diff": memory_diff
                }
            )
            
            # Check for memory leaks
            if memory_diff.get("cuda_allocated_increase", 0) > 100 * 1024 * 1024:  # 100MB
                self._log_debug_event(
                    "memory_leak_warning",
                    "Potential CUDA memory leak detected",
                    {"cuda_increase_mb": memory_diff.get("cuda_allocated_increase", 0) / (1024 * 1024)}
                )
                
        except Exception as e:
            self._log_debug_event(
                "memory_tracking_error",
                f"Memory tracking error: {str(e)}",
                {"error": str(e)}
            )
    
    def _get_memory_snapshot(self) -> Dict[str, Any]:
        """Get current memory snapshot"""
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "cpu_memory_percent": psutil.virtual_memory().percent,
            "cpu_memory_available": psutil.virtual_memory().available,
            "cpu_memory_used": psutil.virtual_memory().used
        }
        
        if torch.cuda.is_available():
            snapshot.update({
                "cuda_allocated": torch.cuda.memory_allocated(),
                "cuda_reserved": torch.cuda.memory_reserved(),
                "cuda_max_allocated": torch.cuda.max_memory_allocated(),
                "cuda_max_reserved": torch.cuda.max_memory_reserved(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device()
            })
        
        return snapshot
    
    def _calculate_memory_diff(self, initial: Dict, final: Dict) -> Dict[str, int]:
        """Calculate memory differences"""
        
        diff = {}
        
        for key in ["cuda_allocated", "cuda_reserved"]:
            if key in initial and key in final:
                diff[f"{key}_increase"] = final[key] - initial[key]
        
        return diff
    
    def check_gradients(self, model: nn.Module, gradient_threshold: float = 1.0):
        """Check gradients for anomalies"""
        
        if not self.enable_gradient_checking:
            return
        
        try:
            total_norm = 0.0
            param_count = 0
            
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
                    # Check for NaN gradients
                    if torch.isnan(p.grad).any():
                        self.nan_detections += 1
                        self._log_debug_event(
                            "nan_gradient_detected",
                            f"NaN gradient detected in parameter {param_count}",
                            {
                                "parameter_index": param_count,
                                "parameter_shape": list(p.grad.shape),
                                "nan_count": torch.isnan(p.grad).sum().item()
                            }
                        )
                    
                    # Check for Inf gradients
                    if torch.isinf(p.grad).any():
                        self.inf_detections += 1
                        self._log_debug_event(
                            "inf_gradient_detected",
                            f"Inf gradient detected in parameter {param_count}",
                            {
                                "parameter_index": param_count,
                                "parameter_shape": list(p.grad.shape),
                                "inf_count": torch.isinf(p.grad).sum().item()
                            }
                        )
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                
                # Check for gradient explosion
                if total_norm > gradient_threshold:
                    self.gradient_explosions += 1
                    self._log_debug_event(
                        "gradient_explosion",
                        f"Gradient explosion detected: {total_norm:.6f} > {gradient_threshold}",
                        {
                            "gradient_norm": total_norm,
                            "threshold": gradient_threshold,
                            "parameter_count": param_count
                        }
                    )
                
                # Check for gradient vanishing
                if total_norm < 1e-6:
                    self.gradient_vanishing += 1
                    self._log_debug_event(
                        "gradient_vanishing",
                        f"Gradient vanishing detected: {total_norm:.6f} < 1e-6",
                        {
                            "gradient_norm": total_norm,
                            "parameter_count": param_count
                        }
                    )
                
                return total_norm
            
        except Exception as e:
            self._log_debug_event(
                "gradient_check_error",
                f"Error checking gradients: {str(e)}",
                {"error": str(e), "stack_trace": traceback.format_exc()}
            )
        
        return 0.0
    
    def check_model_parameters(self, model: nn.Module):
        """Check model parameters for anomalies"""
        
        try:
            param_stats = {
                "total_parameters": 0,
                "nan_parameters": 0,
                "inf_parameters": 0,
                "zero_parameters": 0,
                "parameter_ranges": {}
            }
            
            for name, param in model.named_parameters():
                param_data = param.data
                param_count = param_data.numel()
                param_stats["total_parameters"] += param_count
                
                # Check for NaN parameters
                if torch.isnan(param_data).any():
                    param_stats["nan_parameters"] += torch.isnan(param_data).sum().item()
                    self._log_debug_event(
                        "nan_parameter_detected",
                        f"NaN parameter detected in {name}",
                        {
                            "parameter_name": name,
                            "parameter_shape": list(param_data.shape),
                            "nan_count": torch.isnan(param_data).sum().item()
                        }
                    )
                
                # Check for Inf parameters
                if torch.isinf(param_data).any():
                    param_stats["inf_parameters"] += torch.isinf(param_data).sum().item()
                    self._log_debug_event(
                        "inf_parameter_detected",
                        f"Inf parameter detected in {name}",
                        {
                            "parameter_name": name,
                            "parameter_shape": list(param_data.shape),
                            "inf_count": torch.isinf(param_data).sum().item()
                        }
                    )
                
                # Check for zero parameters
                zero_count = (param_data == 0).sum().item()
                param_stats["zero_parameters"] += zero_count
                
                # Parameter ranges
                param_stats["parameter_ranges"][name] = {
                    "min": param_data.min().item(),
                    "max": param_data.max().item(),
                    "mean": param_data.mean().item(),
                    "std": param_data.std().item()
                }
            
            self._log_debug_event(
                "parameter_check_completed",
                "Model parameter check completed",
                param_stats
            )
            
            return param_stats
            
        except Exception as e:
            self._log_debug_event(
                "parameter_check_error",
                f"Error checking model parameters: {str(e)}",
                {"error": str(e)}
            )
            return {}
    
    def debug_forward_pass(self, model: nn.Module, inputs: torch.Tensor, layer_hooks: bool = True):
        """Debug forward pass with detailed information"""
        
        if not self.debug_mode:
            return model(inputs)
        
        try:
            # Store intermediate activations
            activations = {}
            hooks = []
            
            def hook_fn(name) -> Any:
                def hook(module, input, output) -> Any:
                    activations[name] = {
                        "input_shape": [list(i.shape) if i is not None else None for i in input],
                        "output_shape": list(output.shape) if output is not None else None,
                        "input_stats": {
                            "mean": [i.mean().item() if i is not None else None for i in input],
                            "std": [i.std().item() if i is not None else None for i in input],
                            "min": [i.min().item() if i is not None else None for i in input],
                            "max": [i.max().item() if i is not None else None for i in input]
                        } if input[0] is not None else None,
                        "output_stats": {
                            "mean": output.mean().item() if output is not None else None,
                            "std": output.std().item() if output is not None else None,
                            "min": output.min().item() if output is not None else None,
                            "max": output.max().item() if output is not None else None
                        } if output is not None else None
                    }
                return hook
            
            # Register hooks for all modules
            if layer_hooks:
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules only
                        hook = module.register_forward_hook(hook_fn(name))
                        hooks.append(hook)
            
            # Forward pass with timing
            start_time = time.time()
            with torch.no_grad():
                outputs = model(inputs)
            forward_time = time.time() - start_time
            
            # Log forward pass information
            self._log_debug_event(
                "forward_pass_debug",
                f"Forward pass completed in {forward_time:.4f}s",
                {
                    "input_shape": list(inputs.shape),
                    "output_shape": list(outputs.shape) if outputs is not None else None,
                    "forward_time": forward_time,
                    "activations": activations if layer_hooks else None
                }
            )
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return outputs
            
        except Exception as e:
            self._log_debug_event(
                "forward_pass_error",
                f"Forward pass error: {str(e)}",
                {
                    "error": str(e),
                    "input_shape": list(inputs.shape),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise
    
    def debug_backward_pass(self, loss: torch.Tensor, retain_graph: bool = False):
        """Debug backward pass with gradient information"""
        
        if not self.debug_mode:
            loss.backward(retain_graph=retain_graph)
            return
        
        try:
            # Check loss for anomalies
            if torch.isnan(loss):
                self._log_debug_event(
                    "nan_loss_detected",
                    "NaN loss detected during backward pass",
                    {"loss_value": loss.item()}
                )
            
            if torch.isinf(loss):
                self._log_debug_event(
                    "inf_loss_detected",
                    "Inf loss detected during backward pass",
                    {"loss_value": loss.item()}
                )
            
            # Backward pass with timing
            start_time = time.time()
            loss.backward(retain_graph=retain_graph)
            backward_time = time.time() - start_time
            
            self._log_debug_event(
                "backward_pass_debug",
                f"Backward pass completed in {backward_time:.4f}s",
                {
                    "loss_value": loss.item(),
                    "backward_time": backward_time
                }
            )
            
        except Exception as e:
            self._log_debug_event(
                "backward_pass_error",
                f"Backward pass error: {str(e)}",
                {
                    "error": str(e),
                    "loss_value": loss.item() if not torch.isnan(loss) else "NaN",
                    "stack_trace": traceback.format_exc()
                }
            )
            raise
    
    def debug_training_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        gradient_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """Debug complete training step"""
        
        debug_info = {}
        
        try:
            # Forward pass debugging
            outputs = self.debug_forward_pass(model, inputs)
            debug_info["forward_outputs"] = outputs
            
            # Loss calculation debugging
            loss = loss_fn(outputs, targets)
            debug_info["loss"] = loss.item()
            
            # Check loss for anomalies
            if torch.isnan(loss) or torch.isinf(loss):
                self._log_debug_event(
                    "loss_anomaly",
                    f"Loss anomaly detected: {loss.item()}",
                    {"loss_value": loss.item(), "loss_type": "NaN" if torch.isnan(loss) else "Inf"}
                )
            
            # Backward pass debugging
            self.debug_backward_pass(loss)
            
            # Gradient checking
            gradient_norm = self.check_gradients(model, gradient_threshold)
            debug_info["gradient_norm"] = gradient_norm
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Parameter checking after update
            param_stats = self.check_model_parameters(model)
            debug_info["parameter_stats"] = param_stats
            
            return debug_info
            
        except Exception as e:
            self._log_debug_event(
                "training_step_error",
                f"Training step error: {str(e)}",
                {
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get debug summary"""
        
        return {
            "debug_mode": self.debug_mode,
            "anomaly_detected": self.anomaly_detected,
            "profiler_active": self.profiler_active,
            "nan_detections": self.nan_detections,
            "inf_detections": self.inf_detections,
            "gradient_explosions": self.gradient_explosions,
            "gradient_vanishing": self.gradient_vanishing,
            "debug_events_count": len(self.debug_events),
            "memory_snapshots_count": len(self.memory_snapshots),
            "gradient_violations_count": len(self.gradient_violations)
        }
    
    def save_debug_report(self, filename: str = None):
        """Save debug report to file"""
        
        if filename is None:
            filename = self.log_dir / f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "debug_summary": self.get_debug_summary(),
            "debug_events": self.debug_events,
            "memory_snapshots": self.memory_snapshots,
            "gradient_violations": self.gradient_violations,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        if self.logger:
            self.logger.log_info(f"Debug report saved to {filename}")
        
        return filename
    
    def clear_debug_data(self) -> Any:
        """Clear debug data"""
        
        self.debug_events.clear()
        self.memory_snapshots.clear()
        self.gradient_violations.clear()
        
        # Reset counters
        self.nan_detections = 0
        self.inf_detections = 0
        self.gradient_explosions = 0
        self.gradient_vanishing = 0
        
        if self.logger:
            self.logger.log_info("Debug data cleared")


# Utility functions
def create_pytorch_debugger(
    logger: Optional[TrainingLogger] = None,
    debug_mode: bool = True,
    enable_anomaly_detection: bool = True,
    enable_profiling: bool = False,
    enable_memory_tracking: bool = True,
    enable_gradient_checking: bool = True,
    log_dir: str = "debug_logs"
) -> PyTorchDebugger:
    """Create a PyTorch debugger with default settings"""
    
    return PyTorchDebugger(
        logger=logger,
        debug_mode=debug_mode,
        enable_anomaly_detection=enable_anomaly_detection,
        enable_profiling=enable_profiling,
        enable_memory_tracking=enable_memory_tracking,
        enable_gradient_checking=enable_gradient_checking,
        log_dir=log_dir
    )


@contextmanager
def debug_training_session(
    model: nn.Module,
    logger: Optional[TrainingLogger] = None,
    debug_mode: bool = True,
    enable_anomaly_detection: bool = True,
    enable_profiling: bool = False,
    enable_memory_tracking: bool = True,
    enable_gradient_checking: bool = True
):
    """Context manager for debugging training sessions"""
    
    debugger = create_pytorch_debugger(
        logger=logger,
        debug_mode=debug_mode,
        enable_anomaly_detection=enable_anomaly_detection,
        enable_profiling=enable_profiling,
        enable_memory_tracking=enable_memory_tracking,
        enable_gradient_checking=enable_gradient_checking
    )
    
    try:
        yield debugger
    finally:
        # Save debug report
        debugger.save_debug_report()
        
        if logger:
            logger.log_info("Training debugging session completed")


if __name__ == "__main__":
    # Example usage
    
    # Simple model for testing
    class TestModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x) -> Any:
            return self.linear(x)
    
    # Create debugger
    debugger = create_pytorch_debugger(debug_mode=True)
    
    # Test model
    model = TestModel()
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Debug training step
    with debugger.anomaly_detection():
        with debugger.memory_tracking():
            debug_info = debugger.debug_training_step(
                model, inputs, targets, loss_fn, optimizer
            )
    
    print(f"Debug info: {debug_info}")
    print(f"Debug summary: {debugger.get_debug_summary()}") 