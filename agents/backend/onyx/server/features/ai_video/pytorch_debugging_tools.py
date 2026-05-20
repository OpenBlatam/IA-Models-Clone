from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import psutil
import os
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
PyTorch Debugging Tools

Comprehensive implementation of PyTorch's built-in debugging tools including
autograd.detect_anomaly() and other debugging utilities.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DebugConfig:
    """Configuration for PyTorch debugging tools."""
    enable_anomaly_detection: bool = True
    enable_grad_check: bool = True
    enable_memory_tracking: bool = True
    enable_profiling: bool = True
    enable_tensor_debugging: bool = True
    anomaly_detection_mode: str = "detect_anomaly"  # "detect_anomaly" or "set_detect_anomaly"
    grad_check_numerical: bool = True
    grad_check_analytical: bool = True
    memory_tracking_interval: int = 100  # Track memory every N iterations
    profiling_interval: int = 50  # Profile every N iterations

class PyTorchDebugger:
    """Comprehensive PyTorch debugging tools wrapper."""
    
    def __init__(self, config: DebugConfig = None):
        
    """__init__ function."""
self.config = config or DebugConfig()
        self.debug_state = {
            'anomaly_detection_enabled': False,
            'grad_check_enabled': False,
            'memory_tracking_enabled': False,
            'profiling_enabled': False,
            'tensor_debugging_enabled': False
        }
        self.memory_snapshots = []
        self.profiling_data = []
        self.anomaly_count = 0
        self.grad_check_count = 0
        
        logger.info("PyTorch Debugger initialized")
    
    @contextmanager
    def anomaly_detection(self, enabled: bool = None):
        """Context manager for autograd anomaly detection."""
        enabled = enabled if enabled is not None else self.config.enable_anomaly_detection
        
        if not enabled:
            yield
            return
        
        try:
            # Enable anomaly detection
            if self.config.anomaly_detection_mode == "detect_anomaly":
                autograd.detect_anomaly()
            else:
                autograd.set_detect_anomaly(True)
            
            self.debug_state['anomaly_detection_enabled'] = True
            logger.info("🔍 Anomaly detection enabled")
            
            yield
            
        except Exception as e:
            self.anomaly_count += 1
            logger.error(f"🚨 Anomaly detected: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Disable anomaly detection
            if self.config.anomaly_detection_mode == "detect_anomaly":
                autograd.detect_anomaly()
            else:
                autograd.set_detect_anomaly(False)
            
            self.debug_state['anomaly_detection_enabled'] = False
            logger.info("🔍 Anomaly detection disabled")
    
    @contextmanager
    def grad_check(self, model: nn.Module, enabled: bool = None):
        """Context manager for gradient checking."""
        enabled = enabled if enabled is not None else self.config.enable_grad_check
        
        if not enabled:
            yield
            return
        
        try:
            self.debug_state['grad_check_enabled'] = True
            logger.info("🔍 Gradient checking enabled")
            
            # Store original parameters for comparison
            original_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    original_params[name] = param.data.clone()
            
            yield
            
            # Check gradients after backward pass
            self._check_gradients(model, original_params)
            
        except Exception as e:
            self.grad_check_count += 1
            logger.error(f"🚨 Gradient check failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            self.debug_state['grad_check_enabled'] = False
            logger.info("🔍 Gradient checking disabled")
    
    def _check_gradients(self, model: nn.Module, original_params: Dict[str, torch.Tensor]):
        """Check gradients for anomalies."""
        grad_issues = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                
                # Check for NaN gradients
                if torch.isnan(grad).any():
                    grad_issues.append(f"NaN gradient in {name}")
                
                # Check for infinite gradients
                if torch.isinf(grad).any():
                    grad_issues.append(f"Infinite gradient in {name}")
                
                # Check for extremely large gradients
                grad_norm = grad.norm()
                if grad_norm > 1e6:
                    grad_issues.append(f"Large gradient norm in {name}: {grad_norm}")
                
                # Check for zero gradients (potential dead neurons)
                if grad_norm < 1e-8:
                    grad_issues.append(f"Very small gradient norm in {name}: {grad_norm}")
        
        if grad_issues:
            logger.warning("⚠️ Gradient issues detected:")
            for issue in grad_issues:
                logger.warning(f"  - {issue}")
    
    @contextmanager
    def memory_tracking(self, enabled: bool = None):
        """Context manager for memory tracking."""
        enabled = enabled if enabled is not None else self.config.enable_memory_tracking
        
        if not enabled:
            yield
            return
        
        try:
            self.debug_state['memory_tracking_enabled'] = True
            initial_memory = self._get_memory_usage()
            logger.info(f"🔍 Memory tracking started - Initial: {initial_memory}")
            
            yield
            
            final_memory = self._get_memory_usage()
            memory_diff = {
                'cpu_memory_diff': final_memory['cpu_memory_gb'] - initial_memory['cpu_memory_gb'],
                'gpu_memory_diff': {
                    'allocated_diff': final_memory['gpu_memory_gb']['allocated'] - initial_memory['gpu_memory_gb']['allocated'],
                    'reserved_diff': final_memory['gpu_memory_gb']['reserved'] - initial_memory['gpu_memory_gb']['reserved']
                }
            }
            
            logger.info(f"🔍 Memory tracking completed - Final: {final_memory}")
            logger.info(f"🔍 Memory difference: {memory_diff}")
            
            # Store memory snapshot
            self.memory_snapshots.append({
                'initial': initial_memory,
                'final': final_memory,
                'difference': memory_diff,
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"🚨 Memory tracking error: {e}")
            raise
        finally:
            self.debug_state['memory_tracking_enabled'] = False
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        memory_info = {}
        
        # CPU memory
        try:
            process = psutil.Process()
            memory_info['cpu_memory_gb'] = process.memory_info().rss / (1024**3)
        except Exception:
            memory_info['cpu_memory_gb'] = 0.0
        
        # GPU memory
        if torch.cuda.is_available():
            try:
                memory_info['gpu_memory_gb'] = {
                    'allocated': torch.cuda.memory_allocated() / (1024**3),
                    'reserved': torch.cuda.memory_reserved() / (1024**3),
                    'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)
                }
            except Exception:
                memory_info['gpu_memory_gb'] = {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}
        else:
            memory_info['gpu_memory_gb'] = {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}
        
        return memory_info
    
    @contextmanager
    def profiling(self, enabled: bool = None):
        """Context manager for PyTorch profiling."""
        enabled = enabled if enabled is not None else self.config.enable_profiling
        
        if not enabled:
            yield
            return
        
        try:
            self.debug_state['profiling_enabled'] = True
            logger.info("🔍 Profiling enabled")
            
            # Start profiling
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )
            
            profiler.start()
            
            yield
            
            profiler.stop()
            
            # Log profiling results
            logger.info("🔍 Profiling results:")
            logger.info(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
            # Store profiling data
            self.profiling_data.append({
                'key_averages': profiler.key_averages().table(sort_by="cuda_time_total"),
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"🚨 Profiling error: {e}")
            raise
        finally:
            self.debug_state['profiling_enabled'] = False
            logger.info("🔍 Profiling disabled")
    
    @contextmanager
    def tensor_debugging(self, enabled: bool = None):
        """Context manager for tensor debugging."""
        enabled = enabled if enabled is not None else self.config.enable_tensor_debugging
        
        if not enabled:
            yield
            return
        
        try:
            self.debug_state['tensor_debugging_enabled'] = True
            logger.info("🔍 Tensor debugging enabled")
            
            yield
            
        except Exception as e:
            logger.error(f"🚨 Tensor debugging error: {e}")
            raise
        finally:
            self.debug_state['tensor_debugging_enabled'] = False
            logger.info("🔍 Tensor debugging disabled")
    
    def debug_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> None:
        """Debug tensor properties and values."""
        if not self.debug_state['tensor_debugging_enabled']:
            return
        
        logger.debug(f"🔍 Tensor Debug: {name}")
        logger.debug(f"  Shape: {tensor.shape}")
        logger.debug(f"  Dtype: {tensor.dtype}")
        logger.debug(f"  Device: {tensor.device}")
        logger.debug(f"  Requires grad: {tensor.requires_grad}")
        logger.debug(f"  Min value: {tensor.min().item()}")
        logger.debug(f"  Max value: {tensor.max().item()}")
        logger.debug(f"  Mean value: {tensor.mean().item()}")
        logger.debug(f"  Std value: {tensor.std().item()}")
        logger.debug(f"  Has NaN: {torch.isnan(tensor).any()}")
        logger.debug(f"  Has Inf: {torch.isinf(tensor).any()}")
        
        # Check for potential issues
        if torch.isnan(tensor).any():
            logger.warning(f"⚠️ NaN values detected in {name}")
        
        if torch.isinf(tensor).any():
            logger.warning(f"⚠️ Infinite values detected in {name}")
        
        if tensor.abs().max() > 1e6:
            logger.warning(f"⚠️ Large values detected in {name}: {tensor.abs().max()}")
    
    def debug_model(self, model: nn.Module) -> None:
        """Debug model architecture and parameters."""
        if not self.debug_state['tensor_debugging_enabled']:
            return
        
        logger.debug("🔍 Model Debug:")
        logger.debug(f"  Model type: {type(model).__name__}")
        logger.debug(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
        logger.debug(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        for name, param in model.named_parameters():
            logger.debug(f"  Parameter {name}: {param.shape}, requires_grad: {param.requires_grad}")
            
            if param.grad is not None:
                self.debug_tensor(param.grad, f"grad_{name}")
    
    def debug_gradients(self, model: nn.Module) -> None:
        """Debug gradient information."""
        if not self.debug_state['tensor_debugging_enabled']:
            return
        
        logger.debug("🔍 Gradient Debug:")
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                logger.debug(f"  Gradient {name}: norm={grad_norm:.6f}")
                
                if torch.isnan(param.grad).any():
                    logger.warning(f"⚠️ NaN gradient detected in {name}")
                
                if torch.isinf(param.grad).any():
                    logger.warning(f"⚠️ Infinite gradient detected in {name}")
    
    def check_for_common_issues(self, model: nn.Module, loss: torch.Tensor) -> List[str]:
        """Check for common training issues."""
        issues = []
        
        # Check loss
        if torch.isnan(loss):
            issues.append("Loss is NaN")
        
        if torch.isinf(loss):
            issues.append("Loss is infinite")
        
        if loss.item() > 1e6:
            issues.append("Loss is extremely large")
        
        # Check model parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                issues.append(f"Parameter {name} contains NaN values")
            
            if torch.isinf(param).any():
                issues.append(f"Parameter {name} contains infinite values")
            
            if param.abs().max() > 1e6:
                issues.append(f"Parameter {name} has extremely large values")
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    issues.append(f"Gradient {name} contains NaN values")
                
                if torch.isinf(param.grad).any():
                    issues.append(f"Gradient {name} contains infinite values")
                
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e6:
                    issues.append(f"Gradient {name} has extremely large norm: {grad_norm}")
        
        return issues
    
    def enable_all_debugging(self) -> Any:
        """Enable all debugging features."""
        self.config.enable_anomaly_detection = True
        self.config.enable_grad_check = True
        self.config.enable_memory_tracking = True
        self.config.enable_profiling = True
        self.config.enable_tensor_debugging = True
        logger.info("🔍 All debugging features enabled")
    
    def disable_all_debugging(self) -> Any:
        """Disable all debugging features."""
        self.config.enable_anomaly_detection = False
        self.config.enable_grad_check = False
        self.config.enable_memory_tracking = False
        self.config.enable_profiling = False
        self.config.enable_tensor_debugging = False
        logger.info("🔍 All debugging features disabled")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get debugging summary."""
        return {
            'anomaly_count': self.anomaly_count,
            'grad_check_count': self.grad_check_count,
            'memory_snapshots_count': len(self.memory_snapshots),
            'profiling_data_count': len(self.profiling_data),
            'debug_state': self.debug_state,
            'config': self.config.__dict__
        }
    
    def clear_debug_data(self) -> Any:
        """Clear stored debug data."""
        self.memory_snapshots.clear()
        self.profiling_data.clear()
        self.anomaly_count = 0
        self.grad_check_count = 0
        logger.info("🔍 Debug data cleared")

class DebugTrainer:
    """Training wrapper with integrated debugging tools."""
    
    def __init__(self, model: nn.Module, debugger: PyTorchDebugger = None):
        
    """__init__ function."""
self.model = model
        self.debugger = debugger or PyTorchDebugger()
        self.training_step_count = 0
    
    def training_step(self, 
                     data: torch.Tensor, 
                     targets: torch.Tensor,
                     criterion: nn.Module,
                     optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Training step with comprehensive debugging."""
        self.training_step_count += 1
        
        # Debug input tensors
        if self.debugger.debug_state['tensor_debugging_enabled']:
            self.debugger.debug_tensor(data, "input_data")
            self.debugger.debug_tensor(targets, "targets")
        
        # Enable anomaly detection for this step
        with self.debugger.anomaly_detection():
            # Enable gradient checking
            with self.debugger.grad_check(self.model):
                # Enable memory tracking
                with self.debugger.memory_tracking():
                    # Enable profiling periodically
                    profiling_enabled = (self.training_step_count % self.debugger.config.profiling_interval == 0)
                    with self.debugger.profiling(profiling_enabled):
                        
                        # Forward pass
                        outputs = self.model(data)
                        
                        # Debug outputs
                        if self.debugger.debug_state['tensor_debugging_enabled']:
                            self.debugger.debug_tensor(outputs, "model_outputs")
                        
                        # Compute loss
                        loss = criterion(outputs, targets)
                        
                        # Debug loss
                        if self.debugger.debug_state['tensor_debugging_enabled']:
                            self.debugger.debug_tensor(loss, "loss")
                        
                        # Check for common issues
                        issues = self.debugger.check_for_common_issues(self.model, loss)
                        if issues:
                            logger.warning(f"⚠️ Training issues detected in step {self.training_step_count}:")
                            for issue in issues:
                                logger.warning(f"  - {issue}")
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Debug gradients
                        if self.debugger.debug_state['tensor_debugging_enabled']:
                            self.debugger.debug_gradients(self.model)
                        
                        # Optimizer step
                        optimizer.step()
        
        return {
            'loss': loss.item(),
            'step': self.training_step_count,
            'issues': issues
        }
    
    def debug_model_state(self) -> Any:
        """Debug current model state."""
        if self.debugger.debug_state['tensor_debugging_enabled']:
            self.debugger.debug_model(self.model)

# Example usage
def example_usage():
    """Example of using PyTorch debugging tools."""
    
    # Create debugger
    debug_config = DebugConfig(
        enable_anomaly_detection=True,
        enable_grad_check=True,
        enable_memory_tracking=True,
        enable_profiling=True,
        enable_tensor_debugging=True
    )
    debugger = PyTorchDebugger(debug_config)
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # Create debug trainer
    trainer = DebugTrainer(model, debugger)
    
    # Create dummy data
    data = torch.randn(4, 10)
    targets = torch.randn(4, 1)
    
    # Training components
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop with debugging
    for epoch in range(3):
        logger.info(f"Epoch {epoch + 1}")
        
        for step in range(5):
            result = trainer.training_step(data, targets, criterion, optimizer)
            logger.info(f"Step {step + 1}: Loss = {result['loss']:.4f}")
            
            if result['issues']:
                logger.warning(f"Issues: {result['issues']}")
    
    # Get debug summary
    summary = debugger.get_debug_summary()
    logger.info(f"Debug Summary: {summary}")

match __name__:
    case "__main__":
    example_usage() 