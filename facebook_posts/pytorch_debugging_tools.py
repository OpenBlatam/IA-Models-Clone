from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import logging
import traceback
import time
import os
import gc
import json
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterator, ContextManager
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
import threading
import queue
from contextlib import contextmanager
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
    from deep_learning_framework import DeepLearningFramework, FrameworkConfig, TaskType
    from evaluation_metrics import EvaluationManager, MetricConfig, MetricType
    from gradient_clipping_nan_handling import NumericalStabilityManager
    from early_stopping_scheduling import TrainingManager
    from efficient_data_loading import EfficientDataLoader
    from data_splitting_validation import DataSplitter
    from training_evaluation import TrainingManager as TrainingEvalManager
    from diffusion_models import DiffusionModel, DiffusionConfig
    from advanced_transformers import AdvancedTransformerModel
    from llm_training import AdvancedLLMTrainer
    from model_finetuning import ModelFineTuner
    from custom_modules import AdvancedNeuralNetwork
    from weight_initialization import AdvancedWeightInitializer
    from normalization_techniques import AdvancedLayerNorm
    from loss_functions import AdvancedCrossEntropyLoss
    from optimization_algorithms import AdvancedAdamW
    from attention_mechanisms import MultiHeadAttention
    from tokenization_sequence import AdvancedTokenizer
    from framework_utils import MetricsTracker, ModelAnalyzer, PerformanceMonitor
    from deep_learning_integration import DeepLearningIntegration, IntegrationConfig, IntegrationType, ComponentType
    from robust_error_handling import RobustErrorHandler, RobustDataLoader, RobustModelHandler
    from training_logging_system import TrainingLogger, TrainingProgressTracker, TrainingLoggingManager
            import psutil
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
PyTorch Debugging Tools
Comprehensive debugging system using PyTorch's built-in debugging tools.
"""

warnings.filterwarnings('ignore')

# Import our custom modules
try:
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")


class DebugMode(Enum):
    """PyTorch debugging modes."""
    NONE = "none"
    ANOMALY_DETECTION = "anomaly_detection"
    GRADIENT_CHECKING = "gradient_checking"
    MEMORY_PROFILING = "memory_profiling"
    PERFORMANCE_PROFILING = "performance_profiling"
    FULL_DEBUG = "full_debug"


class DebugLevel(Enum):
    """Debug levels for PyTorch debugging."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class DebugConfig:
    """Configuration for PyTorch debugging."""
    mode: DebugMode = DebugMode.NONE
    level: DebugLevel = DebugLevel.BASIC
    enable_anomaly_detection: bool = False
    enable_gradient_checking: bool = False
    enable_memory_profiling: bool = False
    enable_performance_profiling: bool = False
    anomaly_detection_mode: str = "warn"  # "warn" or "raise"
    gradient_checking_frequency: int = 100
    memory_profiling_frequency: int = 50
    performance_profiling_frequency: int = 10
    save_debug_info: bool = True
    debug_output_dir: str = "debug_outputs"
    log_gradients: bool = False
    log_activations: bool = False
    log_weights: bool = False
    log_memory: bool = True
    log_performance: bool = True


@dataclass
class DebugInfo:
    """Debug information structure."""
    timestamp: datetime
    mode: DebugMode
    level: DebugLevel
    operation: str
    model_name: str
    epoch: int = 0
    batch: int = 0
    gradient_norm: float = 0.0
    memory_usage: float = 0.0
    gpu_memory: float = 0.0
    execution_time: float = 0.0
    anomaly_detected: bool = False
    gradient_check_passed: bool = True
    memory_leak_detected: bool = False
    performance_issue_detected: bool = False
    debug_data: Dict[str, Any] = field(default_factory=dict)


class PyTorchDebugger:
    """Comprehensive PyTorch debugging system using built-in tools."""
    
    def __init__(self, config: DebugConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logger()
        self.debug_history: List[DebugInfo] = []
        self.anomaly_detection_active = False
        self.gradient_checking_active = False
        self.memory_profiling_active = False
        self.performance_profiling_active = False
        
        # Debug output directory
        self.debug_dir = Path(config.debug_output_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.memory_history: List[Dict[str, Any]] = []
        self.gradient_history: List[Dict[str, Any]] = []
        
        # Initialize debugging tools
        self._initialize_debugging_tools()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for debugging."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("pytorch_debugger")
    
    def _initialize_debugging_tools(self) -> Any:
        """Initialize PyTorch debugging tools."""
        if self.config.enable_anomaly_detection:
            self._setup_anomaly_detection()
        
        if self.config.enable_gradient_checking:
            self._setup_gradient_checking()
        
        if self.config.enable_memory_profiling:
            self._setup_memory_profiling()
        
        if self.config.enable_performance_profiling:
            self._setup_performance_profiling()
    
    def _setup_anomaly_detection(self) -> Any:
        """Setup PyTorch anomaly detection."""
        try:
            if self.config.anomaly_detection_mode == "warn":
                torch.autograd.set_detect_anomaly(True)
                self.logger.info("PyTorch anomaly detection enabled (warn mode)")
            elif self.config.anomaly_detection_mode == "raise":
                torch.autograd.set_detect_anomaly(True)
                self.logger.info("PyTorch anomaly detection enabled (raise mode)")
            
            self.anomaly_detection_active = True
            
        except Exception as e:
            self.logger.error(f"Failed to setup anomaly detection: {str(e)}")
    
    def _setup_gradient_checking(self) -> Any:
        """Setup gradient checking."""
        try:
            # Enable gradient checking for specific operations
            self.gradient_checking_active = True
            self.logger.info("Gradient checking enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to setup gradient checking: {str(e)}")
    
    def _setup_memory_profiling(self) -> Any:
        """Setup memory profiling."""
        try:
            self.memory_profiling_active = True
            self.logger.info("Memory profiling enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to setup memory profiling: {str(e)}")
    
    def _setup_performance_profiling(self) -> Any:
        """Setup performance profiling."""
        try:
            self.performance_profiling_active = True
            self.logger.info("Performance profiling enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to setup performance profiling: {str(e)}")
    
    @contextmanager
    def debug_context(self, operation: str, model_name: str = "unknown", epoch: int = 0, batch: int = 0):
        """Context manager for debugging operations."""
        debug_info = DebugInfo(
            timestamp=datetime.now(),
            mode=self.config.mode,
            level=self.config.level,
            operation=operation,
            model_name=model_name,
            epoch=epoch,
            batch=batch
        )
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        
        try:
            # Enable debugging tools if needed
            if self.anomaly_detection_active:
                torch.autograd.set_detect_anomaly(True)
            
            yield debug_info
            
        except Exception as e:
            debug_info.anomaly_detected = True
            self.logger.error(f"Debug anomaly detected in {operation}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        finally:
            # Calculate metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            debug_info.execution_time = end_time - start_time
            debug_info.memory_usage = end_memory - start_memory
            debug_info.gpu_memory = end_gpu_memory - start_gpu_memory
            
            # Disable debugging tools
            if self.anomaly_detection_active:
                torch.autograd.set_detect_anomaly(False)
            
            # Save debug info
            self.debug_history.append(debug_info)
            self._save_debug_info(debug_info)
    
    def debug_forward_pass(self, model: nn.Module, input_data: torch.Tensor, **kwargs):
        """Debug forward pass with comprehensive monitoring."""
        with self.debug_context("forward_pass", model.__class__.__name__, **kwargs) as debug_info:
            # Log input information
            self.logger.debug(f"Forward pass input shape: {input_data.shape}")
            self.logger.debug(f"Input dtype: {input_data.dtype}")
            self.logger.debug(f"Input device: {input_data.device}")
            
            # Check for NaN/Inf in input
            if torch.isnan(input_data).any():
                self.logger.warning("NaN detected in input data")
                debug_info.anomaly_detected = True
            
            if torch.isinf(input_data).any():
                self.logger.warning("Inf detected in input data")
                debug_info.anomaly_detected = True
            
            # Perform forward pass
            output = model(input_data)
            
            # Log output information
            self.logger.debug(f"Forward pass output shape: {output.shape}")
            self.logger.debug(f"Output dtype: {output.dtype}")
            
            # Check for NaN/Inf in output
            if torch.isnan(output).any():
                self.logger.warning("NaN detected in output")
                debug_info.anomaly_detected = True
            
            if torch.isinf(output).any():
                self.logger.warning("Inf detected in output")
                debug_info.anomaly_detected = True
            
            # Log activations if enabled
            if self.config.log_activations:
                self._log_activations(model, debug_info)
            
            return output
    
    def debug_backward_pass(self, loss: torch.Tensor, model: nn.Module, **kwargs):
        """Debug backward pass with gradient monitoring."""
        with self.debug_context("backward_pass", model.__class__.__name__, **kwargs) as debug_info:
            # Log loss information
            self.logger.debug(f"Loss value: {loss.item()}")
            self.logger.debug(f"Loss shape: {loss.shape}")
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss).any():
                self.logger.warning("NaN detected in loss")
                debug_info.anomaly_detected = True
            
            if torch.isinf(loss).any():
                self.logger.warning("Inf detected in loss")
                debug_info.anomaly_detected = True
            
            # Perform backward pass
            loss.backward()
            
            # Check gradients
            gradient_norm = self._check_gradients(model)
            debug_info.gradient_norm = gradient_norm
            
            # Log gradient information
            self.logger.debug(f"Gradient norm: {gradient_norm}")
            
            # Check for gradient anomalies
            if gradient_norm > 10.0:
                self.logger.warning(f"High gradient norm: {gradient_norm}")
                debug_info.anomaly_detected = True
            
            if gradient_norm == 0.0:
                self.logger.warning("Zero gradient norm detected")
                debug_info.anomaly_detected = True
            
            # Log gradients if enabled
            if self.config.log_gradients:
                self._log_gradients(model, debug_info)
            
            return gradient_norm
    
    def debug_optimization_step(self, optimizer: torch.optim.Optimizer, model: nn.Module, **kwargs):
        """Debug optimization step with parameter monitoring."""
        with self.debug_context("optimization_step", model.__class__.__name__, **kwargs) as debug_info:
            # Log optimizer information
            self.logger.debug(f"Optimizer type: {type(optimizer).__name__}")
            self.logger.debug(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            
            # Check parameters before optimization
            param_norm_before = self._check_parameters(model)
            self.logger.debug(f"Parameter norm before optimization: {param_norm_before}")
            
            # Perform optimization step
            optimizer.step()
            
            # Check parameters after optimization
            param_norm_after = self._check_parameters(model)
            self.logger.debug(f"Parameter norm after optimization: {param_norm_after}")
            
            # Check for parameter anomalies
            param_change = abs(param_norm_after - param_norm_before)
            if param_change > 1.0:
                self.logger.warning(f"Large parameter change: {param_change}")
                debug_info.anomaly_detected = True
            
            # Log weights if enabled
            if self.config.log_weights:
                self._log_weights(model, debug_info)
            
            return param_norm_after
    
    def debug_memory_usage(self, **kwargs) -> Any:
        """Debug memory usage with detailed profiling."""
        with self.debug_context("memory_profiling", **kwargs) as debug_info:
            # Get memory information
            cpu_memory = self._get_memory_usage()
            gpu_memory = self._get_gpu_memory_usage()
            
            debug_info.memory_usage = cpu_memory
            debug_info.gpu_memory = gpu_memory
            
            # Log memory information
            self.logger.debug(f"CPU memory usage: {cpu_memory:.2f} MB")
            self.logger.debug(f"GPU memory usage: {gpu_memory:.2f} MB")
            
            # Check for memory issues
            if cpu_memory > 1000:  # 1GB threshold
                self.logger.warning(f"High CPU memory usage: {cpu_memory:.2f} MB")
                debug_info.memory_leak_detected = True
            
            if gpu_memory > 8000:  # 8GB threshold
                self.logger.warning(f"High GPU memory usage: {gpu_memory:.2f} MB")
                debug_info.memory_leak_detected = True
            
            # Save memory history
            memory_data = {
                'timestamp': datetime.now().isoformat(),
                'cpu_memory': cpu_memory,
                'gpu_memory': gpu_memory,
                'epoch': debug_info.epoch,
                'batch': debug_info.batch
            }
            self.memory_history.append(memory_data)
            
            return cpu_memory, gpu_memory
    
    def debug_performance(self, operation: str, **kwargs):
        """Debug performance with timing and profiling."""
        with self.debug_context("performance_profiling", operation, **kwargs) as debug_info:
            # Get performance information
            execution_time = debug_info.execution_time
            
            # Log performance information
            self.logger.debug(f"Operation: {operation}")
            self.logger.debug(f"Execution time: {execution_time:.4f} seconds")
            
            # Check for performance issues
            if execution_time > 10.0:  # 10 second threshold
                self.logger.warning(f"Slow operation detected: {execution_time:.4f} seconds")
                debug_info.performance_issue_detected = True
            
            # Save performance history
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'execution_time': execution_time,
                'epoch': debug_info.epoch,
                'batch': debug_info.batch
            }
            self.performance_history.append(performance_data)
            
            return execution_time
    
    def _check_gradients(self, model: nn.Module) -> float:
        """Check gradients for anomalies."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Check for NaN/Inf in gradients
                if torch.isnan(param.grad).any():
                    self.logger.warning(f"NaN detected in gradients of {param}")
                
                if torch.isinf(param.grad).any():
                    self.logger.warning(f"Inf detected in gradients of {param}")
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        
        return total_norm
    
    def _check_parameters(self, model: nn.Module) -> float:
        """Check parameters for anomalies."""
        total_norm = 0.0
        
        for param in model.parameters():
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # Check for NaN/Inf in parameters
            if torch.isnan(param).any():
                self.logger.warning(f"NaN detected in parameters of {param}")
            
            if torch.isinf(param).any():
                self.logger.warning(f"Inf detected in parameters of {param}")
        
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def _log_activations(self, model: nn.Module, debug_info: DebugInfo):
        """Log activation information."""
        activations = {}
        
        def hook_fn(name) -> Any:
            def hook(module, input, output) -> Any:
                activations[name] = {
                    'shape': output.shape,
                    'dtype': output.dtype,
                    'device': output.device,
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Store activation info
        debug_info.debug_data['activations'] = activations
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    def _log_gradients(self, model: nn.Module, debug_info: DebugInfo):
        """Log gradient information."""
        gradients = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = {
                    'shape': param.grad.shape,
                    'dtype': param.grad.dtype,
                    'device': param.grad.device,
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'min': param.grad.min().item(),
                    'max': param.grad.max().item(),
                    'norm': param.grad.norm().item()
                }
        
        debug_info.debug_data['gradients'] = gradients
    
    def _log_weights(self, model: nn.Module, debug_info: DebugInfo):
        """Log weight information."""
        weights = {}
        
        for name, param in model.named_parameters():
            weights[name] = {
                'shape': param.shape,
                'dtype': param.dtype,
                'device': param.device,
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'norm': param.norm().item()
            }
        
        debug_info.debug_data['weights'] = weights
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
            return 0.0
        except Exception:
            return 0.0
    
    def _save_debug_info(self, debug_info: DebugInfo):
        """Save debug information to file."""
        if not self.config.save_debug_info:
            return
        
        debug_data = {
            'timestamp': debug_info.timestamp.isoformat(),
            'mode': debug_info.mode.value,
            'level': debug_info.level.value,
            'operation': debug_info.operation,
            'model_name': debug_info.model_name,
            'epoch': debug_info.epoch,
            'batch': debug_info.batch,
            'gradient_norm': debug_info.gradient_norm,
            'memory_usage': debug_info.memory_usage,
            'gpu_memory': debug_info.gpu_memory,
            'execution_time': debug_info.execution_time,
            'anomaly_detected': debug_info.anomaly_detected,
            'gradient_check_passed': debug_info.gradient_check_passed,
            'memory_leak_detected': debug_info.memory_leak_detected,
            'performance_issue_detected': debug_info.performance_issue_detected,
            'debug_data': debug_info.debug_data
        }
        
        # Save to JSON file
        debug_file = self.debug_dir / f"debug_info_{debug_info.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(debug_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(debug_data, f, indent=2)
    
    def create_debug_report(self) -> Dict[str, Any]:
        """Create comprehensive debug report."""
        report = {
            'summary': {
                'total_debug_operations': len(self.debug_history),
                'anomalies_detected': sum(1 for info in self.debug_history if info.anomaly_detected),
                'memory_leaks_detected': sum(1 for info in self.debug_history if info.memory_leak_detected),
                'performance_issues_detected': sum(1 for info in self.debug_history if info.performance_issue_detected),
                'debug_mode': self.config.mode.value,
                'debug_level': self.config.level.value
            },
            'performance_analysis': {
                'average_execution_time': np.mean([info.execution_time for info in self.debug_history]),
                'max_execution_time': max([info.execution_time for info in self.debug_history], default=0),
                'min_execution_time': min([info.execution_time for info in self.debug_history], default=0)
            },
            'memory_analysis': {
                'average_memory_usage': np.mean([info.memory_usage for info in self.debug_history]),
                'max_memory_usage': max([info.memory_usage for info in self.debug_history], default=0),
                'average_gpu_memory': np.mean([info.gpu_memory for info in self.debug_history]),
                'max_gpu_memory': max([info.gpu_memory for info in self.debug_history], default=0)
            },
            'gradient_analysis': {
                'average_gradient_norm': np.mean([info.gradient_norm for info in self.debug_history]),
                'max_gradient_norm': max([info.gradient_norm for info in self.debug_history], default=0),
                'gradient_anomalies': sum(1 for info in self.debug_history if info.gradient_norm > 10.0)
            },
            'operation_breakdown': defaultdict(int)
        }
        
        # Count operations
        for info in self.debug_history:
            report['operation_breakdown'][info.operation] += 1
        
        report['operation_breakdown'] = dict(report['operation_breakdown'])
        
        return report
    
    def save_debug_report(self, filename: str = None):
        """Save debug report to file."""
        if filename is None:
            filename = f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.create_debug_report()
        report_file = self.debug_dir / filename
        
        with open(report_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Debug report saved to {report_file}")
    
    def create_debug_plots(self) -> Any:
        """Create debug visualization plots."""
        if not self.debug_history:
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PyTorch Debug Analysis', fontsize=16)
        
        # Extract data
        timestamps = [info.timestamp for info in self.debug_history]
        execution_times = [info.execution_time for info in self.debug_history]
        memory_usage = [info.memory_usage for info in self.debug_history]
        gpu_memory = [info.gpu_memory for info in self.debug_history]
        gradient_norms = [info.gradient_norm for info in self.debug_history]
        
        # Execution time plot
        axes[0, 0].plot(timestamps, execution_times, 'b-')
        axes[0, 0].set_title('Execution Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True)
        
        # Memory usage plot
        axes[0, 1].plot(timestamps, memory_usage, 'g-', label='CPU Memory')
        axes[0, 1].plot(timestamps, gpu_memory, 'r-', label='GPU Memory')
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True)
        
        # Gradient norm plot
        axes[1, 0].plot(timestamps, gradient_norms, 'orange')
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True)
        
        # Anomaly detection plot
        anomaly_timestamps = [info.timestamp for info in self.debug_history if info.anomaly_detected]
        anomaly_counts = [1] * len(anomaly_timestamps)
        
        if anomaly_timestamps:
            axes[1, 1].scatter(anomaly_timestamps, anomaly_counts, color='red', alpha=0.7)
        axes[1, 1].set_title('Anomaly Detection')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Anomalies')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.debug_dir / 'debug_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Debug plots saved to {self.debug_dir}")


class PyTorchDebugManager:
    """High-level manager for PyTorch debugging."""
    
    def __init__(self, config: DebugConfig):
        
    """__init__ function."""
self.debugger = PyTorchDebugger(config)
        self.logger = self.debugger.logger
        self.training_logger = None
    
    def setup_training_logging(self, experiment_name: str = None):
        """Setup training logging integration."""
        self.training_logger = TrainingLoggingManager(experiment_name)
    
    def debug_training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                           loss_fn: Callable, data_batch: torch.Tensor,
                           target_batch: torch.Tensor, epoch: int = 0, batch: int = 0):
        """Debug complete training step."""
        # Debug forward pass
        output = self.debugger.debug_forward_pass(
            model, data_batch,
            epoch=epoch, batch=batch
        )
        
        # Debug loss computation
        loss = loss_fn(output, target_batch)
        
        # Debug backward pass
        gradient_norm = self.debugger.debug_backward_pass(
            loss, model,
            epoch=epoch, batch=batch
        )
        
        # Debug optimization step
        param_norm = self.debugger.debug_optimization_step(
            optimizer, model,
            epoch=epoch, batch=batch
        )
        
        # Debug memory usage
        cpu_memory, gpu_memory = self.debugger.debug_memory_usage(
            epoch=epoch, batch=batch
        )
        
        # Debug performance
        execution_time = self.debugger.debug_performance(
            "training_step",
            epoch=epoch, batch=batch
        )
        
        # Log to training logger if available
        if self.training_logger:
            metrics = {
                'loss': loss.item(),
                'gradient_norm': gradient_norm,
                'param_norm': param_norm,
                'cpu_memory': cpu_memory,
                'gpu_memory': gpu_memory,
                'execution_time': execution_time
            }
            self.training_logger.log_batch_metrics(metrics)
        
        return {
            'loss': loss.item(),
            'gradient_norm': gradient_norm,
            'param_norm': param_norm,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'execution_time': execution_time
        }
    
    def debug_model_inference(self, model: nn.Module, input_data: torch.Tensor, **kwargs):
        """Debug model inference."""
        # Debug forward pass
        output = self.debugger.debug_forward_pass(
            model, input_data, **kwargs
        )
        
        # Debug memory usage
        cpu_memory, gpu_memory = self.debugger.debug_memory_usage(**kwargs)
        
        # Debug performance
        execution_time = self.debugger.debug_performance("inference", **kwargs)
        
        return {
            'output': output,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'execution_time': execution_time
        }
    
    def create_comprehensive_report(self) -> Any:
        """Create comprehensive debug report."""
        # Create debug report
        debug_report = self.debugger.create_debug_report()
        
        # Save debug report
        self.debugger.save_debug_report()
        
        # Create debug plots
        self.debugger.create_debug_plots()
        
        # Print summary
        self._print_debug_summary(debug_report)
        
        return debug_report
    
    def _print_debug_summary(self, report: Dict[str, Any]):
        """Print debug summary."""
        summary = report['summary']
        
        print("\n" + "=" * 60)
        print("PYTORCH DEBUG SUMMARY")
        print("=" * 60)
        print(f"Total debug operations: {summary['total_debug_operations']}")
        print(f"Anomalies detected: {summary['anomalies_detected']}")
        print(f"Memory leaks detected: {summary['memory_leaks_detected']}")
        print(f"Performance issues detected: {summary['performance_issues_detected']}")
        print(f"Debug mode: {summary['debug_mode']}")
        print(f"Debug level: {summary['debug_level']}")
        print("=" * 60)


def demonstrate_pytorch_debugging():
    """Demonstrate PyTorch debugging tools."""
    print("PyTorch Debugging Tools Demonstration")
    print("=" * 60)
    
    # Create debug configuration
    config = DebugConfig(
        mode=DebugMode.FULL_DEBUG,
        level=DebugLevel.ADVANCED,
        enable_anomaly_detection=True,
        enable_gradient_checking=True,
        enable_memory_profiling=True,
        enable_performance_profiling=True,
        log_gradients=True,
        log_activations=True,
        log_weights=True
    )
    
    # Create debug manager
    debug_manager = PyTorchDebugManager(config)
    debug_manager.setup_training_logging("debug_experiment")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Simulate training with debugging
    print("\n1. Debugging Training Step:")
    for epoch in range(2):
        for batch in range(3):
            # Create dummy data
            data_batch = torch.randn(32, 784)
            target_batch = torch.randint(0, 10, (32,))
            
            # Debug training step
            metrics = debug_manager.debug_training_step(
                model, optimizer, loss_fn, data_batch, target_batch,
                epoch=epoch, batch=batch
            )
            
            print(f"Epoch {epoch}, Batch {batch}:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Gradient norm: {metrics['gradient_norm']:.4f}")
            print(f"  Execution time: {metrics['execution_time']:.4f}s")
    
    # Debug model inference
    print("\n2. Debugging Model Inference:")
    input_data = torch.randn(1, 784)
    inference_result = debug_manager.debug_model_inference(model, input_data)
    
    print(f"Inference execution time: {inference_result['execution_time']:.4f}s")
    print(f"Output shape: {inference_result['output'].shape}")
    
    # Create comprehensive report
    print("\n3. Creating Debug Report:")
    report = debug_manager.create_comprehensive_report()
    
    print("\nPyTorch debugging demonstration completed!")
    print(f"Debug files saved to: {debug_manager.debugger.debug_dir}")


if __name__ == "__main__":
    # Demonstrate PyTorch debugging
    demonstrate_pytorch_debugging() 