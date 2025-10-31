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
import numpy as np
import logging
import traceback
import sys
import os
import time
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
import threading
import queue
import gc
import psutil
import inspect
from contextlib import contextmanager
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
            import cProfile
            import pstats
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Error Handling and Debugging System
Comprehensive error handling, debugging, and recovery system for deep learning applications.
"""

warnings.filterwarnings('ignore')

# Import our custom modules
try:
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    INPUT_VALIDATION = "input_validation"
    MODEL_ERROR = "model_error"
    TRAINING_ERROR = "training_error"
    EVALUATION_ERROR = "evaluation_error"
    DATA_ERROR = "data_error"
    MEMORY_ERROR = "memory_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    GRADIENT_ERROR = "gradient_error"
    LOSS_ERROR = "loss_error"
    OPTIMIZATION_ERROR = "optimization_error"


class DebugLevel(Enum):
    """Debug levels."""
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    VERBOSE = "verbose"
    PROFILING = "profiling"


@dataclass
class ErrorInfo:
    """Error information structure."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)
    user_data: Dict[str, Any] = field(default_factory=dict)
    recovery_action: Optional[str] = None
    resolved: bool = False


@dataclass
class DebugConfig:
    """Debug configuration."""
    debug_level: DebugLevel = DebugLevel.BASIC
    enable_profiling: bool = False
    enable_memory_tracking: bool = True
    enable_gradient_tracking: bool = True
    enable_loss_tracking: bool = True
    enable_performance_tracking: bool = True
    log_file: str = "debug.log"
    error_file: str = "errors.log"
    max_log_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    enable_console_output: bool = True
    enable_file_output: bool = True


class ErrorTracker:
    """Comprehensive error tracking system."""
    
    def __init__(self, config: DebugConfig):
        
    """__init__ function."""
self.config = config
        self.errors: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_actions: Dict[str, Callable] = {}
        self.logger = self._setup_logging()
        
        # Thread safety
        self._lock = threading.Lock()
        self._error_queue = queue.Queue()
        
        # Start error processing thread
        self._processing_thread = threading.Thread(target=self._process_errors, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self._processing_thread.start()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger('ErrorTracker')
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handlers
        if self.config.enable_file_output:
            # Error log file
            error_handler = logging.handlers.RotatingFileHandler(
                self.config.error_file,
                maxBytes=self.config.max_log_size,
                backupCount=self.config.backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_handler)
            
            # Debug log file
            debug_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_log_size,
                backupCount=self.config.backup_count
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(detailed_formatter)
            logger.addHandler(debug_handler)
        
        # Console handler
        if self.config.enable_console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def track_error(self, error: Exception, severity: ErrorSeverity, 
                   category: ErrorCategory, context: Optional[Dict[str, Any]] = None,
                   user_data: Optional[Dict[str, Any]] = None) -> str:
        """Track an error with comprehensive information."""
        error_id = f"{category.value}_{int(time.time())}_{len(self.errors)}"
        
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {},
            user_data=user_data or {}
        )
        
        # Add to queue for processing
        self._error_queue.put(error_info)
        
        # Update error counts
        with self._lock:
            self.error_counts[category.value] = self.error_counts.get(category.value, 0) + 1
        
        # Log error
        log_level = getattr(logging, severity.value.upper())
        self.logger.log(log_level, f"Error {error_id}: {str(error)}")
        
        # Attempt recovery
        self._attempt_recovery(error_info)
        
        return error_id
    
    def _process_errors(self) -> Any:
        """Process errors from queue."""
        while True:
            try:
                error_info = self._error_queue.get(timeout=1)
                with self._lock:
                    self.errors.append(error_info)
                
                # Log detailed error information
                self.logger.error(f"Detailed error {error_info.error_id}:")
                self.logger.error(f"  Category: {error_info.category.value}")
                self.logger.error(f"  Severity: {error_info.severity.value}")
                self.logger.error(f"  Message: {error_info.message}")
                self.logger.error(f"  Context: {error_info.context}")
                self.logger.error(f"  Traceback: {error_info.traceback}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing error: {e}")
    
    def _attempt_recovery(self, error_info: ErrorInfo):
        """Attempt automatic recovery based on error category."""
        recovery_action = self.recovery_actions.get(error_info.category.value)
        
        if recovery_action:
            try:
                recovery_action(error_info)
                error_info.recovery_action = "Automatic recovery attempted"
                self.logger.info(f"Recovery attempted for error {error_info.error_id}")
            except Exception as e:
                self.logger.error(f"Recovery failed for error {error_info.error_id}: {e}")
    
    def register_recovery_action(self, category: ErrorCategory, action: Callable):
        """Register a recovery action for an error category."""
        self.recovery_actions[category.value] = action
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        with self._lock:
            return {
                'total_errors': len(self.errors),
                'error_counts': self.error_counts.copy(),
                'recent_errors': [
                    {
                        'id': error.error_id,
                        'category': error.category.value,
                        'severity': error.severity.value,
                        'message': error.message,
                        'timestamp': error.timestamp,
                        'resolved': error.resolved
                    }
                    for error in self.errors[-10:]  # Last 10 errors
                ]
            }
    
    def clear_errors(self) -> Any:
        """Clear all tracked errors."""
        with self._lock:
            self.errors.clear()
            self.error_counts.clear()


class Debugger:
    """Advanced debugging system."""
    
    def __init__(self, config: DebugConfig):
        
    """__init__ function."""
self.config = config
        self.error_tracker = ErrorTracker(config)
        self.logger = self.error_tracker.logger
        self.debug_data: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.memory_snapshots: List[Dict[str, Any]] = []
        
        # Setup profiling
        if self.config.enable_profiling:
            self._setup_profiling()
    
    def _setup_profiling(self) -> Any:
        """Setup profiling tools."""
        try:
            self.profiler = cProfile.Profile()
        except ImportError:
            self.logger.warning("cProfile not available, profiling disabled")
            self.profiler = None
    
    @contextmanager
    def debug_context(self, context_name: str, **kwargs):
        """Context manager for debugging operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            self.logger.debug(f"Starting debug context: {context_name}")
            yield
            
        except Exception as e:
            self.error_tracker.track_error(
                e, ErrorSeverity.ERROR, ErrorCategory.SYSTEM_ERROR,
                context={'context_name': context_name, **kwargs}
            )
            raise
        
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_diff = end_memory - start_memory
            
            self.logger.debug(f"Debug context {context_name} completed in {duration:.3f}s")
            self.logger.debug(f"Memory usage: {start_memory:.2f}MB -> {end_memory:.2f}MB (diff: {memory_diff:.2f}MB)")
            
            # Store performance metrics
            if context_name not in self.performance_metrics:
                self.performance_metrics[context_name] = []
            self.performance_metrics[context_name].append(duration)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def track_memory(self) -> Any:
        """Take a memory snapshot."""
        snapshot = {
            'timestamp': time.time(),
            'memory_usage': self._get_memory_usage(),
            'gpu_memory': self._get_gpu_memory_usage(),
            'gc_stats': gc.get_stats()
        }
        self.memory_snapshots.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self.memory_snapshots) > 100:
            self.memory_snapshots = self.memory_snapshots[-100:]
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage if available."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return None
    
    def track_gradients(self, model: nn.Module):
        """Track gradient statistics."""
        if not self.config.enable_gradient_tracking:
            return
        
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                gradient_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'has_nan': torch.isnan(param.grad).any().item(),
                    'has_inf': torch.isinf(param.grad).any().item()
                }
        
        self.debug_data['gradient_stats'] = gradient_stats
        
        # Check for gradient issues
        for name, stats in gradient_stats.items():
            if stats['has_nan'] or stats['has_inf']:
                self.error_tracker.track_error(
                    ValueError(f"Gradient issues in {name}: NaN={stats['has_nan']}, Inf={stats['has_inf']}"),
                    ErrorSeverity.WARNING,
                    ErrorCategory.GRADIENT_ERROR,
                    context={'parameter_name': name, 'stats': stats}
                )
    
    def track_loss(self, loss: torch.Tensor, step: int):
        """Track loss statistics."""
        if not self.config.enable_loss_tracking:
            return
        
        loss_value = loss.item()
        
        if 'loss_history' not in self.debug_data:
            self.debug_data['loss_history'] = []
        
        self.debug_data['loss_history'].append({
            'step': step,
            'loss': loss_value,
            'timestamp': time.time()
        })
        
        # Check for loss issues
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.error_tracker.track_error(
                ValueError(f"Loss contains NaN or Inf: {loss_value}"),
                ErrorSeverity.ERROR,
                ErrorCategory.LOSS_ERROR,
                context={'step': step, 'loss_value': loss_value}
            )
        
        # Check for loss explosion
        if len(self.debug_data['loss_history']) > 10:
            recent_losses = [entry['loss'] for entry in self.debug_data['loss_history'][-10:]]
            if max(recent_losses) > 1000:  # Arbitrary threshold
                self.error_tracker.track_error(
                    ValueError(f"Loss explosion detected: {recent_losses}"),
                    ErrorSeverity.WARNING,
                    ErrorCategory.LOSS_ERROR,
                    context={'recent_losses': recent_losses}
                )
    
    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a function execution."""
        if not self.profiler:
            return func(*args, **kwargs)
        
        self.profiler.enable()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self.profiler.disable()
            
            # Save profiling stats
            stats = pstats.Stats(self.profiler)
            stats_file = f"profile_{func.__name__}_{int(time.time())}.stats"
            stats.dump_stats(stats_file)
            
            self.logger.info(f"Profiling stats saved to {stats_file}")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debug summary."""
        return {
            'error_summary': self.error_tracker.get_error_summary(),
            'performance_metrics': self.performance_metrics,
            'memory_snapshots': self.memory_snapshots[-10:] if self.memory_snapshots else [],
            'debug_data': self.debug_data,
            'config': {
                'debug_level': self.config.debug_level.value,
                'enable_profiling': self.config.enable_profiling,
                'enable_memory_tracking': self.config.enable_memory_tracking,
                'enable_gradient_tracking': self.config.enable_gradient_tracking,
                'enable_loss_tracking': self.config.enable_loss_tracking
            }
        }


class ModelDebugger:
    """Specialized debugger for deep learning models."""
    
    def __init__(self, debugger: Debugger):
        
    """__init__ function."""
self.debugger = debugger
        self.logger = debugger.logger
        self.model_states: List[Dict[str, Any]] = []
    
    def debug_model_forward(self, model: nn.Module, input_data: torch.Tensor, 
                           expected_output_shape: Optional[Tuple] = None):
        """Debug model forward pass."""
        with self.debugger.debug_context("model_forward"):
            # Track input
            self.logger.debug(f"Input shape: {input_data.shape}")
            self.logger.debug(f"Input dtype: {input_data.dtype}")
            self.logger.debug(f"Input range: [{input_data.min().item():.4f}, {input_data.max().item():.4f}]")
            
            # Check for input issues
            if torch.isnan(input_data).any():
                self.debugger.error_tracker.track_error(
                    ValueError("Input contains NaN values"),
                    ErrorSeverity.ERROR,
                    ErrorCategory.INPUT_VALIDATION
                )
            
            if torch.isinf(input_data).any():
                self.debugger.error_tracker.track_error(
                    ValueError("Input contains Inf values"),
                    ErrorSeverity.ERROR,
                    ErrorCategory.INPUT_VALIDATION
                )
            
            # Forward pass with gradient tracking
            model.train()
            output = model(input_data)
            
            # Track output
            self.logger.debug(f"Output shape: {output.shape}")
            self.logger.debug(f"Output dtype: {output.dtype}")
            self.logger.debug(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            # Check expected output shape
            if expected_output_shape and output.shape != expected_output_shape:
                self.debugger.error_tracker.track_error(
                    ValueError(f"Output shape mismatch: expected {expected_output_shape}, got {output.shape}"),
                    ErrorSeverity.ERROR,
                    ErrorCategory.MODEL_ERROR
                )
            
            # Check for output issues
            if torch.isnan(output).any():
                self.debugger.error_tracker.track_error(
                    ValueError("Output contains NaN values"),
                    ErrorSeverity.ERROR,
                    ErrorCategory.MODEL_ERROR
                )
            
            if torch.isinf(output).any():
                self.debugger.error_tracker.track_error(
                    ValueError("Output contains Inf values"),
                    ErrorSeverity.ERROR,
                    ErrorCategory.MODEL_ERROR
                )
            
            return output
    
    def debug_training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                           loss_fn: Callable, data: torch.Tensor, target: torch.Tensor):
        """Debug a complete training step."""
        with self.debugger.debug_context("training_step"):
            # Forward pass
            output = self.debug_model_forward(model, data)
            
            # Loss calculation
            loss = loss_fn(output, target)
            self.debugger.track_loss(loss, step=len(self.model_states))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradients
            self.debugger.track_gradients(model)
            
            # Optimizer step
            optimizer.step()
            
            # Store model state
            self.model_states.append({
                'step': len(self.model_states),
                'loss': loss.item(),
                'output_shape': output.shape,
                'gradient_norms': {
                    name: param.grad.norm().item() if param.grad is not None else 0.0
                    for name, param in model.named_parameters()
                }
            })
            
            return loss
    
    def analyze_model_parameters(self, model: nn.Module):
        """Analyze model parameters for issues."""
        parameter_analysis = {}
        
        for name, param in model.named_parameters():
            analysis = {
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
                'has_nan': torch.isnan(param).any().item(),
                'has_inf': torch.isinf(param).any().item(),
                'norm': param.norm().item(),
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item()
            }
            
            parameter_analysis[name] = analysis
            
            # Check for parameter issues
            if analysis['has_nan'] or analysis['has_inf']:
                self.debugger.error_tracker.track_error(
                    ValueError(f"Parameter {name} contains NaN or Inf values"),
                    ErrorSeverity.ERROR,
                    ErrorCategory.MODEL_ERROR,
                    context={'parameter_name': name, 'analysis': analysis}
                )
        
        return parameter_analysis


class RecoveryManager:
    """Automatic recovery and error mitigation system."""
    
    def __init__(self, debugger: Debugger):
        
    """__init__ function."""
self.debugger = debugger
        self.logger = debugger.logger
        self.recovery_strategies: Dict[str, Callable] = {}
        self.setup_recovery_strategies()
    
    def setup_recovery_strategies(self) -> Any:
        """Setup automatic recovery strategies."""
        # Memory error recovery
        self.recovery_strategies[ErrorCategory.MEMORY_ERROR.value] = self._recover_memory_error
        
        # Gradient error recovery
        self.recovery_strategies[ErrorCategory.GRADIENT_ERROR.value] = self._recover_gradient_error
        
        # Loss error recovery
        self.recovery_strategies[ErrorCategory.LOSS_ERROR.value] = self._recover_loss_error
        
        # Model error recovery
        self.recovery_strategies[ErrorCategory.MODEL_ERROR.value] = self._recover_model_error
        
        # Register with error tracker
        for category, strategy in self.recovery_strategies.items():
            self.debugger.error_tracker.register_recovery_action(
                ErrorCategory(category), strategy
            )
    
    def _recover_memory_error(self, error_info: ErrorInfo):
        """Recover from memory errors."""
        self.logger.info("Attempting memory error recovery...")
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear debug data
        self.debugger.debug_data.clear()
        
        self.logger.info("Memory error recovery completed")
    
    def _recover_gradient_error(self, error_info: ErrorInfo):
        """Recover from gradient errors."""
        self.logger.info("Attempting gradient error recovery...")
        
        # This would typically involve gradient clipping or resetting
        # Implementation depends on the specific context
        
        self.logger.info("Gradient error recovery completed")
    
    def _recover_loss_error(self, error_info: ErrorInfo):
        """Recover from loss errors."""
        self.logger.info("Attempting loss error recovery...")
        
        # This would typically involve learning rate adjustment or loss scaling
        # Implementation depends on the specific context
        
        self.logger.info("Loss error recovery completed")
    
    def _recover_model_error(self, error_info: ErrorInfo):
        """Recover from model errors."""
        self.logger.info("Attempting model error recovery...")
        
        # This would typically involve model reinitialization or checkpoint loading
        # Implementation depends on the specific context
        
        self.logger.info("Model error recovery completed")


class DebuggingInterface:
    """User-friendly debugging interface."""
    
    def __init__(self, config: DebugConfig):
        
    """__init__ function."""
self.config = config
        self.debugger = Debugger(config)
        self.model_debugger = ModelDebugger(self.debugger)
        self.recovery_manager = RecoveryManager(self.debugger)
    
    def debug_training_loop(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                           loss_fn: Callable, dataloader: torch.utils.data.DataLoader,
                           num_epochs: int = 1):
        """Debug a complete training loop."""
        self.logger.info("Starting debug training loop...")
        
        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, (data, target) in enumerate(dataloader):
                try:
                    with self.debugger.debug_context(f"batch_{batch_idx}"):
                        loss = self.model_debugger.debug_training_step(
                            model, optimizer, loss_fn, data, target
                        )
                        
                        if batch_idx % 10 == 0:
                            self.logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                        # Take memory snapshot periodically
                        if batch_idx % 50 == 0:
                            self.debugger.track_memory()
                
                except Exception as e:
                    self.debugger.error_tracker.track_error(
                        e, ErrorSeverity.ERROR, ErrorCategory.TRAINING_ERROR,
                        context={'epoch': epoch, 'batch': batch_idx}
                    )
                    
                    # Attempt recovery
                    self.recovery_manager._recover_training_error(e)
                    
                    # Continue training if possible
                    continue
            
            # Analyze model parameters after each epoch
            parameter_analysis = self.model_debugger.analyze_model_parameters(model)
            self.logger.info(f"Epoch {epoch + 1} parameter analysis completed")
        
        self.logger.info("Debug training loop completed")
    
    def generate_debug_report(self, output_file: str = "debug_report.json"):
        """Generate comprehensive debug report."""
        report = {
            'timestamp': time.time(),
            'debug_summary': self.debugger.get_debug_summary(),
            'model_states': self.model_debugger.model_states,
            'recommendations': self._generate_recommendations()
        }
        
        with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Debug report saved to {output_file}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on debug data."""
        recommendations = []
        
        error_summary = self.debugger.error_tracker.get_error_summary()
        
        # Check for common issues
        if error_summary['error_counts'].get('memory_error', 0) > 0:
            recommendations.append("Consider reducing batch size or model size to address memory issues")
        
        if error_summary['error_counts'].get('gradient_error', 0) > 0:
            recommendations.append("Consider implementing gradient clipping to address gradient issues")
        
        if error_summary['error_counts'].get('loss_error', 0) > 0:
            recommendations.append("Consider adjusting learning rate or loss function to address loss issues")
        
        # Performance recommendations
        if self.debugger.performance_metrics:
            for context, times in self.debugger.performance_metrics.items():
                avg_time = np.mean(times)
                if avg_time > 1.0:  # More than 1 second
                    recommendations.append(f"Consider optimizing {context} (avg time: {avg_time:.3f}s)")
        
        return recommendations


def demonstrate_error_handling():
    """Demonstrate the error handling and debugging system."""
    print("Error Handling and Debugging System Demonstration")
    print("=" * 60)
    
    # Create debug configuration
    config = DebugConfig(
        debug_level=DebugLevel.DETAILED,
        enable_profiling=True,
        enable_memory_tracking=True,
        enable_gradient_tracking=True,
        enable_loss_tracking=True,
        enable_performance_tracking=True
    )
    
    # Create debugging interface
    debug_interface = DebuggingInterface(config)
    
    # Create simple model for demonstration
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create sample dataset
    class SampleDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100) -> Any:
            self.data = torch.randn(num_samples, 784)
            self.targets = torch.randint(0, 10, (num_samples,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.targets[idx]
    
    dataset = SampleDataset(100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Run debug training
    try:
        debug_interface.debug_training_loop(model, optimizer, loss_fn, dataloader, num_epochs=2)
        
        # Generate debug report
        report = debug_interface.generate_debug_report()
        
        print("\nDebug Report Summary:")
        print(f"Total errors: {report['debug_summary']['error_summary']['total_errors']}")
        print(f"Error counts: {report['debug_summary']['error_summary']['error_counts']}")
        print(f"Recommendations: {report['recommendations']}")
        
    except Exception as e:
        print(f"Demonstration error: {e}")
    
    print("\nError handling and debugging demonstration completed!")


if __name__ == "__main__":
    # Demonstrate error handling and debugging
    demonstrate_error_handling() 