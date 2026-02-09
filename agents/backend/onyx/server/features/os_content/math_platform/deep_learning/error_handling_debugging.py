from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import numpy as np
import logging
import traceback
import sys
import os
import time
import gc
import psutil
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings
from abc import ABC, abstractmethod
import functools
import inspect
import threading
import queue
from contextlib import contextmanager
import signal
import weakref
from pathlib import Path
import pickle
import hashlib
import linecache
import pdb
import ipdb
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
            import pandas as pd
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Error Handling and Debugging for Deep Learning
Comprehensive error handling, debugging utilities, and monitoring for deep learning operations.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling and debugging."""
    # Error handling parameters
    enable_error_handling: bool = True
    enable_debugging: bool = True
    enable_monitoring: bool = True
    enable_logging: bool = True
    
    # Logging parameters
    log_level: str = "INFO"
    log_file: str = "./logs/deep_learning_errors.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring parameters
    monitor_memory: bool = True
    monitor_gpu: bool = True
    monitor_performance: bool = True
    memory_threshold: float = 0.9  # 90% of available memory
    gpu_memory_threshold: float = 0.9  # 90% of GPU memory
    
    # Debugging parameters
    enable_breakpoints: bool = False
    enable_traceback: bool = True
    enable_variable_inspection: bool = True
    max_traceback_depth: int = 10
    
    # Recovery parameters
    enable_auto_recovery: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_graceful_degradation: bool = True
    
    # Data validation parameters
    validate_inputs: bool = True
    validate_outputs: bool = True
    enable_sanitization: bool = True
    
    # Performance parameters
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_time: bool = True
    enable_optimization_suggestions: bool = True


class DeepLearningError(Exception):
    """Base exception for deep learning operations."""
    
    def __init__(self, message: str, error_type: str = "general", context: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(message)
        self.error_type = error_type
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()


class DataLoadingError(DeepLearningError):
    """Exception for data loading errors."""
    
    def __init__(self, message: str, data_path: str = None, context: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(message, "data_loading", context)
        self.data_path = data_path


class ModelInferenceError(DeepLearningError):
    """Exception for model inference errors."""
    
    def __init__(self, message: str, model_name: str = None, input_shape: Tuple = None, context: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(message, "model_inference", context)
        self.model_name = model_name
        self.input_shape = input_shape


class MemoryError(DeepLearningError):
    """Exception for memory-related errors."""
    
    def __init__(self, message: str, memory_type: str = "cpu", usage: float = 0.0, context: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(message, "memory", context)
        self.memory_type = memory_type
        self.usage = usage


class ValidationError(DeepLearningError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, validation_type: str = "input", context: Dict[str, Any] = None):
        
    """__init__ function."""
super().__init__(message, "validation", context)
        self.validation_type = validation_type


class ErrorHandler:
    """Comprehensive error handling for deep learning operations."""
    
    def __init__(self, config: ErrorHandlingConfig):
        
    """__init__ function."""
self.config = config
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self._setup_logging()
        self._setup_monitoring()
    
    def _setup_logging(self) -> Any:
        """Setup logging configuration."""
        if not self.config.enable_logging:
            return
        
        os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format,
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _setup_monitoring(self) -> Any:
        """Setup monitoring systems."""
        if self.config.enable_monitoring:
            self.memory_monitor = MemoryMonitor(self.config)
            self.performance_monitor = PerformanceMonitor(self.config)
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle errors with comprehensive logging and recovery strategies."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc() if self.config.enable_traceback else None
        }
        
        # Log error
        if self.config.enable_logging:
            logger.error(f"Error occurred: {error_info['error_type']}: {error_info['error_message']}")
            if self.config.enable_traceback:
                logger.error(f"Traceback: {error_info['traceback']}")
        
        # Update error counts
        self.error_counts[error_info['error_type']] += 1
        self.error_history.append(error_info)
        
        # Attempt recovery
        if self.config.enable_auto_recovery:
            recovery_result = self._attempt_recovery(error, context)
            error_info['recovery_attempted'] = True
            error_info['recovery_successful'] = recovery_result['success']
            error_info['recovery_action'] = recovery_result['action']
        
        # Monitor system state
        if self.config.enable_monitoring:
            error_info['system_state'] = self._get_system_state()
        
        return error_info
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Attempt to recover from errors."""
        error_type = type(error).__name__
        
        # Memory-related errors
        if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
            return self._recover_from_memory_error(error, context)
        
        # Data loading errors
        elif isinstance(error, (FileNotFoundError, OSError, DataLoadingError)):
            return self._recover_from_data_error(error, context)
        
        # Model inference errors
        elif isinstance(error, (RuntimeError, ModelInferenceError)):
            return self._recover_from_inference_error(error, context)
        
        # Validation errors
        elif isinstance(error, (ValueError, ValidationError)):
            return self._recover_from_validation_error(error, context)
        
        # Default recovery strategy
        else:
            return self._recover_generic(error, context)
    
    def _recover_from_memory_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recover from memory-related errors."""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Clear CPU memory
            gc.collect()
            
            # Reduce batch size if possible
            if context and 'batch_size' in context:
                new_batch_size = max(1, context['batch_size'] // 2)
                context['batch_size'] = new_batch_size
                return {
                    'success': True,
                    'action': f'Reduced batch size to {new_batch_size} and cleared memory'
                }
            
            return {
                'success': True,
                'action': 'Cleared memory cache'
            }
        
        except Exception as recovery_error:
            return {
                'success': False,
                'action': f'Recovery failed: {str(recovery_error)}'
            }
    
    def _recover_from_data_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recover from data loading errors."""
        try:
            # Try alternative data path
            if context and 'data_path' in context:
                alt_path = self._find_alternative_data_path(context['data_path'])
                if alt_path:
                    context['data_path'] = alt_path
                    return {
                        'success': True,
                        'action': f'Found alternative data path: {alt_path}'
                    }
            
            # Try with different data format
            if context and 'data_format' in context:
                alt_formats = ['numpy', 'pickle', 'json', 'csv']
                for fmt in alt_formats:
                    if fmt != context['data_format']:
                        context['data_format'] = fmt
                        return {
                            'success': True,
                            'action': f'Trying alternative format: {fmt}'
                        }
            
            return {
                'success': False,
                'action': 'No recovery strategy available for data error'
            }
        
        except Exception as recovery_error:
            return {
                'success': False,
                'action': f'Recovery failed: {str(recovery_error)}'
            }
    
    def _recover_from_inference_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recover from model inference errors."""
        try:
            # Try with different precision
            if context and 'dtype' in context:
                if context['dtype'] == torch.float32:
                    context['dtype'] = torch.float16
                    return {
                        'success': True,
                        'action': 'Switched to float16 precision'
                    }
            
            # Try with gradient checkpointing
            if context and 'model' in context:
                model = context['model']
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    return {
                        'success': True,
                        'action': 'Enabled gradient checkpointing'
                    }
            
            return {
                'success': False,
                'action': 'No recovery strategy available for inference error'
            }
        
        except Exception as recovery_error:
            return {
                'success': False,
                'action': f'Recovery failed: {str(recovery_error)}'
            }
    
    def _recover_from_validation_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recover from validation errors."""
        try:
            # Try input sanitization
            if context and 'input_data' in context:
                sanitized_data = self._sanitize_input(context['input_data'])
                context['input_data'] = sanitized_data
                return {
                    'success': True,
                    'action': 'Applied input sanitization'
                }
            
            return {
                'success': False,
                'action': 'No recovery strategy available for validation error'
            }
        
        except Exception as recovery_error:
            return {
                'success': False,
                'action': f'Recovery failed: {str(recovery_error)}'
            }
    
    def _recover_generic(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generic recovery strategy."""
        try:
            # Wait and retry
            time.sleep(self.config.retry_delay)
            return {
                'success': True,
                'action': f'Applied generic recovery (wait {self.config.retry_delay}s)'
            }
        
        except Exception as recovery_error:
            return {
                'success': False,
                'action': f'Generic recovery failed: {str(recovery_error)}'
            }
    
    def _find_alternative_data_path(self, original_path: str) -> Optional[str]:
        """Find alternative data path."""
        path = Path(original_path)
        possible_paths = [
            path.with_suffix('.npy'),
            path.with_suffix('.pkl'),
            path.with_suffix('.json'),
            path.with_suffix('.csv'),
            path.parent / f"{path.stem}_backup{path.suffix}",
            path.parent / f"{path.stem}_alt{path.suffix}"
        ]
        
        for alt_path in possible_paths:
            if alt_path.exists():
                return str(alt_path)
        
        return None
    
    def _sanitize_input(self, data: Any) -> Any:
        """Sanitize input data."""
        if isinstance(data, torch.Tensor):
            # Handle NaN and Inf values
            if torch.isnan(data).any():
                data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure finite values
            if not torch.isfinite(data).all():
                data = torch.clamp(data, min=-1e6, max=1e6)
        
        elif isinstance(data, np.ndarray):
            # Handle NaN and Inf values
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure finite values
            if not np.isfinite(data).all():
                data = np.clip(data, -1e6, 1e6)
        
        return data
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        state = {}
        
        if self.config.monitor_memory:
            state['memory'] = self.memory_monitor.get_memory_info()
        
        if self.config.monitor_gpu and torch.cuda.is_available():
            state['gpu'] = self.memory_monitor.get_gpu_info()
        
        if self.config.monitor_performance:
            state['performance'] = self.performance_monitor.get_performance_info()
        
        return state
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'error_counts': dict(self.error_counts),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': list(self.error_history)[-10:],
            'most_common_errors': sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }


class MemoryMonitor:
    """Monitor memory usage for CPU and GPU."""
    
    def __init__(self, config: ErrorHandlingConfig):
        
    """__init__ function."""
self.config = config
        self.memory_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get CPU memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'threshold_exceeded': memory.percent > (self.config.memory_threshold * 100)
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        try:
            if not torch.cuda.is_available():
                return {'available': False}
            
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                cached = torch.cuda.memory_reserved(i)
                
                gpu_info[f'gpu_{i}'] = {
                    'total': memory,
                    'allocated': allocated,
                    'cached': cached,
                    'free': memory - allocated,
                    'percent_used': (allocated / memory) * 100,
                    'threshold_exceeded': (allocated / memory) > self.config.gpu_memory_threshold
                }
            
            return gpu_info
        
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {'error': str(e)}
    
    def check_memory_thresholds(self) -> Dict[str, bool]:
        """Check if memory thresholds are exceeded."""
        warnings = {}
        
        # Check CPU memory
        if self.config.monitor_memory:
            memory_info = self.get_memory_info()
            if memory_info.get('threshold_exceeded', False):
                warnings['cpu_memory'] = True
        
        # Check GPU memory
        if self.config.monitor_gpu and torch.cuda.is_available():
            gpu_info = self.get_gpu_info()
            for gpu_id, info in gpu_info.items():
                if isinstance(info, dict) and info.get('threshold_exceeded', False):
                    warnings[gpu_id] = True
        
        return warnings


class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self, config: ErrorHandlingConfig):
        
    """__init__ function."""
self.config = config
        self.performance_history = deque(maxlen=1000)
        self.start_time = time.time()
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg()
            
            return {
                'cpu_percent': cpu_percent,
                'load_average': load_avg,
                'uptime': time.time() - self.start_time,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get performance info: {e}")
            return {}


class DataValidator:
    """Validate data for deep learning operations."""
    
    def __init__(self, config: ErrorHandlingConfig):
        
    """__init__ function."""
self.config = config
    
    def validate_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> Tuple[bool, str]:
        """Validate tensor data."""
        try:
            # Check for NaN values
            if torch.isnan(tensor).any():
                return False, f"{name} contains NaN values"
            
            # Check for Inf values
            if torch.isinf(tensor).any():
                return False, f"{name} contains Inf values"
            
            # Check for finite values
            if not torch.isfinite(tensor).all():
                return False, f"{name} contains non-finite values"
            
            # Check shape
            if tensor.dim() == 0:
                return False, f"{name} is a scalar, expected tensor"
            
            # Check for empty tensor
            if tensor.numel() == 0:
                return False, f"{name} is empty"
            
            return True, "Valid tensor"
        
        except Exception as e:
            return False, f"Validation error for {name}: {str(e)}"
    
    def validate_numpy_array(self, array: np.ndarray, name: str = "array") -> Tuple[bool, str]:
        """Validate numpy array data."""
        try:
            # Check for NaN values
            if np.isnan(array).any():
                return False, f"{name} contains NaN values"
            
            # Check for Inf values
            if np.isinf(array).any():
                return False, f"{name} contains Inf values"
            
            # Check for finite values
            if not np.isfinite(array).all():
                return False, f"{name} contains non-finite values"
            
            # Check shape
            if array.ndim == 0:
                return False, f"{name} is a scalar, expected array"
            
            # Check for empty array
            if array.size == 0:
                return False, f"{name} is empty"
            
            return True, "Valid array"
        
        except Exception as e:
            return False, f"Validation error for {name}: {str(e)}"
    
    def validate_model_input(self, input_data: Any, expected_shape: Tuple = None, 
                           expected_dtype: torch.dtype = None) -> Tuple[bool, str]:
        """Validate model input data."""
        try:
            if isinstance(input_data, torch.Tensor):
                is_valid, message = self.validate_tensor(input_data, "input")
                if not is_valid:
                    return False, message
                
                # Check shape if provided
                if expected_shape and input_data.shape != expected_shape:
                    return False, f"Input shape {input_data.shape} does not match expected {expected_shape}"
                
                # Check dtype if provided
                if expected_dtype and input_data.dtype != expected_dtype:
                    return False, f"Input dtype {input_data.dtype} does not match expected {expected_dtype}"
            
            elif isinstance(input_data, np.ndarray):
                is_valid, message = self.validate_numpy_array(input_data, "input")
                if not is_valid:
                    return False, message
            
            else:
                return False, f"Unsupported input type: {type(input_data)}"
            
            return True, "Valid model input"
        
        except Exception as e:
            return False, f"Input validation error: {str(e)}"
    
    def validate_model_output(self, output_data: Any, expected_shape: Tuple = None) -> Tuple[bool, str]:
        """Validate model output data."""
        try:
            if isinstance(output_data, torch.Tensor):
                is_valid, message = self.validate_tensor(output_data, "output")
                if not is_valid:
                    return False, message
                
                # Check shape if provided
                if expected_shape and output_data.shape != expected_shape:
                    return False, f"Output shape {output_data.shape} does not match expected {expected_shape}"
            
            elif isinstance(output_data, np.ndarray):
                is_valid, message = self.validate_numpy_array(output_data, "output")
                if not is_valid:
                    return False, message
            
            else:
                return False, f"Unsupported output type: {type(output_data)}"
            
            return True, "Valid model output"
        
        except Exception as e:
            return False, f"Output validation error: {str(e)}"


class Debugger:
    """Advanced debugging utilities for deep learning."""
    
    def __init__(self, config: ErrorHandlingConfig):
        
    """__init__ function."""
self.config = config
        self.breakpoints = set()
        self.variable_history = {}
        self.call_stack = []
    
    def set_breakpoint(self, function_name: str, condition: Callable = None):
        """Set a breakpoint for a function."""
        self.breakpoints.add((function_name, condition))
    
    def debug_function(self, func: Callable) -> Callable:
        """Decorator to add debugging to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            function_name = func.__name__
            
            # Check if breakpoint is set
            if self._should_break(function_name, args, kwargs):
                self._enter_debug_mode(function_name, args, kwargs)
            
            # Track call stack
            self.call_stack.append({
                'function': function_name,
                'args': str(args),
                'kwargs': str(kwargs),
                'timestamp': datetime.now().isoformat()
            })
            
            try:
                result = func(*args, **kwargs)
                
                # Track variable history
                if self.config.enable_variable_inspection:
                    self._track_variables(function_name, result)
                
                return result
            
            except Exception as e:
                if self.config.enable_breakpoints:
                    self._enter_debug_mode(function_name, args, kwargs, error=e)
                raise
        
        return wrapper
    
    def _should_break(self, function_name: str, args: tuple, kwargs: dict) -> bool:
        """Check if we should break at this function."""
        for bp_func, condition in self.breakpoints:
            if bp_func == function_name:
                if condition is None or condition(*args, **kwargs):
                    return True
        return False
    
    def _enter_debug_mode(self, function_name: str, args: tuple, kwargs: dict, error: Exception = None):
        """Enter debug mode with interactive debugging."""
        print(f"\n🔍 DEBUG MODE: {function_name}")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        
        if error:
            print(f"Error: {error}")
            print(f"Traceback: {traceback.format_exc()}")
        
        if self.config.enable_breakpoints:
            try:
                ipdb.set_trace()
            except ImportError:
                pdb.set_trace()
    
    def _track_variables(self, function_name: str, result: Any):
        """Track variable history."""
        if function_name not in self.variable_history:
            self.variable_history[function_name] = []
        
        self.variable_history[function_name].append({
            'result': str(result),
            'result_type': type(result).__name__,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information."""
        return {
            'breakpoints': list(self.breakpoints),
            'call_stack': self.call_stack[-10:],  # Last 10 calls
            'variable_history': {k: v[-5:] for k, v in self.variable_history.items()},  # Last 5 entries per function
            'total_calls': len(self.call_stack)
        }


class SafeDataLoader:
    """Safe data loader with comprehensive error handling."""
    
    def __init__(self, config: ErrorHandlingConfig):
        
    """__init__ function."""
self.config = config
        self.error_handler = ErrorHandler(config)
        self.validator = DataValidator(config)
        self.memory_monitor = MemoryMonitor(config)
    
    def load_data(self, data_path: str, data_format: str = "auto") -> Any:
        """Load data with comprehensive error handling."""
        context = {'data_path': data_path, 'data_format': data_format}
        
        for attempt in range(self.config.max_retries):
            try:
                # Check memory before loading
                memory_warnings = self.memory_monitor.check_memory_thresholds()
                if memory_warnings:
                    logger.warning(f"Memory warnings before loading: {memory_warnings}")
                
                # Determine format if auto
                if data_format == "auto":
                    data_format = self._detect_format(data_path)
                
                # Load data based on format
                if data_format == "numpy":
                    data = self._load_numpy(data_path)
                elif data_format == "pickle":
                    data = self._load_pickle(data_path)
                elif data_format == "json":
                    data = self._load_json(data_path)
                elif data_format == "csv":
                    data = self._load_csv(data_path)
                elif data_format == "torch":
                    data = self._load_torch(data_path)
                else:
                    raise ValueError(f"Unsupported data format: {data_format}")
                
                # Validate loaded data
                if self.config.validate_inputs:
                    if isinstance(data, torch.Tensor):
                        is_valid, message = self.validator.validate_tensor(data, "loaded_data")
                    elif isinstance(data, np.ndarray):
                        is_valid, message = self.validator.validate_numpy_array(data, "loaded_data")
                    else:
                        is_valid, message = True, "Data validation skipped for non-tensor/array data"
                    
                    if not is_valid:
                        raise ValidationError(message, "data_validation", context)
                
                logger.info(f"Successfully loaded data from {data_path}")
                return data
            
            except Exception as e:
                error_info = self.error_handler.handle_error(e, context)
                
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to load data after {self.config.max_retries} attempts")
                    raise DataLoadingError(f"Failed to load data: {str(e)}", data_path, context)
    
    def _detect_format(self, data_path: str) -> str:
        """Detect data format from file extension."""
        path = Path(data_path)
        extension = path.suffix.lower()
        
        format_mapping = {
            '.npy': 'numpy',
            '.npz': 'numpy',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.json': 'json',
            '.csv': 'csv',
            '.pt': 'torch',
            '.pth': 'torch'
        }
        
        return format_mapping.get(extension, 'pickle')  # Default to pickle
    
    def _load_numpy(self, data_path: str) -> np.ndarray:
        """Load numpy array with error handling."""
        try:
            return np.load(data_path)
        except Exception as e:
            raise DataLoadingError(f"Failed to load numpy array: {str(e)}", data_path)
    
    def _load_pickle(self, data_path: str) -> Any:
        """Load pickle file with error handling."""
        try:
            with open(data_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return pickle.load(f)
        except Exception as e:
            raise DataLoadingError(f"Failed to load pickle file: {str(e)}", data_path)
    
    def _load_json(self, data_path: str) -> Any:
        """Load JSON file with error handling."""
        try:
            with open(data_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        except Exception as e:
            raise DataLoadingError(f"Failed to load JSON file: {str(e)}", data_path)
    
    def _load_csv(self, data_path: str) -> np.ndarray:
        """Load CSV file with error handling."""
        try:
            df = pd.read_csv(data_path)
            return df.values
        except Exception as e:
            raise DataLoadingError(f"Failed to load CSV file: {str(e)}", data_path)
    
    def _load_torch(self, data_path: str) -> torch.Tensor:
        """Load torch tensor with error handling."""
        try:
            return torch.load(data_path)
        except Exception as e:
            raise DataLoadingError(f"Failed to load torch tensor: {str(e)}", data_path)


class SafeModelInference:
    """Safe model inference with comprehensive error handling."""
    
    def __init__(self, config: ErrorHandlingConfig):
        
    """__init__ function."""
self.config = config
        self.error_handler = ErrorHandler(config)
        self.validator = DataValidator(config)
        self.memory_monitor = MemoryMonitor(config)
        self.debugger = Debugger(config)
    
    def infer(self, model: nn.Module, input_data: Any, **kwargs) -> Any:
        """Perform model inference with comprehensive error handling."""
        context = {
            'model_name': model.__class__.__name__,
            'input_shape': getattr(input_data, 'shape', None),
            'kwargs': kwargs
        }
        
        for attempt in range(self.config.max_retries):
            try:
                # Check memory before inference
                memory_warnings = self.memory_monitor.check_memory_thresholds()
                if memory_warnings:
                    logger.warning(f"Memory warnings before inference: {memory_warnings}")
                
                # Validate input
                if self.config.validate_inputs:
                    is_valid, message = self.validator.validate_model_input(input_data)
                    if not is_valid:
                        raise ValidationError(message, "input_validation", context)
                
                # Prepare model
                model.eval()
                
                # Move to device if needed
                if isinstance(input_data, torch.Tensor):
                    device = input_data.device
                    model = model.to(device)
                
                # Perform inference
                with torch.no_grad():
                    output = model(input_data, **kwargs)
                
                # Validate output
                if self.config.validate_outputs:
                    is_valid, message = self.validator.validate_model_output(output)
                    if not is_valid:
                        raise ValidationError(message, "output_validation", context)
                
                logger.info(f"Successfully performed inference with {model.__class__.__name__}")
                return output
            
            except Exception as e:
                error_info = self.error_handler.handle_error(e, context)
                
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Inference attempt {attempt + 1} failed, retrying...")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to perform inference after {self.config.max_retries} attempts")
                    raise ModelInferenceError(f"Failed to perform inference: {str(e)}", 
                                            model.__class__.__name__, 
                                            getattr(input_data, 'shape', None), 
                                            context)
    
    def batch_infer(self, model: nn.Module, data_loader: Any, batch_size: int = 32) -> List[Any]:
        """Perform batch inference with error handling."""
        results = []
        context = {'model_name': model.__class__.__name__, 'batch_size': batch_size}
        
        try:
            for batch_idx, batch_data in enumerate(data_loader):
                try:
                    # Check memory before each batch
                    memory_warnings = self.memory_monitor.check_memory_thresholds()
                    if memory_warnings:
                        logger.warning(f"Memory warnings at batch {batch_idx}: {memory_warnings}")
                    
                    # Perform inference on batch
                    batch_output = self.infer(model, batch_data)
                    results.append(batch_output)
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        logger.info(f"Processed batch {batch_idx}")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    if not self.config.enable_graceful_degradation:
                        raise
                    # Continue with next batch if graceful degradation is enabled
            
            return results
        
        except Exception as e:
            error_info = self.error_handler.handle_error(e, context)
            raise ModelInferenceError(f"Failed to perform batch inference: {str(e)}", 
                                    model.__class__.__name__, 
                                    None, 
                                    context)


# Utility functions for error handling
def safe_execute(func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """Safely execute a function with error handling."""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


def retry_on_error(func: Callable, max_retries: int = 3, delay: float = 1.0) -> Callable:
    """Decorator to retry function on error."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                time.sleep(delay)
    return wrapper


@contextmanager
def error_context(context_name: str, **context_vars):
    """Context manager for error handling with context."""
    try:
        yield
    except Exception as e:
        context = {'context_name': context_name, **context_vars}
        logger.error(f"Error in {context_name}: {e}")
        logger.error(f"Context: {context}")
        raise


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = ErrorHandlingConfig(
        enable_error_handling=True,
        enable_debugging=True,
        enable_monitoring=True,
        enable_logging=True,
        max_retries=3,
        retry_delay=1.0
    )
    
    # Create error handling components
    error_handler = ErrorHandler(config)
    data_loader = SafeDataLoader(config)
    model_inference = SafeModelInference(config)
    validator = DataValidator(config)
    debugger = Debugger(config)
    
    # Example: Safe data loading
    try:
        data = data_loader.load_data("example_data.npy", "numpy")
        print("Data loaded successfully")
    except DataLoadingError as e:
        print(f"Data loading failed: {e}")
    
    # Example: Safe model inference
    try:
        # Create dummy model and data
        model = nn.Linear(10, 1)
        input_data = torch.randn(5, 10)
        
        output = model_inference.infer(model, input_data)
        print("Inference completed successfully")
    except ModelInferenceError as e:
        print(f"Model inference failed: {e}")
    
    # Example: Error statistics
    stats = error_handler.get_error_statistics()
    print(f"Error statistics: {stats}") 