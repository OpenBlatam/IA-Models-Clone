#!/usr/bin/env python3
"""
Advanced LLM-Based SEO Engine
Production-ready SEO optimization using transformers, PyTorch, and advanced libraries
Enhanced with custom architectures, mixed precision, and advanced optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.autocast
import unicodedata
import string
import sys
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, TrainingArguments, Trainer, DataCollatorWithPadding,
    PreTrainedModel, PreTrainedTokenizer, AutoConfig
)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, TemplateProcessing
from diffusers import (
    DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
    StableDiffusionUpscalePipeline, StableDiffusionDepth2ImgPipeline,
    DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler,
    UNet2DConditionModel, AutoencoderKL, TextEncoder,
    DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler, HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
    LMSDiscreteScheduler, PNDMScheduler, UniPCMultistepScheduler,
    VQDiffusionScheduler, ScoreSdeVpScheduler, ScoreSdeVeScheduler,
    DDPMWuerstchenScheduler, WuerstchenCombinedScheduler,
    IPNDMScheduler, KarrasVeScheduler, DPMSolverSDEScheduler,
    ControlNetModel, StableDiffusionControlNetPipeline,
    KandinskyPipeline, KandinskyV22Pipeline, KandinskyV22Img2ImgPipeline,
    IFPipeline, IFSuperResolutionPipeline, IFImg2ImgSuperResolutionPipeline,
    DeepFloydIFPipeline, DeepFloydIFSuperResolutionPipeline,
    WuerstchenPipeline, WuerstchenPriorPipeline, WuerstchenDecoderPipeline
)
import gradio as gr
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import logging
import time
import json
import re
import math
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import gc
import psutil
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from transformers import (
    AutoTokenizer, AutoModel, pipeline, TextEncoder,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, 
    StratifiedGroupKFold, LeaveOneOut, LeavePOut,
    RepeatedStratifiedKFold, RepeatedKFold
)
from sklearn.preprocessing import LabelEncoder
import cProfile
import pstats
import io
import line_profiler
import memory_profiler
import tracemalloc
from contextlib import contextmanager
import threading
import queue
from collections import defaultdict, deque

warnings.filterwarnings("ignore")

# ============================================================================
# ERROR HANDLING AND INPUT VALIDATION SYSTEM
# ============================================================================

class GradioErrorHandler:
    """Comprehensive error handling for Gradio applications."""
    
    def __init__(self):
        self.error_log = []
        self.max_error_log_size = 100
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors and return user-friendly error messages."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_friendly_message": self._get_user_friendly_message(error, context)
        }
        
        # Log error
        self._log_error(error_info)
        
        # Return user-friendly response
        return {
            "error": True,
            "message": error_info["user_friendly_message"],
            "details": error_info["error_message"] if self._is_development_mode() else "Contact support for details",
            "error_code": self._get_error_code(error),
            "suggestions": self._get_error_suggestions(error, context)
        }
    
    def _get_user_friendly_message(self, error: Exception, context: str) -> str:
        """Convert technical errors to user-friendly messages."""
        error_type = type(error).__name__
        
        if "CUDA" in str(error) or "GPU" in str(error):
            return "GPU memory or compatibility issue detected. Try reducing batch size or using CPU mode."
        elif "memory" in str(error).lower():
            return "Memory limit exceeded. Try processing smaller batches or clearing cache."
        elif "timeout" in str(error).lower():
            return "Operation timed out. Try with smaller inputs or check your connection."
        elif "validation" in str(error).lower() or "ValueError" in error_type:
            return "Invalid input provided. Please check your input format and try again."
        elif "permission" in str(error).lower():
            return "Permission denied. Check file/directory access rights."
        elif "not found" in str(error).lower():
            return "Resource not found. Please verify the path or resource name."
        elif "connection" in str(error).lower():
            return "Connection failed. Check your internet connection and try again."
        else:
            return f"An unexpected error occurred: {str(error)[:100]}"
    
    def _get_error_code(self, error: Exception) -> str:
        """Generate error codes for categorization."""
        error_type = type(error).__name__
        
        if "CUDA" in str(error) or "GPU" in str(error):
            return "GPU_001"
        elif "memory" in str(error).lower():
            return "MEM_001"
        elif "timeout" in str(error).lower():
            return "TIM_001"
        elif "validation" in str(error).lower() or "ValueError" in error_type:
            return "VAL_001"
        elif "permission" in str(error).lower():
            return "PERM_001"
        elif "not found" in str(error).lower():
            return "NF_001"
        elif "connection" in str(error).lower():
            return "CONN_001"
        else:
            return "UNK_001"
    
    def _get_error_suggestions(self, error: Exception, context: str) -> List[str]:
        """Provide helpful suggestions based on error type."""
        error_type = type(error).__name__
        suggestions = []
        
        if "CUDA" in str(error) or "GPU" in str(error):
            suggestions.extend([
                "Reduce batch size",
                "Use CPU mode if available",
                "Clear GPU cache",
                "Check GPU driver compatibility"
            ])
        elif "memory" in str(error).lower():
            suggestions.extend([
                "Process smaller batches",
                "Clear application cache",
                "Close other applications",
                "Use memory-efficient settings"
            ])
        elif "validation" in str(error).lower() or "ValueError" in error_type:
            suggestions.extend([
                "Check input format requirements",
                "Verify data types",
                "Ensure required fields are filled",
                "Review input validation rules"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error information."""
        self.error_log.append(error_info)
        if len(self.error_log) > self.max_error_log_size:
            self.error_log.pop(0)
        
        # Log to console in development mode
        if self._is_development_mode():
            print(f"ERROR [{error_info['timestamp']}] {error_info['context']}: {error_info['error_message']}")
    
    def _is_development_mode(self) -> bool:
        """Check if running in development mode."""
        return os.getenv("GRADIO_DEBUG", "false").lower() == "true"
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_log:
            return {"total_errors": 0, "recent_errors": []}
        
        error_counts = {}
        for error in self.error_log:
            error_type = error["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_counts,
            "recent_errors": self.error_log[-10:],  # Last 10 errors
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }

class InputValidator:
    """Comprehensive input validation for Gradio applications."""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for different input types."""
        return {
            "text": {
                "min_length": 1,
                "max_length": 10000,
                "allowed_chars": None,  # All characters allowed
                "required": True
            },
            "url": {
                "pattern": r'^https?://[^\s/$.?#].[^\s]*$',
                "max_length": 2048,
                "required": True
            },
            "email": {
                "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                "max_length": 254,
                "required": True
            },
            "number": {
                "min_value": None,
                "max_value": None,
                "integer_only": False,
                "required": True
            },
            "file_path": {
                "allowed_extensions": [".txt", ".md", ".html", ".py", ".js", ".ts", ".tsx"],
                "max_size_mb": 10,
                "required": True
            },
            "json": {
                "max_depth": 10,
                "max_items": 1000,
                "required": True
            }
        }
    
    def validate_text(self, text: str, field_name: str = "text") -> Tuple[bool, Optional[str]]:
        """Validate text input."""
        if not text:
            return False, f"{field_name} cannot be empty"
        
        text = text.strip()
        
        if len(text) < self.validation_rules["text"]["min_length"]:
            return False, f"{field_name} must be at least {self.validation_rules['text']['min_length']} character(s)"
        
        if len(text) > self.validation_rules["text"]["max_length"]:
            return False, f"{field_name} must be no more than {self.validation_rules['text']['max_length']} characters"
        
        return True, None
    
    def validate_url(self, url: str, field_name: str = "URL") -> Tuple[bool, Optional[str]]:
        """Validate URL input."""
        if not url:
            return False, f"{field_name} cannot be empty"
        
        url = url.strip()
        
        if len(url) > self.validation_rules["url"]["max_length"]:
            return False, f"{field_name} must be no more than {self.validation_rules['url']['max_length']} characters"
        
        if not re.match(self.validation_rules["url"]["pattern"], url):
            return False, f"{field_name} must be a valid HTTP/HTTPS URL"
        
        return True, None
    
    def validate_email(self, email: str, field_name: str = "email") -> Tuple[bool, Optional[str]]:
        """Validate email input."""
        if not email:
            return False, f"{field_name} cannot be empty"
        
        email = email.strip()
        
        if len(email) > self.validation_rules["email"]["max_length"]:
            return False, f"{field_name} must be no more than {self.validation_rules['email']['max_length']} characters"
        
        if not re.match(self.validation_rules["email"]["pattern"], email):
            return False, f"{field_name} must be a valid email address"
        
        return True, None
    
    def validate_number(self, value: Union[int, float], field_name: str = "number", 
                       min_val: Optional[Union[int, float]] = None, 
                       max_val: Optional[Union[int, float]] = None,
                       integer_only: bool = False) -> Tuple[bool, Optional[str]]:
        """Validate numeric input."""
        if value is None:
            return False, f"{field_name} cannot be empty"
        
        if integer_only and not isinstance(value, int):
            return False, f"{field_name} must be an integer"
        
        if min_val is not None and value < min_val:
            return False, f"{field_name} must be at least {min_val}"
        
        if max_val is not None and value > max_val:
            return False, f"{field_name} must be no more than {max_val}"
        
        return True, None
    
    def validate_file_path(self, file_path: str, field_name: str = "file path") -> Tuple[bool, Optional[str]]:
        """Validate file path input."""
        if not file_path:
            return False, f"{field_name} cannot be empty"
        
        file_path = file_path.strip()
        
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.validation_rules["file_path"]["allowed_extensions"]:
            allowed = ", ".join(self.validation_rules["file_path"]["allowed_extensions"])
            return False, f"{field_name} must have one of these extensions: {allowed}"
        
        # Check if file exists and size
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.validation_rules["file_path"]["max_size_mb"]:
                return False, f"{field_name} must be no larger than {self.validation_rules['file_path']['max_size_mb']} MB"
        
        return True, None
    
    def validate_json(self, json_str: str, field_name: str = "JSON") -> Tuple[bool, Optional[str]]:
        """Validate JSON input."""
        if not json_str:
            return False, f"{field_name} cannot be empty"
        
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, f"{field_name} must be valid JSON: {str(e)}"
        
        # Check depth and item count
        depth = self._get_json_depth(parsed)
        if depth > self.validation_rules["json"]["max_depth"]:
            return False, f"{field_name} must not exceed {self.validation_rules['json']['max_depth']} levels of nesting"
        
        item_count = self._count_json_items(parsed)
        if item_count > self.validation_rules["json"]["max_items"]:
            return False, f"{field_name} must not exceed {self.validation_rules['json']['max_items']} items"
        
        return True, None
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of a JSON object."""
        if isinstance(obj, dict):
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values()) if obj else current_depth
        elif isinstance(obj, list):
            return max(self._get_json_depth(item, current_depth + 1) for item in obj) if obj else current_depth
        else:
            return current_depth
    
    def _count_json_items(self, obj: Any) -> int:
        """Count the total number of items in a JSON object."""
        if isinstance(obj, dict):
            return sum(1 + self._count_json_items(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(1 + self._count_json_items(item) for item in obj)
        else:
            return 0
    
    def validate_seo_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate SEO-specific inputs."""
        errors = []
        
        # Validate text inputs
        text_fields = ["content", "title", "description", "keywords"]
        for field in text_fields:
            if field in inputs and inputs[field]:
                is_valid, error_msg = self.validate_text(inputs[field], field)
                if not is_valid:
                    errors.append(error_msg)
        
        # Validate numeric inputs
        numeric_fields = {
            "max_length": (1, 10000),
            "batch_size": (1, 128),
            "learning_rate": (1e-6, 1e-2),
            "num_epochs": (1, 1000)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in inputs and inputs[field] is not None:
                is_valid, error_msg = self.validate_number(
                    inputs[field], field, min_val, max_val
                )
                if not is_valid:
                    errors.append(error_msg)
        
        # Validate JSON inputs
        json_fields = ["metadata", "config", "parameters"]
        for field in json_fields:
            if field in inputs and inputs[field]:
                if isinstance(inputs[field], str):
                    is_valid, error_msg = self.validate_json(inputs[field], field)
                    if not is_valid:
                        errors.append(error_msg)
        
        return len(errors) == 0, errors

class GradioErrorBoundary:
    """Error boundary wrapper for Gradio functions."""
    
    def __init__(self, error_handler: GradioErrorHandler):
        self.error_handler = error_handler
    
    def __call__(self, func):
        """Decorator to wrap Gradio functions with error handling."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = f"Function: {func.__name__}, Args: {args[:3]}{'...' if len(args) > 3 else ''}"
                return self.error_handler.handle_error(e, context)
        
        return wrapper

# ============================================================================
# END ERROR HANDLING AND INPUT VALIDATION SYSTEM
# ============================================================================

# ============================================================================
# CODE PROFILING AND PERFORMANCE OPTIMIZATION SYSTEM
# ============================================================================

class CodeProfiler:
    """Comprehensive code profiling system for identifying and optimizing bottlenecks."""
    
    def __init__(self, config: 'SEOConfig'):
        self.config = config
        self.profiling_enabled = config.enable_code_profiling
        self.profiling_data = defaultdict(dict)
        self.profiling_stats = defaultdict(dict)
        self.memory_tracker = tracemalloc
        self.line_profiler = line_profiler.LineProfiler()
        self.profiling_queue = queue.Queue()
        self.profiling_thread = None
        self.profiling_lock = threading.Lock()
        self.start_time = None
        self.end_time = None
        
        # Initialize profiling if enabled
        if self.profiling_enabled:
            self._initialize_profiling()
    
    def _initialize_profiling(self):
        """Initialize profiling system components."""
        try:
            # Start memory tracking
            if self.config.profile_memory_usage:
                self.memory_tracker.start()
            
            # Start profiling thread
            self.profiling_thread = threading.Thread(target=self._profiling_worker, daemon=True)
            self.profiling_thread.start()
            
            self.start_time = time.time()
            print(f"Code profiling initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"Warning: Failed to initialize profiling: {e}")
            self.profiling_enabled = False
    
    def _profiling_worker(self):
        """Background worker for processing profiling data."""
        while True:
            try:
                profiling_task = self.profiling_queue.get(timeout=1)
                if profiling_task is None:  # Shutdown signal
                    break
                
                self._process_profiling_task(profiling_task)
                self.profiling_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Profiling worker error: {e}")
    
    def _process_profiling_task(self, task: Dict[str, Any]):
        """Process a profiling task."""
        try:
            task_type = task.get('type')
            task_data = task.get('data', {})
            
            if task_type == 'memory_snapshot':
                self._capture_memory_snapshot(task_data)
            elif task_type == 'performance_metrics':
                self._capture_performance_metrics(task_data)
            elif task_type == 'gpu_metrics':
                self._capture_gpu_metrics(task_data)
            elif task_type == 'cpu_metrics':
                self._capture_cpu_metrics(task_data)
            elif task_type == 'io_metrics':
                self._capture_io_metrics(task_data)
            
        except Exception as e:
            print(f"Error processing profiling task: {e}")
    
    def _capture_memory_snapshot(self, data: Dict[str, Any]):
        """Capture memory snapshot."""
        try:
            if self.config.profile_memory_usage:
                current, peak = self.memory_tracker.get_traced_memory()
                snapshot = {
                    'timestamp': time.time(),
                    'current_memory': current,
                    'peak_memory': peak,
                    'context': data.get('context', 'unknown')
                }
                self.profiling_stats['memory_snapshots'].append(snapshot)
        except Exception as e:
            print(f"Error capturing memory snapshot: {e}")
    
    def _capture_performance_metrics(self, data: Dict[str, Any]):
        """Capture performance metrics."""
        try:
            metrics = {
                'timestamp': time.time(),
                'operation': data.get('operation', 'unknown'),
                'duration': data.get('duration', 0),
                'memory_delta': data.get('memory_delta', 0),
                'gpu_memory_delta': data.get('gpu_memory_delta', 0),
                'context': data.get('context', 'unknown')
            }
            self.profiling_stats['performance_metrics'].append(metrics)
        except Exception as e:
            print(f"Error capturing performance metrics: {e}")
    
    def _capture_gpu_metrics(self, data: Dict[str, Any]):
        """Capture GPU metrics."""
        try:
            if torch.cuda.is_available() and self.config.profile_gpu_utilization:
                gpu_metrics = {
                    'timestamp': time.time(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated(),
                    'gpu_memory_reserved': torch.cuda.memory_reserved(),
                    'gpu_memory_cached': torch.cuda.memory_reserved() - torch.cuda.memory_allocated(),
                    'context': data.get('context', 'unknown')
                }
                self.profiling_stats['gpu_metrics'].append(gpu_metrics)
        except Exception as e:
            print(f"Error capturing GPU metrics: {e}")
    
    def _capture_cpu_metrics(self, data: Dict[str, Any]):
        """Capture CPU metrics."""
        try:
            if self.config.profile_cpu_utilization:
                cpu_metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'context': data.get('context', 'unknown')
                }
                self.profiling_stats['cpu_metrics'].append(cpu_metrics)
        except Exception as e:
            print(f"Error capturing CPU metrics: {e}")
    
    def _capture_io_metrics(self, data: Dict[str, Any]):
        """Capture I/O metrics."""
        try:
            if self.config.profile_io_operations:
                io_metrics = {
                    'timestamp': time.time(),
                    'operation': data.get('operation', 'unknown'),
                    'bytes_read': data.get('bytes_read', 0),
                    'bytes_written': data.get('bytes_written', 0),
                    'duration': data.get('duration', 0),
                    'context': data.get('context', 'unknown')
                }
                self.profiling_stats['io_metrics'].append(io_metrics)
        except Exception as e:
            print(f"Error capturing I/O metrics: {e}")
    
    @contextmanager
    def profile_operation(self, operation_name: str, operation_type: str = "general"):
        """Context manager for profiling operations."""
        if not self.profiling_enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            self._record_profiling_data(operation_name, operation_type, {
                'duration': duration,
                'memory_delta': memory_delta,
                'gpu_memory_delta': gpu_memory_delta,
                'start_time': start_time,
                'end_time': end_time
            })
    
    def profile_function(self, func):
        """Decorator for profiling functions."""
        if not self.profiling_enabled:
            return func
        
        def wrapper(*args, **kwargs):
            with self.profile_operation(func.__name__, "function"):
                return func(*args, **kwargs)
        
        return wrapper
    
    def profile_data_loading(self, func):
        """Decorator specifically for data loading operations."""
        if not self.profiling_enabled or not self.config.profile_data_loading:
            return func
        
        def wrapper(*args, **kwargs):
            with self.profile_operation(func.__name__, "data_loading"):
                return func(*args, **kwargs)
        
        return wrapper
    
    def profile_preprocessing(self, func):
        """Decorator specifically for preprocessing operations."""
        if not self.profiling_enabled or not self.config.profile_preprocessing:
            return func
        
        def wrapper(*args, **kwargs):
            with self.profile_operation(func.__name__, "preprocessing"):
                return func(*args, **kwargs)
        
        return wrapper
    
    def profile_model_inference(self, func):
        """Decorator specifically for model inference operations."""
        if not self.profiling_enabled or not self.config.profile_model_inference:
            return func
        
        def wrapper(*args, **kwargs):
            with self.profile_operation(func.__name__, "model_inference"):
                return func(*args, **kwargs)
        
        return wrapper
    
    def profile_training_loop(self, func):
        """Decorator specifically for training loop operations."""
        if not self.profiling_enabled or not self.config.profile_training_loop:
            return func
        
        def wrapper(*args, **kwargs):
            with self.profile_operation(func.__name__, "training_loop"):
                return func(*args, **kwargs)
        
        return wrapper
    
    def _record_profiling_data(self, operation_name: str, operation_type: str, metrics: Dict[str, Any]):
        """Record profiling data for an operation."""
        with self.profiling_lock:
            if operation_name not in self.profiling_data:
                self.profiling_data[operation_name] = {
                    'type': operation_type,
                    'calls': 0,
                    'total_duration': 0.0,
                    'total_memory_delta': 0.0,
                    'total_gpu_memory_delta': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0,
                    'min_memory_delta': float('inf'),
                    'max_memory_delta': 0.0,
                    'min_gpu_memory_delta': float('inf'),
                    'max_gpu_memory_delta': 0.0,
                    'timestamps': []
                }
            
            data = self.profiling_data[operation_name]
            data['calls'] += 1
            data['total_duration'] += metrics['duration']
            data['total_memory_delta'] += metrics['memory_delta']
            data['total_gpu_memory_delta'] += metrics['gpu_memory_delta']
            data['min_duration'] = min(data['min_duration'], metrics['duration'])
            data['max_duration'] = max(data['max_duration'], metrics['duration'])
            data['min_memory_delta'] = min(data['min_memory_delta'], metrics['memory_delta'])
            data['max_memory_delta'] = max(data['max_memory_delta'], metrics['memory_delta'])
            data['min_gpu_memory_delta'] = min(data['min_gpu_memory_delta'], metrics['gpu_memory_delta'])
            data['max_gpu_memory_delta'] = max(data['max_gpu_memory_delta'], metrics['gpu_memory_delta'])
            data['timestamps'].append(metrics['start_time'])
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def _get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
            return 0
        except:
            return 0
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get comprehensive profiling summary."""
        if not self.profiling_enabled:
            return {"profiling_enabled": False}
        
        with self.profiling_lock:
            summary = {
                "profiling_enabled": True,
                "total_operations": len(self.profiling_data),
                "total_calls": sum(data['calls'] for data in self.profiling_data.values()),
                "total_duration": sum(data['total_duration'] for data in self.profiling_data.values()),
                "operations": {}
            }
            
            for op_name, op_data in self.profiling_data.items():
                calls = op_data['calls']
                if calls > 0:
                    summary["operations"][op_name] = {
                        "type": op_data['type'],
                        "calls": calls,
                        "total_duration": op_data['total_duration'],
                        "avg_duration": op_data['total_duration'] / calls,
                        "min_duration": op_data['min_duration'],
                        "max_duration": op_data['max_duration'],
                        "total_memory_delta": op_data['total_memory_delta'],
                        "avg_memory_delta": op_data['total_memory_delta'] / calls,
                        "total_gpu_memory_delta": op_data['total_gpu_memory_delta'],
                        "avg_gpu_memory_delta": op_data['total_gpu_memory_delta'] / calls,
                        "memory_efficiency": op_data['total_memory_delta'] / op_data['total_duration'] if op_data['total_duration'] > 0 else 0,
                        "gpu_memory_efficiency": op_data['total_gpu_memory_delta'] / op_data['total_duration'] if op_data['total_duration'] > 0 else 0
                    }
            
            # Sort operations by total duration (descending)
            summary["operations"] = dict(sorted(
                summary["operations"].items(),
                key=lambda x: x[1]['total_duration'],
                reverse=True
            ))
            
            return summary
    
    def get_bottlenecks(self, threshold_duration: float = 1.0) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks above threshold."""
        summary = self.get_profiling_summary()
        if not summary.get("profiling_enabled", False):
            return []
        
        bottlenecks = []
        for op_name, op_data in summary["operations"].items():
            if op_data['avg_duration'] > threshold_duration:
                bottlenecks.append({
                    'operation': op_name,
                    'type': op_data['type'],
                    'avg_duration': op_data['avg_duration'],
                    'total_calls': op_data['calls'],
                    'total_duration': op_data['total_duration'],
                    'memory_efficiency': op_data['memory_efficiency'],
                    'gpu_memory_efficiency': op_data['gpu_memory_efficiency'],
                    'optimization_priority': 'high' if op_data['avg_duration'] > threshold_duration * 2 else 'medium'
                })
        
        # Sort by optimization priority and average duration
        bottlenecks.sort(key=lambda x: (x['optimization_priority'] == 'high', x['avg_duration']), reverse=True)
        return bottlenecks
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Get detailed memory usage analysis."""
        if not self.profiling_enabled or not self.config.profile_memory_usage:
            return {"memory_profiling_enabled": False}
        
        try:
            current, peak = self.memory_tracker.get_traced_memory()
            
            return {
                "memory_profiling_enabled": True,
                "current_memory_mb": current / 1024 / 1024,
                "peak_memory_mb": peak / 1024 / 1024,
                "memory_usage_by_operation": {
                    op_name: {
                        "avg_memory_delta_mb": op_data['avg_memory_delta'] / 1024 / 1024,
                        "total_memory_delta_mb": op_data['total_memory_delta'] / 1024 / 1024,
                        "memory_efficiency": op_data['memory_efficiency']
                    }
                    for op_name, op_data in self.profiling_data.items()
                    if op_data['calls'] > 0
                }
            }
        except Exception as e:
            return {"memory_profiling_enabled": False, "error": str(e)}
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations based on profiling data."""
        bottlenecks = self.get_bottlenecks()
        recommendations = []
        
        for bottleneck in bottlenecks:
            op_name = bottleneck['operation']
            op_type = bottleneck['type']
            avg_duration = bottleneck['avg_duration']
            
            if op_type == "data_loading":
                if avg_duration > 5.0:
                    recommendations.append(f"🚨 CRITICAL: {op_name} is extremely slow ({avg_duration:.2f}s). Consider implementing async loading, caching, or batch processing.")
                elif avg_duration > 1.0:
                    recommendations.append(f"⚠️  WARNING: {op_name} is slow ({avg_duration:.2f}s). Consider optimizing data loading with prefetching or parallel processing.")
                else:
                    recommendations.append(f"✅ {op_name} performance is acceptable ({avg_duration:.2f}s).")
            
            elif op_type == "preprocessing":
                if avg_duration > 2.0:
                    recommendations.append(f"⚠️  WARNING: {op_name} preprocessing is slow ({avg_duration:.2f}s). Consider vectorization or parallel processing.")
                else:
                    recommendations.append(f"✅ {op_name} preprocessing performance is good ({avg_duration:.2f}s).")
            
            elif op_type == "model_inference":
                if avg_duration > 10.0:
                    recommendations.append(f"🚨 CRITICAL: {op_name} inference is extremely slow ({avg_duration:.2f}s). Consider model optimization, quantization, or batch processing.")
                elif avg_duration > 2.0:
                    recommendations.append(f"⚠️  WARNING: {op_name} inference is slow ({avg_duration:.2f}s). Consider mixed precision or model compilation.")
                else:
                    recommendations.append(f"✅ {op_name} inference performance is good ({avg_duration:.2f}s).")
            
            elif op_type == "training_loop":
                if avg_duration > 30.0:
                    recommendations.append(f"🚨 CRITICAL: {op_name} training iteration is extremely slow ({avg_duration:.2f}s). Consider gradient accumulation, mixed precision, or multi-GPU training.")
                elif avg_duration > 10.0:
                    recommendations.append(f"⚠️  WARNING: {op_name} training iteration is slow ({avg_duration:.2f}s). Consider batch size optimization or learning rate scheduling.")
                else:
                    recommendations.append(f"✅ {op_name} training performance is good ({avg_duration:.2f}s).")
        
        return recommendations
    
    def export_profiling_data(self, filepath: str = None) -> str:
        """Export profiling data to JSON file."""
        if not self.profiling_enabled:
            return "Profiling not enabled"
        
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"profiling_data_{timestamp}.json"
        
        try:
            summary = self.get_profiling_summary()
            memory_analysis = self.get_memory_analysis()
            bottlenecks = self.get_bottlenecks()
            recommendations = self.get_performance_recommendations()
            
            export_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "profiling_summary": summary,
                "memory_analysis": memory_analysis,
                "bottlenecks": bottlenecks,
                "recommendations": recommendations,
                "raw_profiling_data": dict(self.profiling_data)
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return f"Profiling data exported to {filepath}"
            
        except Exception as e:
            return f"Failed to export profiling data: {e}"
    
    def cleanup(self):
        """Cleanup profiling resources."""
        if not self.profiling_enabled:
            return
        
        try:
            # Stop memory tracking
            if self.config.profile_memory_usage:
                self.memory_tracker.stop()
            
            # Stop profiling thread
            if self.profiling_thread and self.profiling_thread.is_alive():
                self.profiling_queue.put(None)  # Shutdown signal
                self.profiling_thread.join(timeout=5)
            
            self.end_time = time.time()
            
            # Generate final report
            if self.start_time and self.end_time:
                total_duration = self.end_time - self.start_time
                print(f"Profiling completed. Total duration: {total_duration:.2f}s")
                
                summary = self.get_profiling_summary()
                if summary.get("profiling_enabled", False):
                    print(f"Total operations profiled: {summary['total_operations']}")
                    print(f"Total calls: {summary['total_calls']}")
                    print(f"Total duration: {summary['total_duration']:.2f}s")
                    
                    bottlenecks = self.get_bottlenecks()
                    if bottlenecks:
                        print(f"\nTop bottlenecks:")
                        for i, bottleneck in enumerate(bottlenecks[:5], 1):
                            print(f"{i}. {bottleneck['operation']}: {bottleneck['avg_duration']:.2f}s avg")
                    
                    recommendations = self.get_performance_recommendations()
                    if recommendations:
                        print(f"\nPerformance recommendations:")
                        for rec in recommendations[:5]:
                            print(f"- {rec}")
            
        except Exception as e:
            print(f"Error during profiling cleanup: {e}")

# ============================================================================
# END CODE PROFILING AND PERFORMANCE OPTIMIZATION SYSTEM
# ============================================================================

@dataclass
class SEOConfig:
    """Enhanced configuration for SEO optimization with advanced settings."""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    dataloader_num_workers: int = 4
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    # LoRA Configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # P-tuning Configuration
    use_p_tuning: bool = True
    p_tuning_v2: bool = True
    pre_seq_len: int = 20
    prefix_projection: bool = True
    
    # Diffusion Model Configuration
    use_diffusion: bool = True
    diffusion_model_name: str = "runwayml/stable-diffusion-v1-5"
    diffusion_steps: int = 50
    diffusion_guidance_scale: float = 7.5
    diffusion_height: int = 512
    diffusion_width: int = 512
    diffusion_batch_size: int = 1
    
    # Advanced Diffusion Settings
    scheduler_type: str = "ddim"  # ddim, dpm, euler, heun, kdpm2, lms, pndm, unipc
    use_classifier_free_guidance: bool = True
    negative_prompt: str = "low quality, blurry, distorted, watermark, text"
    num_images_per_prompt: int = 1
    
    # Pipeline Configuration
    pipeline_type: str = "stable-diffusion"  # stable-diffusion, stable-diffusion-xl, img2img, inpainting, upscale, depth2img, controlnet, kandinsky, if, deepfloyd, wuerstchen
    controlnet_model: str = "lllyasviel/sd-controlnet-canny"
    kandinsky_model: str = "kandinsky-community/kandinsky-2.2"
    if_model: str = "DeepFloyd/IF-I-XL-v1.0"
    deepfloyd_model: str = "DeepFloyd/IF-I-XL-v1.0"
    wuerstchen_model: str = "wuerstchen/Wuerstchen"
    eta: float = 0.0  # DDIM eta parameter
    clip_skip: int = 1  # CLIP skip for different styles
    cross_attention_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Training configuration
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Early stopping configuration
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    early_stopping_monitor: str = "val_loss"  # "val_loss", "val_accuracy", "train_loss"
    early_stopping_mode: str = "min"  # "min" for loss, "max" for accuracy
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "linear", "exponential", "step", "plateau"
    lr_scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        "cosine": {"T_max": 1000, "eta_min": 1e-7},
        "linear": {"num_warmup_steps": 100},
        "exponential": {"gamma": 0.95},
        "step": {"step_size": 30, "gamma": 0.1},
        "plateau": {"mode": "min", "factor": 0.5, "patience": 3, "min_lr": 1e-7}
    })
    
    # DataLoader configuration
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    
    # Mixed precision and optimization
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    mixed_precision_enabled: bool = True  # Enable/disable mixed precision
    mixed_precision_memory_efficient: bool = True  # Use memory-efficient mixed precision
    mixed_precision_cast_model: bool = True  # Cast model to mixed precision dtype
    mixed_precision_cast_inputs: bool = True  # Cast inputs to mixed precision dtype
    mixed_precision_cast_outputs: bool = False  # Cast outputs to mixed precision dtype
    mixed_precision_autocast_mode: str = "default"  # "default", "inference", "training"
    mixed_precision_grad_scaler: bool = True  # Use gradient scaler for mixed precision
    mixed_precision_grad_scaler_init_scale: float = 2.0**16  # Initial scale for gradient scaler
    mixed_precision_grad_scaler_growth_factor: float = 2.0  # Growth factor for gradient scaler
    mixed_precision_grad_scaler_backoff_factor: float = 0.5  # Backoff factor for gradient scaler
    mixed_precision_grad_scaler_growth_interval: int = 2000  # Growth interval for gradient scaler
    mixed_precision_grad_scaler_enabled: bool = True  # Enable gradient scaler
    mixed_precision_autocast_enabled: bool = True  # Enable autocast
    mixed_precision_autocast_dtype: str = "auto"  # Autocast dtype
    mixed_precision_autocast_cache_enabled: bool = True  # Enable autocast cache
    mixed_precision_autocast_fast_dtype: str = "auto"  # Fast autocast dtype
    mixed_precision_autocast_fallback_dtype: str = "auto"  # Fallback autocast dtype
    use_diffusion: bool = False
    use_xformers: bool = True
    
    # Device configuration
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    tensorboard_logging: bool = True
    
    # PyTorch debugging and development tools
    enable_autograd_anomaly: bool = False  # Enable autograd.detect_anomaly() for debugging
    enable_autograd_profiler: bool = False  # Enable autograd.profiler.profile() for performance profiling
    enable_tensorboard_profiler: bool = False  # Enable TensorBoard profiler
    debug_memory_usage: bool = False  # Enable memory usage debugging
    debug_gradient_norms: bool = False  # Enable gradient norm debugging
    debug_forward_pass: bool = False  # Enable forward pass debugging
    debug_backward_pass: bool = False  # Enable backward pass debugging
    debug_device_placement: bool = False  # Enable device placement debugging
    debug_mixed_precision: bool = False  # Enable mixed precision debugging
    debug_data_loading: bool = False  # Enable data loading debugging
    debug_validation: bool = False  # Enable validation debugging
    debug_early_stopping: bool = False  # Enable early stopping debugging
    debug_lr_scheduling: bool = False  # Enable learning rate scheduling debugging
    
    # Code profiling and performance optimization
    enable_code_profiling: bool = False  # Enable comprehensive code profiling
    profile_data_loading: bool = True  # Profile data loading operations
    profile_preprocessing: bool = True  # Profile data preprocessing operations
    profile_model_inference: bool = True  # Profile model inference operations
    profile_training_loop: bool = True  # Profile training loop operations
    profile_memory_usage: bool = True  # Profile memory usage patterns
    profile_gpu_utilization: bool = True  # Profile GPU utilization
    profile_cpu_utilization: bool = True  # Profile CPU utilization
    profile_io_operations: bool = True  # Profile I/O operations
    profile_network_operations: bool = False  # Profile network operations (if applicable)
    profile_batch_processing: bool = True  # Profile batch processing operations
    profile_validation_loop: bool = True  # Profile validation loop operations
    profile_early_stopping: bool = True  # Profile early stopping operations
    profile_lr_scheduling: bool = True  # Profile learning rate scheduling operations
    profile_mixed_precision: bool = True  # Profile mixed precision operations
    profile_gradient_accumulation: bool = True  # Profile gradient accumulation operations
    profile_multi_gpu: bool = True  # Profile multi-GPU operations
    profile_autocast: bool = True  # Profile autocast operations
    profile_grad_scaler: bool = True  # Profile gradient scaler operations
    profile_checkpointing: bool = True  # Profile checkpointing operations
    profile_logging: bool = False  # Profile logging operations
    profile_error_handling: bool = False  # Profile error handling operations
    profile_input_validation: bool = False  # Profile input validation operations
    profile_gradio_interface: bool = False  # Profile Gradio interface operations
    profile_metrics_calculation: bool = True  # Profile metrics calculation operations
    profile_cross_validation: bool = True  # Profile cross-validation operations
    profile_data_splitting: bool = True  # Profile data splitting operations
    profile_model_compilation: bool = True  # Profile model compilation operations
    profile_optimization_orchestrator: bool = True  # Profile optimization orchestrator operations
    profile_memory_optimizer: bool = True  # Profile memory optimizer operations
    profile_async_data_loader: bool = True  # Profile async data loader operations
    profile_model_compiler: bool = True  # Profile model compiler operations
    profile_performance_benchmarking: bool = True  # Profile performance benchmarking operations
    profile_testing_framework: bool = False  # Profile testing framework operations
    profile_demo_showcase: bool = False  # Profile demo showcase operations
    profile_evaluation_metrics: bool = True  # Profile evaluation metrics operations
    profile_error_handling_validation: bool = False  # Profile error handling and validation operations
    profile_try_except_blocks: bool = False  # Profile try-except block operations
    profile_comprehensive_logging: bool = False  # Profile comprehensive logging operations
    profile_pytorch_debugging: bool = False  # Profile PyTorch debugging operations
    profile_multi_gpu_training: bool = True  # Profile multi-GPU training operations
    profile_gradient_accumulation: bool = True  # Profile gradient accumulation operations
    profile_enhanced_mixed_precision: bool = True  # Profile enhanced mixed precision operations
    profile_code_profiling: bool = False  # Profile the profiling system itself (meta-profiling)
    
    # Multi-GPU training configuration
    use_multi_gpu: bool = False  # Enable multi-GPU training
    multi_gpu_strategy: str = "dataparallel"  # "dataparallel" or "distributed"
    num_gpus: int = 1  # Number of GPUs to use
    distributed_backend: str = "nccl"  # Backend for distributed training (nccl, gloo, mpi)
    distributed_init_method: str = "env://"  # Initialization method for distributed training
    distributed_world_size: int = 1  # Total number of processes
    distributed_rank: int = 0  # Rank of current process
    distributed_master_addr: str = "localhost"  # Master node address
    distributed_master_port: str = "12355"  # Master node port
    sync_batch_norm: bool = False  # Synchronize batch normalization across GPUs
    find_unused_parameters: bool = False  # Find unused parameters in distributed training
    gradient_as_bucket_view: bool = False  # Use gradient as bucket view for efficiency
    broadcast_buffers: bool = True  # Broadcast buffers in distributed training
    bucket_cap_mb: int = 25  # Bucket size in MB for distributed training
    static_graph: bool = False  # Use static graph optimization for distributed training
    
    # Gradient accumulation configuration
    use_gradient_accumulation: bool = False  # Enable gradient accumulation
    effective_batch_size: Optional[int] = None  # Effective batch size (batch_size * accumulation_steps)
    sync_gradients: bool = True  # Synchronize gradients across accumulation steps
    clip_gradients_before_accumulation: bool = False  # Clip gradients before accumulation
    accumulate_gradients_on_cpu: bool = False  # Accumulate gradients on CPU for memory efficiency
    
    # SEO-specific configuration
    seo_keywords: List[str] = field(default_factory=list)
    seo_domains: List[str] = field(default_factory=list)
    seo_content_types: List[str] = field(default_factory=list)

@dataclass
class DataLoaderConfig:
    """Configuration for efficient data loading with cross-validation support."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    shuffle: bool = True
    drop_last: bool = False
    timeout: int = 0
    collate_fn: Optional[callable] = None
    sampler: Optional[Any] = None
    pin_memory_device: str = ""
    generator: Optional[torch.Generator] = None
    multiprocessing_context: Optional[str] = None
    
    # Cross-validation configuration
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "timeseries", "group", "stratified_group"
    cv_repeats: int = 1
    
    # Split configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Stratification and grouping
    stratify_by: Optional[str] = None
    group_by: Optional[str] = None
    
    # Time series configuration
    time_series_split: bool = False
    preserve_order: bool = False
    
    # SEO-specific splitting
    seo_domain_split: bool = False
    seo_keyword_split: bool = False
    seo_content_type_split: bool = False
    
    # Random state
    random_state: int = 42

class SEODataset(Dataset):
    """Custom dataset for SEO data loading with metadata support for cross-validation."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer: Optional[AutoTokenizer] = None, max_length: int = 512,
                 metadata: Optional[Dict[str, List[Any]]] = None):
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metadata = metadata or {}
        
        # Validate metadata lengths
        for key, values in self.metadata.items():
            if len(values) != len(texts):
                raise ValueError(f"Metadata '{key}' length ({len(values)}) doesn't match texts length ({len(texts)})")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            label = self.labels[idx]
            
            item = {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
            # Add metadata with error handling
            try:
                for key, values in self.metadata.items():
                    item[key] = values[idx]
            except (IndexError, KeyError) as e:
                logger.warning(f"Metadata access error for key '{key}' at index {idx}: {e}")
                # Continue without problematic metadata
            
            if self.tokenizer:
                try:
                    # Tokenize text with proper padding and truncation
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    item.update({
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0)
                    })
                except Exception as e:
                    logger.error(f"Tokenization error for text at index {idx}: {e}")
                    # Return safe default tensors
                    item.update({
                        'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                        'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
                    })
            
            return item
            
        except Exception as e:
            logger.error(f"Dataset item retrieval error for index {idx}: {e}")
            # Return a safe default item
            return {
                'text': f"Error loading text at index {idx}",
                'labels': torch.tensor(0, dtype=torch.long),
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }
    
    def get_metadata_column(self, column_name: str) -> List[Any]:
        """Get a specific metadata column."""
        return self.metadata.get(column_name, [])
    
    def get_stratification_targets(self) -> List[Any]:
        """Get targets for stratification (labels by default)."""
        return self.labels
    
    def get_grouping_targets(self) -> Optional[List[Any]]:
        """Get targets for group-based splitting."""
        return self.metadata.get('group', None)


class EarlyStopping:
    """Early stopping implementation for training."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 monitor: str = "val_loss", mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False
            
        if self.mode == "min":
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class DataLoaderManager:
    """Efficient data loading manager using PyTorch DataLoader."""
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.datasets = {}
        self.dataloaders = {}
        self.samplers = {}
        
    def create_dataset(self, name: str, texts: List[str], labels: Optional[List[int]] = None,
                      tokenizer: Optional[AutoTokenizer] = None, max_length: int = 512) -> SEODataset:
        """Create a new SEO dataset."""
        # Profile data loading operations
        if hasattr(self, 'code_profiler'):
            with self.code_profiler.profile_operation("create_dataset", "data_loading"):
                return self._create_dataset_impl(name, texts, labels, tokenizer, max_length)
        else:
            return self._create_dataset_impl(name, texts, labels, tokenizer, max_length)
    
    def _create_dataset_impl(self, name: str, texts: List[str], labels: Optional[List[int]] = None,
                            tokenizer: Optional[AutoTokenizer] = None, max_length: int = 512) -> SEODataset:
        """Implementation of dataset creation."""
        try:
            if not texts:
                raise ValueError("Texts list cannot be empty")
            
            if labels and len(labels) != len(texts):
                raise ValueError(f"Labels length ({len(labels)}) must match texts length ({len(labels)})")
            
            dataset = SEODataset(texts, labels, tokenizer, max_length)
            self.datasets[name] = dataset
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating dataset '{name}': {e}")
            raise
    
    def create_dataloader(self, name: str, dataset: SEODataset, 
                          config: Optional[DataLoaderConfig] = None) -> DataLoader:
        """Create an efficient DataLoader with the given configuration."""
        # Profile data loading operations
        if hasattr(self, 'code_profiler'):
            with self.code_profiler.profile_operation("create_dataloader", "data_loading"):
                return self._create_dataloader_impl(name, dataset, config)
        else:
            return self._create_dataloader_impl(name, dataset, config)
    
    def _create_dataloader_impl(self, name: str, dataset: SEODataset, 
                               config: Optional[DataLoaderConfig] = None) -> DataLoader:
        """Implementation of DataLoader creation."""
        try:
            config = config or self.config
            
            # Validate dataset
            if not dataset:
                raise ValueError("Dataset cannot be None")
            
            if len(dataset) == 0:
                raise ValueError("Dataset is empty")
            
            # Create DataLoader with optimized settings
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=config.persistent_workers,
                prefetch_factor=config.prefetch_factor,
                drop_last=config.drop_last,
                timeout=config.timeout,
                collate_fn=config.collate_fn,
                sampler=config.sampler,
                pin_memory_device=config.pin_memory_device,
                generator=config.generator,
                multiprocessing_context=config.multiprocessing_context
            )
            
            self.dataloaders[name] = dataloader
            return dataloader
            
        except Exception as e:
            logger.error(f"Error creating DataLoader '{name}': {e}")
            raise
    
    def create_train_val_test_split(self, name: str, dataset: SEODataset, 
                                   config: Optional[DataLoaderConfig] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test split with separate DataLoaders."""
        try:
            config = config or self.config
            
            # Validate dataset
            if not dataset or len(dataset) == 0:
                raise ValueError("Dataset is empty or None")
            
            # Calculate split sizes
            total_size = len(dataset)
            train_size = int(total_size * config.train_ratio)
            val_size = int(total_size * config.val_ratio)
            test_size = total_size - train_size - val_size
            
            # Ensure minimum sizes
            if train_size < 1 or val_size < 1 or test_size < 1:
                raise ValueError(f"Dataset too small for 3-way split. Total: {total_size}")
            
            # Create splits
            try:
                if config.time_series_split:
                    # Time series split (preserve order)
                    train_dataset = Subset(dataset, range(0, train_size))
                    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
                    test_dataset = Subset(dataset, range(train_size + val_size, total_size))
                else:
                    # Random split
                    generator = torch.Generator().manual_seed(config.random_state)
                    train_dataset, val_dataset, test_dataset = random_split(
                        dataset, [train_size, val_size, test_size], generator=generator
                    )
            except Exception as e:
                logger.error(f"Error creating dataset splits: {e}")
                raise
            
            # Create DataLoaders
            try:
                train_loader = self.create_dataloader(f"{name}_train", train_dataset, config)
                val_loader = self.create_dataloader(f"{name}_val", val_dataset, config)
                test_loader = self.create_dataloader(f"{name}_test", test_dataset, config)
            except Exception as e:
                logger.error(f"Error creating DataLoaders for split '{name}': {e}")
                raise
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"Error creating train/val/test split for '{name}': {e}")
            raise
    
    def create_train_val_split(self, name: str, dataset: SEODataset, 
                              val_split: float = 0.2, random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
        """Create train/validation split with separate DataLoaders."""
        # Calculate split sizes
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        # Create DataLoaders
        train_loader = self.create_dataloader(f"{name}_train", train_dataset)
        val_loader = self.create_dataloader(f"{name}_val", val_dataset)
        
        return train_loader, val_loader
    
    def create_cross_validation_folds(self, name: str, dataset: SEODataset,
                                     config: Optional[DataLoaderConfig] = None) -> List[Tuple[DataLoader, DataLoader]]:
        """Create cross-validation folds with train/validation DataLoaders."""
        try:
            config = config or self.config
            
            if not config.use_cross_validation:
                raise ValueError("Cross-validation not enabled in config")
            
            # Validate dataset
            if not dataset or len(dataset) == 0:
                raise ValueError("Dataset is empty or None")
            
            # Get stratification targets
            try:
                stratify_targets = dataset.get_stratification_targets()
                grouping_targets = dataset.get_grouping_targets()
            except Exception as e:
                logger.error(f"Error getting stratification/grouping targets: {e}")
                raise
            
            # Create cross-validation splits
            try:
                cv_splits = self._create_cv_splits(
                    len(dataset), stratify_targets, grouping_targets, config
                )
            except Exception as e:
                logger.error(f"Error creating CV splits: {e}")
                raise
            
            # Create DataLoaders for each fold
            fold_dataloaders = []
            try:
                for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
                    train_dataset = Subset(dataset, train_indices)
                    val_dataset = Subset(dataset, val_indices)
                    
                    train_loader = self.create_dataloader(f"{name}_cv_{fold_idx}_train", train_dataset, config)
                    val_loader = self.create_dataloader(f"{name}_cv_{fold_idx}_val", val_dataset, config)
                    
                    fold_dataloaders.append((train_loader, val_loader))
                    
                    # Store fold information
                    self.datasets[f"{name}_cv_{fold_idx}_train"] = train_dataset
                    self.datasets[f"{name}_cv_{fold_idx}_val"] = val_dataset
            except Exception as e:
                logger.error(f"Error creating DataLoaders for CV fold {fold_idx}: {e}")
                raise
            
            return fold_dataloaders
            
        except Exception as e:
            logger.error(f"Error creating cross-validation folds for '{name}': {e}")
            raise
    
    def _create_cv_splits(self, n_samples: int, stratify_targets: List[Any], 
                          grouping_targets: Optional[List[Any]], config: DataLoaderConfig) -> List[Tuple[List[int], List[int]]]:
        """Create cross-validation splits based on strategy."""
        try:
            if config.cv_strategy == "stratified" and stratify_targets:
                try:
                    cv = StratifiedKFold(
                        n_splits=config.cv_folds,
                        shuffle=config.shuffle,
                        random_state=config.random_state
                    )
                    return list(cv.split(range(n_samples), stratify_targets))
                except Exception as e:
                    logger.error(f"Error creating StratifiedKFold: {e}")
                    raise
                    
            elif config.cv_strategy == "kfold":
                try:
                    cv = KFold(
                        n_splits=config.cv_folds,
                        shuffle=config.shuffle,
                        random_state=config.random_state
                    )
                    return list(cv.split(range(n_samples)))
                except Exception as e:
                    logger.error(f"Error creating KFold: {e}")
                    raise
                    
            elif config.cv_strategy == "timeseries":
                try:
                    cv = TimeSeriesSplit(n_splits=config.cv_folds)
                    return list(cv.split(range(n_samples)))
                except Exception as e:
                    logger.error(f"Error creating TimeSeriesSplit: {e}")
                    raise
                    
            elif config.cv_strategy == "group" and grouping_targets:
                try:
                    cv = GroupKFold(n_splits=config.cv_folds)
                    return list(cv.split(range(n_samples), groups=grouping_targets))
                except Exception as e:
                    logger.error(f"Error creating GroupKFold: {e}")
                    raise
                    
            elif config.cv_strategy == "stratified_group" and grouping_targets and stratify_targets:
                try:
                    cv = StratifiedGroupKFold(
                        n_splits=config.cv_folds,
                        shuffle=config.shuffle,
                        random_state=config.random_state
                    )
                    return list(cv.split(range(n_samples), stratify_targets, groups=grouping_targets))
                except Exception as e:
                    logger.error(f"Error creating StratifiedGroupKFold: {e}")
                    raise
                    
            elif config.cv_strategy == "leave_one_out":
                try:
                    cv = LeaveOneOut()
                    return list(cv.split(range(n_samples)))
                except Exception as e:
                    logger.error(f"Error creating LeaveOneOut: {e}")
                    raise
                    
            elif config.cv_strategy == "leave_p_out":
                try:
                    cv = LeavePOut(p=2)
                    return list(cv.split(range(n_samples)))
                except Exception as e:
                    logger.error(f"Error creating LeavePOut: {e}")
                    raise
                    
            elif config.cv_strategy == "repeated_stratified" and stratify_targets:
                try:
                    cv = RepeatedStratifiedKFold(
                        n_splits=config.cv_folds,
                        n_repeats=config.cv_repeats,
                        random_state=config.random_state
                    )
                    return list(cv.split(range(n_samples), stratify_targets))
                except Exception as e:
                    logger.error(f"Error creating RepeatedStratifiedKFold: {e}")
                    raise
                    
            elif config.cv_strategy == "repeated_kfold":
                try:
                    cv = RepeatedKFold(
                        n_splits=config.cv_folds,
                        n_repeats=config.cv_repeats,
                        random_state=config.random_state
                    )
                    return list(cv.split(range(n_samples)))
                except Exception as e:
                    logger.error(f"Error creating RepeatedKFold: {e}")
                    raise
                    
            else:
                # Fallback to K-fold
                try:
                    cv = KFold(
                        n_splits=config.cv_folds,
                        shuffle=config.shuffle,
                        random_state=config.random_state
                    )
                    return list(cv.split(range(n_samples)))
                except Exception as e:
                    logger.error(f"Error creating default KFold: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error in _create_cv_splits: {e}")
            raise
    
    def create_stratified_split(self, name: str, dataset: SEODataset,
                               val_split: float = 0.2, config: Optional[DataLoaderConfig] = None) -> Tuple[DataLoader, DataLoader]:
        """Create stratified train/validation split."""
        try:
            config = config or self.config
            
            # Validate dataset
            if not dataset or len(dataset) == 0:
                raise ValueError("Dataset is empty or None")
            
            # Get stratification targets
            try:
                stratify_targets = dataset.get_stratification_targets()
            except Exception as e:
                logger.error(f"Error getting stratification targets: {e}")
                raise
            
            # Use sklearn's train_test_split for stratification
            try:
                from sklearn.model_selection import train_test_split
                
                train_indices, val_indices = train_test_split(
                    range(len(dataset)),
                    test_size=val_split,
                    stratify=stratify_targets,
                    random_state=config.random_state,
                    shuffle=config.shuffle
                )
            except Exception as e:
                logger.error(f"Error creating stratified split: {e}")
                raise
            
            # Create subsets
            try:
                train_dataset = Subset(dataset, train_indices)
                val_dataset = Subset(dataset, val_indices)
            except Exception as e:
                logger.error(f"Error creating dataset subsets: {e}")
                raise
            
            # Create DataLoaders
            try:
                train_loader = self.create_dataloader(f"{name}_stratified_train", train_dataset, config)
                val_loader = self.create_dataloader(f"{name}_stratified_val", val_dataset, config)
            except Exception as e:
                logger.error(f"Error creating DataLoaders for stratified split: {e}")
                raise
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error creating stratified split for '{name}': {e}")
            raise
    
    def create_distributed_dataloader(self, name: str, dataset: SEODataset,
                                    world_size: int, rank: int,
                                    config: Optional[DataLoaderConfig] = None) -> DataLoader:
        """Create a distributed DataLoader for multi-GPU training."""
        config = config or self.config
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=config.shuffle
        )
        
        # Create distributed DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            drop_last=config.drop_last,
            timeout=config.timeout,
            collate_fn=config.collate_fn,
            pin_memory_device=config.pin_memory_device,
            generator=config.generator,
            multiprocessing_context=config.multiprocessing_context
        )
        
        self.dataloaders[name] = dataloader
        self.samplers[name] = sampler
        return dataloader
    
    def get_dataloader(self, name: str) -> Optional[DataLoader]:
        """Get a DataLoader by name."""
        return self.dataloaders.get(name)
    
    def get_dataset(self, name: str) -> Optional[SEODataset]:
        """Get a dataset by name."""
        return self.datasets.get(name)
    
    def optimize_for_gpu(self, dataloader: DataLoader, device: str = "cuda") -> DataLoader:
        """Optimize DataLoader for GPU training."""
        # Update pin_memory_device for newer PyTorch versions
        if hasattr(dataloader, 'pin_memory_device'):
            dataloader.pin_memory_device = device
        
        return dataloader
    
    def benchmark_dataloader(self, dataloader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
        """Benchmark DataLoader performance."""
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            batch_start = time.time()
            # Simulate processing
            if isinstance(batch, dict) and 'input_ids' in batch:
                _ = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
        
        total_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        
        return {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'batches_per_second': num_batches / total_time if total_time > 0 else 0,
            'samples_per_second': (num_batches * dataloader.batch_size) / total_time if total_time > 0 else 0
        }
    
    def get_cross_validation_info(self, name: str) -> Dict[str, Any]:
        """Get information about cross-validation folds."""
        cv_info = {
            'name': name,
            'folds': [],
            'total_folds': 0
        }
        
        # Find all CV-related datasets
        for dataset_name, dataset in self.datasets.items():
            if name in dataset_name and ('cv_' in dataset_name):
                fold_info = {
                    'dataset_name': dataset_name,
                    'size': len(dataset),
                    'is_train': 'train' in dataset_name
                }
                cv_info['folds'].append(fold_info)
        
        cv_info['total_folds'] = len([f for f in cv_info['folds'] if f['is_train']])
        return cv_info
    
    def validate_cross_validation_setup(self, name: str) -> Dict[str, Any]:
        """Validate cross-validation setup and provide statistics."""
        cv_info = self.get_cross_validation_info(name)
        
        if cv_info['total_folds'] == 0:
            return {'error': f'No cross-validation folds found for {name}'}
        
        # Analyze fold sizes and balance
        train_folds = [f for f in cv_info['folds'] if f['is_train']]
        val_folds = [f for f in cv_info['folds'] if not f['is_train']]
        
        train_sizes = [f['size'] for f in train_folds]
        val_sizes = [f['size'] for f in val_folds]
        
        validation_result = {
            'name': name,
            'total_folds': cv_info['total_folds'],
            'train_fold_sizes': train_sizes,
            'val_fold_sizes': val_sizes,
            'train_size_mean': np.mean(train_sizes),
            'train_size_std': np.std(train_sizes),
            'val_size_mean': np.mean(val_sizes),
            'val_size_std': np.std(val_sizes),
            'is_balanced': len(set(train_sizes)) <= 2 and len(set(val_sizes)) <= 2,
            'total_samples': sum(train_sizes) + sum(val_sizes)
        }
        
        return validation_result
    
    def cleanup(self):
        """Clean up DataLoaders and datasets."""
        for dataloader in self.dataloaders.values():
            del dataloader
        for dataset in self.datasets.values():
            del dataset
        
        self.dataloaders.clear()
        self.datasets.clear()
        self.samplers.clear()
        gc.collect()

class DiffusionPipelineManager:
    """Comprehensive manager for different diffusion pipelines."""
    
    def __init__(self, config: SEOConfig):
        self.config = config
        self.device = config.device
        self.pipelines = {}
        self.current_pipeline = None
        
    def get_pipeline(self, pipeline_type: str = None) -> Any:
        """Get or create the specified diffusion pipeline."""
        if pipeline_type is None:
            pipeline_type = self.config.pipeline_type
            
        if pipeline_type not in self.pipelines:
            self.pipelines[pipeline_type] = self._create_pipeline(pipeline_type)
            
        self.current_pipeline = pipeline_type
        return self.pipelines[pipeline_type]
    
    def _create_pipeline(self, pipeline_type: str) -> Any:
        """Create a specific diffusion pipeline based on type."""
        try:
            if pipeline_type == "stable-diffusion":
                return StableDiffusionPipeline.from_pretrained(
                    self.config.diffusion_model_name,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "stable-diffusion-xl":
                return StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "img2img":
                return StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.config.diffusion_model_name,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "inpainting":
                return StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "upscale":
                return StableDiffusionUpscalePipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler",
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "depth2img":
                return StableDiffusionDepth2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-depth",
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "controlnet":
                controlnet = ControlNetModel.from_pretrained(
                    self.config.controlnet_model,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                )
                return StableDiffusionControlNetPipeline.from_pretrained(
                    self.config.diffusion_model_name,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "kandinsky":
                return KandinskyPipeline.from_pretrained(
                    self.config.kandinsky_model,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "kandinsky-v2":
                return KandinskyV22Pipeline.from_pretrained(
                    self.config.kandinsky_model,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "if":
                return IFPipeline.from_pretrained(
                    self.config.if_model,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "deepfloyd":
                return DeepFloydIFPipeline.from_pretrained(
                    self.config.deepfloyd_model,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            elif pipeline_type == "wuerstchen":
                return WuerstchenPipeline.from_pretrained(
                    self.config.wuerstchen_model,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                    use_safetensors=True
                ).to(self.device)
                
            else:
                raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
                
        except Exception as e:
            logging.error(f"Error creating {pipeline_type} pipeline: {e}")
            # Fallback to stable-diffusion
            return StableDiffusionPipeline.from_pretrained(
                self.config.diffusion_model_name,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                use_safetensors=True
            ).to(self.device)
    
    def optimize_pipeline(self, pipeline: Any, pipeline_type: str) -> Any:
        """Apply optimizations to the pipeline based on type."""
        try:
            if hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
            if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                pipeline.enable_sequential_cpu_offload()
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                pipeline.enable_xformers_memory_efficient_attention()
                
            return pipeline
            
        except Exception as e:
            logging.warning(f"Pipeline optimization failed: {e}")
            return pipeline
    
    def generate_with_pipeline(self, pipeline_type: str, prompt: str, **kwargs) -> Any:
        """Generate content using the specified pipeline."""
        pipeline = self.get_pipeline(pipeline_type)
        pipeline = self.optimize_pipeline(pipeline, pipeline_type)
        
        # Set default parameters
        default_params = {
            'prompt': prompt,
            'num_inference_steps': self.config.diffusion_steps,
            'guidance_scale': self.config.diffusion_guidance_scale,
            'height': self.config.diffusion_height,
            'width': self.config.diffusion_width,
            'num_images_per_prompt': self.config.num_images_per_prompt,
            'negative_prompt': self.config.negative_prompt
        }
        
        # Update with provided kwargs
        default_params.update(kwargs)
        
        try:
            if pipeline_type == "img2img" and 'image' in kwargs:
                return pipeline(**default_params)
            elif pipeline_type == "inpainting" and 'image' in kwargs and 'mask_image' in kwargs:
                return pipeline(**default_params)
            elif pipeline_type == "upscale" and 'image' in kwargs:
                return pipeline(**default_params)
            elif pipeline_type == "depth2img" and 'image' in kwargs:
                return pipeline(**default_params)
            elif pipeline_type == "controlnet" and 'image' in kwargs:
                return pipeline(**default_params)
            else:
                return pipeline(**default_params)
                
        except Exception as e:
            logging.error(f"Generation failed with {pipeline_type}: {e}")
            raise
    
    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipeline types."""
        return [
            "stable-diffusion", "stable-diffusion-xl", "img2img", "inpainting",
            "upscale", "depth2img", "controlnet", "kandinsky", "kandinsky-v2",
            "if", "deepfloyd", "wuerstchen"
        ]
    
    def cleanup(self):
        """Clean up pipeline resources."""
        for pipeline in self.pipelines.values():
            if hasattr(pipeline, 'to'):
                pipeline.to('cpu')
        self.pipelines.clear()
        torch.cuda.empty_cache()
        gc.collect()

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize B with zeros for stable training
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of LoRA layer."""
        return self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, linear_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, r, alpha, dropout)
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original and LoRA weights."""
        return self.linear(x) + self.lora(x)

class PrefixEncoder(nn.Module):
    """Prefix encoder for P-tuning v2."""
    
    def __init__(self, config: SEOConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Prefix embedding
        self.prefix_projection = config.prefix_projection
        self.pre_seq_len = config.pre_seq_len
        
        if self.prefix_projection:
            # Use MLP for prefix projection
            self.embedding = nn.Embedding(self.pre_seq_len, self.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        else:
            # Direct embedding
            self.embedding = nn.Embedding(self.pre_seq_len, self.hidden_size)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate prefix embeddings."""
        prefix_tokens = torch.arange(self.pre_seq_len, dtype=torch.long, device=self.embedding.weight.device)
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        
        prefix_embeds = self.embedding(prefix_tokens)
        
        if self.prefix_projection:
            prefix_embeds = self.trans(prefix_embeds)
        
        return prefix_embeds

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with proper implementation."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output linear transformation
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights

class SEOAttentionLayer(nn.Module):
    """SEO-specific attention layer with keyword-aware attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class AdvancedDiffusionSEOModel(nn.Module):
    """Advanced diffusion-based SEO content generation model using Diffusers library."""
    
    def __init__(self, config: SEOConfig):
        super().__init__()
        self.config = config
        
        # Initialize diffusion components
        self.unet = UNet2DConditionModel.from_pretrained(
            config.diffusion_model_name,
            subfolder="unet",
            cache_dir=config.cache_dir,
            torch_dtype=torch.float16 if config.use_mixed_precision else torch.float32
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            config.diffusion_model_name,
            subfolder="vae",
            cache_dir=config.cache_dir,
            torch_dtype=torch.float16 if config.use_mixed_precision else torch.float32
        )
        
        self.text_encoder = TextEncoder.from_pretrained(
            config.diffusion_model_name,
            subfolder="text_encoder",
            cache_dir=config.cache_dir,
            torch_dtype=torch.float16 if config.use_mixed_precision else torch.float32
        )
        
        # Initialize advanced schedulers based on config
        self.noise_scheduler = self._get_scheduler(config.scheduler_type)
        
        # Diffusion process parameters
        self.num_train_timesteps = getattr(self.noise_scheduler, 'num_train_timesteps', 1000)
        self.beta_start = getattr(self.noise_scheduler, 'beta_start', 0.00085)
        self.beta_end = getattr(self.noise_scheduler, 'beta_end', 0.012)
        
        # SEO-specific conditioning layers with attention
        self.seo_conditioning = nn.Sequential(
            nn.Linear(768, 512),  # Text encoder output size
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Keyword enhancement layer with residual connections
        self.keyword_enhancer = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 768)
        )
        
        # SEO feature attention mechanism
        self.seo_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _get_scheduler(self, scheduler_type: str):
        """Get appropriate scheduler based on configuration with advanced parameter tuning."""
        scheduler_map = {
            "ddim": DDIMScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "heun": HeunDiscreteScheduler,
            "kdpm2": KDPM2DiscreteScheduler,
            "lms": LMSDiscreteScheduler,
            "pndm": PNDMScheduler,
            "unipc": UniPCMultistepScheduler
        }
        
        scheduler_class = scheduler_map.get(scheduler_type.lower(), DDIMScheduler)
        
        try:
            scheduler = scheduler_class.from_pretrained(
                self.config.diffusion_model_name,
                subfolder="scheduler",
                cache_dir=self.config.cache_dir
            )
            
            # Advanced scheduler-specific parameter configuration
            self._configure_advanced_scheduler_params(scheduler, scheduler_type)
            
            return scheduler
        except Exception as e:
            logging.warning(f"Failed to load {scheduler_type} scheduler, falling back to DDIM: {e}")
            return DDIMScheduler.from_pretrained(
                self.config.diffusion_model_name,
                subfolder="scheduler",
                cache_dir=self.config.cache_dir
            )
    
    def _configure_advanced_scheduler_params(self, scheduler, scheduler_type: str):
        """Configure advanced parameters for different scheduler types."""
        # Common parameters for all schedulers
        if hasattr(scheduler, 'beta_start'):
            scheduler.beta_start = 0.00085
        if hasattr(scheduler, 'beta_end'):
            scheduler.beta_end = 0.012
        if hasattr(scheduler, 'beta_schedule'):
            scheduler.beta_schedule = "scaled_linear"
        
        # Scheduler-specific advanced configurations
        if scheduler_type.lower() == "ddim":
            if hasattr(scheduler, 'eta'):
                scheduler.eta = self.config.eta
            if hasattr(scheduler, 'clip_denoised'):
                scheduler.clip_denoised = True
            if hasattr(scheduler, 'set_alpha_to_one'):
                scheduler.set_alpha_to_one = False
        
        elif scheduler_type.lower() == "dpm":
            if hasattr(scheduler, 'algorithm_type'):
                scheduler.algorithm_type = "dpmsolver++"
            if hasattr(scheduler, 'solver_type'):
                scheduler.solver_type = "midpoint"
            if hasattr(scheduler, 'lower_order_final'):
                scheduler.lower_order_final = True
            if hasattr(scheduler, 'use_karras_sigmas'):
                scheduler.use_karras_sigmas = True
        
        elif scheduler_type.lower() == "euler":
            if hasattr(scheduler, 'use_karras_sigmas'):
                scheduler.use_karras_sigmas = True
            if hasattr(scheduler, 'timestep_spacing'):
                scheduler.timestep_spacing = "leading"
        
        elif scheduler_type.lower() == "heun":
            if hasattr(scheduler, 'use_karras_sigmas'):
                scheduler.use_karras_sigmas = True
            if hasattr(scheduler, 'timestep_spacing'):
                scheduler.timestep_spacing = "leading"
        
        elif scheduler_type.lower() == "kdpm2":
            if hasattr(scheduler, 'use_karras_sigmas'):
                scheduler.use_karras_sigmas = True
            if hasattr(scheduler, 'timestep_spacing'):
                scheduler.timestep_spacing = "leading"
        
        elif scheduler_type.lower() == "lms":
            if hasattr(scheduler, 'use_karras_sigmas'):
                scheduler.use_karras_sigmas = True
            if hasattr(scheduler, 'timestep_spacing'):
                scheduler.timestep_spacing = "leading"
        
        elif scheduler_type.lower() == "pndm":
            if hasattr(scheduler, 'skip_prk_steps'):
                scheduler.skip_prk_steps = False
            if hasattr(scheduler, 'set_alpha_to_one'):
                scheduler.set_alpha_to_one = False
        
        elif scheduler_type.lower() == "unipc":
            if hasattr(scheduler, 'solver_order'):
                scheduler.solver_order = 2
            if hasattr(scheduler, 'use_corrector'):
                scheduler.use_corrector = True
            if hasattr(scheduler, 'use_linear_multistep'):
                scheduler.use_linear_multistep = True
    
    def _initialize_weights(self):
        """Initialize model weights using proper initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, prompt_embeds, seo_features=None, keywords=None):
        """Forward pass with SEO conditioning and advanced diffusion."""
        # Enhance prompt with SEO features
        if seo_features is not None:
            seo_conditioned = self.seo_conditioning(seo_features)
            enhanced_prompt = prompt_embeds + self.keyword_enhancer(seo_conditioned)
            
            # Apply SEO attention mechanism
            enhanced_prompt = enhanced_prompt.unsqueeze(1)  # Add sequence dimension
            attn_output, _ = self.seo_attention(
                enhanced_prompt, enhanced_prompt, enhanced_prompt
            )
            enhanced_prompt = attn_output.squeeze(1)  # Remove sequence dimension
        else:
            enhanced_prompt = prompt_embeds
        
        # Generate content using advanced diffusion
        latents = self._generate_latents_advanced(enhanced_prompt)
        
        # Decode latents to images
        images = self.vae.decode(latents)
        
        return images, latents
    
    def _generate_latents_advanced(self, prompt_embeds):
        """Generate latents using advanced diffusion process with multiple schedulers."""
        # Initialize random latents
        latents = torch.randn(
            (prompt_embeds.shape[0], 4, self.config.diffusion_height // 8, self.config.diffusion_width // 8),
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype
        )
        
        # Scale latents
        latents = latents * self.noise_scheduler.init_noise_sigma
        
        # Advanced denoising loop with adaptive step sizes and scheduler-specific handling
        latents = self._advanced_denoising_loop(latents, prompt_embeds)
        
        return latents
    
    def _advanced_denoising_loop(self, latents: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Advanced denoising loop with adaptive step sizes and optimal sampling."""
        # Get optimal timesteps for sampling
        optimal_timesteps = self._get_optimal_sampling_timesteps()
        
        # Initialize tracking variables
        current_latents = latents
        step_history = []
        
        for step_idx, timestep in enumerate(optimal_timesteps):
            # Prepare timestep tensor
            timestep_tensor = torch.tensor([timestep], device=latents.device, dtype=torch.long)
            
            # Predict noise with UNet
            noise_pred = self.unet(
                current_latents, 
                timestep_tensor, 
                encoder_hidden_states=prompt_embeds
            ).sample
            
            # Apply advanced sampling with scheduler-specific optimizations
            current_latents = self._apply_advanced_sampling_step(
                current_latents, noise_pred, timestep_tensor, step_idx
            )
            
            # Track step information for analysis
            step_history.append({
                'step': step_idx,
                'timestep': timestep,
                'latent_norm': torch.norm(current_latents).item(),
                'noise_norm': torch.norm(noise_pred).item()
            })
        
        return current_latents
    
    def _get_optimal_sampling_timesteps(self) -> List[int]:
        """Get optimal timesteps for sampling based on noise schedule analysis."""
        try:
            # Get noise schedule information
            noise_info = self.get_noise_schedule_info()
            
            # Use optimal timesteps if available
            if 'optimal_sampling_timesteps' in noise_info:
                return noise_info['optimal_sampling_timesteps']
            
            # Fallback to adaptive timestep selection
            return self._adaptive_timestep_selection()
            
        except Exception as e:
            logging.warning(f"Failed to get optimal timesteps, using default: {e}")
            # Default to evenly spaced timesteps
            return torch.linspace(
                self.num_train_timesteps - 1, 0, self.config.diffusion_steps, 
                dtype=torch.long
            ).tolist()
    
    def _adaptive_timestep_selection(self) -> List[int]:
        """Adaptively select timesteps based on noise schedule characteristics."""
        try:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            
            # Calculate noise variance at each timestep
            noise_variance = 1 - alphas_cumprod
            
            # Find regions with significant noise changes
            noise_changes = torch.diff(noise_variance)
            significant_changes = torch.where(torch.abs(noise_changes) > torch.std(noise_changes))[0]
            
            # Select timesteps focusing on significant changes
            if len(significant_changes) >= self.config.diffusion_steps:
                # Use significant change points
                step_indices = torch.linspace(0, len(significant_changes) - 1, 
                                           self.config.diffusion_steps, dtype=torch.long)
                selected_timesteps = significant_changes[step_indices]
            else:
                # Combine significant changes with evenly spaced
                num_even = self.config.diffusion_steps - len(significant_changes)
                even_timesteps = torch.linspace(0, self.num_train_timesteps - 1, 
                                             num_even, dtype=torch.long)
                selected_timesteps = torch.cat([significant_changes, even_timesteps])
                selected_timesteps = torch.sort(selected_timesteps, descending=True)[0]
            
            return selected_timesteps.tolist()
            
        except Exception as e:
            logging.warning(f"Adaptive timestep selection failed: {e}")
            # Fallback to simple linear selection
            return torch.linspace(
                self.num_train_timesteps - 1, 0, self.config.diffusion_steps, 
                dtype=torch.long
            ).tolist()
    
    def _apply_advanced_sampling_step(self, latents: torch.Tensor, noise_pred: torch.Tensor, 
                                    timestep: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Apply advanced sampling step with scheduler-specific optimizations."""
        try:
            # Apply scheduler step with advanced parameters
            if hasattr(self.noise_scheduler, 'step'):
                # Get scheduler-specific parameters
                step_kwargs = self._get_scheduler_step_kwargs(timestep, step_idx)
                
                # Apply scheduler step
                scheduler_output = self.noise_scheduler.step(
                    noise_pred, timestep, latents, **step_kwargs
                )
                
                # Extract result
                if hasattr(scheduler_output, 'prev_sample'):
                    return scheduler_output.prev_sample
                elif hasattr(scheduler_output, 'sample'):
                    return scheduler_output.sample
                else:
                    return scheduler_output
            else:
                # Fallback for schedulers without step method
                return self._fallback_sampling_step(latents, noise_pred, step_idx)
                
        except Exception as e:
            logging.warning(f"Advanced sampling step failed: {e}")
            # Fallback to simple noise reduction
            return self._fallback_sampling_step(latents, noise_pred, step_idx)
    
    def _get_scheduler_step_kwargs(self, timestep: torch.Tensor, step_idx: int) -> Dict[str, Any]:
        """Get scheduler-specific parameters for the current step."""
        kwargs = {}
        
        # DDIM-specific parameters
        if isinstance(self.noise_scheduler, DDIMScheduler):
            kwargs['eta'] = self.config.eta
        
        # DPM-Solver specific parameters
        elif hasattr(self.noise_scheduler, 'algorithm_type'):
            if self.noise_scheduler.algorithm_type == "dpmsolver++":
                kwargs['solver_order'] = min(2, self.config.diffusion_steps - step_idx)
        
        # Euler/Heun specific parameters
        elif hasattr(self.noise_scheduler, 'use_karras_sigmas'):
            kwargs['use_karras_sigmas'] = True
        
        return kwargs
    
    def _fallback_sampling_step(self, latents: torch.Tensor, noise_pred: torch.Tensor, 
                               step_idx: int) -> torch.Tensor:
        """Fallback sampling step when scheduler step method is not available."""
        # Simple noise reduction with adaptive step size
        step_size = 0.1 * (1.0 - step_idx / self.config.diffusion_steps)
        return latents - noise_pred * step_size
    
    def forward_diffusion_process(self, x_0: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Implement forward diffusion process (q(x_t | x_0)).
        Gradually adds noise to the original image according to the noise schedule.
        
        Args:
            x_0: Original clean image [B, C, H, W]
            timesteps: Timesteps for noise addition [B]
        
        Returns:
            x_t: Noisy image at timestep t
        """
        # Get noise schedule parameters
        alphas = self.noise_scheduler.alphas_cumprod[timesteps]
        alphas = alphas.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        x_t = torch.sqrt(alphas) * x_0 + torch.sqrt(1 - alphas) * noise
        
        return x_t, noise
    
    def reverse_diffusion_process(self, x_t: torch.Tensor, timesteps: torch.Tensor, 
                                predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Implement reverse diffusion process (p(x_{t-1} | x_t)).
        Gradually removes noise from the noisy image using predicted noise.
        
        Args:
            x_t: Noisy image at timestep t [B, C, H, W]
            timesteps: Current timesteps [B]
            predicted_noise: Predicted noise from UNet [B, C, H, W]
        
        Returns:
            x_prev: Denoised image at previous timestep
        """
        # Get noise schedule parameters
        alphas = self.noise_scheduler.alphas_cumprod[timesteps]
        alphas = alphas.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Reverse diffusion step
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / sqrt(1 - alpha_t)) * predicted_noise)
        x_prev = (1 / torch.sqrt(alphas)) * (
            x_t - ((1 - alphas) / torch.sqrt(1 - alphas)) * predicted_noise
        )
        
        return x_prev
    
    def compute_loss(self, x_0: torch.Tensor, timesteps: torch.Tensor, 
                    predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion loss for training.
        
        Args:
            x_0: Original clean image
            timesteps: Timesteps
            predicted_noise: Noise predicted by the model
            target_noise: Ground truth noise
        
        Returns:
            loss: Diffusion loss
        """
        # Simple MSE loss between predicted and target noise
        loss = F.mse_loss(predicted_noise, target_noise, reduction='mean')
        return loss
    
    def sample_trajectory(self, x_0: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
        """
        Sample the forward diffusion trajectory to visualize the noise addition process.
        
        Args:
            x_0: Original clean image
            num_steps: Number of steps to sample
        
        Returns:
            trajectory: List of images at different noise levels
        """
        trajectory = [x_0]
        
        # Sample timesteps
        timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_steps, 
                                 dtype=torch.long, device=x_0.device)
        
        for t in timesteps:
            x_t, _ = self.forward_diffusion_process(x_0, t.unsqueeze(0))
            trajectory.append(x_t)
        
        return trajectory
    
    def denoise_trajectory(self, x_T: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
        """
        Sample the reverse diffusion trajectory to visualize the denoising process.
        
        Args:
            x_T: Fully noisy image
            num_steps: Number of denoising steps
        
        Returns:
            trajectory: List of images at different denoising levels
        """
        trajectory = [x_T]
        x_t = x_T
        
        # Sample timesteps in reverse order
        timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_steps, 
                                 dtype=torch.long, device=x_T.device)
        
        for t in timesteps:
            # Predict noise using UNet
            with torch.no_grad():
                noise_pred = self.unet(x_t, t.unsqueeze(0)).sample
            
            # Denoise step
            x_t = self.reverse_diffusion_process(x_t, t.unsqueeze(0), noise_pred)
            trajectory.append(x_t)
        
        return trajectory
    
    def get_noise_schedule_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the noise schedule.
        
        Returns:
            info: Dictionary containing noise schedule parameters
        """
        info = {
            'num_train_timesteps': self.num_train_timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'scheduler_type': self.config.scheduler_type,
            'scheduler_class': self.noise_scheduler.__class__.__name__,
            'alphas_cumprod_shape': self.noise_scheduler.alphas_cumprod.shape,
            'betas_shape': getattr(self.noise_scheduler, 'betas', None),
            'alphas_shape': getattr(self.noise_scheduler, 'alphas', None)
        }
        
        # Add advanced noise schedule analysis
        info.update(self._analyze_noise_schedule_characteristics())
        
        return info
    
    def _analyze_noise_schedule_characteristics(self) -> Dict[str, Any]:
        """Analyze noise schedule characteristics for optimal sampling."""
        try:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            betas = getattr(self.noise_scheduler, 'betas', None)
            
            # Calculate noise schedule statistics
            noise_variance = 1 - alphas_cumprod
            signal_to_noise_ratio = alphas_cumprod / (1 - alphas_cumprod + 1e-8)
            
            # Find optimal sampling points
            optimal_timesteps = self._find_optimal_sampling_timesteps(alphas_cumprod)
            
            # Analyze noise schedule smoothness
            noise_smoothness = self._calculate_noise_schedule_smoothness(alphas_cumprod)
            
            return {
                'noise_variance_range': (noise_variance.min().item(), noise_variance.max().item()),
                'signal_to_noise_range': (signal_to_noise_ratio.min().item(), signal_to_noise_ratio.max().item()),
                'optimal_sampling_timesteps': optimal_timesteps,
                'noise_schedule_smoothness': noise_smoothness,
                'beta_schedule_type': getattr(self.noise_scheduler, 'beta_schedule', 'unknown'),
                'has_karras_sigmas': getattr(self.noise_scheduler, 'use_karras_sigmas', False)
            }
        except Exception as e:
            return {'error': f"Failed to analyze noise schedule: {str(e)}"}
    
    def _find_optimal_sampling_timesteps(self, alphas_cumprod: torch.Tensor, num_steps: int = 10) -> List[int]:
        """Find optimal timesteps for sampling based on noise schedule characteristics."""
        try:
            # Calculate noise variance at each timestep
            noise_variance = 1 - alphas_cumprod
            
            # Find timesteps with significant noise changes
            noise_changes = torch.diff(noise_variance)
            significant_changes = torch.where(torch.abs(noise_changes) > torch.std(noise_changes))[0]
            
            # Select evenly distributed timesteps from significant changes
            if len(significant_changes) >= num_steps:
                step_indices = torch.linspace(0, len(significant_changes) - 1, num_steps, dtype=torch.long)
                optimal_timesteps = significant_changes[step_indices].tolist()
            else:
                # Fallback to evenly distributed timesteps
                step_indices = torch.linspace(0, len(alphas_cumprod) - 1, num_steps, dtype=torch.long)
                optimal_timesteps = step_indices.tolist()
            
            return optimal_timesteps
        except Exception as e:
            # Fallback to simple linear sampling
            return torch.linspace(0, len(alphas_cumprod) - 1, num_steps, dtype=torch.long).tolist()
    
    def _calculate_noise_schedule_smoothness(self, alphas_cumprod: torch.Tensor) -> float:
        """Calculate smoothness of the noise schedule."""
        try:
            # Calculate second derivative of noise variance
            noise_variance = 1 - alphas_cumprod
            first_derivative = torch.diff(noise_variance)
            second_derivative = torch.diff(first_derivative)
            
            # Smoothness is inverse of second derivative magnitude
            smoothness = 1.0 / (1.0 + torch.mean(torch.abs(second_derivative)))
            return smoothness.item()
        except Exception:
            return 0.5  # Default smoothness value

class AdvancedDiffusionSEOGenerator:
    """Advanced diffusion-based SEO content generator using comprehensive Diffusers library features."""
    
    def __init__(self, config: SEOConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.diffusion_model = None
        self.pipeline = None
        self.logger = logging.getLogger(__name__)
        self._initialize_diffusion()
        
        # Initialize diffusion model for training and understanding
        if config.use_diffusion:
            self.diffusion_model = AdvancedDiffusionSEOModel(config)
            self.diffusion_model.to(self.device)
    
    def _initialize_diffusion(self):
        """Initialize advanced diffusion pipeline with multiple model support."""
        try:
            if self.config.use_diffusion:
                # Initialize with advanced configuration
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.config.diffusion_model_name,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32,
                    safety_checker=None,  # Disable safety checker for faster inference
                    requires_safety_checker=False
                )
                
                # Move to device
                self.pipeline = self.pipeline.to(self.device)
                
                # Apply advanced optimizations
                if self.config.use_mixed_precision:
                    self.pipeline.enable_attention_slicing()
                    self.pipeline.enable_vae_slicing()
                    self.pipeline.enable_model_cpu_offload()
                    self.pipeline.enable_sequential_cpu_offload()
                
                # Configure scheduler based on config
                self._configure_scheduler()
                
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except:
                    self.logger.info("xformers not available, using standard attention")
                
                self.logger.info(f"Advanced diffusion pipeline initialized with {self.config.scheduler_type} scheduler")
        except Exception as e:
            self.logger.error(f"Error initializing diffusion pipeline: {e}")
            raise
    
    def _configure_scheduler(self):
        """Configure the appropriate scheduler based on configuration."""
        scheduler_map = {
            "ddim": DDIMScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "heun": HeunDiscreteScheduler,
            "kdpm2": KDPM2DiscreteScheduler,
            "lms": LMSDiscreteScheduler,
            "pndm": PNDMScheduler,
            "unipc": UniPCMultistepScheduler
        }
        
        scheduler_class = scheduler_map.get(self.config.scheduler_type.lower(), DDIMScheduler)
        
        try:
            # Create new scheduler instance
            new_scheduler = scheduler_class.from_config(self.pipeline.scheduler.config)
            
            # Apply advanced scheduler configuration
            self._apply_advanced_scheduler_config(new_scheduler)
            
            # Replace scheduler
            self.pipeline.scheduler = new_scheduler
            self.logger.info(f"Scheduler configured to {self.config.scheduler_type}")
            
        except Exception as e:
            self.logger.warning(f"Failed to configure {self.config.scheduler_type} scheduler: {e}")
    
    def _apply_advanced_scheduler_config(self, scheduler):
        """Apply advanced configuration to scheduler for optimal performance."""
        # Common parameters
        if hasattr(scheduler, 'beta_start'):
            scheduler.beta_start = 0.00085
        if hasattr(scheduler, 'beta_end'):
            scheduler.beta_end = 0.012
        if hasattr(scheduler, 'beta_schedule'):
            scheduler.beta_schedule = "scaled_linear"
        
        # Scheduler-specific advanced configurations
        scheduler_type = self.config.scheduler_type.lower()
        
        if scheduler_type == "ddim":
            if hasattr(scheduler, 'eta'):
                scheduler.eta = self.config.eta
            if hasattr(scheduler, 'clip_denoised'):
                scheduler.clip_denoised = True
            if hasattr(scheduler, 'set_alpha_to_one'):
                scheduler.set_alpha_to_one = False
        
        elif scheduler_type == "dpm":
            if hasattr(scheduler, 'algorithm_type'):
                scheduler.algorithm_type = "dpmsolver++"
            if hasattr(scheduler, 'solver_type'):
                scheduler.solver_type = "midpoint"
            if hasattr(scheduler, 'lower_order_final'):
                scheduler.lower_order_final = True
            if hasattr(scheduler, 'use_karras_sigmas'):
                scheduler.use_karras_sigmas = True
        
        elif scheduler_type in ["euler", "heun", "kdpm2", "lms"]:
            if hasattr(scheduler, 'use_karras_sigmas'):
                scheduler.use_karras_sigmas = True
            if hasattr(scheduler, 'timestep_spacing'):
                scheduler.timestep_spacing = "leading"
        
        elif scheduler_type == "pndm":
            if hasattr(scheduler, 'skip_prk_steps'):
                scheduler.skip_prk_steps = False
            if hasattr(scheduler, 'set_alpha_to_one'):
                scheduler.set_alpha_to_one = False
        
        elif scheduler_type == "unipc":
            if hasattr(scheduler, 'solver_order'):
                scheduler.solver_order = 2
            if hasattr(scheduler, 'use_corrector'):
                scheduler.use_corrector = True
            if hasattr(scheduler, 'use_linear_multistep'):
                scheduler.use_linear_multistep = True
    
    def generate_seo_visual_content(self, prompt: str, seo_features: Dict[str, Any], 
                                  keywords: List[str] = None) -> Dict[str, Any]:
        """Generate SEO-optimized visual content using advanced diffusion."""
        try:
            # Enhance prompt with SEO features
            enhanced_prompt = self._enhance_prompt_with_seo(prompt, seo_features, keywords)
            
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": enhanced_prompt,
                "negative_prompt": self.config.negative_prompt,
                "num_inference_steps": self.config.diffusion_steps,
                "guidance_scale": self.config.diffusion_guidance_scale,
                "height": self.config.diffusion_height,
                "width": self.config.diffusion_width,
                "num_images_per_prompt": self.config.num_images_per_prompt,
                "eta": self.config.eta,
                "generator": torch.Generator(device=self.device).manual_seed(42),  # Reproducible results
                "cross_attention_kwargs": self.config.cross_attention_kwargs
            }
            
            # Add scheduler-specific parameters
            if hasattr(self.pipeline.scheduler, 'algorithm_type'):
                generation_kwargs["algorithm_type"] = self.pipeline.scheduler.algorithm_type
            
            # Generate image with advanced pipeline
            with torch.autocast(self.device.type) if self.config.use_mixed_precision else torch.no_grad():
                result = self.pipeline(**generation_kwargs)
            
            # Extract results
            images = result.images
            nsfw_content_detected = getattr(result, 'nsfw_content_detected', [False] * len(images))
            
            return {
                "images": images,
                "enhanced_prompt": enhanced_prompt,
                "seo_features": seo_features,
                "generation_params": {
                    "steps": self.config.diffusion_steps,
                    "guidance_scale": self.config.diffusion_guidance_scale,
                    "dimensions": (self.config.diffusion_height, self.config.diffusion_width),
                    "scheduler": self.config.scheduler_type,
                    "eta": self.config.eta,
                    "nsfw_detected": nsfw_content_detected
                },
                "pipeline_info": {
                    "model_name": self.config.diffusion_model_name,
                    "device": str(self.device),
                    "mixed_precision": self.config.use_mixed_precision,
                    "attention_slicing": hasattr(self.pipeline, '_attention_slicing_enabled'),
                    "vae_slicing": hasattr(self.pipeline, '_vae_slicing_enabled')
                }
            }
        except Exception as e:
            self.logger.error(f"Error generating visual content: {e}")
            return {"error": str(e)}
    
    def _enhance_prompt_with_seo(self, prompt: str, seo_features: Dict[str, Any], 
                                keywords: List[str] = None) -> str:
        """Enhance prompt with SEO features and keywords."""
        enhanced = prompt
        
        # Add keyword information
        if keywords:
            keyword_text = ", ".join(keywords)
            enhanced += f", featuring keywords: {keyword_text}"
        
        # Add SEO quality indicators
        if seo_features.get("readability_score", 0) > 0.7:
            enhanced += ", high readability content"
        if seo_features.get("keyword_density", {}).get("density", 0) > 0.5:
            enhanced += ", keyword optimized"
        if seo_features.get("content_length", 0) > 500:
            enhanced += ", comprehensive content"
        
        # Add visual style based on content type
        if "guide" in prompt.lower() or "tutorial" in prompt.lower():
            enhanced += ", educational infographic style"
        elif "product" in prompt.lower() or "review" in prompt.lower():
            enhanced += ", product showcase style"
        else:
            enhanced += ", professional business style"
        
        return enhanced
    
    def understand_diffusion_processes(self) -> Dict[str, Any]:
        """
        Comprehensive understanding of forward and reverse diffusion processes.
        
        Returns:
            explanation: Detailed explanation of diffusion processes
        """
        explanation = {
            "forward_diffusion": {
                "description": "Forward diffusion gradually adds noise to clean images",
                "mathematical_formula": "q(x_t | x_0) = N(x_t; sqrt(α_t) * x_0, (1 - α_t) * I)",
                "process": [
                    "Start with clean image x_0",
                    "Sample random noise ε ~ N(0, I)",
                    "Apply noise schedule: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε",
                    "Repeat for T timesteps until image is pure noise"
                ],
                "purpose": "Create training data for noise prediction"
            },
            "reverse_diffusion": {
                "description": "Reverse diffusion gradually removes noise using learned model",
                "mathematical_formula": "p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))",
                "process": [
                    "Start with pure noise x_T",
                    "Predict noise ε_θ using UNet model",
                    "Denoise step: x_{t-1} = denoise(x_t, ε_θ, t)",
                    "Repeat for T timesteps until clean image"
                ],
                "purpose": "Generate new images from noise"
            },
            "noise_schedule": {
                "beta_schedule": "Linear, cosine, or sigmoid noise variance schedule",
                "alphas_cumprod": "Cumulative product of (1 - β_t) values",
                "timesteps": f"Total diffusion steps: {self.config.diffusion_steps}"
            },
            "training_objective": "Minimize MSE between predicted and actual noise",
            "inference": "Use trained model to predict noise and denoise step by step"
        }
        
        # Add advanced noise scheduler information
        if self.diffusion_model:
            noise_info = self.diffusion_model.get_noise_schedule_info()
            explanation["advanced_noise_schedule"] = noise_info
        
        return explanation
    
    def analyze_noise_schedulers(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of different noise schedulers and their characteristics.
        
        Returns:
            analysis: Detailed analysis of noise schedulers
        """
        schedulers = {
            "ddim": {
                "description": "Denoising Diffusion Implicit Models",
                "advantages": ["Deterministic sampling", "Fast inference", "Good quality"],
                "disadvantages": ["Less diverse outputs", "Sensitive to eta parameter"],
                "best_for": ["Quick generation", "Reproducible results", "High quality"],
                "parameters": ["eta", "clip_denoised", "set_alpha_to_one"]
            },
            "dpm": {
                "description": "DPM-Solver: Fast ODE solver for diffusion models",
                "advantages": ["Very fast sampling", "High quality", "Adaptive step sizes"],
                "disadvantages": ["Complex implementation", "Memory intensive"],
                "best_for": ["Fast generation", "High quality", "Production use"],
                "parameters": ["algorithm_type", "solver_type", "use_karras_sigmas"]
            },
            "euler": {
                "description": "Euler method for ODE solving",
                "advantages": ["Simple", "Memory efficient", "Stable"],
                "disadvantages": ["Slower convergence", "Lower quality"],
                "best_for": ["Memory-constrained environments", "Simple applications"],
                "parameters": ["use_karras_sigmas", "timestep_spacing"]
            },
            "heun": {
                "description": "Heun's method (2nd order Runge-Kutta)",
                "advantages": ["Better accuracy than Euler", "Good balance"],
                "disadvantages": ["Slower than Euler", "More complex"],
                "best_for": ["Balanced quality/speed", "Educational purposes"],
                "parameters": ["use_karras_sigmas", "timestep_spacing"]
            },
            "kdpm2": {
                "description": "2nd order Karras DPM-Solver",
                "advantages": ["High quality", "Fast", "Stable"],
                "disadvantages": ["Memory usage", "Complexity"],
                "best_for": ["High quality generation", "Production systems"],
                "parameters": ["use_karras_sigmas", "timestep_spacing"]
            },
            "lms": {
                "description": "Linear Multistep method",
                "advantages": ["Stable", "Good quality", "Predictable"],
                "disadvantages": ["Slower than DPM", "Memory usage"],
                "best_for": ["Stable generation", "Research", "Debugging"],
                "parameters": ["use_karras_sigmas", "timestep_spacing"]
            },
            "pndm": {
                "description": "Pseudo Numerical Methods for Diffusion Models",
                "advantages": ["Fast", "Good quality", "Stable"],
                "disadvantages": ["Complex implementation", "Parameter sensitivity"],
                "best_for": ["Fast generation", "Balanced quality/speed"],
                "parameters": ["skip_prk_steps", "set_alpha_to_one"]
            },
            "unipc": {
                "description": "UniPC: Unified Predictor-Corrector Framework",
                "advantages": ["High quality", "Adaptive", "Flexible"],
                "disadvantages": ["Complex", "Slower than DPM"],
                "best_for": ["High quality generation", "Research", "Custom sampling"],
                "parameters": ["solver_order", "use_corrector", "use_linear_multistep"]
            }
        }
        
        # Add current scheduler analysis
        current_scheduler = self.config.scheduler_type.lower()
        if current_scheduler in schedulers:
            schedulers["current_scheduler"] = {
                "name": current_scheduler,
                "info": schedulers[current_scheduler],
                "recommendations": self._get_scheduler_recommendations(current_scheduler)
            }
        
        return {
            "schedulers": schedulers,
            "current_scheduler": current_scheduler,
            "optimal_choice": self._recommend_optimal_scheduler()
        }
    
    def _get_scheduler_recommendations(self, scheduler_type: str) -> List[str]:
        """Get specific recommendations for a scheduler type."""
        recommendations = {
            "ddim": [
                "Use eta=0.0 for deterministic sampling",
                "Use eta=1.0 for more diverse outputs",
                "Adjust steps based on quality vs speed trade-off"
            ],
            "dpm": [
                "Use algorithm_type='dpmsolver++' for best quality",
                "Enable use_karras_sigmas for better noise handling",
                "Use solver_type='midpoint' for balanced performance"
            ],
            "euler": [
                "Enable use_karras_sigmas for better noise schedules",
                "Use more steps for better quality",
                "Good for memory-constrained environments"
            ],
            "heun": [
                "Better quality than Euler with similar memory usage",
                "Good for educational and research purposes",
                "Balanced performance characteristics"
            ],
            "kdpm2": [
                "High quality with reasonable speed",
                "Enable use_karras_sigmas for optimal performance",
                "Good for production systems"
            ],
            "lms": [
                "Stable and predictable results",
                "Good for debugging and research",
                "Consistent quality across different settings"
            ],
            "pndm": [
                "Fast generation with good quality",
                "Adjust skip_prk_steps based on quality needs",
                "Good for real-time applications"
            ],
            "unipc": [
                "High quality with flexible sampling",
                "Use higher solver_order for better quality",
                "Enable use_corrector for improved results"
            ]
        }
        
        return recommendations.get(scheduler_type, ["No specific recommendations available"])
    
    def _recommend_optimal_scheduler(self) -> Dict[str, Any]:
        """Recommend optimal scheduler based on use case and requirements."""
        use_cases = {
            "speed_priority": {
                "recommendation": "dpm",
                "reason": "Fastest sampling with high quality",
                "alternative": "pndm"
            },
            "quality_priority": {
                "recommendation": "unipc",
                "reason": "Highest quality with flexible sampling",
                "alternative": "kdpm2"
            },
            "memory_constrained": {
                "recommendation": "euler",
                "reason": "Lowest memory usage",
                "alternative": "heun"
            },
            "production": {
                "recommendation": "dpm",
                "reason": "Best balance of speed, quality, and stability",
                "alternative": "ddim"
            },
            "research": {
                "recommendation": "unipc",
                "reason": "Most flexible and configurable",
                "alternative": "lms"
            }
        }
        
        return use_cases
    
    def demonstrate_diffusion_math(self, x_0: torch.Tensor, t: int = 500) -> Dict[str, torch.Tensor]:
        """
        Demonstrate the mathematical principles of diffusion processes.
        
        Args:
            x_0: Clean input image
            t: Timestep to demonstrate
        
        Returns:
            demonstration: Mathematical demonstration results
        """
        # Get noise schedule parameters
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        alpha_t = alphas_cumprod[t]
        beta_t = 1 - alpha_t
        
        # Forward diffusion demonstration
        noise = torch.randn_like(x_0)
        x_t_forward = torch.sqrt(alpha_t) * x_0 + torch.sqrt(beta_t) * noise
        
        # Reverse diffusion demonstration (assuming perfect noise prediction)
        x_0_reconstructed = (x_t_forward - torch.sqrt(beta_t) * noise) / torch.sqrt(alpha_t)
        
        # Demonstrate noise prediction accuracy
        noise_prediction_error = F.mse_loss(noise, torch.zeros_like(noise))  # Placeholder
        
        return {
            "original_image": x_0,
            "noisy_image": x_t_forward,
            "reconstructed_image": x_0_reconstructed,
            "noise": noise,
            "alpha_t": alpha_t,
            "beta_t": beta_t,
            "noise_prediction_error": noise_prediction_error,
            "reconstruction_error": F.mse_loss(x_0, x_0_reconstructed)
        }
    
    def batch_generate_visuals(self, prompts: List[str], seo_features_list: List[Dict[str, Any]], 
                              keywords_list: List[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate multiple visual contents in batch."""
        results = []
        
        for i, prompt in enumerate(prompts):
            seo_features = seo_features_list[i] if i < len(seo_features_list) else {}
            keywords = keywords_list[i] if keywords_list and i < len(keywords_list) else None
            
            result = self.generate_seo_visual_content(prompt, seo_features, keywords)
            results.append(result)
        
        return results
    
    def train_diffusion_step(self, clean_images: torch.Tensor, timesteps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single training step demonstrating forward and reverse diffusion processes.
        
        Args:
            clean_images: Batch of clean images [B, C, H, W]
            timesteps: Random timesteps for training [B]
        
        Returns:
            training_info: Complete training step information
        """
        try:
            # Validate inputs
            if clean_images is None or timesteps is None:
                raise ValueError("Clean images and timesteps cannot be None")
            
            if clean_images.size(0) != timesteps.size(0):
                raise ValueError("Batch size mismatch between images and timesteps")
            
            # Forward diffusion process (q(x_t | x_0)) with error handling
            try:
                noisy_images, target_noise = self.diffusion_model.forward_diffusion_process(clean_images, timesteps)
            except Exception as e:
                logger.error(f"Error in forward diffusion process: {e}")
                raise
            
            # Predict noise using UNet (reverse diffusion process) with error handling
            try:
                predicted_noise = self.diffusion_model.unet(noisy_images, timesteps).sample
            except Exception as e:
                logger.error(f"Error in UNet noise prediction: {e}")
                raise
            
            # Compute loss with error handling
            try:
                loss = self.diffusion_model.compute_loss(clean_images, timesteps, predicted_noise, target_noise)
            except Exception as e:
                logger.error(f"Error computing diffusion loss: {e}")
                raise
            
            # Demonstrate reverse diffusion step with error handling
            try:
                denoised_images = self.diffusion_model.reverse_diffusion_process(noisy_images, timesteps, predicted_noise)
            except Exception as e:
                logger.error(f"Error in reverse diffusion process: {e}")
                raise
            
            # Calculate additional metrics with error handling
            try:
                noise_prediction_error = F.mse_loss(predicted_noise, target_noise)
                reconstruction_error = F.mse_loss(clean_images, denoised_images)
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                noise_prediction_error = torch.tensor(0.0, device=clean_images.device)
                reconstruction_error = torch.tensor(0.0, device=clean_images.device)
            
            return {
                "clean_images": clean_images,
                "noisy_images": noisy_images,
                "target_noise": target_noise,
                "predicted_noise": predicted_noise,
                "denoised_images": denoised_images,
                "loss": loss,
                "timesteps": timesteps,
                "noise_prediction_error": noise_prediction_error,
                "reconstruction_error": reconstruction_error
            }
            
        except Exception as e:
            logger.error(f"Critical error in diffusion training step: {e}")
            # Return safe default values
            return {
                "clean_images": clean_images if clean_images is not None else torch.zeros(1, 3, 64, 64),
                "noisy_images": torch.zeros(1, 3, 64, 64),
                "target_noise": torch.zeros(1, 3, 64, 64),
                "predicted_noise": torch.zeros(1, 3, 64, 64),
                "denoised_images": torch.zeros(1, 3, 64, 64),
                "loss": torch.tensor(float('inf')),
                "timesteps": timesteps if timesteps is not None else torch.zeros(1),
                "noise_prediction_error": torch.tensor(float('inf')),
                "reconstruction_error": torch.tensor(float('inf')),
                "error": str(e)
            }
    
    def evaluate_diffusion_understanding(self, test_images: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive evaluation of diffusion process understanding.
        
        Args:
            test_images: Test images for evaluation
        
        Returns:
            evaluation: Complete evaluation results
        """
        results = {}
        
        # Test forward diffusion at different timesteps
        timesteps_to_test = [100, 500, 900]
        for t in timesteps_to_test:
            t_tensor = torch.full((test_images.size(0),), t, device=test_images.device, dtype=torch.long)
            noisy_imgs, noise = self.diffusion_model.forward_diffusion_process(test_images, t_tensor)
            
            # Calculate noise level
            noise_level = torch.norm(noise, dim=(1, 2, 3)).mean().item()
            image_variance = torch.var(noisy_imgs, dim=(1, 2, 3)).mean().item()
            
            results[f"timestep_{t}"] = {
                "noise_level": noise_level,
                "image_variance": image_variance,
                "noisy_images": noisy_imgs,
                "noise": noise
            }
        
        # Test reverse diffusion process
        # Start with fully noisy images
        t_max = torch.full((test_images.size(0),), self.diffusion_model.num_train_timesteps - 1, 
                          device=test_images.device, dtype=torch.long)
        fully_noisy, _ = self.diffusion_model.forward_diffusion_process(test_images, t_max)
        
        # Denoise step by step
        denoising_steps = 10
        step_size = self.diffusion_model.num_train_timesteps // denoising_steps
        
        denoising_trajectory = [fully_noisy]
        current_images = fully_noisy
        
        for step in range(denoising_steps):
            current_t = self.diffusion_model.num_train_timesteps - 1 - step * step_size
            t_tensor = torch.full((test_images.size(0),), current_t, device=test_images.device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = self.diffusion_model.unet(current_images, t_tensor).sample
            
            # Denoise step
            current_images = self.diffusion_model.reverse_diffusion_process(current_images, t_tensor, predicted_noise)
            denoising_trajectory.append(current_images)
        
        results["reverse_diffusion"] = {
            "fully_noisy": fully_noisy,
            "denoising_trajectory": denoising_trajectory,
            "final_denoised": current_images,
            "final_reconstruction_error": F.mse_loss(test_images, current_images).item()
        }
        
        # Analyze noise schedule understanding
        noise_schedule_info = self.diffusion_model.get_noise_schedule_info()
        results["noise_schedule_analysis"] = noise_schedule_info
        
        return results
    
    def visualize_diffusion_processes(self, sample_images: torch.Tensor, num_steps: int = 8) -> Dict[str, List[torch.Tensor]]:
        """
        Visualize both forward and reverse diffusion processes.
        
        Args:
            sample_images: Sample images to visualize
            num_steps: Number of steps to visualize
        
        Returns:
            visualizations: Forward and reverse process visualizations
        """
        # Forward diffusion visualization
        forward_trajectory = self.diffusion_model.sample_trajectory(sample_images, num_steps)
        
        # Reverse diffusion visualization
        # Start with the most noisy image from forward trajectory
        most_noisy = forward_trajectory[-1]
        reverse_trajectory = self.diffusion_model.denoise_trajectory(most_noisy, num_steps)
        
        return {
            "forward_diffusion": forward_trajectory,
            "reverse_diffusion": reverse_trajectory,
            "process_comparison": {
                "forward_steps": len(forward_trajectory),
                "reverse_steps": len(reverse_trajectory),
                "noise_addition_rate": "Gradual increase from clean to pure noise",
                "noise_removal_rate": "Gradual decrease from pure noise to clean"
            }
        }

class CustomSEOModel(nn.Module):
    """Custom SEO analysis model with advanced architecture, attention mechanisms, and efficient fine-tuning."""
    
    def __init__(self, config: SEOConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            output_hidden_states=True
        )
        
        # Get hidden size from transformer
        hidden_size = self.transformer.config.hidden_size
        
        # Apply LoRA to transformer layers if enabled
        if config.use_lora:
            self._apply_lora_to_transformer()
        
        # P-tuning prefix encoder
        if config.use_p_tuning:
            self.prefix_encoder = PrefixEncoder(config, hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, config.max_length)
        
        # SEO-specific attention layers
        self.seo_attention = SEOAttentionLayer(
            hidden_size, 
            config.num_attention_heads, 
            config.attention_dropout
        )
        
        # Keyword-aware attention
        self.keyword_attention = MultiHeadAttention(
            hidden_size, 
            config.num_attention_heads, 
            config.attention_dropout
        )
        
        # Custom classification head with attention
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # SEO-specific layers with attention
        self.keyword_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(hidden_size // 2, 128)
        )
        
        self.readability_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(hidden_size // 2, 64)
        )
        
        self.content_quality_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(hidden_size // 2, 96)
        )
        
        # Attention pooling for better feature extraction
        self.attention_pooling = nn.MultiheadAttention(
            hidden_size, 
            config.num_attention_heads, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _apply_lora_to_transformer(self):
        """Apply LoRA to transformer layers for efficient fine-tuning."""
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in ['query', 'key', 'value', 'dense']):
                # Apply LoRA to attention and feed-forward layers
                setattr(self.transformer, name, LoRALinear(
                    module, 
                    self.config.lora_r, 
                    self.config.lora_alpha, 
                    self.config.lora_dropout
                ))
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Create proper attention mask for transformer layers."""
        if attention_mask is None:
            return None
        
        # Convert attention mask to proper format for attention layers
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        mask = mask.expand(-1, self.config.num_attention_heads, -1, -1)  # [batch, heads, seq_len, seq_len]
        return mask
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with comprehensive SEO analysis, attention mechanisms, and efficient fine-tuning."""
        batch_size = input_ids.size(0)
        
        # Add P-tuning prefix if enabled
        if self.config.use_p_tuning:
            prefix_embeds = self.prefix_encoder(batch_size)
            prefix_attention_mask = torch.ones(batch_size, self.config.pre_seq_len, device=input_ids.device)
            
            # Concatenate prefix with input
            if attention_mask is not None:
                attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
            else:
                attention_mask = torch.cat([prefix_attention_mask, torch.ones_like(input_ids)], dim=1)
        
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Add P-tuning prefix to hidden states if enabled
        if self.config.use_p_tuning:
            hidden_states = torch.cat([prefix_embeds, hidden_states], dim=1)
        
        # Add positional encoding
        hidden_states = hidden_states.transpose(0, 1)  # [seq_len, batch, hidden_size]
        hidden_states = self.pos_encoding(hidden_states)
        hidden_states = hidden_states.transpose(0, 1)  # [batch, seq_len, hidden_size]
        
        # Create attention mask
        attn_mask = self._create_attention_mask(attention_mask)
        
        # Apply SEO-specific attention
        seo_enhanced = self.seo_attention(hidden_states, attn_mask)
        
        # Apply keyword-aware attention
        keyword_enhanced, keyword_attention_weights = self.keyword_attention(
            seo_enhanced, seo_enhanced, seo_enhanced, attn_mask
        )
        
        # Attention pooling for global representation
        pooled_output, pooling_attention_weights = self.attention_pooling(
            keyword_enhanced, keyword_enhanced, keyword_enhanced,
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        # Global average pooling
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled_output = (pooled_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_output = pooled_output.mean(dim=1)
        
        # Main classification
        logits = self.classifier(pooled_output)
        
        # SEO-specific analyses with attention-enhanced features
        keyword_features = self.keyword_analyzer(pooled_output)
        readability_features = self.readability_analyzer(pooled_output)
        quality_features = self.content_quality_analyzer(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'keyword_features': keyword_features,
            'readability_features': readability_features,
            'quality_features': quality_features,
            'hidden_states': outputs.hidden_states,
            'attention_weights': keyword_attention_weights,
            'pooling_attention_weights': pooling_attention_weights,
            'seo_enhanced_features': seo_enhanced,
            'keyword_enhanced_features': keyword_enhanced
        }
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get only trainable parameters (LoRA + P-tuning + task-specific layers)."""
        trainable_params = []
        
        # Add LoRA parameters
        if self.config.use_lora:
            for name, param in self.named_parameters():
                if 'lora' in name:
                    trainable_params.append(param)
        
        # Add P-tuning parameters
        if self.config.use_p_tuning:
            for name, param in self.named_parameters():
                if 'prefix_encoder' in name:
                    trainable_params.append(param)
        
        # Add task-specific layers
        for name, param in self.named_parameters():
            if any(key in name for key in ['classifier', 'keyword_analyzer', 'readability_analyzer', 
                                         'content_quality_analyzer', 'seo_attention', 'keyword_attention']):
                trainable_params.append(param)
        
        return trainable_params

class AdvancedTokenizer:
    """Advanced tokenizer with custom preprocessing and sequence handling."""
    
    def __init__(self, config: SEOConfig):
        self.config = config
        self.base_tokenizer = None
        self.custom_tokenizer = None
        self.special_tokens = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]',
            'seo_start': '[SEO_START]',
            'seo_end': '[SEO_END]',
            'keyword_start': '[KEYWORD_START]',
            'keyword_end': '[KEYWORD_END]'
        }
        self._initialize_tokenizers()
    
    def _initialize_tokenizers(self):
        """Initialize both base and custom tokenizers."""
        try:
            # Initialize base tokenizer from transformers
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                padding_side='right'
            )
            
            # Add special tokens if not present
            for token_name, token_value in self.special_tokens.items():
                if not hasattr(self.base_tokenizer, token_name) or getattr(self.base_tokenizer, token_name) is None:
                    setattr(self.base_tokenizer, token_name, token_value)
            
            # Initialize custom tokenizer for advanced processing
            self.custom_tokenizer = Tokenizer(models.BPE())
            
            # Configure custom tokenizer
            self.custom_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self.custom_tokenizer.decoder = decoders.ByteLevel()
            self.custom_tokenizer.post_processor = TemplateProcessing(
                single=f"{self.special_tokens['cls_token']} $A {self.special_tokens['sep_token']}",
                pair=f"{self.special_tokens['cls_token']} $A {self.special_tokens['sep_token']} $B:1 {self.special_tokens['sep_token']}:1",
                special_tokens=[
                    (self.special_tokens['cls_token'], self.base_tokenizer.convert_tokens_to_ids(self.special_tokens['cls_token'])),
                    (self.special_tokens['sep_token'], self.base_tokenizer.convert_tokens_to_ids(self.special_tokens['sep_token']))
                ]
            )
            
        except Exception as e:
            logging.error(f"Error initializing tokenizers: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for SEO analysis."""
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Handle special characters for SEO
        text = self._handle_seo_special_chars(text)
        
        # Clean HTML tags if present
        text = self._clean_html_tags(text)
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        return text
    
    def _handle_seo_special_chars(self, text: str) -> str:
        """Handle special characters commonly found in SEO content."""
        # Replace common SEO special characters
        replacements = {
            '&': ' and ',
            '|': ' or ',
            '+': ' plus ',
            '=': ' equals ',
            '#': ' number ',
            '@': ' at ',
            '%': ' percent ',
            '$': ' dollar ',
            '©': ' copyright ',
            '®': ' registered ',
            '™': ' trademark '
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _clean_html_tags(self, text: str) -> str:
        """Remove HTML tags while preserving content."""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation for better tokenization."""
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Standardize dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Add spaces around punctuation for better tokenization
        punctuation = string.punctuation.replace('-', '').replace("'", '')
        for char in punctuation:
            text = text.replace(char, f' {char} ')
        
        return text
    
    def tokenize_with_keywords(self, text: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Tokenize text with special keyword handling."""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Add keyword markers if keywords provided
        if keywords:
            processed_text = self._add_keyword_markers(processed_text, keywords)
        
        # Tokenize with base tokenizer
        tokens = self.base_tokenizer.tokenize(processed_text)
        
        # Convert to IDs
        token_ids = self.base_tokenizer.convert_tokens_to_ids(tokens)
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Truncate if necessary
        if len(token_ids) > self.config.max_length - 2:  # Account for [CLS] and [SEP]
            token_ids = token_ids[:self.config.max_length - 2]
            attention_mask = attention_mask[:self.config.max_length - 2]
        
        # Add special tokens
        token_ids = [self.base_tokenizer.cls_token_id] + token_ids + [self.base_tokenizer.sep_token_id]
        attention_mask = [1] + attention_mask + [1]
        
        # Pad if necessary
        padding_length = self.config.max_length - len(token_ids)
        if padding_length > 0:
            token_ids += [self.base_tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'tokens': tokens,
            'original_text': text,
            'processed_text': processed_text
        }
    
    def _add_keyword_markers(self, text: str, keywords: List[str]) -> str:
        """Add special markers around keywords for better attention."""
        processed_text = text
        for keyword in keywords:
            if keyword.lower() in processed_text.lower():
                # Find and mark keyword occurrences
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                processed_text = pattern.sub(
                    f"{self.special_tokens['keyword_start']} {keyword} {self.special_tokens['keyword_end']}", 
                    processed_text
                )
        return processed_text
    
    def batch_tokenize(self, texts: List[str], keywords_list: List[List[str]] = None) -> List[Dict[str, Any]]:
        """Tokenize multiple texts efficiently."""
        results = []
        
        for i, text in enumerate(texts):
            keywords = keywords_list[i] if keywords_list and i < len(keywords_list) else None
            result = self.tokenize_with_keywords(text, keywords)
            results.append(result)
        
        return results
    
    def decode_tokens(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.base_tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings."""
        return {
            name: self.base_tokenizer.convert_tokens_to_ids(token)
            for name, token in self.special_tokens.items()
        }

class SequenceProcessor:
    """Advanced sequence processing for SEO analysis."""
    
    def __init__(self, config: SEOConfig):
        self.config = config
        self.max_sequence_length = config.max_length
        self.tokenizer = AdvancedTokenizer(config)
    
    def process_sequences(self, texts: List[str], keywords_list: List[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Process multiple sequences for batch processing."""
        # Tokenize all texts
        tokenization_results = self.tokenizer.batch_tokenize(texts, keywords_list)
        
        # Extract tensors
        input_ids = []
        attention_masks = []
        
        for result in tokenization_results:
            input_ids.append(result['input_ids'])
            attention_masks.append(result['attention_mask'])
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long)
        
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_masks_tensor,
            'tokenization_results': tokenization_results
        }
    
    def create_sequence_batches(self, sequences: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Create batches from sequences."""
        batches = []
        num_sequences = sequences['input_ids'].size(0)
        
        for i in range(0, num_sequences, batch_size):
            batch = {
                'input_ids': sequences['input_ids'][i:i + batch_size],
                'attention_mask': sequences['attention_mask'][i:i + batch_size]
            }
            batches.append(batch)
        
        return batches
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int = None) -> torch.Tensor:
        """Pad sequences to the same length."""
        if max_length is None:
            max_length = self.max_sequence_length
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                # Pad with pad token ID
                padded_seq = seq + [0] * (max_length - len(seq))  # Assuming 0 is pad token ID
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences, dtype=torch.long)
    
    def create_attention_masks(self, sequences: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create attention masks for padded sequences."""
        attention_masks = (sequences != pad_token_id).long()
        return attention_masks
    
    def truncate_sequences(self, sequences: List[List[int]], max_length: int) -> List[List[int]]:
        """Truncate sequences to maximum length."""
        truncated = []
        for seq in sequences:
            if len(seq) > max_length:
                truncated.append(seq[:max_length])
            else:
                truncated.append(seq)
        return truncated

class AdvancedLLMSEOEngine:
    """Advanced SEO engine using LLMs and transformers with enhanced features and efficient fine-tuning."""
    
    def __init__(self, config: SEOConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.seo_model = None
        self.seo_pipeline = None
        self.optimizer = None
        self.scheduler = None
        # Enhanced mixed precision setup
        self.scaler = self._setup_mixed_precision()
        self.mixed_precision_dtype = self._get_optimal_mixed_precision_dtype()
        self.logger = self._setup_logging()
        self.sequence_processor = SequenceProcessor(config)
        self.diffusion_generator = AdvancedDiffusionSEOGenerator(config) if config.use_diffusion else None
        self.pipeline_manager = DiffusionPipelineManager(config) if config.use_diffusion else None
        
        # Initialize data loader manager
        self.data_loader_manager = DataLoaderManager(DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        ))
        
        # Training state
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_loss': float('inf'),
            'training_history': []
        }
        
        # Setup debugging tools
        self._setup_debugging_tools()
        
        # Setup multi-GPU training
        self._setup_multi_gpu_training()
        
        # Setup gradient accumulation
        self._setup_gradient_accumulation()
        
        # Initialize code profiler
        self.code_profiler = CodeProfiler(config)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging configuration for training progress and errors."""
        import logging.handlers
        from datetime import datetime
        import json
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", '
            '"function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # General application log file
        app_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"seo_engine_{timestamp}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(app_handler)
        
        # Training progress log file
        training_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"training_progress_{timestamp}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        training_handler.setLevel(logging.INFO)
        training_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(training_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"errors_{timestamp}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance metrics log file
        performance_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"performance_metrics_{timestamp}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(json_formatter)
        root_logger.addHandler(performance_handler)
        
        # Debug log file (only in debug mode)
        if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
            debug_handler = logging.handlers.RotatingFileHandler(
                log_dir / f"debug_{timestamp}.log",
                maxBytes=5*1024*1024,  # 5MB
                backupCount=2,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(debug_handler)
        
        # Create specialized loggers
        self._setup_specialized_loggers(log_dir, timestamp)
        
        # Log initial setup
        logger = logging.getLogger(__name__)
        logger.info("Comprehensive logging system initialized")
        logger.info(f"Log directory: {log_dir.absolute()}")
        logger.info(f"Timestamp: {timestamp}")
        
        return logger
    
    def _setup_debugging_tools(self):
        """Setup PyTorch debugging tools based on configuration."""
        try:
            # Setup autograd anomaly detection
            if self.config.enable_autograd_anomaly:
                torch.autograd.set_detect_anomaly(True)
                self.logger.info("✅ Autograd anomaly detection enabled")
                self.logger.warning("⚠️  Autograd anomaly detection will slow down training significantly")
            
            # Setup device placement debugging
            if self.config.debug_device_placement:
                self.logger.info("✅ Device placement debugging enabled")
                self.logger.info(f"Current device: {self.device}")
                if torch.cuda.is_available():
                    self.logger.info(f"CUDA device: {torch.cuda.current_device()}")
                    self.logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
                    self.logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Setup memory debugging
            if self.config.debug_memory_usage:
                self.logger.info("✅ Memory usage debugging enabled")
                if hasattr(psutil, 'virtual_memory'):
                    memory = psutil.virtual_memory()
                    self.logger.info(f"System memory: {memory.total / 1024**3:.2f} GB total, "
                                   f"{memory.available / 1024**3:.2f} GB available")
                if torch.cuda.is_available():
                    self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB total")
            
            # Log debugging configuration
            debug_options = [
                "autograd_anomaly", "autograd_profiler", "tensorboard_profiler",
                "memory_usage", "gradient_norms", "forward_pass", "backward_pass",
                "device_placement", "mixed_precision", "data_loading", "validation",
                "early_stopping", "lr_scheduling"
            ]
            
            enabled_debug_options = [
                opt for opt in debug_options 
                if getattr(self.config, f"debug_{opt.replace('_', '')}", False) or 
                   getattr(self.config, f"enable_{opt.replace('_', '')}", False)
            ]
            
            if enabled_debug_options:
                self.logger.info(f"🔧 Debugging tools enabled: {', '.join(enabled_debug_options)}")
            else:
                self.logger.info("🔧 No debugging tools enabled")
                
        except Exception as e:
            self.logger.error(f"Error setting up debugging tools: {e}")
            self.logger.warning("Continuing without debugging tools")
    
    def enable_debugging(self, debug_options: Optional[List[str]] = None):
        """Enable specific debugging options dynamically."""
        try:
            if debug_options is None:
                # Enable all debugging options
                debug_options = [
                    "autograd_anomaly", "autograd_profiler", "tensorboard_profiler",
                    "memory_usage", "gradient_norms", "forward_pass", "backward_pass",
                    "device_placement", "mixed_precision", "data_loading", "validation",
                    "early_stopping", "lr_scheduling"
                ]
            
            for option in debug_options:
                if hasattr(self.config, f"debug_{option.replace('_', '')}"):
                    setattr(self.config, f"debug_{option.replace('_', '')}", True)
                elif hasattr(self.config, f"enable_{option.replace('_', '')}"):
                    setattr(self.config, f"enable_{option.replace('_', '')}", True)
            
            # Re-setup debugging tools
            self._setup_debugging_tools()
            self.logger.info(f"✅ Debugging options enabled: {', '.join(debug_options)}")
            
        except Exception as e:
            self.logger.error(f"Error enabling debugging options: {e}")
    
    def disable_debugging(self, debug_options: Optional[List[str]] = None):
        """Disable specific debugging options dynamically."""
        try:
            if debug_options is None:
                # Disable all debugging options
                debug_options = [
                    "autograd_anomaly", "autograd_profiler", "tensorboard_profiler",
                    "memory_usage", "gradient_norms", "forward_pass", "backward_pass",
                    "device_placement", "mixed_precision", "data_loading", "validation",
                    "early_stopping", "lr_scheduling"
                ]
            
            for option in debug_options:
                if hasattr(self.config, f"debug_{option.replace('_', '')}"):
                    setattr(self.config, f"debug_{option.replace('_', '')}", False)
                elif hasattr(self.config, f"enable_{option.replace('_', '')}"):
                    setattr(self.config, f"enable_{option.replace('_', '')}", False)
            
            # Disable autograd anomaly detection
            if hasattr(self.config, 'enable_autograd_anomaly') and not self.config.enable_autograd_anomaly:
                torch.autograd.set_detect_anomaly(False)
            
            self.logger.info(f"🔧 Debugging options disabled: {', '.join(debug_options)}")
            
        except Exception as e:
            self.logger.error(f"Error disabling debugging options: {e}")
    
    def _setup_multi_gpu_training(self):
        """Setup multi-GPU training using DataParallel or DistributedDataParallel."""
        try:
            if not self.config.use_multi_gpu:
                self.logger.info("🔧 Multi-GPU training disabled")
                return
            
            if not torch.cuda.is_available():
                self.logger.warning("⚠️  CUDA not available, multi-GPU training disabled")
                return
            
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2:
                self.logger.warning(f"⚠️  Only {num_gpus} GPU(s) available, multi-GPU training disabled")
                return
            
            self.logger.info(f"🚀 Setting up multi-GPU training with {num_gpus} GPUs")
            
            if self.config.multi_gpu_strategy == "dataparallel":
                self._setup_dataparallel_training(num_gpus)
            elif self.config.multi_gpu_strategy == "distributed":
                self._setup_distributed_training(num_gpus)
            else:
                self.logger.error(f"❌ Unknown multi-GPU strategy: {self.config.multi_gpu_strategy}")
                return
            
            self.logger.info("✅ Multi-GPU training setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up multi-GPU training: {e}")
            self.logger.warning("Continuing with single-GPU training")
    
    def _setup_dataparallel_training(self, num_gpus: int):
        """Setup DataParallel training."""
        try:
            self.logger.info(f"🔧 Setting up DataParallel training with {num_gpus} GPUs")
            
            # Update batch size for multi-GPU
            original_batch_size = self.config.batch_size
            self.config.batch_size = original_batch_size * num_gpus
            self.logger.info(f"📊 Batch size adjusted: {original_batch_size} -> {self.config.batch_size} (per GPU: {original_batch_size})")
            
            # Update number of workers
            self.config.dataloader_num_workers = min(self.config.dataloader_num_workers * num_gpus, 16)
            self.logger.info(f"📊 DataLoader workers adjusted: {self.config.dataloader_num_workers}")
            
            # Set device to first GPU for DataParallel
            self.device = torch.device("cuda:0")
            self.logger.info(f"📱 Primary device set to: {self.device}")
            
            # Mark as DataParallel mode
            self.is_dataparallel = True
            self.num_gpus = num_gpus
            
            self.logger.info("✅ DataParallel training setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up DataParallel training: {e}")
            raise
    
    def _setup_distributed_training(self, num_gpus: int):
        """Setup DistributedDataParallel training."""
        try:
            self.logger.info(f"🔧 Setting up DistributedDataParallel training with {num_gpus} GPUs")
            
            # Check if distributed training is already initialized
            if torch.distributed.is_initialized():
                self.logger.info("✅ Distributed training already initialized")
                return
            
            # Set environment variables for distributed training
            os.environ['MASTER_ADDR'] = self.config.distributed_master_addr
            os.environ['MASTER_PORT'] = self.config.distributed_master_port
            os.environ['WORLD_SIZE'] = str(self.config.distributed_world_size)
            os.environ['RANK'] = str(self.config.distributed_rank)
            
            # Initialize distributed training
            torch.distributed.init_process_group(
                backend=self.config.distributed_backend,
                init_method=self.config.distributed_init_method,
                world_size=self.config.distributed_world_size,
                rank=self.config.distributed_rank
            )
            
            # Set device based on local rank
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            
            # Update batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = original_batch_size * num_gpus
            self.logger.info(f"📊 Batch size adjusted: {original_batch_size} -> {self.config.batch_size} (per GPU: {original_batch_size})")
            
            # Mark as distributed mode
            self.is_distributed = True
            self.num_gpus = num_gpus
            self.local_rank = local_rank
            
            self.logger.info(f"✅ DistributedDataParallel training setup completed on device {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error setting up DistributedDataParallel training: {e}")
            raise
    
    def _wrap_model_for_multi_gpu(self):
        """Wrap the model for multi-GPU training."""
        try:
            if not hasattr(self, 'is_dataparallel') and not hasattr(self, 'is_distributed'):
                self.logger.info("🔧 Single-GPU training mode")
                return
            
            if hasattr(self, 'is_dataparallel') and self.is_dataparallel:
                self._wrap_model_dataparallel()
            elif hasattr(self, 'is_distributed') and self.is_distributed:
                self._wrap_model_distributed()
            
        except Exception as e:
            self.logger.error(f"Error wrapping model for multi-GPU: {e}")
            self.logger.warning("Continuing with unwrapped model")
    
    def _wrap_model_dataparallel(self):
        """Wrap model with DataParallel."""
        try:
            self.logger.info("🔧 Wrapping model with DataParallel")
            
            # Move model to primary GPU
            self.seo_model = self.seo_model.to(self.device)
            
            # Wrap with DataParallel
            self.seo_model = torch.nn.DataParallel(
                self.seo_model,
                device_ids=list(range(self.num_gpus)),
                output_device=0,
                dim=0
            )
            
            self.logger.info(f"✅ Model wrapped with DataParallel on {self.num_gpus} GPUs")
            
        except Exception as e:
            self.logger.error(f"Error wrapping model with DataParallel: {e}")
            raise
    
    def _wrap_model_distributed(self):
        """Wrap model with DistributedDataParallel."""
        try:
            self.logger.info("🔧 Wrapping model with DistributedDataParallel")
            
            # Move model to local device
            self.seo_model = self.seo_model.to(self.device)
            
            # Wrap with DistributedDataParallel
            self.seo_model = torch.nn.parallel.DistributedDataParallel(
                self.seo_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=self.config.broadcast_buffers,
                bucket_cap_mb=self.config.bucket_cap_mb,
                static_graph=self.config.static_graph,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view
            )
            
            self.logger.info(f"✅ Model wrapped with DistributedDataParallel on device {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error wrapping model with DistributedDataParallel: {e}")
            raise
    
    def get_multi_gpu_status(self) -> Dict[str, Any]:
        """Get current multi-GPU training status."""
        status = {
            'use_multi_gpu': getattr(self.config, 'use_multi_gpu', False),
            'strategy': getattr(self.config, 'multi_gpu_strategy', 'none'),
            'num_gpus': getattr(self, 'num_gpus', 1),
            'is_dataparallel': getattr(self, 'is_dataparallel', False),
            'is_distributed': getattr(self, 'is_distributed', False),
            'device': str(self.device),
            'local_rank': getattr(self, 'local_rank', 0)
        }
        return status
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status including multi-GPU and gradient accumulation."""
        try:
            status = {
                'multi_gpu': self.get_multi_gpu_status(),
                'gradient_accumulation': self._get_gradient_accumulation_status(),
                'training_state': {
                    'epoch': self.training_state.get('epoch', 0),
                    'step': self.training_state.get('step', 0),
                    'epochs_completed': len(self.training_state.get('training_history', []))
                },
                'optimizer': {
                    'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else None,
                    'weight_decay': self.optimizer.param_groups[0].get('weight_decay', None) if self.optimizer else None
                },
                'scheduler': {
                    'type': type(self.scheduler).__name__ if self.scheduler else None,
                    'last_lr': self.scheduler.get_last_lr() if self.scheduler else None
                },
                'device': str(self.device),
                'mixed_precision': self.config.use_mixed_precision,
                'debugging': self.get_debugging_status()
            }
            return status
        except Exception as e:
            self.logger.error(f"Error getting training status: {e}")
            return {}
    
    def cleanup_multi_gpu(self):
        """Cleanup multi-GPU training resources."""
        try:
            if hasattr(self, 'is_distributed') and self.is_distributed:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                    self.logger.info("✅ Distributed training process group destroyed")
                
                # Reset distributed flags
                self.is_distributed = False
                self.local_rank = 0
            
            if hasattr(self, 'is_dataparallel') and self.is_dataparallel:
                # Reset DataParallel flags
                self.is_dataparallel = False
                self.num_gpus = 1
            
            self.logger.info("✅ Multi-GPU training cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during multi-GPU cleanup: {e}")
    
    def synchronize_gpus(self):
        """Synchronize all GPUs during training."""
        try:
            if hasattr(self, 'is_distributed') and self.is_distributed:
                torch.distributed.barrier()
                self.logger.debug("🔧 GPUs synchronized via distributed barrier")
            elif hasattr(self, 'is_dataparallel') and self.is_dataparallel:
                torch.cuda.synchronize()
                self.logger.debug("🔧 GPUs synchronized via CUDA synchronize")
            else:
                self.logger.debug("🔧 Single GPU, no synchronization needed")
                
        except Exception as e:
            self.logger.warning(f"GPU synchronization failed: {e}")
    
    def _setup_gradient_accumulation(self):
        """Setup gradient accumulation configuration and validation."""
        try:
            if not self.config.use_gradient_accumulation:
                self.logger.info("🔧 Gradient accumulation disabled")
                return
            
            # Validate gradient accumulation configuration
            if self.config.gradient_accumulation_steps < 1:
                raise ValueError(f"Gradient accumulation steps must be >= 1, got {self.config.gradient_accumulation_steps}")
            
            # Calculate effective batch size
            if self.config.effective_batch_size is None:
                self.config.effective_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps
            
            # Validate effective batch size
            if self.config.effective_batch_size <= 0:
                raise ValueError(f"Effective batch size must be > 0, got {self.config.effective_batch_size}")
            
            # Log gradient accumulation setup
            self.logger.info(f"🔧 Gradient accumulation enabled:")
            self.logger.info(f"   Steps: {self.config.gradient_accumulation_steps}")
            self.logger.info(f"   Base batch size: {self.config.batch_size}")
            self.logger.info(f"   Effective batch size: {self.config.effective_batch_size}")
            self.logger.info(f"   Sync gradients: {self.config.sync_gradients}")
            self.logger.info(f"   Clip before accumulation: {self.config.clip_gradients_before_accumulation}")
            self.logger.info(f"   Accumulate on CPU: {self.config.accumulate_gradients_on_cpu}")
            
            # Setup CPU accumulation if enabled
            if self.config.accumulate_gradients_on_cpu:
                self.logger.info("🔧 CPU gradient accumulation enabled - gradients will be moved to CPU during accumulation")
            
        except Exception as e:
            self.logger.error(f"Error setting up gradient accumulation: {e}")
            raise
    
    def _get_gradient_accumulation_status(self) -> Dict[str, Any]:
        """Get current gradient accumulation status."""
        try:
            status = {
                'enabled': self.config.use_gradient_accumulation,
                'steps': self.config.gradient_accumulation_steps if self.config.use_gradient_accumulation else 1,
                'effective_batch_size': self.config.effective_batch_size,
                'sync_gradients': self.config.sync_gradients,
                'clip_before_accumulation': self.config.clip_gradients_before_accumulation,
                'accumulate_on_cpu': self.config.accumulate_gradients_on_cpu
            }
            return status
        except Exception as e:
            self.logger.error(f"Error getting gradient accumulation status: {e}")
            return {}
    
    def _validate_gradient_accumulation_batch(self, batch_size: int) -> bool:
        """Validate that batch size is compatible with gradient accumulation."""
        try:
            if not self.config.use_gradient_accumulation:
                return True
            
            # Check if batch size is compatible with accumulation steps
            if batch_size != self.config.batch_size:
                self.logger.warning(f"⚠️  Batch size mismatch: expected {self.config.batch_size}, got {batch_size}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating gradient accumulation batch: {e}")
            return False
    
    def _setup_mixed_precision(self) -> Optional[amp.GradScaler]:
        """Setup enhanced mixed precision training with optimal configuration."""
        try:
            if not self.config.use_mixed_precision:
                self.logger.info("🔧 Mixed precision disabled")
                return None
            
            # Validate device compatibility
            if self.device.type != "cuda":
                self.logger.warning("⚠️  Mixed precision requires CUDA device. Disabling mixed precision.")
                return None
            
            # Check CUDA compute capability for optimal dtype selection
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()
                self.logger.info(f"🔧 CUDA compute capability: {compute_capability[0]}.{compute_capability[1]}")
                
                # Enable bfloat16 for Ampere+ GPUs (compute capability >= 8.0)
                if compute_capability[0] >= 8:
                    self.logger.info("🔧 Ampere+ GPU detected - bfloat16 will be used for optimal performance")
                else:
                    self.logger.info("🔧 Pre-Ampere GPU detected - float16 will be used")
            
            # Create gradient scaler with enhanced configuration
            scaler = None
            if self.config.mixed_precision_grad_scaler:
                scaler = amp.GradScaler(
                    init_scale=self.config.mixed_precision_grad_scaler_init_scale,
                    growth_factor=self.config.mixed_precision_grad_scaler_growth_factor,
                    backoff_factor=self.config.mixed_precision_grad_scaler_backoff_factor,
                    growth_interval=self.config.mixed_precision_grad_scaler_growth_interval,
                    enabled=self.config.mixed_precision_grad_scaler_enabled
                )
                self.logger.info("🔧 Enhanced gradient scaler configured")
            
            # Log mixed precision configuration
            self.logger.info(f"🔧 Mixed precision enabled:")
            self.logger.info(f"   Dtype: {self.config.mixed_precision_dtype}")
            self.logger.info(f"   Memory efficient: {self.config.mixed_precision_memory_efficient}")
            self.logger.info(f"   Cast model: {self.config.mixed_precision_cast_model}")
            self.logger.info(f"   Cast inputs: {self.config.mixed_precision_cast_inputs}")
            self.logger.info(f"   Cast outputs: {self.config.mixed_precision_cast_outputs}")
            self.logger.info(f"   Autocast mode: {self.config.mixed_precision_autocast_mode}")
            self.logger.info(f"   Gradient scaler: {self.config.mixed_precision_grad_scaler}")
            
            return scaler
            
        except Exception as e:
            self.logger.error(f"Error setting up mixed precision: {e}")
            return None
    
    def _get_optimal_mixed_precision_dtype(self) -> torch.dtype:
        """Get optimal mixed precision dtype based on hardware and configuration."""
        try:
            if not self.config.use_mixed_precision:
                return torch.float32
            
            if self.config.mixed_precision_dtype == "auto":
                if torch.cuda.is_available():
                    compute_capability = torch.cuda.get_device_capability()
                    
                    # Use bfloat16 for Ampere+ GPUs (compute capability >= 8.0)
                    if compute_capability[0] >= 8:
                        if torch.cuda.is_bf16_supported():
                            return torch.bfloat16
                        else:
                            self.logger.warning("⚠️  bfloat16 not supported on this GPU, falling back to float16")
                            return torch.float16
                    else:
                        return torch.float16
                else:
                    return torch.float32
            elif self.config.mixed_precision_dtype == "bfloat16":
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    self.logger.warning("⚠️  bfloat16 not supported, falling back to float16")
                    return torch.float16
            elif self.config.mixed_precision_dtype == "float16":
                return torch.float16
            else:
                return torch.float32
                
        except Exception as e:
            self.logger.error(f"Error determining optimal mixed precision dtype: {e}")
            return torch.float32
    
    def get_debugging_status(self) -> Dict[str, bool]:
        """Get current debugging configuration status."""
        try:
            debug_options = [
                "autograd_anomaly", "autograd_profiler", "tensorboard_profiler",
                "memory_usage", "gradient_norms", "forward_pass", "backward_pass",
                "device_placement", "mixed_precision", "data_loading", "validation",
                "early_stopping", "lr_scheduling"
            ]
            
            status = {}
            for option in debug_options:
                if hasattr(self.config, f"debug_{option.replace('_', '')}"):
                    status[option] = getattr(self.config, f"debug_{option.replace('_', '')}")
                elif hasattr(self.config, f"enable_{option.replace('_', '')}"):
                    status[option] = getattr(self.config, f"enable_{option.replace('_', '')}")
                else:
                    status[option] = False
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting debugging status: {e}")
            return {}
    
    def get_mixed_precision_status(self) -> Dict[str, Any]:
        """Get comprehensive mixed precision training status and performance metrics."""
        try:
            status = {
                'enabled': self.config.use_mixed_precision,
                'dtype': str(self.mixed_precision_dtype) if hasattr(self, 'mixed_precision_dtype') else 'unknown',
                'gradient_scaler': {
                    'enabled': self.config.mixed_precision_grad_scaler,
                    'available': self.scaler is not None,
                    'init_scale': self.config.mixed_precision_grad_scaler_init_scale,
                    'growth_factor': self.config.mixed_precision_grad_scaler_growth_factor,
                    'backoff_factor': self.config.mixed_precision_grad_scaler_backoff_factor,
                    'growth_interval': self.config.mixed_precision_grad_scaler_growth_interval
                },
                'autocast': {
                    'enabled': self.config.mixed_precision_autocast_enabled,
                    'dtype': self.config.mixed_precision_autocast_dtype,
                    'cache_enabled': self.config.mixed_precision_autocast_cache_enabled,
                    'fast_dtype': self.config.mixed_precision_autocast_fast_dtype,
                    'fallback_dtype': self.config.mixed_precision_autocast_fallback_dtype
                },
                'casting': {
                    'model': self.config.mixed_precision_cast_model,
                    'inputs': self.config.mixed_precision_cast_inputs,
                    'outputs': self.config.mixed_precision_cast_outputs
                },
                'memory_efficient': self.config.mixed_precision_memory_efficient,
                'hardware_support': {
                    'cuda_available': torch.cuda.is_available(),
                    'compute_capability': torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
                    'bf16_supported': torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None
                }
            }
            
            # Add performance metrics if available
            if hasattr(self, 'training_state') and 'training_history' in self.training_state:
                training_history = self.training_state['training_history']
                if training_history:
                    # Calculate mixed precision performance metrics
                    mixed_precision_epochs = [epoch for epoch in training_history if epoch.get('mixed_precision_enabled', False)]
                    if mixed_precision_epochs:
                        status['performance'] = {
                            'total_epochs': len(training_history),
                            'mixed_precision_epochs': len(mixed_precision_epochs),
                            'avg_loss_with_mp': np.mean([epoch.get('train_loss', 0) for epoch in mixed_precision_epochs]),
                            'avg_loss_without_mp': np.mean([epoch.get('train_loss', 0) for epoch in training_history if not epoch.get('mixed_precision_enabled', False)])
                        }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting mixed precision status: {e}")
            return {}
    
    def enable_mixed_precision(self, dtype: str = "auto", memory_efficient: bool = True) -> bool:
        """Dynamically enable mixed precision training with specified configuration."""
        try:
            if not torch.cuda.is_available():
                self.logger.warning("⚠️  Mixed precision requires CUDA device")
                return False
            
            # Update configuration
            self.config.use_mixed_precision = True
            self.config.mixed_precision_dtype = dtype
            self.config.mixed_precision_memory_efficient = memory_efficient
            
            # Reinitialize mixed precision
            self.scaler = self._setup_mixed_precision()
            self.mixed_precision_dtype = self._get_optimal_mixed_precision_dtype()
            
            if self.scaler:
                self.logger.info(f"✅ Mixed precision enabled with dtype: {self.mixed_precision_dtype}")
                return True
            else:
                self.logger.error("❌ Failed to enable mixed precision")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enabling mixed precision: {e}")
            return False
    
    def disable_mixed_precision(self) -> bool:
        """Dynamically disable mixed precision training."""
        try:
            self.config.use_mixed_precision = False
            self.scaler = None
            self.mixed_precision_dtype = torch.float32
            self.logger.info("✅ Mixed precision disabled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disabling mixed precision: {e}")
            return False
    
    def optimize_mixed_precision_for_hardware(self) -> Dict[str, Any]:
        """Automatically optimize mixed precision settings for current hardware."""
        try:
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}
            
            compute_capability = torch.cuda.get_device_capability()
            device_name = torch.cuda.get_device_name()
            
            optimizations = {
                'device_name': device_name,
                'compute_capability': f"{compute_capability[0]}.{compute_capability[1]}",
                'recommended_dtype': 'bfloat16' if compute_capability[0] >= 8 else 'float16',
                'bf16_supported': torch.cuda.is_bf16_supported(),
                'recommendations': []
            }
            
            # Ampere+ GPUs (compute capability >= 8.0)
            if compute_capability[0] >= 8:
                if torch.cuda.is_bf16_supported():
                    optimizations['recommendations'].append("Use bfloat16 for optimal performance")
                    optimizations['recommendations'].append("Enable memory-efficient mixed precision")
                else:
                    optimizations['recommendations'].append("bfloat16 not supported, use float16")
            
            # Pre-Ampere GPUs
            else:
                optimizations['recommendations'].append("Use float16 for optimal performance")
                optimizations['recommendations'].append("Consider memory-efficient settings")
            
            # General recommendations
            optimizations['recommendations'].extend([
                "Enable autocast cache for repeated operations",
                "Use gradient scaler for stable training",
                "Monitor memory usage during training"
            ])
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Error optimizing mixed precision for hardware: {e}")
            return {"error": str(e)}
    
    def profile_model_performance(self, dataloader: DataLoader, num_batches: int = 10) -> Dict[str, Any]:
        """Profile model performance using PyTorch's profiler."""
        try:
            if not self.config.enable_autograd_profiler:
                self.logger.warning("Autograd profiler not enabled. Enable it first with enable_debugging(['autograd_profiler'])")
                return {}
            
            self.logger.info(f"🔍 Starting model performance profiling for {num_batches} batches")
            
            # Setup profiler
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=num_batches,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True
            )
            
            profiler.start()
            
            self.seo_model.eval()
            batch_count = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_count >= num_batches:
                        break
                    
                    try:
                        if isinstance(batch, dict) and 'input_ids' in batch:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            
                            outputs = self.seo_model(input_ids, attention_mask)
                            loss = F.cross_entropy(outputs, labels)
                            
                            batch_count += 1
                            profiler.step()
                            
                    except Exception as e:
                        self.log_error(e, "Profiling batch", "profile_model_performance", {"batch_idx": batch_idx})
                        continue
            
            profiler.stop()
            
            # Get profiling results
            profiler_results = {
                'total_events': profiler.total_events,
                'cpu_events': profiler.cpu_events,
                'cuda_events': profiler.cuda_events,
                'key_averages': profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            }
            
            self.logger.info("✅ Model performance profiling completed")
            self.logger.info(f"   Total events: {profiler.total_events}")
            self.logger.info(f"   CPU events: {profiler.cpu_events}")
            self.logger.info(f"   CUDA events: {profiler.cuda_events}")
            
            return profiler_results
            
        except Exception as e:
            self.log_error(e, "Model performance profiling", "profile_model_performance")
            return {}
    
    def _setup_specialized_loggers(self, log_dir: Path, timestamp: str):
        """Setup specialized loggers for different components."""
        import logging.handlers
        
        # Training progress logger
        training_logger = logging.getLogger("training_progress")
        training_logger.setLevel(logging.INFO)
        training_logger.handlers.clear()
        
        training_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"training_detailed_{timestamp}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        training_file_handler.setLevel(logging.INFO)
        training_file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] TRAINING - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        training_logger.addHandler(training_file_handler)
        
        # Model performance logger
        model_logger = logging.getLogger("model_performance")
        model_logger.setLevel(logging.INFO)
        model_logger.handlers.clear()
        
        model_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"model_performance_{timestamp}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        model_file_handler.setLevel(logging.INFO)
        model_file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] MODEL - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        model_logger.addHandler(model_file_handler)
        
        # Data loading logger
        data_logger = logging.getLogger("data_loading")
        data_logger.setLevel(logging.INFO)
        data_logger.handlers.clear()
        
        data_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"data_loading_{timestamp}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        data_file_handler.setLevel(logging.INFO)
        data_file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] DATA - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        data_logger.addHandler(data_file_handler)
        
        # Error tracking logger
        error_tracker = logging.getLogger("error_tracker")
        error_tracker.setLevel(logging.ERROR)
        error_tracker.handlers.clear()
        
        error_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"error_tracking_{timestamp}.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] ERROR - %(levelname)s - %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        error_tracker.addHandler(error_file_handler)
    
    def log_training_progress(self, epoch: int, step: int, loss: float, learning_rate: float, 
                            validation_loss: Optional[float] = None, metrics: Optional[Dict[str, float]] = None):
        """Log comprehensive training progress information."""
        training_logger = logging.getLogger("training_progress")
        
        # Basic training info
        training_logger.info(f"Epoch {epoch}, Step {step}: Loss={loss:.6f}, LR={learning_rate:.2e}")
        
        # Validation info if available
        if validation_loss is not None:
            training_logger.info(f"Validation Loss: {validation_loss:.6f}")
        
        # Additional metrics if available
        if metrics:
            metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            training_logger.info(f"Metrics: {metrics_str}")
        
        # Performance metrics logger for structured data
        performance_logger = logging.getLogger("model_performance")
        performance_data = {
            "epoch": epoch,
            "step": step,
            "training_loss": loss,
            "learning_rate": learning_rate,
            "validation_loss": validation_loss,
            "metrics": metrics or {},
            "timestamp": time.time()
        }
        performance_logger.info(json.dumps(performance_data))
    
    def log_model_performance(self, operation: str, duration: float, memory_usage: Optional[float] = None,
                            gpu_utilization: Optional[float] = None, additional_metrics: Optional[Dict[str, Any]] = None):
        """Log model performance metrics."""
        model_logger = logging.getLogger("model_performance")
        
        # Basic performance info
        model_logger.info(f"{operation} completed in {duration:.4f}s")
        
        # Memory and GPU info if available
        if memory_usage is not None:
            model_logger.info(f"Memory usage: {memory_usage:.2f}MB")
        if gpu_utilization is not None:
            model_logger.info(f"GPU utilization: {gpu_utilization:.2f}%")
        
        # Additional metrics if available
        if additional_metrics:
            metrics_str = ", ".join([f"{k}={v}" for k, v in additional_metrics.items()])
            model_logger.info(f"Additional metrics: {metrics_str}")
        
        # Performance metrics logger for structured data
        performance_logger = logging.getLogger("model_performance")
        performance_data = {
            "operation": operation,
            "duration": duration,
            "memory_usage_mb": memory_usage,
            "gpu_utilization_percent": gpu_utilization,
            "additional_metrics": additional_metrics or {},
            "timestamp": time.time()
        }
        performance_logger.info(json.dumps(performance_data))
    
    def log_data_loading(self, operation: str, dataset_size: int, batch_size: int, 
                        duration: float, memory_usage: Optional[float] = None):
        """Log data loading operations and performance."""
        data_logger = logging.getLogger("data_loading")
        
        # Basic data loading info
        data_logger.info(f"{operation}: Dataset size={dataset_size}, Batch size={batch_size}, Duration={duration:.4f}s")
        
        # Memory info if available
        if memory_usage is not None:
            data_logger.info(f"Memory usage: {memory_usage:.2f}MB")
        
        # Performance metrics logger for structured data
        performance_logger = logging.getLogger("model_performance")
        performance_data = {
            "operation": f"data_loading_{operation}",
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "duration": duration,
            "memory_usage_mb": memory_usage,
            "timestamp": time.time()
        }
        performance_logger.info(json.dumps(performance_data))
    
    def log_error(self, error: Exception, context: str = "", operation: str = "", 
                 additional_info: Optional[Dict[str, Any]] = None):
        """Log errors with comprehensive context and tracking."""
        error_logger = logging.getLogger("error_tracker")
        
        # Basic error info
        error_logger.error(f"Error in {operation}: {type(error).__name__}: {str(error)}")
        error_logger.error(f"Context: {context}")
        
        # Additional info if available
        if additional_info:
            info_str = ", ".join([f"{k}={v}" for k, v in additional_info.items()])
            error_logger.error(f"Additional info: {info_str}")
        
        # Log to main error file as well
        main_logger = logging.getLogger(__name__)
        main_logger.error(f"Error in {operation}: {str(error)}", exc_info=True)
        
        # Performance metrics logger for error tracking
        performance_logger = logging.getLogger("model_performance")
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation,
            "context": context,
            "additional_info": additional_info or {},
            "timestamp": time.time()
        }
        performance_logger.error(json.dumps(error_data))
    
    def log_training_summary(self, total_epochs: int, total_steps: int, final_loss: float, 
                           best_loss: float, training_duration: float, early_stopping_triggered: bool = False):
        """Log comprehensive training summary."""
        training_logger = logging.getLogger("training_progress")
        
        # Training summary
        training_logger.info("=" * 80)
        training_logger.info("TRAINING SUMMARY")
        training_logger.info("=" * 80)
        training_logger.info(f"Total epochs: {total_epochs}")
        training_logger.info(f"Total steps: {total_steps}")
        training_logger.info(f"Final loss: {final_loss:.6f}")
        training_logger.info(f"Best loss: {best_loss:.6f}")
        training_logger.info(f"Training duration: {training_duration:.2f}s")
        training_logger.info(f"Early stopping triggered: {early_stopping_triggered}")
        training_logger.info("=" * 80)
        
        # Performance metrics logger for structured data
        performance_logger = logging.getLogger("model_performance")
        summary_data = {
            "training_summary": {
                "total_epochs": total_epochs,
                "total_steps": total_steps,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "training_duration": training_duration,
                "early_stopping_triggered": early_stopping_triggered
            },
            "timestamp": time.time()
        }
        performance_logger.info(json.dumps(summary_data))
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters and configuration."""
        training_logger = logging.getLogger("training_progress")
        
        training_logger.info("=" * 80)
        training_logger.info("HYPERPARAMETERS")
        training_logger.info("=" * 80)
        
        for key, value in config.items():
            training_logger.info(f"{key}: {value}")
        
        training_logger.info("=" * 80)
        
        # Performance metrics logger for structured data
        performance_logger = logging.getLogger("model_performance")
        config_data = {
            "hyperparameters": config,
            "timestamp": time.time()
        }
        performance_logger.info(json.dumps(config_data))
    
    async def initialize_models(self):
        """Initialize all required models asynchronously with error handling."""
        self.logger.info("Initializing advanced SEO models with attention mechanisms and efficient fine-tuning...")
        
        try:
            # Initialize advanced tokenizer
            self.tokenizer = AdvancedTokenizer(self.config)
            
            # Initialize custom model with attention and efficient fine-tuning
            self.seo_model = CustomSEOModel(self.config).to(self.device)
            
            # Enable mixed precision if available
            if self.config.use_mixed_precision and self.device.type == "cuda":
                self.seo_model = self.seo_model.half()
            
            # Wrap model for multi-GPU training
            self._wrap_model_for_multi_gpu()
            
            # Initialize optimizer and scheduler
            self._initialize_optimizer()
            self._initialize_scheduler()
            
            # Initialize SEO analysis pipeline
            self.seo_pipeline = pipeline(
                "text-classification",
                model=self.seo_model,
                tokenizer=self.tokenizer.base_tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            # Log trainable parameters
            trainable_params = self.seo_model.get_trainable_parameters()
            total_params = sum(p.numel() for p in self.seo_model.parameters())
            trainable_params_count = sum(p.numel() for p in trainable_params)
            
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)")
            self.logger.info("Models initialized successfully with attention mechanisms and efficient fine-tuning")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def _initialize_optimizer(self):
        """Initialize optimizer for training."""
        if not self.seo_model:
            raise ValueError("SEO model must be initialized before creating optimizer")
        
        # Get trainable parameters
        trainable_params = self.seo_model.get_trainable_parameters()
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return self.optimizer
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler with multiple options."""
        if not self.optimizer:
            raise ValueError("Optimizer must be initialized before creating scheduler")
        
        # Calculate total training steps
        total_steps = self.config.num_epochs * (1000 // self.config.batch_size)  # Estimate
        
        if self.config.lr_scheduler == "cosine":
            params = self.config.lr_scheduler_params.get("cosine", {})
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
                **params
            )
        elif self.config.lr_scheduler == "linear":
            params = self.config.lr_scheduler_params.get("linear", {})
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
                **params
            )
        elif self.config.lr_scheduler == "exponential":
            params = self.config.lr_scheduler_params.get("exponential", {"gamma": 0.95})
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=params["gamma"]
            )
        elif self.config.lr_scheduler == "step":
            params = self.config.lr_scheduler_params.get("step", {"step_size": 30, "gamma": 0.1})
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=params["step_size"], gamma=params["gamma"]
            )
        elif self.config.lr_scheduler == "plateau":
            params = self.config.lr_scheduler_params.get("plateau", {
                "mode": "min", "factor": 0.5, "patience": 3, "min_lr": 1e-7
            })
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=params["mode"],
                factor=params["factor"],
                patience=params["patience"],
                min_lr=params["min_lr"]
            )
        else:
            # Default to cosine
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
        
        return self.scheduler
    
    @torch.no_grad()
    def analyze_seo_score(self, text: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Analyze SEO score with enhanced metrics, GPU optimization, and attention analysis."""
        # Profile model inference
        with self.code_profiler.profile_operation("analyze_seo_score", "model_inference"):
            try:
                # Validate input
                if not text or not text.strip():
                    raise ValueError("Text input cannot be empty")
            
            # Use advanced tokenization with error handling
            try:
                tokenization_result = self.tokenizer.tokenize_with_keywords(text, keywords)
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                raise
            
            # Convert to tensors with error handling
            try:
                inputs = {
                    'input_ids': torch.tensor([tokenization_result['input_ids']], dtype=torch.long).to(self.device),
                    'attention_mask': torch.tensor([tokenization_result['attention_mask']], dtype=torch.long).to(self.device)
                }
            except Exception as e:
                logger.error(f"Tensor conversion error: {e}")
                raise
            
            # Use mixed precision for inference with error handling
            try:
                # Enhanced mixed precision inference with optimal configuration
                if self.config.use_mixed_precision:
                    autocast_kwargs = {}
                    
                    # Configure autocast based on settings
                    if self.config.mixed_precision_autocast_dtype == "auto":
                        autocast_kwargs['dtype'] = self.mixed_precision_dtype
                    elif self.config.mixed_precision_autocast_dtype == "bfloat16":
                        autocast_kwargs['dtype'] = torch.bfloat16
                    elif self.config.mixed_precision_autocast_dtype == "float16":
                        autocast_kwargs['dtype'] = torch.float16
                    
                    # Enable cache if configured
                    if self.config.mixed_precision_autocast_cache_enabled:
                        autocast_kwargs['cache_enabled'] = True
                    
                    # Use enhanced autocast context
                    with amp.autocast(**autocast_kwargs):
                        # Cast inputs if configured
                        if self.config.mixed_precision_cast_inputs:
                            for key, value in inputs.items():
                                if isinstance(value, torch.Tensor):
                                    inputs[key] = value.to(dtype=self.mixed_precision_dtype)
                        
                        outputs = self.seo_model(**inputs)
                        
                        # Cast outputs if configured
                        if self.config.mixed_precision_cast_outputs:
                            for key, value in outputs.items():
                                if isinstance(value, torch.Tensor):
                                    outputs[key] = value.to(dtype=torch.float32)
                else:
                    with torch.no_grad():
                        outputs = self.seo_model(**inputs)
                    logits = outputs['logits']
                    probabilities = F.softmax(logits, dim=-1)
                    seo_score = probabilities[0][1].item()
                    
                    # Extract additional features with error handling
                    try:
                        keyword_features = outputs['keyword_features'].cpu().numpy()
                        readability_features = outputs['keyword_features'].cpu().numpy()
                        quality_features = outputs['keyword_features'].cpu().numpy()
                    except Exception as e:
                        logger.error(f"Feature extraction error: {e}")
                        # Provide default features
                        keyword_features = np.zeros((1, 10))
                        readability_features = np.zeros((1, 10))
                        quality_features = np.zeros((1, 10))
                    
                    # Extract attention weights for analysis with error handling
                    try:
                        attention_weights = outputs['attention_weights'].cpu().numpy()
                        pooling_attention_weights = outputs['attention_weights'].cpu().numpy()
                    except Exception as e:
                        logger.error(f"Attention weights extraction error: {e}")
                        # Provide default attention weights
                        attention_weights = np.ones((1, 10, 10))
                        pooling_attention_weights = np.ones((1, 10, 10))
            except Exception as e:
                logger.error(f"Model inference error: {e}")
                raise
            
            # Generate analysis with error handling
            try:
                return {
                    "seo_score": seo_score,
                    "confidence": max(probabilities[0]).item(),
                    "keyword_analysis": self._analyze_keyword_features(keyword_features[0]),
                    "readability_analysis": self._analyze_readability_features(readability_features[0]),
                    "quality_analysis": self._analyze_quality_features(quality_features[0]),
                    "attention_analysis": self._analyze_attention_patterns(attention_weights[0], text),
                    "detailed_analysis": self._generate_comprehensive_analysis(text, seo_score),
                    "tokenization_info": {
                        "num_tokens": len(tokenization_result['tokens']),
                        "processed_text": tokenization_result['processed_text'][:100] + "..." if len(tokenization_result['processed_text']) > 100 else tokenization_result['processed_text']
                    }
                }
            except Exception as e:
                logger.error(f"Analysis generation error: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error analyzing SEO: {e}")
            return {"error": str(e), "error_type": type(e).__name__}
    
    def _analyze_attention_patterns(self, attention_weights: np.ndarray, text: str) -> Dict[str, Any]:
        """Analyze attention patterns for SEO insights."""
        try:
            # Validate inputs
            if attention_weights is None or attention_weights.size == 0:
                raise ValueError("Attention weights cannot be empty")
            
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Get tokens for analysis with error handling
            try:
                tokens = self.tokenizer.tokenize(text)[:attention_weights.shape[0]]
            except Exception as e:
                logger.error(f"Tokenization error in attention analysis: {e}")
                tokens = []
            
            # Calculate attention statistics with error handling
            try:
                mean_attention = np.mean(attention_weights, axis=0)  # Average across heads
                max_attention = np.max(attention_weights, axis=0)    # Maximum attention per position
                
                # Find most attended positions
                top_attention_positions = np.argsort(mean_attention)[-5:]  # Top 5 positions
                
                # Analyze attention distribution
                attention_entropy = -np.sum(mean_attention * np.log(mean_attention + 1e-8))
            except Exception as e:
                logger.error(f"Attention statistics calculation error: {e}")
                # Provide default values
                mean_attention = np.ones(attention_weights.shape[1])
                max_attention = np.ones(attention_weights.shape[1])
                top_attention_positions = [0, 1, 2, 3, 4]
                attention_entropy = 0.0
            
            return {
                "attention_entropy": float(attention_entropy),
                "attention_concentration": float(np.max(mean_attention)),
                "attention_diversity": float(1.0 - np.max(mean_attention)),
                "top_attended_positions": [int(pos) for pos in top_attention_positions],
                "attention_weights_shape": attention_weights.shape,
                "mean_attention_score": float(np.mean(mean_attention))
            }
            
        except Exception as e:
            logger.error(f"Error in attention pattern analysis: {e}")
            return {
                "attention_entropy": 0.0,
                "attention_concentration": 0.0,
                "attention_diversity": 0.0,
                "top_attended_positions": [0, 1, 2, 3, 4],
                "attention_weights_shape": (0, 0),
                "mean_attention_score": 0.0,
                "error": str(e)
            }
    
    def _analyze_keyword_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze keyword-related features."""
        try:
            # Validate input
            if features is None or features.size == 0:
                raise ValueError("Features cannot be empty")
            
            # Calculate features with error handling
            try:
                mean_features = float(np.mean(features))
                std_features = float(np.std(features))
                
                # Determine optimization level
                if mean_features > 0.7:
                    optimization_level = "high"
                elif mean_features > 0.4:
                    optimization_level = "medium"
                else:
                    optimization_level = "low"
                    
            except Exception as e:
                logger.error(f"Feature calculation error: {e}")
                mean_features = 0.0
                std_features = 0.0
                optimization_level = "low"
            
            return {
                "keyword_density_score": mean_features,
                "keyword_variety_score": std_features,
                "keyword_optimization_level": optimization_level
            }
            
        except Exception as e:
            logger.error(f"Error in keyword feature analysis: {e}")
            return {
                "keyword_density_score": 0.0,
                "keyword_variety_score": 0.0,
                "keyword_optimization_level": "low",
                "error": str(e)
            }
    
    def _analyze_readability_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze readability features."""
        try:
            # Validate input
            if features is None or features.size == 0:
                raise ValueError("Features cannot be empty")
            
            # Calculate features with error handling
            try:
                mean_features = float(np.mean(features))
                std_features = float(np.std(features))
                
                # Determine complexity level
                if mean_features > 0.6:
                    complexity_level = "simple"
                elif mean_features > 0.3:
                    complexity_level = "moderate"
                else:
                    complexity_level = "complex"
                    
            except Exception as e:
                logger.error(f"Readability feature calculation error: {e}")
                mean_features = 0.0
                std_features = 0.0
                complexity_level = "complex"
            
            return {
                "readability_score": mean_features,
                "complexity_level": complexity_level,
                "sentence_structure": std_features
            }
            
        except Exception as e:
            logger.error(f"Error in readability feature analysis: {e}")
            return {
                "readability_score": 0.0,
                "complexity_level": "complex",
                "sentence_structure": 0.0,
                "error": str(e)
            }
    
    def _analyze_quality_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze content quality features."""
        try:
            # Validate input
            if features is None or features.size == 0:
                raise ValueError("Features cannot be empty")
            
            # Calculate features with error handling
            try:
                mean_features = float(np.mean(features))
                max_features = float(np.max(features))
                std_features = float(np.std(features))
                consistency_score = float(1 - std_features)
                
                # Ensure consistency score is within valid range
                consistency_score = max(0.0, min(1.0, consistency_score))
                
            except Exception as e:
                logger.error(f"Quality feature calculation error: {e}")
                mean_features = 0.0
                max_features = 0.0
                consistency_score = 0.0
            
            return {
                "content_quality_score": mean_features,
                "engagement_potential": max_features,
                "consistency_score": consistency_score
            }
            
        except Exception as e:
            logger.error(f"Error in quality feature analysis: {e}")
            return {
                "content_quality_score": 0.0,
                "engagement_potential": 0.0,
                "consistency_score": 0.0,
                "error": str(e)
            }
    
    def _generate_comprehensive_analysis(self, text: str, score: float) -> Dict[str, Any]:
        """Generate comprehensive SEO analysis with advanced metrics."""
        analysis = {
            "keyword_density": self._calculate_advanced_keyword_density(text),
            "readability_score": self._calculate_enhanced_readability(text),
            "content_length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "recommendations": self._generate_advanced_recommendations(text, score),
            "technical_seo": self._analyze_technical_seo(text),
            "semantic_analysis": self._perform_semantic_analysis(text)
        }
        
        return analysis
    
    def _calculate_advanced_keyword_density(self, text: str) -> Dict[str, float]:
        """Calculate advanced keyword density metrics."""
        words = text.lower().split()
        if not words:
            return {"density": 0.0, "variety": 0.0, "frequency": 0.0}
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        total_words = len(words)
        unique_words = len(word_freq)
        
        return {
            "density": unique_words / total_words,
            "variety": unique_words / total_words,
            "frequency": max(word_freq.values()) / total_words if word_freq else 0.0
        }
    
    def _calculate_enhanced_readability(self, text: str) -> Dict[str, float]:
        """Calculate enhanced readability metrics."""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return {"flesch_score": 0.0, "grade_level": 0.0, "complexity": 0.0}
        
        # Flesch Reading Ease
        flesch_score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
        
        # Flesch-Kincaid Grade Level
        grade_level = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
        
        return {
            "flesch_score": max(0, min(100, flesch_score)),
            "grade_level": max(0, grade_level),
            "complexity": min(1.0, grade_level / 20.0)
        }
    
    def _generate_advanced_recommendations(self, text: str, score: float) -> List[str]:
        """Generate advanced SEO recommendations."""
        recommendations = []
        
        if score < 0.3:
            recommendations.extend([
                "Improve keyword optimization and density",
                "Increase content length to at least 300 words",
                "Enhance readability with shorter sentences",
                "Add more relevant internal and external links",
                "Optimize meta descriptions and title tags"
            ])
        elif score < 0.7:
            recommendations.extend([
                "Fine-tune keyword placement in headings",
                "Improve content structure with better headings",
                "Add more engaging multimedia content",
                "Optimize for featured snippets",
                "Enhance user engagement metrics"
            ])
        else:
            recommendations.extend([
                "Maintain current optimization level",
                "Focus on user experience improvements",
                "Monitor competitor strategies",
                "Consider advanced technical SEO",
                "Optimize for voice search queries"
            ])
        
        return recommendations
    
    def _analyze_technical_seo(self, text: str) -> Dict[str, Any]:
        """Analyze technical SEO aspects."""
        return {
            "has_headings": bool(re.search(r'<h[1-6]>', text, re.IGNORECASE)),
            "has_links": bool(re.search(r'<a\s+href', text, re.IGNORECASE)),
            "has_images": bool(re.search(r'<img', text, re.IGNORECASE)),
            "has_lists": bool(re.search(r'<[uo]l>', text, re.IGNORECASE)),
            "content_structure": "good" if len(text.split()) > 300 else "needs_improvement"
        }
    
    def _perform_semantic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform semantic analysis of content."""
        words = text.lower().split()
        unique_words = set(words)
        
        return {
            "vocabulary_richness": len(unique_words) / len(words) if words else 0.0,
            "content_uniqueness": len(unique_words) / len(words) if words else 0.0,
            "semantic_coherence": self._calculate_semantic_coherence(text)
        }
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence score."""
        # Simple implementation - can be enhanced with embeddings
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 1.0
        
        # Calculate sentence similarity (simplified)
        return 0.8  # Placeholder for actual semantic analysis
    
    def _count_syllables(self, word: str) -> int:
        """Enhanced syllable counting with better accuracy."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        # Handle edge cases
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
        
        return count
    
    async def optimize_content(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """Optimize content for target keywords with advanced techniques."""
        try:
            # Analyze current content
            current_analysis = self.analyze_seo_score(content)
            
            # Generate optimized content using advanced techniques
            optimized_content = await self._generate_advanced_optimized_content(
                content, target_keywords
            )
            
            # Analyze optimized content
            optimized_analysis = self.analyze_seo_score(optimized_content)
            
            # Generate visual content if diffusion is enabled
            visual_content = None
            if self.diffusion_generator:
                visual_content = self.diffusion_generator.generate_seo_visual_content(
                    optimized_content, 
                    optimized_analysis.get("detailed_analysis", {}),
                    target_keywords
                )
            
            return {
                "original_content": content,
                "optimized_content": optimized_content,
                "original_score": current_analysis.get("seo_score", 0),
                "optimized_score": optimized_analysis.get("seo_score", 0),
                "improvement": optimized_analysis.get("seo_score", 0) - current_analysis.get("seo_score", 0),
                "keyword_optimization": self._analyze_keyword_optimization(content, optimized_content, target_keywords),
                "recommendations": optimized_analysis.get("detailed_analysis", {}).get("recommendations", []),
                "visual_content": visual_content
            }
        except Exception as e:
            self.logger.error(f"Error optimizing content: {e}")
            return {"error": str(e)}
    
    async def _generate_advanced_optimized_content(self, content: str, keywords: List[str]) -> str:
        """Generate optimized content using advanced LLM techniques."""
        optimized = content
        
        # Advanced keyword integration
        for keyword in keywords:
            if keyword.lower() not in optimized.lower():
                # Find optimal insertion points
                sentences = optimized.split('.')
                if sentences:
                    # Insert keyword naturally in first sentence
                    first_sentence = sentences[0].strip()
                    if first_sentence:
                        sentences[0] = f"{first_sentence} {keyword}."
                        optimized = '. '.join(sentences)
        
        # Add semantic variations
        semantic_variations = self._generate_semantic_variations(keywords)
        for variation in semantic_variations:
            if variation.lower() not in optimized.lower():
                optimized += f" {variation}."
        
        return optimized
    
    def _generate_semantic_variations(self, keywords: List[str]) -> List[str]:
        """Generate semantic variations of keywords."""
        variations = []
        for keyword in keywords:
            # Add common variations
            variations.extend([
                f"best {keyword}",
                f"top {keyword}",
                f"{keyword} guide",
                f"{keyword} tips"
            ])
        return variations[:5]  # Limit to 5 variations
    
    def _analyze_keyword_optimization(self, original: str, optimized: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze keyword optimization effectiveness."""
        original_lower = original.lower()
        optimized_lower = optimized.lower()
        
        keyword_analysis = {}
        for keyword in keywords:
            original_count = original_lower.count(keyword.lower())
            optimized_count = optimized_lower.count(keyword.lower())
            
            keyword_analysis[keyword] = {
                "original_frequency": original_count,
                "optimized_frequency": optimized_count,
                "improvement": optimized_count - original_count,
                "density": optimized_count / len(optimized.split()) if optimized.split() else 0
            }
        
        return keyword_analysis
    
    def batch_analyze(self, texts: List[str], keywords_list: List[List[str]] = None) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch with advanced sequence processing."""
        results = []
        
        try:
            # Process all sequences
            sequences = self.sequence_processor.process_sequences(texts, keywords_list)
            
            # Create batches
            batches = self.sequence_processor.create_sequence_batches(sequences, self.config.batch_size)
            
            for batch in batches:
                # Move to device
                batch_inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                
                # Process batch
                with torch.no_grad():
                    outputs = self.model(**batch_inputs)
                    logits = outputs['logits']
                    probabilities = F.softmax(logits, dim=-1)
                    
                    # Process each sample in batch
                    for i in range(batch['input_ids'].size(0)):
                        seo_score = probabilities[i][1].item()
                        
                        # Extract features
                        keyword_features = outputs['keyword_features'][i].cpu().numpy()
                        readability_features = outputs['readability_features'][i].cpu().numpy()
                        quality_features = outputs['quality_features'][i].cpu().numpy()
                        
                        # Get original text index
                        text_idx = len(results)
                        original_text = texts[text_idx] if text_idx < len(texts) else ""
                        
                        result = {
                            "seo_score": seo_score,
                            "confidence": max(probabilities[i]).item(),
                            "keyword_analysis": self._analyze_keyword_features(keyword_features),
                            "readability_analysis": self._analyze_readability_features(readability_features),
                            "quality_analysis": self._analyze_quality_features(quality_features),
                            "detailed_analysis": self._generate_comprehensive_analysis(original_text, seo_score)
                        }
                        results.append(result)
                
                # Clear GPU cache periodically
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        except Exception as e:
            self.logger.error(f"Error in batch analysis: {e}")
            # Return error results for all texts
            results = [{"error": str(e)} for _ in texts]
        
        return results
    
    # Pipeline Manager Methods
    def get_available_pipelines(self) -> List[str]:
        """Get list of available diffusion pipeline types."""
        if not self.pipeline_manager:
            return []
        return self.pipeline_manager.get_available_pipelines()
    
    def switch_pipeline(self, pipeline_type: str) -> Dict[str, Any]:
        """Switch to a different diffusion pipeline."""
        if not self.pipeline_manager:
            return {"error": "Pipeline manager not initialized"}
        
        try:
            pipeline = self.pipeline_manager.get_pipeline(pipeline_type)
            return {
                "status": "success",
                "pipeline_type": pipeline_type,
                "pipeline_class": pipeline.__class__.__name__,
                "device": str(pipeline.device) if hasattr(pipeline, 'device') else str(self.device)
            }
        except Exception as e:
            return {"error": f"Failed to switch pipeline: {str(e)}"}
    
    def generate_with_pipeline(self, pipeline_type: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate content using a specific diffusion pipeline."""
        if not self.pipeline_manager:
            return {"error": "Pipeline manager not initialized"}
        
        try:
            result = self.pipeline_manager.generate_with_pipeline(pipeline_type, prompt, **kwargs)
            return {
                "status": "success",
                "pipeline_type": pipeline_type,
                "result": result,
                "prompt": prompt,
                "parameters": kwargs
            }
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}
    
    def get_pipeline_info(self, pipeline_type: str = None) -> Dict[str, Any]:
        """Get information about a specific pipeline."""
        if not self.pipeline_manager:
            return {"error": "Pipeline manager not initialized"}
        
        if pipeline_type is None:
            pipeline_type = self.pipeline_manager.current_pipeline
        
        try:
            pipeline = self.pipeline_manager.get_pipeline(pipeline_type)
            info = {
                "pipeline_type": pipeline_type,
                "pipeline_class": pipeline.__class__.__name__,
                "device": str(pipeline.device) if hasattr(pipeline, 'device') else str(self.device),
                "config": {
                    "diffusion_steps": self.config.diffusion_steps,
                    "guidance_scale": self.config.diffusion_guidance_scale,
                    "height": self.config.diffusion_height,
                    "width": self.config.diffusion_width,
                    "negative_prompt": self.config.negative_prompt
                }
            }
            return info
        except Exception as e:
            return {"error": f"Failed to get pipeline info: {str(e)}"}
    
    def cleanup_pipelines(self):
        """Clean up pipeline resources."""
        if self.pipeline_manager:
            self.pipeline_manager.cleanup()

    # Data Loading Methods
    def create_training_dataset(self, texts: List[str], labels: Optional[List[int]] = None,
                               name: str = "training") -> SEODataset:
        """Create a training dataset for SEO analysis."""
        try:
            # Debug data loading if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"🔍 Data loading debugging - Creating dataset '{name}'")
                self.logger.debug(f"   Number of texts: {len(texts)}")
                self.logger.debug(f"   Labels provided: {labels is not None}")
                if labels is not None:
                    self.logger.debug(f"   Number of labels: {len(labels)}")
                    unique_labels = set(labels)
                    self.logger.debug(f"   Unique labels: {sorted(unique_labels)}")
                    label_counts = {label: labels.count(label) for label in unique_labels}
                    self.logger.debug(f"   Label distribution: {label_counts}")
                self.logger.debug(f"   Tokenizer available: {self.tokenizer is not None}")
            
            dataset = self.data_loader_manager.create_dataset(name, texts, labels, self.tokenizer)
            
            # Debug dataset creation if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"   Dataset created successfully: {len(dataset)} samples")
                if hasattr(dataset, 'metadata') and dataset.metadata:
                    self.logger.debug(f"   Metadata keys: {list(dataset.metadata.keys())}")
            
            return dataset
            
        except Exception as e:
            self.log_error(e, "Creating training dataset", "create_training_dataset", 
                          {"name": name, "texts_count": len(texts), "labels_count": len(labels) if labels else 0})
            raise
    
    def create_training_dataloaders(self, texts: List[str], labels: Optional[List[int]] = None,
                                   name: str = "training", val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create train/validation DataLoaders for training."""
        try:
            # Debug data loading if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"🔍 Data loading debugging - Creating train/val dataloaders '{name}'")
                self.logger.debug(f"   Validation split: {val_split}")
            
            dataset = self.create_training_dataset(texts, labels, name)
            train_loader, val_loader = self.data_loader_manager.create_train_val_split(name, dataset, val_split)
            
            # Debug dataloader creation if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"   Train loader: {len(train_loader)} batches")
                self.logger.debug(f"   Validation loader: {len(val_loader)} batches")
                if hasattr(train_loader, 'dataset'):
                    self.logger.debug(f"   Train samples: {len(train_loader.dataset)}")
                if hasattr(val_loader, 'dataset'):
                    self.logger.debug(f"   Validation samples: {len(val_loader.dataset)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.log_error(e, "Creating training dataloaders", "create_training_dataloaders", 
                          {"name": name, "val_split": val_split})
            raise
    
    def create_training_dataloaders_with_test(self, texts: List[str], labels: Optional[List[int]] = None,
                                             name: str = "training") -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test DataLoaders for training."""
        dataset = self.create_training_dataset(texts, labels, name)
        return self.data_loader_manager.create_train_val_test_split(name, dataset)
    
    def create_cross_validation_dataloaders(self, texts: List[str], labels: Optional[List[int]] = None,
                                           name: str = "training", metadata: Optional[Dict[str, List[Any]]] = None,
                                           cv_config: Optional[DataLoaderConfig] = None) -> List[Tuple[DataLoader, DataLoader]]:
        """Create cross-validation DataLoaders for training."""
        try:
            # Debug data loading if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"🔍 Data loading debugging - Creating cross-validation dataloaders '{name}'")
                self.logger.debug(f"   Metadata provided: {metadata is not None}")
                if metadata:
                    self.logger.debug(f"   Metadata keys: {list(metadata.keys())}")
            
            # Create dataset with metadata
            dataset = self.create_training_dataset(texts, labels, name, metadata)
            
            # Create CV configuration
            if cv_config is None:
                cv_config = DataLoaderConfig(
                    use_cross_validation=True,
                    cv_folds=5,
                    cv_strategy="stratified",
                    batch_size=self.config.batch_size,
                    num_workers=self.config.dataloader_num_workers
                )
            
            # Debug CV configuration if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"   CV folds: {cv_config.cv_folds}")
                self.logger.debug(f"   CV strategy: {cv_config.cv_strategy}")
                self.logger.debug(f"   Batch size: {cv_config.batch_size}")
                self.logger.debug(f"   Num workers: {cv_config.num_workers}")
            
            cv_loaders = self.data_loader_manager.create_cross_validation_folds(name, dataset, cv_config)
            
            # Debug CV creation if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"   Cross-validation loaders created: {len(cv_loaders)} folds")
                for i, (train_loader, val_loader) in enumerate(cv_loaders):
                    self.logger.debug(f"   Fold {i+1}: {len(train_loader)} train batches, {len(val_loader)} val batches")
            
            return cv_loaders
            
        except Exception as e:
            self.log_error(e, "Creating cross-validation dataloaders", "create_cross_validation_dataloaders", 
                          {"name": name, "metadata_keys": list(metadata.keys()) if metadata else None})
            raise
    
    def create_stratified_dataloaders(self, texts: List[str], labels: Optional[List[int]] = None,
                                     name: str = "training", val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create stratified train/validation DataLoaders."""
        try:
            # Debug data loading if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"🔍 Data loading debugging - Creating stratified dataloaders '{name}'")
                self.logger.debug(f"   Validation split: {val_split}")
                if labels is not None:
                    unique_labels = set(labels)
                    label_counts = {label: labels.count(label) for label in unique_labels}
                    self.logger.debug(f"   Label distribution: {label_counts}")
            
            dataset = self.create_training_dataset(texts, labels, name)
            train_loader, val_loader = self.data_loader_manager.create_stratified_split(name, dataset, val_split)
            
            # Debug stratified creation if enabled
            if self.config.debug_data_loading:
                self.logger.debug(f"   Stratified split completed")
                self.logger.debug(f"   Train loader: {len(train_loader)} batches")
                self.logger.debug(f"   Validation loader: {len(val_loader)} batches")
                if hasattr(train_loader, 'dataset'):
                    self.logger.debug(f"   Train samples: {len(train_loader.dataset)}")
                if hasattr(val_loader, 'dataset'):
                    self.logger.debug(f"   Validation samples: {len(val_loader.dataset)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.log_error(e, "Creating stratified dataloaders", "create_stratified_dataloaders", 
                          {"name": name, "val_split": val_split})
            raise
    
    def create_distributed_dataloaders(self, texts: List[str], labels: Optional[List[int]] = None,
                                      name: str = "training", val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create distributed DataLoaders for multi-GPU training."""
        try:
            if not hasattr(self, 'is_distributed') or not self.is_distributed:
                self.logger.warning("⚠️  Not in distributed mode, creating regular dataloaders")
                return self.create_training_dataloaders(texts, labels, name, val_split)
            
            self.logger.info("🔧 Creating distributed DataLoaders")
            
            # Create dataset
            dataset = self.create_training_dataset(texts, labels, name)
            
            # Create train/val split
            train_size = int((1 - val_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create distributed samplers
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.config.distributed_world_size,
                rank=self.config.distributed_rank,
                shuffle=True
            )
            
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.config.distributed_world_size,
                rank=self.config.distributed_rank,
                shuffle=False
            )
            
            # Create DataLoaders with distributed samplers
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size // self.config.distributed_world_size,
                sampler=train_sampler,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                persistent_workers=self.config.dataloader_persistent_workers
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size // self.config.distributed_world_size,
                sampler=val_sampler,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                persistent_workers=self.config.dataloader_persistent_workers
            )
            
            self.logger.info(f"✅ Distributed DataLoaders created with {len(train_loader)} train batches and {len(val_loader)} val batches")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.log_error(e, "Creating distributed dataloaders", "create_distributed_dataloaders", 
                          {"name": name, "val_split": val_split})
            raise
    
    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
                   early_stopping: Optional[EarlyStopping] = None) -> Dict[str, float]:
        """Train for one epoch using the provided DataLoader with early stopping support and comprehensive logging."""
        try:
            # Validate inputs
            if not train_loader:
                raise ValueError("Train loader cannot be None")
            
            # Log epoch start
            epoch_start_time = time.time()
            self.log_training_progress(
                epoch=self.training_state['epoch'],
                step=self.training_state['step'],
                loss=0.0,  # Will be updated during training
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
            
            self.seo_model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Profile training loop
            with self.code_profiler.profile_operation("train_epoch", "training_loop"):
            
            # Gradient accumulation setup
            accumulation_steps = self.config.gradient_accumulation_steps if self.config.use_gradient_accumulation else 1
            effective_batch_size = self.config.batch_size * accumulation_steps
            self.logger.info(f"🔧 Gradient accumulation: {accumulation_steps} steps, effective batch size: {effective_batch_size}")
            
            # Initialize gradient accumulation variables
            accumulated_loss = 0.0
            gradient_accumulation_counter = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch_start_time = time.time()
                
                try:
                    # Move batch to device with error handling
                    try:
                        if isinstance(batch, dict) and 'input_ids' in batch:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)
                        else:
                            self.log_error(
                                error=ValueError(f"Invalid batch format at index {batch_idx}"),
                                context="Batch processing",
                                operation="train_epoch",
                                additional_info={"batch_idx": batch_idx, "batch_keys": list(batch.keys()) if isinstance(batch, dict) else "Not a dict"}
                            )
                            continue
                    except Exception as e:
                        self.log_error(e, "Moving batch to device", "train_epoch", {"batch_idx": batch_idx})
                        continue
                    
                    # Zero gradients only at the start of accumulation cycle
                    try:
                        if gradient_accumulation_counter == 0:
                            self.optimizer.zero_grad()
                            if self.config.debug_gradient_norms:
                                self.logger.debug(f"🔧 Zeroed gradients at start of accumulation cycle - Batch {batch_idx}")
                    except Exception as e:
                        self.log_error(e, "Zeroing gradients", "train_epoch", {"batch_idx": batch_idx})
                        continue
                    
                    # Forward pass with mixed precision and error handling
                    try:
                        # Debug forward pass if enabled
                        if self.config.debug_forward_pass:
                            self.logger.debug(f"🔍 Forward pass debugging - Batch {batch_idx}")
                            self.logger.debug(f"   Input shape: {input_ids.shape}")
                            self.logger.debug(f"   Attention mask shape: {attention_mask.shape}")
                            self.logger.debug(f"   Labels shape: {labels.shape}")
                            self.logger.debug(f"   Device: {input_ids.device}")
                        
                        # Debug mixed precision if enabled
                        if self.config.debug_mixed_precision:
                            self.logger.debug(f"🔍 Mixed precision debugging - Batch {batch_idx}")
                            self.logger.debug(f"   Mixed precision enabled: {self.config.use_mixed_precision}")
                            self.logger.debug(f"   Scaler available: {self.scaler is not None}")
                        
                        if self.config.use_mixed_precision and self.scaler:
                            # Enhanced mixed precision training with optimal dtype
                            autocast_kwargs = {}
                            
                            # Configure autocast based on settings
                            if self.config.mixed_precision_autocast_dtype == "auto":
                                autocast_kwargs['dtype'] = self.mixed_precision_dtype
                            elif self.config.mixed_precision_autocast_dtype == "bfloat16":
                                autocast_kwargs['dtype'] = torch.bfloat16
                            elif self.config.mixed_precision_autocast_dtype == "float16":
                                autocast_kwargs['dtype'] = torch.float16
                            
                            # Enable cache if configured
                            if self.config.mixed_precision_autocast_cache_enabled:
                                autocast_kwargs['cache_enabled'] = True
                            
                            # Use enhanced autocast context
                            with autocast(**autocast_kwargs):
                                # Cast inputs if configured
                                if self.config.mixed_precision_cast_inputs:
                                    input_ids = input_ids.to(dtype=self.mixed_precision_dtype)
                                    attention_mask = attention_mask.to(dtype=self.mixed_precision_dtype)
                                
                                outputs = self.seo_model(input_ids, attention_mask)
                                loss = F.cross_entropy(outputs, labels)
                                
                                # Cast outputs if configured
                                if self.config.mixed_precision_cast_outputs:
                                    outputs = outputs.to(dtype=torch.float32)
                                    loss = loss.to(dtype=torch.float32)
                            
                            # Debug outputs if enabled
                            if self.config.debug_forward_pass:
                                self.logger.debug(f"   Output shape: {outputs.shape}")
                                self.logger.debug(f"   Loss value: {loss.item():.6f}")
                            
                            # Scale loss for gradient accumulation
                            scaled_loss = loss / accumulation_steps
                            
                            # Backward pass with gradient scaling
                            self.scaler.scale(scaled_loss).backward()
                            
                            # Gradient clipping before accumulation if enabled
                            if self.config.clip_gradients_before_accumulation and self.config.max_grad_norm > 0:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.seo_model.parameters(), self.config.max_grad_norm)
                            
                            # Update accumulation counter
                            gradient_accumulation_counter += 1
                            
                            # Step optimizer and update scaler only at the end of accumulation cycle
                            if gradient_accumulation_counter >= accumulation_steps:
                                # Gradient clipping after accumulation if not done before
                                if not self.config.clip_gradients_before_accumulation and self.config.max_grad_norm > 0:
                                    self.scaler.unscale_(self.optimizer)
                                    torch.nn.utils.clip_grad_norm_(self.seo_model.parameters(), self.config.max_grad_norm)
                                
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                gradient_accumulation_counter = 0
                        else:
                            outputs = self.seo_model(input_ids, attention_mask)
                            loss = F.cross_entropy(outputs, labels)
                            
                            # Debug outputs if enabled
                            if self.config.debug_forward_pass:
                                self.logger.debug(f"   Output shape: {outputs.shape}")
                                self.logger.debug(f"   Loss value: {loss.item():.6f}")
                            
                            # Scale loss for gradient accumulation
                            scaled_loss = loss / accumulation_steps
                            scaled_loss.backward()
                            
                            # Gradient clipping before accumulation if enabled
                            if self.config.clip_gradients_before_accumulation and self.config.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.seo_model.parameters(), self.config.max_grad_norm)
                            
                            # Update accumulation counter
                            gradient_accumulation_counter += 1
                            
                            # Step optimizer only at the end of accumulation cycle
                            if gradient_accumulation_counter >= accumulation_steps:
                                # Gradient clipping after accumulation if not done before
                                if not self.config.clip_gradients_before_accumulation and self.config.max_grad_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(self.seo_model.parameters(), self.config.max_grad_norm)
                                
                                self.optimizer.step()
                                gradient_accumulation_counter = 0
                        
                        # Debug backward pass if enabled
                        if self.config.debug_backward_pass:
                            self.logger.debug(f"🔍 Backward pass debugging - Batch {batch_idx}")
                            # Check for NaN/Inf in gradients
                            for name, param in self.seo_model.named_parameters():
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()
                                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                        self.logger.warning(f"⚠️  NaN/Inf gradient detected in {name}: {grad_norm}")
                                    elif self.config.debug_gradient_norms:
                                        self.logger.debug(f"   {name} gradient norm: {grad_norm:.6f}")
                        
                        # Debug memory usage if enabled
                        if self.config.debug_memory_usage:
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated() / 1024**2
                                reserved = torch.cuda.memory_reserved() / 1024**2
                                self.logger.debug(f"🔍 Memory usage - Batch {batch_idx}: "
                                               f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
                    except Exception as e:
                        self.log_error(e, "Forward/backward pass", "train_epoch", {"batch_idx": batch_idx})
                        continue
                    
                    # Update scheduler (step per batch for some schedulers)
                    try:
                        if self.scheduler and isinstance(self.scheduler, (torch.optim.lr_scheduler.ExponentialLR, 
                                                                       torch.optim.lr_scheduler.StepLR)):
                            self.scheduler.step()
                    except Exception as e:
                        self.log_error(e, "Updating scheduler", "train_epoch", {"batch_idx": batch_idx})
                    
                    # Handle remaining gradients at the end of accumulation cycle
                    if self.config.use_gradient_accumulation and gradient_accumulation_counter > 0:
                        try:
                            if self.config.debug_gradient_norms:
                                self.logger.debug(f"🔧 Processing remaining gradients: {gradient_accumulation_counter} steps")
                            
                            # Gradient clipping for remaining gradients
                            if self.config.max_grad_norm > 0:
                                if self.config.use_mixed_precision and self.scaler:
                                    self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.seo_model.parameters(), self.config.max_grad_norm)
                            
                            # Step optimizer for remaining gradients
                            if self.config.use_mixed_precision and self.scaler:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            
                            gradient_accumulation_counter = 0
                            
                        except Exception as e:
                            self.log_error(e, "Processing remaining gradients", "train_epoch", {"batch_idx": batch_idx})
                    
                                            # Track mixed precision performance
                        if self.config.use_mixed_precision:
                            try:
                                # Record mixed precision metrics
                                if hasattr(self, 'training_state') and 'training_history' in self.training_state:
                                    current_epoch = self.training_state.get('epoch', 0)
                                    if current_epoch > 0:
                                        # Find or create epoch record
                                        epoch_record = None
                                        for record in self.training_state['training_history']:
                                            if record.get('epoch') == current_epoch:
                                                epoch_record = record
                                                break
                                        
                                        if epoch_record is None:
                                            epoch_record = {
                                                'epoch': current_epoch,
                                                'mixed_precision_enabled': True,
                                                'mixed_precision_dtype': str(self.mixed_precision_dtype),
                                                'gradient_scaler_scale': self.scaler.get_scale() if self.scaler else None
                                            }
                                            self.training_state['training_history'].append(epoch_record)
                                        else:
                                            epoch_record['mixed_precision_enabled'] = True
                                            epoch_record['mixed_precision_dtype'] = str(self.mixed_precision_dtype)
                                            epoch_record['gradient_scaler_scale'] = self.scaler.get_scale() if self.scaler else None
                            except Exception as e:
                                self.logger.warning(f"Failed to track mixed precision metrics: {e}")
                        
                        # Update metrics
                        try:
                        total_loss += loss.item()
                        num_batches += 1
                        self.training_state['step'] += 1
                        
                        # Log batch progress with comprehensive metrics
                        batch_duration = time.time() - batch_start_time
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        if batch_idx % 10 == 0:
                            self.log_training_progress(
                                epoch=self.training_state['epoch'],
                                step=self.training_state['step'],
                                loss=loss.item(),
                                learning_rate=current_lr,
                                metrics={
                                    "batch_duration": batch_duration,
                                    "batch_idx": batch_idx,
                                    "total_batches": len(train_loader)
                                }
                            )
                            
                            # Log model performance for this batch
                            self.log_model_performance(
                                operation=f"batch_training_{batch_idx}",
                                duration=batch_duration,
                                memory_usage=psutil.virtual_memory().used / 1024 / 1024 if hasattr(psutil, 'virtual_memory') else None,
                                additional_metrics={
                                    "batch_loss": loss.item(),
                                    "learning_rate": current_lr,
                                    "batch_size": input_ids.size(0)
                                }
                            )
                    except Exception as e:
                        self.log_error(e, "Updating metrics", "train_epoch", {"batch_idx": batch_idx})
                            
                except Exception as e:
                    self.log_error(e, "Processing batch", "train_epoch", {"batch_idx": batch_idx})
                    continue
            
            # Calculate average training loss
            try:
                avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            except Exception as e:
                self.log_error(e, "Calculating average training loss", "train_epoch")
                avg_train_loss = 0.0
            
            # Validation with error handling
            val_loss = 0.0
            if val_loader:
                try:
                    # Debug validation if enabled
                    if self.config.debug_validation:
                        self.logger.debug(f"🔍 Validation debugging - Epoch {self.training_state['epoch']}")
                        self.logger.debug(f"   Validation samples: {len(val_loader.dataset)}")
                        self.logger.debug(f"   Validation batches: {len(val_loader)}")
                        if torch.cuda.is_available():
                            self.logger.debug(f"   CUDA memory before validation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    
                    val_start_time = time.time()
                    val_loss = self.evaluate_model(val_loader)
                    val_duration = time.time() - val_start_time
                    
                    # Debug validation results if enabled
                    if self.config.debug_validation:
                        self.logger.debug(f"   Validation loss: {val_loss:.6f}")
                        self.logger.debug(f"   Validation duration: {val_duration:.4f}s")
                        if torch.cuda.is_available():
                            self.logger.debug(f"   CUDA memory after validation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    
                    # Log validation performance
                    self.log_model_performance(
                        operation="validation",
                        duration=val_duration,
                        additional_metrics={
                            "validation_loss": val_loss,
                            "validation_samples": len(val_loader.dataset)
                        }
                    )
                except Exception as e:
                    self.log_error(e, "Validation", "train_epoch")
                    val_loss = float('inf')
                
                # Update scheduler (step per epoch for some schedulers)
                try:
                    # Debug learning rate scheduling if enabled
                    if self.config.debug_lr_scheduling:
                        self.logger.debug(f"🔍 Learning rate scheduling debugging - Epoch {self.training_state['epoch']}")
                        self.logger.debug(f"   Scheduler type: {type(self.scheduler).__name__}")
                        self.logger.debug(f"   Current learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                    
                    if self.scheduler and isinstance(self.scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau,
                                                                   get_cosine_schedule_with_warmup,
                                                                   get_linear_schedule_with_warmup)):
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            old_lr = self.optimizer.param_groups[0]['lr']
                            self.scheduler.step(val_loss)
                            new_lr = self.optimizer.param_groups[0]['lr']
                            
                            # Debug LR change if enabled
                            if self.config.debug_lr_scheduling and old_lr != new_lr:
                                self.logger.debug(f"   Learning rate changed: {old_lr:.2e} → {new_lr:.2e}")
                        else:
                            old_lr = self.optimizer.param_groups[0]['lr']
                            self.scheduler.step()
                            new_lr = self.optimizer.param_groups[0]['lr']
                            
                            # Debug LR change if enabled
                            if self.config.debug_lr_scheduling and old_lr != new_lr:
                                self.logger.debug(f"   Learning rate changed: {old_lr:.2e} → {new_lr:.2e}")
                except Exception as e:
                    self.log_error(e, "Updating epoch scheduler", "train_epoch")
            
            # Check early stopping with error handling
            should_stop = False
            if early_stopping and val_loader:
                try:
                    # Debug early stopping if enabled
                    if self.config.debug_early_stopping:
                        self.logger.debug(f"🔍 Early stopping debugging - Epoch {self.training_state['epoch']}")
                        self.logger.debug(f"   Monitor metric: {early_stopping.monitor}")
                        self.logger.debug(f"   Current patience: {early_stopping.counter}")
                        self.logger.debug(f"   Best value: {early_stopping.best_score}")
                        self.logger.debug(f"   Min delta: {early_stopping.min_delta}")
                    
                    monitor_value = val_loss if early_stopping.monitor == "val_loss" else avg_train_loss
                    should_stop = early_stopping(monitor_value)
                    
                    # Debug early stopping decision if enabled
                    if self.config.debug_early_stopping:
                        self.logger.debug(f"   Monitor value: {monitor_value:.6f}")
                        self.logger.debug(f"   Should stop: {should_stop}")
                        if should_stop:
                            self.logger.debug(f"   Early stopping triggered! Patience exceeded: {early_stopping.counter}")
                    
                    if should_stop:
                        self.log_training_progress(
                            epoch=self.training_state['epoch'],
                            step=self.training_state['step'],
                            loss=avg_train_loss,
                            learning_rate=self.optimizer.param_groups[0]['lr'],
                            validation_loss=val_loss,
                            metrics={"early_stopping_triggered": True, "patience_exceeded": early_stopping.counter}
                        )
                except Exception as e:
                    self.log_error(e, "Checking early stopping", "train_epoch")
            
            # Update training state with error handling
            try:
                self.training_state['epoch'] += 1
                self.training_state['training_history'].append({
                    'epoch': self.training_state['epoch'],
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'early_stopping_triggered': should_stop
                })
                
                # Log epoch completion
                epoch_duration = time.time() - epoch_start_time
                self.log_training_progress(
                    epoch=self.training_state['epoch'],
                    step=self.training_state['step'],
                    loss=avg_train_loss,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    validation_loss=val_loss,
                    metrics={
                        "epoch_duration": epoch_duration,
                        "avg_batch_loss": avg_train_loss,
                        "total_batches": num_batches
                    }
                )
                
                # Log epoch performance
                self.log_model_performance(
                    operation="epoch_training",
                    duration=epoch_duration,
                    memory_usage=psutil.virtual_memory().used / 1024 / 1024 if hasattr(psutil, 'virtual_memory') else None,
                    additional_metrics={
                        "epoch": self.training_state['epoch'],
                        "avg_train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "total_batches": num_batches
                    }
                )
                
            except Exception as e:
                self.log_error(e, "Updating training state", "train_epoch")
            
            # Close profiling context
            # (profiling context is automatically closed)
            
            return {
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'early_stopping_triggered': should_stop
            }
            
        except Exception as e:
            self.log_error(e, "Critical error in train_epoch", "train_epoch")
            return {
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'learning_rate': 0.0,
                'early_stopping_triggered': True,
                'error': str(e)
            }
    
    def train_with_early_stopping(self, train_loader: DataLoader, val_loader: DataLoader,
                                 max_epochs: int = 100, early_stopping: Optional[EarlyStopping] = None) -> Dict[str, Any]:
        """Train the model with early stopping and learning rate scheduling with comprehensive logging."""
        training_start_time = time.time()
        
        if early_stopping is None:
            early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                monitor=self.config.early_stopping_monitor,
                mode=self.config.early_stopping_mode
            )
        
        # Log hyperparameters
        config_dict = {
            'max_epochs': max_epochs,
            'early_stopping_patience': early_stopping.patience,
            'early_stopping_min_delta': early_stopping.min_delta,
            'early_stopping_monitor': early_stopping.monitor,
            'early_stopping_mode': early_stopping.mode,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': getattr(self.config, 'batch_size', 'N/A'),
            'use_mixed_precision': getattr(self.config, 'use_mixed_precision', False),
            'max_grad_norm': getattr(self.config, 'max_grad_norm', 'N/A')
        }
        self.log_hyperparameters(config_dict)
        
        training_results = {
            'epochs_completed': 0,
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'early_stopping_triggered': False,
            'training_history': [],
            'learning_rate_history': []
        }
        
        # Log training start
        self.log_training_progress(
            epoch=0,
            step=0,
            loss=0.0,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            metrics={
                "max_epochs": max_epochs,
                "early_stopping_patience": early_stopping.patience,
                "training_start_time": training_start_time
            }
        )
        
        self.logger.info(f"Starting training with early stopping (patience: {early_stopping.patience})")
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            self.logger.info(f"Epoch {epoch + 1}/{max_epochs}")
            
            # Train one epoch
            epoch_result = self.train_epoch(train_loader, val_loader, early_stopping)
            
            # Store results
            training_results['training_history'].append(epoch_result)
            training_results['learning_rate_history'].append(epoch_result['learning_rate'])
            training_results['epochs_completed'] = epoch + 1
            
            # Track best performance
            if epoch_result['val_loss'] < training_results['best_val_loss']:
                training_results['best_val_loss'] = epoch_result['val_loss']
                training_results['best_epoch'] = epoch + 1
                
                # Save best model checkpoint
                if self.config.save_checkpoints:
                    self.save_checkpoint(f"best_model_epoch_{epoch + 1}.pt")
            
            # Log epoch summary
            self.logger.info(f"Epoch {epoch + 1}: Train Loss: {epoch_result['train_loss']:.4f}, "
                           f"Val Loss: {epoch_result['val_loss']:.4f}, "
                           f"LR: {epoch_result['learning_rate']:.2e}")
            
            # Check early stopping
            if epoch_result['early_stopping_triggered']:
                training_results['early_stopping_triggered'] = True
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        self.logger.info(f"Training completed. Best val loss: {training_results['best_val_loss']:.4f} "
                        f"at epoch {training_results['best_epoch']}")
        
        return training_results
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.seo_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'training_state': self.training_state,
            'epoch': self.training_state['epoch']
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.seo_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_state = checkpoint['training_state']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_learning_rate_info(self) -> Dict[str, Any]:
        """Get current learning rate information."""
        if not self.optimizer:
            return {"error": "Optimizer not initialized"}
        
        lr_info = {
            "current_lr": self.optimizer.param_groups[0]['lr'],
            "scheduler_type": self.config.lr_scheduler,
            "scheduler_params": self.config.lr_scheduler_params.get(self.config.lr_scheduler, {}),
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay
        }
        
        if self.scheduler:
            lr_info["scheduler_state"] = str(type(self.scheduler))
        
        return lr_info
    
    def evaluate_model(self, dataloader: DataLoader) -> float:
        """Evaluate the model on a validation/test DataLoader."""
        try:
            # Validate input
            if not dataloader:
                raise ValueError("DataLoader cannot be None")
            
            # Debug evaluation if enabled
            if self.config.debug_validation:
                self.logger.debug(f"🔍 Evaluation debugging - Starting model evaluation")
                self.logger.debug(f"   DataLoader batches: {len(dataloader)}")
                if hasattr(dataloader, 'dataset'):
                    self.logger.debug(f"   Dataset samples: {len(dataloader.dataset)}")
                self.logger.debug(f"   Device: {self.device}")
                if torch.cuda.is_available():
                    self.logger.debug(f"   CUDA memory before evaluation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            self.seo_model.eval()
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        # Debug batch processing if enabled
                        if self.config.debug_validation and batch_idx % 10 == 0:
                            self.logger.debug(f"🔍 Evaluation debugging - Processing batch {batch_idx}")
                        
                        # Validate batch format
                        if isinstance(batch, dict) and 'input_ids' in batch:
                            try:
                                input_ids = batch['input_ids'].to(self.device)
                                attention_mask = batch['attention_mask'].to(self.device)
                                labels = batch['labels'].to(self.device)
                                
                                # Debug batch details if enabled
                                if self.config.debug_validation and batch_idx % 10 == 0:
                                    self.logger.debug(f"   Batch {batch_idx} shapes: input_ids={input_ids.shape}, "
                                                   f"attention_mask={attention_mask.shape}, labels={labels.shape}")
                                
                            except Exception as e:
                                self.log_error(e, "Moving batch to device", "evaluate_model", {"batch_idx": batch_idx})
                                continue
                        else:
                            self.logger.warning(f"Skipping invalid batch format at index {batch_idx}")
                            continue
                        
                        # Forward pass with error handling
                        try:
                            outputs = self.seo_model(input_ids, attention_mask)
                            loss = F.cross_entropy(outputs, labels)
                            
                            # Debug forward pass if enabled
                            if self.config.debug_forward_pass and batch_idx % 10 == 0:
                                self.logger.debug(f"   Batch {batch_idx} outputs: shape={outputs.shape}, loss={loss.item():.6f}")
                            
                        except Exception as e:
                            self.log_error(e, "Forward pass", "evaluate_model", {"batch_idx": batch_idx})
                            continue
                        
                        # Update metrics with error handling
                        try:
                            total_loss += loss.item()
                            num_batches += 1
                        except Exception as e:
                            self.log_error(e, "Updating metrics", "evaluate_model", {"batch_idx": batch_idx})
                            continue
                            
                    except Exception as e:
                        self.log_error(e, "Processing batch", "evaluate_model", {"batch_idx": batch_idx})
                        continue
            
            # Calculate average loss with error handling
            try:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                
                # Debug evaluation results if enabled
                if self.config.debug_validation:
                    self.logger.debug(f"🔍 Evaluation debugging - Evaluation completed")
                    self.logger.debug(f"   Total batches processed: {num_batches}")
                    self.logger.debug(f"   Total loss: {total_loss:.6f}")
                    self.logger.debug(f"   Average loss: {avg_loss:.6f}")
                    if torch.cuda.is_available():
                        self.logger.debug(f"   CUDA memory after evaluation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
                return avg_loss
                
            except Exception as e:
                self.log_error(e, "Calculating average loss", "evaluate_model")
                return 0.0
                
        except Exception as e:
            self.log_error(e, "Critical error in model evaluation", "evaluate_model")
            return float('inf')
    
    def train_cross_validation(self, cv_dataloaders: List[Tuple[DataLoader, DataLoader]], 
                              num_epochs: int = 1) -> Dict[str, Any]:
        """Train model using cross-validation."""
        cv_results = {
            'fold_results': [],
            'mean_train_loss': 0.0,
            'mean_val_loss': 0.0,
            'std_train_loss': 0.0,
            'std_val_loss': 0.0,
            'best_fold': 0,
            'best_val_loss': float('inf')
        }
        
        train_losses = []
        val_losses = []
        
        for fold_idx, (train_loader, val_loader) in enumerate(cv_dataloaders):
            self.logger.info(f"Training fold {fold_idx + 1}/{len(cv_dataloaders)}")
            
            # Reset model weights for each fold (optional)
            if fold_idx > 0:
                self._reset_model_weights()
            
            fold_results = {
                'fold': fold_idx + 1,
                'epoch_results': []
            }
            
            for epoch in range(num_epochs):
                epoch_result = self.train_epoch(train_loader, val_loader)
                fold_results['epoch_results'].append(epoch_result)
                
                # Log progress
                self.logger.info(f"Fold {fold_idx + 1}, Epoch {epoch + 1}: "
                               f"Train Loss: {epoch_result['train_loss']:.4f}, "
                               f"Val Loss: {epoch_result['val_loss']:.4f}")
            
            # Get final fold performance
            final_epoch = fold_results['epoch_results'][-1]
            train_losses.append(final_epoch['train_loss'])
            val_losses.append(final_epoch['val_loss'])
            
            # Track best fold
            if final_epoch['val_loss'] < cv_results['best_val_loss']:
                cv_results['best_val_loss'] = final_epoch['val_loss']
                cv_results['best_fold'] = fold_idx
            
            fold_results['final_train_loss'] = final_epoch['train_loss']
            fold_results['final_val_loss'] = final_epoch['val_loss']
            cv_results['fold_results'].append(fold_results)
        
        # Calculate statistics
        cv_results['mean_train_loss'] = np.mean(train_losses)
        cv_results['mean_val_loss'] = np.mean(val_losses)
        cv_results['std_train_loss'] = np.std(train_losses)
        cv_results['std_val_loss'] = np.std(val_losses)
        
        self.logger.info(f"Cross-validation completed. "
                        f"Mean Val Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
        
        return cv_results
    
    def _reset_model_weights(self):
        """Reset model weights to initial state for cross-validation."""
        # This is a simple approach - in practice, you might want to save/restore initial weights
        for module in self.seo_model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def evaluate_cross_validation(self, cv_dataloaders: List[Tuple[DataLoader, DataLoader]]) -> Dict[str, Any]:
        """Evaluate model performance across all cross-validation folds."""
        cv_eval_results = {
            'fold_evaluations': [],
            'overall_metrics': {}
        }
        
        all_predictions = []
        all_labels = []
        
        for fold_idx, (train_loader, val_loader) in enumerate(cv_dataloaders):
            self.logger.info(f"Evaluating fold {fold_idx + 1}/{len(cv_dataloaders)}")
            
            fold_eval = self._evaluate_fold(val_loader)
            cv_eval_results['fold_evaluations'].append({
                'fold': fold_idx + 1,
                'metrics': fold_eval
            })
            
            # Collect predictions and labels for overall metrics
            if 'predictions' in fold_eval and 'labels' in fold_eval:
                all_predictions.extend(fold_eval['predictions'])
                all_labels.extend(fold_eval['labels'])
        
        # Calculate overall metrics
        if all_predictions and all_labels:
            cv_eval_results['overall_metrics'] = self._calculate_overall_metrics(all_predictions, all_labels)
        
        return cv_eval_results
    
    def _evaluate_fold(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate a single fold."""
        self.seo_model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict) and 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    continue
                
                outputs = self.seo_model(input_ids, attention_mask)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        return {
            'loss': avg_loss,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def _calculate_overall_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate overall metrics from all fold predictions."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_seo_classification(self, y_true: List[int], y_pred: List[int], 
                                  task_type: str = "seo_optimization") -> Dict[str, float]:
        """Evaluate SEO classification tasks with specialized metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support, roc_auc_score,
            confusion_matrix, classification_report, cohen_kappa_score
        )
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_recall_fscore_support(y_true, y_pred, average='macro')[0],
            'precision_weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted')[0],
            'recall_macro': precision_recall_fscore_support(y_true, y_pred, average='macro')[1],
            'recall_weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted')[1],
            'f1_macro': precision_recall_fscore_support(y_true, y_pred, average='macro')[2],
            'f1_weighted': precision_recall_fscore_support(y_true, y_pred, average='weighted')[2],
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Task-specific metrics
        if task_type == "seo_optimization":
            metrics.update(self._calculate_seo_optimization_metrics(y_true, y_pred))
        elif task_type == "content_quality":
            metrics.update(self._calculate_content_quality_classification_metrics(y_true, y_pred))
        elif task_type == "keyword_relevance":
            metrics.update(self._calculate_keyword_relevance_metrics(y_true, y_pred))
        
        return metrics
    
    def evaluate_seo_regression(self, y_true: List[float], y_pred: List[float],
                               task_type: str = "seo_score") -> Dict[str, float]:
        """Evaluate SEO regression tasks with specialized metrics."""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            mean_absolute_percentage_error, max_error
        )
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic regression metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'max_error': max_error(y_true, y_pred)
        }
        
        # Task-specific metrics
        if task_type == "seo_score":
            metrics.update(self._calculate_seo_score_regression_metrics(y_true, y_pred))
        elif task_type == "readability_score":
            metrics.update(self._calculate_readability_regression_metrics(y_true, y_pred))
        elif task_type == "content_quality_score":
            metrics.update(self._calculate_content_quality_regression_metrics(y_true, y_pred))
        
        return metrics
    
    def evaluate_seo_ranking(self, y_true: List[float], y_pred: List[float],
                            query_ids: Optional[List[str]] = None,
                            task_type: str = "content_ranking") -> Dict[str, float]:
        """Evaluate SEO ranking tasks with specialized metrics."""
        from sklearn.metrics import ndcg_score
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic ranking metrics
        metrics = {
            'ndcg_5': ndcg_score([y_true], [y_pred], k=5),
            'ndcg_10': ndcg_score([y_true], [y_pred], k=10),
            'ndcg_20': ndcg_score([y_true], [y_pred], k=20)
        }
        
        # Task-specific metrics
        if task_type == "content_ranking":
            metrics.update(self._calculate_content_ranking_metrics(y_true, y_pred))
        elif task_type == "keyword_ranking":
            metrics.update(self._calculate_keyword_ranking_metrics(y_true, y_pred))
        elif task_type == "seo_ranking":
            metrics.update(self._calculate_seo_ranking_metrics(y_true, y_pred))
        
        return metrics
    
    def evaluate_seo_content_quality(self, texts: List[str], 
                                   html_contents: Optional[List[str]] = None,
                                   task_type: str = "overall_quality") -> Dict[str, float]:
        """Evaluate SEO content quality with specialized metrics."""
        if html_contents is None:
            html_contents = [""] * len(texts)
        
        metrics = {
            'readability_scores': [],
            'keyword_density_scores': [],
            'technical_seo_scores': [],
            'content_structure_scores': [],
            'engagement_potential_scores': []
        }
        
        for text, html in zip(texts, html_contents):
            # Calculate individual content metrics
            content_metrics = self._calculate_single_content_metrics(text, html)
            
            metrics['readability_scores'].append(content_metrics['readability_score'])
            metrics['keyword_density_scores'].append(content_metrics['keyword_density_score'])
            metrics['technical_seo_scores'].append(content_metrics['technical_seo_score'])
            metrics['content_structure_scores'].append(content_metrics['content_structure_score'])
            metrics['engagement_potential_scores'].append(content_metrics['engagement_potential_score'])
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'avg_readability': np.mean(metrics['readability_scores']),
            'avg_keyword_density': np.mean(metrics['keyword_density_scores']),
            'avg_technical_seo': np.mean(metrics['technical_seo_scores']),
            'avg_content_structure': np.mean(metrics['content_structure_scores']),
            'avg_engagement_potential': np.mean(metrics['engagement_potential_scores']),
            'content_consistency': 1.0 - np.std(metrics['readability_scores']),
            'quality_distribution': {
                'high_quality': np.sum(np.array(metrics['readability_scores']) > 0.7),
                'medium_quality': np.sum((np.array(metrics['readability_scores']) <= 0.7) & 
                                       (np.array(metrics['readability_scores']) > 0.4)),
                'low_quality': np.sum(np.array(metrics['readability_scores']) <= 0.4)
            }
        }
        
        metrics.update(aggregate_metrics)
        return metrics
    
    def _calculate_seo_optimization_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate SEO-specific optimization metrics."""
        # SEO relevance metrics
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        metrics = {}
        
        # SEO precision (how many predicted SEO-optimized are actually good)
        if true_positives + false_positives > 0:
            metrics['seo_precision'] = true_positives / (true_positives + false_positives)
        else:
            metrics['seo_precision'] = 0.0
        
        # SEO recall (how many actual SEO-optimized were detected)
        if true_positives + false_negatives > 0:
            metrics['seo_recall'] = true_positives / (true_positives + false_negatives)
        else:
            metrics['seo_recall'] = 0.0
        
        # SEO F1 score
        if metrics['seo_precision'] + metrics['seo_recall'] > 0:
            metrics['seo_f1'] = 2 * (metrics['seo_precision'] * metrics['seo_recall']) / (metrics['seo_precision'] + metrics['seo_recall'])
        else:
            metrics['seo_f1'] = 0.0
        
        # SEO optimization accuracy
        metrics['seo_optimization_accuracy'] = (true_positives + np.sum(y_true == 0 and y_pred == 0)) / len(y_true)
        
        return metrics
    
    def _calculate_content_quality_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate content quality classification metrics."""
        # Content quality detection metrics
        high_quality_true = y_true == 1
        high_quality_pred = y_pred == 1
        
        metrics = {
            'content_quality_detection_accuracy': np.mean(high_quality_true == high_quality_pred),
            'high_quality_precision': np.sum((high_quality_true) & (high_quality_pred)) / (np.sum(high_quality_pred) + 1e-8),
            'high_quality_recall': np.sum((high_quality_true) & (high_quality_pred)) / (np.sum(high_quality_true) + 1e-8)
        }
        
        return metrics
    
    def _calculate_keyword_relevance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate keyword relevance classification metrics."""
        # Keyword relevance detection metrics
        relevant_true = y_true == 1
        relevant_pred = y_pred == 1
        
        metrics = {
            'keyword_relevance_detection_accuracy': np.mean(relevant_true == relevant_pred),
            'relevant_keyword_precision': np.sum((relevant_true) & (relevant_pred)) / (np.sum(relevant_pred) + 1e-8),
            'relevant_keyword_recall': np.sum((relevant_true) & (relevant_pred)) / (np.sum(relevant_true) + 1e-8)
        }
        
        return metrics
    
    def _calculate_seo_score_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate SEO score regression metrics."""
        # SEO score accuracy within threshold
        seo_threshold = 0.1  # 10% threshold for SEO scores
        within_threshold = np.abs(y_true - y_pred) <= seo_threshold
        
        metrics = {
            'seo_accuracy_within_threshold': np.mean(within_threshold),
            'high_quality_detection_accuracy': np.mean((y_true >= 0.7) == (y_pred >= 0.7)),
            'seo_score_correlation': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        }
        
        return metrics
    
    def _calculate_readability_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate readability regression metrics."""
        # Readability score accuracy within threshold
        readability_threshold = 0.15  # 15% threshold for readability scores
        within_threshold = np.abs(y_true - y_pred) <= readability_threshold
        
        metrics = {
            'readability_accuracy_within_threshold': np.mean(within_threshold),
            'readability_level_detection_accuracy': np.mean(
                (y_true >= 0.6) == (y_pred >= 0.6)  # High readability threshold
            ),
            'readability_correlation': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        }
        
        return metrics
    
    def _calculate_content_quality_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate content quality regression metrics."""
        # Content quality score accuracy within threshold
        quality_threshold = 0.12  # 12% threshold for quality scores
        within_threshold = np.abs(y_true - y_pred) <= quality_threshold
        
        metrics = {
            'quality_accuracy_within_threshold': np.mean(within_threshold),
            'high_quality_detection_accuracy': np.mean((y_true >= 0.7) == (y_pred >= 0.7)),
            'quality_correlation': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        }
        
        return metrics
    
    def _calculate_content_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate content ranking metrics."""
        # Top-k content relevance
        k_values = [1, 3, 5, 10]
        metrics = {}
        
        for k in k_values:
            if len(y_pred) >= k:
                top_k_indices = np.argsort(y_pred)[-k:]
                top_k_relevance = y_true[top_k_indices]
                metrics[f'top_{k}_content_relevance'] = np.mean(top_k_relevance)
                metrics[f'top_{k}_content_precision'] = np.mean(top_k_relevance >= 0.7)
        
        return metrics
    
    def _calculate_keyword_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate keyword ranking metrics."""
        # Top-k keyword relevance
        k_values = [1, 3, 5, 10]
        metrics = {}
        
        for k in k_values:
            if len(y_pred) >= k:
                top_k_indices = np.argsort(y_pred)[-k:]
                top_k_relevance = y_true[top_k_indices]
                metrics[f'top_{k}_keyword_relevance'] = np.mean(top_k_relevance)
                metrics[f'top_{k}_keyword_precision'] = np.mean(top_k_relevance >= 0.6)
        
        return metrics
    
    def _calculate_seo_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate SEO ranking metrics."""
        # Top-k SEO relevance
        k_values = [1, 3, 5, 10]
        metrics = {}
        
        for k in k_values:
            if len(y_pred) >= k:
                top_k_indices = np.argsort(y_pred)[-k:]
                top_k_relevance = y_true[top_k_indices]
                metrics[f'top_{k}_seo_relevance'] = np.mean(top_k_relevance)
                metrics[f'top_{k}_seo_precision'] = np.mean(top_k_relevance >= 0.7)
        
        return metrics
    
    def _calculate_single_content_metrics(self, text: str, html_content: str = "") -> Dict[str, float]:
        """Calculate metrics for a single piece of content."""
        # Readability score
        readability_score = self._calculate_enhanced_readability(text)
        
        # Keyword density score
        keyword_density_score = self._calculate_advanced_keyword_density(text)
        
        # Technical SEO score
        technical_seo_score = self._analyze_technical_seo(text)['technical_seo_score']
        
        # Content structure score
        content_structure_score = self._calculate_content_structure_score(text)
        
        # Engagement potential score
        engagement_potential_score = self._calculate_engagement_potential(text)
        
        return {
            'readability_score': readability_score,
            'keyword_density_score': keyword_density_score,
            'technical_seo_score': technical_seo_score,
            'content_structure_score': content_structure_score,
            'engagement_potential_score': engagement_potential_score
        }
    
    def _calculate_content_structure_score(self, text: str) -> float:
        """Calculate content structure score."""
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        # Sentence length variety
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        if sentence_lengths:
            sentence_variety = 1.0 - (np.std(sentence_lengths) / np.mean(sentence_lengths)) if np.mean(sentence_lengths) > 0 else 0.0
        else:
            sentence_variety = 0.0
        
        # Paragraph structure
        paragraph_scores = []
        for para in paragraphs:
            if para.strip():
                para_sentences = para.split('.')
                if len(para_sentences) >= 2:  # Good paragraph has multiple sentences
                    paragraph_scores.append(1.0)
                else:
                    paragraph_scores.append(0.5)
        
        paragraph_score = np.mean(paragraph_scores) if paragraph_scores else 0.0
        
        # Overall structure score
        structure_score = (sentence_variety * 0.4 + paragraph_score * 0.6)
        return min(max(structure_score, 0.0), 1.0)
    
    def _calculate_engagement_potential(self, text: str) -> float:
        """Calculate engagement potential score."""
        # Question marks indicate engagement
        question_count = text.count('?')
        question_score = min(question_count * 0.1, 0.3)
        
        # Exclamation marks indicate excitement
        exclamation_count = text.count('!')
        exclamation_score = min(exclamation_count * 0.05, 0.2)
        
        # Personal pronouns indicate engagement
        personal_pronouns = ['you', 'your', 'we', 'our', 'us']
        pronoun_count = sum(text.lower().count(pronoun) for pronoun in personal_pronouns)
        pronoun_score = min(pronoun_count * 0.05, 0.3)
        
        # Action words
        action_words = ['learn', 'discover', 'find', 'get', 'try', 'start', 'begin']
        action_count = sum(text.lower().count(word) for word in action_words)
        action_score = min(action_count * 0.03, 0.2)
        
        # Total engagement score
        engagement_score = question_score + exclamation_score + pronoun_score + action_score
        return min(max(engagement_score, 0.0), 1.0)
    
    def get_evaluation_summary(self, task_type: str, metrics: Dict[str, float]) -> str:
        """Generate a human-readable evaluation summary."""
        if task_type == "classification":
            return self._generate_classification_summary(metrics)
        elif task_type == "regression":
            return self._generate_regression_summary(metrics)
        elif task_type == "ranking":
            return self._generate_ranking_summary(metrics)
        elif task_type == "content_quality":
            return self._generate_content_quality_summary(metrics)
        else:
            return f"Evaluation summary for {task_type}: {json.dumps(metrics, indent=2)}"
    
    def _generate_classification_summary(self, metrics: Dict[str, float]) -> str:
        """Generate classification evaluation summary."""
        summary = f"""
Classification Performance Summary:
- Accuracy: {metrics.get('accuracy', 0):.3f}
- Precision (Weighted): {metrics.get('precision_weighted', 0):.3f}
- Recall (Weighted): {metrics.get('recall_weighted', 0):.3f}
- F1 Score (Weighted): {metrics.get('f1_weighted', 0):.3f}
- Cohen's Kappa: {metrics.get('cohen_kappa', 0):.3f}
"""
        
        if 'seo_f1' in metrics:
            summary += f"- SEO F1 Score: {metrics['seo_f1']:.3f}\n"
        
        return summary
    
    def _generate_regression_summary(self, metrics: Dict[str, float]) -> str:
        """Generate regression evaluation summary."""
        summary = f"""
Regression Performance Summary:
- R² Score: {metrics.get('r2', 0):.3f}
- RMSE: {metrics.get('rmse', 0):.3f}
- MAE: {metrics.get('mae', 0):.3f}
- MAPE: {metrics.get('mape', 0):.3f}
"""
        
        if 'seo_accuracy_within_threshold' in metrics:
            summary += f"- SEO Accuracy within Threshold: {metrics['seo_accuracy_within_threshold']:.3f}\n"
        
        return summary
    
    def _generate_ranking_summary(self, metrics: Dict[str, float]) -> str:
        """Generate ranking evaluation summary."""
        summary = f"""
Ranking Performance Summary:
- NDCG@5: {metrics.get('ndcg_5', 0):.3f}
- NDCG@10: {metrics.get('ndcg_10', 0):.3f}
- NDCG@20: {metrics.get('ndcg_20', 0):.3f}
"""
        
        # Add task-specific metrics
        for key, value in metrics.items():
            if key.startswith('top_') and key.endswith('_relevance'):
                summary += f"- {key.replace('_', ' ').title()}: {value:.3f}\n"
        
        return summary
    
    def _generate_content_quality_summary(self, metrics: Dict[str, float]) -> str:
        """Generate content quality evaluation summary."""
        summary = f"""
Content Quality Summary:
- Average Readability: {metrics.get('avg_readability', 0):.3f}
- Average Keyword Density: {metrics.get('avg_keyword_density', 0):.3f}
- Average Technical SEO: {metrics.get('avg_technical_seo', 0):.3f}
- Content Consistency: {metrics.get('content_consistency', 0):.3f}
"""
        
        if 'quality_distribution' in metrics:
            dist = metrics['quality_distribution']
            summary += f"- Quality Distribution: High({dist['high_quality']}), Medium({dist['medium_quality']}), Low({dist['low_quality']})\n"
        
        return summary
    
    def benchmark_data_loading(self, dataloader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
        """Benchmark DataLoader performance."""
        return self.data_loader_manager.benchmark_dataloader(dataloader, num_batches)
    
    def get_data_loading_stats(self) -> Dict[str, Any]:
        """Get statistics about data loading performance."""
        stats = {
            'num_datasets': len(self.data_loader_manager.datasets),
            'num_dataloaders': len(self.data_loader_manager.dataloaders),
            'config': {
                'batch_size': self.data_loader_manager.config.batch_size,
                'num_workers': self.data_loader_manager.config.num_workers,
                'pin_memory': self.data_loader_manager.config.pin_memory,
                'persistent_workers': self.data_loader_manager.config.persistent_workers,
                'prefetch_factor': self.data_loader_manager.config.prefetch_factor
            }
        }
        
        # Add benchmark results for each dataloader
        for name, dataloader in self.data_loader_manager.dataloaders.items():
            try:
                benchmark = self.benchmark_data_loading(dataloader, num_batches=5)
                stats[f'benchmark_{name}'] = benchmark
            except Exception as e:
                stats[f'benchmark_{name}_error'] = str(e)
        
        return stats
    
    def get_cross_validation_stats(self, name: str = "training") -> Dict[str, Any]:
        """Get cross-validation statistics and information."""
        return self.data_loader_manager.get_cross_validation_info(name)
    
    def validate_cross_validation_setup(self, name: str = "training") -> Dict[str, Any]:
        """Validate cross-validation setup and provide statistics."""
        return self.data_loader_manager.validate_cross_validation_setup(name)
    
    def get_all_split_types(self, name: str = "training") -> Dict[str, Any]:
        """Get information about all available split types for a dataset."""
        split_info = {
            'name': name,
            'available_splits': [],
            'cross_validation': {},
            'stratified': {},
            'three_way': {}
        }
        
        # Check for cross-validation
        cv_info = self.get_cross_validation_stats(name)
        if cv_info['total_folds'] > 0:
            split_info['cross_validation'] = cv_info
            split_info['available_splits'].append('cross_validation')
        
        # Check for other splits
        for split_name in ['train', 'val', 'test', 'stratified_train', 'stratified_val']:
            if f"{name}_{split_name}" in self.data_loader_manager.datasets:
                split_info['available_splits'].append(split_name)
        
        return split_info

    def cleanup(self):
        """Clean up all resources."""
        if self.diffusion_generator:
            self.diffusion_generator.cleanup()
        if self.pipeline_manager:
            self.pipeline_manager.cleanup()
        if self.data_loader_manager:
            self.data_loader_manager.cleanup()
        
        # Clean up model
        del self.seo_model
        del self.optimizer
        if self.scheduler:
            del self.scheduler
        if self.scaler:
            del self.scaler
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Enhanced Gradio Interface
def create_advanced_gradio_interface():
    """Create advanced Gradio interface for SEO engine."""
    
    # Initialize error handling and input validation
    error_handler = GradioErrorHandler()
    input_validator = InputValidator()
    error_boundary = GradioErrorBoundary(error_handler)
    
    # Initialize engine with advanced config
    config = SEOConfig(
        use_mixed_precision=True,
        batch_size=8,
        max_length=512,
        num_attention_heads=8,
        use_lora=True,
        use_p_tuning=True,
        use_diffusion=True,
        scheduler_type="ddim",
        diffusion_guidance_scale=7.5,
        diffusion_steps=50,
        eta=0.0
    )
    engine = AdvancedLLMSEOEngine(config)
    
    async def analyze_seo_advanced(text):
        """Advanced SEO analysis with attention mechanisms and efficient fine-tuning."""
        await engine.initialize_models()
        return engine.analyze_seo_score(text)
    
    async def optimize_content_advanced(content, keywords):
        """Advanced content optimization."""
        await engine.initialize_models()
        keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
        return await engine.optimize_content(content, keywords_list)
    
    async def batch_analyze_advanced(texts):
        """Batch analysis of multiple texts."""
        await engine.initialize_models()
        texts_list = [t.strip() for t in texts.split('\n\n') if t.strip()]
        return engine.batch_analyze(texts_list)
    
    # Create advanced interface
    with gr.Blocks(title="Advanced LLM SEO Engine", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🚀 Advanced LLM SEO Engine")
        gr.Markdown("### Powered by Transformers, PyTorch, and Advanced Deep Learning")
        gr.Markdown("#### Enhanced with Attention Mechanisms, Positional Encodings, Efficient Fine-tuning, and Diffusion Models")
        
        with gr.Tab("🔍 SEO Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Content to Analyze",
                        placeholder="Enter your content here for comprehensive SEO analysis...",
                        lines=12
                    )
                    analyze_btn = gr.Button("🚀 Analyze SEO", variant="primary")
                
                with gr.Column(scale=1):
                    score_output = gr.Number(label="SEO Score", precision=3)
                    confidence_output = gr.Number(label="Confidence", precision=3)
                    keyword_analysis_output = gr.JSON(label="Keyword Analysis")
                    readability_output = gr.JSON(label="Readability Analysis")
                    quality_output = gr.JSON(label="Quality Analysis")
                    attention_output = gr.JSON(label="Attention Analysis")
            
            analyze_btn.click(
                analyze_seo_advanced,
                inputs=[text_input],
                outputs=[score_output, confidence_output, keyword_analysis_output, readability_output, quality_output, attention_output]
            )
        
        with gr.Tab("⚡ Content Optimization"):
            with gr.Row():
                with gr.Column():
                    content_input = gr.Textbox(
                        label="Original Content",
                        placeholder="Enter your content for optimization...",
                        lines=10
                    )
                    keywords_input = gr.Textbox(
                        label="Target Keywords (comma-separated)",
                        placeholder="keyword1, keyword2, keyword3"
                    )
                    optimize_btn = gr.Button("⚡ Optimize Content", variant="primary")
                
                with gr.Column():
                    optimized_output = gr.Textbox(
                        label="Optimized Content",
                        lines=10
                    )
                    improvement_output = gr.Number(label="Score Improvement", precision=3)
                    keyword_optimization_output = gr.JSON(label="Keyword Optimization")
                    recommendations_output = gr.JSON(label="Recommendations")
            
            optimize_btn.click(
                optimize_content_advanced,
                inputs=[content_input, keywords_input],
                outputs=[optimized_output, improvement_output, keyword_optimization_output, recommendations_output]
            )
        
        with gr.Tab("📊 Batch Analysis"):
            with gr.Row():
                with gr.Column():
                    batch_texts_input = gr.Textbox(
                        label="Multiple Texts (separated by double newlines)",
                        placeholder="Text 1\n\nText 2\n\nText 3",
                        lines=15
                    )
                    batch_analyze_btn = gr.Button("📊 Batch Analyze", variant="primary")
                
                with gr.Column():
                    batch_results_output = gr.JSON(label="Batch Analysis Results")
            
            batch_analyze_btn.click(
                batch_analyze_advanced,
                inputs=[batch_texts_input],
                outputs=[batch_results_output]
            )
        
        with gr.Tab("🎨 Advanced Visual Content Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    visual_prompt_input = gr.Textbox(
                        label="Content Description for Visual Generation",
                        placeholder="Describe the visual content you want to generate...",
                        lines=8
                    )
                    keywords_input = gr.Textbox(
                        label="Target Keywords (comma-separated)",
                        placeholder="keyword1, keyword2, keyword3"
                    )
                    
                    with gr.Row():
                        scheduler_dropdown = gr.Dropdown(
                            choices=["ddim", "dpm", "euler", "heun", "kdpm2", "lms", "pndm", "unipc"],
                            value="ddim",
                            label="Scheduler Type",
                            info="Choose the diffusion scheduler for different generation styles"
                        )
                        guidance_scale_slider = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale",
                            info="Higher values = more prompt adherence"
                        )
                    
                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Generation Steps",
                            info="More steps = higher quality, slower generation"
                        )
                        eta_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.1,
                            label="DDIM Eta",
                            info="Controls randomness in DDIM scheduler"
                        )
                    
                    generate_visual_btn = gr.Button("🎨 Generate Visual Content", variant="primary")
                
                with gr.Column(scale=1):
                    visual_output = gr.Image(label="Generated Visual Content")
                    generation_info_output = gr.JSON(label="Generation Information")
                    pipeline_status = gr.JSON(label="Pipeline Status")
            
            def advanced_generate_visual(prompt, keywords, scheduler, guidance, steps, eta):
                """Advanced visual generation with custom parameters."""
                if not engine.diffusion_generator:
                    return None, {"error": "Diffusion not enabled"}, {"status": "disabled"}
                
                # Update config temporarily
                engine.config.scheduler_type = scheduler
                engine.config.diffusion_guidance_scale = guidance
                engine.config.diffusion_steps = steps
                engine.config.eta = eta
                
                # Reconfigure scheduler if changed
                engine.diffusion_generator._configure_scheduler()
                
                # Generate content
                result = engine.diffusion_generator.generate_seo_visual_content(
                    prompt, 
                    {}, 
                    [k.strip() for k in keywords.split(',') if k.strip()]
                )
                
                # Get pipeline status
                status = {
                    "scheduler": scheduler,
                    "guidance_scale": guidance,
                    "steps": steps,
                    "eta": eta,
                    "device": str(engine.device),
                    "mixed_precision": engine.config.use_mixed_precision
                }
                
                return result.get("images", [None])[0] if result.get("images") else None, result, status
            
            generate_visual_btn.click(
                advanced_generate_visual,
                inputs=[visual_prompt_input, keywords_input, scheduler_dropdown, 
                       guidance_scale_slider, steps_slider, eta_slider],
                outputs=[visual_output, generation_info_output, pipeline_status]
            )
        
        with gr.Tab("⚙️ Diffusion Model Management"):
            with gr.Row():
                with gr.Column():
                    model_info_btn = gr.Button("📊 Model Information", variant="secondary")
                    scheduler_info_btn = gr.Button("🔄 Scheduler Status", variant="secondary")
                    optimize_btn = gr.Button("🚀 Optimize Pipeline", variant="primary")
                    clear_cache_btn = gr.Button("🧹 Clear GPU Cache", variant="secondary")
                
                with gr.Column():
                    model_info_output = gr.JSON(label="Model Information")
                    scheduler_info_output = gr.JSON(label="Scheduler Status")
                    optimization_output = gr.JSON(label="Optimization Results")
        
        with gr.Tab("🧠 Diffusion Process Understanding"):
            with gr.Row():
                with gr.Column():
                    diffusion_explanation_btn = gr.Button("📚 Understand Diffusion", variant="secondary")
                    diffusion_math_btn = gr.Button("🧮 Mathematical Demo", variant="secondary")
                    trajectory_btn = gr.Button("🔄 View Trajectories", variant="primary")
                    noise_schedule_btn = gr.Button("📊 Noise Schedule Info", variant="secondary")
                
                with gr.Column():
                    diffusion_explanation_output = gr.JSON(label="Diffusion Process Explanation")
                    diffusion_math_output = gr.JSON(label="Mathematical Demonstration")
                    trajectory_output = gr.Gallery(label="Diffusion Trajectories", columns=5, rows=2)
                    noise_schedule_output = gr.JSON(label="Noise Schedule Information")
            
            def get_diffusion_explanation():
                """Get comprehensive explanation of diffusion processes."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                return engine.diffusion_generator.understand_diffusion_processes()
            
            def get_diffusion_math():
                """Demonstrate diffusion mathematics."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                # Create a dummy image for demonstration
                dummy_image = torch.randn(1, 3, 64, 64, device=engine.device)
                return engine.diffusion_generator.demonstrate_diffusion_math(dummy_image)
            
            def get_trajectories():
                """Generate and display diffusion trajectories."""
                if not engine.diffusion_generator:
                    return []
                
                # Create a dummy image for trajectory demonstration
                dummy_image = torch.randn(1, 3, 64, 64, device=engine.device)
                
                # Forward trajectory (adding noise)
                forward_traj = engine.diffusion_generator.diffusion_model.sample_trajectory(dummy_image, 5)
                
                # Convert tensors to images for display
                trajectory_images = []
                for i, img_tensor in enumerate(forward_traj):
                    # Convert tensor to PIL image (simplified)
                    img_array = img_tensor.squeeze(0).cpu().numpy()
                    img_array = np.clip(img_array, 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                    img_array = np.transpose(img_array, (1, 2, 0))
                    trajectory_images.append(img_array)
                
                return trajectory_images
            
            def get_noise_schedule_info():
                """Get noise schedule information."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                pipeline = engine.diffusion_generator.pipeline
                if not pipeline:
                    return {"error": "Pipeline not initialized"}
                
                return {
                    "scheduler_type": engine.config.scheduler_type,
                    "num_train_timesteps": getattr(pipeline.scheduler, 'num_train_timesteps', 'N/A'),
                    "beta_start": getattr(pipeline.scheduler, 'beta_start', 'N/A'),
                    "beta_end": getattr(pipeline.scheduler, 'beta_end', 'N/A'),
                    "beta_schedule": getattr(pipeline.scheduler, 'beta_schedule', 'N/A'),
                    "alphas_cumprod_shape": pipeline.scheduler.alphas_cumprod.shape if hasattr(pipeline.scheduler, 'alphas_cumprod') else 'N/A'
                }
            
            # Connect buttons to functions
            diffusion_explanation_btn.click(get_diffusion_explanation, outputs=[diffusion_explanation_output])
            diffusion_math_btn.click(get_diffusion_math, outputs=[diffusion_math_output])
            trajectory_btn.click(get_trajectories, outputs=[trajectory_output])
            noise_schedule_btn.click(get_noise_schedule_info, outputs=[noise_schedule_output])
        
        with gr.Tab("🚀 Pipeline Management"):
            with gr.Row():
                with gr.Column():
                    pipeline_dropdown = gr.Dropdown(
                        choices=engine.get_available_pipelines(),
                        value="stable-diffusion",
                        label="Select Pipeline Type",
                        interactive=True
                    )
                    switch_pipeline_btn = gr.Button("🔄 Switch Pipeline", variant="primary")
                    pipeline_info_btn = gr.Button("📊 Pipeline Information", variant="secondary")
                    generate_pipeline_btn = gr.Button("🎨 Generate with Pipeline", variant="primary")
                    cleanup_pipelines_btn = gr.Button("🧹 Cleanup Pipelines", variant="secondary")
                
                with gr.Column():
                    pipeline_status_output = gr.JSON(label="Pipeline Status")
                    pipeline_info_output = gr.JSON(label="Pipeline Information")
                    pipeline_generation_output = gr.JSON(label="Generation Results")
            
            with gr.Row():
                with gr.Column():
                    pipeline_prompt_input = gr.Textbox(
                        label="Generation Prompt",
                        placeholder="Enter prompt for image generation...",
                        lines=3
                    )
                    pipeline_params_input = gr.JSON(
                        label="Additional Parameters (optional)",
                        value={},
                        interactive=True
                    )
                
                with gr.Column():
                    pipeline_result_output = gr.Gallery(label="Generated Images", columns=3, rows=2)
            
            def switch_pipeline(pipeline_type):
                """Switch to a different diffusion pipeline."""
                return engine.switch_pipeline(pipeline_type)
            
            def get_pipeline_info(pipeline_type):
                """Get information about the selected pipeline."""
                return engine.get_pipeline_info(pipeline_type)
            
            def generate_with_pipeline(pipeline_type, prompt, params):
                """Generate content using the selected pipeline."""
                if not params:
                    params = {}
                return engine.generate_with_pipeline(pipeline_type, prompt, **params)
            
            def cleanup_pipelines():
                """Clean up pipeline resources."""
                engine.cleanup_pipelines()
                return {"status": "success", "message": "Pipelines cleaned up"}
            
            # Connect buttons to functions
            switch_pipeline_btn.click(switch_pipeline, inputs=[pipeline_dropdown], outputs=[pipeline_status_output])
            pipeline_info_btn.click(get_pipeline_info, inputs=[pipeline_dropdown], outputs=[pipeline_info_output])
            generate_pipeline_btn.click(
                generate_with_pipeline,
                inputs=[pipeline_dropdown, pipeline_prompt_input, pipeline_params_input],
                outputs=[pipeline_generation_output]
            )
            cleanup_pipelines_btn.click(cleanup_pipelines, outputs=[pipeline_status_output])
        
        with gr.Tab("🎯 Diffusion Training & Evaluation"):
            with gr.Row():
                with gr.Column():
                    training_demo_btn = gr.Button("🚀 Training Demo", variant="primary")
                    evaluation_btn = gr.Button("📊 Evaluate Understanding", variant="secondary")
                    visualization_btn = gr.Button("🖼️ Visualize Processes", variant="secondary")
                    batch_size_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        label="Batch Size for Demo",
                        info="Number of images to process in demonstration"
                    )
                
                with gr.Column():
                    training_output = gr.JSON(label="Training Step Results")
                    evaluation_output = gr.JSON(label="Evaluation Results")
                    visualization_output = gr.Gallery(label="Process Visualizations", columns=4, rows=2)
        
        with gr.Tab("⚙️ Advanced Noise Schedulers & Sampling"):
            with gr.Row():
                with gr.Column():
                    scheduler_analysis_btn = gr.Button("📊 Analyze Schedulers", variant="primary")
                    optimal_scheduler_btn = gr.Button("🎯 Find Optimal Scheduler", variant="secondary")
                    sampling_methods_btn = gr.Button("🔄 Sampling Methods", variant="secondary")
                    noise_schedule_analysis_btn = gr.Button("📈 Noise Schedule Analysis", variant="secondary")
                    
                    with gr.Row():
                        scheduler_type_dropdown = gr.Dropdown(
                            choices=["ddim", "dpm", "euler", "heun", "kdpm2", "lms", "pndm", "unipc"],
                            value="ddim",
                            label="Scheduler Type",
                            info="Select scheduler for detailed analysis"
                        )
                        use_case_dropdown = gr.Dropdown(
                            choices=["speed_priority", "quality_priority", "memory_constrained", "production", "research"],
                            value="production",
                            label="Use Case",
                            info="Select your primary use case for recommendations"
                        )
                
                with gr.Column():
                    scheduler_analysis_output = gr.JSON(label="Scheduler Analysis")
                    optimal_scheduler_output = gr.JSON(label="Optimal Scheduler Recommendations")
                    sampling_methods_output = gr.JSON(label="Sampling Methods Analysis")
                    noise_schedule_output = gr.JSON(label="Noise Schedule Analysis")
            
            def analyze_schedulers():
                """Analyze all available noise schedulers."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                return engine.diffusion_generator.analyze_noise_schedulers()
            
            def get_optimal_scheduler(use_case):
                """Get optimal scheduler recommendations for specific use case."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                analysis = engine.diffusion_generator.analyze_noise_schedulers()
                use_cases = analysis.get("optimal_choice", {})
                
                if use_case in use_cases:
                    return {
                        "use_case": use_case,
                        "recommendation": use_cases[use_case],
                        "all_recommendations": use_cases
                    }
                else:
                    return {"error": f"Use case '{use_case}' not found"}
            
            def analyze_sampling_methods():
                """Analyze available sampling methods and their characteristics."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                return {
                    "sampling_methods": {
                        "deterministic": {
                            "description": "Deterministic sampling with fixed seed",
                            "advantages": ["Reproducible results", "Consistent quality"],
                            "disadvantages": ["Less diverse outputs", "Fixed patterns"],
                            "best_for": ["Production systems", "Debugging", "Consistent results"]
                        },
                        "stochastic": {
                            "description": "Stochastic sampling with random elements",
                            "advantages": ["Diverse outputs", "Creative variations"],
                            "disadvantages": ["Less predictable", "Variable quality"],
                            "best_for": ["Creative applications", "Exploration", "Variety generation"]
                        },
                        "adaptive": {
                            "description": "Adaptive sampling with dynamic step sizes",
                            "advantages": ["Optimal performance", "Quality-speed balance"],
                            "disadvantages": ["Complex implementation", "Computational overhead"],
                            "best_for": ["High-performance systems", "Research", "Custom applications"]
                        }
                    },
                    "current_method": "adaptive" if engine.diffusion_generator.diffusion_model else "standard",
                    "recommendations": [
                        "Use deterministic sampling for production systems",
                        "Use stochastic sampling for creative applications",
                        "Use adaptive sampling for optimal performance"
                    ]
                }
            
            def analyze_noise_schedule():
                """Analyze current noise schedule characteristics."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                if not engine.diffusion_generator.diffusion_model:
                    return {"error": "Diffusion model not initialized"}
                
                return engine.diffusion_generator.diffusion_model.get_noise_schedule_info()
            
            # Connect buttons to functions
            scheduler_analysis_btn.click(analyze_schedulers, outputs=[scheduler_analysis_output])
            optimal_scheduler_btn.click(get_optimal_scheduler, inputs=[use_case_dropdown], outputs=[optimal_scheduler_output])
            sampling_methods_btn.click(analyze_sampling_methods, outputs=[sampling_methods_output])
            noise_schedule_analysis_btn.click(analyze_noise_schedule, outputs=[noise_schedule_output])
            
            def run_training_demo(batch_size):
                """Demonstrate diffusion training process."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                try:
                    # Create dummy training data
                    clean_images = torch.randn(batch_size, 3, 64, 64, device=engine.device)
                    timesteps = torch.randint(0, 1000, (batch_size,), device=engine.device)
                    
                    # Run training step
                    training_info = engine.diffusion_generator.train_diffusion_step(clean_images, timesteps)
                    
                    # Convert tensors to serializable format
                    serializable_info = {}
                    for key, value in training_info.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                serializable_info[key] = value.item()
                            else:
                                serializable_info[key] = f"Tensor shape: {value.shape}"
                        else:
                            serializable_info[key] = value
                    
                    return serializable_info
                except Exception as e:
                    return {"error": f"Training demo failed: {str(e)}"}
            
            def run_evaluation(batch_size):
                """Evaluate diffusion process understanding."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                try:
                    # Create dummy test data
                    test_images = torch.randn(batch_size, 3, 64, 64, device=engine.device)
                    
                    # Run evaluation
                    evaluation_results = engine.diffusion_generator.evaluate_diffusion_understanding(test_images)
                    
                    # Convert tensors to serializable format
                    serializable_results = {}
                    for key, value in evaluation_results.items():
                        if isinstance(value, torch.Tensor):
                            serializable_results[key] = f"Tensor shape: {value.shape}"
                        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                            serializable_results[key] = f"List of {len(value)} tensors, shape: {value[0].shape}"
                        else:
                            serializable_results[key] = value
                    
                    return serializable_results
                except Exception as e:
                    return {"error": f"Evaluation failed: {str(e)}"}
            
            def run_visualization(batch_size):
                """Visualize diffusion processes."""
                if not engine.diffusion_generator:
                    return []
                
                try:
                    # Create dummy sample data
                    sample_images = torch.randn(batch_size, 3, 64, 64, device=engine.device)
                    
                    # Get visualizations
                    visualizations = engine.diffusion_generator.visualize_diffusion_processes(sample_images, 6)
                    
                    # Convert tensors to images for display
                    all_images = []
                    
                    # Forward diffusion images
                    for i, img_tensor in enumerate(visualizations["forward_diffusion"]):
                        img_array = img_tensor.squeeze(0).cpu().numpy()
                        img_array = np.clip(img_array, 0, 1)
                        img_array = (img_array * 255).astype(np.uint8)
                        img_array = np.transpose(img_array, (1, 2, 0))
                        all_images.append(img_array)
                    
                    # Reverse diffusion images
                    for i, img_tensor in enumerate(visualizations["reverse_diffusion"]):
                        img_array = img_tensor.squeeze(0).cpu().numpy()
                        img_array = np.clip(img_array, 0, 1)
                        img_array = (img_array * 255).astype(np.uint8)
                        img_array = np.transpose(img_array, (1, 2, 0))
                        all_images.append(img_array)
                    
                    return all_images
                except Exception as e:
                    return []
            
            # Connect buttons to functions
            training_demo_btn.click(run_training_demo, inputs=[batch_size_slider], outputs=[training_output])
            evaluation_btn.click(run_evaluation, inputs=[batch_size_slider], outputs=[evaluation_output])
            visualization_btn.click(run_visualization, inputs=[batch_size_slider], outputs=[visualization_output])
            
            def get_model_info():
                """Get comprehensive model information."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                pipeline = engine.diffusion_generator.pipeline
                if not pipeline:
                    return {"error": "Pipeline not initialized"}
                
                info = {
                    "model_name": engine.config.diffusion_model_name,
                    "device": str(engine.device),
                    "dtype": str(pipeline.unet.dtype),
                    "unet_parameters": sum(p.numel() for p in pipeline.unet.parameters()),
                    "vae_parameters": sum(p.numel() for p in pipeline.vae.parameters()),
                    "text_encoder_parameters": sum(p.numel() for p in pipeline.text_encoder.parameters()),
                    "total_parameters": sum(p.numel() for p in pipeline.parameters()),
                    "memory_usage": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if engine.device.type == "cuda" else "N/A",
                    "mixed_precision": engine.config.use_mixed_precision,
                    "attention_slicing": hasattr(pipeline, '_attention_slicing_enabled'),
                    "vae_slicing": hasattr(pipeline, '_vae_slicing_enabled')
                }
                return info
            
            def get_scheduler_info():
                """Get scheduler configuration and status."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                pipeline = engine.diffusion_generator.pipeline
                if not pipeline:
                    return {"error": "Pipeline not initialized"}
                
                scheduler = pipeline.scheduler
                info = {
                    "scheduler_type": engine.config.scheduler_type,
                    "scheduler_class": scheduler.__class__.__name__,
                    "num_train_timesteps": getattr(scheduler, 'num_train_timesteps', 'N/A'),
                    "beta_start": getattr(scheduler, 'beta_start', 'N/A'),
                    "beta_end": getattr(scheduler, 'beta_end', 'N/A'),
                    "beta_schedule": getattr(scheduler, 'beta_schedule', 'N/A'),
                    "algorithm_type": getattr(scheduler, 'algorithm_type', 'N/A'),
                    "solver_type": getattr(scheduler, 'solver_type', 'N/A'),
                    "current_config": str(scheduler.config)
                }
                return info
            
            def optimize_pipeline():
                """Apply advanced optimizations to the pipeline."""
                if not engine.diffusion_generator:
                    return {"error": "Diffusion not enabled"}
                
                try:
                    pipeline = engine.diffusion_generator.pipeline
                    
                    # Apply memory optimizations
                    if engine.device.type == "cuda":
                        torch.cuda.empty_cache()
                        if hasattr(pipeline, 'enable_model_cpu_offload'):
                            pipeline.enable_model_cpu_offload()
                        if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                            pipeline.enable_sequential_cpu_offload()
                    
                    # Enable attention optimizations
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing()
                    if hasattr(pipeline, 'enable_vae_slicing'):
                        pipeline.enable_vae_slicing()
                    
                    # Try to enable xformers if available
                    try:
                        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                            pipeline.enable_xformers_memory_efficient_attention()
                            xformers_enabled = True
                        else:
                            xformers_enabled = False
                    except:
                        xformers_enabled = False
                    
                    return {
                        "status": "success",
                        "optimizations_applied": [
                            "memory_cache_cleared",
                            "model_cpu_offload",
                            "sequential_cpu_offload",
                            "attention_slicing",
                            "vae_slicing",
                            f"xformers_attention: {xformers_enabled}"
                        ],
                        "memory_after": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if engine.device.type == "cuda" else "N/A"
                    }
                except Exception as e:
                    return {"error": f"Optimization failed: {str(e)}"}
            
            def clear_gpu_cache():
                """Clear GPU memory cache."""
                if engine.device.type == "cuda":
                    torch.cuda.empty_cache()
                    return {"status": "success", "message": "GPU cache cleared"}
                else:
                    return {"status": "info", "message": "Not using GPU"}
            
            # Connect buttons to functions
            model_info_btn.click(get_model_info, outputs=[model_info_output])
            scheduler_info_btn.click(get_scheduler_info, outputs=[scheduler_info_output])
            optimize_btn.click(optimize_pipeline, outputs=[optimization_output])
            clear_cache_btn.click(clear_gpu_cache, outputs=[optimization_output])
        
        with gr.Tab("📊 Data Loading & Training"):
            with gr.Row():
                with gr.Column():
                    training_texts_input = gr.Textbox(
                        label="Training Texts (one per line)",
                        placeholder="Enter training texts here...",
                        lines=10
                    )
                    training_labels_input = gr.Textbox(
                        label="Training Labels (comma-separated, optional)",
                        placeholder="0,1,0,1... (leave empty for default)",
                        lines=2
                    )
                    metadata_input = gr.Textbox(
                        label="Metadata (JSON format, optional)",
                        placeholder='{"group": ["A", "B", "A", "B"], "domain": ["tech", "health", "tech", "health"]}',
                        lines=3
                    )
                    dataset_name_input = gr.Textbox(
                        label="Dataset Name",
                        value="training",
                        lines=1
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            val_split_input = gr.Slider(
                                minimum=0.1,
                                maximum=0.5,
                                value=0.2,
                                step=0.05,
                                label="Validation Split"
                            )
                            test_split_input = gr.Slider(
                                minimum=0.1,
                                maximum=0.3,
                                value=0.15,
                                step=0.05,
                                label="Test Split"
                            )
                        
                        with gr.Column():
                            cv_folds_input = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=5,
                                step=1,
                                label="CV Folds"
                            )
                            cv_strategy_input = gr.Dropdown(
                                choices=["stratified", "kfold", "timeseries", "group", "stratified_group", 
                                        "leave_one_out", "leave_p_out", "repeated_stratified", "repeated_kfold"],
                                value="stratified",
                                label="CV Strategy"
                            )
                    
                    create_dataset_btn = gr.Button("📚 Create Dataset", variant="primary")
                    create_dataloaders_btn = gr.Button("🔄 Create DataLoaders", variant="primary")
                    create_cv_dataloaders_btn = gr.Button("🔄 Create CV DataLoaders", variant="primary")
                    create_three_way_btn = gr.Button("📊 Create 3-Way Split", variant="primary")
                    train_epoch_btn = gr.Button("🚀 Train Epoch", variant="primary")
                    train_cv_btn = gr.Button("🚀 Train Cross-Validation", variant="primary")
                    benchmark_btn = gr.Button("⚡ Benchmark Performance", variant="secondary")
                
                with gr.Column():
                    dataset_status_output = gr.JSON(label="Dataset Status")
                    dataloader_status_output = gr.JSON(label="DataLoader Status")
                    cv_status_output = gr.JSON(label="Cross-Validation Status")
                    training_output_output = gr.JSON(label="Training Output")
                    cv_training_output = gr.JSON(label="CV Training Output")
                    benchmark_output_output = gr.JSON(label="Benchmark Results")
            
            with gr.Row():
                with gr.Column():
                    batch_size_input = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=32,
                        step=1,
                        label="Batch Size"
                    )
                    num_workers_input = gr.Slider(
                        minimum=0,
                        maximum=16,
                        value=4,
                        step=1,
                        label="Number of Workers"
                    )
                    pin_memory_input = gr.Checkbox(
                        label="Pin Memory",
                        value=True
                    )
                    persistent_workers_input = gr.Checkbox(
                        label="Persistent Workers",
                        value=True
                    )
                    update_config_btn = gr.Button("⚙️ Update Config", variant="secondary")
                
                with gr.Column():
                    config_status_output = gr.JSON(label="Configuration Status")
                    stats_output_output = gr.JSON(label="Data Loading Statistics")
            
            # Early Stopping and Learning Rate Scheduling Controls
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🛑 Early Stopping Configuration")
                    early_stopping_patience_input = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Early Stopping Patience"
                    )
                    early_stopping_min_delta_input = gr.Slider(
                        minimum=0.0001,
                        maximum=0.01,
                        value=0.001,
                        step=0.0001,
                        label="Minimum Delta for Improvement"
                    )
                    early_stopping_monitor_input = gr.Dropdown(
                        choices=["val_loss", "val_accuracy", "train_loss"],
                        value="val_loss",
                        label="Monitor Metric"
                    )
                    early_stopping_mode_input = gr.Dropdown(
                        choices=["min", "max"],
                        value="min",
                        label="Mode (min for loss, max for accuracy)"
                    )
                
                with gr.Column():
                    gr.Markdown("### 📈 Learning Rate Scheduling")
                    lr_scheduler_input = gr.Dropdown(
                        choices=["cosine", "linear", "exponential", "step", "plateau"],
                        value="cosine",
                        label="LR Scheduler Type"
                    )
                    learning_rate_input = gr.Slider(
                        minimum=1e-6,
                        maximum=1e-2,
                        value=2e-5,
                        step=1e-6,
                        label="Learning Rate"
                    )
                    warmup_steps_input = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        value=100,
                        step=10,
                        label="Warmup Steps"
                    )
                    max_grad_norm_input = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="Max Gradient Norm"
                    )
            
            # Training with Early Stopping Button
            train_with_early_stopping_btn = gr.Button("🚀 Train with Early Stopping", variant="primary")
            get_lr_info_btn = gr.Button("📊 Get LR Info", variant="secondary")
            
            def create_dataset(texts, labels, name):
                try:
                    if not texts.strip():
                        return {"error": "No training texts provided"}
                    
                    text_list = [t.strip() for t in texts.split('\n') if t.strip()]
                    label_list = None
                    
                    if labels.strip():
                        label_list = [int(l.strip()) for l in labels.split(',') if l.strip()]
                        if len(label_list) != len(text_list):
                            return {"error": f"Number of labels ({len(label_list)}) doesn't match number of texts ({len(text_list)})"}
                    
                    dataset = engine.create_training_dataset(text_list, label_list, name)
                    return {
                        "success": True,
                        "dataset_name": name,
                        "num_samples": len(dataset),
                        "num_labels": len(set(label_list)) if label_list else 1
                    }
                except Exception as e:
                    return {"error": str(e)}
            
            def create_dataloaders(name, val_split):
                try:
                    dataset = engine.data_loader_manager.get_dataset(name)
                    if not dataset:
                        return {"error": f"Dataset '{name}' not found. Create it first."}
                    
                    train_loader, val_loader = engine.create_training_dataloaders(
                        [], [], name, val_split
                    )
                    
                    return {
                        "success": True,
                        "train_loader": f"{name}_train",
                        "val_loader": f"{name}_val",
                        "train_batches": len(train_loader),
                        "val_batches": len(val_loader),
                        "batch_size": train_loader.batch_size
                    }
                except Exception as e:
                    return {"error": str(e)}
            
            def train_epoch():
                try:
                    train_loader = engine.data_loader_manager.get_dataloader("training_train")
                    val_loader = engine.data_loader_manager.get_dataloader("training_val")
                    
                    if not train_loader:
                        return {"error": "Training DataLoader not found. Create dataloaders first."}
                    
                    results = engine.train_epoch(train_loader, val_loader)
                    return results
                except Exception as e:
                    return {"error": str(e)}
            
            def benchmark_performance():
                try:
                    train_loader = engine.data_loader_manager.get_dataloader("training_train")
                    if not train_loader:
                        return {"error": "Training DataLoader not found. Create dataloaders first."}
                    
                    benchmark = engine.benchmark_data_loading(train_loader)
                    return benchmark
                except Exception as e:
                    return {"error": str(e)}
            
            def update_config(batch_size, num_workers, pin_memory, persistent_workers):
                try:
                    new_config = DataLoaderConfig(
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        prefetch_factor=2
                    )
                    
                    # Update existing dataloaders with new config
                    for name, dataloader in engine.data_loader_manager.dataloaders.items():
                        if "train" in name:
                            engine.data_loader_manager.create_dataloader(name, 
                                engine.data_loader_manager.get_dataset(name.replace("_train", "")), 
                                new_config)
                    
                    return {"success": True, "config_updated": True}
                except Exception as e:
                    return {"error": str(e)}
            
            def get_stats():
                try:
                    stats = engine.get_data_loading_stats()
                    return stats
                except Exception as e:
                    return {"error": str(e)}
            
            def train_with_early_stopping(patience, min_delta, monitor, mode, lr_scheduler, 
                                        learning_rate, warmup_steps, max_grad_norm, max_epochs=50):
                """Train model with early stopping and learning rate scheduling."""
                try:
                    # Update engine config
                    engine.config.early_stopping_patience = patience
                    engine.config.early_stopping_min_delta = min_delta
                    engine.config.early_stopping_monitor = monitor
                    engine.config.early_stopping_mode = mode
                    engine.config.lr_scheduler = lr_scheduler
                    engine.config.learning_rate = learning_rate
                    engine.config.warmup_steps = warmup_steps
                    engine.config.max_grad_norm = max_grad_norm
                    
                    # Get dataloaders
                    train_loader = engine.data_loader_manager.get_dataloader("training_train")
                    val_loader = engine.data_loader_manager.get_dataloader("training_val")
                    
                    if not train_loader or not val_loader:
                        return {"error": "Training DataLoaders not found. Create dataloaders first."}
                    
                    # Reinitialize optimizer and scheduler with new config
                    engine._initialize_optimizer()
                    engine._initialize_scheduler()
                    
                    # Train with early stopping
                    results = engine.train_with_early_stopping(train_loader, val_loader, max_epochs)
                    
                    return {
                        "success": True,
                        "epochs_completed": results['epochs_completed'],
                        "best_val_loss": results['best_val_loss'],
                        "best_epoch": results['best_epoch'],
                        "early_stopping_triggered": results['early_stopping_triggered'],
                        "final_learning_rate": results['training_history'][-1]['learning_rate'] if results['training_history'] else None
                    }
                except Exception as e:
                    return {"error": str(e)}
            
            def get_learning_rate_info():
                """Get current learning rate information."""
                try:
                    lr_info = engine.get_learning_rate_info()
                    return lr_info
                except Exception as e:
                    return {"error": str(e)}
            
            # Connect buttons to functions
            create_dataset_btn.click(
                create_dataset,
                inputs=[training_texts_input, training_labels_input, dataset_name_input],
                outputs=[dataset_status_output]
            )
            
            create_dataloaders_btn.click(
                create_dataloaders,
                inputs=[dataset_name_input, val_split_input],
                outputs=[dataloader_status_output]
            )
            
            train_epoch_btn.click(
                train_epoch,
                outputs=[training_output_output]
            )
            
            benchmark_btn.click(
                benchmark_performance,
                outputs=[benchmark_output_output]
            )
            
            update_config_btn.click(
                update_config,
                inputs=[batch_size_input, num_workers_input, pin_memory_input, persistent_workers_input],
                outputs=[config_status_output]
            )
            
            # Connect early stopping and LR scheduling buttons
            train_with_early_stopping_btn.click(
                train_with_early_stopping,
                inputs=[early_stopping_patience_input, early_stopping_min_delta_input, 
                       early_stopping_monitor_input, early_stopping_mode_input,
                       lr_scheduler_input, learning_rate_input, warmup_steps_input, max_grad_norm_input],
                outputs=[training_output_output]
            )
            
            get_lr_info_btn.click(
                get_learning_rate_info,
                outputs=[config_status_output]
            )
            
            # Auto-refresh stats
            gr.on(
                [create_dataset_btn.click, create_dataloaders_btn.click, update_config_btn.click],
                get_stats,
                outputs=[stats_output_output]
            )
        
        with gr.Tab("🎨 Interactive Demos"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚀 Real-Time SEO Analysis Demo")
                    demo_text_input = gr.Textbox(
                        label="Input Text for SEO Analysis",
                        placeholder="Enter your content here for real-time SEO optimization...",
                        lines=8
                    )
                    demo_analysis_type = gr.Dropdown(
                        choices=["comprehensive", "keyword_density", "readability", "sentiment", "technical_seo"],
                        value="comprehensive",
                        label="Analysis Type"
                    )
                    demo_language = gr.Dropdown(
                        choices=["en", "es", "fr", "de", "it", "pt"],
                        value="en",
                        label="Language"
                    )
                    run_demo_analysis_btn = gr.Button("🔍 Run Demo Analysis", variant="primary")
                    
                    gr.Markdown("### 🎭 Diffusion Model Demo")
                    diffusion_prompt_input = gr.Textbox(
                        label="Diffusion Prompt",
                        placeholder="A beautiful landscape with mountains and lake, high quality, detailed",
                        lines=3
                    )
                    diffusion_negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="blurry, low quality, distorted",
                        lines=2
                    )
                    diffusion_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Diffusion Steps"
                    )
                    diffusion_guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                    diffusion_seed = gr.Number(
                        label="Seed (optional)",
                        value=-1
                    )
                    run_diffusion_demo_btn = gr.Button("🎨 Generate Image", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 📊 Analysis Results")
                    demo_analysis_output = gr.JSON(label="SEO Analysis Results")
                    demo_visualization_output = gr.Plot(label="SEO Metrics Visualization")
                    
                    gr.Markdown("### 🖼️ Generated Image")
                    diffusion_output_image = gr.Image(label="Generated Image", type="pil")
                    diffusion_metadata_output = gr.JSON(label="Generation Metadata")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔄 Batch Processing Demo")
                    batch_texts_input = gr.Textbox(
                        label="Batch Texts (one per line)",
                        placeholder="Enter multiple texts for batch processing...",
                        lines=6
                    )
                    batch_processing_type = gr.Dropdown(
                        choices=["seo_optimization", "content_generation", "keyword_extraction", "sentiment_analysis"],
                        value="seo_optimization",
                        label="Processing Type"
                    )
                    run_batch_demo_btn = gr.Button("⚡ Run Batch Processing", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 📈 Batch Results")
                    batch_results_output = gr.JSON(label="Batch Processing Results")
                    batch_progress_output = gr.Progress(label="Processing Progress")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Evaluation Metrics Demo")
                    gr.Markdown("Test comprehensive evaluation metrics for different SEO tasks")
                    run_evaluation_metrics_btn = gr.Button("📊 Run Evaluation Metrics Demo", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 📊 Evaluation Results")
                    evaluation_metrics_output = gr.JSON(label="Evaluation Metrics Results")
                    evaluation_visualization_output = gr.Plot(label="Evaluation Metrics Visualization")
            
            # Demo Functions with Error Handling and Input Validation
            @error_boundary
            def run_demo_analysis(text, analysis_type, language):
                """Run real-time SEO analysis demo with comprehensive validation."""
                # Input validation
                validation_errors = []
                
                # Validate text input
                is_valid, error_msg = input_validator.validate_text(text, "Content text")
                if not is_valid:
                    validation_errors.append(error_msg)
                
                # Validate analysis type
                valid_analysis_types = ["comprehensive", "keyword_density", "readability", "sentiment", "technical_seo"]
                if analysis_type not in valid_analysis_types:
                    validation_errors.append(f"Invalid analysis type. Must be one of: {', '.join(valid_analysis_types)}")
                
                # Validate language
                valid_languages = ["en", "es", "fr", "de", "it", "pt", "auto"]
                if language not in valid_languages:
                    validation_errors.append(f"Invalid language. Must be one of: {', '.join(valid_languages)}")
                
                if validation_errors:
                    return {"error": True, "message": "Validation failed", "details": validation_errors}, None
                
                try:
                    if not text.strip():
                        return {"error": "No text provided"}, None
                    
                    # Simulate real-time analysis
                    results = {
                        "text_length": len(text),
                        "word_count": len(text.split()),
                        "analysis_type": analysis_type,
                        "language": language,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "processing_time": f"{np.random.uniform(0.1, 0.5):.3f}s"
                    }
                    
                    # Add analysis-specific results
                    if analysis_type == "comprehensive":
                        results.update({
                            "seo_score": np.random.uniform(65, 95),
                            "keyword_density": np.random.uniform(1.5, 4.0),
                            "readability_score": np.random.uniform(70, 90),
                            "sentiment_score": np.random.uniform(-0.3, 0.8),
                            "technical_seo_score": np.random.uniform(75, 95)
                        })
                    elif analysis_type == "keyword_density":
                        words = text.lower().split()
                        common_keywords = ["seo", "optimization", "content", "marketing", "digital"]
                        keyword_count = sum(1 for word in words if word in common_keywords)
                        results["keyword_density"] = (keyword_count / len(words)) * 100 if words else 0
                    elif analysis_type == "readability":
                        sentences = text.split('.')
                        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
                        results["readability_score"] = max(0, 100 - avg_sentence_length * 2)
                    elif analysis_type == "sentiment":
                        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
                        negative_words = ["bad", "terrible", "awful", "horrible", "worst"]
                        pos_count = sum(1 for word in text.lower().split() if word in positive_words)
                        neg_count = sum(1 for word in text.lower().split() if word in negative_words)
                        results["sentiment_score"] = (pos_count - neg_count) / max(len(text.split()), 1)
                    elif analysis_type == "technical_seo":
                        results.update({
                            "title_length": np.random.uniform(30, 60),
                            "meta_description_length": np.random.uniform(120, 160),
                            "heading_structure": "H1, H2, H3" if np.random.random() > 0.3 else "H1, H2",
                            "internal_links": np.random.randint(0, 10),
                            "external_links": np.random.randint(0, 5)
                        })
                    
                    # Create visualization
                    if analysis_type == "comprehensive":
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        metrics = ['SEO Score', 'Readability', 'Technical SEO', 'Sentiment']
                        values = [
                            results.get('seo_score', 0),
                            results.get('readability_score', 0),
                            results.get('technical_seo_score', 0),
                            (results.get('sentiment_score', 0) + 1) * 50  # Normalize to 0-100
                        ]
                        
                        bars = ax.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                        ax.set_ylim(0, 100)
                        ax.set_ylabel('Score')
                        ax.set_title('SEO Analysis Results')
                        
                        # Add value labels on bars
                        for bar, value in zip(bars, values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{value:.1f}', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        return results, fig
                    else:
                        return results, None
                        
                except Exception as e:
                    return {"error": f"Demo analysis failed: {str(e)}"}, None
            
            @error_boundary
            def run_diffusion_demo(prompt, negative_prompt, steps, guidance_scale, seed):
                """Run diffusion model demo with comprehensive validation."""
                # Input validation
                validation_errors = []
                
                # Validate prompt
                is_valid, error_msg = input_validator.validate_text(prompt, "Generation prompt")
                if not is_valid:
                    validation_errors.append(error_msg)
                
                # Validate negative prompt (optional)
                if negative_prompt and negative_prompt.strip():
                    is_valid, error_msg = input_validator.validate_text(negative_prompt, "Negative prompt")
                    if not is_valid:
                        validation_errors.append(error_msg)
                
                # Validate numeric parameters
                is_valid, error_msg = input_validator.validate_number(steps, "Steps", 1, 100, True)
                if not is_valid:
                    validation_errors.append(error_msg)
                
                is_valid, error_msg = input_validator.validate_number(guidance_scale, "Guidance scale", 1.0, 20.0)
                if not is_valid:
                    validation_errors.append(error_msg)
                
                # Validate seed
                if seed != -1:
                    is_valid, error_msg = input_validator.validate_number(seed, "Seed", 0, 2**32-1, True)
                    if not is_valid:
                        validation_errors.append(error_msg)
                
                if validation_errors:
                    return None, {"error": True, "message": "Validation failed", "details": validation_errors}
                
                try:
                    if not prompt.strip():
                        return None, {"error": "No prompt provided"}
                    
                    if not engine.diffusion_generator:
                        return None, {"error": "Diffusion models not enabled"}
                    
                    # Set seed if provided
                    if seed != -1:
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(seed)
                    
                    # Generate image
                    start_time = time.time()
                    image = engine.diffusion_generator.generate_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        steps=steps,
                        guidance_scale=guidance_scale
                    )
                    generation_time = time.time() - start_time
                    
                    # Metadata
                    metadata = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed if seed != -1 else "random",
                        "generation_time": f"{generation_time:.2f}s",
                        "model": engine.diffusion_generator.pipeline.__class__.__name__,
                        "device": str(engine.device)
                    }
                    
                    return image, metadata
                    
                except Exception as e:
                    return None, {"error": f"Diffusion demo failed: {str(e)}"}
            
            @error_boundary
            def run_batch_demo(texts, processing_type):
                """Run batch processing demo with comprehensive validation."""
                # Input validation
                validation_errors = []
                
                # Validate texts input
                is_valid, error_msg = input_validator.validate_text(texts, "Batch texts")
                if not is_valid:
                    validation_errors.append(error_msg)
                
                # Validate processing type
                valid_processing_types = ["seo_optimization", "keyword_analysis", "readability_check", "quality_assessment"]
                if processing_type not in valid_processing_types:
                    validation_errors.append(f"Invalid processing type. Must be one of: {', '.join(valid_processing_types)}")
                
                if validation_errors:
                    return {"error": True, "message": "Validation failed", "details": validation_errors}
                
                try:
                    if not texts.strip():
                        return {"error": "No texts provided"}
                    
                    text_list = [t.strip() for t in texts.split('\n') if t.strip()]
                    if not text_list:
                        return {"error": "No valid texts found"}
                    
                    # Validate individual text lengths
                    for i, text in enumerate(text_list):
                        if len(text) > 5000:  # Limit individual text length
                            validation_errors.append(f"Text {i+1} exceeds maximum length of 5000 characters")
                    
                    if validation_errors:
                        return {"error": True, "message": "Text validation failed", "details": validation_errors}
                    
                    results = {
                        "processing_type": processing_type,
                        "total_texts": len(text_list),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "results": []
                    }
                    
                    # Process each text
                    for i, text in enumerate(text_list):
                        # Simulate processing time
                        time.sleep(0.1)
                        
                        if processing_type == "seo_optimization":
                            result = {
                                "text_id": i + 1,
                                "text_preview": text[:50] + "..." if len(text) > 50 else text,
                                "seo_score": np.random.uniform(60, 95),
                                "suggestions": [
                                    "Add more relevant keywords",
                                    "Improve heading structure",
                                    "Optimize meta description"
                                ][:np.random.randint(1, 4)]
                            }
                        elif processing_type == "content_generation":
                            result = {
                                "text_id": i + 1,
                                "original_length": len(text),
                                "generated_content": f"Enhanced version of: {text[:30]}...",
                                "improvement_score": np.random.uniform(20, 60)
                            }
                        elif processing_type == "keyword_extraction":
                            words = text.lower().split()
                            keywords = [word for word in words if len(word) > 4 and word.isalpha()]
                            result = {
                                "text_id": i + 1,
                                "extracted_keywords": keywords[:5],
                                "keyword_count": len(keywords),
                                "relevance_score": np.random.uniform(0.5, 1.0)
                            }
                        elif processing_type == "sentiment":
                            sentiment = np.random.choice(["positive", "neutral", "negative"])
                            result = {
                                "text_id": i + 1,
                                "sentiment": sentiment,
                                "confidence": np.random.uniform(0.6, 0.95),
                                "key_phrases": [text.split()[i] for i in np.random.choice(len(text.split()), 3, replace=False)]
                            }
                        
                        results["results"].append(result)
                    
                    return results
                    
                except Exception as e:
                    return {"error": f"Batch demo failed: {str(e)}"}
            
            @error_boundary
            def run_evaluation_metrics_demo():
                """Run evaluation metrics demo with comprehensive validation."""
                try:
                    # Validate engine state
                    if not hasattr(engine, 'evaluate_seo_classification'):
                        return {"error": True, "message": "Evaluation methods not available"}, None
                    
                    # Mock data for demonstration with validation
                    y_true_classification = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
                    y_pred_classification = [1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
                    
                    y_true_regression = [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 0.8]
                    y_pred_regression = [0.75, 0.65, 0.85, 0.75, 0.55, 0.8, 0.7, 0.85, 0.75, 0.8]
                    
                    y_true_ranking = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
                    y_pred_ranking = [0.85, 0.75, 0.8, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
                    
                    # Validate data consistency
                    if len(y_true_classification) != len(y_pred_classification):
                        return {"error": True, "message": "Classification data length mismatch"}, None
                    
                    if len(y_true_regression) != len(y_pred_regression):
                        return {"error": True, "message": "Regression data length mismatch"}, None
                    
                    if len(y_true_ranking) != len(y_pred_ranking):
                        return {"error": True, "message": "Ranking data length mismatch"}, None
                    
                    # Calculate evaluation metrics with error handling
                    try:
                        classification_metrics = engine.evaluate_seo_classification(
                            y_true_classification, y_pred_classification, "seo_optimization"
                        )
                    except Exception as e:
                        classification_metrics = {"error": f"Classification evaluation failed: {str(e)}"}
                    
                    try:
                        regression_metrics = engine.evaluate_seo_regression(
                            y_true_regression, y_pred_regression, "seo_score"
                        )
                    except Exception as e:
                        regression_metrics = {"error": f"Regression evaluation failed: {str(e)}"}
                    
                    try:
                        ranking_metrics = engine.evaluate_seo_ranking(
                            y_true_ranking, y_pred_ranking, "content_ranking"
                        )
                    except Exception as e:
                        ranking_metrics = {"error": f"Ranking evaluation failed: {str(e)}"}
                    
                    # Create comprehensive evaluation report
                    evaluation_report = {
                        "classification_metrics": classification_metrics,
                        "regression_metrics": regression_metrics,
                        "ranking_metrics": ranking_metrics,
                        "summary": {
                            "classification": engine.get_evaluation_summary("classification", classification_metrics) if isinstance(classification_metrics, dict) else "N/A",
                            "regression": engine.get_evaluation_summary("regression", regression_metrics) if isinstance(regression_metrics, dict) else "N/A",
                            "ranking": engine.get_evaluation_summary("ranking", ranking_metrics) if isinstance(ranking_metrics, dict) else "N/A"
                        }
                    }
                    
                    # Create visualization with error handling
                    try:
                        import matplotlib.pyplot as plt
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                        
                        # Classification metrics
                        class_metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
                        class_values = [
                            classification_metrics.get('accuracy', 0) if isinstance(classification_metrics, dict) else 0,
                            classification_metrics.get('precision_weighted', 0) if isinstance(classification_metrics, dict) else 0,
                            classification_metrics.get('recall_weighted', 0) if isinstance(classification_metrics, dict) else 0,
                            classification_metrics.get('f1_weighted', 0) if isinstance(classification_metrics, dict) else 0
                        ]
                        ax1.bar(class_metrics, class_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                        ax1.set_title('Classification Metrics')
                        ax1.set_ylim(0, 1)
                        
                        # Regression metrics
                        reg_metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
                        reg_values = [
                            regression_metrics.get('r2', 0) if isinstance(regression_metrics, dict) else 0,
                            regression_metrics.get('rmse', 0) if isinstance(regression_metrics, dict) else 0,
                            regression_metrics.get('mae', 0) if isinstance(regression_metrics, dict) else 0,
                            regression_metrics.get('mape', 0) if isinstance(regression_metrics, dict) else 0
                        ]
                        ax2.bar(reg_metrics, reg_values, color=['#FF9F43', '#00D2D3', '#54A0FF', '#5F27CD'])
                        ax2.set_title('Regression Metrics')
                        
                        # Ranking metrics
                        rank_metrics = ['NDCG@5', 'NDCG@10', 'NDCG@20']
                        rank_values = [
                            ranking_metrics.get('ndcg_5', 0) if isinstance(ranking_metrics, dict) else 0,
                            ranking_metrics.get('ndcg_10', 0) if isinstance(ranking_metrics, dict) else 0,
                            ranking_metrics.get('ndcg_20', 0) if isinstance(ranking_metrics, dict) else 0
                        ]
                        ax3.bar(rank_metrics, rank_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                        ax3.set_title('Ranking Metrics')
                        ax3.set_ylim(0, 1)
                        
                        # SEO-specific metrics
                        seo_metrics = ['SEO F1', 'SEO Precision', 'SEO Recall']
                        seo_values = [
                            classification_metrics.get('seo_f1', 0) if isinstance(classification_metrics, dict) else 0,
                            classification_metrics.get('seo_precision', 0) if isinstance(classification_metrics, dict) else 0,
                            classification_metrics.get('seo_recall', 0) if isinstance(classification_metrics, dict) else 0
                        ]
                        ax4.bar(seo_metrics, seo_values, color=['#FF9F43', '#00D2D3', '#54A0FF'])
                        ax4.set_title('SEO-Specific Metrics')
                        ax4.set_ylim(0, 1)
                        
                        plt.tight_layout()
                        
                    except Exception as e:
                        return {"error": True, "message": f"Visualization failed: {str(e)}"}, None
                    
                    return json.dumps(evaluation_report, indent=2), fig
                except Exception as e:
                    return f"Error in evaluation metrics demo: {str(e)}", None
            
            # Connect demo buttons
            run_demo_analysis_btn.click(
                run_demo_analysis,
                inputs=[demo_text_input, demo_analysis_type, demo_language],
                outputs=[demo_analysis_output, demo_visualization_output]
            )
            
            run_diffusion_demo_btn.click(
                run_diffusion_demo,
                inputs=[diffusion_prompt_input, diffusion_negative_prompt, diffusion_steps, 
                       diffusion_guidance_scale, diffusion_seed],
                outputs=[diffusion_output_image, diffusion_metadata_output]
            )
            
            run_batch_demo_btn.click(
                run_batch_demo,
                inputs=[batch_texts_input, batch_processing_type],
                outputs=[batch_results_output]
            )
            
            run_evaluation_metrics_btn.click(
                run_evaluation_metrics_demo,
                inputs=[],
                outputs=[evaluation_metrics_output, evaluation_visualization_output]
            )
        
        with gr.Tab("🛡️ Error Handling & Monitoring"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚨 Error Management")
                    view_errors_btn = gr.Button("👁️ View Recent Errors", variant="secondary")
                    clear_errors_btn = gr.Button("🧹 Clear Error Log", variant="secondary")
                    export_errors_btn = gr.Button("📥 Export Error Report", variant="secondary")
                    
                    gr.Markdown("### 📊 Error Statistics")
                    error_summary_output = gr.JSON(label="Error Summary")
                    error_trends_output = gr.Plot(label="Error Trends")
                
                with gr.Column():
                    gr.Markdown("### ✅ Input Validation")
                    validation_test_btn = gr.Button("🧪 Test Input Validation", variant="primary")
                    validation_rules_btn = gr.Button("📋 View Validation Rules", variant="secondary")
                    
                    gr.Markdown("### 🔧 System Health")
                    system_health_output = gr.JSON(label="System Health Status")
                    health_check_btn = gr.Button("🔍 Run Health Check", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧪 Validation Testing")
                    test_text_input = gr.Textbox(
                        label="Test Text Input",
                        placeholder="Enter text to test validation...",
                        lines=3
                    )
                    test_url_input = gr.Textbox(
                        label="Test URL Input",
                        placeholder="https://example.com",
                        lines=1
                    )
                    test_json_input = gr.Textbox(
                        label="Test JSON Input",
                        placeholder='{"key": "value"}',
                        lines=2
                    )
                    run_validation_test_btn = gr.Button("🧪 Run Validation Test", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 📋 Validation Results")
                    validation_results_output = gr.JSON(label="Validation Results")
                    validation_suggestions_output = gr.Textbox(
                        label="Validation Suggestions",
                        lines=5,
                        interactive=False
                    )
            
            # Error handling and monitoring functions
            def view_recent_errors():
                """View recent errors from the error handler."""
                try:
                    return error_handler.get_error_summary()
                except Exception as e:
                    return {"error": f"Failed to get error summary: {str(e)}"}
            
            def clear_error_log():
                """Clear the error log."""
                try:
                    error_handler.error_log.clear()
                    return {"success": True, "message": "Error log cleared", "total_errors": 0}
                except Exception as e:
                    return {"error": f"Failed to clear error log: {str(e)}"}
            
            def export_error_report():
                """Export error report to JSON."""
                try:
                    error_summary = error_handler.get_error_summary()
                    export_data = {
                        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error_summary": error_summary,
                        "system_info": {
                            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                            "torch_version": torch.__version__,
                            "gradio_version": gr.__version__,
                            "device": str(engine.device) if hasattr(engine, 'device') else "N/A"
                        }
                    }
                    
                    export_file = f"error_report_{int(time.time())}.json"
                    with open(export_file, 'w') as f:
                        json.dump(export_data, f, indent=2)
                    
                    return {
                        "success": True,
                        "export_file": export_file,
                        "total_errors": error_summary.get("total_errors", 0)
                    }
                    
                except Exception as e:
                    return {"error": f"Failed to export error report: {str(e)}"}
            
            def test_input_validation():
                """Test input validation with sample inputs."""
                try:
                    test_results = {}
                    
                    # Test text validation
                    test_text = test_text_input.strip() if test_text_input else "Sample text"
                    is_valid, error_msg = input_validator.validate_text(test_text, "Test text")
                    test_results["text_validation"] = {
                        "input": test_text,
                        "is_valid": is_valid,
                        "error_message": error_msg
                    }
                    
                    # Test URL validation
                    test_url = test_url_input.strip() if test_url_input else "https://example.com"
                    is_valid, error_msg = input_validator.validate_url(test_url, "Test URL")
                    test_results["url_validation"] = {
                        "input": test_url,
                        "is_valid": is_valid,
                        "error_message": error_msg
                    }
                    
                    # Test JSON validation
                    test_json = test_json_input.strip() if test_json_input else '{"key": "value"}'
                    is_valid, error_msg = input_validator.validate_json(test_json, "Test JSON")
                    test_results["json_validation"] = {
                        "input": test_json,
                        "is_valid": is_valid,
                        "error_message": error_msg
                    }
                    
                    # Generate suggestions
                    suggestions = []
                    for field, result in test_results.items():
                        if not result["is_valid"]:
                            suggestions.append(f"{field.replace('_', ' ').title()}: {result['error_message']}")
                    
                    if not suggestions:
                        suggestions.append("All inputs passed validation successfully!")
                    
                    return test_results, "\n".join(suggestions)
                    
                except Exception as e:
                    return {"error": f"Validation test failed: {str(e)}"}, "Test failed due to an error"
            
            def view_validation_rules():
                """View current validation rules."""
                try:
                    return {
                        "validation_rules": input_validator.validation_rules,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                except Exception as e:
                    return {"error": f"Failed to get validation rules: {str(e)}"}
            
            def run_health_check():
                """Run comprehensive system health check."""
                try:
                    health_status = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "system": {},
                        "engine": {},
                        "models": {},
                        "validation": {}
                    }
                    
                    # System health
                    try:
                        import psutil
                        health_status["system"] = {
                            "cpu_usage": f"{psutil.cpu_percent()}%",
                            "memory_usage": f"{psutil.virtual_memory().percent}%",
                            "disk_usage": f"{psutil.disk_usage('/').percent}%",
                            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                        }
                    except Exception as e:
                        health_status["system"]["error"] = str(e)
                    
                    # Engine health
                    try:
                        health_status["engine"] = {
                            "initialized": hasattr(engine, 'seo_model'),
                            "device": str(engine.device) if hasattr(engine, 'device') else "N/A",
                            "config_loaded": hasattr(engine, 'config'),
                            "error_handler_active": error_handler is not None,
                            "validator_active": input_validator is not None
                        }
                    except Exception as e:
                        health_status["engine"]["error"] = str(e)
                    
                    # Model health
                    try:
                        health_status["models"] = {
                            "seo_model": engine.seo_model is not None if hasattr(engine, 'seo_model') else False,
                            "diffusion_model": engine.diffusion_generator is not None if hasattr(engine, 'diffusion_generator') else False,
                            "tokenizer": hasattr(engine, 'tokenizer') and engine.tokenizer is not None,
                            "gpu_available": torch.cuda.is_available(),
                            "gpu_memory": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
                        }
                    except Exception as e:
                        health_status["models"]["error"] = str(e)
                    
                    # Validation health
                    try:
                        health_status["validation"] = {
                            "rules_loaded": len(input_validator.validation_rules) > 0,
                            "error_handler_active": len(error_handler.error_log) >= 0,
                            "validation_methods": [method for method in dir(input_validator) if method.startswith('validate_')]
                        }
                    except Exception as e:
                        health_status["validation"]["error"] = str(e)
                    
                    return health_status
                    
                except Exception as e:
                    return {"error": f"Health check failed: {str(e)}"}
            
            # Connect error handling buttons
            view_errors_btn.click(view_recent_errors, outputs=[error_summary_output])
            clear_errors_btn.click(clear_error_log, outputs=[error_summary_output])
            export_errors_btn.click(export_error_report, outputs=[error_summary_output])
            validation_test_btn.click(test_input_validation, 
                                   inputs=[test_text_input, test_url_input, test_json_input],
                                   outputs=[validation_results_output, validation_suggestions_output])
            validation_rules_btn.click(view_validation_rules, outputs=[validation_results_output])
            health_check_btn.click(run_health_check, outputs=[system_health_output])
        
        with gr.Tab("📊 Performance Monitoring"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚀 Real-Time Performance Metrics")
                    refresh_metrics_btn = gr.Button("🔄 Refresh Metrics", variant="primary")
                    clear_metrics_btn = gr.Button("🗑️ Clear Metrics", variant="secondary")
                    
                    gr.Markdown("### 📈 Training Progress")
                    training_progress_output = gr.Plot(label="Training Progress")
                    validation_metrics_output = gr.Plot(label="Validation Metrics")
                
                with gr.Column():
                    gr.Markdown("### 💾 System Resources")
                    system_metrics_output = gr.JSON(label="System Metrics")
                    memory_usage_output = gr.Plot(label="Memory Usage")
                    
                    gr.Markdown("### 🔍 Model Performance")
                    model_performance_output = gr.JSON(label="Model Performance")
                    inference_speed_output = gr.Plot(label="Inference Speed")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Custom Metrics")
                    custom_metric_name = gr.Textbox(
                        label="Metric Name",
                        placeholder="e.g., accuracy, f1_score, custom_metric"
                    )
                    custom_metric_value = gr.Number(
                        label="Metric Value",
                        value=0.0
                    )
                    add_custom_metric_btn = gr.Button("➕ Add Metric", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 📋 Metrics History")
                    metrics_history_output = gr.JSON(label="Metrics History")
                    export_metrics_btn = gr.Button("📥 Export Metrics", variant="secondary")
            
            # Performance monitoring functions with error handling
            @error_boundary
            def refresh_performance_metrics():
                """Refresh real-time performance metrics with comprehensive error handling."""
                try:
                    # System metrics
                    import psutil
                    system_metrics = {
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage": psutil.disk_usage('/').percent,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # GPU metrics if available
                    if torch.cuda.is_available():
                        system_metrics.update({
                            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                            "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                            "gpu_utilization": "N/A"  # Would need nvidia-ml-py for this
                        })
                    
                    # Model performance metrics
                    model_metrics = {
                        "model_loaded": engine.seo_model is not None,
                        "diffusion_loaded": engine.diffusion_generator is not None,
                        "current_device": str(engine.device),
                        "mixed_precision": engine.config.use_mixed_precision,
                        "batch_size": engine.config.batch_size
                    }
                    
                    # Training metrics if available
                    if hasattr(engine, 'training_history') and engine.training_history:
                        latest = engine.training_history[-1]
                        model_metrics.update({
                            "last_epoch": latest.get('epoch', 0),
                            "last_train_loss": latest.get('train_loss', 0),
                            "last_val_loss": latest.get('val_loss', 0),
                            "learning_rate": latest.get('learning_rate', 0)
                        })
                    
                    return system_metrics, model_metrics
                    
                except Exception as e:
                    return {"error": f"Failed to refresh metrics: {str(e)}"}, {"error": str(e)}
            
            @error_boundary
            def create_performance_plots():
                """Create performance visualization plots with comprehensive error handling."""
                try:
                    import matplotlib.pyplot as plt
                    
                    # Training progress plot
                    if hasattr(engine, 'training_history') and engine.training_history:
                        epochs = [h.get('epoch', i) for i, h in enumerate(engine.training_history)]
                        train_losses = [h.get('train_loss', 0) for h in engine.training_history]
                        val_losses = [h.get('val_loss', 0) for h in engine.training_history]
                        
                        # Validate data before plotting
                        if len(epochs) > 0 and len(train_losses) > 0 and len(val_losses) > 0:
                            fig1, ax1 = plt.subplots(figsize=(8, 6))
                            ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
                            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Loss')
                            ax1.set_title('Training Progress')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            plt.tight_layout()
                        else:
                            fig1 = None
                    else:
                        fig1 = None
                    
                    # Memory usage plot
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    if torch.cuda.is_available():
                        # Simulate memory usage over time
                        time_points = list(range(10))
                        memory_usage = [np.random.uniform(2, 8) for _ in time_points]
                        
                        ax2.plot(time_points, memory_usage, 'g-', linewidth=2, marker='o')
                        ax2.set_xlabel('Time (minutes)')
                        ax2.set_ylabel('GPU Memory (GB)')
                        ax2.set_title('GPU Memory Usage')
                        ax2.grid(True, alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, 'GPU not available', ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('GPU Memory Usage')
                    
                    plt.tight_layout()
                    
                    return fig1, fig2
                    
                except Exception as e:
                    return None, None
            
            @error_boundary
            def add_custom_metric(name, value):
                """Add a custom metric to tracking with comprehensive validation."""
                try:
                    # Input validation
                    if not name or not name.strip():
                        return {"error": True, "message": "Metric name cannot be empty"}
                    
                    if not isinstance(value, (int, float)):
                        return {"error": True, "message": "Metric value must be a number"}
                    
                    if not hasattr(engine, 'custom_metrics'):
                        engine.custom_metrics = {}
                    
                    if not hasattr(engine, 'custom_metrics_history'):
                        engine.custom_metrics_history = []
                    
                    # Add to current metrics
                    engine.custom_metrics[name] = value
                    
                    # Add to history
                    engine.custom_metrics_history.append({
                        "metric": name,
                        "value": value,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    return {
                        "success": True,
                        "metric_added": name,
                        "value": value,
                        "total_metrics": len(engine.custom_metrics)
                    }
                    
                except Exception as e:
                    return {"error": f"Failed to add metric: {str(e)}"}
            
            @error_boundary
            def get_metrics_history():
                """Get metrics history with comprehensive error handling."""
                try:
                    if not hasattr(engine, 'custom_metrics_history'):
                        return {"metrics": []}
                    
                    # Validate history data
                    history = engine.custom_metrics_history[-20:]  # Last 20 entries
                    if not isinstance(history, list):
                        return {"error": "Invalid metrics history format"}
                    
                    return {
                        "total_metrics": len(engine.custom_metrics_history),
                        "metrics": history
                    }
                    
                except Exception as e:
                    return {"error": f"Failed to get metrics history: {str(e)}"}
            
            @error_boundary
            def export_metrics():
                """Export metrics to JSON file with comprehensive validation."""
                try:
                    if not hasattr(engine, 'custom_metrics_history'):
                        return {"error": "No metrics to export"}
                    
                    # Validate export data
                    export_data = {
                        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_metrics": len(engine.custom_metrics_history),
                        "metrics": engine.custom_metrics_history,
                        "custom_metrics": getattr(engine, 'custom_metrics', {}),
                        "training_history": getattr(engine, 'training_history', [])
                    }
                    
                    # Validate file path
                    export_file = f"metrics_export_{int(time.time())}.json"
                    if not export_file.endswith('.json'):
                        return {"error": "Invalid export file format"}
                    
                    # Write to file with error handling
                    try:
                        with open(export_file, 'w') as f:
                            json.dump(export_data, f, indent=2)
                    except (IOError, OSError) as e:
                        return {"error": f"Failed to write export file: {str(e)}"}
                    
                    return {
                        "success": True,
                        "export_file": export_file,
                        "metrics_exported": len(engine.custom_metrics_history)
                    }
                    
                except Exception as e:
                    return {"error": f"Failed to export metrics: {str(e)}"}
            
            # Connect performance monitoring buttons
            refresh_metrics_btn.click(
                refresh_performance_metrics,
                outputs=[system_metrics_output, model_performance_output]
            )
            
            refresh_metrics_btn.click(
                create_performance_plots,
                outputs=[training_progress_output, memory_usage_output]
            )
            
            add_custom_metric_btn.click(
                add_custom_metric,
                inputs=[custom_metric_name, custom_metric_value],
                outputs=[metrics_history_output]
            )
            
            clear_metrics_btn.click(
                lambda: ({"success": True, "message": "Metrics cleared"}, []),
                outputs=[metrics_history_output, training_progress_output]
            )
            
            export_metrics_btn.click(
                export_metrics,
                outputs=[metrics_history_output]
            )
            
            # Auto-refresh metrics
            gr.on(
                [refresh_metrics_btn.click],
                get_metrics_history,
                outputs=[metrics_history_output]
            )
        
        with gr.Tab("🔍 Code Profiling & Bottleneck Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚀 Code Profiling System")
                    enable_profiling_btn = gr.Button("✅ Enable Profiling", variant="primary")
                    disable_profiling_btn = gr.Button("❌ Disable Profiling", variant="secondary")
                    get_profiling_status_btn = gr.Button("📊 Profiling Status", variant="primary")
                    get_bottlenecks_btn = gr.Button("🐌 Identify Bottlenecks", variant="primary")
                    get_recommendations_btn = gr.Button("💡 Performance Recommendations", variant="primary")
                    export_profiling_btn = gr.Button("📥 Export Profiling Data", variant="secondary")
                    cleanup_profiling_btn = gr.Button("🧹 Cleanup Profiling", variant="secondary")
                
                with gr.Column():
                    gr.Markdown("### 📈 Profiling Results")
                    profiling_status_output = gr.JSON(label="Profiling Status")
                    bottlenecks_output = gr.JSON(label="Performance Bottlenecks")
                    recommendations_output = gr.JSON(label="Optimization Recommendations")
                    memory_analysis_output = gr.JSON(label="Memory Analysis")
                    profiling_summary_output = gr.JSON(label="Profiling Summary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ Profiling Configuration")
                    profile_data_loading_cb = gr.Checkbox(label="Profile Data Loading", value=True)
                    profile_preprocessing_cb = gr.Checkbox(label="Profile Preprocessing", value=True)
                    profile_model_inference_cb = gr.Checkbox(label="Profile Model Inference", value=True)
                    profile_training_loop_cb = gr.Checkbox(label="Profile Training Loop", value=True)
                    profile_memory_usage_cb = gr.Checkbox(label="Profile Memory Usage", value=True)
                    profile_gpu_utilization_cb = gr.Checkbox(label="Profile GPU Utilization", value=True)
                
                with gr.Column():
                    gr.Markdown("### 📊 Real-Time Monitoring")
                    real_time_metrics_output = gr.JSON(label="Real-Time Metrics")
                    operation_timings_output = gr.JSON(label="Operation Timings")
                    memory_trends_output = gr.JSON(label="Memory Trends")
            
            # Profiling functions with error handling
            @error_boundary
            def enable_profiling():
                """Enable code profiling with comprehensive error handling."""
                try:
                    if hasattr(engine, 'code_profiler'):
                        engine.config.enable_code_profiling = True
                        engine.code_profiler.profiling_enabled = True
                        return {"success": True, "message": "Code profiling enabled successfully"}
                    else:
                        return {"error": "Code profiler not available"}
                except Exception as e:
                    return {"error": f"Failed to enable profiling: {str(e)}"}
            
            @error_boundary
            def disable_profiling():
                """Disable code profiling with comprehensive error handling."""
                try:
                    if hasattr(engine, 'code_profiler'):
                        engine.config.enable_code_profiling = False
                        engine.code_profiler.profiling_enabled = False
                        return {"success": True, "message": "Code profiling disabled successfully"}
                    else:
                        return {"error": "Code profiler not available"}
                except Exception as e:
                    return {"error": f"Failed to disable profiling: {str(e)}"}
            
            @error_boundary
            def get_profiling_status():
                """Get comprehensive profiling status with error handling."""
                try:
                    if hasattr(engine, 'code_profiler') and engine.code_profiler.profiling_enabled:
                        status = engine.code_profiler.get_profiling_summary()
                        return status
                    else:
                        return {"profiling_enabled": False, "message": "Profiling not enabled"}
                except Exception as e:
                    return {"error": f"Failed to get profiling status: {str(e)}"}
            
            @error_boundary
            def identify_bottlenecks():
                """Identify performance bottlenecks with error handling."""
                try:
                    if hasattr(engine, 'code_profiler') and engine.code_profiler.profiling_enabled:
                        bottlenecks = engine.code_profiler.get_bottlenecks()
                        return bottlenecks
                    else:
                        return {"error": "Profiling not enabled"}
                except Exception as e:
                    return {"error": f"Failed to identify bottlenecks: {str(e)}"}
            
            @error_boundary
            def get_performance_recommendations():
                """Get performance optimization recommendations with error handling."""
                try:
                    if hasattr(engine, 'code_profiler') and engine.code_profiler.profiling_enabled:
                        recommendations = engine.code_profiler.get_performance_recommendations()
                        return {"recommendations": recommendations}
                    else:
                        return {"error": "Profiling not enabled"}
                except Exception as e:
                    return {"error": f"Failed to get recommendations: {str(e)}"}
            
            @error_boundary
            def export_profiling_data():
                """Export profiling data to JSON file with error handling."""
                try:
                    if hasattr(engine, 'code_profiler') and engine.code_profiler.profiling_enabled:
                        result = engine.code_profiler.export_profiling_data()
                        return {"success": True, "message": result}
                    else:
                        return {"error": "Profiling not enabled"}
                except Exception as e:
                    return {"error": f"Failed to export profiling data: {str(e)}"}
            
            @error_boundary
            def cleanup_profiling():
                """Cleanup profiling resources with error handling."""
                try:
                    if hasattr(engine, 'code_profiler'):
                        engine.code_profiler.cleanup()
                        return {"success": True, "message": "Profiling cleanup completed"}
                    else:
                        return {"error": "Code profiler not available"}
                except Exception as e:
                    return {"error": f"Failed to cleanup profiling: {str(e)}"}
            
            # Connect profiling buttons
            enable_profiling_btn.click(
                enable_profiling,
                outputs=[profiling_status_output]
            )
            
            disable_profiling_btn.click(
                disable_profiling,
                outputs=[profiling_status_output]
            )
            
            get_profiling_status_btn.click(
                get_profiling_status,
                outputs=[profiling_status_output]
            )
            
            get_bottlenecks_btn.click(
                identify_bottlenecks,
                outputs=[bottlenecks_output]
            )
            
            get_recommendations_btn.click(
                get_performance_recommendations,
                outputs=[recommendations_output]
            )
            
            export_profiling_btn.click(
                export_profiling_data,
                outputs=[profiling_summary_output]
            )
            
            cleanup_profiling_btn.click(
                cleanup_profiling,
                outputs=[profiling_summary_output]
            )
    
    return interface

if __name__ == "__main__":
    # Create and launch advanced interface
    interface = create_advanced_gradio_interface()
    interface.launch(
        share=True, 
        debug=True,
        server_name="0.0.0.0",
        server_