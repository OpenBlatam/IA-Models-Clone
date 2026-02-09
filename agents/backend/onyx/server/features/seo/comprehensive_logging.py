#!/usr/bin/env python3
"""
Comprehensive Logging System for Training Progress and Errors
Advanced logging with structured output, real-time monitoring, and error tracking
Integrated with PyTorch debugging tools for enhanced ML/DL debugging
"""

import logging
import logging.handlers
import os
import sys
import time
import json
import traceback
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
import queue
import signal
import psutil
import torch
import numpy as np
import pandas as pd
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

@dataclass
class LoggingConfig:
    """Configuration for comprehensive logging system."""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_json: bool = True
    enable_tensorboard: bool = True
    enable_remote: bool = False
    remote_endpoint: Optional[str] = None
    log_training_metrics: bool = True
    log_system_metrics: bool = True
    log_gpu_metrics: bool = True
    log_memory_usage: bool = True
    log_performance: bool = True
    log_errors: bool = True
    log_warnings: bool = True
    log_debug: bool = False
    structured_format: bool = True
    include_timestamp: bool = True
    include_context: bool = True
    include_stack_trace: bool = True
    max_queue_size: int = 1000
    flush_interval: float = 1.0  # seconds
    enable_async_logging: bool = True
    enable_thread_safety: bool = True
    
    # PyTorch Debugging Tools Configuration
    enable_pytorch_debugging: bool = True
    enable_autograd_anomaly_detection: bool = False  # Can be expensive
    enable_gradient_debugging: bool = True
    enable_memory_debugging: bool = True
    enable_profiler: bool = False  # Can be expensive
    enable_tensor_debugging: bool = True
    max_grad_norm: float = 1.0
    memory_fraction: float = 0.8
    enable_cuda_memory_stats: bool = True

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging output."""
    
    def __init__(self, include_timestamp: bool = True, include_context: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_context = include_context
    
    def format(self, record):
        """Format log record with structured information."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() if self.include_timestamp else None,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context information
        if self.include_context and hasattr(record, 'context'):
            log_entry["context"] = record.context
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename', 
                          'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'context']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for non-blocking logging."""
    
    def __init__(self, target_handler: logging.Handler, max_queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.running = True
    
    def emit(self, record):
        """Emit log record to queue."""
        try:
            if not self.queue.full():
                self.queue.put(record, block=False)
            else:
                # Queue is full, log to stderr as fallback
                sys.stderr.write(f"Log queue full, dropping record: {record.getMessage()}\n")
        except Exception:
            # Fallback to stderr
            sys.stderr.write(f"Failed to queue log record: {record.getMessage()}\n")
    
    def _worker(self):
        """Worker thread to process queued log records."""
        while self.running:
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:
                    break
                self.target_handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                sys.stderr.write(f"Error in async log worker: {e}\n")
    
    def close(self):
        """Close the async handler."""
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=5.0)

class TrainingMetricsLogger:
    """Specialized logger for training metrics and progress."""
    
    def __init__(self, log_dir: str, config: LoggingConfig):
        self.log_dir = Path(log_dir)
        self.config = config
        self.metrics_file = self.log_dir / "training_metrics.jsonl"
        self.progress_file = self.log_dir / "training_progress.csv"
        self.lock = threading.Lock() if config.enable_thread_safety else None
        
        # Initialize files
        self._init_files()
        
        # Metrics storage
        self.current_epoch = 0
        self.current_step = 0
        self.metrics_history = []
        self.start_time = time.time()
    
    def _init_files(self):
        """Initialize logging files."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress CSV
        if not self.progress_file.exists():
            with open(self.progress_file, 'w') as f:
                f.write("timestamp,epoch,step,loss,accuracy,learning_rate,gradient_norm,memory_usage,time_elapsed\n")
    
    def log_training_step(self, epoch: int, step: int, loss: float, accuracy: Optional[float] = None,
                         learning_rate: Optional[float] = None, gradient_norm: Optional[float] = None,
                         memory_usage: Optional[float] = None, **kwargs):
        """Log training step metrics."""
        timestamp = datetime.now().isoformat()
        time_elapsed = time.time() - self.start_time
        
        # Update current state
        self.current_epoch = epoch
        self.current_step = step
        
        # Prepare metrics
        metrics = {
            "timestamp": timestamp,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "memory_usage": memory_usage,
            "time_elapsed": time_elapsed,
            **kwargs
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Write to files
        self._write_metrics(metrics)
        self._write_progress(metrics)
    
    def log_validation_step(self, epoch: int, step: int, val_loss: float, val_accuracy: Optional[float] = None,
                           **kwargs):
        """Log validation step metrics."""
        self.log_training_step(epoch, step, val_loss, val_accuracy, 
                             metric_type="validation", **kwargs)
    
    def log_epoch_summary(self, epoch: int, train_loss: float, val_loss: float, 
                         train_accuracy: Optional[float] = None, val_accuracy: Optional[float] = None,
                         learning_rate: Optional[float] = None, **kwargs):
        """Log epoch summary metrics."""
        timestamp = datetime.now().isoformat()
        time_elapsed = time.time() - self.start_time
        
        summary = {
            "timestamp": timestamp,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
            "time_elapsed": time_elapsed,
            "metric_type": "epoch_summary",
            **kwargs
        }
        
        # Write to metrics file
        self._write_metrics(summary)
    
    def _write_metrics(self, metrics: Dict[str, Any):
        """Write metrics to JSONL file."""
        try:
            if self.lock:
                with self.lock:
                    with open(self.metrics_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
            else:
                with open(self.metrics_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            sys.stderr.write(f"Failed to write metrics: {e}\n")
    
    def _write_progress(self, metrics: Dict[str, Any):
        """Write progress to CSV file."""
        try:
            if self.lock:
                with self.lock:
                    with open(self.progress_file, 'a', encoding='utf-8') as f:
                        f.write(f"{metrics['timestamp']},{metrics['epoch']},{metrics['step']},"
                               f"{metrics['loss']},{metrics.get('accuracy', '')},{metrics.get('learning_rate', '')},"
                               f"{metrics.get('gradient_norm', '')},{metrics.get('memory_usage', '')},"
                               f"{metrics['time_elapsed']}\n")
            else:
                with open(self.progress_file, 'a', encoding='utf-8') as f:
                    f.write(f"{metrics['timestamp']},{metrics['epoch']},{metrics['step']},"
                           f"{metrics['loss']},{metrics.get('accuracy', '')},{metrics.get('learning_rate', '')},"
                           f"{metrics.get('gradient_norm', '')},{metrics.get('memory_usage', '')},"
                           f"{metrics['time_elapsed']}\n")
        except Exception as e:
            sys.stderr.write(f"Failed to write progress: {e}\n")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        summary = {
            "total_epochs": df['epoch'].max() if 'epoch' in df.columns else 0,
            "total_steps": df['step'].max() if 'step' in df.columns else 0,
            "total_time": time.time() - self.start_time,
            "metrics_count": len(self.metrics_history)
        }
        
        # Add loss statistics
        if 'loss' in df.columns:
            summary.update({
                "min_loss": df['loss'].min(),
                "max_loss": df['loss'].max(),
                "mean_loss": df['loss'].mean(),
                "last_loss": df['loss'].iloc[-1] if len(df) > 0 else None
            })
        
        # Add accuracy statistics
        if 'accuracy' in df.columns:
            accuracy_data = df['accuracy'].dropna()
            if len(accuracy_data) > 0:
                summary.update({
                    "min_accuracy": accuracy_data.min(),
                    "max_accuracy": accuracy_data.max(),
                    "mean_accuracy": accuracy_data.mean(),
                    "last_accuracy": accuracy_data.iloc[-1] if len(accuracy_data) > 0 else None
                })
        
        return summary

class ErrorTracker:
    """Comprehensive error tracking and analysis."""
    
    def __init__(self, log_dir: str, config: LoggingConfig):
        self.log_dir = Path(log_dir)
        self.config = config
        self.errors_file = self.log_dir / "errors.jsonl"
        self.error_summary_file = self.log_dir / "error_summary.json"
        self.lock = threading.Lock() if config.enable_thread_safety else None
        
        # Error statistics
        self.error_counts = {}
        self.error_timeline = []
        self.critical_errors = []
        self.recovery_attempts = []
        
        # Initialize files
        self._init_files()
    
    def _init_files(self):
        """Initialize error tracking files."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing error summary if exists
        if self.error_summary_file.exists():
            try:
                with open(self.error_summary_file, 'r') as f:
                    data = json.load(f)
                    self.error_counts = data.get('error_counts', {})
                    self.error_timeline = data.get('error_timeline', [])
                    self.critical_errors = data.get('critical_errors', [])
                    self.recovery_attempts = data.get('recovery_attempts', [])
            except Exception:
                pass
    
    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, 
                   severity: str = "ERROR", recovery_attempted: bool = False):
        """Track an error with context and severity."""
        timestamp = datetime.now().isoformat()
        error_info = {
            "timestamp": timestamp,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity,
            "context": context or {},
            "stack_trace": traceback.format_exc(),
            "recovery_attempted": recovery_attempted
        }
        
        # Update statistics
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to timeline
        self.error_timeline.append({
            "timestamp": timestamp,
            "error_type": error_type,
            "severity": severity
        })
        
        # Track critical errors
        if severity == "CRITICAL":
            self.critical_errors.append(error_info)
        
        # Track recovery attempts
        if recovery_attempted:
            self.recovery_attempts.append({
                "timestamp": timestamp,
                "error_type": error_type,
                "success": False  # Will be updated if recovery succeeds
            })
        
        # Write to file
        self._write_error(error_info)
        
        # Update summary
        self._update_summary()
    
    def track_recovery_success(self, error_type: str, recovery_method: str):
        """Track successful error recovery."""
        timestamp = datetime.now().isoformat()
        
        # Update last recovery attempt
        if self.recovery_attempts:
            last_attempt = self.recovery_attempts[-1]
            if last_attempt['error_type'] == error_type:
                last_attempt['success'] = True
                last_attempt['recovery_method'] = recovery_method
                last_attempt['recovery_timestamp'] = timestamp
        
        # Add new successful recovery
        self.recovery_attempts.append({
            "timestamp": timestamp,
            "error_type": error_type,
            "success": True,
            "recovery_method": recovery_method
        })
        
        # Update summary
        self._update_summary()
    
    def _write_error(self, error_info: Dict[str, Any]):
        """Write error information to file."""
        try:
            if self.lock:
                with self.lock:
                    with open(self.errors_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(error_info, ensure_ascii=False) + '\n')
            else:
                with open(self.errors_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(error_info, ensure_ascii=False) + '\n')
        except Exception as e:
            sys.stderr.write(f"Failed to write error: {e}\n")
    
    def _update_summary(self):
        """Update error summary file."""
        try:
            summary = {
                "total_errors": sum(self.error_counts.values()),
                "error_counts": self.error_counts,
                "error_timeline": self.error_timeline[-100:],  # Keep last 100
                "critical_errors": self.critical_errors[-50:],  # Keep last 50
                "recovery_attempts": self.recovery_attempts[-100:],  # Keep last 100
                "last_updated": datetime.now().isoformat()
            }
            
            if self.lock:
                with self.lock:
                    with open(self.error_summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
            else:
                with open(self.error_summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            sys.stderr.write(f"Failed to update error summary: {e}\n")
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get comprehensive error analysis."""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "critical_errors_count": len(self.critical_errors),
            "recovery_success_rate": self._calculate_recovery_rate(),
            "most_common_errors": self._get_most_common_errors(),
            "error_trends": self._analyze_error_trends()
        }
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate error recovery success rate."""
        if not self.recovery_attempts:
            return 0.0
        
        successful = sum(1 for attempt in self.recovery_attempts if attempt.get('success', False))
        return successful / len(self.recovery_attempts)
    
    def _get_most_common_errors(self) -> List[Dict[str, Any]]:
        """Get most common error types."""
        sorted_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"error_type": error_type, "count": count} for error_type, count in sorted_errors[:10]]
    
    def _analyze_error_trends(self) -> Dict[str, Any]:
        """Analyze error trends over time."""
        if not self.error_timeline:
            return {}
        
        # Group by hour
        hourly_counts = {}
        for error in self.error_timeline:
            hour = error['timestamp'][:13]  # YYYY-MM-DDTHH
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        return {
            "hourly_distribution": hourly_counts,
            "peak_error_hour": max(hourly_counts.items(), key=lambda x: x[1]) if hourly_counts else None
        }

class SystemMonitor:
    """System resource monitoring and logging."""
    
    def __init__(self, log_dir: str, config: LoggingConfig):
        self.log_dir = Path(log_dir)
        self.config = config
        self.metrics_file = self.log_dir / "system_metrics.jsonl"
        self.lock = threading.Lock() if config.enable_thread_safety else None
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # seconds
        
        # Initialize files
        self._init_files()
    
    def _init_files(self):
        """Initialize system monitoring files."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitor_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self._log_metrics(metrics)
                time.sleep(self.monitor_interval)
            except Exception as e:
                sys.stderr.write(f"System monitoring error: {e}\n")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        timestamp = datetime.now().isoformat()
        metrics = {"timestamp": timestamp}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            metrics.update({
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.update({
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_percent": memory.percent,
                "memory_used": memory.used
            })
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.update({
                "disk_total": disk.total,
                "disk_used": disk.used,
                "disk_free": disk.free,
                "disk_percent": disk.percent
            })
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.update({
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            })
            
            # GPU metrics (if available)
            if torch.cuda.is_available():
                gpu_metrics = self._collect_gpu_metrics()
                metrics.update(gpu_metrics)
            
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU-specific metrics."""
        try:
            gpu_count = torch.cuda.device_count()
            gpu_metrics = {"gpu_count": gpu_count}
            
            for i in range(gpu_count):
                gpu_metrics.update({
                    f"gpu_{i}_memory_allocated": torch.cuda.memory_allocated(i),
                    f"gpu_{i}_memory_reserved": torch.cuda.memory_reserved(i),
                    f"gpu_{i}_memory_total": torch.cuda.get_device_properties(i).total_memory
                })
            
            return gpu_metrics
        except Exception:
            return {}
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log system metrics to file."""
        try:
            if self.lock:
                with self.lock:
                    with open(self.metrics_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
            else:
                with open(self.metrics_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            sys.stderr.write(f"Failed to log system metrics: {e}\n")

class PyTorchDebugTools:
    """PyTorch debugging tools integration for comprehensive logging."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.anomaly_detection_enabled = False
        self.profiler_active = False
        self.memory_tracker = {}
        self.gradient_history = []
        self.debug_context = {}
        
        # Initialize PyTorch debugging if enabled
        if self.config.enable_pytorch_debugging:
            self._setup_pytorch_debugging()
    
    def _setup_pytorch_debugging(self):
        """Setup PyTorch debugging tools."""
        try:
            # Enable anomaly detection if requested
            if self.config.enable_autograd_anomaly_detection:
                torch.autograd.set_detect_anomaly(True)
                self.anomaly_detection_enabled = True
            
            # Setup memory debugging
            if self.config.enable_memory_debugging and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Setup profiler if enabled
            if self.config.enable_profiler:
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=3,
                        repeat=2
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                )
                
        except Exception as e:
            warnings.warn(f"Failed to setup PyTorch debugging tools: {e}")
    
    def enable_anomaly_detection(self, enable: bool = True):
        """Enable/disable autograd anomaly detection."""
        try:
            torch.autograd.set_detect_anomaly(enable)
            self.anomaly_detection_enabled = enable
            return True
        except Exception as e:
            warnings.warn(f"Failed to set anomaly detection: {e}")
            return False
    
    def start_profiler(self):
        """Start PyTorch profiler."""
        if hasattr(self, 'profiler') and self.config.enable_profiler:
            try:
                self.profiler.start()
                self.profiler_active = True
                return True
            except Exception as e:
                warnings.warn(f"Failed to start profiler: {e}")
                return False
        return False
    
    def stop_profiler(self):
        """Stop PyTorch profiler."""
        if hasattr(self, 'profiler') and self.profiler_active:
            try:
                self.profiler.stop()
                self.profiler_active = False
                return True
            except Exception as e:
                warnings.warn(f"Failed to stop profiler: {e}")
                return False
        return False
    
    def debug_gradients(self, model: torch.nn.Module, loss: torch.Tensor):
        """Debug gradient information for a model."""
        if not self.config.enable_gradient_debugging:
            return {}
        
        debug_info = {}
        
        try:
            # Check for NaN/Inf in gradients
            total_norm = 0.0
            param_count = 0
            nan_count = 0
            inf_count = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_count += 1
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
                    # Check for NaN/Inf
                    if torch.isnan(param.grad).any():
                        nan_count += 1
                        debug_info[f"{name}_nan"] = True
                    
                    if torch.isinf(param.grad).any():
                        inf_count += 1
                        debug_info[f"{name}_inf"] = True
            
            total_norm = total_norm ** (1. / 2)
            
            debug_info.update({
                "total_grad_norm": total_norm,
                "param_count": param_count,
                "nan_gradients": nan_count,
                "inf_gradients": inf_count,
                "loss_value": loss.item() if hasattr(loss, 'item') else float(loss),
                "loss_requires_grad": loss.requires_grad,
                "loss_grad_fn": str(loss.grad_fn) if loss.grad_fn else None
            })
            
            # Store gradient history
            self.gradient_history.append({
                "timestamp": datetime.now().isoformat(),
                "total_norm": total_norm,
                "nan_count": nan_count,
                "inf_count": inf_count
            })
            
            # Keep only last 100 entries
            if len(self.gradient_history) > 100:
                self.gradient_history = self.gradient_history[-100:]
                
        except Exception as e:
            debug_info["gradient_debug_error"] = str(e)
        
        return debug_info
    
    def debug_memory_usage(self):
        """Debug PyTorch memory usage."""
        if not self.config.enable_memory_debugging:
            return {}
        
        memory_info = {}
        
        try:
            if torch.cuda.is_available():
                # CUDA memory stats
                memory_info.update({
                    "cuda_memory_allocated": torch.cuda.memory_allocated() / 1e9,  # GB
                    "cuda_memory_reserved": torch.cuda.memory_reserved() / 1e9,    # GB
                    "cuda_memory_cached": torch.cuda.memory_reserved() / 1e9,      # GB
                    "cuda_max_memory_allocated": torch.cuda.max_memory_allocated() / 1e9,  # GB
                    "cuda_max_memory_reserved": torch.cuda.max_memory_reserved() / 1e9,    # GB
                })
                
                # Per-device memory info
                for i in range(torch.cuda.device_count()):
                    device_name = f"cuda:{i}"
                    memory_info.update({
                        f"{device_name}_allocated": torch.cuda.memory_allocated(i) / 1e9,
                        f"{device_name}_reserved": torch.cuda.memory_reserved(i) / 1e9,
                        f"{device_name}_total": torch.cuda.get_device_properties(i).total_memory / 1e9,
                    })
            
            # CPU memory info
            if hasattr(torch, 'get_num_threads'):
                memory_info["torch_num_threads"] = torch.get_num_threads()
            
            # Memory fragmentation info
            if torch.cuda.is_available():
                memory_info["cuda_memory_fragmentation"] = (
                    torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                ) / torch.cuda.memory_reserved() if torch.cuda.memory_reserved() > 0 else 0
            
        except Exception as e:
            memory_info["memory_debug_error"] = str(e)
        
        return memory_info
    
    def debug_tensor_info(self, tensor: torch.Tensor, name: str = "tensor"):
        """Debug tensor information."""
        if not self.config.enable_tensor_debugging:
            return {}
        
        tensor_info = {}
        
        try:
            tensor_info.update({
                f"{name}_shape": list(tensor.shape),
                f"{name}_dtype": str(tensor.dtype),
                f"{name}_device": str(tensor.device),
                f"{name}_requires_grad": tensor.requires_grad,
                f"{name}_is_leaf": tensor.is_leaf,
                f"{name}_grad_fn": str(tensor.grad_fn) if tensor.grad_fn else None,
                f"{name}_numel": tensor.numel(),
                f"{name}_storage_size": tensor.storage().size() if tensor.storage() else None,
            })
            
            # Check for NaN/Inf values
            if tensor.numel() > 0:
                tensor_info.update({
                    f"{name}_has_nan": torch.isnan(tensor).any().item(),
                    f"{name}_has_inf": torch.isinf(tensor).any().item(),
                    f"{name}_nan_count": torch.isnan(tensor).sum().item(),
                    f"{name}_inf_count": torch.isinf(tensor).sum().item(),
                })
                
                # Statistical information
                if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                    tensor_info.update({
                        f"{name}_min": tensor.min().item() if tensor.numel() > 0 else None,
                        f"{name}_max": tensor.max().item() if tensor.numel() > 0 else None,
                        f"{name}_mean": tensor.mean().item() if tensor.numel() > 0 else None,
                        f"{name}_std": tensor.std().item() if tensor.numel() > 0 else None,
                    })
            
        except Exception as e:
            tensor_info[f"{name}_debug_error"] = str(e)
        
        return tensor_info
    
    def debug_model_state(self, model: torch.nn.Module):
        """Debug model state information."""
        if not self.config.enable_tensor_debugging:
            return {}
        
        model_info = {}
        
        try:
            model_info.update({
                "model_training_mode": model.training,
                "model_device": next(model.parameters()).device if list(model.parameters()) else "CPU",
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "non_trainable_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            })
            
            # Check for NaN/Inf in model parameters
            nan_params = 0
            inf_params = 0
            
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params += 1
                if torch.isinf(param).any():
                    inf_params += 1
            
            model_info.update({
                "parameters_with_nan": nan_params,
                "parameters_with_inf": inf_params,
            })
            
        except Exception as e:
            model_info["model_debug_error"] = str(e)
        
        return model_info
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debug summary."""
        return {
            "anomaly_detection_enabled": self.anomaly_detection_enabled,
            "profiler_active": self.profiler_active,
            "gradient_history": self.gradient_history[-20:],  # Last 20 entries
            "debug_context": self.debug_context,
            "memory_tracker": self.memory_tracker,
        }
    
    def cleanup(self):
        """Cleanup PyTorch debugging tools."""
        try:
            if self.anomaly_detection_enabled:
                torch.autograd.set_detect_anomaly(False)
            
            if self.profiler_active:
                self.stop_profiler()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            warnings.warn(f"Error during PyTorch debug cleanup: {e}")

class ComprehensiveLogger:
    """Main comprehensive logging system."""
    
    def __init__(self, name: str = "seo_evaluation", config: Optional[LoggingConfig] = None):
        self.name = name
        self.config = config or LoggingConfig()
        
        # Create log directory
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.logger = self._setup_logger()
        self.training_logger = TrainingMetricsLogger(str(self.log_dir), self.config)
        self.error_tracker = ErrorTracker(str(self.log_dir), self.config)
        self.system_monitor = SystemMonitor(str(self.log_dir), self.config)
        self.pytorch_debug = PyTorchDebugTools(self.config)
        
        # Start system monitoring if enabled
        if self.config.log_system_metrics:
            self.system_monitor.start_monitoring()
        
        # Log initialization
        self.logger.info("Comprehensive logging system initialized", extra={
            "context": {
                "log_dir": str(self.log_dir),
                "config": asdict(self.config),
                "pytorch_debugging_enabled": self.config.enable_pytorch_debugging,
                "anomaly_detection_enabled": self.pytorch_debug.anomaly_detection_enabled
            }
        })
    
    def _setup_logger(self) -> logging.Logger:
        """Setup the main logger with handlers and formatters."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = StructuredFormatter(
                include_timestamp=self.config.include_timestamp,
                include_context=self.config.include_context
            )
            console_handler.setFormatter(console_formatter)
            
            if self.config.enable_async_logging:
                console_handler = AsyncLogHandler(console_handler, self.config.max_queue_size)
            
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "application.log",
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = StructuredFormatter(
                include_timestamp=self.config.include_timestamp,
                include_context=self.config.include_context
            )
            file_handler.setFormatter(file_formatter)
            
            if self.config.enable_async_logging:
                file_handler = AsyncLogHandler(file_handler, self.config.max_queue_size)
            
            logger.addHandler(file_handler)
        
        # JSON handler
        if self.config.enable_json:
            json_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "application.jsonl",
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(logging.DEBUG)
            json_formatter = logging.Formatter('%(message)s')  # Raw JSON output
            json_handler.setFormatter(json_formatter)
            
            if self.config.enable_async_logging:
                json_handler = AsyncLogHandler(json_handler, self.config.max_queue_size)
            
            logger.addHandler(json_handler)
        
        return logger
    
    def log_training_step(self, epoch: int, step: int, loss: float, **kwargs):
        """Log training step with comprehensive metrics."""
        # Log to main logger
        self.logger.info(f"Training step {step} in epoch {epoch}", extra={
            "context": {
                "epoch": epoch,
                "step": step,
                "loss": loss,
                **kwargs
            }
        })
        
        # Log to training metrics logger
        if self.config.log_training_metrics:
            self.training_logger.log_training_step(epoch, step, loss, **kwargs)
    
    def log_validation_step(self, epoch: int, step: int, val_loss: float, **kwargs):
        """Log validation step with comprehensive metrics."""
        # Log to main logger
        self.logger.info(f"Validation step {step} in epoch {epoch}", extra={
            "context": {
                "epoch": epoch,
                "step": step,
                "val_loss": val_loss,
                **kwargs
            }
        })
        
        # Log to training metrics logger
        if self.config.log_training_metrics:
            self.training_logger.log_validation_step(epoch, step, val_loss, **kwargs)
    
    def log_epoch_summary(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Log epoch summary with comprehensive metrics."""
        # Log to main logger
        self.logger.info(f"Epoch {epoch} completed", extra={
            "context": {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **kwargs
            }
        })
        
        # Log to training metrics logger
        if self.config.log_training_metrics:
            self.training_logger.log_epoch_summary(epoch, train_loss, val_loss, **kwargs)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None, 
                  severity: str = "ERROR", recovery_attempted: bool = False):
        """Log error with comprehensive tracking."""
        # Log to main logger
        self.logger.error(f"Error occurred: {error}", extra={
            "context": context or {},
            "error_type": type(error).__name__,
            "severity": severity
        }, exc_info=True)
        
        # Track error
        if self.config.log_errors:
            self.error_tracker.track_error(error, context, severity, recovery_attempted)
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning with context."""
        self.logger.warning(message, extra={
            "context": context or {}
        })
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message with context."""
        self.logger.info(message, extra={
            "context": context or {}
        })
    
    def log_debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log debug message with context."""
        if self.config.log_debug:
            self.logger.debug(message, extra={
                "context": context or {}
            })
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        if self.config.log_performance:
            self.logger.info(f"Performance: {operation} completed in {duration:.4f}s", extra={
                "context": {
                    "operation": operation,
                    "duration": duration,
                    **kwargs
                }
            })
    
    @contextmanager
    def performance_tracking(self, operation: str):
        """Context manager for performance tracking."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log_performance(operation, duration)
    
    @contextmanager
    def pytorch_debugging(self, operation: str, model: Optional[torch.nn.Module] = None, 
                         enable_anomaly_detection: bool = False, enable_profiler: bool = False):
        """Context manager for PyTorch debugging operations."""
        if not self.config.enable_pytorch_debugging:
            yield
            return
        
        # Store original states
        original_anomaly_detection = self.pytorch_debug.anomaly_detection_enabled
        original_profiler_state = self.pytorch_debug.profiler_active
        
        try:
            # Enable requested debugging features
            if enable_anomaly_detection:
                self.enable_autograd_anomaly_detection(True)
            
            if enable_profiler:
                self.start_profiler()
            
            # Log debugging start
            self.logger.info(f"PyTorch debugging started for: {operation}", extra={
                "context": {
                    "operation": operation,
                    "anomaly_detection": enable_anomaly_detection,
                    "profiler": enable_profiler
                }
            })
            
            yield
            
            # Log debugging completion
            self.logger.info(f"PyTorch debugging completed for: {operation}")
            
        except Exception as e:
            # Log debugging errors
            self.logger.error(f"PyTorch debugging error in {operation}: {e}", exc_info=True)
            raise
        
        finally:
            # Restore original states
            if enable_anomaly_detection:
                self.enable_autograd_anomaly_detection(original_anomaly_detection)
            
            if enable_profiler:
                if original_profiler_state:
                    self.start_profiler()
                else:
                    self.stop_profiler()
    
    @contextmanager
    def gradient_debugging(self, model: torch.nn.Module, operation: str = "gradient_debug"):
        """Context manager for gradient debugging."""
        if not self.config.enable_gradient_debugging:
            yield
            return
        
        try:
            # Log gradient debugging start
            self.logger.info(f"Gradient debugging started for: {operation}")
            
            yield
            
            # Debug gradients after operation
            if hasattr(model, 'parameters'):
                # Create a dummy loss for debugging if none exists
                dummy_loss = torch.tensor(0.0, requires_grad=True)
                if torch.cuda.is_available():
                    dummy_loss = dummy_loss.cuda()
                
                gradient_debug = self.debug_model_gradients(model, dummy_loss, operation=operation)
                
                # Log gradient debugging results
                self.logger.info(f"Gradient debugging completed for: {operation}", extra={
                    "context": {
                        "operation": operation,
                        "gradient_debug": gradient_debug
                    }
                })
            
        except Exception as e:
            self.logger.error(f"Gradient debugging error in {operation}: {e}", exc_info=True)
            raise
    
    def get_logging_summary(self) -> Dict[str, Any]:
        """Get comprehensive logging summary."""
        return {
            "training_metrics": self.training_logger.get_metrics_summary(),
            "error_analysis": self.error_tracker.get_error_analysis(),
            "pytorch_debug": self.pytorch_debug.get_debug_summary(),
            "log_files": {
                "application_log": str(self.log_dir / "application.log"),
                "application_jsonl": str(self.log_dir / "application.jsonl"),
                "training_metrics": str(self.log_dir / "training_metrics.jsonl"),
                "training_progress": str(self.log_dir / "training_progress.csv"),
                "errors": str(self.log_dir / "errors.jsonl"),
                "error_summary": str(self.log_dir / "error_summary.json"),
                "system_metrics": str(self.log_dir / "system_metrics.jsonl")
            },
            "config": asdict(self.config)
        }
    
    # PyTorch Debugging Methods
    def debug_model_gradients(self, model: torch.nn.Module, loss: torch.Tensor, **kwargs):
        """Debug model gradients with comprehensive logging."""
        debug_info = self.pytorch_debug.debug_gradients(model, loss)
        
        # Log gradient debugging information
        self.logger.info("Model gradient debugging completed", extra={
            "context": {
                "gradient_debug": debug_info,
                **kwargs
            }
        })
        
        return debug_info
    
    def debug_model_memory(self, **kwargs):
        """Debug PyTorch memory usage with comprehensive logging."""
        memory_info = self.pytorch_debug.debug_memory_usage()
        
        # Log memory debugging information
        self.logger.info("Model memory debugging completed", extra={
            "context": {
                "memory_debug": memory_info,
                **kwargs
            }
        })
        
        return memory_info
    
    def debug_tensor(self, tensor: torch.Tensor, name: str = "tensor", **kwargs):
        """Debug tensor information with comprehensive logging."""
        tensor_info = self.pytorch_debug.debug_tensor_info(tensor, name)
        
        # Log tensor debugging information
        self.logger.info(f"Tensor '{name}' debugging completed", extra={
            "context": {
                "tensor_debug": tensor_info,
                **kwargs
            }
        })
        
        return tensor_info
    
    def debug_model_state(self, model: torch.nn.Module, **kwargs):
        """Debug model state with comprehensive logging."""
        model_info = self.pytorch_debug.debug_model_state(model)
        
        # Log model state debugging information
        self.logger.info("Model state debugging completed", extra={
            "context": {
                "model_debug": model_info,
                **kwargs
            }
        })
        
        return model_info
    
    def enable_autograd_anomaly_detection(self, enable: bool = True):
        """Enable/disable autograd anomaly detection."""
        success = self.pytorch_debug.enable_anomaly_detection(enable)
        
        if success:
            self.logger.info(f"Autograd anomaly detection {'enabled' if enable else 'disabled'}")
        else:
            self.logger.warning(f"Failed to {'enable' if enable else 'disable'} autograd anomaly detection")
        
        return success
    
    def start_profiler(self):
        """Start PyTorch profiler."""
        success = self.pytorch_debug.start_profiler()
        
        if success:
            self.logger.info("PyTorch profiler started")
        else:
            self.logger.warning("Failed to start PyTorch profiler")
        
        return success
    
    def stop_profiler(self):
        """Stop PyTorch profiler."""
        success = self.pytorch_debug.stop_profiler()
        
        if success:
            self.logger.info("PyTorch profiler stopped")
        else:
            self.logger.warning("Failed to stop PyTorch profiler")
        
        return success
    
    def log_training_step_with_debug(self, epoch: int, step: int, loss: float, model: torch.nn.Module, **kwargs):
        """Log training step with comprehensive PyTorch debugging."""
        # Standard training step logging
        self.log_training_step(epoch, step, loss, **kwargs)
        
        # PyTorch debugging if enabled
        if self.config.enable_pytorch_debugging:
            # Debug gradients
            gradient_debug = self.debug_model_gradients(model, loss, epoch=epoch, step=step)
            
            # Debug memory
            memory_debug = self.debug_model_memory(epoch=epoch, step=step)
            
            # Debug model state
            model_debug = self.debug_model_state(model, epoch=epoch, step=step)
            
            # Log comprehensive debug information
            self.logger.info(f"Training step {step} debugging completed", extra={
                "context": {
                    "epoch": epoch,
                    "step": step,
                    "gradient_debug": gradient_debug,
                    "memory_debug": memory_debug,
                    "model_debug": model_debug,
                    **kwargs
                }
            })
            
            return {
                "gradient_debug": gradient_debug,
                "memory_debug": memory_debug,
                "model_debug": model_debug
            }
        
        return {}
    
    def cleanup(self):
        """Cleanup logging resources."""
        try:
            # Stop system monitoring
            self.system_monitor.stop_monitoring()
            
            # Cleanup PyTorch debugging tools
            self.pytorch_debug.cleanup()
            
            # Close handlers
            for handler in self.logger.handlers:
                if hasattr(handler, 'close'):
                    handler.close()
            
            # Log cleanup
            self.logger.info("Logging system cleanup completed")
            
        except Exception as e:
            sys.stderr.write(f"Error during logging cleanup: {e}\n")

# Utility functions for easy integration
def setup_logging(name: str = "seo_evaluation", **kwargs) -> ComprehensiveLogger:
    """Setup comprehensive logging system with custom configuration."""
    config = LoggingConfig(**kwargs)
    return ComprehensiveLogger(name, config)

def get_logger(name: str = "seo_evaluation") -> ComprehensiveLogger:
    """Get existing logger instance or create new one."""
    return ComprehensiveLogger(name)

# Example usage
if __name__ == "__main__":
    # Setup logging with PyTorch debugging enabled
    logger = setup_logging(
        log_level="DEBUG",
        log_dir="./logs",
        enable_console=True,
        enable_file=True,
        enable_json=True,
        log_training_metrics=True,
        log_system_metrics=True,
        log_gpu_metrics=True,
        enable_pytorch_debugging=True,
        enable_gradient_debugging=True,
        enable_memory_debugging=True,
        enable_tensor_debugging=True
    )
    
    try:
        # Example PyTorch model for debugging demonstration
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        ).to(device)
        
        # Create dummy data
        x = torch.randn(32, 10).to(device)
        y = torch.randn(32, 1).to(device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Example training loop with PyTorch debugging
        for epoch in range(3):
            for step in range(5):
                # Forward pass
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                
                # Backward pass
                loss.backward()
                
                # Debug gradients before optimization
                gradient_debug = logger.debug_model_gradients(model, loss, epoch=epoch, step=step)
                
                # Debug memory usage
                memory_debug = logger.debug_model_memory(epoch=epoch, step=step)
                
                # Debug model state
                model_debug = logger.debug_model_state(model, epoch=epoch, step=step)
                
                # Optimizer step
                optimizer.step()
                
                # Log training step with comprehensive debugging
                logger.log_training_step_with_debug(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    model=model,
                    accuracy=0.8,  # Dummy accuracy
                    learning_rate=optimizer.param_groups[0]['lr'],
                    gradient_norm=gradient_debug.get('total_grad_norm', 0.0)
                )
                
                time.sleep(0.1)  # Simulate work
            
            # Log epoch summary
            logger.log_epoch_summary(
                epoch=epoch,
                train_loss=0.5,
                val_loss=0.6,
                train_accuracy=0.8,
                val_accuracy=0.75
            )
        
        # Example of using context managers for PyTorch debugging
        print("\n=== PyTorch Debugging Context Managers Demo ===")
        
        # Performance tracking with PyTorch debugging
        with logger.performance_tracking("model_inference_with_debug"):
            with logger.pytorch_debugging("model_inference", model=model, enable_anomaly_detection=True):
                # Simulate model inference
                with torch.no_grad():
                    test_output = model(x[:5])
                    print(f"Test output shape: {test_output.shape}")
        
        # Gradient debugging context manager
        with logger.gradient_debugging(model, "test_operation"):
            # Simulate some operation that affects gradients
            test_loss = criterion(model(x[:5]), y[:5])
            test_loss.backward()
        
        # Example error logging
        try:
            raise ValueError("Example error for testing")
        except Exception as e:
            logger.log_error(e, context={"operation": "example"}, severity="ERROR")
        
        # Get comprehensive summary including PyTorch debugging
        summary = logger.get_logging_summary()
        print("\n=== Comprehensive Logging Summary ===")
        print("Logging Summary:", json.dumps(summary, indent=2, default=str))
        
        # Show PyTorch debugging summary
        pytorch_debug_summary = summary.get('pytorch_debug', {})
        print(f"\n=== PyTorch Debugging Summary ===")
        print(f"Anomaly Detection Enabled: {pytorch_debug_summary.get('anomaly_detection_enabled', False)}")
        print(f"Profiler Active: {pytorch_debug_summary.get('profiler_active', False)}")
        print(f"Gradient History Entries: {len(pytorch_debug_summary.get('gradient_history', []))}")
        
    finally:
        # Cleanup
        logger.cleanup()
