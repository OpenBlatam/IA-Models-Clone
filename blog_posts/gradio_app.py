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

import gradio as gr
import torch
from PIL import Image
from advanced_diffusion_pipelines import PipelineConfig, create_pipeline, PipelineManager
from advanced_model_training import ModelEvaluator
import numpy as np
import logging
import time
import gc
import psutil
import threading
import asyncio
from functools import lru_cache
from collections import defaultdict, deque
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import io
from typing import List, Optional, Tuple, Dict, Any, Union
import traceback
import re
import sys
from datetime import datetime
import os
from pathlib import Path
            import xformers
            import torch.distributed as dist
                import torch.distributed as dist
        from mixed_precision_training import (
        from mixed_precision_training import benchmark_mixed_precision
        from mixed_precision_training import get_mixed_precision_recommendations
        from mixed_precision_training import MixedPrecisionConfig
from typing import Any, List, Dict, Optional
"""
🚀 Gradio Integration for Model Inference & Evaluation
=====================================================

Production-ready Gradio app for interactive model inference and evaluation.
Supports classification and regression tasks, metrics, and visualization.
Enhanced with comprehensive error handling and debugging capabilities.
"""


# Enhanced logging configuration with comprehensive training progress and error logging

# PyTorch debugging tools and utilities
class PyTorchDebugger:
    """Comprehensive PyTorch debugging utilities."""
    
    def __init__(self, debug_mode: bool = False):
        
    """__init__ function."""
self.debug_mode = debug_mode
        self.anomaly_detection_enabled = False
        self.profiler_active = False
        self.memory_tracking_enabled = False
        self.gradient_tracking_enabled = False
        self.debug_info = {}
        
    def enable_anomaly_detection(self, enabled: bool = True):
        """Enable/disable autograd anomaly detection."""
        try:
            if enabled and not self.anomaly_detection_enabled:
                torch.autograd.set_detect_anomaly(True)
                self.anomaly_detection_enabled = True
                logger.info("✅ PyTorch autograd anomaly detection enabled")
            elif not enabled and self.anomaly_detection_enabled:
                torch.autograd.set_detect_anomaly(False)
                self.anomaly_detection_enabled = False
                logger.info("❌ PyTorch autograd anomaly detection disabled")
        except Exception as e:
            logger.error(f"Failed to toggle anomaly detection: {e}")
    
    def start_profiler(self, record_shapes: bool = True, profile_memory: bool = True, 
                      with_stack: bool = True, use_cuda: bool = True):
        """Start PyTorch profiler for performance analysis."""
        try:
            if not self.profiler_active:
                profiler_config = {
                    'record_shapes': record_shapes,
                    'profile_memory': profile_memory,
                    'with_stack': with_stack,
                    'use_cuda': use_cuda and torch.cuda.is_available()
                }
                
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA if use_cuda and torch.cuda.is_available() else None
                    ],
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=3,
                        repeat=2
                    )
                )
                
                self.profiler.start()
                self.profiler_active = True
                self.debug_info['profiler_config'] = profiler_config
                logger.info("✅ PyTorch profiler started")
                
        except Exception as e:
            logger.error(f"Failed to start profiler: {e}")
    
    def stop_profiler(self, export_path: str = None):
        """Stop PyTorch profiler and optionally export results."""
        try:
            if self.profiler_active:
                self.profiler.stop()
                self.profiler_active = False
                
                # Export profiler results
                if export_path:
                    self.profiler.export_chrome_trace(export_path)
                    logger.info(f"✅ Profiler results exported to {export_path}")
                
                # Log profiler summary
                self._log_profiler_summary()
                logger.info("❌ PyTorch profiler stopped")
                
        except Exception as e:
            logger.error(f"Failed to stop profiler: {e}")
    
    def _log_profiler_summary(self) -> Any:
        """Log profiler summary information."""
        try:
            if hasattr(self, 'profiler'):
                # Get key metrics from profiler
                key_averages = self.profiler.key_averages()
                
                # Log CPU and CUDA time
                cpu_time = sum(event.cpu_time_total for event in key_averages) / 1000  # Convert to seconds
                cuda_time = sum(event.cuda_time_total for event in key_averages) / 1000 if torch.cuda.is_available() else 0
                
                logger.info(f"Profiler Summary - CPU Time: {cpu_time:.3f}s, CUDA Time: {cuda_time:.3f}s")
                
                # Log top operations by time
                top_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:5]
                for i, op in enumerate(top_ops):
                    logger.info(f"  Top {i+1}: {op.name} - CPU: {op.cpu_time_total/1000:.3f}s, CUDA: {op.cuda_time_total/1000:.3f}s")
                
        except Exception as e:
            logger.error(f"Failed to log profiler summary: {e}")
    
    def enable_memory_tracking(self, enabled: bool = True):
        """Enable/disable memory tracking."""
        try:
            if enabled and not self.memory_tracking_enabled:
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                self.memory_tracking_enabled = True
                self.debug_info['memory_tracking_start'] = {
                    'cpu_memory': psutil.virtual_memory().percent,
                    'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                }
                logger.info("✅ Memory tracking enabled")
            elif not enabled and self.memory_tracking_enabled:
                self.memory_tracking_enabled = False
                logger.info("❌ Memory tracking disabled")
        except Exception as e:
            logger.error(f"Failed to toggle memory tracking: {e}")
    
    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get current memory statistics."""
        try:
            stats = {
                'cpu_memory_percent': psutil.virtual_memory().percent,
                'cpu_memory_available_gb': psutil.virtual_memory().available / 1024**3
            }
            
            if torch.cuda.is_available():
                stats.update({
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                    'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                    'gpu_memory_max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3
                })
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    def enable_gradient_tracking(self, enabled: bool = True):
        """Enable/disable gradient tracking for debugging."""
        try:
            if enabled and not self.gradient_tracking_enabled:
                self.gradient_tracking_enabled = True
                logger.info("✅ Gradient tracking enabled")
            elif not enabled and self.gradient_tracking_enabled:
                self.gradient_tracking_enabled = False
                logger.info("❌ Gradient tracking disabled")
        except Exception as e:
            logger.error(f"Failed to toggle gradient tracking: {e}")
    
    def check_gradients(self, model: torch.nn.Module, log_gradients: bool = False):
        """Check gradients for NaN/Inf values."""
        try:
            if not self.gradient_tracking_enabled:
                return True, "Gradient tracking not enabled"
            
            has_nan = False
            has_inf = False
            gradient_info = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_nan = torch.isnan(param.grad).any().item()
                    grad_inf = torch.isinf(param.grad).any().item()
                    
                    if grad_nan or grad_inf:
                        has_nan = has_nan or grad_nan
                        has_inf = has_inf or grad_inf
                        
                        if log_gradients:
                            gradient_info[name] = {
                                'has_nan': grad_nan,
                                'has_inf': grad_inf,
                                'grad_norm': param.grad.norm().item(),
                                'param_norm': param.norm().item()
                            }
            
            if has_nan or has_inf:
                error_msg = f"Gradient issues detected: NaN={has_nan}, Inf={has_inf}"
                if log_gradients:
                    logger.error(f"{error_msg} - Details: {gradient_info}")
                return False, error_msg
            
            return True, "Gradients are valid"
            
        except Exception as e:
            logger.error(f"Failed to check gradients: {e}")
            return False, f"Gradient check failed: {e}"
    
    def debug_tensor(self, tensor: torch.Tensor, name: str = "tensor", log_details: bool = True):
        """Debug tensor properties and values."""
        try:
            debug_info = {
                'name': name,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'requires_grad': tensor.requires_grad,
                'is_leaf': tensor.is_leaf,
                'numel': tensor.numel(),
                'memory_size_mb': tensor.element_size() * tensor.numel() / 1024**2
            }
            
            # Check for NaN/Inf values
            if tensor.numel() > 0:
                debug_info.update({
                    'has_nan': torch.isnan(tensor).any().item(),
                    'has_inf': torch.isinf(tensor).any().item(),
                    'min_value': tensor.min().item() if tensor.numel() > 0 else None,
                    'max_value': tensor.max().item() if tensor.numel() > 0 else None,
                    'mean_value': tensor.mean().item() if tensor.numel() > 0 else None,
                    'std_value': tensor.std().item() if tensor.numel() > 0 else None
                })
            
            if log_details:
                logger.info(f"Tensor Debug - {name}: {debug_info}")
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Failed to debug tensor {name}: {e}")
            return {'name': name, 'error': str(e)}
    
    def debug_model(self, model: torch.nn.Module, log_details: bool = True):
        """Debug model parameters and structure."""
        try:
            debug_info = {
                'model_name': model.__class__.__name__,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'modules': len(list(model.modules())),
                'device': next(model.parameters()).device if list(model.parameters()) else None
            }
            
            # Check parameter statistics
            param_stats = {}
            for name, param in model.named_parameters():
                param_stats[name] = {
                    'shape': list(param.shape),
                    'requires_grad': param.requires_grad,
                    'has_nan': torch.isnan(param).any().item(),
                    'has_inf': torch.isinf(param).any().item(),
                    'norm': param.norm().item(),
                    'grad_norm': param.grad.norm().item() if param.grad is not None else None
                }
            
            debug_info['parameter_stats'] = param_stats
            
            if log_details:
                logger.info(f"Model Debug - {debug_info['model_name']}: {debug_info}")
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Failed to debug model: {e}")
            return {'error': str(e)}
    
    def context_manager(self, operation_name: str):
        """Context manager for debugging operations."""
        return PyTorchDebugContext(self, operation_name)

class PyTorchDebugContext:
    """Context manager for PyTorch debugging operations."""
    
    def __init__(self, debugger: PyTorchDebugger, operation_name: str):
        
    """__init__ function."""
self.debugger = debugger
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self) -> Any:
        """Enter debugging context."""
        self.start_time = time.time()
        self.start_memory = self.debugger.get_memory_stats()
        
        logger.info(f"🔍 Starting debug context: {self.operation_name}")
        return self.debugger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Exit debugging context."""
        end_time = time.time()
        end_memory = self.debugger.get_memory_stats()
        
        duration = end_time - self.start_time
        
        # Log operation summary
        logger.info(f"🔍 Debug context completed: {self.operation_name} - Duration: {duration:.3f}s")
        
        # Log memory changes
        if self.start_memory and end_memory:
            cpu_diff = end_memory.get('cpu_memory_percent', 0) - self.start_memory.get('cpu_memory_percent', 0)
            gpu_diff = end_memory.get('gpu_memory_allocated_gb', 0) - self.start_memory.get('gpu_memory_allocated_gb', 0)
            
            logger.info(f"  Memory changes - CPU: {cpu_diff:+.1f}%, GPU: {gpu_diff:+.2f}GB")
        
        # Log any exceptions
        if exc_type is not None:
            logger.error(f"  Exception in {self.operation_name}: {exc_type.__name__}: {exc_val}")
            return False  # Re-raise the exception
        
        return True

# Global debugger instance
pytorch_debugger = PyTorchDebugger(debug_mode=False)

# Performance Optimization System
class PerformanceOptimizer:
    """Comprehensive performance optimization utilities."""
    
    def __init__(self) -> Any:
        self.optimization_config = {
            'memory_efficient_attention': True,
            'compile_models': True,
            'use_channels_last': True,
            'enable_xformers': True,
            'optimize_for_inference': True,
            'use_torch_compile': True,
            'enable_amp': True,
            'use_fast_math': True
        }
        self.performance_metrics = defaultdict(list)
        self.optimization_history = []
        self.current_optimizations = set()
        
    def optimize_pipeline_performance(self, pipeline, config: Dict[str, Any] = None):
        """Apply comprehensive performance optimizations to pipeline."""
        try:
            if config:
                self.optimization_config.update(config)
            
            optimizations_applied = []
            
            # 1. Enable memory efficient attention
            if self.optimization_config['memory_efficient_attention']:
                if hasattr(pipeline, 'unet'):
                    self._enable_memory_efficient_attention(pipeline.unet)
                    optimizations_applied.append('memory_efficient_attention')
            
            # 2. Use channels last memory format
            if self.optimization_config['use_channels_last']:
                if hasattr(pipeline, 'unet'):
                    self._convert_to_channels_last(pipeline.unet)
                    optimizations_applied.append('channels_last')
            
            # 3. Enable xformers attention
            if self.optimization_config['enable_xformers']:
                if hasattr(pipeline, 'unet'):
                    self._enable_xformers_attention(pipeline.unet)
                    optimizations_applied.append('xformers_attention')
            
            # 4. Optimize for inference
            if self.optimization_config['optimize_for_inference']:
                self._optimize_for_inference(pipeline)
                optimizations_applied.append('inference_optimization')
            
            # 5. Compile models with torch.compile
            if self.optimization_config['use_torch_compile']:
                if hasattr(pipeline, 'unet'):
                    self._compile_model(pipeline.unet)
                    optimizations_applied.append('torch_compile')
            
            # 6. Enable fast math
            if self.optimization_config['use_fast_math']:
                self._enable_fast_math()
                optimizations_applied.append('fast_math')
            
            # Record optimizations
            self.current_optimizations.update(optimizations_applied)
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'optimizations': optimizations_applied,
                'config': self.optimization_config.copy()
            })
            
            logger.info(f"✅ Performance optimizations applied: {optimizations_applied}")
            return optimizations_applied
            
        except Exception as e:
            logger.error(f"Failed to apply performance optimizations: {e}")
            return []
    
    def _enable_memory_efficient_attention(self, model) -> Any:
        """Enable memory efficient attention in model."""
        try:
            for module in model.modules():
                if hasattr(module, 'attention_head_size'):
                    # Set attention to use memory efficient implementation
                    if hasattr(module, 'set_attention_slice'):
                        module.set_attention_slice(slice_size=1)
                    if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
                        module.set_use_memory_efficient_attention_xformers(True)
        except Exception as e:
            logger.warning(f"Failed to enable memory efficient attention: {e}")
    
    def _convert_to_channels_last(self, model) -> Any:
        """Convert model to channels last memory format."""
        try:
            model.to(memory_format=torch.channels_last)
            logger.info("✅ Model converted to channels last memory format")
        except Exception as e:
            logger.warning(f"Failed to convert to channels last: {e}")
    
    def _enable_xformers_attention(self, model) -> Any:
        """Enable xformers attention if available."""
        try:
            for module in model.modules():
                if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
                    module.set_use_memory_efficient_attention_xformers(True)
            logger.info("✅ Xformers attention enabled")
        except ImportError:
            logger.warning("Xformers not available, skipping xformers optimization")
        except Exception as e:
            logger.warning(f"Failed to enable xformers attention: {e}")
    
    def _optimize_for_inference(self, pipeline) -> Any:
        """Optimize pipeline for inference."""
        try:
            # Set to evaluation mode
            if hasattr(pipeline, 'unet'):
                pipeline.unet.eval()
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder.eval()
            if hasattr(pipeline, 'vae'):
                pipeline.vae.eval()
            
            # Disable gradient computation
            if hasattr(pipeline, 'unet'):
                for param in pipeline.unet.parameters():
                    param.requires_grad = False
            
            logger.info("✅ Pipeline optimized for inference")
        except Exception as e:
            logger.warning(f"Failed to optimize for inference: {e}")
    
    def _compile_model(self, model) -> Any:
        """Compile model with torch.compile."""
        try:
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(model, mode='reduce-overhead')
                # Replace the model with compiled version
                if hasattr(model, '_compiled_model'):
                    model._compiled_model = compiled_model
                logger.info("✅ Model compiled with torch.compile")
            else:
                logger.warning("torch.compile not available")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
    
    def _enable_fast_math(self) -> Any:
        """Enable fast math operations."""
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ Fast math operations enabled")
        except Exception as e:
            logger.warning(f"Failed to enable fast math: {e}")
    
    def optimize_batch_processing(self, batch_size: int, available_memory: float) -> Dict[str, Any]:
        """Optimize batch processing based on available memory."""
        try:
            # Calculate optimal batch size based on memory
            optimal_batch_size = self._calculate_optimal_batch_size(available_memory)
            
            # Calculate optimal accumulation steps
            accumulation_steps = max(1, batch_size // optimal_batch_size)
            
            # Calculate memory usage per batch
            memory_per_batch = available_memory / optimal_batch_size
            
            optimization_config = {
                'optimal_batch_size': optimal_batch_size,
                'accumulation_steps': accumulation_steps,
                'memory_per_batch_gb': memory_per_batch,
                'total_batches': (batch_size + optimal_batch_size - 1) // optimal_batch_size
            }
            
            logger.info(f"✅ Batch optimization: {optimization_config}")
            return optimization_config
            
        except Exception as e:
            logger.error(f"Failed to optimize batch processing: {e}")
            return {'optimal_batch_size': batch_size, 'accumulation_steps': 1}
    
    def _calculate_optimal_batch_size(self, available_memory: float) -> int:
        """Calculate optimal batch size based on available memory."""
        try:
            # Base memory requirements (in GB)
            base_memory_gb = 2.0  # Base memory for model and operations
            
            # Memory per image (estimated)
            memory_per_image_gb = 0.5  # Conservative estimate
            
            # Calculate optimal batch size
            usable_memory = available_memory - base_memory_gb
            optimal_batch_size = max(1, int(usable_memory / memory_per_image_gb))
            
            # Cap at reasonable maximum
            optimal_batch_size = min(optimal_batch_size, 16)
            
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal batch size: {e}")
            return 1
    
    def optimize_memory_usage(self, pipeline) -> Dict[str, Any]:
        """Optimize memory usage for pipeline."""
        try:
            memory_optimizations = {}
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_optimizations['cache_cleared'] = True
            
            # Set memory fraction
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.9)
                memory_optimizations['memory_fraction_set'] = True
            
            # Enable gradient checkpointing if available
            if hasattr(pipeline, 'unet'):
                if hasattr(pipeline.unet, 'enable_gradient_checkpointing'):
                    pipeline.unet.enable_gradient_checkpointing()
                    memory_optimizations['gradient_checkpointing'] = True
            
            # Use mixed precision
            if self.optimization_config['enable_amp']:
                memory_optimizations['mixed_precision'] = True
            
            logger.info(f"✅ Memory optimizations applied: {list(memory_optimizations.keys())}")
            return memory_optimizations
            
        except Exception as e:
            logger.error(f"Failed to optimize memory usage: {e}")
            return {}
    
    def measure_performance(self, operation_name: str, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance of an operation."""
        try:
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Run operation
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate metrics
            duration = end_time - start_time
            memory_used = (end_memory - start_memory) / 1024**3  # GB
            throughput = 1.0 / duration if duration > 0 else 0
            
            metrics = {
                'operation': operation_name,
                'duration_seconds': duration,
                'memory_used_gb': memory_used,
                'throughput_ops_per_second': throughput,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store metrics
            self.performance_metrics[operation_name].append(metrics)
            
            logger.info(f"📊 Performance: {operation_name} - {duration:.3f}s, {memory_used:.2f}GB")
            return result, metrics
            
        except Exception as e:
            logger.error(f"Failed to measure performance for {operation_name}: {e}")
            return func(*args, **kwargs), {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all operations."""
        try:
            summary = {
                'total_operations': len(self.performance_metrics),
                'optimizations_applied': list(self.current_optimizations),
                'optimization_history': self.optimization_history[-10:],  # Last 10
                'performance_by_operation': {}
            }
            
            for operation, metrics_list in self.performance_metrics.items():
                if metrics_list:
                    avg_duration = sum(m['duration_seconds'] for m in metrics_list) / len(metrics_list)
                    avg_memory = sum(m['memory_used_gb'] for m in metrics_list) / len(metrics_list)
                    avg_throughput = sum(m['throughput_ops_per_second'] for m in metrics_list) / len(metrics_list)
                    
                    summary['performance_by_operation'][operation] = {
                        'total_runs': len(metrics_list),
                        'avg_duration_seconds': avg_duration,
                        'avg_memory_gb': avg_memory,
                        'avg_throughput_ops_per_second': avg_throughput,
                        'min_duration': min(m['duration_seconds'] for m in metrics_list),
                        'max_duration': max(m['duration_seconds'] for m in metrics_list)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def auto_tune_parameters(self, pipeline, sample_input: str, target_throughput: float = 1.0) -> Dict[str, Any]:
        """Auto-tune parameters for optimal performance."""
        try:
            tuning_results = {}
            
            # Test different batch sizes
            batch_sizes = [1, 2, 4, 8, 16]
            batch_performance = {}
            
            for batch_size in batch_sizes:
                try:
                    # Measure performance with this batch size
                    _, metrics = self.measure_performance(
                        f"batch_size_{batch_size}",
                        lambda: self._test_batch_performance(pipeline, sample_input, batch_size),
                        pipeline, sample_input, batch_size
                    )
                    
                    batch_performance[batch_size] = metrics
                    
                    # Check if we've reached target throughput
                    if metrics.get('throughput_ops_per_second', 0) >= target_throughput:
                        tuning_results['optimal_batch_size'] = batch_size
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to test batch size {batch_size}: {e}")
                    continue
            
            # Test different precision modes
            precision_modes = ['fp32', 'fp16', 'bf16']
            precision_performance = {}
            
            for precision in precision_modes:
                try:
                    _, metrics = self.measure_performance(
                        f"precision_{precision}",
                        lambda: self._test_precision_performance(pipeline, sample_input, precision),
                        pipeline, sample_input, precision
                    )
                    
                    precision_performance[precision] = metrics
                    
                except Exception as e:
                    logger.warning(f"Failed to test precision {precision}: {e}")
                    continue
            
            # Find optimal configuration
            if batch_performance:
                optimal_batch = max(batch_performance.keys(), 
                                  key=lambda x: batch_performance[x].get('throughput_ops_per_second', 0))
                tuning_results['optimal_batch_size'] = optimal_batch
            
            if precision_performance:
                optimal_precision = max(precision_performance.keys(),
                                      key=lambda x: precision_performance[x].get('throughput_ops_per_second', 0))
                tuning_results['optimal_precision'] = optimal_precision
            
            tuning_results['batch_performance'] = batch_performance
            tuning_results['precision_performance'] = precision_performance
            
            logger.info(f"✅ Auto-tuning completed: {tuning_results}")
            return tuning_results
            
        except Exception as e:
            logger.error(f"Failed to auto-tune parameters: {e}")
            return {}
    
    def _test_batch_performance(self, pipeline, input_text: str, batch_size: int):
        """Test performance with specific batch size."""
        try:
            # Create dummy generator
            generator = torch.manual_seed(42)
            
            # Run inference
            output = pipeline(
                input_text,
                num_images_per_prompt=batch_size,
                generator=generator,
                num_inference_steps=20  # Reduced for testing
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Failed to test batch performance: {e}")
            raise
    
    def _test_precision_performance(self, pipeline, input_text: str, precision: str):
        """Test performance with specific precision."""
        try:
            generator = torch.manual_seed(42)
            
            if precision == 'fp16':
                with torch.cuda.amp.autocast():
                    output = pipeline(input_text, num_images_per_prompt=1, generator=generator)
            elif precision == 'bf16':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = pipeline(input_text, num_images_per_prompt=1, generator=generator)
            else:  # fp32
                output = pipeline(input_text, num_images_per_prompt=1, generator=generator)
            
            return output
            
        except Exception as e:
            logger.error(f"Failed to test precision performance: {e}")
            raise

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure different loggers for different purposes
def setup_logging():
    """Setup comprehensive logging configuration for training progress and errors."""
    
    # Main application logger
    main_logger = logging.getLogger("gradio_app")
    main_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    main_logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    main_logger.addHandler(console_handler)
    
    # File handler for general logs
    general_handler = logging.FileHandler(logs_dir / "gradio_app.log")
    general_handler.setLevel(logging.INFO)
    general_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    general_handler.setFormatter(general_formatter)
    main_logger.addHandler(general_handler)
    
    # Training progress logger
    training_logger = logging.getLogger("training_progress")
    training_logger.setLevel(logging.INFO)
    training_logger.handlers.clear()
    
    training_handler = logging.FileHandler(logs_dir / "training_progress.log")
    training_handler.setLevel(logging.INFO)
    training_formatter = logging.Formatter(
        "[%(asctime)s] TRAINING - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    training_handler.setFormatter(training_formatter)
    training_logger.addHandler(training_handler)
    
    # Error logger
    error_logger = logging.getLogger("error_logger")
    error_logger.setLevel(logging.ERROR)
    error_logger.handlers.clear()
    
    error_handler = logging.FileHandler(logs_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        "[%(asctime)s] ERROR - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s\nTraceback: %(pathname)s:%(lineno)d\n",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    error_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_handler)
    
    # Performance logger
    performance_logger = logging.getLogger("performance")
    performance_logger.setLevel(logging.INFO)
    performance_logger.handlers.clear()
    
    performance_handler = logging.FileHandler(logs_dir / "performance.log")
    performance_handler.setLevel(logging.INFO)
    performance_formatter = logging.Formatter(
        "[%(asctime)s] PERFORMANCE - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    performance_handler.setFormatter(performance_formatter)
    performance_logger.addHandler(performance_handler)
    
    # Model operations logger
    model_logger = logging.getLogger("model_operations")
    model_logger.setLevel(logging.INFO)
    model_logger.handlers.clear()
    
    model_handler = logging.FileHandler(logs_dir / "model_operations.log")
    model_handler.setLevel(logging.INFO)
    model_formatter = logging.Formatter(
        "[%(asctime)s] MODEL - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    model_handler.setFormatter(model_formatter)
    model_logger.addHandler(model_handler)
    
    return main_logger

# Setup logging
logger = setup_logging()
training_logger = logging.getLogger("training_progress")
error_logger = logging.getLogger("error_logger")
performance_logger = logging.getLogger("performance")
model_logger = logging.getLogger("model_operations")

# Custom exception classes for better error handling
class GradioAppError(Exception):
    """Base exception for Gradio app errors."""
    pass

class InputValidationError(GradioAppError):
    """Raised when input validation fails."""
    pass

class ModelLoadingError(GradioAppError):
    """Raised when model loading fails."""
    pass

class InferenceError(GradioAppError):
    """Raised when inference fails."""
    pass

class MemoryError(GradioAppError):
    """Raised when memory issues occur."""
    pass

# Enhanced input validation patterns
INPUT_PATTERNS = {
    'prompt': {
        'min_length': 1,
        'max_length': 1000,
        'forbidden_chars': ['<script>', 'javascript:', 'onerror='],
        'required_words': [],
        'max_words': 200
    },
    'seed': {
        'min_value': -2**31,
        'max_value': 2**31 - 1
    },
    'num_images': {
        'min_value': 1,
        'max_value': 8
    }
}

PIPELINE_CONFIGS = {
    "Stable Diffusion v1.5": PipelineConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        model_type="stable-diffusion",
        num_inference_steps=30,
        guidance_scale=7.5
    ),
    "Stable Diffusion XL": PipelineConfig(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        model_type="stable-diffusion-xl",
        num_inference_steps=30,
        guidance_scale=7.5
    ),
}

manager = PipelineManager()
for name, config in PIPELINE_CONFIGS.items():
    try:
        logger.info(f"Loading pipeline '{name}'...")
        pipeline = create_pipeline(config)
        manager.add_pipeline(name, pipeline)
        logger.info(f"Pipeline '{name}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load pipeline '{name}': {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

# Enhanced monitoring state with error tracking
monitoring_data = {
    'inference_history': deque(maxlen=100),
    'memory_history': deque(maxlen=100),
    'error_history': deque(maxlen=50),
    'performance_metrics': defaultdict(list),
    'model_health': {},
    'system_stats': {},
    'error_counts': defaultdict(int),
    'debug_logs': deque(maxlen=200),
    'input_validation_failures': deque(maxlen=50)
}

def log_debug_info(message: str, data: Dict[str, Any] = None):
    """Log debug information with timestamp and context."""
    debug_entry = {
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'data': data or {},
        'memory_usage': psutil.virtual_memory().percent,
        'gpu_memory': get_gpu_utilization() if torch.cuda.is_available() else None
    }
    monitoring_data['debug_logs'].append(debug_entry)
    logger.debug(f"DEBUG: {message}")

def log_training_progress(epoch: int, step: int, total_steps: int, loss: float, 
                         learning_rate: float, metrics: Dict[str, float] = None,
                         phase: str = "training"):
    """Log comprehensive training progress information."""
    progress_percentage = (step / total_steps) * 100 if total_steps > 0 else 0
    
    # Calculate ETA
    elapsed_time = time.time() - monitoring_data.get('training_start_time', time.time())
    if step > 0:
        time_per_step = elapsed_time / step
        remaining_steps = total_steps - step
        eta_seconds = time_per_step * remaining_steps
        eta_str = f"{eta_seconds/3600:.1f}h {eta_seconds%3600/60:.0f}m"
    else:
        eta_str = "Unknown"
    
    # Log to training progress file
    training_logger.info(
        f"Epoch {epoch}, Step {step}/{total_steps} ({progress_percentage:.1f}%) - "
        f"Loss: {loss:.6f}, LR: {learning_rate:.2e}, ETA: {eta_str}, Phase: {phase}"
    )
    
    # Log metrics if provided
    if metrics:
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        training_logger.info(f"Metrics: {metrics_str}")
    
    # Update monitoring data
    monitoring_data['training_progress'] = {
        'epoch': epoch,
        'step': step,
        'total_steps': total_steps,
        'progress_percentage': progress_percentage,
        'loss': loss,
        'learning_rate': learning_rate,
        'eta': eta_str,
        'phase': phase,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }

def log_training_start(model_name: str, total_epochs: int, total_steps: int, 
                      batch_size: int, learning_rate: float, optimizer: str):
    """Log training session start information."""
    monitoring_data['training_start_time'] = time.time()
    monitoring_data['training_config'] = {
        'model_name': model_name,
        'total_epochs': total_epochs,
        'total_steps': total_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'optimizer': optimizer,
        'start_time': datetime.now().isoformat()
    }
    
    training_logger.info(
        f"🚀 Training started - Model: {model_name}, Epochs: {total_epochs}, "
        f"Steps: {total_steps}, Batch Size: {batch_size}, LR: {learning_rate}, "
        f"Optimizer: {optimizer}"
    )
    
    # Log system information
    system_stats = get_system_stats()
    training_logger.info(f"System stats at training start: {json.dumps(system_stats, indent=2)}")

def log_training_end(success: bool, final_loss: float = None, 
                    total_training_time: float = None, final_metrics: Dict[str, float] = None):
    """Log training session end information."""
    end_time = time.time()
    training_time = total_training_time or (end_time - monitoring_data.get('training_start_time', end_time))
    
    if success:
        training_logger.info(
            f"✅ Training completed successfully - Duration: {training_time/3600:.2f}h, "
            f"Final Loss: {final_loss:.6f if final_loss else 'N/A'}"
        )
    else:
        training_logger.error(
            f"❌ Training failed - Duration: {training_time/3600:.2f}h, "
            f"Final Loss: {final_loss:.6f if final_loss else 'N/A'}"
        )
    
    if final_metrics:
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in final_metrics.items()])
        training_logger.info(f"Final metrics: {metrics_str}")
    
    # Update monitoring data
    monitoring_data['training_end'] = {
        'success': success,
        'final_loss': final_loss,
        'total_training_time': training_time,
        'final_metrics': final_metrics or {},
        'end_time': datetime.now().isoformat()
    }

def log_model_checkpoint(epoch: int, step: int, loss: float, 
                        checkpoint_path: str, metrics: Dict[str, float] = None):
    """Log model checkpoint information."""
    training_logger.info(
        f"💾 Checkpoint saved - Epoch {epoch}, Step {step}, Loss: {loss:.6f}, "
        f"Path: {checkpoint_path}"
    )
    
    if metrics:
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        training_logger.info(f"Checkpoint metrics: {metrics_str}")
    
    # Update monitoring data
    if 'checkpoints' not in monitoring_data:
        monitoring_data['checkpoints'] = []
    
    monitoring_data['checkpoints'].append({
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'checkpoint_path': checkpoint_path,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    })

def log_validation_results(epoch: int, step: int, val_loss: float, 
                          val_metrics: Dict[str, float], is_best: bool = False):
    """Log validation results."""
    status = "🏆 BEST" if is_best else "📊"
    training_logger.info(
        f"{status} Validation - Epoch {epoch}, Step {step}, Loss: {val_loss:.6f}"
    )
    
    if val_metrics:
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in val_metrics.items()])
        training_logger.info(f"Validation metrics: {metrics_str}")
    
    # Update monitoring data
    if 'validation_results' not in monitoring_data:
        monitoring_data['validation_results'] = []
    
    monitoring_data['validation_results'].append({
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'is_best': is_best,
        'timestamp': datetime.now().isoformat()
    })

def log_error_with_context(error: Exception, context: str, additional_data: Dict[str, Any] = None):
    """Log errors with comprehensive context information."""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'timestamp': datetime.now().isoformat(),
        'traceback': traceback.format_exc(),
        'additional_data': additional_data or {}
    }
    
    # Log to error logger
    error_logger.error(
        f"Error in {context}: {type(error).__name__}: {str(error)}\n"
        f"Traceback: {traceback.format_exc()}"
    )
    
    # Log to main logger
    logger.error(f"Error in {context}: {str(error)}")
    
    # Update monitoring data
    if 'errors' not in monitoring_data:
        monitoring_data['errors'] = []
    monitoring_data['errors'].append(error_info)

def log_performance_metrics(operation: str, duration: float, 
                           memory_usage: Dict[str, float] = None,
                           throughput: float = None, batch_size: int = None):
    """Log performance metrics for operations."""
    performance_logger.info(
        f"Performance - {operation}: Duration: {duration:.3f}s"
    )
    
    if memory_usage:
        memory_str = ", ".join([f"{k}: {v:.2f}GB" for k, v in memory_usage.items()])
        performance_logger.info(f"Memory usage: {memory_str}")
    
    if throughput:
        performance_logger.info(f"Throughput: {throughput:.2f} samples/sec")
    
    if batch_size:
        performance_logger.info(f"Batch size: {batch_size}")
    
    # Update monitoring data
    if 'performance_metrics' not in monitoring_data:
        monitoring_data['performance_metrics'] = []
    
    monitoring_data['performance_metrics'].append({
        'operation': operation,
        'duration': duration,
        'memory_usage': memory_usage or {},
        'throughput': throughput,
        'batch_size': batch_size,
        'timestamp': datetime.now().isoformat()
    })

def log_model_operation(operation: str, model_name: str, 
                       success: bool, details: Dict[str, Any] = None):
    """Log model-related operations."""
    status = "✅" if success else "❌"
    model_logger.info(
        f"{status} {operation} - Model: {model_name}"
    )
    
    if details:
        details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
        model_logger.info(f"Details: {details_str}")
    
    # Update monitoring data
    if 'model_operations' not in monitoring_data:
        monitoring_data['model_operations'] = []
    
    monitoring_data['model_operations'].append({
        'operation': operation,
        'model_name': model_name,
        'success': success,
        'details': details or {},
        'timestamp': datetime.now().isoformat()
    })

def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Comprehensive prompt validation."""
    if not isinstance(prompt, str):
        return False, "Prompt must be a string."
    
    if not prompt.strip():
        return False, "Prompt cannot be empty."
    
    prompt = prompt.strip()
    
    # Length validation
    if len(prompt) < INPUT_PATTERNS['prompt']['min_length']:
        return False, f"Prompt too short. Minimum length: {INPUT_PATTERNS['prompt']['min_length']} characters."
    
    if len(prompt) > INPUT_PATTERNS['prompt']['max_length']:
        return False, f"Prompt too long. Maximum length: {INPUT_PATTERNS['prompt']['max_length']} characters."
    
    # Word count validation
    word_count = len(prompt.split())
    if word_count > INPUT_PATTERNS['prompt']['max_words']:
        return False, f"Prompt has too many words. Maximum: {INPUT_PATTERNS['prompt']['max_words']} words."
    
    # Security validation - check for potentially harmful content
    for forbidden in INPUT_PATTERNS['prompt']['forbidden_chars']:
        if forbidden.lower() in prompt.lower():
            return False, f"Prompt contains forbidden content: {forbidden}"
    
    # Check for excessive repetition
    words = prompt.split()
    if len(words) > 3:
        word_freq = {}
        for word in words:
            word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
        max_freq = max(word_freq.values())
        if max_freq > len(words) * 0.3:  # More than 30% repetition
            return False, "Prompt contains excessive word repetition."
    
    return True, ""

def validate_seed(seed: Union[int, float, str]) -> Tuple[bool, str]:
    """Validate random seed input."""
    if seed is None:
        return True, ""  # None is valid for random seed
    
    try:
        seed_int = int(seed)
    except (ValueError, TypeError):
        return False, "Seed must be a valid integer."
    
    if not (INPUT_PATTERNS['seed']['min_value'] <= seed_int <= INPUT_PATTERNS['seed']['max_value']):
        return False, f"Seed must be between {INPUT_PATTERNS['seed']['min_value']} and {INPUT_PATTERNS['seed']['max_value']}."
    
    return True, ""

def validate_num_images(num_images: Union[int, float]) -> Tuple[bool, str]:
    """Validate number of images input."""
    try:
        num_int = int(num_images)
    except (ValueError, TypeError):
        return False, "Number of images must be a valid integer."
    
    if not (INPUT_PATTERNS['num_images']['min_value'] <= num_int <= INPUT_PATTERNS['num_images']['max_value']):
        return False, f"Number of images must be between {INPUT_PATTERNS['num_images']['min_value']} and {INPUT_PATTERNS['num_images']['max_value']}."
    
    return True, ""

def validate_model_name(model_name: str) -> Tuple[bool, str]:
    """Validate model name input."""
    if not isinstance(model_name, str):
        return False, "Model name must be a string."
    
    if model_name not in PIPELINE_CONFIGS:
        available_models = list(PIPELINE_CONFIGS.keys())
        return False, f"Invalid model name. Available models: {', '.join(available_models)}"
    
    return True, ""

def comprehensive_input_validation(prompt: str, model_name: str, seed: Union[int, float, str], 
                                 num_images: Union[int, float], debug_mode: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    """Comprehensive input validation with detailed error reporting."""
    validation_results = {}
    errors = []
    
    # Validate prompt
    is_valid, error = validate_prompt(prompt)
    validation_results['prompt'] = {'valid': is_valid, 'error': error}
    if not is_valid:
        errors.append(f"Prompt: {error}")
    
    # Validate model name
    is_valid, error = validate_model_name(model_name)
    validation_results['model_name'] = {'valid': is_valid, 'error': error}
    if not is_valid:
        errors.append(f"Model: {error}")
    
    # Validate seed
    is_valid, error = validate_seed(seed)
    validation_results['seed'] = {'valid': is_valid, 'error': error}
    if not is_valid:
        errors.append(f"Seed: {error}")
    
    # Validate number of images
    is_valid, error = validate_num_images(num_images)
    validation_results['num_images'] = {'valid': is_valid, 'error': error}
    if not is_valid:
        errors.append(f"Number of images: {error}")
    
    # Check system resources
    try:
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 90:
            errors.append(f"High memory usage: {memory_usage}%. Consider reducing batch size.")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if gpu_memory > 0.9:
                errors.append(f"High GPU memory usage: {gpu_memory:.1%}. Consider clearing cache.")
    except Exception as e:
        if debug_mode:
            errors.append(f"System resource check failed: {e}")
    
    # Log validation results
    if debug_mode:
        log_debug_info("Input validation completed", validation_results)
    
    if errors:
        error_msg = "Validation errors:\n" + "\n".join(f"• {error}" for error in errors)
        monitoring_data['input_validation_failures'].append({
            'timestamp': datetime.now().isoformat(),
            'errors': errors,
            'inputs': {'prompt': prompt[:100], 'model_name': model_name, 'seed': seed, 'num_images': num_images}
        })
        return False, error_msg, validation_results
    
    return True, "", validation_results

def safe_model_loading(model_name: str, debug_mode: bool = False) -> Tuple[Any, str]:
    """Safely load model with comprehensive error handling."""
    start_time = time.time()
    
    try:
        log_debug_info(f"Attempting to load model: {model_name}")
        log_model_operation("model_loading_start", model_name, True, {"debug_mode": debug_mode})
        
        if model_name not in PIPELINE_CONFIGS:
            raise ModelLoadingError(f"Model '{model_name}' not found in available configurations.")
        
        # Log system stats before loading
        system_stats = get_system_stats()
        log_debug_info(f"System stats before model loading", system_stats)
        
        pipeline = get_cached_pipeline(model_name)
        
        if pipeline is None:
            raise ModelLoadingError(f"Failed to load pipeline for model '{model_name}'.")
        
        # Calculate loading time
        loading_time = time.time() - start_time
        
        # Log successful loading
        log_debug_info(f"Model loaded successfully: {model_name}")
        log_model_operation("model_loading_success", model_name, True, {
            "loading_time": loading_time,
            "system_stats": system_stats
        })
        
        # Log performance metrics
        log_performance_metrics("model_loading", loading_time, {
            "cpu_memory": system_stats.get('memory_percent', 0),
            "gpu_memory": system_stats.get('gpu_stats', {}).get('gpu_0', {}).get('memory_allocated_gb', 0) if system_stats.get('gpu_stats') else 0
        })
        
        return pipeline, ""
        
    except Exception as e:
        loading_time = time.time() - start_time
        
        # Log error with context
        log_error_with_context(e, f"model_loading_{model_name}", {
            "model_name": model_name,
            "loading_time": loading_time,
            "debug_mode": debug_mode,
            "available_models": list(PIPELINE_CONFIGS.keys())
        })
        
        error_msg = f"Model loading error: {str(e)}"
        if debug_mode:
            error_msg += f"\nTraceback: {traceback.format_exc()}"
        
        # Log failed operation
        log_model_operation("model_loading_failed", model_name, False, {
            "error": str(e),
            "loading_time": loading_time
        })
        
        monitoring_data['error_counts']['model_loading'] += 1
        return None, error_msg

def safe_inference(pipeline: Any, prompt: str, num_images: int, generator: Any, 
                  use_mixed_precision: bool, debug_mode: bool = False) -> Tuple[Any, str]:
    """Safely perform inference with comprehensive error handling and PyTorch debugging."""
    start_time = time.time()
    
    # Use PyTorch debugging context if debug mode is enabled
    debug_context = pytorch_debugger.context_manager("inference") if debug_mode else None
    
    try:
        if debug_context:
            debug_context.__enter__()
        
        log_debug_info(f"Starting inference: {num_images} images, mixed_precision={use_mixed_precision}")
        log_model_operation("inference_start", "pipeline", True, {
            "num_images": num_images,
            "use_mixed_precision": use_mixed_precision,
            "prompt_length": len(prompt)
        })
        
        # Enable PyTorch debugging tools if in debug mode
        if debug_mode:
            pytorch_debugger.enable_anomaly_detection(True)
            pytorch_debugger.enable_memory_tracking(True)
            
            # Start profiler for detailed performance analysis
            pytorch_debugger.start_profiler(
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                use_cuda=torch.cuda.is_available()
            )
        
        # Check memory before inference
        memory_before = {}
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated()
            memory_before['gpu_allocated'] = gpu_memory_before / 1024**3  # GB
            memory_before['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            
            if gpu_memory_before > torch.cuda.max_memory_allocated() * 0.8:
                logger.warning("High GPU memory usage before inference")
        
        memory_before['cpu_percent'] = psutil.virtual_memory().percent
        
        # Debug pipeline if in debug mode
        if debug_mode and hasattr(pipeline, 'unet'):
            pytorch_debugger.debug_model(pipeline.unet, log_details=True)
        
        # Perform inference with debugging
        inference_start = time.time()
        try:
            if use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = pipeline(prompt, num_images_per_prompt=num_images, generator=generator)
            else:
                output = pipeline(prompt, num_images_per_prompt=num_images, generator=generator)
        except Exception as inference_error:
            # Log detailed inference error with PyTorch debugging info
            if debug_mode:
                logger.error(f"Inference failed with error: {inference_error}")
                logger.error(f"PyTorch version: {torch.__version__}")
                logger.error(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.error(f"CUDA version: {torch.version.cuda}")
                    logger.error(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
            raise inference_error
        
        inference_time = time.time() - inference_start
        
        # Validate output
        if not hasattr(output, 'images') and 'images' not in output:
            raise InferenceError("Pipeline output does not contain images.")
        
        images = output.images if hasattr(output, 'images') else output["images"]
        if not images or len(images) == 0:
            raise InferenceError("No images were generated.")
        
        if len(images) != num_images:
            logger.warning(f"Expected {num_images} images, got {len(images)}")
        
        # Debug output tensors if in debug mode
        if debug_mode and hasattr(output, 'images'):
            for i, img in enumerate(output.images[:2]):  # Debug first 2 images
                if hasattr(img, 'shape'):
                    pytorch_debugger.debug_tensor(torch.tensor(img), f"output_image_{i}", log_details=True)
        
        # Calculate total time and throughput
        total_time = time.time() - start_time
        throughput = num_images / total_time if total_time > 0 else 0
        
        # Log successful inference
        log_debug_info(f"Inference completed successfully: {len(images)} images generated")
        log_model_operation("inference_success", "pipeline", True, {
            "images_generated": len(images),
            "inference_time": inference_time,
            "total_time": total_time,
            "throughput": throughput
        })
        
        # Log performance metrics
        log_performance_metrics("inference", total_time, memory_before, throughput, num_images)
        
        # Stop profiler and export results if in debug mode
        if debug_mode:
            pytorch_debugger.stop_profiler(export_path=f"logs/inference_profile_{int(time.time())}.json")
        
        return output, ""
        
    except torch.cuda.OutOfMemoryError as e:
        total_time = time.time() - start_time
        
        # Log memory error with context
        log_error_with_context(e, "inference_out_of_memory", {
            "num_images": num_images,
            "use_mixed_precision": use_mixed_precision,
            "memory_before": memory_before,
            "inference_time": total_time
        })
        
        error_msg = "GPU out of memory. Try reducing batch size or using CPU."
        if debug_mode:
            error_msg += f"\nError details: {str(e)}"
            # Get detailed memory info
            memory_stats = pytorch_debugger.get_memory_stats()
            error_msg += f"\nMemory stats: {memory_stats}"
        
        # Log failed operation
        log_model_operation("inference_failed", "pipeline", False, {
            "error_type": "out_of_memory",
            "error": str(e),
            "inference_time": total_time
        })
        
        monitoring_data['error_counts']['out_of_memory'] += 1
        return None, error_msg
        
    except Exception as e:
        total_time = time.time() - start_time
        
        # Log error with context
        log_error_with_context(e, "inference_general_error", {
            "num_images": num_images,
            "use_mixed_precision": use_mixed_precision,
            "memory_before": memory_before,
            "inference_time": total_time
        })
        
        error_msg = f"Inference error: {str(e)}"
        if debug_mode:
            error_msg += f"\nTraceback: {traceback.format_exc()}"
            # Get PyTorch debugging info
            error_msg += f"\nPyTorch version: {torch.__version__}"
            error_msg += f"\nCUDA available: {torch.cuda.is_available()}"
            if torch.cuda.is_available():
                error_msg += f"\nCUDA version: {torch.version.cuda}"
        
        # Log failed operation
        log_model_operation("inference_failed", "pipeline", False, {
            "error_type": "general_error",
            "error": str(e),
            "inference_time": total_time
        })
        
        monitoring_data['error_counts']['inference'] += 1
        return None, error_msg
        
    finally:
        # Clean up debugging tools
        if debug_mode:
            pytorch_debugger.enable_anomaly_detection(False)
            pytorch_debugger.enable_memory_tracking(False)
        
        if debug_context:
            debug_context.__exit__(None, None, None)

def get_detailed_error_info(error: Exception, debug_mode: bool = False) -> Dict[str, Any]:
    """Get detailed error information for debugging."""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
            'gpu_available': torch.cuda.is_available(),
        }
    }
    
    if torch.cuda.is_available():
        error_info['system_info'].update({
            'gpu_memory_allocated': torch.cuda.memory_allocated(),
            'gpu_memory_reserved': torch.cuda.memory_reserved(),
            'gpu_device_count': torch.cuda.device_count()
        })
    
    if debug_mode:
        error_info['traceback'] = traceback.format_exc()
        error_info['debug_logs'] = list(monitoring_data['debug_logs'])[-10:]
    
    return error_info

def optimize_pipeline_settings(pipeline, use_mixed_precision=False, use_multi_gpu=False) -> Any:
    """Optimize pipeline settings for better performance using the performance optimizer."""
    try:
        # Use the performance optimizer for comprehensive optimization
        optimization_config = {
            'memory_efficient_attention': True,
            'compile_models': True,
            'use_channels_last': True,
            'enable_xformers': True,
            'optimize_for_inference': True,
            'use_torch_compile': True,
            'enable_amp': use_mixed_precision,
            'use_fast_math': True
        }
        
        # Apply performance optimizations
        optimizations_applied = performance_optimizer.optimize_pipeline_performance(
            pipeline, optimization_config
        )
        
        # Optimize memory usage
        memory_optimizations = performance_optimizer.optimize_memory_usage(pipeline)
        
        # Apply legacy optimizations for compatibility
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
            logger.info("Enabled attention slicing for memory efficiency")
        
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
            logger.info("Enabled VAE slicing for memory efficiency")
        
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Failed to enable xformers: {e}")
        
        # Set optimal device placement
        if torch.cuda.is_available():
            pipeline = pipeline.to('cuda')
            if use_mixed_precision:
                pipeline = pipeline.half()
        
        # Log optimization results
        logger.info(f"Pipeline optimization completed:")
        logger.info(f"  Performance optimizations: {optimizations_applied}")
        logger.info(f"  Memory optimizations: {list(memory_optimizations.keys())}")
        
        return pipeline
        
    except Exception as e:
        logger.warning(f"Pipeline optimization failed: {e}")
        return pipeline

def preprocess_images_batch(images: List[Image.Image], target_size: Tuple[int, int] = (512, 512)) -> List[torch.Tensor]:
    """Efficiently preprocess a batch of images with comprehensive error handling."""
    processed_images = []
    
    if not images:
        logger.warning("No images provided for preprocessing")
        return processed_images
    
    def preprocess_single_image(img) -> Any:
        try:
            # Validate input image
            if img is None:
                logger.warning("Received None image in preprocessing")
                return None
            
            if not isinstance(img, Image.Image):
                logger.warning(f"Invalid image type: {type(img)}, expected PIL.Image")
                return None
            
            # Check image dimensions
            if img.size[0] == 0 or img.size[1] == 0:
                logger.warning("Image has zero dimensions")
                return None
            
            # Convert to RGB if needed
            try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            except Exception as e:
                logger.error(f"Failed to convert image to RGB: {e}")
                return None
            
            # Resize efficiently
            try:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            except Exception as e:
                logger.error(f"Failed to resize image: {e}")
                return None
            
            # Convert to tensor efficiently
            try:
            img_array = np.array(img).astype(np.float32) / 255.0
                
                # Validate array values
                if np.isnan(img_array).any() or np.isinf(img_array).any():
                    logger.warning("Image contains NaN or Inf values")
                    return None
                
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            
                # Validate tensor
                if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                    logger.warning("Tensor contains NaN or Inf values")
                    return None
                
            return img_tensor
                
        except Exception as e:
                logger.error(f"Failed to convert image to tensor: {e}")
            return None
    
        except Exception as e:
            logger.error(f"Unexpected error in image preprocessing: {e}")
            return None
    
    # Process images in parallel with error handling
    try:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(preprocess_single_image, img) for img in images]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
            if result is not None:
                processed_images.append(result)
                    else:
                        logger.warning(f"Failed to preprocess image {i}")
                except Exception as e:
                    logger.error(f"Error processing image {i}: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to create thread pool for image preprocessing: {e}")
        # Fallback to sequential processing
        for i, img in enumerate(images):
            try:
                result = preprocess_single_image(img)
                if result is not None:
                    processed_images.append(result)
            except Exception as e:
                logger.error(f"Sequential preprocessing failed for image {i}: {e}")
    
    logger.info(f"Successfully preprocessed {len(processed_images)}/{len(images)} images")
    return processed_images

def optimize_image_conversion(images: List) -> List[Image.Image]:
    """Optimize image conversion and handling with comprehensive error handling."""
    optimized_images = []
    
    if not images:
        logger.warning("No images provided for conversion")
        return optimized_images
    
    def convert_image(img) -> Any:
        try:
            # Validate input
            if img is None:
                logger.warning("Received None image in conversion")
                return None
            
            # Handle PIL Image
            if isinstance(img, Image.Image):
                try:
                    # Validate image
                    if img.size[0] == 0 or img.size[1] == 0:
                        logger.warning("PIL image has zero dimensions")
                        return None
                    
                    # Ensure RGB mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                return img
                except Exception as e:
                    logger.error(f"Failed to process PIL image: {e}")
                    return None
            
            # Handle numpy array
            elif isinstance(img, np.ndarray):
                try:
                    # Validate array
                    if img.size == 0:
                        logger.warning("Numpy array is empty")
                        return None
                    
                    if np.isnan(img).any() or np.isinf(img).any():
                        logger.warning("Numpy array contains NaN or Inf values")
                        return None
                    
                    # Handle different array shapes
                    if img.ndim == 2:
                        # Grayscale to RGB
                        img = np.stack([img] * 3, axis=-1)
                    elif img.ndim == 3 and img.shape[2] == 1:
                        # Single channel to RGB
                        img = np.concatenate([img] * 3, axis=2)
                    
                    # Normalize to uint8 if needed
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                return Image.fromarray(img)
                except Exception as e:
                    logger.error(f"Failed to convert numpy array to image: {e}")
                    return None
            
            # Handle PyTorch tensor
            elif isinstance(img, torch.Tensor):
                try:
                    # Move to CPU if needed
                    if img.device.type != 'cpu':
                        img = img.cpu()
                    
                # Handle tensor conversion efficiently
                if img.dim() == 4:
                    img = img.squeeze(0)
                    
                if img.dim() == 3 and img.shape[0] in [1, 3]:
                    img = img.permute(1, 2, 0)
                    
                    # Convert to numpy
                    img_array = img.numpy()
                    
                    # Validate array
                    if np.isnan(img_array).any() or np.isinf(img_array).any():
                        logger.warning("Tensor contains NaN or Inf values")
                        return None
                    
                    # Normalize to uint8
                if img_array.dtype != np.uint8:
                        if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                    
                return Image.fromarray(img_array)
                except Exception as e:
                    logger.error(f"Failed to convert tensor to image: {e}")
                    return None
            
            else:
                logger.warning(f"Unknown image type: {type(img)}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in image conversion: {e}")
            return None
    
    # Process conversions in parallel with error handling
    try:
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(convert_image, img) for img in images]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
            if result is not None:
                optimized_images.append(result)
                    else:
                        logger.warning(f"Failed to convert image {i}")
                except Exception as e:
                    logger.error(f"Error converting image {i}: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to create thread pool for image conversion: {e}")
        # Fallback to sequential processing
        for i, img in enumerate(images):
            try:
                result = convert_image(img)
                if result is not None:
                    optimized_images.append(result)
            except Exception as e:
                logger.error(f"Sequential conversion failed for image {i}: {e}")
    
    logger.info(f"Successfully converted {len(optimized_images)}/{len(images)} images")
    return optimized_images

def get_system_stats():
    """Get comprehensive system statistics with error handling."""
    stats = {
        'timestamp': time.time(),
        'status': 'success'
    }
    
    # CPU statistics
    try:
        stats['cpu_percent'] = psutil.cpu_percent(interval=1)
    except Exception as e:
        logger.error(f"Failed to get CPU stats: {e}")
        stats['cpu_percent'] = None
        stats['status'] = 'partial'
    
    # Memory statistics
    try:
        memory = psutil.virtual_memory()
        stats['memory_percent'] = memory.percent
        stats['memory_available_gb'] = round(memory.available / 1024**3, 2)
        stats['memory_total_gb'] = round(memory.total / 1024**3, 2)
        stats['memory_used_gb'] = round(memory.used / 1024**3, 2)
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        stats['memory_percent'] = None
        stats['memory_available_gb'] = None
        stats['memory_total_gb'] = None
        stats['memory_used_gb'] = None
        stats['status'] = 'partial'
    
    # Disk statistics
    try:
        disk = psutil.disk_usage('/')
        stats['disk_usage_percent'] = disk.percent
        stats['disk_free_gb'] = round(disk.free / 1024**3, 2)
        stats['disk_total_gb'] = round(disk.total / 1024**3, 2)
    except Exception as e:
        logger.error(f"Failed to get disk stats: {e}")
        stats['disk_usage_percent'] = None
        stats['disk_free_gb'] = None
        stats['disk_total_gb'] = None
        stats['status'] = 'partial'
    
    # Network statistics (if available)
    try:
        network = psutil.net_io_counters()
        stats['network_bytes_sent'] = network.bytes_sent
        stats['network_bytes_recv'] = network.bytes_recv
    except Exception as e:
        logger.warning(f"Failed to get network stats: {e}")
        stats['network_bytes_sent'] = None
        stats['network_bytes_recv'] = None
    
    # GPU statistics
    try:
    if torch.cuda.is_available():
        stats['gpu_count'] = torch.cuda.device_count()
        stats['gpu_stats'] = {}
            
        for i in range(torch.cuda.device_count()):
            try:
                    device_props = torch.cuda.get_device_properties(i)
                stats['gpu_stats'][f'gpu_{i}'] = {
                        'name': device_props.name,
                    'memory_allocated_gb': round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                    'memory_reserved_gb': round(torch.cuda.memory_reserved(i) / 1024**3, 2),
                        'memory_total_gb': round(device_props.total_memory / 1024**3, 2),
                        'memory_free_gb': round((device_props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3, 2),
                        'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'utilization_percent': 0  # Would need nvidia-ml-py for actual utilization
                }
            except Exception as e:
                    logger.error(f"Failed to get GPU {i} stats: {e}")
                    stats['gpu_stats'][f'gpu_{i}'] = {'error': str(e)}
        else:
            stats['gpu_count'] = 0
            stats['gpu_stats'] = {}
            
    except Exception as e:
        logger.error(f"Failed to get GPU stats: {e}")
        stats['gpu_count'] = None
        stats['gpu_stats'] = {}
        stats['status'] = 'partial'
    
    # Process-specific statistics
    try:
        process = psutil.Process()
        stats['process_memory_gb'] = round(process.memory_info().rss / 1024**3, 2)
        stats['process_cpu_percent'] = process.cpu_percent()
        stats['process_threads'] = process.num_threads()
    except Exception as e:
        logger.warning(f"Failed to get process stats: {e}")
        stats['process_memory_gb'] = None
        stats['process_cpu_percent'] = None
        stats['process_threads'] = None
    
    return stats

def update_monitoring_data(operation_type, metrics, success=True, error=None) -> Any:
    """Update global monitoring data."""
    timestamp = time.time()
    
    # Update inference history
    monitoring_data['inference_history'].append({
        'timestamp': timestamp,
        'operation_type': operation_type,
        'success': success,
        'duration_seconds': metrics.get('inference_time_seconds', 0),
        'images_generated': metrics.get('images_generated', 0),
        'model_used': metrics.get('model_used', 'unknown'),
        'gpu_count': metrics.get('gpu_count', 0),
        'mixed_precision': metrics.get('mixed_precision', False),
        'multi_gpu_enabled': metrics.get('multi_gpu_enabled', False),
        'gradient_accumulation_steps': metrics.get('gradient_accumulation_steps', 1)
    })
    
    # Update memory history
    system_stats = get_system_stats()
    monitoring_data['memory_history'].append({
        'timestamp': timestamp,
        'cpu_percent': system_stats['cpu_percent'],
        'memory_percent': system_stats['memory_percent'],
        'gpu_stats': system_stats.get('gpu_stats', {})
    })
    
    # Update error history if there was an error
    if error:
        monitoring_data['error_history'].append({
            'timestamp': timestamp,
            'operation_type': operation_type,
            'error': str(error),
            'model_used': metrics.get('model_used', 'unknown')
        })
    
    # Update performance metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            monitoring_data['performance_metrics'][key].append(value)
            if len(monitoring_data['performance_metrics'][key]) > 100:
                monitoring_data['performance_metrics'][key].pop(0)
    
    # Update system stats
    monitoring_data['system_stats'] = system_stats

def get_model_health_status():
    """Get model health status."""
    health = {
        'pipelines_loaded': len(manager.pipelines),
        'available_models': list(PIPELINE_CONFIGS.keys()),
        'last_error': None,
        'error_count': len(monitoring_data['error_history']),
        'success_rate': 0,
        'average_inference_time': 0,
        'memory_usage': {}
    }
    
    if monitoring_data['inference_history']:
        recent_inferences = list(monitoring_data['inference_history'])[-20:]
        success_count = sum(1 for inf in recent_inferences if inf['success'])
        health['success_rate'] = success_count / len(recent_inferences) if recent_inferences else 0
        
        inference_times = [inf['duration_seconds'] for inf in recent_inferences if inf['success']]
        health['average_inference_time'] = sum(inference_times) / len(inference_times) if inference_times else 0
    
    if torch.cuda.is_available():
        health['memory_usage'] = {
            f'gpu_{i}_allocated_gb': round(torch.cuda.memory_allocated(i) / 1024**3, 2)
            for i in range(torch.cuda.device_count())
        }
    
    if monitoring_data['error_history']:
        health['last_error'] = monitoring_data['error_history'][-1]
    
    return health

def get_performance_summary():
    """Get performance summary statistics."""
    if not monitoring_data['performance_metrics']:
        return {}
    
    summary = {}
    for metric, values in monitoring_data['performance_metrics'].items():
        if values:
            summary[f'{metric}_mean'] = round(sum(values) / len(values), 2)
            summary[f'{metric}_min'] = round(min(values), 2)
            summary[f'{metric}_max'] = round(max(values), 2)
            summary[f'{metric}_count'] = len(values)
    
    return summary

@lru_cache(maxsize=100)
def get_cached_pipeline(model_name) -> Optional[Dict[str, Any]]:
    return manager.get_pipeline(model_name)

def get_available_gpus():
    
    """get_available_gpus function."""
if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []

# Multi-GPU Training System
class MultiGPUTrainer:
    """Comprehensive multi-GPU training utilities for DataParallel and DistributedDataParallel."""
    
    def __init__(self) -> Any:
        self.ddp_initialized = False
        self.dp_initialized = False
        self.current_strategy = None
        self.gpu_config = {}
        self.training_metrics = defaultdict(list)
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        try:
            gpu_info = {
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': 0,
                'gpu_details': [],
                'total_memory_gb': 0,
                'compute_capability': [],
                'status': 'success'
            }
            
            if not torch.cuda.is_available():
                return gpu_info
            
            gpu_count = torch.cuda.device_count()
            gpu_info['gpu_count'] = gpu_count
            
            total_memory = 0
            for i in range(gpu_count):
                try:
                    device_props = torch.cuda.get_device_properties(i)
                    memory_gb = device_props.total_memory / 1024**3
                    total_memory += memory_gb
                    
                    gpu_detail = {
                        'id': i,
                        'name': device_props.name,
                        'memory_gb': round(memory_gb, 2),
                        'compute_capability': f"{device_props.major}.{device_props.minor}",
                        'multi_processor_count': device_props.multi_processor_count,
                        'max_threads_per_block': device_props.max_threads_per_block,
                        'max_shared_memory_per_block': device_props.max_shared_memory_per_block,
                        'is_integrated': device_props.is_integrated,
                        'is_multi_gpu_board': device_props.is_multi_gpu_board,
                        'status': 'healthy'
                    }
                    
                    gpu_info['gpu_details'].append(gpu_detail)
                    gpu_info['compute_capability'].append(f"{device_props.major}.{device_props.minor}")
                    
                except Exception as e:
                    logger.error(f"Failed to get GPU {i} info: {e}")
                    gpu_info['gpu_details'].append({
                        'id': i,
                        'status': 'error',
                        'error': str(e)
                    })
            
            gpu_info['total_memory_gb'] = round(total_memory, 2)
            return gpu_info
            
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def setup_data_parallel(self, model: torch.nn.Module, device_ids: List[int] = None) -> Tuple[torch.nn.Module, bool]:
        """Setup DataParallel for multi-GPU training."""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping DataParallel setup")
                return model, False
            
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                logger.warning(f"Only {gpu_count} GPU available, skipping DataParallel setup")
                return model, False
            
            # Determine device IDs
            if device_ids is None:
                device_ids = list(range(gpu_count))
            else:
                device_ids = [i for i in device_ids if i < gpu_count]
            
            if len(device_ids) < 2:
                logger.warning(f"Only {len(device_ids)} valid GPU IDs provided, skipping DataParallel setup")
                return model, False
            
            # Move model to first GPU
            model = model.to(f'cuda:{device_ids[0]}')
            
            # Setup DataParallel
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            
            self.dp_initialized = True
            self.current_strategy = 'DataParallel'
            self.gpu_config = {
                'strategy': 'DataParallel',
                'device_ids': device_ids,
                'gpu_count': len(device_ids),
                'setup_time': datetime.now().isoformat()
            }
            
            logger.info(f"✅ DataParallel setup completed with {len(device_ids)} GPUs: {device_ids}")
            return model, True
            
        except Exception as e:
            logger.error(f"Failed to setup DataParallel: {e}")
            return model, False
    
    def setup_distributed_data_parallel(self, model: torch.nn.Module, 
                                      backend: str = 'nccl',
                                      init_method: str = 'env://',
                                      world_size: int = None,
                                      rank: int = None,
                                      device_ids: List[int] = None) -> Tuple[torch.nn.Module, bool]:
        """Setup DistributedDataParallel for multi-GPU training."""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, skipping DistributedDataParallel setup")
                return model, False
            
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                logger.warning(f"Only {gpu_count} GPU available, skipping DistributedDataParallel setup")
                return model, False
            
            
            # Initialize distributed process group if not already initialized
            if not dist.is_initialized():
                # Set environment variables if not provided
                if world_size is None:
                    world_size = gpu_count
                if rank is None:
                    rank = 0
                
                # Set environment variables for distributed training
                os.environ['WORLD_SIZE'] = str(world_size)
                os.environ['RANK'] = str(rank)
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                
                # Initialize process group
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank
                )
                
                self.ddp_initialized = True
                logger.info(f"✅ Distributed process group initialized: backend={backend}, world_size={world_size}, rank={rank}")
            
            # Determine device IDs
            if device_ids is None:
                device_ids = list(range(gpu_count))
            else:
                device_ids = [i for i in device_ids if i < gpu_count]
            
            if len(device_ids) < 2:
                logger.warning(f"Only {len(device_ids)} valid GPU IDs provided, skipping DistributedDataParallel setup")
                return model, False
            
            # Move model to first GPU
            model = model.to(f'cuda:{device_ids[0]}')
            
            # Setup DistributedDataParallel
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=device_ids,
                output_device=device_ids[0],
                find_unused_parameters=False,
                broadcast_buffers=True
            )
            
            self.current_strategy = 'DistributedDataParallel'
            self.gpu_config = {
                'strategy': 'DistributedDataParallel',
                'backend': backend,
                'world_size': world_size,
                'rank': rank,
                'device_ids': device_ids,
                'gpu_count': len(device_ids),
                'setup_time': datetime.now().isoformat()
            }
            
            logger.info(f"✅ DistributedDataParallel setup completed with {len(device_ids)} GPUs: {device_ids}")
            return model, True
            
        except Exception as e:
            logger.error(f"Failed to setup DistributedDataParallel: {e}")
            return model, False
    
    def setup_multi_gpu_training(self, model: torch.nn.Module, 
                               strategy: str = 'auto',
                               device_ids: List[int] = None,
                               ddp_backend: str = 'nccl',
                               ddp_init_method: str = 'env://') -> Tuple[torch.nn.Module, bool, Dict[str, Any]]:
        """Setup multi-GPU training with automatic strategy selection."""
        try:
            gpu_info = self.get_gpu_info()
            
            if not gpu_info['cuda_available'] or gpu_info['gpu_count'] < 2:
                logger.warning("Multi-GPU training not available")
                return model, False, gpu_info
            
            # Auto-select strategy if not specified
            if strategy == 'auto':
                if gpu_info['gpu_count'] <= 4:
                    strategy = 'DataParallel'  # Better for small number of GPUs
    else:
                    strategy = 'DistributedDataParallel'  # Better for large number of GPUs
            
            setup_success = False
            
            if strategy.lower() == 'dataparallel':
                model, setup_success = self.setup_data_parallel(model, device_ids)
            elif strategy.lower() in ['distributeddataparallel', 'ddp']:
                model, setup_success = self.setup_distributed_data_parallel(
                    model, ddp_backend, ddp_init_method, device_ids=device_ids
                )
            else:
                logger.error(f"Unknown multi-GPU strategy: {strategy}")
                return model, False, gpu_info
            
            if setup_success:
                logger.info(f"✅ Multi-GPU training setup completed: {strategy}")
                return model, True, gpu_info
            else:
                logger.warning(f"Failed to setup {strategy}, falling back to single GPU")
                return model, False, gpu_info
                
        except Exception as e:
            logger.error(f"Failed to setup multi-GPU training: {e}")
            return model, False, gpu_info
    
    def optimize_batch_size_for_multi_gpu(self, base_batch_size: int, gpu_count: int, strategy: str) -> Dict[str, Any]:
        """Optimize batch size for multi-GPU training."""
        try:
            if strategy == 'DataParallel':
                # DataParallel automatically distributes batch across GPUs
                effective_batch_size = base_batch_size * gpu_count
                batch_per_gpu = base_batch_size
            elif strategy == 'DistributedDataParallel':
                # DDP requires manual batch distribution
                effective_batch_size = base_batch_size * gpu_count
                batch_per_gpu = base_batch_size
            else:
                effective_batch_size = base_batch_size
                batch_per_gpu = base_batch_size
            
            optimization_config = {
                'base_batch_size': base_batch_size,
                'effective_batch_size': effective_batch_size,
                'batch_per_gpu': batch_per_gpu,
                'gpu_count': gpu_count,
                'strategy': strategy,
                'scaling_factor': gpu_count
            }
            
            logger.info(f"✅ Batch size optimization: {optimization_config}")
            return optimization_config
            
        except Exception as e:
            logger.error(f"Failed to optimize batch size: {e}")
            return {
                'base_batch_size': base_batch_size,
                'effective_batch_size': base_batch_size,
                'batch_per_gpu': base_batch_size,
                'gpu_count': 1,
                'strategy': 'single_gpu',
                'scaling_factor': 1
            }
    
    def get_multi_gpu_metrics(self) -> Dict[str, Any]:
        """Get multi-GPU training metrics."""
        try:
            metrics = {
                'strategy': self.current_strategy,
                'gpu_config': self.gpu_config,
                'ddp_initialized': self.ddp_initialized,
                'dp_initialized': self.dp_initialized,
                'gpu_info': self.get_gpu_info(),
                'training_metrics': dict(self.training_metrics)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get multi-GPU metrics: {e}")
            return {'error': str(e)}
    
    def cleanup_distributed(self) -> Any:
        """Cleanup distributed training resources."""
        try:
            if self.ddp_initialized:
                if dist.is_initialized():
                    dist.destroy_process_group()
                    logger.info("✅ Distributed process group destroyed")
                
                self.ddp_initialized = False
                self.current_strategy = None
                self.gpu_config = {}
                
        except Exception as e:
            logger.error(f"Failed to cleanup distributed resources: {e}")
    
    def log_training_metrics(self, epoch: int, step: int, loss: float, 
                           learning_rate: float, gpu_utilization: Dict[str, Any] = None):
        """Log training metrics for multi-GPU training."""
        try:
            metric_entry = {
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'learning_rate': learning_rate,
                'gpu_utilization': gpu_utilization,
                'strategy': self.current_strategy,
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_metrics[f'epoch_{epoch}'].append(metric_entry)
            
            logger.info(f"📊 Multi-GPU Training - Epoch {epoch}, Step {step}: Loss={loss:.4f}, LR={learning_rate:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")

# Global multi-GPU trainer instance
multi_gpu_trainer = MultiGPUTrainer()

def setup_multi_gpu_pipeline(pipeline, use_ddp=False) -> Any:
    """Enhanced multi-GPU pipeline setup with comprehensive error handling."""
    try:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping multi-GPU setup")
            return pipeline, False
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            logger.warning(f"Only {gpu_count} GPU available, skipping multi-GPU setup")
            return pipeline, False
        
        # Use the multi-GPU trainer for setup
        strategy = 'DistributedDataParallel' if use_ddp else 'DataParallel'
        pipeline, success, gpu_info = multi_gpu_trainer.setup_multi_gpu_training(
            pipeline, strategy=strategy
        )
        
        if success:
            logger.info(f"✅ Multi-GPU setup completed: {strategy}")
            return pipeline, True
        else:
            logger.warning(f"Failed to setup {strategy}, using single GPU")
            return pipeline, False
            
        except Exception as e:
        logger.error(f"Failed to setup multi-GPU pipeline: {e}")
            return pipeline, False

def clear_gpu_memory():
    """Clear GPU memory across all devices."""
    if torch.cuda.is_available():
        # Clear memory on all GPUs
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ GPU memory cleared across all devices")

def train_with_multi_gpu(model: torch.nn.Module, 
                        train_loader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer,
                        criterion: torch.nn.Module,
                        num_epochs: int = 10,
                        strategy: str = 'auto',
                        device_ids: List[int] = None,
                        use_mixed_precision: bool = True,
                        gradient_accumulation_steps: int = 1) -> Dict[str, Any]:
    """Train model using multi-GPU with comprehensive monitoring."""
    try:
        start_time = time.time()
        training_results = {
            'success': False,
            'epochs_completed': 0,
            'final_loss': None,
            'training_time': 0,
            'gpu_utilization': {},
            'multi_gpu_metrics': {},
            'error': None
        }
        
        # Setup multi-GPU training
        model, multi_gpu_success, gpu_info = multi_gpu_trainer.setup_multi_gpu_training(
            model, strategy=strategy, device_ids=device_ids
        )
        
        if not multi_gpu_success:
            logger.warning("Multi-GPU setup failed, falling back to single GPU")
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get GPU information
        training_results['gpu_info'] = gpu_info
        training_results['multi_gpu_metrics'] = multi_gpu_trainer.get_multi_gpu_metrics()
        
        # Setup mixed precision
        scaler = None
        if use_mixed_precision and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    # Move data to device
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()
                    
                    # Forward pass with mixed precision
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    num_batches += 1
                    
                    # Log training metrics
                    if batch_idx % 10 == 0:  # Log every 10 batches
                        current_loss = loss.item() * gradient_accumulation_steps
                        learning_rate = optimizer.param_groups[0]['lr']
                        
                        # Get GPU utilization
                        gpu_utilization = get_gpu_utilization()
                        
                        # Log multi-GPU metrics
                        multi_gpu_trainer.log_training_metrics(
                            epoch, batch_idx, current_loss, learning_rate, gpu_utilization
                        )
                        
                        # Log training progress
                        log_training_progress(
                            epoch, batch_idx, len(train_loader), current_loss, learning_rate
                        )
                
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch completion
            log_training_progress(
                epoch, len(train_loader), len(train_loader), avg_epoch_loss, 
                optimizer.param_groups[0]['lr'], phase="epoch_complete"
            )
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed - Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
            
            training_results['epochs_completed'] = epoch + 1
            training_results['final_loss'] = avg_epoch_loss
        
        # Training completed
        total_training_time = time.time() - start_time
        training_results['training_time'] = total_training_time
        training_results['success'] = True
        
        # Get final GPU utilization
        training_results['gpu_utilization'] = get_gpu_utilization()
        
        # Log training completion
        log_training_end(
            success=True,
            final_loss=training_results['final_loss'],
            total_training_time=total_training_time
        )
        
        logger.info(f"✅ Multi-GPU training completed successfully in {total_training_time:.2f}s")
        return training_results
        
    except Exception as e:
        error_msg = f"Multi-GPU training failed: {e}"
        logger.error(error_msg)
        training_results['error'] = error_msg
        training_results['success'] = False
        
        # Log training failure
        log_training_end(success=False, total_training_time=time.time() - start_time)
        
        return training_results
    
    finally:
        # Cleanup distributed resources
        multi_gpu_trainer.cleanup_distributed()
        clear_gpu_memory()

def evaluate_with_multi_gpu(model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          criterion: torch.nn.Module,
                          strategy: str = 'auto',
                          device_ids: List[int] = None) -> Dict[str, Any]:
    """Evaluate model using multi-GPU with comprehensive monitoring."""
    try:
        evaluation_results = {
            'success': False,
            'test_loss': None,
            'accuracy': None,
            'gpu_utilization': {},
            'multi_gpu_metrics': {},
            'error': None
        }
        
        # Setup multi-GPU evaluation
        model, multi_gpu_success, gpu_info = multi_gpu_trainer.setup_multi_gpu_training(
            model, strategy=strategy, device_ids=device_ids
        )
        
        if not multi_gpu_success:
            logger.warning("Multi-GPU setup failed, falling back to single GPU")
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        evaluation_results['gpu_info'] = gpu_info
        evaluation_results['multi_gpu_metrics'] = multi_gpu_trainer.get_multi_gpu_metrics()
        
        # Evaluation loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                try:
                    # Move data to device
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()
                    
                    # Forward pass
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Calculate metrics
                    test_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        logger.info(f"Evaluation batch {batch_idx}/{len(test_loader)}")
                
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
        
        # Calculate final metrics
        avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        
        evaluation_results['test_loss'] = avg_test_loss
        evaluation_results['accuracy'] = accuracy
        evaluation_results['success'] = True
        evaluation_results['gpu_utilization'] = get_gpu_utilization()
        
        logger.info(f"✅ Multi-GPU evaluation completed - Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return evaluation_results
        
    except Exception as e:
        error_msg = f"Multi-GPU evaluation failed: {e}"
        logger.error(error_msg)
        evaluation_results['error'] = error_msg
        evaluation_results['success'] = False
        return evaluation_results
    
    finally:
        # Cleanup distributed resources
        multi_gpu_trainer.cleanup_distributed()
        clear_gpu_memory()

def get_multi_gpu_status() -> Dict[str, Any]:
    """Get comprehensive multi-GPU status and metrics."""
    try:
        status = {
            'multi_gpu_available': False,
            'gpu_info': {},
            'current_strategy': None,
            'training_metrics': {},
            'performance_summary': {},
            'status': 'success'
        }
        
        # Get GPU information
        gpu_info = multi_gpu_trainer.get_gpu_info()
        status['gpu_info'] = gpu_info
        
        # Check multi-GPU availability
        if gpu_info['cuda_available'] and gpu_info['gpu_count'] >= 2:
            status['multi_gpu_available'] = True
        
        # Get current strategy
        status['current_strategy'] = multi_gpu_trainer.current_strategy
        
        # Get training metrics
        status['training_metrics'] = multi_gpu_trainer.get_multi_gpu_metrics()
        
        # Get performance summary
        status['performance_summary'] = performance_optimizer.get_performance_summary()
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get multi-GPU status: {e}")
        return {'status': 'error', 'error': str(e)}

def get_gpu_utilization():
    """Get GPU utilization statistics with comprehensive error handling."""
    gpu_stats = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'total_memory_allocated_gb': 0,
        'total_memory_reserved_gb': 0,
        'status': 'success'
    }
    
    if not torch.cuda.is_available():
        return gpu_stats
    
    try:
        gpu_count = torch.cuda.device_count()
        gpu_stats['gpu_count'] = gpu_count
        
        if gpu_count == 0:
            return gpu_stats
        
        # Individual GPU stats
        gpu_stats['individual_gpus'] = {}
        total_allocated = 0
        total_reserved = 0
        
        for i in range(gpu_count):
            try:
                # Get device properties
                device_props = torch.cuda.get_device_properties(i)
                
                # Get memory stats
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = device_props.total_memory
                
                # Calculate percentages
                memory_allocated_gb = round(memory_allocated / 1024**3, 2)
                memory_reserved_gb = round(memory_reserved / 1024**3, 2)
                memory_total_gb = round(memory_total / 1024**3, 2)
                memory_free_gb = round((memory_total - memory_reserved) / 1024**3, 2)
                
                # Calculate utilization percentages
                allocated_percent = round((memory_allocated / memory_total) * 100, 1)
                reserved_percent = round((memory_reserved / memory_total) * 100, 1)
                
                gpu_stats['individual_gpus'][f'gpu_{i}'] = {
                    'name': device_props.name,
                    'memory_allocated_gb': memory_allocated_gb,
                    'memory_reserved_gb': memory_reserved_gb,
                    'memory_total_gb': memory_total_gb,
                    'memory_free_gb': memory_free_gb,
                    'memory_allocated_percent': allocated_percent,
                    'memory_reserved_percent': reserved_percent,
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'multi_processor_count': device_props.multi_processor_count,
                    'max_threads_per_block': device_props.max_threads_per_block,
                    'max_shared_memory_per_block': device_props.max_shared_memory_per_block,
                    'status': 'healthy'
                }
                
                total_allocated += memory_allocated_gb
                total_reserved += memory_reserved_gb
                
        except Exception as e:
                logger.error(f"Failed to get GPU {i} utilization: {e}")
                gpu_stats['individual_gpus'][f'gpu_{i}'] = {
                    'error': str(e),
                    'status': 'error'
                }
                gpu_stats['status'] = 'partial'
        
        # Aggregate stats
        gpu_stats['total_memory_allocated_gb'] = round(total_allocated, 2)
        gpu_stats['total_memory_reserved_gb'] = round(total_reserved, 2)
        
        # Check for memory warnings
        if total_reserved > 0:
            healthy_gpus = sum(1 for gpu in gpu_stats['individual_gpus'].values() 
                             if gpu.get('status') == 'healthy')
            if healthy_gpus > 0:
                avg_reserved_percent = total_reserved / (healthy_gpus * 8)  # Assuming 8GB per GPU
                if avg_reserved_percent > 90:
                    gpu_stats['memory_warning'] = 'High GPU memory usage detected'
                elif avg_reserved_percent > 80:
                    gpu_stats['memory_warning'] = 'Moderate GPU memory usage detected'
        
    except Exception as e:
        logger.error(f"Failed to get GPU utilization: {e}")
        gpu_stats['status'] = 'error'
        gpu_stats['error'] = str(e)
    
    return gpu_stats

def validate_inputs(prompt, model_name, seed, num_images) -> bool:
    """Legacy validation function - kept for compatibility."""
    is_valid, error_msg, _ = comprehensive_input_validation(prompt, model_name, seed, num_images, debug_mode=False)
    return is_valid, error_msg if not is_valid else None

def generate_with_gradient_accumulation(pipeline, prompt, num_images, generator, accumulation_steps=1, use_mixed_precision=False) -> Any:
    """Generate images with gradient accumulation for large effective batch sizes with comprehensive error handling."""
    
    # Validate inputs
    try:
        if accumulation_steps <= 0:
            raise ValueError("Accumulation steps must be positive")
        if num_images <= 0:
            raise ValueError("Number of images must be positive")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
    except Exception as e:
        logger.error(f"Input validation failed for gradient accumulation: {e}")
        raise
    
    if accumulation_steps <= 1:
        # No accumulation needed - use safe inference
        try:
            output, error = safe_inference(pipeline, prompt, num_images, generator, use_mixed_precision, debug_mode=False)
            if output is None:
                raise InferenceError(error)
            return output
        except Exception as e:
            logger.error(f"Single-step inference failed: {e}")
            raise
    
    # Split the generation into accumulation steps
    try:
    images_per_step = max(1, num_images // accumulation_steps)
    all_images = []
    
    logger.info(f"Using gradient accumulation: {accumulation_steps} steps, {images_per_step} images per step")
    
    for step in range(accumulation_steps):
        step_start_time = time.time()
        
            try:
        # Calculate images for this step
        if step == accumulation_steps - 1:
            # Last step gets remaining images
            step_images = num_images - (accumulation_steps - 1) * images_per_step
        else:
            step_images = images_per_step
        
        if step_images <= 0:
                    logger.warning(f"Step {step + 1}: No images to generate, skipping")
            break
            
                # Check memory before generation
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated()
                    if gpu_memory_before > torch.cuda.max_memory_allocated() * 0.9:
                        logger.warning(f"High GPU memory usage before step {step + 1}: {gpu_memory_before / 1024**3:.2f}GB")
                
                # Generate images for this step
        try:
            if use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = pipeline(prompt, num_images_per_prompt=step_images, generator=generator)
            else:
                output = pipeline(prompt, num_images_per_prompt=step_images, generator=generator)
            
                    # Validate output
                    if not hasattr(output, 'images') and 'images' not in output:
                        raise InferenceError("Pipeline output does not contain images")
                    
            step_images_list = output.images if hasattr(output, 'images') else output["images"]
                    
                    if not step_images_list or len(step_images_list) == 0:
                        raise InferenceError(f"No images generated in step {step + 1}")
                    
                    # Optimize image conversion with error handling
                    try:
            optimized_images = optimize_image_conversion(step_images_list)
                        if not optimized_images:
                            raise InferenceError(f"Failed to convert images in step {step + 1}")
                        
            all_images.extend(optimized_images)
                        
                    except Exception as e:
                        logger.error(f"Image conversion failed in step {step + 1}: {e}")
                        raise
            
            step_time = time.time() - step_start_time
                    logger.info(f"Accumulation step {step + 1}/{accumulation_steps}: {len(optimized_images)} images in {step_time:.2f}s")
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU out of memory in step {step + 1}: {e}")
                    # Try to clear memory and continue with fewer images
                    clear_gpu_memory()
                    if step_images > 1:
                        logger.info(f"Retrying step {step + 1} with fewer images")
                        step_images = max(1, step_images // 2)
                        continue
                    else:
                        raise
                        
                except Exception as e:
                    logger.error(f"Generation failed in step {step + 1}: {e}")
                    raise
            
            # Clear memory between steps
                try:
            clear_gpu_memory()
                except Exception as e:
                    logger.warning(f"Failed to clear GPU memory after step {step + 1}: {e}")
            
        except Exception as e:
            logger.error(f"Error in accumulation step {step + 1}: {e}")
                # Continue with next step if possible
                if step == 0:
                    # If first step fails, raise the error
            raise
                else:
                    logger.warning(f"Continuing with {len(all_images)} images generated so far")
                    break
        
        # Validate final results
        if not all_images:
            raise InferenceError("No images were generated in any accumulation step")
        
        logger.info(f"Gradient accumulation completed: {len(all_images)} total images generated")
    
    # Return combined results
    return type('Output', (), {'images': all_images})()
        
    except Exception as e:
        logger.error(f"Gradient accumulation failed: {e}")
        # Clean up any partial results
        if 'all_images' in locals():
            del all_images
        clear_gpu_memory()
        raise

def generate(prompt, model_name, seed, num_images, debug_mode, use_mixed_precision, use_multi_gpu, use_ddp, gradient_accumulation_steps) -> Any:
    """Enhanced generate function with comprehensive error handling and debugging."""
    start_time = time.time()
    
    try:
        # Comprehensive input validation
        is_valid, error_msg, validation_results = comprehensive_input_validation(
            prompt, model_name, seed, num_images, debug_mode
        )
        
        if not is_valid:
            logger.warning(f"Input validation failed: {error_msg}")
            update_monitoring_data('generation', {}, success=False, error=error_msg)
            
            # Return detailed error information in debug mode
            if debug_mode:
                error_details = {
                    'validation_results': validation_results,
                    'error_type': 'input_validation',
                    'timestamp': datetime.now().isoformat()
                }
                return gr.update(value=None), gr.update(value=None), error_msg, error_details
            else:
                return gr.update(value=None), gr.update(value=None), error_msg, {}
        
        # Safe model loading
        pipeline, model_error = safe_model_loading(model_name, debug_mode)
        if pipeline is None:
            update_monitoring_data('generation', {'model_used': model_name}, success=False, error=model_error)
            return gr.update(value=None), gr.update(value=None), model_error, {}
        
        # Optimize pipeline settings with performance optimization
        try:
            pipeline = optimize_pipeline_settings(pipeline, use_mixed_precision, use_multi_gpu)
            
            # Apply additional performance optimizations
            if torch.cuda.is_available():
                # Get available memory for batch optimization
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                # Optimize batch processing
                batch_optimization = performance_optimizer.optimize_batch_processing(
                    num_images, available_memory
                )
                
                # Auto-tune parameters if in debug mode
                if debug_mode:
                    tuning_results = performance_optimizer.auto_tune_parameters(
                        pipeline, prompt, target_throughput=0.5
                    )
                    logger.info(f"Auto-tuning results: {tuning_results}")
                
                logger.info(f"Batch optimization: {batch_optimization}")
                
        except Exception as e:
            error_msg = f"Pipeline optimization failed: {str(e)}"
            if debug_mode:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
            logger.error(error_msg)
            update_monitoring_data('generation', {'model_used': model_name}, success=False, error=error_msg)
            return gr.update(value=None), gr.update(value=None), error_msg, {}
            
            # Setup multi-GPU if requested
            multi_gpu_enabled = False
            if use_multi_gpu and torch.cuda.device_count() > 1:
            try:
                pipeline, multi_gpu_enabled = setup_multi_gpu_pipeline(pipeline, use_ddp)
                logger.info(f"Multi-GPU setup: {'DDP' if use_ddp else 'DataParallel'} enabled: {multi_gpu_enabled}")
        except Exception as e:
                error_msg = f"Multi-GPU setup failed: {str(e)}"
                if debug_mode:
                    error_msg += f"\nTraceback: {traceback.format_exc()}"
                logger.warning(error_msg)
                # Continue without multi-GPU
        
        # Prepare generator
        generator = None
        if seed is not None:
            try:
                generator = torch.manual_seed(int(seed))
            except Exception as e:
                logger.warning(f"Failed to set seed {seed}: {e}")
        
        # Enable PyTorch debugging tools if debug mode is requested
            if debug_mode:
            # Enable comprehensive PyTorch debugging
            pytorch_debugger.enable_anomaly_detection(True)
            pytorch_debugger.enable_memory_tracking(True)
            pytorch_debugger.enable_gradient_tracking(True)
            
            # Start profiler for detailed performance analysis
            pytorch_debugger.start_profiler(
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                use_cuda=torch.cuda.is_available()
            )
            
            logger.info("✅ PyTorch debugging tools enabled: anomaly detection, memory tracking, gradient tracking, profiler")
            log_debug_info("Starting generation with comprehensive PyTorch debugging", {
                'prompt': prompt[:100], 'model_name': model_name, 'seed': seed,
                'num_images': num_images, 'mixed_precision': use_mixed_precision,
                'multi_gpu': multi_gpu_enabled,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            })
            
            # Debug pipeline model if available
            if hasattr(pipeline, 'unet'):
                pytorch_debugger.debug_model(pipeline.unet, log_details=True)
        
        # Perform inference with performance measurement
        try:
            if gradient_accumulation_steps > 1:
                # Measure performance for gradient accumulation
                output, perf_metrics = performance_optimizer.measure_performance(
                    "gradient_accumulation_inference",
                    generate_with_gradient_accumulation,
                    pipeline, prompt, num_images, generator, 
                    gradient_accumulation_steps, use_mixed_precision
                )
            else:
                # Measure performance for single inference
                output, perf_metrics = performance_optimizer.measure_performance(
                    "single_inference",
                    lambda: safe_inference(pipeline, prompt, num_images, generator, use_mixed_precision, debug_mode),
                    pipeline, prompt, num_images, generator, use_mixed_precision, debug_mode
                )
                
                # Extract output and error from safe_inference result
                if isinstance(output, tuple) and len(output) == 2:
                    output, inference_error = output
                    if output is None:
                        raise InferenceError(inference_error)
            
            # Process output
            images = output.images if hasattr(output, 'images') else output["images"]
            images = optimize_image_conversion(images)
            
            inference_time = time.time() - start_time
            gpu_utilization = get_gpu_utilization()
            
            logger.info(f"Inference completed successfully for model '{model_name}' in {inference_time:.2f}s.")
            
            if debug_mode:
                torch.autograd.set_detect_anomaly(False)
                # Stop profiler and export results
                pytorch_debugger.stop_profiler(export_path=f"logs/generation_profile_{int(time.time())}.json")
                
                # Disable debugging tools
                pytorch_debugger.enable_anomaly_detection(False)
                pytorch_debugger.enable_memory_tracking(False)
                pytorch_debugger.enable_gradient_tracking(False)
                
                # Get final memory stats
                final_memory_stats = pytorch_debugger.get_memory_stats()
                
                logger.info("✅ PyTorch debugging tools disabled and profiler results exported")
                log_debug_info("Generation completed successfully with debugging", {
                    'inference_time': inference_time,
                    'images_generated': len(images),
                    'gpu_utilization': gpu_utilization,
                    'final_memory_stats': final_memory_stats
                })
            
            # Get performance summary
            performance_summary = performance_optimizer.get_performance_summary()
            
            performance_metrics = {
                "inference_time_seconds": round(inference_time, 2),
                "images_generated": len(images),
                "model_used": model_name,
                "mixed_precision": use_mixed_precision,
                "multi_gpu_enabled": multi_gpu_enabled,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": num_images * gradient_accumulation_steps if gradient_accumulation_steps > 1 else num_images,
                "validation_results": validation_results if debug_mode else None,
                "performance_optimizations": list(performance_optimizer.current_optimizations),
                "performance_summary": performance_summary,
                "latest_performance_metrics": perf_metrics if 'perf_metrics' in locals() else {},
                **gpu_utilization
            }
            
            # Update monitoring data
            update_monitoring_data('generation', performance_metrics, success=True)
            
            clear_gpu_memory()
            return images, None, None, performance_metrics
            
        except Exception as e:
            error_info = get_detailed_error_info(e, debug_mode)
            error_msg = f"Generation failed: {str(e)}"
            
            if debug_mode:
                error_msg += f"\n\nDebug Information:\n{json.dumps(error_info, indent=2)}"
            
            logger.error(f"Error during generation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if debug_mode:
                # Stop profiler and disable debugging tools
                pytorch_debugger.stop_profiler()
                pytorch_debugger.enable_anomaly_detection(False)
                pytorch_debugger.enable_memory_tracking(False)
                pytorch_debugger.enable_gradient_tracking(False)
            
            update_monitoring_data('generation', {'model_used': model_name}, success=False, error=str(e))
            clear_gpu_memory()
            return gr.update(value=None), gr.update(value=None), error_msg, error_info if debug_mode else {}
            
    except Exception as e:
        error_info = get_detailed_error_info(e, debug_mode)
        error_msg = f"Unexpected error: {str(e)}"
        
        if debug_mode:
            error_msg += f"\n\nDebug Information:\n{json.dumps(error_info, indent=2)}"
        
        logger.error(f"Unexpected error during generation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        if debug_mode:
            # Stop profiler and disable debugging tools
            pytorch_debugger.stop_profiler()
            pytorch_debugger.enable_anomaly_detection(False)
            pytorch_debugger.enable_memory_tracking(False)
            pytorch_debugger.enable_gradient_tracking(False)
        
        update_monitoring_data('generation', {'model_used': model_name}, success=False, error=str(e))
        clear_gpu_memory()
        return gr.update(value=None), gr.update(value=None), error_msg, error_info if debug_mode else {}

def evaluate_images(generated, reference, debug_mode, use_mixed_precision, use_multi_gpu, use_ddp) -> Any:
    """Enhanced evaluation function with comprehensive error handling and debugging."""
    start_time = time.time()
    
    try:
        # Input validation for evaluation
        if not generated or not reference:
            error_msg = "Both generated and reference images are required for evaluation."
            update_monitoring_data('evaluation', {}, success=False, error=error_msg)
            return None, error_msg, {}
        
        if len(generated) == 0 or len(reference) == 0:
            error_msg = "Generated and reference image lists cannot be empty."
            update_monitoring_data('evaluation', {}, success=False, error=error_msg)
            return None, error_msg, {}
        
        # Log evaluation start
        if debug_mode:
            log_debug_info("Starting image evaluation", {
                'generated_count': len(generated),
                'reference_count': len(reference),
                'mixed_precision': use_mixed_precision,
                'multi_gpu': use_multi_gpu
            })
        
        # Setup device and evaluator
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluator = ModelEvaluator(None, device)
        except Exception as e:
            error_msg = f"Failed to initialize evaluator: {str(e)}"
            if debug_mode:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
            logger.error(error_msg)
            update_monitoring_data('evaluation', {}, success=False, error=error_msg)
            return None, error_msg, {}
        
        # Setup multi-GPU for evaluation if requested
        multi_gpu_enabled = False
        if use_multi_gpu and torch.cuda.device_count() > 1:
            try:
                if use_ddp:
                    evaluator.model = torch.nn.parallel.DistributedDataParallel(evaluator.model)
                else:
                    evaluator.model = torch.nn.DataParallel(evaluator.model)
                multi_gpu_enabled = True
                logger.info(f"Multi-GPU evaluation setup: {'DDP' if use_ddp else 'DataParallel'} enabled")
            except Exception as e:
                error_msg = f"Multi-GPU setup failed: {str(e)}"
                if debug_mode:
                    error_msg += f"\nTraceback: {traceback.format_exc()}"
                logger.warning(error_msg)
                # Continue without multi-GPU
        
        # Perform evaluation
        try:
            logger.info("Starting evaluation of generated images.")
            if debug_mode:
                torch.autograd.set_detect_anomaly(True)
                logger.info("Autograd anomaly detection enabled for evaluation debugging.")
            
            # Preprocess images efficiently with error handling
            try:
                generated_processed = preprocess_images_batch(generated)
                reference_processed = preprocess_images_batch(reference)
            except Exception as e:
                error_msg = f"Image preprocessing failed: {str(e)}"
                if debug_mode:
                    error_msg += f"\nTraceback: {traceback.format_exc()}"
                raise Exception(error_msg)
            
            # Perform evaluation with mixed precision if requested
            try:
            if use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    metrics = evaluator.evaluate_quality(generated_processed, reference_processed)
            else:
                metrics = evaluator.evaluate_quality(generated_processed, reference_processed)
            except Exception as e:
                error_msg = f"Evaluation computation failed: {str(e)}"
                if debug_mode:
                    error_msg += f"\nTraceback: {traceback.format_exc()}"
                raise Exception(error_msg)
            
            evaluation_time = time.time() - start_time
            gpu_utilization = get_gpu_utilization()
            
            logger.info(f"Evaluation completed successfully in {evaluation_time:.2f}s.")
            
            if debug_mode:
                torch.autograd.set_detect_anomaly(False)
                logger.info("Autograd anomaly detection disabled.")
                log_debug_info("Evaluation completed successfully", {
                    'evaluation_time': evaluation_time,
                    'metrics_keys': list(metrics.keys()) if metrics else [],
                    'gpu_utilization': gpu_utilization
                })
            
            performance_metrics = {
                "evaluation_time_seconds": round(evaluation_time, 2),
                "mixed_precision": use_mixed_precision,
                "multi_gpu_enabled": multi_gpu_enabled,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "generated_images_count": len(generated),
                "reference_images_count": len(reference),
                **gpu_utilization
            }
            
            # Update monitoring data
            update_monitoring_data('evaluation', performance_metrics, success=True)
            
            clear_gpu_memory()
            return metrics, None, performance_metrics
            
        except Exception as e:
            error_info = get_detailed_error_info(e, debug_mode)
            error_msg = f"Evaluation failed: {str(e)}"
            
            if debug_mode:
                error_msg += f"\n\nDebug Information:\n{json.dumps(error_info, indent=2)}"
            
            logger.error(f"Error during evaluation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if debug_mode:
                torch.autograd.set_detect_anomaly(False)
            
            update_monitoring_data('evaluation', {}, success=False, error=str(e))
            clear_gpu_memory()
            return None, error_msg, error_info if debug_mode else {}
            
    except Exception as e:
        error_info = get_detailed_error_info(e, debug_mode)
        error_msg = f"Unexpected evaluation error: {str(e)}"
        
        if debug_mode:
            error_msg += f"\n\nDebug Information:\n{json.dumps(error_info, indent=2)}"
        
        logger.error(f"Unexpected error during evaluation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        if debug_mode:
            torch.autograd.set_detect_anomaly(False)
        
        update_monitoring_data('evaluation', {}, success=False, error=str(e))
        clear_gpu_memory()
        return None, error_msg, error_info if debug_mode else {}

def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data with enhanced error tracking."""
    try:
        # Get basic monitoring data
        dashboard_data = {
        'model_health': get_model_health_status(),
        'performance_summary': get_performance_summary(),
        'system_stats': get_system_stats(),
        'recent_inferences': list(monitoring_data['inference_history'])[-10:],
        'recent_errors': list(monitoring_data['error_history'])[-5:],
            'memory_trend': list(monitoring_data['memory_history'])[-20:],
            'error_analytics': {
                'error_counts': dict(monitoring_data['error_counts']),
                'total_errors': sum(monitoring_data['error_counts'].values()),
                'most_common_error': max(monitoring_data['error_counts'].items(), key=lambda x: x[1]) if monitoring_data['error_counts'] else None,
                'recent_validation_failures': list(monitoring_data['input_validation_failures'])[-5:],
                'error_trend': len(monitoring_data['error_history'])
            },
            'debug_info': {
                'debug_logs_count': len(monitoring_data['debug_logs']),
                'recent_debug_logs': list(monitoring_data['debug_logs'])[-5:],
                'system_health': {
                    'memory_usage': psutil.virtual_memory().percent,
                    'cpu_usage': psutil.cpu_percent(),
                    'gpu_available': torch.cuda.is_available(),
                    'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                }
            }
        }
        
        # Add error rate calculation
        total_operations = len(monitoring_data['inference_history']) + len(monitoring_data['error_history'])
        if total_operations > 0:
            dashboard_data['error_analytics']['error_rate'] = len(monitoring_data['error_history']) / total_operations
        else:
            dashboard_data['error_analytics']['error_rate'] = 0.0
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating monitoring dashboard: {e}")
        return {
            'error': f"Failed to generate dashboard: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }

def get_error_summary():
    """Get a summary of recent errors for debugging."""
    try:
        recent_errors = list(monitoring_data['error_history'])[-10:]
        error_summary = {
            'total_errors': len(monitoring_data['error_history']),
            'recent_errors': recent_errors,
            'error_types': defaultdict(int),
            'error_timeline': []
        }
        
        for error in recent_errors:
            error_type = error.get('error_type', 'unknown')
            error_summary['error_types'][error_type] += 1
            error_summary['error_timeline'].append({
                'timestamp': error.get('timestamp', ''),
                'error_type': error_type,
                'message': error.get('message', '')[:100]
            })
        
        return error_summary
        
    except Exception as e:
        logger.error(f"Error generating error summary: {e}")
        return {'error': f"Failed to generate error summary: {str(e)}"}

def clear_error_logs():
    """Clear error logs and reset error counters."""
    try:
        monitoring_data['error_history'].clear()
        monitoring_data['error_counts'].clear()
        monitoring_data['input_validation_failures'].clear()
        monitoring_data['debug_logs'].clear()
        logger.info("Error logs cleared successfully")
        return {"status": "success", "message": "Error logs cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear error logs: {e}")
        return {"status": "error", "message": f"Failed to clear error logs: {str(e)}"}

def export_debug_info():
    """Export debug information for external analysis."""
    try:
        debug_export = {
            'timestamp': datetime.now().isoformat(),
            'error_history': list(monitoring_data['error_history']),
            'error_counts': dict(monitoring_data['error_counts']),
            'input_validation_failures': list(monitoring_data['input_validation_failures']),
            'debug_logs': list(monitoring_data['debug_logs']),
            'system_info': {
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'memory_usage': psutil.virtual_memory()._asdict(),
                'cpu_info': {
                    'count': psutil.cpu_count(),
                    'usage': psutil.cpu_percent(interval=1)
                }
            }
        }
        
        # Save to file
        filename = f"debug_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(debug_export, f, indent=2, default=str)
        
        logger.info(f"Debug information exported to {filename}")
        return {"status": "success", "filename": filename, "message": f"Debug information exported to {filename}"}
        
    except Exception as e:
        logger.error(f"Failed to export debug information: {e}")
        return {"status": "error", "message": f"Failed to export debug information: {str(e)}"}


# =============================================================================
# MIXED PRECISION TRAINING INTERFACE FUNCTIONS
# =============================================================================

def train_model_with_mixed_precision_interface(model_type: str, num_epochs: int, 
                                             batch_size: int, learning_rate: float,
                                             use_mixed_precision: bool, adaptive: bool) -> str:
    """Train model with mixed precision for the Gradio interface."""
    try:
        # Import mixed precision training system
            MixedPrecisionConfig, train_with_mixed_precision,
            optimize_mixed_precision_settings
        )
        
        # Create model based on type
        if model_type == "linear":
            model = torch.nn.Linear(10, 2)
        elif model_type == "mlp":
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )
        else:
            model = torch.nn.Linear(10, 2)
        
        # Create dataset and data loader
        X = torch.randn(1000, 10)
        y = torch.randint(0, 2, (1000,))
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create mixed precision config
        if use_mixed_precision:
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 8.0
            config = optimize_mixed_precision_settings(model, batch_size, available_memory)
        else:
            config = MixedPrecisionConfig(enabled=False)
        
        # Train with mixed precision
        training_metrics = train_with_mixed_precision(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            config=config,
            adaptive=adaptive
        )
        
        return json.dumps(training_metrics, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Mixed precision training error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def benchmark_mixed_precision_interface(model_type: str, batch_size: int) -> str:
    """Benchmark mixed precision performance for the Gradio interface."""
    try:
        # Import mixed precision training system
        
        # Create model based on type
        if model_type == "linear":
            model = torch.nn.Linear(10, 2)
        elif model_type == "mlp":
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 32), torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )
        else:
            model = torch.nn.Linear(10, 2)
        
        # Move model to GPU
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create sample data
        data = torch.randn(batch_size, 10)
        if torch.cuda.is_available():
            data = data.cuda()
        
        # Run benchmark
        benchmark_results = benchmark_mixed_precision(model, data, num_iterations=50)
        
        return json.dumps(benchmark_results, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return json.dumps({"error": str(e)}, indent=2)


def get_mixed_precision_recommendations_interface() -> str:
    """Get mixed precision recommendations based on system."""
    try:
        # Import mixed precision training system
        
        recommendations = get_mixed_precision_recommendations()
        return recommendations
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def get_mixed_precision_status_interface() -> str:
    """Get current mixed precision status and configuration."""
    try:
        # Import mixed precision training system
        
        status = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_info': {},
            'recommended_config': {}
        }
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            status['gpu_info'] = {
                'name': gpu_props.name,
                'memory_gb': gpu_props.total_memory / (1024**3),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            }
            
            # Get recommended configuration
            total_memory = status['gpu_info']['memory_gb']
            if total_memory >= 16:
                status['recommended_config'] = {
                    'use_mixed_precision': True,
                    'init_scale': 2**16,
                    'growth_factor': 2.0,
                    'memory_efficient': False
                }
            elif total_memory >= 8:
                status['recommended_config'] = {
                    'use_mixed_precision': True,
                    'init_scale': 2**18,
                    'growth_factor': 1.5,
                    'memory_efficient': True
                }
            else:
                status['recommended_config'] = {
                    'use_mixed_precision': True,
                    'init_scale': 2**20,
                    'growth_factor': 1.2,
                    'memory_efficient': True
                }
        else:
            status['recommended_config'] = {
                'use_mixed_precision': False,
                'reason': 'CUDA not available'
            }
        
        return json.dumps(status, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# Transformers Integration Interface Functions
def initialize_transformers_interface() -> str:
    """Interface function for initializing transformers system."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return json.dumps({"error": "Transformers integration system not available"})
        
        success = initialize_transformers_system()
        if success:
            return json.dumps({"status": "success", "message": "Transformers system initialized successfully"})
        else:
            return json.dumps({"status": "error", "message": "Failed to initialize transformers system"})
            
    except Exception as e:
        logger.error(f"Transformers initialization failed: {e}")
        return json.dumps({"error": str(e)})


def get_available_models_interface() -> str:
    """Interface function for getting available models."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return json.dumps({"error": "Transformers integration system not available"})
        
        models = get_available_models()
        return json.dumps(models, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return json.dumps({"error": str(e)})


def train_transformers_model_interface(model_name: str, model_type: str, task: str,
                                     train_texts: str, val_texts: str = "",
                                     num_epochs: int = 3, batch_size: int = 4,
                                     learning_rate: float = 5e-5, use_peft: bool = True) -> str:
    """Interface function for training transformers models."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return json.dumps({"error": "Transformers integration system not available"})
        
        # Parse training texts
        train_text_list = [text.strip() for text in train_texts.split("\n") if text.strip()]
        val_text_list = [text.strip() for text in val_texts.split("\n") if text.strip()] if val_texts else None
        
        if not train_text_list:
            return json.dumps({"error": "No training texts provided"})
        
        # Create configuration
        config = TransformersConfig(
            model_name=model_name,
            model_type=model_type,
            task=task,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_peft=use_peft
        )
        
        # Initialize trainer
        trainer = AdvancedTransformersTrainer(config)
        
        # Train model
        result = trainer.train(train_text_list, val_texts=val_text_list)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Transformers training failed: {e}")
        return json.dumps({"error": str(e)})


def generate_text_interface(prompt: str, model_path: str = "./transformers_final_model",
                          max_new_tokens: int = 100, temperature: float = 0.7,
                          top_p: float = 0.9, do_sample: bool = True) -> str:
    """Interface function for text generation."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return json.dumps({"error": "Transformers integration system not available"})
        
        # Validate inputs
        is_valid, error_msg = validate_transformers_inputs(prompt, model_path, 512)
        if not is_valid:
            return json.dumps({"error": error_msg})
        
        # Create pipeline
        config = TransformersConfig()
        pipeline = TransformersPipeline(model_path, config)
        
        # Generate text
        generated_text = pipeline.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        
        return json.dumps({
            "prompt": prompt,
            "generated_text": generated_text,
            "model_path": model_path
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        return json.dumps({"error": str(e)})


def batch_generate_interface(prompts: str, model_path: str = "./transformers_final_model",
                           max_new_tokens: int = 100, temperature: float = 0.7) -> str:
    """Interface function for batch text generation."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return json.dumps({"error": "Transformers integration system not available"})
        
        # Parse prompts
        prompt_list = [prompt.strip() for prompt in prompts.split("\n") if prompt.strip()]
        if not prompt_list:
            return json.dumps({"error": "No prompts provided"})
        
        # Create pipeline
        config = TransformersConfig()
        pipeline = TransformersPipeline(model_path, config)
        
        # Generate texts
        generated_texts = pipeline.batch_generate(
            prompt_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Format results
        results = []
        for prompt, generated in zip(prompt_list, generated_texts):
            results.append({
                "prompt": prompt,
                "generated_text": generated
            })
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Batch text generation failed: {e}")
        return json.dumps({"error": str(e)})


def classify_texts_interface(texts: str, model_path: str = "./transformers_final_model") -> str:
    """Interface function for text classification."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return json.dumps({"error": "Transformers integration system not available"})
        
        # Parse texts
        text_list = [text.strip() for text in texts.split("\n") if text.strip()]
        if not text_list:
            return json.dumps({"error": "No texts provided"})
        
        # Create pipeline
        config = TransformersConfig(model_type="sequence_classification")
        pipeline = TransformersPipeline(model_path, config)
        
        # Classify texts
        classifications = pipeline.classify(text_list)
        
        return json.dumps(classifications, indent=2)
        
    except Exception as e:
        logger.error(f"Text classification failed: {e}")
        return json.dumps({"error": str(e)})


def get_transformers_status_interface() -> str:
    """Interface function for getting transformers system status."""
    try:
        if not TRANSFORMERS_AVAILABLE:
            return json.dumps({"status": "unavailable", "message": "Transformers system not available"})
        
        # Get available models
        models = get_available_models()
        
        # Check if model files exist
        model_path = "./transformers_final_model"
        model_exists = os.path.exists(model_path)
        
        status = {
            "status": "available",
            "available_models": models,
            "trained_model_exists": model_exists,
            "model_path": model_path if model_exists else None,
            "transformers_version": "4.35.0" if TRANSFORMERS_AVAILABLE else "not_installed",
            "peft_available": PEFT_AVAILABLE if 'PEFT_AVAILABLE' in globals() else False
        }
        
        return json.dumps(status, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to get transformers status: {e}")
        return json.dumps({"error": str(e)})


with gr.Blocks(title="Enhanced Diffusion Model Playground with Error Handling") as demo:
    gr.Markdown("# 🚀 Enhanced Diffusion Model Playground")
    gr.Markdown("### Production-ready with comprehensive error handling and debugging capabilities")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🎨 Generation Settings")
            prompt = gr.Textbox(
                label="Prompt", 
                value="A beautiful landscape with mountains",
                placeholder="Enter your prompt here...",
                lines=3,
                max_lines=5
            )
            model_name = gr.Dropdown(
                list(PIPELINE_CONFIGS.keys()), 
                label="Model", 
                value="Stable Diffusion v1.5",
                info="Select the diffusion model to use"
            )
            seed = gr.Number(
                label="Random Seed", 
                value=42, 
                precision=0,
                info="Set to -1 for random seed"
            )
            num_images = gr.Slider(
                1, 8, value=1, step=1, 
                label="Number of Images",
                info="Maximum 8 images per generation"
            )
            gradient_accumulation_steps = gr.Slider(
                1, 8, value=1, step=1, 
                label="Gradient Accumulation Steps",
                info="For large batch processing"
            )
            
            with gr.Accordion("🔧 Advanced Settings", open=False):
                debug_mode = gr.Checkbox(
                    label="Debug Mode", 
                    value=False,
                    info="Enable detailed error reporting and autograd anomaly detection"
                )
                use_mixed_precision = gr.Checkbox(
                    label="Use Mixed Precision (FP16)", 
                    value=True,
                    info="Reduces memory usage and speeds up inference"
                )
                use_multi_gpu = gr.Checkbox(
                    label="Use Multi-GPU", 
                    value=False,
                    info="Distribute computation across multiple GPUs"
                )
                use_ddp = gr.Checkbox(
                    label="Use DistributedDataParallel", 
                    value=False,
                    info="More efficient than DataParallel for multi-GPU"
                )
            
            generate_btn = gr.Button("🚀 Generate Images", variant="primary", size="lg")
            
            with gr.Accordion("⚠️ Error Information", open=False):
                error_box = gr.Textbox(
                    label="Error Details", 
                    visible=True, 
                    interactive=False,
                    lines=5,
                    max_lines=10
                )
            
            performance_box = gr.JSON(label="📊 Performance Metrics")
            
        with gr.Column(scale=1):
            gr.Markdown("## 🖼️ Generated Images")
            gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery"
            ).style(grid=[2], height="auto")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📊 Monitoring & Debugging Dashboard")
            with gr.Row():
                refresh_monitoring_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
                clear_logs_btn = gr.Button("🗑️ Clear Error Logs", variant="secondary")
                export_debug_btn = gr.Button("📤 Export Debug Info", variant="secondary")
            
            with gr.Tabs():
                with gr.TabItem("📈 System Health"):
                    monitoring_dashboard = gr.JSON(label="Monitoring Dashboard")
                with gr.TabItem("🚨 Error Analytics"):
                    error_summary = gr.JSON(label="Error Summary")
                with gr.TabItem("🔍 Debug Logs"):
                    debug_logs = gr.JSON(label="Recent Debug Logs")
                with gr.TabItem("🤖 Transformers"):
                    gr.Markdown("### 🚀 Advanced Transformers Integration")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 🔧 System Setup")
                            init_transformers_btn = gr.Button("🚀 Initialize Transformers", variant="primary")
                            get_models_btn = gr.Button("📋 Get Available Models", variant="secondary")
                            get_status_btn = gr.Button("📊 Check Status", variant="secondary")
                            
                            gr.Markdown("#### 🏋️ Model Training")
                            train_model_name = gr.Dropdown(
                                choices=["microsoft/DialoGPT-medium", "gpt2", "bert-base-uncased"],
                                label="Model Name",
                                value="microsoft/DialoGPT-medium"
                            )
                            train_model_type = gr.Dropdown(
                                choices=["causal", "sequence_classification", "token_classification"],
                                label="Model Type",
                                value="causal"
                            )
                            train_task = gr.Dropdown(
                                choices=["text_generation", "classification", "token_classification"],
                                label="Task",
                                value="text_generation"
                            )
                            train_texts = gr.Textbox(
                                label="Training Texts",
                                placeholder="Enter training texts (one per line)...",
                                lines=5
                            )
                            val_texts = gr.Textbox(
                                label="Validation Texts (Optional)",
                                placeholder="Enter validation texts (one per line)...",
                                lines=3
                            )
                            train_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                            train_batch_size = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                            train_lr = gr.Slider(1e-6, 1e-3, value=5e-5, step=1e-6, label="Learning Rate")
                            use_peft = gr.Checkbox(label="Use PEFT", value=True)
                            
                            train_btn = gr.Button("🏋️ Train Model", variant="primary")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📝 Text Generation")
                            gen_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Enter your prompt here...",
                                lines=3
                            )
                            gen_max_tokens = gr.Slider(10, 500, value=100, step=10, label="Max New Tokens")
                            gen_temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                            gen_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="Top P")
                            gen_do_sample = gr.Checkbox(label="Do Sample", value=True)
                            
                            generate_text_btn = gr.Button("✨ Generate Text", variant="primary")
                            
                            gr.Markdown("#### 📦 Batch Generation")
                            batch_prompts = gr.Textbox(
                                label="Prompts (one per line)",
                                placeholder="Enter multiple prompts...",
                                lines=4
                            )
                            batch_generate_btn = gr.Button("📦 Batch Generate", variant="secondary")
                            
                            gr.Markdown("#### 🏷️ Text Classification")
                            classify_texts = gr.Textbox(
                                label="Texts to Classify (one per line)",
                                placeholder="Enter texts to classify...",
                                lines=4
                            )
                            classify_btn = gr.Button("🏷️ Classify Texts", variant="secondary")
                    
                    with gr.Accordion("📊 Results", open=True):
                        transformers_output = gr.JSON(label="Transformers Results")
    
    with gr.Row():
        gr.Markdown("## 🎯 Image Evaluation")
        with gr.Column():
        generated_input = gr.Gallery(label="Generated Images for Evaluation")
        reference_input = gr.Gallery(label="Reference Images for Evaluation")
            
            with gr.Row():
        eval_debug_mode = gr.Checkbox(label="Debug Mode for Evaluation", value=False)
        eval_mixed_precision = gr.Checkbox(label="Use Mixed Precision for Evaluation", value=True)
        eval_multi_gpu = gr.Checkbox(label="Use Multi-GPU for Evaluation", value=False)
        eval_ddp = gr.Checkbox(label="Use DistributedDataParallel for Evaluation", value=False)
            
            eval_btn = gr.Button("🔍 Evaluate Images", variant="primary")
            
            with gr.Accordion("📊 Evaluation Results", open=True):
        metrics_output = gr.JSON(label="Evaluation Metrics")
                eval_error_box = gr.Textbox(label="Evaluation Error", visible=False, interactive=False)
        eval_performance_box = gr.JSON(label="Evaluation Performance Metrics")

    with gr.Row():
        gr.Markdown("## 🚀 Mixed Precision Training")
        with gr.Column():
            with gr.Row():
                mp_model_type = gr.Dropdown(
                    choices=["linear", "mlp", "conv"],
                    label="Model Type",
                    value="mlp",
                    info="Select model architecture for training"
                )
                mp_num_epochs = gr.Slider(
                    1, 50, value=5, step=1,
                    label="Number of Epochs",
                    info="Training epochs"
                )
                mp_batch_size = gr.Slider(
                    8, 128, value=32, step=8,
                    label="Batch Size",
                    info="Training batch size"
                )
                mp_learning_rate = gr.Slider(
                    1e-5, 1e-2, value=1e-4, step=1e-5,
                    label="Learning Rate",
                    info="Training learning rate"
                )
            
            with gr.Row():
                mp_use_mixed_precision = gr.Checkbox(
                    label="Use Mixed Precision",
                    value=True,
                    info="Enable automatic mixed precision training"
                )
                mp_adaptive = gr.Checkbox(
                    label="Adaptive Training",
                    value=True,
                    info="Adapt mixed precision settings based on performance"
                )
            
            with gr.Row():
                mp_train_btn = gr.Button("🚀 Train with Mixed Precision", variant="primary")
                mp_benchmark_btn = gr.Button("📊 Benchmark Performance", variant="secondary")
                mp_recommendations_btn = gr.Button("💡 Get Recommendations", variant="secondary")
                mp_status_btn = gr.Button("📋 Check Status", variant="secondary")
            
            with gr.Accordion("📊 Training Results", open=True):
                mp_training_output = gr.JSON(label="Training Metrics")
                mp_benchmark_output = gr.JSON(label="Benchmark Results")
                mp_recommendations_output = gr.JSON(label="System Recommendations")
                mp_status_output = gr.JSON(label="Mixed Precision Status")

    # Event handlers
    generate_btn.click(
        fn=generate,
        inputs=[prompt, model_name, seed, num_images, debug_mode, use_mixed_precision, use_multi_gpu, use_ddp, gradient_accumulation_steps],
        outputs=[gallery, error_box, performance_box],
        show_progress=True
    )
    
    eval_btn.click(
        fn=evaluate_images,
        inputs=[generated_input, reference_input, eval_debug_mode, eval_mixed_precision, eval_multi_gpu, eval_ddp],
        outputs=[metrics_output, eval_error_box, eval_performance_box],
        show_progress=True
    )
    
    refresh_monitoring_btn.click(
        fn=get_monitoring_dashboard,
        outputs=monitoring_dashboard
    )
    
    clear_logs_btn.click(
        fn=clear_error_logs,
        outputs=gr.JSON(label="Clear Logs Result")
    )
    
    export_debug_btn.click(
        fn=export_debug_info,
        outputs=gr.JSON(label="Export Result")
    )
    
    # Error summary tab
    refresh_monitoring_btn.click(
        fn=get_error_summary,
        outputs=error_summary
    )
    
    # Debug logs tab
    refresh_monitoring_btn.click(
        fn=lambda: list(monitoring_data['debug_logs'])[-10:],
        outputs=debug_logs
    )
    
    # Mixed precision training event handlers
    mp_train_btn.click(
        fn=train_model_with_mixed_precision_interface,
        inputs=[mp_model_type, mp_num_epochs, mp_batch_size, mp_learning_rate, 
                mp_use_mixed_precision, mp_adaptive],
        outputs=mp_training_output,
        show_progress=True
    )
    
    mp_benchmark_btn.click(
        fn=benchmark_mixed_precision_interface,
        inputs=[mp_model_type, mp_batch_size],
        outputs=mp_benchmark_output,
        show_progress=True
    )
    
    mp_recommendations_btn.click(
        fn=get_mixed_precision_recommendations_interface,
        outputs=mp_recommendations_output
    )
    
    mp_status_btn.click(
        fn=get_mixed_precision_status_interface,
        outputs=mp_status_output
    )
    
    # Transformers event handlers
    init_transformers_btn.click(
        fn=initialize_transformers_interface,
        outputs=transformers_output
    )
    
    get_models_btn.click(
        fn=get_available_models_interface,
        outputs=transformers_output
    )
    
    get_status_btn.click(
        fn=get_transformers_status_interface,
        outputs=transformers_output
    )
    
    train_btn.click(
        fn=train_transformers_model_interface,
        inputs=[train_model_name, train_model_type, train_task, train_texts, val_texts,
                train_epochs, train_batch_size, train_lr, use_peft],
        outputs=transformers_output,
        show_progress=True
    )
    
    generate_text_btn.click(
        fn=generate_text_interface,
        inputs=[gen_prompt, gr.Textbox(value="./transformers_final_model"), gen_max_tokens,
                gen_temperature, gen_top_p, gen_do_sample],
        outputs=transformers_output
    )
    
    batch_generate_btn.click(
        fn=batch_generate_interface,
        inputs=[batch_prompts, gr.Textbox(value="./transformers_final_model")],
        outputs=transformers_output
    )
    
    classify_btn.click(
        fn=classify_texts_interface,
        inputs=[classify_texts, gr.Textbox(value="./transformers_final_model")],
        outputs=transformers_output
    )
    
    # Auto-refresh monitoring dashboard
    demo.load(
        fn=get_monitoring_dashboard,
        outputs=monitoring_dashboard
    )

match __name__:
    case "__main__":
    demo.launch() 