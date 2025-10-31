#!/usr/bin/env python3
"""
Performance Metrics - Comprehensive performance measurement and tracking
Provides detailed performance metrics for optimization evaluation
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
from collections import defaultdict, deque
import threading
import queue
import psutil
import gc

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a specific time."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: float
    inference_time: float
    throughput: float
    latency: float

@dataclass
class PerformanceStatistics:
    """Performance statistics over time."""
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_gpu_usage: float
    avg_gpu_memory: float
    avg_inference_time: float
    avg_throughput: float
    avg_latency: float
    max_cpu_usage: float
    max_memory_usage: float
    max_gpu_usage: float
    max_gpu_memory: float
    min_inference_time: float
    max_inference_time: float
    std_inference_time: float
    std_throughput: float

class PerformanceMetrics:
    """Comprehensive performance metrics collector and analyzer."""
    
    def __init__(self, collection_interval: float = 1.0, max_history: int = 1000):
        self.collection_interval = collection_interval
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # Performance data
        self.performance_history = deque(maxlen=max_history)
        self.current_metrics = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.benchmark_results = {}
        self.optimization_baselines = {}
        
        # Initialize system info
        self._initialize_system_info()
    
    def _initialize_system_info(self):
        """Initialize system information."""
        try:
            self.system_info = {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            if torch.cuda.is_available():
                self.system_info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            self.logger.info(f"System info initialized: {self.system_info}")
            
        except Exception as e:
            self.logger.error(f"System info initialization failed: {e}")
            self.system_info = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        try:
            self.is_monitoring = True
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.is_monitoring = False
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            self.logger.warning("Monitoring not started")
            return
        
        try:
            self.is_monitoring = False
            self.stop_event.set()
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Collect performance metrics
                snapshot = self._collect_performance_snapshot()
                self.performance_history.append(snapshot)
                
                # Update current metrics
                self.current_metrics = {
                    'cpu_usage': snapshot.cpu_usage,
                    'memory_usage': snapshot.memory_usage,
                    'gpu_usage': snapshot.gpu_usage,
                    'gpu_memory': snapshot.gpu_memory,
                    'inference_time': snapshot.inference_time,
                    'throughput': snapshot.throughput,
                    'latency': snapshot.latency
                }
                
                # Wait for next collection
                self.stop_event.wait(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                break
    
    def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot."""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # GPU usage
            gpu_usage = 0.0
            gpu_memory = 0.0
            
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0.0
                gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            
            # Inference metrics (placeholder - would be updated by actual inference)
            inference_time = self.current_metrics.get('inference_time', 0.0)
            throughput = self.current_metrics.get('throughput', 0.0)
            latency = self.current_metrics.get('latency', 0.0)
            
            return PerformanceSnapshot(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                inference_time=inference_time,
                throughput=throughput,
                latency=latency
            )
            
        except Exception as e:
            self.logger.error(f"Performance snapshot collection failed: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_usage=0.0,
                gpu_memory=0.0,
                inference_time=0.0,
                throughput=0.0,
                latency=0.0
            )
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       num_iterations: int = 100, warmup_iterations: int = 10) -> Dict[str, Any]:
        """Benchmark model performance."""
        try:
            model.eval()
            device = next(model.parameters()).device
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = model(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            inference_times = []
            for _ in range(num_iterations):
                iter_start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                iter_end = time.time()
                inference_times.append(iter_end - iter_start)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.time() - start_time
            
            # Calculate metrics
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            min_inference_time = np.min(inference_times)
            max_inference_time = np.max(inference_times)
            
            throughput = 1.0 / avg_inference_time
            latency = avg_inference_time * 1000  # Convert to ms
            
            # Memory usage
            memory_peak = 0.0
            memory_average = 0.0
            
            if torch.cuda.is_available():
                memory_peak = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                memory_average = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            # Store benchmark results
            benchmark_id = str(uuid.uuid4())
            self.benchmark_results[benchmark_id] = {
                'model_id': id(model),
                'input_shape': input_shape,
                'num_iterations': num_iterations,
                'avg_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'min_inference_time': min_inference_time,
                'max_inference_time': max_inference_time,
                'throughput': throughput,
                'latency': latency,
                'memory_peak': memory_peak,
                'memory_average': memory_average,
                'total_time': total_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Update current metrics
            self.current_metrics.update({
                'inference_time': avg_inference_time,
                'throughput': throughput,
                'latency': latency
            })
            
            return {
                'benchmark_id': benchmark_id,
                'avg_inference_time': avg_inference_time,
                'std_inference_time': std_inference_time,
                'min_inference_time': min_inference_time,
                'max_inference_time': max_inference_time,
                'throughput': throughput,
                'latency': latency,
                'memory_peak': memory_peak,
                'memory_average': memory_average,
                'total_time': total_time
            }
            
        except Exception as e:
            self.logger.error(f"Model benchmarking failed: {e}")
            return {'error': str(e)}
    
    def compare_models(self, model1: nn.Module, model2: nn.Module, 
                      input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Compare performance of two models."""
        try:
            # Benchmark both models
            results1 = self.benchmark_model(model1, input_shape)
            results2 = self.benchmark_model(model2, input_shape)
            
            if 'error' in results1 or 'error' in results2:
                return {'error': 'Benchmarking failed for one or both models'}
            
            # Calculate improvements
            speed_improvement = (results1['avg_inference_time'] - results2['avg_inference_time']) / results1['avg_inference_time']
            throughput_improvement = (results2['throughput'] - results1['throughput']) / results1['throughput']
            memory_improvement = (results1['memory_peak'] - results2['memory_peak']) / results1['memory_peak']
            
            return {
                'model1_results': results1,
                'model2_results': results2,
                'speed_improvement': speed_improvement,
                'throughput_improvement': throughput_improvement,
                'memory_improvement': memory_improvement,
                'overall_improvement': (speed_improvement + throughput_improvement + memory_improvement) / 3.0
            }
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            return {'error': str(e)}
    
    def get_performance_statistics(self) -> PerformanceStatistics:
        """Get performance statistics over time."""
        try:
            if not self.performance_history:
                return PerformanceStatistics(
                    avg_cpu_usage=0.0, avg_memory_usage=0.0, avg_gpu_usage=0.0, avg_gpu_memory=0.0,
                    avg_inference_time=0.0, avg_throughput=0.0, avg_latency=0.0,
                    max_cpu_usage=0.0, max_memory_usage=0.0, max_gpu_usage=0.0, max_gpu_memory=0.0,
                    min_inference_time=0.0, max_inference_time=0.0, std_inference_time=0.0, std_throughput=0.0
                )
            
            # Extract metrics
            cpu_usage = [snapshot.cpu_usage for snapshot in self.performance_history]
            memory_usage = [snapshot.memory_usage for snapshot in self.performance_history]
            gpu_usage = [snapshot.gpu_usage for snapshot in self.performance_history]
            gpu_memory = [snapshot.gpu_memory for snapshot in self.performance_history]
            inference_times = [snapshot.inference_time for snapshot in self.performance_history if snapshot.inference_time > 0]
            throughputs = [snapshot.throughput for snapshot in self.performance_history if snapshot.throughput > 0]
            latencies = [snapshot.latency for snapshot in self.performance_history if snapshot.latency > 0]
            
            return PerformanceStatistics(
                avg_cpu_usage=np.mean(cpu_usage),
                avg_memory_usage=np.mean(memory_usage),
                avg_gpu_usage=np.mean(gpu_usage),
                avg_gpu_memory=np.mean(gpu_memory),
                avg_inference_time=np.mean(inference_times) if inference_times else 0.0,
                avg_throughput=np.mean(throughputs) if throughputs else 0.0,
                avg_latency=np.mean(latencies) if latencies else 0.0,
                max_cpu_usage=np.max(cpu_usage),
                max_memory_usage=np.max(memory_usage),
                max_gpu_usage=np.max(gpu_usage),
                max_gpu_memory=np.max(gpu_memory),
                min_inference_time=np.min(inference_times) if inference_times else 0.0,
                max_inference_time=np.max(inference_times) if inference_times else 0.0,
                std_inference_time=np.std(inference_times) if inference_times else 0.0,
                std_throughput=np.std(throughputs) if throughputs else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Performance statistics calculation failed: {e}")
            return PerformanceStatistics(
                avg_cpu_usage=0.0, avg_memory_usage=0.0, avg_gpu_usage=0.0, avg_gpu_memory=0.0,
                avg_inference_time=0.0, avg_throughput=0.0, avg_latency=0.0,
                max_cpu_usage=0.0, max_memory_usage=0.0, max_gpu_usage=0.0, max_gpu_memory=0.0,
                min_inference_time=0.0, max_inference_time=0.0, std_inference_time=0.0, std_throughput=0.0
            )
    
    def set_optimization_baseline(self, model_id: str, baseline_metrics: Dict[str, Any]):
        """Set optimization baseline for a model."""
        try:
            self.optimization_baselines[model_id] = {
                'baseline_metrics': baseline_metrics,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"Optimization baseline set for model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to set optimization baseline: {e}")
    
    def calculate_optimization_improvement(self, model_id: str, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate improvement over optimization baseline."""
        try:
            if model_id not in self.optimization_baselines:
                return {'error': 'No baseline found for model'}
            
            baseline = self.optimization_baselines[model_id]['baseline_metrics']
            
            improvements = {}
            
            # Calculate improvements for each metric
            for metric in ['avg_inference_time', 'throughput', 'latency', 'memory_peak']:
                if metric in baseline and metric in current_metrics:
                    baseline_val = baseline[metric]
                    current_val = current_metrics[metric]
                    
                    if baseline_val > 0:
                        if metric == 'avg_inference_time' or metric == 'latency':
                            # Lower is better
                            improvement = (baseline_val - current_val) / baseline_val
                        else:
                            # Higher is better
                            improvement = (current_val - baseline_val) / baseline_val
                        
                        improvements[metric] = improvement
            
            # Calculate overall improvement
            if improvements:
                improvements['overall_improvement'] = np.mean(list(improvements.values()))
            
            return improvements
            
        except Exception as e:
            self.logger.error(f"Optimization improvement calculation failed: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self, filepath: str) -> bool:
        """Export performance data to file."""
        try:
            export_data = {
                'system_info': self.system_info,
                'performance_history': [
                    {
                        'timestamp': snapshot.timestamp.isoformat(),
                        'cpu_usage': snapshot.cpu_usage,
                        'memory_usage': snapshot.memory_usage,
                        'gpu_usage': snapshot.gpu_usage,
                        'gpu_memory': snapshot.gpu_memory,
                        'inference_time': snapshot.inference_time,
                        'throughput': snapshot.throughput,
                        'latency': snapshot.latency
                    }
                    for snapshot in self.performance_history
                ],
                'benchmark_results': self.benchmark_results,
                'optimization_baselines': self.optimization_baselines,
                'export_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Performance data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Performance data export failed: {e}")
            return False
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.current_metrics.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return self.system_info.copy()
    
    def clear_history(self):
        """Clear performance history."""
        self.performance_history.clear()
        self.benchmark_results.clear()
        self.optimization_baselines.clear()
        self.logger.info("Performance history cleared")
    
    def get_history_size(self) -> int:
        """Get current history size."""
        return len(self.performance_history)
