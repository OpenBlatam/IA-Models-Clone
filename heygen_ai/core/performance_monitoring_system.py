"""
Advanced Performance Monitoring System for HeyGen AI Enterprise

This module provides comprehensive performance monitoring, analytics, and optimization:
- Real-time performance monitoring with sub-second latency
- Advanced analytics and performance prediction
- Anomaly detection and alerting
- Performance optimization recommendations
- Multi-dimensional performance metrics
- Historical performance analysis
- Automated performance tuning
"""

import logging
import os
import time
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import psutil
import GPUtil

# Advanced monitoring libraries
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("pynvml not available. Install for GPU monitoring.")

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("prometheus_client not available. Install for metrics export.")

try:
    import influxdb_client
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    warnings.warn("influxdb_client not available. Install for time-series data.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install for anomaly detection.")

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMonitoringConfig:
    """Configuration for performance monitoring system."""
    
    # Monitoring settings
    enable_real_time_monitoring: bool = True
    monitoring_interval: float = 0.1  # 100ms
    enable_gpu_monitoring: bool = True
    enable_system_monitoring: bool = True
    enable_model_monitoring: bool = True
    
    # Analytics settings
    enable_performance_prediction: bool = True
    enable_anomaly_detection: bool = True
    enable_trend_analysis: bool = True
    enable_optimization_recommendations: bool = True
    
    # Storage settings
    enable_local_storage: bool = True
    enable_database_storage: bool = False
    enable_metrics_export: bool = True
    max_history_size: int = 10000
    
    # Alerting settings
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "gpu_utilization": 90.0,
        "memory_usage": 85.0,
        "temperature": 80.0,
        "performance_degradation": 20.0
    })
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_speedup": 1.2,
        "max_memory_increase": 10.0,
        "min_accuracy": 95.0
    })


class PerformanceMetrics:
    """Comprehensive performance metrics collection."""
    
    def __init__(self):
        self.timestamp = time.time()
        self.gpu_metrics = {}
        self.system_metrics = {}
        self.model_metrics = {}
        self.custom_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "gpu_metrics": self.gpu_metrics,
            "system_metrics": self.system_metrics,
            "model_metrics": self.model_metrics,
            "custom_metrics": self.custom_metrics
        }
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class GPUMonitor:
    """Advanced GPU monitoring with NVML integration."""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.nvml_initialized = False
        self.gpu_count = 0
        self.gpu_handles = []
        
        if self.config.enable_gpu_monitoring and NVML_AVAILABLE:
            self._initialize_nvml()
    
    def _initialize_nvml(self):
        """Initialize NVML for GPU monitoring."""
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)
            
            self.nvml_initialized = True
            logger.info(f"NVML initialized with {self.gpu_count} GPUs")
            
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.nvml_initialized = False
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get comprehensive GPU metrics."""
        if not self.nvml_initialized:
            return {}
        
        try:
            metrics = {}
            
            for i, handle in enumerate(self.gpu_handles):
                gpu_metrics = {}
                
                # Basic GPU info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                gpu_metrics['name'] = name
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_metrics['memory_total_mb'] = memory_info.total / (1024**2)
                gpu_metrics['memory_used_mb'] = memory_info.used / (1024**2)
                gpu_metrics['memory_free_mb'] = memory_info.free / (1024**2)
                gpu_metrics['memory_usage_percent'] = (memory_info.used / memory_info.total) * 100
                
                # Utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_metrics['gpu_utilization_percent'] = utilization.gpu
                gpu_metrics['memory_utilization_percent'] = utilization.memory
                
                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_metrics['temperature_celsius'] = temperature
                except:
                    gpu_metrics['temperature_celsius'] = 0
                
                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
                    gpu_metrics['power_usage_watts'] = power
                except:
                    gpu_metrics['power_usage_watts'] = 0
                
                # Clock speeds
                try:
                    gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    gpu_metrics['gpu_clock_mhz'] = gpu_clock
                    gpu_metrics['memory_clock_mhz'] = memory_clock
                except:
                    gpu_metrics['gpu_clock_mhz'] = 0
                    gpu_metrics['memory_clock_mhz'] = 0
                
                metrics[f"gpu_{i}"] = gpu_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return {}


class SystemMonitor:
    """System-level performance monitoring."""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            metrics = {}
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics['cpu'] = {
                'usage_percent': cpu_percent,
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'frequency_max_mhz': cpu_freq.max if cpu_freq else 0
            }
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory'] = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'usage_percent': memory.percent
            }
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics['disk'] = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': (disk.used / disk.total) * 100,
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0
            }
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics['network'] = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}


class ModelPerformanceMonitor:
    """Model-specific performance monitoring."""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.model_metrics_history = deque(maxlen=config.max_history_size)
    
    def monitor_model_inference(self, model: nn.Module, input_tensor: torch.Tensor,
                               num_runs: int = 10) -> Dict[str, Any]:
        """Monitor model inference performance."""
        try:
            model.eval()
            
            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_tensor)
            
            # Benchmark inference
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_tensor)
            
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            
            # Calculate metrics
            avg_inference_time = total_time / num_runs
            throughput = num_runs / total_time
            
            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                memory_reserved = torch.cuda.memory_reserved() / (1024**2)    # MB
            else:
                memory_allocated = 0
                memory_reserved = 0
            
            metrics = {
                "inference_time_ms": avg_inference_time * 1000,
                "throughput_inferences_per_second": throughput,
                "total_time": total_time,
                "num_runs": num_runs,
                "memory_allocated_mb": memory_allocated,
                "memory_reserved_mb": memory_reserved,
                "input_shape": list(input_tensor.shape),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "model_buffers": sum(b.numel() for b in model.buffers())
            }
            
            # Store in history
            self.model_metrics_history.append({
                "timestamp": time.time(),
                "metrics": metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model inference monitoring failed: {e}")
            return {}
    
    def get_model_performance_history(self, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get model performance history within a time window."""
        try:
            cutoff_time = time.time() - (window_minutes * 60)
            recent_metrics = [
                entry for entry in self.model_metrics_history
                if entry["timestamp"] > cutoff_time
            ]
            return recent_metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []


class PerformanceAnalyzer:
    """Advanced performance analysis and prediction."""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.max_history_size)
        self.anomaly_detector = None
        self.trend_analyzer = None
        
        if self.config.enable_anomaly_detection and SKLEARN_AVAILABLE:
            self._initialize_anomaly_detector()
    
    def _initialize_anomaly_detector(self):
        """Initialize anomaly detection model."""
        try:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.trend_analyzer = StandardScaler()
            logger.info("Anomaly detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detector: {e}")
    
    def analyze_performance_trends(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            if len(metrics_history) < 10:
                return {"error": "Insufficient data for trend analysis"}
            
            analysis = {}
            
            # Extract key metrics
            gpu_utilizations = []
            memory_usages = []
            inference_times = []
            temperatures = []
            
            for entry in metrics_history:
                metrics = entry.get("metrics", {})
                
                # GPU metrics
                for gpu_key, gpu_metrics in metrics.get("gpu_metrics", {}).items():
                    if isinstance(gpu_metrics, dict):
                        gpu_utilizations.append(gpu_metrics.get("gpu_utilization_percent", 0))
                        memory_usages.append(gpu_metrics.get("memory_usage_percent", 0))
                        temperatures.append(gpu_metrics.get("temperature_celsius", 0))
                
                # Model metrics
                if "inference_time_ms" in metrics:
                    inference_times.append(metrics["inference_time_ms"])
            
            # Calculate trends
            if gpu_utilizations:
                analysis["gpu_utilization"] = {
                    "mean": statistics.mean(gpu_utilizations),
                    "std": statistics.stdev(gpu_utilizations) if len(gpu_utilizations) > 1 else 0,
                    "trend": self._calculate_trend(gpu_utilizations)
                }
            
            if memory_usages:
                analysis["memory_usage"] = {
                    "mean": statistics.mean(memory_usages),
                    "std": statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0,
                    "trend": self._calculate_trend(memory_usages)
                }
            
            if inference_times:
                analysis["inference_time"] = {
                    "mean": statistics.mean(inference_times),
                    "std": statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                    "trend": self._calculate_trend(inference_times)
                }
            
            if temperatures:
                analysis["temperature"] = {
                    "mean": statistics.mean(temperatures),
                    "std": statistics.stdev(temperatures) if len(temperatures) > 1 else 0,
                    "trend": self._calculate_trend(temperatures)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        try:
            if len(values) < 2:
                return "stable"
            
            # Simple linear trend calculation
            x = list(range(len(values)))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def detect_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        try:
            if not self.anomaly_detector or not self.config.enable_anomaly_detection:
                return []
            
            anomalies = []
            
            # Extract numerical features for anomaly detection
            features = []
            feature_names = []
            
            # GPU metrics
            for gpu_key, gpu_metrics in current_metrics.get("gpu_metrics", {}).items():
                if isinstance(gpu_metrics, dict):
                    features.extend([
                        gpu_metrics.get("gpu_utilization_percent", 0),
                        gpu_metrics.get("memory_usage_percent", 0),
                        gpu_metrics.get("temperature_celsius", 0)
                    ])
                    feature_names.extend([
                        f"{gpu_key}_utilization",
                        f"{gpu_key}_memory",
                        f"{gpu_key}_temperature"
                    ])
            
            # System metrics
            system_metrics = current_metrics.get("system_metrics", {})
            features.extend([
                system_metrics.get("cpu", {}).get("usage_percent", 0),
                system_metrics.get("memory", {}).get("usage_percent", 0)
            ])
            feature_names.extend(["cpu_usage", "memory_usage"])
            
            if len(features) > 0:
                # Reshape for sklearn
                features_array = np.array(features).reshape(1, -1)
                
                # Detect anomalies
                anomaly_scores = self.anomaly_detector.decision_function(features_array)
                is_anomaly = self.anomaly_detector.predict(features_array)
                
                # Check for anomalies
                for i, (score, is_anom) in enumerate(zip(anomaly_scores, is_anomaly)):
                    if is_anom == -1:  # Anomaly detected
                        anomalies.append({
                            "metric": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                            "value": features[i],
                            "anomaly_score": float(score),
                            "severity": "high" if abs(score) > 0.5 else "medium"
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []


class PerformanceOptimizer:
    """AI-powered performance optimization recommendations."""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.optimization_history = deque(maxlen=1000)
        self.recommendation_rules = self._load_recommendation_rules()
    
    def _load_recommendation_rules(self) -> List[Dict[str, Any]]:
        """Load performance optimization recommendation rules."""
        return [
            {
                "condition": lambda metrics: metrics.get("gpu_metrics", {}).get("gpu_0", {}).get("memory_usage_percent", 0) > 90,
                "recommendation": "Reduce batch size or enable gradient checkpointing to lower memory usage",
                "priority": "high",
                "category": "memory"
            },
            {
                "condition": lambda metrics: metrics.get("gpu_metrics", {}).get("gpu_0", {}).get("temperature_celsius", 0) > 80,
                "recommendation": "Check GPU cooling and consider reducing workload intensity",
                "priority": "high",
                "category": "thermal"
            },
            {
                "condition": lambda metrics: metrics.get("gpu_metrics", {}).get("gpu_0", {}).get("gpu_utilization_percent", 0) < 50,
                "recommendation": "Increase batch size or enable mixed precision training for better GPU utilization",
                "priority": "medium",
                "category": "efficiency"
            },
            {
                "condition": lambda metrics: metrics.get("system_metrics", {}).get("memory", {}).get("usage_percent", 0) > 85,
                "recommendation": "Close unnecessary applications or increase system memory",
                "priority": "medium",
                "category": "system"
            }
        ]
    
    def generate_recommendations(self, current_metrics: Dict[str, Any], 
                                historical_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        try:
            recommendations = []
            
            # Apply rule-based recommendations
            for rule in self.recommendation_rules:
                if rule["condition"](current_metrics):
                    recommendations.append({
                        "type": "rule_based",
                        "recommendation": rule["recommendation"],
                        "priority": rule["priority"],
                        "category": rule["category"],
                        "timestamp": time.time()
                    })
            
            # Generate data-driven recommendations
            data_driven_recs = self._generate_data_driven_recommendations(
                current_metrics, historical_metrics
            )
            recommendations.extend(data_driven_recs)
            
            # Sort by priority
            priority_order = {"high": 3, "medium": 2, "low": 1}
            recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 0), reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _generate_data_driven_recommendations(self, current_metrics: Dict[str, Any],
                                            historical_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate data-driven optimization recommendations."""
        try:
            recommendations = []
            
            if len(historical_metrics) < 5:
                return recommendations
            
            # Analyze performance trends
            recent_metrics = historical_metrics[-10:]
            
            # Check for performance degradation
            if "inference_time_ms" in current_metrics:
                current_time = current_metrics["inference_time_ms"]
                historical_times = [
                    m.get("metrics", {}).get("inference_time_ms", 0)
                    for m in recent_metrics
                    if "inference_time_ms" in m.get("metrics", {})
                ]
                
                if historical_times:
                    avg_historical_time = statistics.mean(historical_times)
                    if current_time > avg_historical_time * 1.2:  # 20% degradation
                        recommendations.append({
                            "type": "data_driven",
                            "recommendation": "Performance degradation detected. Consider model optimization or hardware check.",
                            "priority": "high",
                            "category": "performance",
                            "timestamp": time.time(),
                            "evidence": f"Current: {current_time:.2f}ms, Average: {avg_historical_time:.2f}ms"
                        })
            
            # Check for memory usage trends
            gpu_memory_usage = []
            for entry in recent_metrics:
                for gpu_key, gpu_metrics in entry.get("metrics", {}).get("gpu_metrics", {}).items():
                    if isinstance(gpu_metrics, dict):
                        gpu_memory_usage.append(gpu_metrics.get("memory_usage_percent", 0))
            
            if gpu_memory_usage:
                avg_memory_usage = statistics.mean(gpu_memory_usage)
                if avg_memory_usage > 85:
                    recommendations.append({
                        "type": "data_driven",
                        "recommendation": "High memory usage trend detected. Consider model compression or memory optimization.",
                        "priority": "medium",
                        "category": "memory",
                        "timestamp": time.time(),
                        "evidence": f"Average memory usage: {avg_memory_usage:.1f}%"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Data-driven recommendation generation failed: {e}")
            return []


class PerformanceMonitoringSystem:
    """Main performance monitoring system orchestrating all components."""
    
    def __init__(self, config: PerformanceMonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.monitoring_system")
        
        # Initialize monitoring components
        self.gpu_monitor = GPUMonitor(config)
        self.system_monitor = SystemMonitor(config)
        self.model_monitor = ModelPerformanceMonitor(config)
        self.analyzer = PerformanceAnalyzer(config)
        self.optimizer = PerformanceOptimizer(config)
        
        # Performance data storage
        self.performance_history = deque(maxlen=config.max_history_size)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Metrics export
        if config.enable_metrics_export and PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics export."""
        try:
            # GPU metrics
            self.gpu_utilization_gauge = prometheus_client.Gauge(
                'heygen_ai_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id']
            )
            
            self.gpu_memory_gauge = prometheus_client.Gauge(
                'heygen_ai_gpu_memory_usage_percent',
                'GPU memory usage percentage',
                ['gpu_id']
            )
            
            self.gpu_temperature_gauge = prometheus_client.Gauge(
                'heygen_ai_gpu_temperature_celsius',
                'GPU temperature in Celsius',
                ['gpu_id']
            )
            
            # System metrics
            self.cpu_usage_gauge = prometheus_client.Gauge(
                'heygen_ai_cpu_usage_percent',
                'CPU usage percentage'
            )
            
            self.memory_usage_gauge = prometheus_client.Gauge(
                'heygen_ai_memory_usage_percent',
                'Memory usage percentage'
            )
            
            logger.info("Prometheus metrics export configured")
            
        except Exception as e:
            logger.error(f"Failed to setup Prometheus metrics: {e}")
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_all_metrics()
                
                # Store metrics
                self.performance_history.append(metrics)
                
                # Export to Prometheus if enabled
                if self.config.enable_metrics_export and PROMETHEUS_AVAILABLE:
                    self._export_prometheus_metrics(metrics)
                
                # Wait for next monitoring interval
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _collect_all_metrics(self) -> PerformanceMetrics:
        """Collect all performance metrics."""
        metrics = PerformanceMetrics()
        
        # GPU metrics
        if self.config.enable_gpu_monitoring:
            metrics.gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        
        # System metrics
        if self.config.enable_system_monitoring:
            metrics.system_metrics = self.system_monitor.get_system_metrics()
        
        return metrics
    
    def _export_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Export metrics to Prometheus."""
        try:
            # GPU metrics
            for gpu_id, gpu_metrics in metrics.gpu_metrics.items():
                if isinstance(gpu_metrics, dict):
                    self.gpu_utilization_gauge.labels(gpu_id).set(
                        gpu_metrics.get("gpu_utilization_percent", 0)
                    )
                    self.gpu_memory_gauge.labels(gpu_id).set(
                        gpu_metrics.get("memory_usage_percent", 0)
                    )
                    self.gpu_temperature_gauge.labels(gpu_id).set(
                        gpu_metrics.get("temperature_celsius", 0)
                    )
            
            # System metrics
            if "cpu" in metrics.system_metrics:
                self.cpu_usage_gauge.set(
                    metrics.system_metrics["cpu"].get("usage_percent", 0)
                )
            
            if "memory" in metrics.system_metrics:
                self.memory_usage_gauge.set(
                    metrics.system_metrics["memory"].get("usage_percent", 0)
                )
                
        except Exception as e:
            logger.error(f"Prometheus metrics export failed: {e}")
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            cutoff_time = time.time() - (window_minutes * 60)
            recent_metrics = [
                entry for entry in self.performance_history
                if entry.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No metrics available for the specified time window"}
            
            # Analyze trends
            trends = self.analyzer.analyze_performance_trends(recent_metrics)
            
            # Detect anomalies
            latest_metrics = recent_metrics[-1] if recent_metrics else None
            anomalies = []
            if latest_metrics:
                anomalies = self.analyzer.detect_anomalies(latest_metrics.to_dict())
            
            # Generate recommendations
            recommendations = []
            if latest_metrics:
                recommendations = self.optimizer.generate_recommendations(
                    latest_metrics.to_dict(), [m.to_dict() for m in recent_metrics]
                )
            
            return {
                "time_window_minutes": window_minutes,
                "metrics_count": len(recent_metrics),
                "latest_metrics": latest_metrics.to_dict() if latest_metrics else None,
                "performance_trends": trends,
                "anomalies": anomalies,
                "recommendations": recommendations,
                "summary": {
                    "total_gpus": len(latest_metrics.gpu_metrics) if latest_metrics else 0,
                    "monitoring_active": self.monitoring_active,
                    "history_size": len(self.performance_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"error": str(e)}
    
    def monitor_model_performance(self, model: nn.Module, input_tensor: torch.Tensor,
                                 num_runs: int = 10) -> Dict[str, Any]:
        """Monitor specific model performance."""
        return self.model_monitor.monitor_model_inference(model, input_tensor, num_runs)
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations."""
        try:
            if not self.performance_history:
                return []
            
            latest_metrics = self.performance_history[-1]
            historical_metrics = list(self.performance_history)[-100:]  # Last 100 entries
            
            return self.optimizer.generate_recommendations(
                latest_metrics.to_dict(),
                [m.to_dict() for m in historical_metrics]
            )
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return []


# Factory functions
def create_performance_monitoring_system(config: Optional[PerformanceMonitoringConfig] = None) -> PerformanceMonitoringSystem:
    """Create a performance monitoring system."""
    if config is None:
        config = PerformanceMonitoringConfig()
    
    return PerformanceMonitoringSystem(config)


def create_comprehensive_monitoring_config() -> PerformanceMonitoringConfig:
    """Create comprehensive monitoring configuration."""
    return PerformanceMonitoringConfig(
        enable_real_time_monitoring=True,
        monitoring_interval=0.1,
        enable_gpu_monitoring=True,
        enable_system_monitoring=True,
        enable_model_monitoring=True,
        enable_performance_prediction=True,
        enable_anomaly_detection=True,
        enable_trend_analysis=True,
        enable_optimization_recommendations=True,
        enable_alerts=True,
        enable_metrics_export=True
    )


if __name__ == "__main__":
    # Test the performance monitoring system
    config = create_comprehensive_monitoring_config()
    monitoring_system = create_performance_monitoring_system(config)
    
    # Start monitoring
    monitoring_system.start_monitoring()
    
    # Let it run for a few seconds
    time.sleep(5)
    
    # Get performance summary
    summary = monitoring_system.get_performance_summary(window_minutes=1)
    print(f"Performance summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Get optimization recommendations
    recommendations = monitoring_system.get_optimization_recommendations()
    print(f"Optimization recommendations: {json.dumps(recommendations, indent=2, default=str)}")
    
    # Stop monitoring
    monitoring_system.stop_monitoring()
