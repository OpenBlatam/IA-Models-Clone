"""
Monitoring System
Comprehensive monitoring and metrics collection for the TruthGPT system
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import psutil
import torch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: float
    gpu_memory_total: float
    timestamp: float

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    inference_time: float
    tokens_per_second: float
    memory_usage: float
    cache_hit_rate: float
    timestamp: float

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    epoch_time: float
    timestamp: float

class MetricsCollector:
    """Collects and manages various metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics = deque(maxlen=max_history)
        self.model_metrics = deque(maxlen=max_history)
        self.training_metrics = deque(maxlen=max_history)
        
        # Custom metrics
        self.custom_metrics = defaultdict(lambda: deque(maxlen=max_history))
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds
        
        logger.info("MetricsCollector initialized")
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous system monitoring"""
        self.monitor_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self.collect_system_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def collect_system_metrics(self) -> None:
        """Collect current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU metrics
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                timestamp=time.time()
            )
            
            self.system_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def record_model_metrics(self, 
                            inference_time: float,
                            tokens_per_second: float,
                            memory_usage: float = 0.0,
                            cache_hit_rate: float = 0.0) -> None:
        """Record model performance metrics"""
        metrics = ModelMetrics(
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            memory_usage=memory_usage,
            cache_hit_rate=cache_hit_rate,
            timestamp=time.time()
        )
        
        self.model_metrics.append(metrics)
    
    def record_training_metrics(self,
                               epoch: int,
                               train_loss: float,
                               val_loss: float,
                               learning_rate: float,
                               epoch_time: float) -> None:
        """Record training performance metrics"""
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            epoch_time=epoch_time,
            timestamp=time.time()
        )
        
        self.training_metrics.append(metrics)
    
    def record_custom_metric(self, name: str, value: float, metadata: Optional[Dict] = None) -> None:
        """Record a custom metric"""
        metric_data = {
            'value': value,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.custom_metrics[name].append(metric_data)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary"""
        if not self.system_metrics:
            return {}
        
        recent_metrics = list(self.system_metrics)[-100:]  # Last 100 measurements
        
        return {
            'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
            'max_cpu_percent': np.max([m.cpu_percent for m in recent_metrics]),
            'avg_memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
            'max_memory_percent': np.max([m.memory_percent for m in recent_metrics]),
            'avg_gpu_memory_used': np.mean([m.gpu_memory_used for m in recent_metrics]),
            'max_gpu_memory_used': np.max([m.gpu_memory_used for m in recent_metrics]),
            'gpu_memory_total': recent_metrics[-1].gpu_memory_total if recent_metrics else 0,
            'measurement_count': len(recent_metrics)
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model performance summary"""
        if not self.model_metrics:
            return {}
        
        recent_metrics = list(self.model_metrics)[-100:]  # Last 100 measurements
        
        return {
            'avg_inference_time': np.mean([m.inference_time for m in recent_metrics]),
            'avg_tokens_per_second': np.mean([m.tokens_per_second for m in recent_metrics]),
            'max_tokens_per_second': np.max([m.tokens_per_second for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_cache_hit_rate': np.mean([m.cache_hit_rate for m in recent_metrics]),
            'total_inferences': len(recent_metrics)
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training performance summary"""
        if not self.training_metrics:
            return {}
        
        return {
            'total_epochs': len(self.training_metrics),
            'final_train_loss': self.training_metrics[-1].train_loss,
            'final_val_loss': self.training_metrics[-1].val_loss,
            'best_val_loss': min([m.val_loss for m in self.training_metrics]),
            'avg_epoch_time': np.mean([m.epoch_time for m in self.training_metrics]),
            'total_training_time': sum([m.epoch_time for m in self.training_metrics])
        }
    
    def export_metrics(self, filepath: str) -> None:
        """Export all metrics to JSON file"""
        try:
            data = {
                'system_metrics': [asdict(m) for m in self.system_metrics],
                'model_metrics': [asdict(m) for m in self.model_metrics],
                'training_metrics': [asdict(m) for m in self.training_metrics],
                'custom_metrics': {
                    name: list(metrics) for name, metrics in self.custom_metrics.items()
                },
                'export_timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

class MonitoringSystem:
    """Main monitoring system that coordinates all monitoring activities"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector(
            max_history=self.config.get('max_history', 1000)
        )
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'gpu_memory_percent': 90.0
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info("MonitoringSystem initialized")
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start the monitoring system"""
        self.metrics_collector.start_monitoring(interval)
        logger.info("Monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        self.metrics_collector.stop_monitoring()
        logger.info("Monitoring system stopped")
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """Add an alert callback function"""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self) -> None:
        """Check for alert conditions"""
        if not self.metrics_collector.system_metrics:
            return
        
        latest_metrics = self.metrics_collector.system_metrics[-1]
        
        # Check CPU alert
        if latest_metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            self._trigger_alert('high_cpu', {
                'cpu_percent': latest_metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']
            })
        
        # Check memory alert
        if latest_metrics.memory_percent > self.alert_thresholds['memory_percent']:
            self._trigger_alert('high_memory', {
                'memory_percent': latest_metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_percent']
            })
        
        # Check GPU memory alert
        if latest_metrics.gpu_memory_total > 0:
            gpu_memory_percent = (latest_metrics.gpu_memory_used / latest_metrics.gpu_memory_total) * 100
            if gpu_memory_percent > self.alert_thresholds['gpu_memory_percent']:
                self._trigger_alert('high_gpu_memory', {
                    'gpu_memory_percent': gpu_memory_percent,
                    'threshold': self.alert_thresholds['gpu_memory_percent']
                })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger an alert"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Alert triggered: {alert_type} - {data}")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report"""
        return {
            'system': self.metrics_collector.get_system_summary(),
            'model': self.metrics_collector.get_model_summary(),
            'training': self.metrics_collector.get_training_summary(),
            'timestamp': time.time()
        }
    
    def export_report(self, filepath: str) -> None:
        """Export comprehensive report to file"""
        report = self.get_comprehensive_report()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export report: {e}")

