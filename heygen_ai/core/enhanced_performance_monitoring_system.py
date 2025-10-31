#!/usr/bin/env python3
"""
Enhanced Performance Monitoring System for Advanced Distributed AI
Integrated with the refactored architecture for comprehensive monitoring
"""

import logging
import time
import json
import psutil
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import numpy as np
from collections import deque, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== ENHANCED ENUMS =====

class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    QUANTUM = "quantum"
    NEUROMORPHIC = "neuromorphic"
    HYBRID = "hybrid"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MonitoringMode(Enum):
    """Monitoring operation modes."""
    PASSIVE = "passive"
    ACTIVE = "active"
    INTELLIGENT = "intelligent"
    ADAPTIVE = "adaptive"

# ===== ENHANCED CONFIGURATION =====

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    enabled: bool = True
    monitoring_mode: MonitoringMode = MonitoringMode.INTELLIGENT
    sampling_interval: float = 1.0  # seconds
    retention_period: int = 3600  # seconds
    max_metrics: int = 10000
    enable_alerts: bool = True
    enable_anomaly_detection: bool = True
    enable_auto_optimization: bool = True

@dataclass
class AlertConfig:
    """Configuration for alerting system."""
    enabled: bool = True
    severity_threshold: AlertSeverity = AlertSeverity.WARNING
    notification_channels: List[str] = field(default_factory=lambda: ["console", "log"])
    alert_cooldown: int = 300  # seconds
    max_alerts_per_hour: int = 100

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    enabled: bool = True
    algorithm: str = "isolation_forest"
    sensitivity: float = 0.8
    window_size: int = 100
    min_samples: int = 10

# ===== ABSTRACT BASE CLASSES =====

class BaseMonitor(ABC):
    """Abstract base class for monitoring components."""
    
    def __init__(self, name: str, config: PerformanceConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.enabled = config.enabled
        self.metrics = {}
        self.last_update = time.time()
    
    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect monitoring metrics."""
        pass
    
    @abstractmethod
    def process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze collected metrics."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self.enabled
    
    def get_last_update(self) -> float:
        """Get timestamp of last update."""
        return self.last_update

class BaseAlertManager(ABC):
    """Abstract base class for alert management."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AlertManager")
        self.alerts = deque(maxlen=1000)
        self.alert_history = defaultdict(int)
        self.last_alert_time = {}
    
    @abstractmethod
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        pass
    
    @abstractmethod
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send an alert."""
        pass
    
    def can_send_alert(self, alert_key: str) -> bool:
        """Check if an alert can be sent (cooldown)."""
        if alert_key not in self.last_alert_time:
            return True
        
        time_since_last = time.time() - self.last_alert_time[alert_key]
        return time_since_last >= self.config.alert_cooldown

# ===== CONCRETE MONITORING IMPLEMENTATIONS =====

class SystemResourceMonitor(BaseMonitor):
    """Monitor system resources (CPU, memory, disk, network)."""
    
    def __init__(self, config: PerformanceConfig):
        super().__init__("SystemResourceMonitor", config)
        self.metric_history = defaultdict(lambda: deque(maxlen=100))
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            metrics = {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else 0
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "percent": memory.percent,
                    "swap_percent": swap.percent
                },
                "disk": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "percent": disk_usage.percent
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                }
            }
            
            # Store in history
            for key, value in self._flatten_metrics(metrics).items():
                self.metric_history[key].append(value)
            
            self.last_update = time.time()
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process system metrics for analysis."""
        if "error" in metrics:
            return metrics
        
        processed = {
            "timestamp": metrics["timestamp"],
            "summary": {},
            "trends": {},
            "anomalies": {}
        }
        
        # Calculate summary statistics
        for key, value in self._flatten_metrics(metrics).items():
            if key in self.metric_history and len(self.metric_history[key]) > 0:
                history = list(self.metric_history[key])
                processed["summary"][key] = {
                    "current": value,
                    "average": np.mean(history),
                    "min": np.min(history),
                    "max": np.max(history),
                    "trend": self._calculate_trend(history)
                }
        
        return processed
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested metrics dictionary."""
        flattened = {}
        for key, value in metrics.items():
            if key == "timestamp":
                continue
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics(value, new_key))
            else:
                flattened[new_key] = value
        return flattened
    
    def _calculate_trend(self, history: List[float]) -> str:
        """Calculate trend direction from history."""
        if len(history) < 2:
            return "stable"
        
        recent = history[-5:] if len(history) >= 5 else history
        if len(recent) < 2:
            return "stable"
        
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

class QuantumPerformanceMonitor(BaseMonitor):
    """Monitor quantum computing performance metrics."""
    
    def __init__(self, config: PerformanceConfig):
        super().__init__("QuantumPerformanceMonitor", config)
        self.quantum_metrics = {}
        self.circuit_performance = defaultdict(list)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect quantum performance metrics."""
        try:
            metrics = {
                "timestamp": time.time(),
                "quantum_system": {
                    "active_qubits": self._get_active_qubits(),
                    "circuit_depth": self._get_circuit_depth(),
                    "error_rates": self._get_error_rates(),
                    "quantum_advantage": self._calculate_quantum_advantage(),
                    "optimization_level": self._get_optimization_level()
                },
                "performance": {
                    "circuits_executed": len(self.circuit_performance),
                    "average_execution_time": self._calculate_average_execution_time(),
                    "success_rate": self._calculate_success_rate(),
                    "quantum_volume": self._calculate_quantum_volume()
                }
            }
            
            self.last_update = time.time()
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect quantum metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process quantum metrics for analysis."""
        if "error" in metrics:
            return metrics
        
        processed = {
            "timestamp": metrics["timestamp"],
            "quantum_analysis": {},
            "performance_insights": {},
            "optimization_recommendations": []
        }
        
        # Analyze quantum system performance
        quantum_system = metrics["quantum_system"]
        processed["quantum_analysis"] = {
            "qubit_efficiency": quantum_system["active_qubits"] / max(quantum_system["active_qubits"], 1),
            "error_tolerance": 1.0 - quantum_system["error_rates"],
            "advantage_ratio": quantum_system["quantum_advantage"]
        }
        
        # Generate optimization recommendations
        if quantum_system["error_rates"] > 0.1:
            processed["optimization_recommendations"].append({
                "type": "error_mitigation",
                "priority": "high",
                "description": "High error rates detected, consider enabling error mitigation techniques"
            })
        
        if quantum_system["quantum_advantage"] < 1.0:
            processed["optimization_recommendations"].append({
                "type": "quantum_advantage",
                "priority": "medium",
                "description": "Quantum advantage not achieved, consider circuit optimization"
            })
        
        return processed
    
    def _get_active_qubits(self) -> int:
        """Get number of active qubits."""
        return self.quantum_metrics.get("active_qubits", 0)
    
    def _get_circuit_depth(self) -> int:
        """Get current circuit depth."""
        return self.quantum_metrics.get("circuit_depth", 0)
    
    def _get_error_rates(self) -> float:
        """Get current error rates."""
        return self.quantum_metrics.get("error_rates", 0.0)
    
    def _calculate_quantum_advantage(self) -> float:
        """Calculate quantum advantage ratio."""
        return self.quantum_metrics.get("quantum_advantage", 1.0)
    
    def _get_optimization_level(self) -> int:
        """Get current optimization level."""
        return self.quantum_metrics.get("optimization_level", 1)
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average circuit execution time."""
        if not self.circuit_performance:
            return 0.0
        times = [perf["execution_time"] for perf in self.circuit_performance.values()]
        return np.mean(times) if times else 0.0
    
    def _calculate_success_rate(self) -> float:
        """Calculate circuit success rate."""
        if not self.circuit_performance:
            return 0.0
        successful = sum(1 for perf in self.circuit_performance.values() if perf.get("success", False))
        return successful / len(self.circuit_performance) if self.circuit_performance else 0.0
    
    def _calculate_quantum_volume(self) -> float:
        """Calculate quantum volume metric."""
        # Simplified quantum volume calculation
        qubits = self._get_active_qubits()
        depth = self._get_circuit_depth()
        error_rate = self._get_error_rates()
        
        if error_rate == 0:
            return qubits * depth
        
        return qubits * depth * (1 - error_rate)

class NeuromorphicPerformanceMonitor(BaseMonitor):
    """Monitor neuromorphic computing performance metrics."""
    
    def __init__(self, config: PerformanceConfig):
        super().__init__("NeuromorphicPerformanceMonitor", config)
        self.neuromorphic_metrics = {}
        self.network_performance = defaultdict(list)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect neuromorphic performance metrics."""
        try:
            metrics = {
                "timestamp": time.time(),
                "neuromorphic_system": {
                    "active_neurons": self._get_active_neurons(),
                    "network_connectivity": self._get_network_connectivity(),
                    "plasticity_rate": self._get_plasticity_rate(),
                    "spiking_frequency": self._get_spiking_frequency(),
                    "learning_efficiency": self._calculate_learning_efficiency()
                },
                "performance": {
                    "networks_active": len(self.network_performance),
                    "average_response_time": self._calculate_average_response_time(),
                    "adaptation_rate": self._calculate_adaptation_rate(),
                    "emergent_behavior_score": self._calculate_emergent_behavior_score()
                }
            }
            
            self.last_update = time.time()
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect neuromorphic metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process neuromorphic metrics for analysis."""
        if "error" in metrics:
            return metrics
        
        processed = {
            "timestamp": metrics["timestamp"],
            "neuromorphic_analysis": {},
            "performance_insights": {},
            "optimization_recommendations": []
        }
        
        # Analyze neuromorphic system performance
        neuromorphic_system = metrics["neuromorphic_system"]
        processed["neuromorphic_analysis"] = {
            "neuron_efficiency": neuromorphic_system["active_neurons"] / max(neuromorphic_system["active_neurons"], 1),
            "plasticity_effectiveness": neuromorphic_system["plasticity_rate"],
            "learning_progress": neuromorphic_system["learning_efficiency"]
        }
        
        # Generate optimization recommendations
        if neuromorphic_system["plasticity_rate"] < 0.1:
            processed["optimization_recommendations"].append({
                "type": "plasticity_optimization",
                "priority": "medium",
                "description": "Low plasticity rate, consider adjusting learning parameters"
            })
        
        if neuromorphic_system["learning_efficiency"] < 0.5:
            processed["optimization_recommendations"].append({
                "type": "learning_optimization",
                "priority": "high",
                "description": "Low learning efficiency, review network architecture and training data"
            })
        
        return processed
    
    def _get_active_neurons(self) -> int:
        """Get number of active neurons."""
        return self.neuromorphic_metrics.get("active_neurons", 0)
    
    def _get_network_connectivity(self) -> float:
        """Get network connectivity ratio."""
        return self.neuromorphic_metrics.get("network_connectivity", 0.0)
    
    def _get_plasticity_rate(self) -> float:
        """Get current plasticity rate."""
        return self.neuromorphic_metrics.get("plasticity_rate", 0.0)
    
    def _get_spiking_frequency(self) -> float:
        """Get average spiking frequency."""
        return self.neuromorphic_metrics.get("spiking_frequency", 0.0)
    
    def _calculate_learning_efficiency(self) -> float:
        """Calculate learning efficiency score."""
        return self.neuromorphic_metrics.get("learning_efficiency", 0.0)
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average network response time."""
        if not self.network_performance:
            return 0.0
        times = [perf["response_time"] for perf in self.network_performance.values()]
        return np.mean(times) if times else 0.0
    
    def _calculate_adaptation_rate(self) -> float:
        """Calculate system adaptation rate."""
        return self.neuromorphic_metrics.get("adaptation_rate", 0.0)
    
    def _calculate_emergent_behavior_score(self) -> float:
        """Calculate emergent behavior score."""
        # Simplified emergent behavior calculation
        connectivity = self._get_network_connectivity()
        plasticity = self._get_plasticity_rate()
        learning = self._calculate_learning_efficiency()
        
        return (connectivity + plasticity + learning) / 3.0

# ===== ALERT MANAGEMENT IMPLEMENTATIONS =====

class PerformanceAlertManager(BaseAlertManager):
    """Manage performance-related alerts."""
    
    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self.alert_rules = self._initialize_alert_rules()
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions in metrics."""
        alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if self._evaluate_rule(rule, metrics):
                alert = self._create_alert(rule_name, rule, metrics)
                if self.can_send_alert(alert["key"]):
                    alerts.append(alert)
                    self.last_alert_time[alert["key"]] = time.time()
        
        return alerts
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send an alert through configured channels."""
        try:
            for channel in self.config.notification_channels:
                if channel == "console":
                    self._send_console_alert(alert)
                elif channel == "log":
                    self._send_log_alert(alert)
                # Add more channels as needed
            
            self.alerts.append(alert)
            self.alert_history[alert["severity"]] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False
    
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert rules."""
        return {
            "high_cpu": {
                "metric": "cpu.percent",
                "threshold": 80.0,
                "severity": AlertSeverity.WARNING,
                "description": "High CPU usage detected"
            },
            "high_memory": {
                "metric": "memory.percent",
                "threshold": 85.0,
                "severity": AlertSeverity.WARNING,
                "description": "High memory usage detected"
            },
            "low_disk_space": {
                "metric": "disk.percent",
                "threshold": 90.0,
                "severity": AlertSeverity.ERROR,
                "description": "Low disk space detected"
            },
            "quantum_error_rate": {
                "metric": "quantum_system.error_rates",
                "threshold": 0.15,
                "severity": AlertSeverity.ERROR,
                "description": "High quantum error rate detected"
            },
            "neuromorphic_learning_failure": {
                "metric": "neuromorphic_system.learning_efficiency",
                "threshold": 0.3,
                "severity": AlertSeverity.WARNING,
                "description": "Low neuromorphic learning efficiency"
            }
        }
    
    def _evaluate_rule(self, rule: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """Evaluate if an alert rule should trigger."""
        try:
            metric_path = rule["metric"].split(".")
            value = metrics
            for key in metric_path:
                value = value.get(key, {})
            
            if not isinstance(value, (int, float)):
                return False
            
            threshold = rule["threshold"]
            return value > threshold
            
        except (KeyError, TypeError):
            return False
    
    def _create_alert(self, rule_name: str, rule: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create an alert from a triggered rule."""
        return {
            "id": f"{rule_name}_{int(time.time())}",
            "key": rule_name,
            "severity": rule["severity"].value,
            "description": rule["description"],
            "metric": rule["metric"],
            "threshold": rule["threshold"],
            "current_value": self._extract_metric_value(rule["metric"], metrics),
            "timestamp": time.time(),
            "source": "performance_monitor"
        }
    
    def _extract_metric_value(self, metric_path: str, metrics: Dict[str, Any]) -> Any:
        """Extract metric value from nested dictionary."""
        try:
            keys = metric_path.split(".")
            value = metrics
            for key in keys:
                value = value.get(key, {})
            return value
        except (KeyError, TypeError):
            return None
    
    def _send_console_alert(self, alert: Dict[str, Any]):
        """Send alert to console."""
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }
        
        emoji = severity_emoji.get(alert["severity"], "ℹ️")
        print(f"{emoji} ALERT: {alert['description']}")
        print(f"   Metric: {alert['metric']}")
        print(f"   Current: {alert['current_value']}")
        print(f"   Threshold: {alert['threshold']}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert['timestamp']))}")
        print()
    
    def _send_log_alert(self, alert: Dict[str, Any]):
        """Send alert to log."""
        self.logger.warning(f"ALERT: {alert['description']} - {alert['metric']}: {alert['current_value']}")

# ===== MAIN ENHANCED PERFORMANCE MONITORING SYSTEM =====

class EnhancedPerformanceMonitoringSystem:
    """Main enhanced performance monitoring system."""
    
    def __init__(self, performance_config: PerformanceConfig, alert_config: AlertConfig):
        self.performance_config = performance_config
        self.alert_config = alert_config
        self.logger = logging.getLogger(f"{__name__}.EnhancedPerformanceMonitoringSystem")
        
        # Initialize monitoring components
        self.monitors: Dict[str, BaseMonitor] = {}
        self.alert_manager = PerformanceAlertManager(alert_config)
        
        # Initialize monitoring thread
        self.monitoring_thread = None
        self.running = False
        self.metrics_history = deque(maxlen=performance_config.max_metrics)
        
        # Initialize monitoring components
        self._initialize_monitors()
    
    def _initialize_monitors(self):
        """Initialize monitoring components."""
        try:
            # System resource monitor
            self.monitors["system"] = SystemResourceMonitor(self.performance_config)
            
            # Quantum performance monitor
            self.monitors["quantum"] = QuantumPerformanceMonitor(self.performance_config)
            
            # Neuromorphic performance monitor
            self.monitors["neuromorphic"] = NeuromorphicPerformanceMonitor(self.performance_config)
            
            self.logger.info("Performance monitoring components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring components: {e}")
            raise
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.running:
            self.logger.warning("Monitoring system is already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics from all monitors
                all_metrics = {}
                for name, monitor in self.monitors.items():
                    if monitor.is_enabled():
                        metrics = monitor.collect_metrics()
                        processed_metrics = monitor.process_metrics(metrics)
                        all_metrics[name] = processed_metrics
                
                # Store metrics in history
                if all_metrics:
                    self.metrics_history.append({
                        "timestamp": time.time(),
                        "metrics": all_metrics
                    })
                
                # Check for alerts
                if self.alert_config.enabled:
                    alerts = self.alert_manager.check_alerts(all_metrics)
                    for alert in alerts:
                        self.alert_manager.send_alert(alert)
                
                # Sleep until next sampling interval
                time.sleep(self.performance_config.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics from all monitors."""
        current_metrics = {}
        for name, monitor in self.monitors.items():
            if monitor.is_enabled():
                current_metrics[name] = monitor.collect_metrics()
        return current_metrics
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics history."""
        if limit is None:
            return list(self.metrics_history)
        return list(self.metrics_history)[-limit:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "status": "running" if self.running else "stopped",
            "monitors": {
                name: {
                    "enabled": monitor.is_enabled(),
                    "last_update": monitor.get_last_update(),
                    "status": "active" if monitor.is_enabled() else "disabled"
                }
                for name, monitor in self.monitors.items()
            },
            "alert_manager": {
                "enabled": self.alert_config.enabled,
                "active_alerts": len(self.alert_manager.alerts),
                "alert_history": dict(self.alert_manager.alert_history)
            },
            "metrics_history": {
                "total_entries": len(self.metrics_history),
                "oldest_entry": self.metrics_history[0]["timestamp"] if self.metrics_history else None,
                "newest_entry": self.metrics_history[-1]["timestamp"] if self.metrics_history else None
            }
        }
    
    def export_metrics(self, file_path: str, format: str = "json") -> bool:
        """Export metrics to file."""
        try:
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(list(self.metrics_history), f, indent=2, default=str)
            else:
                self.logger.warning(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Metrics exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False

# ===== FACTORY FUNCTIONS =====

def create_enhanced_performance_monitoring_system(
    performance_config: Optional[PerformanceConfig] = None,
    alert_config: Optional[AlertConfig] = None
) -> EnhancedPerformanceMonitoringSystem:
    """Create enhanced performance monitoring system."""
    if performance_config is None:
        performance_config = PerformanceConfig()
    
    if alert_config is None:
        alert_config = AlertConfig()
    
    return EnhancedPerformanceMonitoringSystem(performance_config, alert_config)

def create_minimal_performance_config() -> PerformanceConfig:
    """Create minimal performance monitoring configuration."""
    return PerformanceConfig(
        enabled=True,
        monitoring_mode=MonitoringMode.PASSIVE,
        sampling_interval=5.0,
        retention_period=1800,
        max_metrics=1000,
        enable_alerts=False,
        enable_anomaly_detection=False,
        enable_auto_optimization=False
    )

def create_maximum_performance_config() -> PerformanceConfig:
    """Create maximum performance monitoring configuration."""
    return PerformanceConfig(
        enabled=True,
        monitoring_mode=MonitoringMode.INTELLIGENT,
        sampling_interval=0.5,
        retention_period=7200,
        max_metrics=50000,
        enable_alerts=True,
        enable_anomaly_detection=True,
        enable_auto_optimization=True
    )

# ===== EXPORT MAIN CLASSES =====

__all__ = [
    "EnhancedPerformanceMonitoringSystem",
    "PerformanceConfig",
    "AlertConfig",
    "AnomalyDetectionConfig",
    "MetricType",
    "AlertSeverity",
    "MonitoringMode",
    "BaseMonitor",
    "BaseAlertManager",
    "SystemResourceMonitor",
    "QuantumPerformanceMonitor",
    "NeuromorphicPerformanceMonitor",
    "PerformanceAlertManager",
    "create_enhanced_performance_monitoring_system",
    "create_minimal_performance_config",
    "create_maximum_performance_config"
]
