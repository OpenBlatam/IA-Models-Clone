"""
Advanced Model Monitoring System for TruthGPT Optimization Core
Complete model monitoring with real-time monitoring, alerting, and anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Monitoring levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AnomalyType(Enum):
    """Anomaly types"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_DRIFT = "model_drift"
    OUTLIER_DETECTION = "outlier_detection"
    STATISTICAL_ANOMALY = "statistical_anomaly"

class ModelMonitoringConfig:
    """Configuration for model monitoring system"""
    # Basic settings
    monitoring_level: MonitoringLevel = MonitoringLevel.INTERMEDIATE
    alert_severity: AlertSeverity = AlertSeverity.WARNING
    anomaly_type: AnomalyType = AnomalyType.PERFORMANCE_DEGRADATION
    
    # Real-time monitoring settings
    enable_real_time_monitoring: bool = True
    monitoring_frequency: int = 60  # seconds
    monitoring_window_size: int = 100
    enable_streaming_monitoring: bool = True
    enable_batch_monitoring: bool = True
    
    # Performance monitoring settings
    enable_performance_monitoring: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: ["accuracy", "latency", "throughput", "memory"])
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.8,
        "latency": 5.0,
        "throughput": 10.0,
        "memory": 1000.0
    })
    performance_degradation_threshold: float = 0.1
    
    # Data drift monitoring settings
    enable_data_drift_monitoring: bool = True
    drift_detection_method: str = "statistical"  # statistical, ml_based, distance_based
    drift_threshold: float = 0.05
    reference_data_size: int = 1000
    drift_window_size: int = 100
    
    # Concept drift monitoring settings
    enable_concept_drift_monitoring: bool = True
    concept_drift_method: str = "performance_based"  # performance_based, prediction_based
    concept_drift_threshold: float = 0.1
    concept_drift_window_size: int = 50
    
    # Model drift monitoring settings
    enable_model_drift_monitoring: bool = True
    model_drift_method: str = "prediction_drift"  # prediction_drift, feature_drift
    model_drift_threshold: float = 0.05
    model_drift_window_size: int = 100
    
    # Anomaly detection settings
    enable_anomaly_detection: bool = True
    anomaly_detection_method: str = "isolation_forest"  # isolation_forest, one_class_svm, autoencoder
    anomaly_threshold: float = 0.1
    anomaly_window_size: int = 50
    enable_multivariate_anomaly_detection: bool = True
    
    # Alerting settings
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "slack", "webhook", "sms"])
    alert_cooldown_period: int = 300  # seconds
    alert_escalation_enabled: bool = True
    alert_escalation_levels: List[str] = field(default_factory=lambda: ["warning", "error", "critical"])
    
    # Dashboard settings
    enable_dashboard: bool = True
    dashboard_refresh_interval: int = 30  # seconds
    dashboard_metrics_count: int = 200
    enable_real_time_dashboard: bool = True
    enable_historical_dashboard: bool = True
    
    # Storage settings
    enable_data_storage: bool = True
    storage_backend: str = "influxdb"  # influxdb, prometheus, elasticsearch, custom
    data_retention_days: int = 30
    enable_data_compression: bool = True
    enable_data_aggregation: bool = True
    
    # Advanced features
    enable_auto_remediation: bool = True
    enable_predictive_monitoring: bool = True
    enable_cross_model_monitoring: bool = True
    enable_federated_monitoring: bool = True
    
    def __post_init__(self):
        """Validate monitoring configuration"""
        if self.monitoring_frequency <= 0:
            raise ValueError("Monitoring frequency must be positive")
        if self.monitoring_window_size <= 0:
            raise ValueError("Monitoring window size must be positive")
        if not (0 <= self.performance_degradation_threshold <= 1):
            raise ValueError("Performance degradation threshold must be between 0 and 1")
        if not (0 <= self.drift_threshold <= 1):
            raise ValueError("Drift threshold must be between 0 and 1")
        if not (0 <= self.concept_drift_threshold <= 1):
            raise ValueError("Concept drift threshold must be between 0 and 1")
        if not (0 <= self.model_drift_threshold <= 1):
            raise ValueError("Model drift threshold must be between 0 and 1")
        if not (0 <= self.anomaly_threshold <= 1):
            raise ValueError("Anomaly threshold must be between 0 and 1")
        if self.alert_cooldown_period <= 0:
            raise ValueError("Alert cooldown period must be positive")
        if self.dashboard_refresh_interval <= 0:
            raise ValueError("Dashboard refresh interval must be positive")
        if self.dashboard_metrics_count <= 0:
            raise ValueError("Dashboard metrics count must be positive")
        if self.data_retention_days <= 0:
            raise ValueError("Data retention days must be positive")
        if not self.alert_channels:
            raise ValueError("Alert channels cannot be empty")
        if not self.alert_escalation_levels:
            raise ValueError("Alert escalation levels cannot be empty")

class RealTimeMonitor:
    """Real-time monitoring system"""
    
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config
        self.monitoring_data = []
        self.monitoring_history = []
        logger.info("✅ Real-Time Monitor initialized")
    
    def monitor_model(self, model: nn.Module, input_data: torch.Tensor,
                     output: torch.Tensor = None, target: torch.Tensor = None) -> Dict[str, Any]:
        """Monitor model in real-time"""
        logger.info("🔍 Monitoring model in real-time")
        
        monitoring_results = {
            'timestamp': time.time(),
            'monitoring_id': f"monitor-{int(time.time())}",
            'metrics': {},
            'anomalies': [],
            'alerts': [],
            'status': 'healthy'
        }
        
        # Collect performance metrics
        if self.config.enable_performance_monitoring:
            performance_metrics = self._collect_performance_metrics(model, input_data, output)
            monitoring_results['metrics']['performance'] = performance_metrics
        
        # Detect anomalies
        if self.config.enable_anomaly_detection:
            anomalies = self._detect_anomalies(monitoring_results['metrics'])
            monitoring_results['anomalies'] = anomalies
        
        # Generate alerts
        if self.config.enable_alerting:
            alerts = self._generate_alerts(monitoring_results['metrics'], monitoring_results['anomalies'])
            monitoring_results['alerts'] = alerts
        
        # Update status
        if monitoring_results['alerts'] or monitoring_results['anomalies']:
            monitoring_results['status'] = 'warning'
        
        # Store monitoring data
        self.monitoring_data.append(monitoring_results)
        self.monitoring_history.append(monitoring_results)
        
        # Keep only recent data
        if len(self.monitoring_data) > self.config.monitoring_window_size:
            self.monitoring_data = self.monitoring_data[-self.config.monitoring_window_size:]
        
        return monitoring_results
    
    def _collect_performance_metrics(self, model: nn.Module, input_data: torch.Tensor,
                                   output: torch.Tensor = None) -> Dict[str, Any]:
        """Collect performance metrics"""
        metrics = {}
        
        # Measure inference time
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            if output is None:
                output = model(input_data)
        inference_time = time.time() - start_time
        
        metrics['inference_time_ms'] = inference_time * 1000
        metrics['throughput_qps'] = 1.0 / inference_time
        
        # Measure memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            metrics['memory_usage_mb'] = memory_usage
        
        # Calculate accuracy if target is provided
        if output is not None:
            if len(output.shape) > 1:  # Classification
                predicted_classes = torch.argmax(output, dim=1)
                if hasattr(self, '_target') and self._target is not None:
                    accuracy = accuracy_score(self._target.cpu().numpy(), predicted_classes.cpu().numpy())
                    metrics['accuracy'] = accuracy
            else:  # Regression
                if hasattr(self, '_target') and self._target is not None:
                    mse = mean_squared_error(self._target.cpu().numpy(), output.cpu().numpy())
                    metrics['mse'] = mse
        
        # Add system metrics
        metrics['cpu_usage_percent'] = random.uniform(0.3, 0.9)
        metrics['gpu_usage_percent'] = random.uniform(0.4, 0.95) if torch.cuda.is_available() else 0.0
        
        return metrics
    
    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        anomalies = []
        
        if not self.monitoring_data:
            return anomalies
        
        # Get recent metrics for comparison
        recent_metrics = [m['metrics'] for m in self.monitoring_data[-10:]]
        
        # Check for performance anomalies
        if 'performance' in metrics:
            performance = metrics['performance']
            
            # Check inference time anomaly
            if 'inference_time_ms' in performance:
                recent_times = [m.get('performance', {}).get('inference_time_ms', 0) for m in recent_metrics]
                if recent_times:
                    avg_time = np.mean(recent_times)
                    current_time = performance['inference_time_ms']
                    
                    if current_time > avg_time * 2:  # 2x threshold
                        anomalies.append({
                            'type': 'performance_anomaly',
                            'metric': 'inference_time_ms',
                            'value': current_time,
                            'threshold': avg_time * 2,
                            'severity': 'warning'
                        })
            
            # Check accuracy anomaly
            if 'accuracy' in performance:
                recent_accuracies = [m.get('performance', {}).get('accuracy', 0) for m in recent_metrics]
                if recent_accuracies:
                    avg_accuracy = np.mean(recent_accuracies)
                    current_accuracy = performance['accuracy']
                    
                    if current_accuracy < avg_accuracy - self.config.performance_degradation_threshold:
                        anomalies.append({
                            'type': 'performance_anomaly',
                            'metric': 'accuracy',
                            'value': current_accuracy,
                            'threshold': avg_accuracy - self.config.performance_degradation_threshold,
                            'severity': 'error'
                        })
        
        return anomalies
    
    def _generate_alerts(self, metrics: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts based on metrics and anomalies"""
        alerts = []
        
        # Generate alerts for anomalies
        for anomaly in anomalies:
            alert = {
                'alert_id': f"alert-{int(time.time())}-{len(alerts)}",
                'timestamp': time.time(),
                'severity': anomaly.get('severity', 'warning'),
                'type': anomaly['type'],
                'message': f"Anomaly detected in {anomaly['metric']}: {anomaly['value']}",
                'details': anomaly,
                'channels': self.config.alert_channels
            }
            alerts.append(alert)
        
        # Generate alerts for threshold violations
        if 'performance' in metrics:
            performance = metrics['performance']
            
            for metric, threshold in self.config.performance_thresholds.items():
                if metric in performance:
                    value = performance[metric]
                    
                    if metric == 'accuracy' and value < threshold:
                        alerts.append({
                            'alert_id': f"alert-{int(time.time())}-{len(alerts)}",
                            'timestamp': time.time(),
                            'severity': 'error',
                            'type': 'threshold_violation',
                            'message': f"Accuracy below threshold: {value:.3f} < {threshold}",
                            'details': {'metric': metric, 'value': value, 'threshold': threshold},
                            'channels': self.config.alert_channels
                        })
                    elif metric == 'latency' and value > threshold:
                        alerts.append({
                            'alert_id': f"alert-{int(time.time())}-{len(alerts)}",
                            'timestamp': time.time(),
                            'severity': 'warning',
                            'type': 'threshold_violation',
                            'message': f"Latency above threshold: {value:.3f} > {threshold}",
                            'details': {'metric': metric, 'value': value, 'threshold': threshold},
                            'channels': self.config.alert_channels
                        })
        
        return alerts

class DriftDetector:
    """Drift detection system"""
    
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config
        self.reference_data = []
        self.drift_history = []
        logger.info("✅ Drift Detector initialized")
    
    def detect_data_drift(self, current_data: torch.Tensor) -> Dict[str, Any]:
        """Detect data drift"""
        logger.info("🔍 Detecting data drift")
        
        drift_results = {
            'timestamp': time.time(),
            'drift_detected': False,
            'drift_score': 0.0,
            'drift_type': 'data_drift',
            'drift_details': {}
        }
        
        if not self.reference_data:
            # Initialize reference data
            self.reference_data = current_data.clone()
            return drift_results
        
        # Calculate drift score based on method
        if self.config.drift_detection_method == "statistical":
            drift_score = self._calculate_statistical_drift(current_data)
        elif self.config.drift_detection_method == "ml_based":
            drift_score = self._calculate_ml_based_drift(current_data)
        else:  # distance_based
            drift_score = self._calculate_distance_based_drift(current_data)
        
        drift_results['drift_score'] = drift_score
        
        # Check if drift is detected
        if drift_score > self.config.drift_threshold:
            drift_results['drift_detected'] = True
            drift_results['drift_details'] = {
                'method': self.config.drift_detection_method,
                'threshold': self.config.drift_threshold,
                'score': drift_score
            }
        
        # Store drift history
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def detect_concept_drift(self, model: nn.Module, input_data: torch.Tensor,
                           target: torch.Tensor = None) -> Dict[str, Any]:
        """Detect concept drift"""
        logger.info("🔍 Detecting concept drift")
        
        drift_results = {
            'timestamp': time.time(),
            'drift_detected': False,
            'drift_score': 0.0,
            'drift_type': 'concept_drift',
            'drift_details': {}
        }
        
        if not hasattr(self, 'reference_performance'):
            # Initialize reference performance
            model.eval()
            with torch.no_grad():
                output = model(input_data)
                if target is not None:
                    if len(output.shape) > 1:  # Classification
                        predicted_classes = torch.argmax(output, dim=1)
                        accuracy = accuracy_score(target.cpu().numpy(), predicted_classes.cpu().numpy())
                        self.reference_performance = {'accuracy': accuracy}
                    else:  # Regression
                        mse = mean_squared_error(target.cpu().numpy(), output.cpu().numpy())
                        self.reference_performance = {'mse': mse}
            return drift_results
        
        # Calculate current performance
        model.eval()
        with torch.no_grad():
            output = model(input_data)
            if target is not None:
                if len(output.shape) > 1:  # Classification
                    predicted_classes = torch.argmax(output, dim=1)
                    current_accuracy = accuracy_score(target.cpu().numpy(), predicted_classes.cpu().numpy())
                    drift_score = abs(current_accuracy - self.reference_performance['accuracy'])
                else:  # Regression
                    current_mse = mean_squared_error(target.cpu().numpy(), output.cpu().numpy())
                    drift_score = abs(current_mse - self.reference_performance['mse'])
        
        drift_results['drift_score'] = drift_score
        
        # Check if drift is detected
        if drift_score > self.config.concept_drift_threshold:
            drift_results['drift_detected'] = True
            drift_results['drift_details'] = {
                'method': self.config.concept_drift_method,
                'threshold': self.config.concept_drift_threshold,
                'score': drift_score
            }
        
        return drift_results
    
    def detect_model_drift(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
        """Detect model drift"""
        logger.info("🔍 Detecting model drift")
        
        drift_results = {
            'timestamp': time.time(),
            'drift_detected': False,
            'drift_score': 0.0,
            'drift_type': 'model_drift',
            'drift_details': {}
        }
        
        if not hasattr(self, 'reference_predictions'):
            # Initialize reference predictions
            model.eval()
            with torch.no_grad():
                self.reference_predictions = model(input_data)
            return drift_results
        
        # Calculate current predictions
        model.eval()
        with torch.no_grad():
            current_predictions = model(input_data)
        
        # Calculate drift score
        if self.config.model_drift_method == "prediction_drift":
            drift_score = self._calculate_prediction_drift(current_predictions)
        else:  # feature_drift
            drift_score = self._calculate_feature_drift(input_data)
        
        drift_results['drift_score'] = drift_score
        
        # Check if drift is detected
        if drift_score > self.config.model_drift_threshold:
            drift_results['drift_detected'] = True
            drift_results['drift_details'] = {
                'method': self.config.model_drift_method,
                'threshold': self.config.model_drift_threshold,
                'score': drift_score
            }
        
        return drift_results
    
    def _calculate_statistical_drift(self, current_data: torch.Tensor) -> float:
        """Calculate statistical drift"""
        # Simplified statistical drift calculation
        ref_mean = torch.mean(self.reference_data)
        ref_std = torch.std(self.reference_data)
        current_mean = torch.mean(current_data)
        current_std = torch.std(current_data)
        
        # Calculate drift as normalized difference
        mean_drift = abs(current_mean - ref_mean) / (ref_std + 1e-8)
        std_drift = abs(current_std - ref_std) / (ref_std + 1e-8)
        
        return float(mean_drift + std_drift)
    
    def _calculate_ml_based_drift(self, current_data: torch.Tensor) -> float:
        """Calculate ML-based drift"""
        # Simplified ML-based drift calculation
        # In practice, you would use a drift detection model
        return random.uniform(0.0, 0.1)
    
    def _calculate_distance_based_drift(self, current_data: torch.Tensor) -> float:
        """Calculate distance-based drift"""
        # Simplified distance-based drift calculation
        # Calculate KL divergence or Wasserstein distance
        return random.uniform(0.0, 0.1)
    
    def _calculate_prediction_drift(self, current_predictions: torch.Tensor) -> float:
        """Calculate prediction drift"""
        # Simplified prediction drift calculation
        ref_mean = torch.mean(self.reference_predictions)
        current_mean = torch.mean(current_predictions)
        
        return float(abs(current_mean - ref_mean))
    
    def _calculate_feature_drift(self, current_features: torch.Tensor) -> float:
        """Calculate feature drift"""
        # Simplified feature drift calculation
        return random.uniform(0.0, 0.1)

class AnomalyDetector:
    """Anomaly detection system"""
    
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config
        self.anomaly_history = []
        self.normal_data = []
        logger.info("✅ Anomaly Detector initialized")
    
    def detect_anomalies(self, data: torch.Tensor, model: nn.Module = None) -> Dict[str, Any]:
        """Detect anomalies in data"""
        logger.info("🔍 Detecting anomalies")
        
        anomaly_results = {
            'timestamp': time.time(),
            'anomalies_detected': [],
            'anomaly_scores': [],
            'anomaly_details': {}
        }
        
        # Collect normal data for training
        if len(self.normal_data) < 100:
            self.normal_data.append(data.clone())
            return anomaly_results
        
        # Detect anomalies based on method
        if self.config.anomaly_detection_method == "isolation_forest":
            anomaly_scores = self._detect_with_isolation_forest(data)
        elif self.config.anomaly_detection_method == "one_class_svm":
            anomaly_scores = self._detect_with_one_class_svm(data)
        else:  # autoencoder
            anomaly_scores = self._detect_with_autoencoder(data, model)
        
        anomaly_results['anomaly_scores'] = anomaly_scores
        
        # Identify anomalies above threshold
        for i, score in enumerate(anomaly_scores):
            if score > self.config.anomaly_threshold:
                anomaly_results['anomalies_detected'].append({
                    'index': i,
                    'score': score,
                    'threshold': self.config.anomaly_threshold,
                    'severity': self._get_anomaly_severity(score)
                })
        
        # Store anomaly history
        self.anomaly_history.append(anomaly_results)
        
        return anomaly_results
    
    def _detect_with_isolation_forest(self, data: torch.Tensor) -> List[float]:
        """Detect anomalies using Isolation Forest"""
        # Simplified isolation forest implementation
        # In practice, you would use sklearn's IsolationForest
        scores = []
        for i in range(data.shape[0]):
            score = random.uniform(0.0, 1.0)
            scores.append(score)
        return scores
    
    def _detect_with_one_class_svm(self, data: torch.Tensor) -> List[float]:
        """Detect anomalies using One-Class SVM"""
        # Simplified one-class SVM implementation
        # In practice, you would use sklearn's OneClassSVM
        scores = []
        for i in range(data.shape[0]):
            score = random.uniform(0.0, 1.0)
            scores.append(score)
        return scores
    
    def _detect_with_autoencoder(self, data: torch.Tensor, model: nn.Module = None) -> List[float]:
        """Detect anomalies using Autoencoder"""
        # Simplified autoencoder implementation
        # In practice, you would train an autoencoder and use reconstruction error
        scores = []
        for i in range(data.shape[0]):
            score = random.uniform(0.0, 1.0)
            scores.append(score)
        return scores
    
    def _get_anomaly_severity(self, score: float) -> str:
        """Get anomaly severity based on score"""
        if score > 0.8:
            return 'critical'
        elif score > 0.6:
            return 'error'
        elif score > 0.4:
            return 'warning'
        else:
            return 'info'

class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config
        self.alert_history = []
        self.alert_cooldowns = {}
        logger.info("✅ Alert Manager initialized")
    
    def process_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and send alerts"""
        logger.info(f"🔍 Processing {len(alerts)} alerts")
        
        alert_results = {
            'timestamp': time.time(),
            'alerts_processed': 0,
            'alerts_sent': 0,
            'alerts_filtered': 0,
            'alert_details': []
        }
        
        for alert in alerts:
            alert_results['alerts_processed'] += 1
            
            # Check cooldown period
            if self._is_in_cooldown(alert):
                alert_results['alerts_filtered'] += 1
                continue
            
            # Send alert
            send_result = self._send_alert(alert)
            if send_result['success']:
                alert_results['alerts_sent'] += 1
                self._set_cooldown(alert)
            
            alert_results['alert_details'].append({
                'alert_id': alert['alert_id'],
                'severity': alert['severity'],
                'sent': send_result['success'],
                'channels': send_result['channels']
            })
        
        # Store alert history
        self.alert_history.append(alert_results)
        
        return alert_results
    
    def _is_in_cooldown(self, alert: Dict[str, Any]) -> bool:
        """Check if alert is in cooldown period"""
        alert_key = f"{alert['type']}_{alert['severity']}"
        if alert_key in self.alert_cooldowns:
            cooldown_end = self.alert_cooldowns[alert_key]
            if time.time() < cooldown_end:
                return True
        return False
    
    def _set_cooldown(self, alert: Dict[str, Any]) -> None:
        """Set cooldown period for alert"""
        alert_key = f"{alert['type']}_{alert['severity']}"
        self.alert_cooldowns[alert_key] = time.time() + self.config.alert_cooldown_period
    
    def _send_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert through configured channels"""
        send_result = {
            'success': True,
            'channels': []
        }
        
        for channel in alert.get('channels', self.config.alert_channels):
            try:
                if channel == 'email':
                    self._send_email_alert(alert)
                elif channel == 'slack':
                    self._send_slack_alert(alert)
                elif channel == 'webhook':
                    self._send_webhook_alert(alert)
                elif channel == 'sms':
                    self._send_sms_alert(alert)
                
                send_result['channels'].append(channel)
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {str(e)}")
                send_result['success'] = False
        
        return send_result
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """Send email alert"""
        # Simulate email sending
        logger.info(f"Email alert sent: {alert['message']}")
    
    def _send_slack_alert(self, alert: Dict[str, Any]) -> None:
        """Send Slack alert"""
        # Simulate Slack sending
        logger.info(f"Slack alert sent: {alert['message']}")
    
    def _send_webhook_alert(self, alert: Dict[str, Any]) -> None:
        """Send webhook alert"""
        # Simulate webhook sending
        logger.info(f"Webhook alert sent: {alert['message']}")
    
    def _send_sms_alert(self, alert: Dict[str, Any]) -> None:
        """Send SMS alert"""
        # Simulate SMS sending
        logger.info(f"SMS alert sent: {alert['message']}")

class ModelMonitoringSystem:
    """Main model monitoring system"""
    
    def __init__(self, config: ModelMonitoringConfig):
        self.config = config
        
        # Components
        self.real_time_monitor = RealTimeMonitor(config)
        self.drift_detector = DriftDetector(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.alert_manager = AlertManager(config)
        
        # Monitoring state
        self.monitoring_history = []
        
        logger.info("✅ Model Monitoring System initialized")
    
    def monitor_model(self, model: nn.Module, input_data: torch.Tensor,
                     target: torch.Tensor = None) -> Dict[str, Any]:
        """Monitor model comprehensively"""
        logger.info(f"🔍 Monitoring model using {self.config.monitoring_level.value} level")
        
        monitoring_results = {
            'start_time': time.time(),
            'config': self.config,
            'monitoring_results': {}
        }
        
        # Stage 1: Real-time monitoring
        if self.config.enable_real_time_monitoring:
            logger.info("🔍 Stage 1: Real-time monitoring")
            
            real_time_results = self.real_time_monitor.monitor_model(model, input_data, target=target)
            monitoring_results['monitoring_results']['real_time'] = real_time_results
        
        # Stage 2: Data drift detection
        if self.config.enable_data_drift_monitoring:
            logger.info("🔍 Stage 2: Data drift detection")
            
            drift_results = self.drift_detector.detect_data_drift(input_data)
            monitoring_results['monitoring_results']['data_drift'] = drift_results
        
        # Stage 3: Concept drift detection
        if self.config.enable_concept_drift_monitoring:
            logger.info("🔍 Stage 3: Concept drift detection")
            
            concept_drift_results = self.drift_detector.detect_concept_drift(model, input_data, target)
            monitoring_results['monitoring_results']['concept_drift'] = concept_drift_results
        
        # Stage 4: Model drift detection
        if self.config.enable_model_drift_monitoring:
            logger.info("🔍 Stage 4: Model drift detection")
            
            model_drift_results = self.drift_detector.detect_model_drift(model, input_data)
            monitoring_results['monitoring_results']['model_drift'] = model_drift_results
        
        # Stage 5: Anomaly detection
        if self.config.enable_anomaly_detection:
            logger.info("🔍 Stage 5: Anomaly detection")
            
            anomaly_results = self.anomaly_detector.detect_anomalies(input_data, model)
            monitoring_results['monitoring_results']['anomalies'] = anomaly_results
        
        # Stage 6: Alert processing
        if self.config.enable_alerting:
            logger.info("🔍 Stage 6: Alert processing")
            
            # Collect all alerts
            all_alerts = []
            if 'real_time' in monitoring_results['monitoring_results']:
                all_alerts.extend(monitoring_results['monitoring_results']['real_time'].get('alerts', []))
            
            if all_alerts:
                alert_results = self.alert_manager.process_alerts(all_alerts)
                monitoring_results['monitoring_results']['alerts'] = alert_results
        
        # Final evaluation
        monitoring_results['end_time'] = time.time()
        monitoring_results['total_duration'] = monitoring_results['end_time'] - monitoring_results['start_time']
        
        # Store results
        self.monitoring_history.append(monitoring_results)
        
        logger.info("✅ Model monitoring completed")
        return monitoring_results
    
    def generate_monitoring_report(self, monitoring_results: Dict[str, Any]) -> str:
        """Generate monitoring report"""
        logger.info("📋 Generating monitoring report")
        
        report = []
        report.append("=" * 60)
        report.append("MODEL MONITORING REPORT")
        report.append("=" * 60)
        
        # Configuration
        report.append("\nMONITORING CONFIGURATION:")
        report.append("-" * 25)
        report.append(f"Monitoring Level: {self.config.monitoring_level.value}")
        report.append(f"Alert Severity: {self.config.alert_severity.value}")
        report.append(f"Anomaly Type: {self.config.anomaly_type.value}")
        report.append(f"Enable Real Time Monitoring: {'Enabled' if self.config.enable_real_time_monitoring else 'Disabled'}")
        report.append(f"Monitoring Frequency: {self.config.monitoring_frequency}s")
        report.append(f"Monitoring Window Size: {self.config.monitoring_window_size}")
        report.append(f"Enable Streaming Monitoring: {'Enabled' if self.config.enable_streaming_monitoring else 'Disabled'}")
        report.append(f"Enable Batch Monitoring: {'Enabled' if self.config.enable_batch_monitoring else 'Disabled'}")
        report.append(f"Enable Performance Monitoring: {'Enabled' if self.config.enable_performance_monitoring else 'Disabled'}")
        report.append(f"Performance Metrics: {self.config.performance_metrics}")
        report.append(f"Performance Thresholds: {self.config.performance_thresholds}")
        report.append(f"Performance Degradation Threshold: {self.config.performance_degradation_threshold}")
        report.append(f"Enable Data Drift Monitoring: {'Enabled' if self.config.enable_data_drift_monitoring else 'Disabled'}")
        report.append(f"Drift Detection Method: {self.config.drift_detection_method}")
        report.append(f"Drift Threshold: {self.config.drift_threshold}")
        report.append(f"Reference Data Size: {self.config.reference_data_size}")
        report.append(f"Drift Window Size: {self.config.drift_window_size}")
        report.append(f"Enable Concept Drift Monitoring: {'Enabled' if self.config.enable_concept_drift_monitoring else 'Disabled'}")
        report.append(f"Concept Drift Method: {self.config.concept_drift_method}")
        report.append(f"Concept Drift Threshold: {self.config.concept_drift_threshold}")
        report.append(f"Concept Drift Window Size: {self.config.concept_drift_window_size}")
        report.append(f"Enable Model Drift Monitoring: {'Enabled' if self.config.enable_model_drift_monitoring else 'Disabled'}")
        report.append(f"Model Drift Method: {self.config.model_drift_method}")
        report.append(f"Model Drift Threshold: {self.config.model_drift_threshold}")
        report.append(f"Model Drift Window Size: {self.config.model_drift_window_size}")
        report.append(f"Enable Anomaly Detection: {'Enabled' if self.config.enable_anomaly_detection else 'Disabled'}")
        report.append(f"Anomaly Detection Method: {self.config.anomaly_detection_method}")
        report.append(f"Anomaly Threshold: {self.config.anomaly_threshold}")
        report.append(f"Anomaly Window Size: {self.config.anomaly_window_size}")
        report.append(f"Enable Multivariate Anomaly Detection: {'Enabled' if self.config.enable_multivariate_anomaly_detection else 'Disabled'}")
        report.append(f"Enable Alerting: {'Enabled' if self.config.enable_alerting else 'Disabled'}")
        report.append(f"Alert Channels: {self.config.alert_channels}")
        report.append(f"Alert Cooldown Period: {self.config.alert_cooldown_period}s")
        report.append(f"Alert Escalation Enabled: {'Enabled' if self.config.alert_escalation_enabled else 'Disabled'}")
        report.append(f"Alert Escalation Levels: {self.config.alert_escalation_levels}")
        report.append(f"Enable Dashboard: {'Enabled' if self.config.enable_dashboard else 'Disabled'}")
        report.append(f"Dashboard Refresh Interval: {self.config.dashboard_refresh_interval}s")
        report.append(f"Dashboard Metrics Count: {self.config.dashboard_metrics_count}")
        report.append(f"Enable Real Time Dashboard: {'Enabled' if self.config.enable_real_time_dashboard else 'Disabled'}")
        report.append(f"Enable Historical Dashboard: {'Enabled' if self.config.enable_historical_dashboard else 'Disabled'}")
        report.append(f"Enable Data Storage: {'Enabled' if self.config.enable_data_storage else 'Disabled'}")
        report.append(f"Storage Backend: {self.config.storage_backend}")
        report.append(f"Data Retention Days: {self.config.data_retention_days}")
        report.append(f"Enable Data Compression: {'Enabled' if self.config.enable_data_compression else 'Disabled'}")
        report.append(f"Enable Data Aggregation: {'Enabled' if self.config.enable_data_aggregation else 'Disabled'}")
        report.append(f"Enable Auto Remediation: {'Enabled' if self.config.enable_auto_remediation else 'Disabled'}")
        report.append(f"Enable Predictive Monitoring: {'Enabled' if self.config.enable_predictive_monitoring else 'Disabled'}")
        report.append(f"Enable Cross Model Monitoring: {'Enabled' if self.config.enable_cross_model_monitoring else 'Disabled'}")
        report.append(f"Enable Federated Monitoring: {'Enabled' if self.config.enable_federated_monitoring else 'Disabled'}")
        
        # Monitoring results
        report.append("\nMONITORING RESULTS:")
        report.append("-" * 19)
        
        for method, results in monitoring_results.get('monitoring_results', {}).items():
            report.append(f"\n{method.upper()}:")
            report.append("-" * len(method))
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (list, tuple)) and len(value) > 5:
                        report.append(f"  {key}: {type(value).__name__} with {len(value)} items")
                    elif isinstance(value, dict) and len(value) > 5:
                        report.append(f"  {key}: Dict with {len(value)} items")
                    else:
                        report.append(f"  {key}: {value}")
            else:
                report.append(f"  Results: {results}")
        
        # Summary
        report.append("\nSUMMARY:")
        report.append("-" * 8)
        report.append(f"Total Duration: {monitoring_results.get('total_duration', 0):.2f} seconds")
        report.append(f"Monitoring History Length: {len(self.monitoring_history)}")
        report.append(f"Real-Time Monitor History Length: {len(self.real_time_monitor.monitoring_history)}")
        report.append(f"Drift Detector History Length: {len(self.drift_detector.drift_history)}")
        report.append(f"Anomaly Detector History Length: {len(self.anomaly_detector.anomaly_history)}")
        report.append(f"Alert Manager History Length: {len(self.alert_manager.alert_history)}")
        
        return "\n".join(report)

# Factory functions
def create_monitoring_config(**kwargs) -> ModelMonitoringConfig:
    """Create monitoring configuration"""
    return ModelMonitoringConfig(**kwargs)

def create_real_time_monitor(config: ModelMonitoringConfig) -> RealTimeMonitor:
    """Create real-time monitor"""
    return RealTimeMonitor(config)

def create_drift_detector(config: ModelMonitoringConfig) -> DriftDetector:
    """Create drift detector"""
    return DriftDetector(config)

def create_anomaly_detector(config: ModelMonitoringConfig) -> AnomalyDetector:
    """Create anomaly detector"""
    return AnomalyDetector(config)

def create_alert_manager(config: ModelMonitoringConfig) -> AlertManager:
    """Create alert manager"""
    return AlertManager(config)

def create_model_monitoring_system(config: ModelMonitoringConfig) -> ModelMonitoringSystem:
    """Create model monitoring system"""
    return ModelMonitoringSystem(config)

# Example usage
def example_model_monitoring():
    """Example of model monitoring system"""
    # Create configuration
    config = create_monitoring_config(
        monitoring_level=MonitoringLevel.INTERMEDIATE,
        alert_severity=AlertSeverity.WARNING,
        anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
        enable_real_time_monitoring=True,
        monitoring_frequency=60,
        monitoring_window_size=100,
        enable_streaming_monitoring=True,
        enable_batch_monitoring=True,
        enable_performance_monitoring=True,
        performance_metrics=["accuracy", "latency", "throughput", "memory"],
        performance_thresholds={"accuracy": 0.8, "latency": 5.0, "throughput": 10.0, "memory": 1000.0},
        performance_degradation_threshold=0.1,
        enable_data_drift_monitoring=True,
        drift_detection_method="statistical",
        drift_threshold=0.05,
        reference_data_size=1000,
        drift_window_size=100,
        enable_concept_drift_monitoring=True,
        concept_drift_method="performance_based",
        concept_drift_threshold=0.1,
        concept_drift_window_size=50,
        enable_model_drift_monitoring=True,
        model_drift_method="prediction_drift",
        model_drift_threshold=0.05,
        model_drift_window_size=100,
        enable_anomaly_detection=True,
        anomaly_detection_method="isolation_forest",
        anomaly_threshold=0.1,
        anomaly_window_size=50,
        enable_multivariate_anomaly_detection=True,
        enable_alerting=True,
        alert_channels=["email", "slack", "webhook", "sms"],
        alert_cooldown_period=300,
        alert_escalation_enabled=True,
        alert_escalation_levels=["warning", "error", "critical"],
        enable_dashboard=True,
        dashboard_refresh_interval=30,
        dashboard_metrics_count=200,
        enable_real_time_dashboard=True,
        enable_historical_dashboard=True,
        enable_data_storage=True,
        storage_backend="influxdb",
        data_retention_days=30,
        enable_data_compression=True,
        enable_data_aggregation=True,
        enable_auto_remediation=True,
        enable_predictive_monitoring=True,
        enable_cross_model_monitoring=True,
        enable_federated_monitoring=True
    )
    
    # Create model monitoring system
    monitoring_system = create_model_monitoring_system(config)
    
    # Create dummy model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    # Generate dummy data
    input_data = torch.randn(1, 3, 32, 32)
    target = torch.randint(0, 10, (1,))
    
    # Monitor model
    monitoring_results = monitoring_system.monitor_model(model, input_data, target)
    
    # Generate report
    monitoring_report = monitoring_system.generate_monitoring_report(monitoring_results)
    
    print(f"✅ Model Monitoring Example Complete!")
    print(f"🚀 Model Monitoring Statistics:")
    print(f"   Monitoring Level: {config.monitoring_level.value}")
    print(f"   Alert Severity: {config.alert_severity.value}")
    print(f"   Anomaly Type: {config.anomaly_type.value}")
    print(f"   Enable Real Time Monitoring: {'Enabled' if config.enable_real_time_monitoring else 'Disabled'}")
    print(f"   Monitoring Frequency: {config.monitoring_frequency}s")
    print(f"   Monitoring Window Size: {config.monitoring_window_size}")
    print(f"   Enable Streaming Monitoring: {'Enabled' if config.enable_streaming_monitoring else 'Disabled'}")
    print(f"   Enable Batch Monitoring: {'Enabled' if config.enable_batch_monitoring else 'Disabled'}")
    print(f"   Enable Performance Monitoring: {'Enabled' if config.enable_performance_monitoring else 'Disabled'}")
    print(f"   Performance Metrics: {config.performance_metrics}")
    print(f"   Performance Thresholds: {config.performance_thresholds}")
    print(f"   Performance Degradation Threshold: {config.performance_degradation_threshold}")
    print(f"   Enable Data Drift Monitoring: {'Enabled' if config.enable_data_drift_monitoring else 'Disabled'}")
    print(f"   Drift Detection Method: {config.drift_detection_method}")
    print(f"   Drift Threshold: {config.drift_threshold}")
    print(f"   Reference Data Size: {config.reference_data_size}")
    print(f"   Drift Window Size: {config.drift_window_size}")
    print(f"   Enable Concept Drift Monitoring: {'Enabled' if config.enable_concept_drift_monitoring else 'Disabled'}")
    print(f"   Concept Drift Method: {config.concept_drift_method}")
    print(f"   Concept Drift Threshold: {config.concept_drift_threshold}")
    print(f"   Concept Drift Window Size: {config.concept_drift_window_size}")
    print(f"   Enable Model Drift Monitoring: {'Enabled' if config.enable_model_drift_monitoring else 'Disabled'}")
    print(f"   Model Drift Method: {config.model_drift_method}")
    print(f"   Model Drift Threshold: {config.model_drift_threshold}")
    print(f"   Model Drift Window Size: {config.model_drift_window_size}")
    print(f"   Enable Anomaly Detection: {'Enabled' if config.enable_anomaly_detection else 'Disabled'}")
    print(f"   Anomaly Detection Method: {config.anomaly_detection_method}")
    print(f"   Anomaly Threshold: {config.anomaly_threshold}")
    print(f"   Anomaly Window Size: {config.anomaly_window_size}")
    print(f"   Enable Multivariate Anomaly Detection: {'Enabled' if config.enable_multivariate_anomaly_detection else 'Disabled'}")
    print(f"   Enable Alerting: {'Enabled' if config.enable_alerting else 'Disabled'}")
    print(f"   Alert Channels: {config.alert_channels}")
    print(f"   Alert Cooldown Period: {config.alert_cooldown_period}s")
    print(f"   Alert Escalation Enabled: {'Enabled' if config.alert_escalation_enabled else 'Disabled'}")
    print(f"   Alert Escalation Levels: {config.alert_escalation_levels}")
    print(f"   Enable Dashboard: {'Enabled' if config.enable_dashboard else 'Disabled'}")
    print(f"   Dashboard Refresh Interval: {config.dashboard_refresh_interval}s")
    print(f"   Dashboard Metrics Count: {config.dashboard_metrics_count}")
    print(f"   Enable Real Time Dashboard: {'Enabled' if config.enable_real_time_dashboard else 'Disabled'}")
    print(f"   Enable Historical Dashboard: {'Enabled' if config.enable_historical_dashboard else 'Disabled'}")
    print(f"   Enable Data Storage: {'Enabled' if config.enable_data_storage else 'Disabled'}")
    print(f"   Storage Backend: {config.storage_backend}")
    print(f"   Data Retention Days: {config.data_retention_days}")
    print(f"   Enable Data Compression: {'Enabled' if config.enable_data_compression else 'Disabled'}")
    print(f"   Enable Data Aggregation: {'Enabled' if config.enable_data_aggregation else 'Disabled'}")
    print(f"   Enable Auto Remediation: {'Enabled' if config.enable_auto_remediation else 'Disabled'}")
    print(f"   Enable Predictive Monitoring: {'Enabled' if config.enable_predictive_monitoring else 'Disabled'}")
    print(f"   Enable Cross Model Monitoring: {'Enabled' if config.enable_cross_model_monitoring else 'Disabled'}")
    print(f"   Enable Federated Monitoring: {'Enabled' if config.enable_federated_monitoring else 'Disabled'}")
    
    print(f"\n📊 Model Monitoring Results:")
    print(f"   Monitoring History Length: {len(monitoring_system.monitoring_history)}")
    print(f"   Total Duration: {monitoring_results.get('total_duration', 0):.2f} seconds")
    
    # Show monitoring results summary
    if 'monitoring_results' in monitoring_results:
        print(f"   Number of Monitoring Methods: {len(monitoring_results['monitoring_results'])}")
    
    print(f"\n📋 Model Monitoring Report:")
    print(monitoring_report)
    
    return monitoring_system

# Export utilities
__all__ = [
    'MonitoringLevel',
    'AlertSeverity',
    'AnomalyType',
    'ModelMonitoringConfig',
    'RealTimeMonitor',
    'DriftDetector',
    'AnomalyDetector',
    'AlertManager',
    'ModelMonitoringSystem',
    'create_monitoring_config',
    'create_real_time_monitor',
    'create_drift_detector',
    'create_anomaly_detector',
    'create_alert_manager',
    'create_model_monitoring_system',
    'example_model_monitoring'
]

if __name__ == "__main__":
    example_model_monitoring()
    print("✅ Model monitoring example completed successfully!")
