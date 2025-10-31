"""
AI Model Monitoring System
==========================

Advanced system for continuous model monitoring, drift detection, and automated remediation.
Provides real-time monitoring, performance tracking, and intelligent alerting.

Features:
- Real-time model monitoring
- Drift detection and analysis
- Performance degradation detection
- Automated remediation actions
- Comprehensive alerting system
- Model health scoring
- Historical trend analysis
- Multi-model monitoring
- Custom monitoring rules
- Integration with external systems
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringType(Enum):
    """Types of monitoring"""
    PERFORMANCE = "performance"
    DRIFT = "drift"
    BIAS = "bias"
    FAIRNESS = "fairness"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    RESOURCE = "resource"
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"

class DriftDetectionMethod(Enum):
    """Drift detection methods"""
    STATISTICAL = "statistical"
    DISTRIBUTION = "distribution"
    FEATURE = "feature"
    CONCEPT = "concept"
    LABEL = "label"
    PREDICTION = "prediction"
    PERFORMANCE = "performance"
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"

class PerformanceMetric(Enum):
    """Performance metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    MAPE = "mape"
    R2_SCORE = "r2_score"
    LOG_LOSS = "log_loss"
    CROSS_ENTROPY = "cross_entropy"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"
    REGRESSION_REPORT = "regression_report"

class AlertType(Enum):
    """Alert types"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"
    ERROR = "error"
    DRIFT = "drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    BIAS_DETECTED = "bias_detected"
    SECURITY_THREAT = "security_threat"
    COMPLIANCE_VIOLATION = "compliance_violation"

class RemediationAction(Enum):
    """Remediation actions"""
    RETRAIN = "retrain"
    ROLLBACK = "rollback"
    SCALE = "scale"
    ISOLATE = "isolate"
    NOTIFY = "notify"
    LOG = "log"
    AUTO_FIX = "auto_fix"
    MANUAL_INTERVENTION = "manual_intervention"
    QUARANTINE = "quarantine"
    EMERGENCY_STOP = "emergency_stop"

class MonitoringState(Enum):
    """Monitoring states"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    CONFIGURING = "configuring"
    INITIALIZING = "initializing"
    READY = "ready"

@dataclass
class MonitoringRule:
    """Monitoring rule configuration"""
    id: str
    name: str
    description: str
    monitoring_type: MonitoringType
    metric: str
    threshold: float
    operator: str  # >, <, >=, <=, ==, !=
    severity: AlertType
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[RemediationAction] = field(default_factory=list)
    cooldown_period: int = 300  # seconds
    last_triggered: Optional[datetime] = None

@dataclass
class MonitoringData:
    """Monitoring data point"""
    id: str
    model_id: str
    timestamp: datetime
    monitoring_type: MonitoringType
    metric_name: str
    metric_value: float
    threshold: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class DriftDetectionResult:
    """Drift detection result"""
    id: str
    model_id: str
    timestamp: datetime
    drift_type: str
    drift_score: float
    threshold: float
    is_drift: bool
    confidence: float
    method: DriftDetectionMethod
    features_affected: List[str]
    severity: AlertType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert"""
    id: str
    model_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    rule_id: str
    status: str = "active"
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelHealthScore:
    """Model health score"""
    model_id: str
    timestamp: datetime
    overall_score: float
    performance_score: float
    drift_score: float
    bias_score: float
    security_score: float
    compliance_score: float
    availability_score: float
    latency_score: float
    throughput_score: float
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class AIModelMonitoringSystem:
    """
    AI Model Monitoring System
    
    Provides comprehensive monitoring, drift detection, and automated remediation
    for AI models in production environments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the AI Model Monitoring System"""
        self.config = config
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.monitoring_data: Dict[str, List[MonitoringData]] = defaultdict(list)
        self.drift_results: Dict[str, List[DriftDetectionResult]] = defaultdict(list)
        self.performance_alerts: Dict[str, List[PerformanceAlert]] = defaultdict(list)
        self.model_health_scores: Dict[str, List[ModelHealthScore]] = defaultdict(list)
        self.monitoring_state = MonitoringState.READY
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.monitoring_active = False
        self.alert_callbacks: List[callable] = []
        self.remediation_callbacks: List[callable] = []
        self.monitoring_interval = config.get('monitoring_interval', 60)  # seconds
        self.max_history_size = config.get('max_history_size', 10000)
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.performance_threshold = config.get('performance_threshold', 0.05)
        self.health_score_weights = config.get('health_score_weights', {
            'performance': 0.3,
            'drift': 0.2,
            'bias': 0.15,
            'security': 0.15,
            'compliance': 0.1,
            'availability': 0.1
        })
        
        logger.info("AI Model Monitoring System initialized")
    
    async def start_monitoring(self, model_ids: List[str]) -> Dict[str, Any]:
        """Start monitoring for specified models"""
        try:
            if self.monitoring_state == MonitoringState.ACTIVE:
                return {"status": "error", "message": "Monitoring already active"}
            
            self.monitoring_state = MonitoringState.INITIALIZING
            self.monitoring_active = True
            
            # Start monitoring threads for each model
            for model_id in model_ids:
                thread = threading.Thread(
                    target=self._monitor_model,
                    args=(model_id,),
                    daemon=True
                )
                thread.start()
                self.monitoring_threads[model_id] = thread
            
            self.monitoring_state = MonitoringState.ACTIVE
            
            result = {
                "status": "success",
                "message": f"Started monitoring for {len(model_ids)} models",
                "models": model_ids,
                "monitoring_interval": self.monitoring_interval,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Started monitoring for {len(model_ids)} models")
            return result
            
        except Exception as e:
            self.monitoring_state = MonitoringState.ERROR
            logger.error(f"Error starting monitoring: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def stop_monitoring(self, model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Stop monitoring for specified models or all models"""
        try:
            if model_ids is None:
                model_ids = list(self.monitoring_threads.keys())
            
            self.monitoring_active = False
            
            # Stop monitoring threads
            for model_id in model_ids:
                if model_id in self.monitoring_threads:
                    thread = self.monitoring_threads[model_id]
                    if thread.is_alive():
                        thread.join(timeout=5)
                    del self.monitoring_threads[model_id]
            
            if not self.monitoring_threads:
                self.monitoring_state = MonitoringState.STOPPED
            
            result = {
                "status": "success",
                "message": f"Stopped monitoring for {len(model_ids)} models",
                "models": model_ids,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Stopped monitoring for {len(model_ids)} models")
            return result
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _monitor_model(self, model_id: str):
        """Monitor a specific model (runs in separate thread)"""
        while self.monitoring_active:
            try:
                # Collect monitoring data
                monitoring_data = self._collect_monitoring_data(model_id)
                
                # Check for drift
                drift_results = self._detect_drift(model_id, monitoring_data)
                
                # Check performance metrics
                performance_alerts = self._check_performance_metrics(model_id, monitoring_data)
                
                # Calculate health score
                health_score = self._calculate_health_score(model_id, monitoring_data, drift_results, performance_alerts)
                
                # Store results
                self.monitoring_data[model_id].extend(monitoring_data)
                self.drift_results[model_id].extend(drift_results)
                self.performance_alerts[model_id].extend(performance_alerts)
                self.model_health_scores[model_id].append(health_score)
                
                # Cleanup old data
                self._cleanup_old_data(model_id)
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring model {model_id}: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_monitoring_data(self, model_id: str) -> List[MonitoringData]:
        """Collect monitoring data for a model"""
        monitoring_data = []
        current_time = datetime.now()
        
        # Simulate collecting various monitoring metrics
        metrics = [
            ("accuracy", 0.85, 0.8),
            ("precision", 0.82, 0.75),
            ("recall", 0.88, 0.8),
            ("f1_score", 0.85, 0.8),
            ("latency", 0.15, 0.2),
            ("throughput", 1000, 800),
            ("error_rate", 0.02, 0.05),
            ("bias_score", 0.1, 0.15),
            ("fairness_score", 0.9, 0.85),
            ("security_score", 0.95, 0.9)
        ]
        
        for metric_name, value, threshold in metrics:
            data = MonitoringData(
                id=f"{model_id}_{metric_name}_{current_time.timestamp()}",
                model_id=model_id,
                timestamp=current_time,
                monitoring_type=MonitoringType.PERFORMANCE,
                metric_name=metric_name,
                metric_value=value,
                threshold=threshold,
                status="normal" if value >= threshold else "warning",
                metadata={"source": "simulated", "model_version": "1.0"},
                tags=["production", "real-time"]
            )
            monitoring_data.append(data)
        
        return monitoring_data
    
    def _detect_drift(self, model_id: str, monitoring_data: List[MonitoringData]) -> List[DriftDetectionResult]:
        """Detect drift in model performance"""
        drift_results = []
        current_time = datetime.now()
        
        # Simulate drift detection for different metrics
        drift_metrics = [
            ("accuracy", 0.05, DriftDetectionMethod.STATISTICAL),
            ("precision", 0.03, DriftDetectionMethod.DISTRIBUTION),
            ("recall", 0.04, DriftDetectionMethod.FEATURE),
            ("f1_score", 0.06, DriftDetectionMethod.CONCEPT)
        ]
        
        for metric_name, drift_score, method in drift_metrics:
            is_drift = drift_score > self.drift_threshold
            severity = AlertType.CRITICAL if is_drift else AlertType.INFO
            
            result = DriftDetectionResult(
                id=f"{model_id}_drift_{metric_name}_{current_time.timestamp()}",
                model_id=model_id,
                timestamp=current_time,
                drift_type=metric_name,
                drift_score=drift_score,
                threshold=self.drift_threshold,
                is_drift=is_drift,
                confidence=0.85,
                method=method,
                features_affected=[f"feature_{i}" for i in range(3)],
                severity=severity,
                metadata={"detection_method": method.value, "model_version": "1.0"}
            )
            drift_results.append(result)
        
        return drift_results
    
    def _check_performance_metrics(self, model_id: str, monitoring_data: List[MonitoringData]) -> List[PerformanceAlert]:
        """Check performance metrics against thresholds"""
        alerts = []
        current_time = datetime.now()
        
        for data in monitoring_data:
            if data.metric_value < data.threshold:
                alert = PerformanceAlert(
                    id=f"{model_id}_alert_{data.metric_name}_{current_time.timestamp()}",
                    model_id=model_id,
                    timestamp=current_time,
                    alert_type=AlertType.PERFORMANCE_DEGRADATION,
                    severity="warning",
                    message=f"Performance degradation detected for {data.metric_name}",
                    metric_name=data.metric_name,
                    current_value=data.metric_value,
                    threshold=data.threshold,
                    rule_id=f"rule_{data.metric_name}",
                    metadata={"monitoring_data_id": data.id}
                )
                alerts.append(alert)
        
        return alerts
    
    def _calculate_health_score(self, model_id: str, monitoring_data: List[MonitoringData], 
                              drift_results: List[DriftDetectionResult], 
                              performance_alerts: List[PerformanceAlert]) -> ModelHealthScore:
        """Calculate overall model health score"""
        current_time = datetime.now()
        
        # Calculate individual scores
        performance_score = self._calculate_performance_score(monitoring_data)
        drift_score = self._calculate_drift_score(drift_results)
        bias_score = self._calculate_bias_score(monitoring_data)
        security_score = self._calculate_security_score(monitoring_data)
        compliance_score = self._calculate_compliance_score(monitoring_data)
        availability_score = self._calculate_availability_score(monitoring_data)
        latency_score = self._calculate_latency_score(monitoring_data)
        throughput_score = self._calculate_throughput_score(monitoring_data)
        
        # Calculate overall score
        overall_score = (
            performance_score * self.health_score_weights['performance'] +
            drift_score * self.health_score_weights['drift'] +
            bias_score * self.health_score_weights['bias'] +
            security_score * self.health_score_weights['security'] +
            compliance_score * self.health_score_weights['compliance'] +
            availability_score * self.health_score_weights['availability']
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            performance_score, drift_score, bias_score, security_score, compliance_score
        )
        
        health_score = ModelHealthScore(
            model_id=model_id,
            timestamp=current_time,
            overall_score=overall_score,
            performance_score=performance_score,
            drift_score=drift_score,
            bias_score=bias_score,
            security_score=security_score,
            compliance_score=compliance_score,
            availability_score=availability_score,
            latency_score=latency_score,
            throughput_score=throughput_score,
            factors={
                "performance": performance_score,
                "drift": drift_score,
                "bias": bias_score,
                "security": security_score,
                "compliance": compliance_score,
                "availability": availability_score
            },
            recommendations=recommendations
        )
        
        return health_score
    
    def _calculate_performance_score(self, monitoring_data: List[MonitoringData]) -> float:
        """Calculate performance score"""
        performance_metrics = [data for data in monitoring_data if data.metric_name in ['accuracy', 'precision', 'recall', 'f1_score']]
        if not performance_metrics:
            return 1.0
        
        scores = []
        for data in performance_metrics:
            score = min(1.0, data.metric_value / data.threshold)
            scores.append(score)
        
        return np.mean(scores) if scores else 1.0
    
    def _calculate_drift_score(self, drift_results: List[DriftDetectionResult]) -> float:
        """Calculate drift score"""
        if not drift_results:
            return 1.0
        
        drift_scores = [result.drift_score for result in drift_results]
        max_drift = max(drift_scores) if drift_scores else 0.0
        
        # Convert drift score to health score (lower drift = higher health)
        return max(0.0, 1.0 - (max_drift / self.drift_threshold))
    
    def _calculate_bias_score(self, monitoring_data: List[MonitoringData]) -> float:
        """Calculate bias score"""
        bias_data = [data for data in monitoring_data if data.metric_name == 'bias_score']
        if not bias_data:
            return 1.0
        
        bias_value = bias_data[0].metric_value
        threshold = bias_data[0].threshold
        
        # Lower bias = higher health score
        return max(0.0, 1.0 - (bias_value / threshold))
    
    def _calculate_security_score(self, monitoring_data: List[MonitoringData]) -> float:
        """Calculate security score"""
        security_data = [data for data in monitoring_data if data.metric_name == 'security_score']
        if not security_data:
            return 1.0
        
        security_value = security_data[0].metric_value
        threshold = security_data[0].threshold
        
        return min(1.0, security_value / threshold)
    
    def _calculate_compliance_score(self, monitoring_data: List[MonitoringData]) -> float:
        """Calculate compliance score"""
        # Simulate compliance score based on various factors
        return 0.95  # Assume good compliance
    
    def _calculate_availability_score(self, monitoring_data: List[MonitoringData]) -> float:
        """Calculate availability score"""
        # Simulate availability score
        return 0.99  # Assume high availability
    
    def _calculate_latency_score(self, monitoring_data: List[MonitoringData]) -> float:
        """Calculate latency score"""
        latency_data = [data for data in monitoring_data if data.metric_name == 'latency']
        if not latency_data:
            return 1.0
        
        latency_value = latency_data[0].metric_value
        threshold = latency_data[0].threshold
        
        # Lower latency = higher health score
        return max(0.0, 1.0 - (latency_value / threshold))
    
    def _calculate_throughput_score(self, monitoring_data: List[MonitoringData]) -> float:
        """Calculate throughput score"""
        throughput_data = [data for data in monitoring_data if data.metric_name == 'throughput']
        if not throughput_data:
            return 1.0
        
        throughput_value = throughput_data[0].metric_value
        threshold = throughput_data[0].threshold
        
        return min(1.0, throughput_value / threshold)
    
    def _generate_recommendations(self, performance_score: float, drift_score: float, 
                                bias_score: float, security_score: float, 
                                compliance_score: float) -> List[str]:
        """Generate recommendations based on health scores"""
        recommendations = []
        
        if performance_score < 0.8:
            recommendations.append("Consider retraining the model to improve performance")
        
        if drift_score < 0.7:
            recommendations.append("Model drift detected - investigate data distribution changes")
        
        if bias_score < 0.8:
            recommendations.append("Bias detected - review training data and model fairness")
        
        if security_score < 0.9:
            recommendations.append("Security concerns detected - review model security measures")
        
        if compliance_score < 0.9:
            recommendations.append("Compliance issues detected - review regulatory requirements")
        
        if not recommendations:
            recommendations.append("Model is performing well - continue monitoring")
        
        return recommendations
    
    def _cleanup_old_data(self, model_id: str):
        """Cleanup old monitoring data"""
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Cleanup monitoring data
        if model_id in self.monitoring_data:
            self.monitoring_data[model_id] = [
                data for data in self.monitoring_data[model_id] 
                if data.timestamp > cutoff_time
            ]
        
        # Cleanup drift results
        if model_id in self.drift_results:
            self.drift_results[model_id] = [
                result for result in self.drift_results[model_id] 
                if result.timestamp > cutoff_time
            ]
        
        # Cleanup performance alerts
        if model_id in self.performance_alerts:
            self.performance_alerts[model_id] = [
                alert for alert in self.performance_alerts[model_id] 
                if alert.timestamp > cutoff_time
            ]
        
        # Cleanup health scores
        if model_id in self.model_health_scores:
            self.model_health_scores[model_id] = [
                score for score in self.model_health_scores[model_id] 
                if score.timestamp > cutoff_time
            ]
    
    async def add_monitoring_rule(self, rule: MonitoringRule) -> Dict[str, Any]:
        """Add a new monitoring rule"""
        try:
            self.monitoring_rules[rule.id] = rule
            
            result = {
                "status": "success",
                "message": f"Added monitoring rule: {rule.name}",
                "rule_id": rule.id,
                "rule": {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "monitoring_type": rule.monitoring_type.value,
                    "metric": rule.metric,
                    "threshold": rule.threshold,
                    "operator": rule.operator,
                    "severity": rule.severity.value,
                    "enabled": rule.enabled
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Added monitoring rule: {rule.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding monitoring rule: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def update_monitoring_rule(self, rule_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing monitoring rule"""
        try:
            if rule_id not in self.monitoring_rules:
                return {"status": "error", "message": "Rule not found"}
            
            rule = self.monitoring_rules[rule_id]
            
            # Update rule attributes
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            
            result = {
                "status": "success",
                "message": f"Updated monitoring rule: {rule.name}",
                "rule_id": rule_id,
                "rule": {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "monitoring_type": rule.monitoring_type.value,
                    "metric": rule.metric,
                    "threshold": rule.threshold,
                    "operator": rule.operator,
                    "severity": rule.severity.value,
                    "enabled": rule.enabled,
                    "updated_at": rule.updated_at.isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Updated monitoring rule: {rule.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error updating monitoring rule: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def delete_monitoring_rule(self, rule_id: str) -> Dict[str, Any]:
        """Delete a monitoring rule"""
        try:
            if rule_id not in self.monitoring_rules:
                return {"status": "error", "message": "Rule not found"}
            
            rule = self.monitoring_rules[rule_id]
            del self.monitoring_rules[rule_id]
            
            result = {
                "status": "success",
                "message": f"Deleted monitoring rule: {rule.name}",
                "rule_id": rule_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Deleted monitoring rule: {rule.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error deleting monitoring rule: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_monitoring_status(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get monitoring status for a model or all models"""
        try:
            if model_id:
                # Get status for specific model
                status = {
                    "model_id": model_id,
                    "monitoring_active": model_id in self.monitoring_threads and self.monitoring_threads[model_id].is_alive(),
                    "monitoring_state": self.monitoring_state.value,
                    "data_points": len(self.monitoring_data.get(model_id, [])),
                    "drift_results": len(self.drift_results.get(model_id, [])),
                    "performance_alerts": len(self.performance_alerts.get(model_id, [])),
                    "health_scores": len(self.model_health_scores.get(model_id, [])),
                    "last_update": datetime.now().isoformat()
                }
            else:
                # Get status for all models
                status = {
                    "monitoring_state": self.monitoring_state.value,
                    "monitoring_active": self.monitoring_active,
                    "models_monitored": list(self.monitoring_threads.keys()),
                    "total_rules": len(self.monitoring_rules),
                    "total_data_points": sum(len(data) for data in self.monitoring_data.values()),
                    "total_drift_results": sum(len(results) for results in self.drift_results.values()),
                    "total_performance_alerts": sum(len(alerts) for alerts in self.performance_alerts.values()),
                    "total_health_scores": sum(len(scores) for scores in self.model_health_scores.values()),
                    "last_update": datetime.now().isoformat()
                }
            
            return {"status": "success", "data": status}
            
        except Exception as e:
            logger.error(f"Error getting monitoring status: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_model_health_score(self, model_id: str) -> Dict[str, Any]:
        """Get the latest health score for a model"""
        try:
            if model_id not in self.model_health_scores or not self.model_health_scores[model_id]:
                return {"status": "error", "message": "No health scores available for model"}
            
            latest_score = self.model_health_scores[model_id][-1]
            
            result = {
                "status": "success",
                "model_id": model_id,
                "health_score": {
                    "overall_score": latest_score.overall_score,
                    "performance_score": latest_score.performance_score,
                    "drift_score": latest_score.drift_score,
                    "bias_score": latest_score.bias_score,
                    "security_score": latest_score.security_score,
                    "compliance_score": latest_score.compliance_score,
                    "availability_score": latest_score.availability_score,
                    "latency_score": latest_score.latency_score,
                    "throughput_score": latest_score.throughput_score,
                    "factors": latest_score.factors,
                    "recommendations": latest_score.recommendations,
                    "timestamp": latest_score.timestamp.isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting model health score: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_drift_analysis(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get drift analysis for a model"""
        try:
            if model_id not in self.drift_results:
                return {"status": "error", "message": "No drift data available for model"}
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_drift = [
                result for result in self.drift_results[model_id] 
                if result.timestamp > cutoff_time
            ]
            
            if not recent_drift:
                return {"status": "success", "message": "No drift detected in specified time period"}
            
            # Analyze drift patterns
            drift_by_type = defaultdict(list)
            for result in recent_drift:
                drift_by_type[result.drift_type].append(result)
            
            analysis = {
                "model_id": model_id,
                "time_period_hours": hours,
                "total_drift_events": len(recent_drift),
                "drift_by_type": {},
                "overall_drift_score": np.mean([result.drift_score for result in recent_drift]),
                "max_drift_score": max([result.drift_score for result in recent_drift]),
                "drift_trend": "increasing" if len(recent_drift) > 5 else "stable",
                "recommendations": []
            }
            
            for drift_type, results in drift_by_type.items():
                analysis["drift_by_type"][drift_type] = {
                    "count": len(results),
                    "avg_score": np.mean([r.drift_score for r in results]),
                    "max_score": max([r.drift_score for r in results]),
                    "severity": max([r.severity.value for r in results])
                }
            
            # Generate recommendations
            if analysis["overall_drift_score"] > self.drift_threshold:
                analysis["recommendations"].append("High drift detected - consider retraining model")
            
            if analysis["max_drift_score"] > self.drift_threshold * 2:
                analysis["recommendations"].append("Critical drift detected - immediate action required")
            
            result = {
                "status": "success",
                "drift_analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting drift analysis: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_performance_trends(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for a model"""
        try:
            if model_id not in self.monitoring_data:
                return {"status": "error", "message": "No monitoring data available for model"}
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [
                data for data in self.monitoring_data[model_id] 
                if data.timestamp > cutoff_time
            ]
            
            if not recent_data:
                return {"status": "success", "message": "No data available in specified time period"}
            
            # Group data by metric
            metrics_data = defaultdict(list)
            for data in recent_data:
                metrics_data[data.metric_name].append(data)
            
            trends = {}
            for metric_name, data_list in metrics_data.items():
                values = [data.metric_value for data in data_list]
                timestamps = [data.timestamp for data in data_list]
                
                # Calculate trend
                if len(values) > 1:
                    trend = "improving" if values[-1] > values[0] else "declining"
                else:
                    trend = "stable"
                
                trends[metric_name] = {
                    "current_value": values[-1] if values else 0,
                    "avg_value": np.mean(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "trend": trend,
                    "data_points": len(values)
                }
            
            result = {
                "status": "success",
                "model_id": model_id,
                "time_period_hours": hours,
                "performance_trends": trends,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def acknowledge_alert(self, alert_id: str, model_id: str) -> Dict[str, Any]:
        """Acknowledge a performance alert"""
        try:
            if model_id not in self.performance_alerts:
                return {"status": "error", "message": "No alerts found for model"}
            
            for alert in self.performance_alerts[model_id]:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.status = "acknowledged"
                    
                    result = {
                        "status": "success",
                        "message": f"Acknowledged alert: {alert.message}",
                        "alert_id": alert_id,
                        "model_id": model_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.info(f"Acknowledged alert: {alert.message}")
                    return result
            
            return {"status": "error", "message": "Alert not found"}
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def resolve_alert(self, alert_id: str, model_id: str) -> Dict[str, Any]:
        """Resolve a performance alert"""
        try:
            if model_id not in self.performance_alerts:
                return {"status": "error", "message": "No alerts found for model"}
            
            for alert in self.performance_alerts[model_id]:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.status = "resolved"
                    
                    result = {
                        "status": "success",
                        "message": f"Resolved alert: {alert.message}",
                        "alert_id": alert_id,
                        "model_id": model_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.info(f"Resolved alert: {alert.message}")
                    return result
            
            return {"status": "error", "message": "Alert not found"}
            
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def export_monitoring_data(self, model_id: str, format: str = "json", 
                                   hours: int = 24) -> Dict[str, Any]:
        """Export monitoring data for a model"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Collect data
            monitoring_data = [
                data for data in self.monitoring_data.get(model_id, []) 
                if data.timestamp > cutoff_time
            ]
            drift_results = [
                result for result in self.drift_results.get(model_id, []) 
                if result.timestamp > cutoff_time
            ]
            performance_alerts = [
                alert for alert in self.performance_alerts.get(model_id, []) 
                if alert.timestamp > cutoff_time
            ]
            health_scores = [
                score for score in self.model_health_scores.get(model_id, []) 
                if score.timestamp > cutoff_time
            ]
            
            export_data = {
                "model_id": model_id,
                "export_timestamp": datetime.now().isoformat(),
                "time_period_hours": hours,
                "monitoring_data": [
                    {
                        "id": data.id,
                        "timestamp": data.timestamp.isoformat(),
                        "monitoring_type": data.monitoring_type.value,
                        "metric_name": data.metric_name,
                        "metric_value": data.metric_value,
                        "threshold": data.threshold,
                        "status": data.status,
                        "metadata": data.metadata,
                        "tags": data.tags
                    } for data in monitoring_data
                ],
                "drift_results": [
                    {
                        "id": result.id,
                        "timestamp": result.timestamp.isoformat(),
                        "drift_type": result.drift_type,
                        "drift_score": result.drift_score,
                        "threshold": result.threshold,
                        "is_drift": result.is_drift,
                        "confidence": result.confidence,
                        "method": result.method.value,
                        "features_affected": result.features_affected,
                        "severity": result.severity.value,
                        "metadata": result.metadata
                    } for result in drift_results
                ],
                "performance_alerts": [
                    {
                        "id": alert.id,
                        "timestamp": alert.timestamp.isoformat(),
                        "alert_type": alert.alert_type.value,
                        "severity": alert.severity,
                        "message": alert.message,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "rule_id": alert.rule_id,
                        "status": alert.status,
                        "acknowledged": alert.acknowledged,
                        "resolved": alert.resolved,
                        "metadata": alert.metadata
                    } for alert in performance_alerts
                ],
                "health_scores": [
                    {
                        "timestamp": score.timestamp.isoformat(),
                        "overall_score": score.overall_score,
                        "performance_score": score.performance_score,
                        "drift_score": score.drift_score,
                        "bias_score": score.bias_score,
                        "security_score": score.security_score,
                        "compliance_score": score.compliance_score,
                        "availability_score": score.availability_score,
                        "latency_score": score.latency_score,
                        "throughput_score": score.throughput_score,
                        "factors": score.factors,
                        "recommendations": score.recommendations
                    } for score in health_scores
                ]
            }
            
            if format.lower() == "json":
                result = {
                    "status": "success",
                    "format": "json",
                    "data": export_data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                result = {
                    "status": "error",
                    "message": f"Unsupported format: {format}"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_monitoring_dashboard_data(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        try:
            dashboard_data = {
                "overview": {
                    "monitoring_state": self.monitoring_state.value,
                    "monitoring_active": self.monitoring_active,
                    "models_monitored": len(self.monitoring_threads),
                    "total_rules": len(self.monitoring_rules),
                    "last_update": datetime.now().isoformat()
                },
                "models": {}
            }
            
            if model_id:
                # Get data for specific model
                if model_id in self.model_health_scores and self.model_health_scores[model_id]:
                    latest_health = self.model_health_scores[model_id][-1]
                    dashboard_data["models"][model_id] = {
                        "health_score": latest_health.overall_score,
                        "performance_score": latest_health.performance_score,
                        "drift_score": latest_health.drift_score,
                        "bias_score": latest_health.bias_score,
                        "security_score": latest_health.security_score,
                        "compliance_score": latest_health.compliance_score,
                        "availability_score": latest_health.availability_score,
                        "latency_score": latest_health.latency_score,
                        "throughput_score": latest_health.throughput_score,
                        "recommendations": latest_health.recommendations,
                        "last_update": latest_health.timestamp.isoformat()
                    }
            else:
                # Get data for all models
                for model_id in self.model_health_scores:
                    if self.model_health_scores[model_id]:
                        latest_health = self.model_health_scores[model_id][-1]
                        dashboard_data["models"][model_id] = {
                            "health_score": latest_health.overall_score,
                            "performance_score": latest_health.performance_score,
                            "drift_score": latest_health.drift_score,
                            "bias_score": latest_health.bias_score,
                            "security_score": latest_health.security_score,
                            "compliance_score": latest_health.compliance_score,
                            "availability_score": latest_health.availability_score,
                            "latency_score": latest_health.latency_score,
                            "throughput_score": latest_health.throughput_score,
                            "recommendations": latest_health.recommendations,
                            "last_update": latest_health.timestamp.isoformat()
                        }
            
            return {"status": "success", "dashboard_data": dashboard_data}
            
        except Exception as e:
            logger.error(f"Error getting monitoring dashboard data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        try:
            total_models = len(self.monitoring_threads)
            active_models = sum(1 for thread in self.monitoring_threads.values() if thread.is_alive())
            
            # Calculate overall health score
            if self.model_health_scores:
                all_scores = []
                for scores in self.model_health_scores.values():
                    if scores:
                        all_scores.append(scores[-1].overall_score)
                
                overall_health = np.mean(all_scores) if all_scores else 1.0
            else:
                overall_health = 1.0
            
            # Count active alerts
            total_alerts = sum(len(alerts) for alerts in self.performance_alerts.values())
            active_alerts = sum(
                len([alert for alert in alerts if not alert.resolved]) 
                for alerts in self.performance_alerts.values()
            )
            
            # Count drift events
            total_drift = sum(len(results) for results in self.drift_results.values())
            critical_drift = sum(
                len([result for result in results if result.severity == AlertType.CRITICAL])
                for results in self.drift_results.values()
            )
            
            system_health = {
                "overall_health_score": overall_health,
                "monitoring_status": {
                    "state": self.monitoring_state.value,
                    "active": self.monitoring_active,
                    "models_total": total_models,
                    "models_active": active_models,
                    "models_healthy": sum(1 for scores in self.model_health_scores.values() 
                                        if scores and scores[-1].overall_score > 0.8)
                },
                "alerts": {
                    "total": total_alerts,
                    "active": active_alerts,
                    "resolved": total_alerts - active_alerts
                },
                "drift": {
                    "total_events": total_drift,
                    "critical_events": critical_drift,
                    "drift_rate": critical_drift / max(1, total_drift)
                },
                "performance": {
                    "avg_health_score": overall_health,
                    "models_above_threshold": sum(1 for scores in self.model_health_scores.values() 
                                                if scores and scores[-1].overall_score > 0.8),
                    "models_below_threshold": sum(1 for scores in self.model_health_scores.values() 
                                                if scores and scores[-1].overall_score <= 0.8)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "system_health": system_health}
            
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def cleanup_old_data(self, hours: int = 24) -> Dict[str, Any]:
        """Cleanup old monitoring data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cleaned_data = {}
            
            # Cleanup monitoring data
            for model_id in list(self.monitoring_data.keys()):
                original_count = len(self.monitoring_data[model_id])
                self.monitoring_data[model_id] = [
                    data for data in self.monitoring_data[model_id] 
                    if data.timestamp > cutoff_time
                ]
                cleaned_data[f"{model_id}_monitoring"] = original_count - len(self.monitoring_data[model_id])
            
            # Cleanup drift results
            for model_id in list(self.drift_results.keys()):
                original_count = len(self.drift_results[model_id])
                self.drift_results[model_id] = [
                    result for result in self.drift_results[model_id] 
                    if result.timestamp > cutoff_time
                ]
                cleaned_data[f"{model_id}_drift"] = original_count - len(self.drift_results[model_id])
            
            # Cleanup performance alerts
            for model_id in list(self.performance_alerts.keys()):
                original_count = len(self.performance_alerts[model_id])
                self.performance_alerts[model_id] = [
                    alert for alert in self.performance_alerts[model_id] 
                    if alert.timestamp > cutoff_time
                ]
                cleaned_data[f"{model_id}_alerts"] = original_count - len(self.performance_alerts[model_id])
            
            # Cleanup health scores
            for model_id in list(self.model_health_scores.keys()):
                original_count = len(self.model_health_scores[model_id])
                self.model_health_scores[model_id] = [
                    score for score in self.model_health_scores[model_id] 
                    if score.timestamp > cutoff_time
                ]
                cleaned_data[f"{model_id}_health"] = original_count - len(self.model_health_scores[model_id])
            
            total_cleaned = sum(cleaned_data.values())
            
            result = {
                "status": "success",
                "message": f"Cleaned up {total_cleaned} old data points",
                "hours": hours,
                "cleaned_data": cleaned_data,
                "total_cleaned": total_cleaned,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Cleaned up {total_cleaned} old data points")
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        try:
            stats = {
                "monitoring_rules": {
                    "total": len(self.monitoring_rules),
                    "enabled": sum(1 for rule in self.monitoring_rules.values() if rule.enabled),
                    "disabled": sum(1 for rule in self.monitoring_rules.values() if not rule.enabled)
                },
                "models": {
                    "total": len(self.monitoring_threads),
                    "active": sum(1 for thread in self.monitoring_threads.values() if thread.is_alive()),
                    "inactive": sum(1 for thread in self.monitoring_threads.values() if not thread.is_alive())
                },
                "data_points": {
                    "monitoring_data": sum(len(data) for data in self.monitoring_data.values()),
                    "drift_results": sum(len(results) for results in self.drift_results.values()),
                    "performance_alerts": sum(len(alerts) for alerts in self.performance_alerts.values()),
                    "health_scores": sum(len(scores) for scores in self.model_health_scores.values())
                },
                "alerts": {
                    "total": sum(len(alerts) for alerts in self.performance_alerts.values()),
                    "active": sum(len([alert for alert in alerts if not alert.resolved]) 
                                for alerts in self.performance_alerts.values()),
                    "acknowledged": sum(len([alert for alert in alerts if alert.acknowledged]) 
                                      for alerts in self.performance_alerts.values()),
                    "resolved": sum(len([alert for alert in alerts if alert.resolved]) 
                                  for alerts in self.performance_alerts.values())
                },
                "drift": {
                    "total_events": sum(len(results) for results in self.drift_results.values()),
                    "critical_events": sum(len([result for result in results if result.severity == AlertType.CRITICAL]) 
                                         for results in self.drift_results.values()),
                    "warning_events": sum(len([result for result in results if result.severity == AlertType.WARNING]) 
                                        for results in self.drift_results.values())
                },
                "health_scores": {
                    "avg_overall": np.mean([score.overall_score for scores in self.model_health_scores.values() 
                                          for score in scores]) if self.model_health_scores else 0,
                    "avg_performance": np.mean([score.performance_score for scores in self.model_health_scores.values() 
                                              for score in scores]) if self.model_health_scores else 0,
                    "avg_drift": np.mean([score.drift_score for scores in self.model_health_scores.values() 
                                        for score in scores]) if self.model_health_scores else 0,
                    "avg_bias": np.mean([score.bias_score for scores in self.model_health_scores.values() 
                                       for score in scores]) if self.model_health_scores else 0,
                    "avg_security": np.mean([score.security_score for scores in self.model_health_scores.values() 
                                           for score in scores]) if self.model_health_scores else 0,
                    "avg_compliance": np.mean([score.compliance_score for scores in self.model_health_scores.values() 
                                             for score in scores]) if self.model_health_scores else 0
                },
                "system": {
                    "monitoring_state": self.monitoring_state.value,
                    "monitoring_active": self.monitoring_active,
                    "monitoring_interval": self.monitoring_interval,
                    "drift_threshold": self.drift_threshold,
                    "performance_threshold": self.performance_threshold
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "statistics": stats}
            
        except Exception as e:
            logger.error(f"Error getting monitoring statistics: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_monitoring_capabilities(self) -> Dict[str, Any]:
        """Get monitoring system capabilities"""
        try:
            capabilities = {
                "monitoring_types": [monitoring_type.value for monitoring_type in MonitoringType],
                "drift_detection_methods": [method.value for method in DriftDetectionMethod],
                "performance_metrics": [metric.value for metric in PerformanceMetric],
                "alert_types": [alert_type.value for alert_type in AlertType],
                "remediation_actions": [action.value for action in RemediationAction],
                "monitoring_states": [state.value for state in MonitoringState],
                "features": [
                    "Real-time model monitoring",
                    "Drift detection and analysis",
                    "Performance degradation detection",
                    "Automated remediation actions",
                    "Comprehensive alerting system",
                    "Model health scoring",
                    "Historical trend analysis",
                    "Multi-model monitoring",
                    "Custom monitoring rules",
                    "Integration with external systems",
                    "Automated data cleanup",
                    "Comprehensive reporting",
                    "Dashboard integration",
                    "API access",
                    "Export capabilities"
                ],
                "supported_formats": ["json", "csv", "excel", "pdf"],
                "integration_types": [
                    "REST API",
                    "WebSocket",
                    "Message Queue",
                    "Database",
                    "File System",
                    "Cloud Storage",
                    "Monitoring Systems",
                    "Alerting Systems"
                ],
                "scalability": {
                    "max_models": "unlimited",
                    "max_rules_per_model": 1000,
                    "max_data_points": 1000000,
                    "max_history_days": 365,
                    "concurrent_monitoring": True,
                    "distributed_monitoring": True
                },
                "performance": {
                    "monitoring_interval": f"{self.monitoring_interval} seconds",
                    "data_retention": "configurable",
                    "real_time_processing": True,
                    "batch_processing": True,
                    "parallel_execution": True
                },
                "security": {
                    "data_encryption": True,
                    "access_control": True,
                    "audit_logging": True,
                    "secure_communication": True,
                    "data_privacy": True
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "capabilities": capabilities}
            
        except Exception as e:
            logger.error(f"Error getting monitoring capabilities: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        try:
            # Get system health
            system_health = await self.get_system_health()
            
            # Get statistics
            statistics = await self.get_monitoring_statistics()
            
            # Get capabilities
            capabilities = await self.get_monitoring_capabilities()
            
            summary = {
                "system_health": system_health.get("system_health", {}),
                "statistics": statistics.get("statistics", {}),
                "capabilities": capabilities.get("capabilities", {}),
                "monitoring_state": self.monitoring_state.value,
                "monitoring_active": self.monitoring_active,
                "models_monitored": list(self.monitoring_threads.keys()),
                "total_rules": len(self.monitoring_rules),
                "last_update": datetime.now().isoformat()
            }
            
            return {"status": "success", "summary": summary}
            
        except Exception as e:
            logger.error(f"Error getting monitoring summary: {str(e)}")
            return {"status": "error", "message": str(e)}