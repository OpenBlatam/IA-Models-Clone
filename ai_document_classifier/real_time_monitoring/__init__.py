"""
Real-time Monitoring and Alerting System
========================================

Advanced monitoring system for the AI Document Classifier with real-time metrics,
intelligent alerting, and automated response capabilities.

Modules:
- monitoring_system: Main monitoring system with metrics collection, anomaly detection, and alerting
"""

from .monitoring_system import (
    RealTimeMonitoringSystem,
    MetricsCollector,
    AnomalyDetector,
    AlertManager,
    HealthChecker,
    SystemMetric,
    Alert,
    HealthCheck,
    AlertSeverity,
    AlertStatus,
    monitoring_system,
    app
)

__all__ = [
    "RealTimeMonitoringSystem",
    "MetricsCollector", 
    "AnomalyDetector",
    "AlertManager",
    "HealthChecker",
    "SystemMetric",
    "Alert",
    "HealthCheck",
    "AlertSeverity",
    "AlertStatus",
    "monitoring_system",
    "app"
]
























