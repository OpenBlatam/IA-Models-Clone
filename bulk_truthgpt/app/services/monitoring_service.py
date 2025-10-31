"""
Monitoring service for Ultimate Enhanced Supreme Production system
"""

import time
import logging
from typing import Dict, Any, Optional
from app.models.monitoring import SystemMetrics, PerformanceMetrics, HealthStatus
from app.core.monitoring_core import MonitoringCore

logger = logging.getLogger(__name__)

class MonitoringService:
    """Monitoring service."""
    
    def __init__(self):
        """Initialize service."""
        self.core = MonitoringCore()
        self.logger = logger
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics."""
        try:
            metrics = self.core.get_system_metrics()
            return metrics
        except Exception as e:
            self.logger.error(f"❌ Error getting system metrics: {e}")
            return SystemMetrics(
                supreme_optimization_level='unknown',
                ultra_fast_level='unknown',
                refactored_ultimate_hybrid_level='unknown',
                cuda_kernel_level='unknown',
                gpu_utilization_level='unknown',
                memory_optimization_level='unknown',
                reward_function_level='unknown',
                truthgpt_adapter_level='unknown',
                microservices_level='unknown',
                max_concurrent_generations=0,
                max_documents_per_query=0,
                max_continuous_documents=0,
                ultimate_enhanced_supreme_ready=False,
                ultra_fast_ready=False,
                refactored_ultimate_hybrid_ready=False,
                cuda_kernel_ready=False,
                gpu_utils_ready=False,
                memory_utils_ready=False,
                reward_function_ready=False,
                truthgpt_adapter_ready=False,
                microservices_ready=False,
                ultimate_ready=False,
                ultra_advanced_ready=False,
                advanced_ready=False,
                performance_metrics=PerformanceMetrics()
            )
    
    def get_performance_metrics(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            metrics = self.core.get_performance_metrics(query_params)
            return metrics
        except Exception as e:
            self.logger.error(f"❌ Error getting performance metrics: {e}")
            return {}
    
    def get_health_status(self) -> HealthStatus:
        """Get health status."""
        try:
            status = self.core.get_health_status()
            return status
        except Exception as e:
            self.logger.error(f"❌ Error getting health status: {e}")
            return HealthStatus.UNKNOWN
    
    def get_health_details(self) -> Dict[str, Any]:
        """Get health details."""
        try:
            details = self.core.get_health_details()
            return details
        except Exception as e:
            self.logger.error(f"❌ Error getting health details: {e}")
            return {}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        try:
            alerts = self.core.get_active_alerts()
            return alerts
        except Exception as e:
            self.logger.error(f"❌ Error getting active alerts: {e}")
            return []
    
    def create_alert_config(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create alert configuration."""
        try:
            config = self.core.create_alert_config(alert_data)
            return config
        except Exception as e:
            self.logger.error(f"❌ Error creating alert config: {e}")
            return {}
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        try:
            dashboard_data = self.core.get_dashboard_data()
            return dashboard_data
        except Exception as e:
            self.logger.error(f"❌ Error getting dashboard data: {e}")
            return {}









