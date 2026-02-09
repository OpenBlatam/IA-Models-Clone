"""
Prometheus Monitoring Plugin
============================
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Response
from aws.core.interfaces import MonitoringPlugin

logger = logging.getLogger(__name__)


class PrometheusMonitoringPlugin(MonitoringPlugin):
    """Prometheus metrics plugin."""
    
    def get_name(self) -> str:
        return "prometheus"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        monitoring_config = config.get("monitoring", {})
        return monitoring_config.get("enable_prometheus", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup Prometheus metrics."""
        try:
            from prometheus_client import generate_latest, REGISTRY, make_asgi_app
            from aws.monitoring.prometheus_config import setup_prometheus
            
            app = setup_prometheus(
                app,
                config.get("app_name", "robot-movement-ai"),
                config.get("app_version", "1.0.0")
            )
            
            logger.info("Prometheus monitoring enabled")
            
        except ImportError:
            logger.warning("Prometheus client not installed. Monitoring disabled.")
        except Exception as e:
            logger.error(f"Failed to setup Prometheus: {e}")
        
        return app















