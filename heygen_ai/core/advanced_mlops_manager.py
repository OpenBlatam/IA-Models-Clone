#!/usr/bin/env python3
"""
Advanced MLOps & Model Lifecycle Manager
========================================

Comprehensive MLOps and model lifecycle management:
- Automated model training pipelines
- Model versioning and registry
- Automated deployment and scaling
- Performance monitoring and alerting
- A/B testing and model comparison
- CI/CD pipelines for ML
- Model governance and compliance
- Automated rollback and recovery
"""

import asyncio
import logging
import time
import json
import yaml
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import weakref
import gc
import os
import shutil
from pathlib import Path

# ML and deployment libraries
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import kubeflow
    KUBEFLOW_AVAILABLE = True
except ImportError:
    KUBEFlow_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"       # Development phase
    STAGING = "staging"              # Staging/testing phase
    PRODUCTION = "production"        # Production deployment
    ARCHIVED = "archived"            # Archived/retired
    FAILED = "failed"                # Failed deployment

class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"        # Blue-green deployment
    CANARY = "canary"                # Canary deployment
    ROLLING = "rolling"              # Rolling update
    RECREATE = "recreate"            # Recreate deployment

class MonitoringLevel(Enum):
    """Monitoring levels."""
    BASIC = "basic"                  # Basic monitoring
    ADVANCED = "advanced"            # Advanced monitoring
    ENTERPRISE = "enterprise"        # Enterprise monitoring
    CUSTOM = "custom"                # Custom monitoring

@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_name: str
    version_number: str
    stage: ModelStage
    created_at: datetime
    created_by: str
    git_commit: str
    model_path: str
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    dependencies: Dict[str, str]
    artifacts: List[str]

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    replicas: int = 3
    resources: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "1000m",
        "memory": "2Gi",
        "gpu": "1"
    })
    environment_variables: Dict[str, str] = field(default_factory=dict)
    health_check_path: str = "/health"
    readiness_probe: Dict[str, Any] = field(default_factory=dict)
    liveness_probe: Dict[str, Any] = field(default_factory=dict)
    autoscaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    monitoring_level: MonitoringLevel = MonitoringLevel.ADVANCED
    metrics_interval: int = 30  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "error_rate": 5.0,
        "latency_p95": 1000.0
    })
    log_retention_days: int = 30
    enable_anomaly_detection: bool = True
    enable_auto_scaling: bool = True
    enable_cost_monitoring: bool = True

@dataclass
class PipelineConfig:
    """CI/CD pipeline configuration."""
    enable_automated_testing: bool = True
    enable_automated_deployment: bool = True
    enable_rollback_on_failure: bool = True
    test_timeout_minutes: int = 30
    deployment_timeout_minutes: int = 60
    required_approvals: int = 1
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])

class AdvancedMLOpsManager:
    """
    Advanced MLOps and model lifecycle manager.
    """

    def __init__(self, 
                 mlflow_tracking_uri: Optional[str] = None,
                 model_registry_uri: Optional[str] = None):
        self.mlflow_tracking_uri = mlflow_tracking_uri or "sqlite:///mlflow.db"
        self.model_registry_uri = model_registry_uri or "sqlite:///mlflow.db"
        
        # Initialize MLflow
        self._initialize_mlflow()
        
        # Model registry
        self.model_versions = {}
        self.model_stages = {}
        self.deployment_history = []
        
        # Active deployments
        self.active_deployments = {}
        self.deployment_configs = {}
        
        # Monitoring
        self.monitoring_configs = {}
        self.performance_metrics = deque(maxlen=10000)
        self.alert_history = []
        
        # CI/CD pipelines
        self.pipeline_configs = {}
        self.pipeline_runs = []
        
        # Background tasks
        self._monitoring_running = False
        self._monitoring_thread = None
        self._start_monitoring()

    def _initialize_mlflow(self):
        """Initialize MLflow tracking and model registry."""
        try:
            if MLFLOW_AVAILABLE:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                mlflow.set_registry_uri(self.model_registry_uri)
                logger.info("MLflow initialized successfully")
            else:
                logger.warning("MLflow not available, using local tracking")
        except Exception as e:
            logger.error(f"Error initializing MLflow: {e}")

    def _start_monitoring(self):
        """Start background monitoring."""
        try:
            if not self._monitoring_running:
                self._monitoring_running = True
                self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self._monitoring_thread.start()
                logger.info("Background monitoring started")
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        try:
            while self._monitoring_running:
                # Collect metrics from active deployments
                self._collect_deployment_metrics()
                
                # Check alert thresholds
                self._check_alerts()
                
                # Auto-scaling decisions
                self._make_autoscaling_decisions()
                
                # Sleep for monitoring interval
                time.sleep(30)  # 30 seconds
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

    async def register_model(self, 
                           model_name: str,
                           model_path: str,
                           metadata: Dict[str, Any],
                           performance_metrics: Dict[str, float],
                           dependencies: Dict[str, str],
                           created_by: str,
                           git_commit: str = "") -> ModelVersion:
        """Register a new model version."""
        try:
            # Generate version ID
            version_id = str(uuid.uuid4())
            version_number = self._generate_version_number(model_name)
            
            # Create model version
            model_version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                version_number=version_number,
                stage=ModelStage.DEVELOPMENT,
                created_at=datetime.now(),
                created_by=created_by,
                git_commit=git_commit,
                model_path=model_path,
                metadata=metadata,
                performance_metrics=performance_metrics,
                dependencies=dependencies,
                artifacts=self._collect_model_artifacts(model_path)
            )
            
            # Store in registry
            self.model_versions[version_id] = model_version
            if model_name not in self.model_stages:
                self.model_stages[model_name] = {}
            self.model_stages[model_name][version_number] = model_version
            
            # Register with MLflow if available
            if MLFLOW_AVAILABLE:
                await self._register_with_mlflow(model_version)
            
            logger.info(f"Model {model_name} version {version_number} registered successfully")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def _generate_version_number(self, model_name: str) -> str:
        """Generate semantic version number."""
        try:
            if model_name in self.model_stages:
                existing_versions = list(self.model_stages[model_name].keys())
                if existing_versions:
                    # Parse existing versions and increment
                    latest_version = max(existing_versions, key=lambda v: [int(x) for x in v.split('.')])
                    major, minor, patch = map(int, latest_version.split('.'))
                    return f"{major}.{minor}.{patch + 1}"
            
            return "1.0.0"
            
        except Exception as e:
            logger.error(f"Error generating version number: {e}")
            return "1.0.0"

    def _collect_model_artifacts(self, model_path: str) -> List[str]:
        """Collect model artifacts from model path."""
        try:
            artifacts = []
            model_dir = Path(model_path)
            
            if model_dir.exists():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        artifacts.append(str(file_path))
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error collecting model artifacts: {e}")
            return []

    async def _register_with_mlflow(self, model_version: ModelVersion):
        """Register model with MLflow."""
        try:
            if not MLFLOW_AVAILABLE:
                return
            
            # This would implement MLflow model registration
            # For now, log the intention
            logger.debug(f"Would register {model_version.model_name} with MLflow")
            
        except Exception as e:
            logger.error(f"Error registering with MLflow: {e}")

    async def promote_model(self, 
                          model_name: str,
                          version_number: str,
                          target_stage: ModelStage,
                          promoted_by: str) -> bool:
        """Promote model to a new stage."""
        try:
            if model_name not in self.model_stages:
                raise ValueError(f"Model {model_name} not found")
            
            if version_number not in self.model_stages[model_name]:
                raise ValueError(f"Version {version_number} not found for model {model_name}")
            
            model_version = self.model_stages[model_name][version_number]
            
            # Update stage
            old_stage = model_version.stage
            model_version.stage = target_stage
            
            # Update MLflow if available
            if MLFLOW_AVAILABLE:
                await self._update_mlflow_stage(model_version, target_stage)
            
            logger.info(f"Model {model_name} version {version_number} promoted from {old_stage.value} to {target_stage.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False

    async def _update_mlflow_stage(self, model_version: ModelVersion, new_stage: ModelStage):
        """Update MLflow model stage."""
        try:
            if not MLFLOW_AVAILABLE:
                return
            
            # This would implement MLflow stage update
            logger.debug(f"Would update MLflow stage for {model_version.model_name} to {new_stage.value}")
            
        except Exception as e:
            logger.error(f"Error updating MLflow stage: {e}")

    async def deploy_model(self, 
                          model_name: str,
                          version_number: str,
                          deployment_config: DeploymentConfig,
                          deployment_name: Optional[str] = None) -> Dict[str, Any]:
        """Deploy a model version."""
        try:
            if model_name not in self.model_stages:
                raise ValueError(f"Model {model_name} not found")
            
            if version_number not in self.model_stages[model_name]:
                raise ValueError(f"Version {version_number} not found for model {model_name}")
            
            model_version = self.model_stages[model_name][version_number]
            
            # Generate deployment name if not provided
            if not deployment_name:
                deployment_name = f"{model_name}-{version_number}-{int(time.time())}"
            
            # Create deployment configuration
            self.deployment_configs[deployment_name] = deployment_config
            
            # Deploy based on strategy
            if deployment_config.deployment_strategy == DeploymentStrategy.BLUE_GREEN:
                deployment_result = await self._deploy_blue_green(model_version, deployment_name, deployment_config)
            elif deployment_config.deployment_strategy == DeploymentStrategy.CANARY:
                deployment_result = await self._deploy_canary(model_version, deployment_name, deployment_config)
            elif deployment_config.deployment_strategy == DeploymentStrategy.ROLLING:
                deployment_result = await self._deploy_rolling(model_version, deployment_name, deployment_config)
            else:
                deployment_result = await self._deploy_recreate(model_version, deployment_name, deployment_config)
            
            # Store deployment info
            deployment_info = {
                "deployment_name": deployment_name,
                "model_name": model_name,
                "version_number": version_number,
                "deployment_config": deployment_config,
                "deployment_result": deployment_result,
                "deployed_at": datetime.now(),
                "status": "active"
            }
            
            self.active_deployments[deployment_name] = deployment_info
            self.deployment_history.append(deployment_info)
            
            logger.info(f"Model {model_name} version {version_number} deployed successfully as {deployment_name}")
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise

    async def _deploy_blue_green(self, model_version: ModelVersion, deployment_name: str, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy using blue-green strategy."""
        try:
            # This would implement blue-green deployment
            # For now, simulate deployment
            deployment_result = {
                "strategy": "blue_green",
                "status": "success",
                "blue_deployment": f"{deployment_name}-blue",
                "green_deployment": f"{deployment_name}-green",
                "active_deployment": f"{deployment_name}-green",
                "replicas": config.replicas,
                "resources": config.resources
            }
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Error in blue-green deployment: {e}")
            raise

    async def _deploy_canary(self, model_version: ModelVersion, deployment_name: str, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy using canary strategy."""
        try:
            # This would implement canary deployment
            deployment_result = {
                "strategy": "canary",
                "status": "success",
                "stable_deployment": f"{deployment_name}-stable",
                "canary_deployment": f"{deployment_name}-canary",
                "traffic_split": {"stable": 90, "canary": 10},
                "replicas": config.replicas,
                "resources": config.resources
            }
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Error in canary deployment: {e}")
            raise

    async def _deploy_rolling(self, model_version: ModelVersion, deployment_name: str, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy using rolling strategy."""
        try:
            # This would implement rolling deployment
            deployment_result = {
                "strategy": "rolling",
                "status": "success",
                "deployment_name": deployment_name,
                "replicas": config.replicas,
                "resources": config.resources,
                "rolling_update_config": {
                    "max_surge": 1,
                    "max_unavailable": 0
                }
            }
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Error in rolling deployment: {e}")
            raise

    async def _deploy_recreate(self, model_version: ModelVersion, deployment_name: str, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy using recreate strategy."""
        try:
            # This would implement recreate deployment
            deployment_result = {
                "strategy": "recreate",
                "status": "success",
                "deployment_name": deployment_name,
                "replicas": config.replicas,
                "resources": config.resources
            }
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Error in recreate deployment: {e}")
            raise

    async def setup_monitoring(self, 
                              deployment_name: str,
                              monitoring_config: MonitoringConfig) -> bool:
        """Setup monitoring for a deployment."""
        try:
            if deployment_name not in self.active_deployments:
                raise ValueError(f"Deployment {deployment_name} not found")
            
            self.monitoring_configs[deployment_name] = monitoring_config
            
            # Setup monitoring infrastructure
            await self._setup_monitoring_infrastructure(deployment_name, monitoring_config)
            
            logger.info(f"Monitoring setup completed for deployment {deployment_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up monitoring: {e}")
            return False

    async def _setup_monitoring_infrastructure(self, deployment_name: str, config: MonitoringConfig):
        """Setup monitoring infrastructure."""
        try:
            # This would implement monitoring infrastructure setup
            # For now, log the configuration
            logger.debug(f"Setting up monitoring for {deployment_name} with level {config.monitoring_level.value}")
            
        except Exception as e:
            logger.error(f"Error setting up monitoring infrastructure: {e}")

    def _collect_deployment_metrics(self):
        """Collect metrics from active deployments."""
        try:
            for deployment_name, deployment_info in self.active_deployments.items():
                if deployment_name in self.monitoring_configs:
                    metrics = self._collect_single_deployment_metrics(deployment_name)
                    if metrics:
                        self.performance_metrics.append(metrics)
                        
        except Exception as e:
            logger.error(f"Error collecting deployment metrics: {e}")

    def _collect_single_deployment_metrics(self, deployment_name: str) -> Optional[Dict[str, Any]]:
        """Collect metrics for a single deployment."""
        try:
            # This would collect actual metrics from the deployment
            # For now, simulate metrics
            metrics = {
                "deployment_name": deployment_name,
                "timestamp": time.time(),
                "cpu_usage": 45.0 + (time.time() % 30),  # Simulated
                "memory_usage": 60.0 + (time.time() % 20),  # Simulated
                "error_rate": 0.5 + (time.time() % 2),  # Simulated
                "latency_p95": 500.0 + (time.time() % 200),  # Simulated
                "request_count": int(1000 + (time.time() % 500))  # Simulated
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {deployment_name}: {e}")
            return None

    def _check_alerts(self):
        """Check alert thresholds and trigger alerts."""
        try:
            for deployment_name, config in self.monitoring_configs.items():
                if deployment_name in self.active_deployments:
                    recent_metrics = [m for m in self.performance_metrics 
                                    if m["deployment_name"] == deployment_name and 
                                    time.time() - m["timestamp"] < 300]  # Last 5 minutes
                    
                    if recent_metrics:
                        latest_metrics = recent_metrics[-1]
                        alerts = self._check_single_deployment_alerts(deployment_name, latest_metrics, config)
                        
                        for alert in alerts:
                            self.alert_history.append(alert)
                            logger.warning(f"ALERT: {alert['message']}")
                            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def _check_single_deployment_alerts(self, deployment_name: str, metrics: Dict[str, Any], config: MonitoringConfig) -> List[Dict[str, Any]]:
        """Check alerts for a single deployment."""
        try:
            alerts = []
            
            for metric_name, threshold in config.alert_thresholds.items():
                if metric_name in metrics:
                    current_value = metrics[metric_name]
                    
                    if current_value > threshold:
                        alert = {
                            "deployment_name": deployment_name,
                            "metric_name": metric_name,
                            "current_value": current_value,
                            "threshold": threshold,
                            "severity": "high" if current_value > threshold * 1.5 else "medium",
                            "timestamp": time.time(),
                            "message": f"{deployment_name}: {metric_name} = {current_value:.2f} exceeds threshold {threshold:.2f}"
                        }
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts for {deployment_name}: {e}")
            return []

    def _make_autoscaling_decisions(self):
        """Make autoscaling decisions based on metrics."""
        try:
            for deployment_name, config in self.monitoring_configs.items():
                if (config.enable_auto_scaling and 
                    deployment_name in self.active_deployments and
                    deployment_name in self.deployment_configs):
                    
                    deployment_config = self.deployment_configs[deployment_name]
                    recent_metrics = [m for m in self.performance_metrics 
                                    if m["deployment_name"] == deployment_name and 
                                    time.time() - m["timestamp"] < 300]  # Last 5 minutes
                    
                    if recent_metrics:
                        avg_cpu = sum(m["cpu_usage"] for m in recent_metrics) / len(recent_metrics)
                        
                        if avg_cpu > deployment_config.target_cpu_utilization:
                            # Scale up
                            new_replicas = min(
                                deployment_config.max_replicas,
                                deployment_config.replicas + 1
                            )
                            if new_replicas != deployment_config.replicas:
                                logger.info(f"Scaling up {deployment_name} to {new_replicas} replicas")
                                deployment_config.replicas = new_replicas
                        
                        elif avg_cpu < deployment_config.target_cpu_utilization * 0.5:
                            # Scale down
                            new_replicas = max(
                                deployment_config.min_replicas,
                                deployment_config.replicas - 1
                            )
                            if new_replicas != deployment_config.replicas:
                                logger.info(f"Scaling down {deployment_name} to {new_replicas} replicas")
                                deployment_config.replicas = new_replicas
                                
        except Exception as e:
            logger.error(f"Error making autoscaling decisions: {e}")

    async def rollback_deployment(self, deployment_name: str, target_version: str) -> bool:
        """Rollback a deployment to a previous version."""
        try:
            if deployment_name not in self.active_deployments:
                raise ValueError(f"Deployment {deployment_name} not found")
            
            deployment_info = self.active_deployments[deployment_name]
            current_version = deployment_info["version_number"]
            
            # Perform rollback
            rollback_result = await self._perform_rollback(deployment_name, target_version)
            
            if rollback_result:
                # Update deployment info
                deployment_info["version_number"] = target_version
                deployment_info["rolled_back_from"] = current_version
                deployment_info["rolled_back_at"] = datetime.now()
                
                logger.info(f"Deployment {deployment_name} rolled back from {current_version} to {target_version}")
                return True
            else:
                logger.error(f"Rollback failed for deployment {deployment_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error rolling back deployment: {e}")
            return False

    async def _perform_rollback(self, deployment_name: str, target_version: str) -> bool:
        """Perform the actual rollback operation."""
        try:
            # This would implement the actual rollback logic
            # For now, simulate successful rollback
            logger.debug(f"Rolling back {deployment_name} to version {target_version}")
            
            # Simulate rollback delay
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing rollback: {e}")
            return False

    def get_model_versions(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """Get model versions."""
        try:
            if model_name:
                if model_name in self.model_stages:
                    return list(self.model_stages[model_name].values())
                else:
                    return []
            else:
                return list(self.model_versions.values())
                
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []

    def get_active_deployments(self) -> Dict[str, Any]:
        """Get active deployments."""
        return self.active_deployments.copy()

    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self.deployment_history.copy()

    def get_performance_metrics(self, deployment_name: Optional[str] = None, 
                               time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get performance metrics."""
        try:
            cutoff_time = time.time() - (time_window_minutes * 60)
            
            if deployment_name:
                metrics = [m for m in self.performance_metrics 
                          if m["deployment_name"] == deployment_name and 
                          m["timestamp"] > cutoff_time]
            else:
                metrics = [m for m in self.performance_metrics 
                          if m["timestamp"] > cutoff_time]
            
            return sorted(metrics, key=lambda x: x["timestamp"])
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []

    def get_alerts(self, deployment_name: Optional[str] = None, 
                   severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts."""
        try:
            alerts = self.alert_history
            
            if deployment_name:
                alerts = [a for a in alerts if a["deployment_name"] == deployment_name]
            
            if severity:
                alerts = [a for a in alerts if a["severity"] == severity]
            
            return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop monitoring
            self._monitoring_running = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5)
            
            # Clear data structures
            self.model_versions.clear()
            self.model_stages.clear()
            self.active_deployments.clear()
            self.deployment_configs.clear()
            self.monitoring_configs.clear()
            self.performance_metrics.clear()
            self.alert_history.clear()
            self.deployment_history.clear()
            self.pipeline_runs.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Advanced MLOps manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
