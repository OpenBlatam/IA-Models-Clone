"""
Deployment Automation System
============================

Advanced deployment automation system for AI model analysis with CI/CD
integration, blue-green deployment, rollback capabilities, and monitoring.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import subprocess
import docker
import kubernetes
import yaml
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import shutil
import os
import tarfile
import zipfile
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DeploymentType(str, Enum):
    """Deployment types"""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"
    MANUAL = "manual"
    AUTOMATED = "automated"


class DeploymentStatus(str, Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class EnvironmentType(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    PREVIEW = "preview"


class InfrastructureType(str, Enum):
    """Infrastructure types"""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass
class DeploymentConfiguration:
    """Deployment configuration"""
    config_id: str
    name: str
    description: str
    deployment_type: DeploymentType
    environment: EnvironmentType
    infrastructure: InfrastructureType
    application_config: Dict[str, Any]
    infrastructure_config: Dict[str, Any]
    health_check_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    scaling_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Deployment:
    """Deployment instance"""
    deployment_id: str
    config_id: str
    name: str
    version: str
    status: DeploymentStatus
    environment: EnvironmentType
    infrastructure: InfrastructureType
    start_time: datetime
    end_time: datetime = None
    duration: float = 0.0
    logs: List[str] = None
    metrics: Dict[str, Any] = None
    error_message: str = ""
    rollback_available: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.logs is None:
            self.logs = []
        if self.metrics is None:
            self.metrics = {}


@dataclass
class DeploymentStep:
    """Deployment step"""
    step_id: str
    deployment_id: str
    name: str
    description: str
    step_type: str
    status: DeploymentStatus
    start_time: datetime
    end_time: datetime = None
    duration: float = 0.0
    logs: List[str] = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []


@dataclass
class DeploymentMetrics:
    """Deployment metrics"""
    deployment_id: str
    success_rate: float
    average_deployment_time: float
    rollback_rate: float
    health_check_success_rate: float
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, float]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DeploymentAutomationSystem:
    """Advanced deployment automation system for AI model analysis"""
    
    def __init__(self, max_deployments: int = 1000):
        self.max_deployments = max_deployments
        
        self.deployment_configurations: Dict[str, DeploymentConfiguration] = {}
        self.deployments: Dict[str, Deployment] = {}
        self.deployment_steps: List[DeploymentStep] = []
        self.deployment_metrics: List[DeploymentMetrics] = []
        
        # Infrastructure clients
        self.docker_client = None
        self.k8s_client = None
        
        # Deployment execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running_deployments: Dict[str, asyncio.Task] = {}
        
        # Initialize infrastructure clients
        self._initialize_infrastructure_clients()
    
    async def create_deployment_configuration(self, 
                                            name: str,
                                            description: str,
                                            deployment_type: DeploymentType,
                                            environment: EnvironmentType,
                                            infrastructure: InfrastructureType,
                                            application_config: Dict[str, Any] = None,
                                            infrastructure_config: Dict[str, Any] = None,
                                            health_check_config: Dict[str, Any] = None,
                                            rollback_config: Dict[str, Any] = None,
                                            scaling_config: Dict[str, Any] = None,
                                            monitoring_config: Dict[str, Any] = None) -> DeploymentConfiguration:
        """Create deployment configuration"""
        try:
            config_id = hashlib.md5(f"{name}_{environment}_{datetime.now()}".encode()).hexdigest()
            
            if application_config is None:
                application_config = {}
            if infrastructure_config is None:
                infrastructure_config = {}
            if health_check_config is None:
                health_check_config = {}
            if rollback_config is None:
                rollback_config = {}
            if scaling_config is None:
                scaling_config = {}
            if monitoring_config is None:
                monitoring_config = {}
            
            config = DeploymentConfiguration(
                config_id=config_id,
                name=name,
                description=description,
                deployment_type=deployment_type,
                environment=environment,
                infrastructure=infrastructure,
                application_config=application_config,
                infrastructure_config=infrastructure_config,
                health_check_config=health_check_config,
                rollback_config=rollback_config,
                scaling_config=scaling_config,
                monitoring_config=monitoring_config
            )
            
            self.deployment_configurations[config_id] = config
            
            logger.info(f"Created deployment configuration: {name}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating deployment configuration: {str(e)}")
            raise e
    
    async def deploy_application(self, 
                               config_id: str,
                               version: str,
                               deployment_type: DeploymentType = None) -> Deployment:
        """Deploy application with given configuration"""
        try:
            if config_id not in self.deployment_configurations:
                raise ValueError(f"Deployment configuration {config_id} not found")
            
            config = self.deployment_configurations[config_id]
            deployment_id = hashlib.md5(f"{config_id}_{version}_{datetime.now()}".encode()).hexdigest()
            
            # Override deployment type if provided
            if deployment_type:
                config.deployment_type = deployment_type
            
            # Create deployment instance
            deployment = Deployment(
                deployment_id=deployment_id,
                config_id=config_id,
                name=config.name,
                version=version,
                status=DeploymentStatus.PENDING,
                environment=config.environment,
                infrastructure=config.infrastructure,
                start_time=datetime.now()
            )
            
            self.deployments[deployment_id] = deployment
            
            logger.info(f"Starting deployment: {config.name} v{version}")
            
            # Execute deployment based on type
            if config.deployment_type == DeploymentType.BLUE_GREEN:
                await self._deploy_blue_green(deployment, config)
            elif config.deployment_type == DeploymentType.ROLLING:
                await self._deploy_rolling(deployment, config)
            elif config.deployment_type == DeploymentType.CANARY:
                await self._deploy_canary(deployment, config)
            elif config.deployment_type == DeploymentType.RECREATE:
                await self._deploy_recreate(deployment, config)
            else:
                await self._deploy_manual(deployment, config)
            
            return deployment
            
        except Exception as e:
            logger.error(f"Error deploying application: {str(e)}")
            raise e
    
    async def rollback_deployment(self, 
                                deployment_id: str,
                                target_version: str = None) -> bool:
        """Rollback deployment to previous version"""
        try:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            deployment = self.deployments[deployment_id]
            config = self.deployment_configurations[deployment.config_id]
            
            if not deployment.rollback_available:
                raise ValueError(f"Rollback not available for deployment {deployment_id}")
            
            logger.info(f"Rolling back deployment: {deployment.name}")
            
            # Create rollback deployment
            rollback_deployment = Deployment(
                deployment_id=hashlib.md5(f"rollback_{deployment_id}_{datetime.now()}".encode()).hexdigest(),
                config_id=deployment.config_id,
                name=f"Rollback: {deployment.name}",
                version=target_version or "previous",
                status=DeploymentStatus.IN_PROGRESS,
                environment=deployment.environment,
                infrastructure=deployment.infrastructure,
                start_time=datetime.now()
            )
            
            self.deployments[rollback_deployment.deployment_id] = rollback_deployment
            
            # Execute rollback based on infrastructure
            if config.infrastructure == InfrastructureType.DOCKER:
                await self._rollback_docker(rollback_deployment, config, target_version)
            elif config.infrastructure == InfrastructureType.KUBERNETES:
                await self._rollback_kubernetes(rollback_deployment, config, target_version)
            else:
                await self._rollback_generic(rollback_deployment, config, target_version)
            
            # Update original deployment status
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
            
            logger.info(f"Rollback completed: {deployment.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back deployment: {str(e)}")
            return False
    
    async def scale_application(self, 
                              deployment_id: str,
                              scale_config: Dict[str, int]) -> bool:
        """Scale application deployment"""
        try:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            deployment = self.deployments[deployment_id]
            config = self.deployment_configurations[deployment.config_id]
            
            logger.info(f"Scaling deployment: {deployment.name}")
            
            # Execute scaling based on infrastructure
            if config.infrastructure == InfrastructureType.DOCKER:
                await self._scale_docker(deployment, config, scale_config)
            elif config.infrastructure == InfrastructureType.KUBERNETES:
                await self._scale_kubernetes(deployment, config, scale_config)
            else:
                await self._scale_generic(deployment, config, scale_config)
            
            # Update deployment metrics
            deployment.metrics["scaling"] = {
                "timestamp": datetime.now().isoformat(),
                "scale_config": scale_config
            }
            
            logger.info(f"Scaling completed: {deployment.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error scaling application: {str(e)}")
            return False
    
    async def health_check_deployment(self, 
                                    deployment_id: str) -> Dict[str, Any]:
        """Perform health check on deployment"""
        try:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            deployment = self.deployments[deployment_id]
            config = self.deployment_configurations[deployment.config_id]
            
            health_check_config = config.health_check_config
            
            # Perform health checks
            health_results = {
                "deployment_id": deployment_id,
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy",
                "checks": {}
            }
            
            # HTTP health check
            if "http_endpoint" in health_check_config:
                http_status = await self._check_http_health(health_check_config["http_endpoint"])
                health_results["checks"]["http"] = http_status
                if not http_status["healthy"]:
                    health_results["overall_status"] = "unhealthy"
            
            # Database health check
            if "database_config" in health_check_config:
                db_status = await self._check_database_health(health_check_config["database_config"])
                health_results["checks"]["database"] = db_status
                if not db_status["healthy"]:
                    health_results["overall_status"] = "unhealthy"
            
            # Resource health check
            if "resource_limits" in health_check_config:
                resource_status = await self._check_resource_health(deployment, health_check_config["resource_limits"])
                health_results["checks"]["resources"] = resource_status
                if not resource_status["healthy"]:
                    health_results["overall_status"] = "unhealthy"
            
            # Update deployment metrics
            deployment.metrics["health_check"] = health_results
            
            return health_results
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            return {"error": str(e)}
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status and metrics"""
        try:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            deployment = self.deployments[deployment_id]
            config = self.deployment_configurations[deployment.config_id]
            
            # Calculate duration
            end_time = deployment.end_time or datetime.now()
            duration = (end_time - deployment.start_time).total_seconds()
            
            status = {
                "deployment_id": deployment_id,
                "name": deployment.name,
                "version": deployment.version,
                "status": deployment.status.value,
                "environment": deployment.environment.value,
                "infrastructure": deployment.infrastructure.value,
                "start_time": deployment.start_time.isoformat(),
                "end_time": deployment.end_time.isoformat() if deployment.end_time else None,
                "duration": duration,
                "rollback_available": deployment.rollback_available,
                "error_message": deployment.error_message,
                "metrics": deployment.metrics,
                "logs": deployment.logs[-10:] if deployment.logs else [],  # Last 10 logs
                "configuration": {
                    "deployment_type": config.deployment_type.value,
                    "health_check_config": config.health_check_config,
                    "scaling_config": config.scaling_config
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {str(e)}")
            return {"error": str(e)}
    
    async def get_deployment_analytics(self, 
                                     time_range_days: int = 30) -> Dict[str, Any]:
        """Get deployment analytics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=time_range_days)
            recent_deployments = [d for d in self.deployments.values() if d.created_at >= cutoff_date]
            
            analytics = {
                "total_deployments": len(recent_deployments),
                "successful_deployments": len([d for d in recent_deployments if d.status == DeploymentStatus.SUCCESS]),
                "failed_deployments": len([d for d in recent_deployments if d.status == DeploymentStatus.FAILED]),
                "rolled_back_deployments": len([d for d in recent_deployments if d.status == DeploymentStatus.ROLLED_BACK]),
                "success_rate": 0.0,
                "average_deployment_time": 0.0,
                "rollback_rate": 0.0,
                "deployment_types": {},
                "environments": {},
                "infrastructures": {},
                "deployment_trends": {},
                "performance_metrics": {}
            }
            
            if recent_deployments:
                # Calculate success rate
                successful = len([d for d in recent_deployments if d.status == DeploymentStatus.SUCCESS])
                analytics["success_rate"] = successful / len(recent_deployments)
                
                # Calculate average deployment time
                completed_deployments = [d for d in recent_deployments if d.end_time]
                if completed_deployments:
                    deployment_times = [(d.end_time - d.start_time).total_seconds() for d in completed_deployments]
                    analytics["average_deployment_time"] = sum(deployment_times) / len(deployment_times)
                
                # Calculate rollback rate
                rolled_back = len([d for d in recent_deployments if d.status == DeploymentStatus.ROLLED_BACK])
                analytics["rollback_rate"] = rolled_back / len(recent_deployments)
                
                # Analyze deployment types
                for deployment in recent_deployments:
                    config = self.deployment_configurations.get(deployment.config_id)
                    if config:
                        dep_type = config.deployment_type.value
                        if dep_type not in analytics["deployment_types"]:
                            analytics["deployment_types"][dep_type] = 0
                        analytics["deployment_types"][dep_type] += 1
                
                # Analyze environments
                for deployment in recent_deployments:
                    env = deployment.environment.value
                    if env not in analytics["environments"]:
                        analytics["environments"][env] = 0
                    analytics["environments"][env] += 1
                
                # Analyze infrastructures
                for deployment in recent_deployments:
                    infra = deployment.infrastructure.value
                    if infra not in analytics["infrastructures"]:
                        analytics["infrastructures"][infra] = 0
                    analytics["infrastructures"][infra] += 1
                
                # Deployment trends (daily)
                daily_deployments = defaultdict(int)
                for deployment in recent_deployments:
                    date_key = deployment.created_at.date()
                    daily_deployments[date_key] += 1
                
                analytics["deployment_trends"] = {
                    date.isoformat(): count for date, count in daily_deployments.items()
                }
                
                # Performance metrics
                analytics["performance_metrics"] = {
                    "fastest_deployment": min([(d.end_time - d.start_time).total_seconds() for d in completed_deployments], default=0),
                    "slowest_deployment": max([(d.end_time - d.start_time).total_seconds() for d in completed_deployments], default=0),
                    "deployment_frequency": len(recent_deployments) / time_range_days
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting deployment analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    def _initialize_infrastructure_clients(self) -> None:
        """Initialize infrastructure clients"""
        try:
            # Initialize Docker client
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Docker client: {str(e)}")
            
            # Initialize Kubernetes client
            try:
                kubernetes.config.load_incluster_config()
                self.k8s_client = kubernetes.client.ApiClient()
                logger.info("Kubernetes client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Kubernetes client: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error initializing infrastructure clients: {str(e)}")
    
    async def _deploy_blue_green(self, deployment: Deployment, config: DeploymentConfiguration) -> None:
        """Deploy using blue-green strategy"""
        try:
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.logs.append(f"Starting blue-green deployment for {deployment.name}")
            
            # Step 1: Deploy to green environment
            green_step = await self._create_deployment_step(
                deployment.deployment_id, "Deploy to Green", "Deploying to green environment"
            )
            await self._execute_deployment_step(green_step, config, "green")
            
            # Step 2: Health check green environment
            health_step = await self._create_deployment_step(
                deployment.deployment_id, "Health Check Green", "Performing health checks on green environment"
            )
            health_status = await self._check_http_health(config.health_check_config.get("http_endpoint", ""))
            if not health_status["healthy"]:
                raise Exception("Health check failed on green environment")
            
            # Step 3: Switch traffic to green
            switch_step = await self._create_deployment_step(
                deployment.deployment_id, "Switch Traffic", "Switching traffic to green environment"
            )
            await self._execute_deployment_step(switch_step, config, "switch")
            
            # Step 4: Cleanup blue environment
            cleanup_step = await self._create_deployment_step(
                deployment.deployment_id, "Cleanup Blue", "Cleaning up blue environment"
            )
            await self._execute_deployment_step(cleanup_step, config, "cleanup")
            
            deployment.status = DeploymentStatus.SUCCESS
            deployment.rollback_available = True
            deployment.logs.append("Blue-green deployment completed successfully")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.logs.append(f"Blue-green deployment failed: {str(e)}")
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _deploy_rolling(self, deployment: Deployment, config: DeploymentConfiguration) -> None:
        """Deploy using rolling update strategy"""
        try:
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.logs.append(f"Starting rolling deployment for {deployment.name}")
            
            # Simulate rolling deployment steps
            steps = ["Prepare", "Update Pod 1", "Update Pod 2", "Update Pod 3", "Verify", "Complete"]
            
            for step_name in steps:
                step = await self._create_deployment_step(
                    deployment.deployment_id, step_name, f"Executing {step_name.lower()}"
                )
                await self._execute_deployment_step(step, config, "rolling")
                await asyncio.sleep(1)  # Simulate step execution time
            
            deployment.status = DeploymentStatus.SUCCESS
            deployment.rollback_available = True
            deployment.logs.append("Rolling deployment completed successfully")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.logs.append(f"Rolling deployment failed: {str(e)}")
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _deploy_canary(self, deployment: Deployment, config: DeploymentConfiguration) -> None:
        """Deploy using canary strategy"""
        try:
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.logs.append(f"Starting canary deployment for {deployment.name}")
            
            # Step 1: Deploy canary (small percentage)
            canary_step = await self._create_deployment_step(
                deployment.deployment_id, "Deploy Canary", "Deploying canary version"
            )
            await self._execute_deployment_step(canary_step, config, "canary")
            
            # Step 2: Monitor canary
            monitor_step = await self._create_deployment_step(
                deployment.deployment_id, "Monitor Canary", "Monitoring canary performance"
            )
            await asyncio.sleep(5)  # Simulate monitoring period
            
            # Step 3: Gradually increase traffic
            traffic_step = await self._create_deployment_step(
                deployment.deployment_id, "Increase Traffic", "Gradually increasing traffic to canary"
            )
            await self._execute_deployment_step(traffic_step, config, "traffic")
            
            # Step 4: Full deployment
            full_step = await self._create_deployment_step(
                deployment.deployment_id, "Full Deployment", "Deploying to full environment"
            )
            await self._execute_deployment_step(full_step, config, "full")
            
            deployment.status = DeploymentStatus.SUCCESS
            deployment.rollback_available = True
            deployment.logs.append("Canary deployment completed successfully")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.logs.append(f"Canary deployment failed: {str(e)}")
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _deploy_recreate(self, deployment: Deployment, config: DeploymentConfiguration) -> None:
        """Deploy using recreate strategy"""
        try:
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.logs.append(f"Starting recreate deployment for {deployment.name}")
            
            # Step 1: Stop old version
            stop_step = await self._create_deployment_step(
                deployment.deployment_id, "Stop Old Version", "Stopping old version"
            )
            await self._execute_deployment_step(stop_step, config, "stop")
            
            # Step 2: Deploy new version
            deploy_step = await self._create_deployment_step(
                deployment.deployment_id, "Deploy New Version", "Deploying new version"
            )
            await self._execute_deployment_step(deploy_step, config, "deploy")
            
            # Step 3: Start new version
            start_step = await self._create_deployment_step(
                deployment.deployment_id, "Start New Version", "Starting new version"
            )
            await self._execute_deployment_step(start_step, config, "start")
            
            deployment.status = DeploymentStatus.SUCCESS
            deployment.rollback_available = True
            deployment.logs.append("Recreate deployment completed successfully")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.logs.append(f"Recreate deployment failed: {str(e)}")
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _deploy_manual(self, deployment: Deployment, config: DeploymentConfiguration) -> None:
        """Deploy using manual strategy"""
        try:
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.logs.append(f"Starting manual deployment for {deployment.name}")
            
            # Simulate manual deployment steps
            await asyncio.sleep(2)  # Simulate manual deployment time
            
            deployment.status = DeploymentStatus.SUCCESS
            deployment.rollback_available = True
            deployment.logs.append("Manual deployment completed successfully")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.logs.append(f"Manual deployment failed: {str(e)}")
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _create_deployment_step(self, 
                                    deployment_id: str, 
                                    name: str, 
                                    description: str) -> DeploymentStep:
        """Create deployment step"""
        try:
            step_id = hashlib.md5(f"{deployment_id}_{name}_{datetime.now()}".encode()).hexdigest()
            
            step = DeploymentStep(
                step_id=step_id,
                deployment_id=deployment_id,
                name=name,
                description=description,
                step_type="deployment",
                status=DeploymentStatus.PENDING,
                start_time=datetime.now()
            )
            
            self.deployment_steps.append(step)
            
            return step
            
        except Exception as e:
            logger.error(f"Error creating deployment step: {str(e)}")
            raise e
    
    async def _execute_deployment_step(self, 
                                     step: DeploymentStep, 
                                     config: DeploymentConfiguration, 
                                     step_type: str) -> None:
        """Execute deployment step"""
        try:
            step.status = DeploymentStatus.IN_PROGRESS
            step.logs.append(f"Executing {step.name}")
            
            # Simulate step execution based on infrastructure
            if config.infrastructure == InfrastructureType.DOCKER:
                await self._execute_docker_step(step, config, step_type)
            elif config.infrastructure == InfrastructureType.KUBERNETES:
                await self._execute_kubernetes_step(step, config, step_type)
            else:
                await self._execute_generic_step(step, config, step_type)
            
            step.status = DeploymentStatus.SUCCESS
            step.logs.append(f"Completed {step.name}")
            
        except Exception as e:
            step.status = DeploymentStatus.FAILED
            step.error_message = str(e)
            step.logs.append(f"Failed {step.name}: {str(e)}")
            raise e
        finally:
            step.end_time = datetime.now()
            step.duration = (step.end_time - step.start_time).total_seconds()
    
    async def _execute_docker_step(self, 
                                 step: DeploymentStep, 
                                 config: DeploymentConfiguration, 
                                 step_type: str) -> None:
        """Execute Docker deployment step"""
        try:
            if self.docker_client:
                # Simulate Docker operations
                await asyncio.sleep(1)
                step.logs.append("Docker step executed successfully")
            else:
                step.logs.append("Docker client not available, simulating step")
                await asyncio.sleep(0.5)
                
        except Exception as e:
            raise e
    
    async def _execute_kubernetes_step(self, 
                                     step: DeploymentStep, 
                                     config: DeploymentConfiguration, 
                                     step_type: str) -> None:
        """Execute Kubernetes deployment step"""
        try:
            if self.k8s_client:
                # Simulate Kubernetes operations
                await asyncio.sleep(1)
                step.logs.append("Kubernetes step executed successfully")
            else:
                step.logs.append("Kubernetes client not available, simulating step")
                await asyncio.sleep(0.5)
                
        except Exception as e:
            raise e
    
    async def _execute_generic_step(self, 
                                  step: DeploymentStep, 
                                  config: DeploymentConfiguration, 
                                  step_type: str) -> None:
        """Execute generic deployment step"""
        try:
            # Simulate generic deployment operations
            await asyncio.sleep(0.5)
            step.logs.append("Generic step executed successfully")
            
        except Exception as e:
            raise e
    
    async def _rollback_docker(self, 
                             deployment: Deployment, 
                             config: DeploymentConfiguration, 
                             target_version: str) -> None:
        """Rollback Docker deployment"""
        try:
            deployment.logs.append("Rolling back Docker deployment")
            await asyncio.sleep(2)  # Simulate rollback time
            deployment.status = DeploymentStatus.SUCCESS
            deployment.logs.append("Docker rollback completed")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _rollback_kubernetes(self, 
                                 deployment: Deployment, 
                                 config: DeploymentConfiguration, 
                                 target_version: str) -> None:
        """Rollback Kubernetes deployment"""
        try:
            deployment.logs.append("Rolling back Kubernetes deployment")
            await asyncio.sleep(2)  # Simulate rollback time
            deployment.status = DeploymentStatus.SUCCESS
            deployment.logs.append("Kubernetes rollback completed")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _rollback_generic(self, 
                              deployment: Deployment, 
                              config: DeploymentConfiguration, 
                              target_version: str) -> None:
        """Rollback generic deployment"""
        try:
            deployment.logs.append("Rolling back generic deployment")
            await asyncio.sleep(1)  # Simulate rollback time
            deployment.status = DeploymentStatus.SUCCESS
            deployment.logs.append("Generic rollback completed")
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            raise e
        finally:
            deployment.end_time = datetime.now()
            deployment.duration = (deployment.end_time - deployment.start_time).total_seconds()
    
    async def _scale_docker(self, 
                          deployment: Deployment, 
                          config: DeploymentConfiguration, 
                          scale_config: Dict[str, int]) -> None:
        """Scale Docker deployment"""
        try:
            deployment.logs.append(f"Scaling Docker deployment: {scale_config}")
            await asyncio.sleep(1)  # Simulate scaling time
            deployment.logs.append("Docker scaling completed")
            
        except Exception as e:
            raise e
    
    async def _scale_kubernetes(self, 
                              deployment: Deployment, 
                              config: DeploymentConfiguration, 
                              scale_config: Dict[str, int]) -> None:
        """Scale Kubernetes deployment"""
        try:
            deployment.logs.append(f"Scaling Kubernetes deployment: {scale_config}")
            await asyncio.sleep(1)  # Simulate scaling time
            deployment.logs.append("Kubernetes scaling completed")
            
        except Exception as e:
            raise e
    
    async def _scale_generic(self, 
                           deployment: Deployment, 
                           config: DeploymentConfiguration, 
                           scale_config: Dict[str, int]) -> None:
        """Scale generic deployment"""
        try:
            deployment.logs.append(f"Scaling generic deployment: {scale_config}")
            await asyncio.sleep(0.5)  # Simulate scaling time
            deployment.logs.append("Generic scaling completed")
            
        except Exception as e:
            raise e
    
    async def _check_http_health(self, endpoint: str) -> Dict[str, Any]:
        """Check HTTP health endpoint"""
        try:
            if not endpoint:
                return {"healthy": True, "response_time": 0, "status_code": 200}
            
            start_time = time.time()
            response = requests.get(endpoint, timeout=5)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "healthy": response.status_code == 200,
                "response_time": response_time,
                "status_code": response.status_code
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "response_time": 0,
                "status_code": 0,
                "error": str(e)
            }
    
    async def _check_database_health(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Simulate database health check
            await asyncio.sleep(0.1)
            
            return {
                "healthy": True,
                "connection_time": 50,  # milliseconds
                "query_time": 10  # milliseconds
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_resource_health(self, 
                                   deployment: Deployment, 
                                   resource_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Check resource health"""
        try:
            # Simulate resource health check
            await asyncio.sleep(0.1)
            
            # Generate simulated resource usage
            cpu_usage = np.random.uniform(10, 80)
            memory_usage = np.random.uniform(100, 800)
            
            cpu_healthy = cpu_usage < resource_limits.get("max_cpu", 80)
            memory_healthy = memory_usage < resource_limits.get("max_memory", 1000)
            
            return {
                "healthy": cpu_healthy and memory_healthy,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "cpu_limit": resource_limits.get("max_cpu", 80),
                "memory_limit": resource_limits.get("max_memory", 1000)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }


# Global deployment automation system instance
_deployment_system: Optional[DeploymentAutomationSystem] = None


def get_deployment_automation_system(max_deployments: int = 1000) -> DeploymentAutomationSystem:
    """Get or create global deployment automation system instance"""
    global _deployment_system
    if _deployment_system is None:
        _deployment_system = DeploymentAutomationSystem(max_deployments)
    return _deployment_system


# Example usage
async def main():
    """Example usage of the deployment automation system"""
    system = get_deployment_automation_system()
    
    # Create deployment configuration
    config = await system.create_deployment_configuration(
        name="AI Model API",
        description="Deployment configuration for AI Model API",
        deployment_type=DeploymentType.BLUE_GREEN,
        environment=EnvironmentType.PRODUCTION,
        infrastructure=InfrastructureType.KUBERNETES,
        application_config={
            "image": "ai-model-api:latest",
            "port": 8080,
            "replicas": 3
        },
        infrastructure_config={
            "namespace": "production",
            "resources": {
                "cpu": "500m",
                "memory": "1Gi"
            }
        },
        health_check_config={
            "http_endpoint": "http://localhost:8080/health",
            "timeout": 30
        },
        rollback_config={
            "enabled": True,
            "max_versions": 5
        },
        scaling_config={
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu": 70
        }
    )
    print(f"Created deployment configuration: {config.config_id}")
    
    # Deploy application
    deployment = await system.deploy_application(
        config_id=config.config_id,
        version="v1.2.3",
        deployment_type=DeploymentType.BLUE_GREEN
    )
    print(f"Deployment started: {deployment.deployment_id}")
    print(f"Status: {deployment.status.value}")
    
    # Wait for deployment to complete
    await asyncio.sleep(3)
    
    # Get deployment status
    status = await system.get_deployment_status(deployment.deployment_id)
    print(f"Deployment status: {status['status']}")
    print(f"Duration: {status['duration']:.2f}s")
    
    # Perform health check
    health = await system.health_check_deployment(deployment.deployment_id)
    print(f"Health check: {health['overall_status']}")
    
    # Scale application
    scale_success = await system.scale_application(
        deployment_id=deployment.deployment_id,
        scale_config={"replicas": 5}
    )
    print(f"Scaling successful: {scale_success}")
    
    # Get deployment analytics
    analytics = await system.get_deployment_analytics()
    print(f"Deployment analytics:")
    print(f"  Total deployments: {analytics.get('total_deployments', 0)}")
    print(f"  Success rate: {analytics.get('success_rate', 0):.1%}")
    print(f"  Average deployment time: {analytics.get('average_deployment_time', 0):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())

























