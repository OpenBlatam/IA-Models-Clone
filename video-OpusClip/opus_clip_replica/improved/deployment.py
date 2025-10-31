"""
Advanced Deployment System for OpusClip Improved
==============================================

Comprehensive deployment automation with CI/CD, monitoring, and rollback capabilities.
"""

import asyncio
import logging
import json
import yaml
import subprocess
import shutil
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
from pathlib import Path
import docker
import kubernetes
from kubernetes import client, config
import requests
import psutil

from .schemas import get_settings
from .exceptions import DeploymentError, create_deployment_error

logger = logging.getLogger(__name__)


class DeploymentStatus(str, Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class EnvironmentType(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStrategy(str, Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str
    name: str
    version: str
    environment: EnvironmentType
    strategy: DeploymentStrategy
    replicas: int
    resources: Dict[str, Any]
    health_check: Dict[str, Any]
    rollback_config: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class DeploymentResult:
    """Deployment result"""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    logs: List[str] = None
    metrics: Dict[str, Any] = None
    error_message: Optional[str] = None


class DockerManager:
    """Docker container management"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.registry_url = "registry.opusclip.com"
        self.image_prefix = "opusclip"
    
    def build_image(self, dockerfile_path: str, tag: str, build_args: Dict[str, str] = None) -> bool:
        """Build Docker image"""
        try:
            logger.info(f"Building Docker image: {tag}")
            
            # Build image
            image, build_logs = self.client.images.build(
                path=dockerfile_path,
                tag=tag,
                buildargs=build_args or {},
                rm=True,
                forcerm=True
            )
            
            logger.info(f"Docker image built successfully: {tag}")
            return True
            
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            raise create_deployment_error("docker_build", tag, e)
    
    def push_image(self, tag: str) -> bool:
        """Push Docker image to registry"""
        try:
            logger.info(f"Pushing Docker image: {tag}")
            
            # Push image
            push_logs = self.client.images.push(tag)
            
            logger.info(f"Docker image pushed successfully: {tag}")
            return True
            
        except Exception as e:
            logger.error(f"Docker push failed: {e}")
            raise create_deployment_error("docker_push", tag, e)
    
    def pull_image(self, tag: str) -> bool:
        """Pull Docker image from registry"""
        try:
            logger.info(f"Pulling Docker image: {tag}")
            
            # Pull image
            self.client.images.pull(tag)
            
            logger.info(f"Docker image pulled successfully: {tag}")
            return True
            
        except Exception as e:
            logger.error(f"Docker pull failed: {e}")
            raise create_deployment_error("docker_pull", tag, e)
    
    def run_container(self, image: str, name: str, ports: Dict[str, str] = None, 
                     environment: Dict[str, str] = None, volumes: Dict[str, str] = None) -> str:
        """Run Docker container"""
        try:
            logger.info(f"Running Docker container: {name}")
            
            # Run container
            container = self.client.containers.run(
                image=image,
                name=name,
                ports=ports or {},
                environment=environment or {},
                volumes=volumes or {},
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            container_id = container.id
            logger.info(f"Docker container started: {name} ({container_id})")
            return container_id
            
        except Exception as e:
            logger.error(f"Docker container run failed: {e}")
            raise create_deployment_error("docker_run", name, e)
    
    def stop_container(self, container_id: str) -> bool:
        """Stop Docker container"""
        try:
            container = self.client.containers.get(container_id)
            container.stop()
            container.remove()
            
            logger.info(f"Docker container stopped: {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Docker container stop failed: {e}")
            raise create_deployment_error("docker_stop", container_id, e)
    
    def get_container_logs(self, container_id: str, tail: int = 100) -> List[str]:
        """Get Docker container logs"""
        try:
            container = self.client.containers.get(container_id)
            logs = container.logs(tail=tail).decode('utf-8').split('\n')
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get container logs: {e}")
            return []
    
    def get_container_status(self, container_id: str) -> Dict[str, Any]:
        """Get Docker container status"""
        try:
            container = self.client.containers.get(container_id)
            return {
                "id": container.id,
                "name": container.name,
                "status": container.status,
                "image": container.image.tags[0] if container.image.tags else container.image.id,
                "created": container.attrs["Created"],
                "ports": container.attrs["NetworkSettings"]["Ports"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return {}


class KubernetesManager:
    """Kubernetes cluster management"""
    
    def __init__(self):
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.namespace = "opusclip"
    
    def create_namespace(self, namespace: str) -> bool:
        """Create Kubernetes namespace"""
        try:
            namespace_manifest = {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {
                    "name": namespace
                }
            }
            
            self.v1.create_namespace(body=namespace_manifest)
            logger.info(f"Kubernetes namespace created: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes namespace creation failed: {e}")
            raise create_deployment_error("k8s_namespace", namespace, e)
    
    def deploy_application(self, deployment_config: DeploymentConfig) -> bool:
        """Deploy application to Kubernetes"""
        try:
            # Create deployment manifest
            deployment_manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": deployment_config.name,
                    "namespace": self.namespace,
                    "labels": {
                        "app": deployment_config.name,
                        "version": deployment_config.version
                    }
                },
                "spec": {
                    "replicas": deployment_config.replicas,
                    "selector": {
                        "matchLabels": {
                            "app": deployment_config.name
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": deployment_config.name,
                                "version": deployment_config.version
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": deployment_config.name,
                                "image": f"{deployment_config.name}:{deployment_config.version}",
                                "ports": [{
                                    "containerPort": 8000
                                }],
                                "resources": deployment_config.resources,
                                "livenessProbe": deployment_config.health_check.get("liveness"),
                                "readinessProbe": deployment_config.health_check.get("readiness")
                            }]
                        }
                    }
                }
            }
            
            # Create deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment_manifest
            )
            
            logger.info(f"Kubernetes deployment created: {deployment_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            raise create_deployment_error("k8s_deploy", deployment_config.name, e)
    
    def create_service(self, service_name: str, app_name: str, port: int = 8000) -> bool:
        """Create Kubernetes service"""
        try:
            service_manifest = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": service_name,
                    "namespace": self.namespace
                },
                "spec": {
                    "selector": {
                        "app": app_name
                    },
                    "ports": [{
                        "port": port,
                        "targetPort": port
                    }],
                    "type": "LoadBalancer"
                }
            }
            
            self.v1.create_namespaced_service(
                namespace=self.namespace,
                body=service_manifest
            )
            
            logger.info(f"Kubernetes service created: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes service creation failed: {e}")
            raise create_deployment_error("k8s_service", service_name, e)
    
    def create_ingress(self, ingress_name: str, service_name: str, host: str) -> bool:
        """Create Kubernetes ingress"""
        try:
            ingress_manifest = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "metadata": {
                    "name": ingress_name,
                    "namespace": self.namespace,
                    "annotations": {
                        "nginx.ingress.kubernetes.io/rewrite-target": "/"
                    }
                },
                "spec": {
                    "rules": [{
                        "host": host,
                        "http": {
                            "paths": [{
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": service_name,
                                        "port": {
                                            "number": 8000
                                        }
                                    }
                                }
                            }]
                        }
                    }]
                }
            }
            
            self.networking_v1.create_namespaced_ingress(
                namespace=self.namespace,
                body=ingress_manifest
            )
            
            logger.info(f"Kubernetes ingress created: {ingress_name}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes ingress creation failed: {e}")
            raise create_deployment_error("k8s_ingress", ingress_name, e)
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get Kubernetes deployment status"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            return {
                "name": deployment.metadata.name,
                "namespace": deployment.metadata.namespace,
                "replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas,
                "available_replicas": deployment.status.available_replicas,
                "updated_replicas": deployment.status.updated_replicas,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "message": condition.message
                    } for condition in deployment.status.conditions
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    def scale_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Scale Kubernetes deployment"""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Kubernetes deployment scaled: {deployment_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes deployment scaling failed: {e}")
            raise create_deployment_error("k8s_scale", deployment_name, e)
    
    def rollback_deployment(self, deployment_name: str, revision: int = None) -> bool:
        """Rollback Kubernetes deployment"""
        try:
            if revision is None:
                # Get deployment rollout history
                rollout_history = self.apps_v1.read_namespaced_deployment_rollback(
                    name=deployment_name,
                    namespace=self.namespace
                )
                revision = rollout_history.revision - 1
            
            # Rollback to previous revision
            rollback_body = {
                "name": deployment_name,
                "rollbackTo": {
                    "revision": revision
                }
            }
            
            self.apps_v1.create_namespaced_deployment_rollback(
                name=deployment_name,
                namespace=self.namespace,
                body=rollback_body
            )
            
            logger.info(f"Kubernetes deployment rolled back: {deployment_name} to revision {revision}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes deployment rollback failed: {e}")
            raise create_deployment_error("k8s_rollback", deployment_name, e)


class HealthChecker:
    """Application health checking"""
    
    def __init__(self):
        self.health_endpoints = {
            "api": "/api/v2/opus-clip/health",
            "metrics": "/metrics",
            "readiness": "/api/v2/opus-clip/health/readiness",
            "liveness": "/api/v2/opus-clip/health/liveness"
        }
    
    def check_health(self, base_url: str, timeout: int = 30) -> Dict[str, Any]:
        """Check application health"""
        try:
            health_status = {
                "overall": "healthy",
                "checks": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            for check_name, endpoint in self.health_endpoints.items():
                try:
                    response = requests.get(
                        f"{base_url}{endpoint}",
                        timeout=timeout
                    )
                    
                    if response.status_code == 200:
                        health_status["checks"][check_name] = {
                            "status": "healthy",
                            "response_time": response.elapsed.total_seconds(),
                            "status_code": response.status_code
                        }
                    else:
                        health_status["checks"][check_name] = {
                            "status": "unhealthy",
                            "response_time": response.elapsed.total_seconds(),
                            "status_code": response.status_code
                        }
                        health_status["overall"] = "unhealthy"
                        
                except Exception as e:
                    health_status["checks"][check_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["overall"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def wait_for_healthy(self, base_url: str, max_wait: int = 300, check_interval: int = 10) -> bool:
        """Wait for application to become healthy"""
        try:
            start_time = datetime.utcnow()
            
            while (datetime.utcnow() - start_time).total_seconds() < max_wait:
                health_status = self.check_health(base_url)
                
                if health_status["overall"] == "healthy":
                    logger.info(f"Application is healthy: {base_url}")
                    return True
                
                logger.info(f"Waiting for application to become healthy: {base_url}")
                time.sleep(check_interval)
            
            logger.error(f"Application did not become healthy within {max_wait} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Health wait failed: {e}")
            return False


class DeploymentManager:
    """Main deployment manager"""
    
    def __init__(self):
        self.settings = get_settings()
        self.docker_manager = DockerManager()
        self.k8s_manager = KubernetesManager()
        self.health_checker = HealthChecker()
        self.deployments: Dict[str, DeploymentResult] = {}
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
    
    def create_deployment_config(self, name: str, version: str, environment: EnvironmentType,
                               strategy: DeploymentStrategy, replicas: int = 3) -> DeploymentConfig:
        """Create deployment configuration"""
        try:
            config = DeploymentConfig(
                deployment_id=str(uuid4()),
                name=name,
                version=version,
                environment=environment,
                strategy=strategy,
                replicas=replicas,
                resources={
                    "requests": {
                        "cpu": "500m",
                        "memory": "1Gi"
                    },
                    "limits": {
                        "cpu": "2000m",
                        "memory": "4Gi"
                    }
                },
                health_check={
                    "liveness": {
                        "httpGet": {
                            "path": "/api/v2/opus-clip/health/liveness",
                            "port": 8000
                        },
                        "initialDelaySeconds": 30,
                        "periodSeconds": 10
                    },
                    "readiness": {
                        "httpGet": {
                            "path": "/api/v2/opus-clip/health/readiness",
                            "port": 8000
                        },
                        "initialDelaySeconds": 5,
                        "periodSeconds": 5
                    }
                },
                rollback_config={
                    "enabled": True,
                    "max_revisions": 10
                }
            )
            
            self.deployment_configs[config.deployment_id] = config
            logger.info(f"Deployment configuration created: {name} v{version}")
            return config
            
        except Exception as e:
            logger.error(f"Deployment configuration creation failed: {e}")
            raise create_deployment_error("deployment_config", name, e)
    
    async def deploy_application(self, deployment_config: DeploymentConfig) -> DeploymentResult:
        """Deploy application"""
        try:
            deployment_id = deployment_config.deployment_id
            start_time = datetime.utcnow()
            
            # Create deployment result
            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.IN_PROGRESS,
                start_time=start_time,
                logs=[]
            )
            
            self.deployments[deployment_id] = result
            
            logger.info(f"Starting deployment: {deployment_config.name} v{deployment_config.version}")
            
            try:
                # Step 1: Build Docker image
                image_tag = f"{deployment_config.name}:{deployment_config.version}"
                result.logs.append("Building Docker image...")
                
                self.docker_manager.build_image(
                    dockerfile_path=".",
                    tag=image_tag
                )
                result.logs.append("Docker image built successfully")
                
                # Step 2: Push image to registry
                result.logs.append("Pushing Docker image to registry...")
                self.docker_manager.push_image(image_tag)
                result.logs.append("Docker image pushed successfully")
                
                # Step 3: Deploy to Kubernetes
                result.logs.append("Deploying to Kubernetes...")
                self.k8s_manager.deploy_application(deployment_config)
                result.logs.append("Kubernetes deployment created")
                
                # Step 4: Create service
                service_name = f"{deployment_config.name}-service"
                result.logs.append("Creating Kubernetes service...")
                self.k8s_manager.create_service(service_name, deployment_config.name)
                result.logs.append("Kubernetes service created")
                
                # Step 5: Create ingress
                ingress_name = f"{deployment_config.name}-ingress"
                host = f"{deployment_config.name}.{deployment_config.environment.value}.opusclip.com"
                result.logs.append("Creating Kubernetes ingress...")
                self.k8s_manager.create_ingress(ingress_name, service_name, host)
                result.logs.append("Kubernetes ingress created")
                
                # Step 6: Wait for deployment to be ready
                result.logs.append("Waiting for deployment to be ready...")
                await asyncio.sleep(30)  # Wait for pods to start
                
                # Step 7: Health check
                base_url = f"https://{host}"
                result.logs.append("Performing health checks...")
                
                if self.health_checker.wait_for_healthy(base_url):
                    result.logs.append("Health checks passed")
                    result.status = DeploymentStatus.COMPLETED
                else:
                    result.logs.append("Health checks failed")
                    result.status = DeploymentStatus.FAILED
                    result.error_message = "Health checks failed"
                
                # Update result
                result.end_time = datetime.utcnow()
                result.duration = (result.end_time - result.start_time).total_seconds()
                
                logger.info(f"Deployment completed: {deployment_config.name} v{deployment_config.version}")
                
            except Exception as e:
                result.status = DeploymentStatus.FAILED
                result.error_message = str(e)
                result.end_time = datetime.utcnow()
                result.duration = (result.end_time - result.start_time).total_seconds()
                result.logs.append(f"Deployment failed: {e}")
                
                logger.error(f"Deployment failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Deployment process failed: {e}")
            raise create_deployment_error("deployment_process", deployment_config.name, e)
    
    def rollback_deployment(self, deployment_name: str, revision: int = None) -> bool:
        """Rollback deployment"""
        try:
            logger.info(f"Rolling back deployment: {deployment_name}")
            
            # Rollback Kubernetes deployment
            self.k8s_manager.rollback_deployment(deployment_name, revision)
            
            # Wait for rollback to complete
            time.sleep(30)
            
            # Verify rollback
            status = self.k8s_manager.get_deployment_status(deployment_name)
            if status.get("ready_replicas", 0) > 0:
                logger.info(f"Deployment rollback successful: {deployment_name}")
                return True
            else:
                logger.error(f"Deployment rollback failed: {deployment_name}")
                return False
                
        except Exception as e:
            logger.error(f"Deployment rollback failed: {e}")
            raise create_deployment_error("deployment_rollback", deployment_name, e)
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            # Get Kubernetes deployment status
            k8s_status = self.k8s_manager.get_deployment_status(deployment_name)
            
            # Get deployment result
            deployment_results = [r for r in self.deployments.values() 
                                if r.deployment_id in [c.deployment_id for c in self.deployment_configs.values() 
                                                      if c.name == deployment_name]]
            
            latest_result = max(deployment_results, key=lambda x: x.start_time) if deployment_results else None
            
            return {
                "deployment_name": deployment_name,
                "kubernetes_status": k8s_status,
                "deployment_result": asdict(latest_result) if latest_result else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    def get_deployment_logs(self, deployment_id: str) -> List[str]:
        """Get deployment logs"""
        if deployment_id in self.deployments:
            return self.deployments[deployment_id].logs
        return []
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        try:
            deployments = []
            
            for config in self.deployment_configs.values():
                status = self.get_deployment_status(config.name)
                deployments.append({
                    "deployment_id": config.deployment_id,
                    "name": config.name,
                    "version": config.version,
                    "environment": config.environment.value,
                    "strategy": config.strategy.value,
                    "replicas": config.replicas,
                    "status": status.get("kubernetes_status", {}).get("ready_replicas", 0),
                    "created_at": config.created_at.isoformat() if config.created_at else None
                })
            
            return deployments
            
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")
            return []
    
    def cleanup_old_deployments(self, keep_count: int = 5):
        """Cleanup old deployments"""
        try:
            # Get all deployments sorted by creation time
            all_deployments = list(self.deployment_configs.values())
            all_deployments.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
            
            # Keep only the most recent deployments
            deployments_to_remove = all_deployments[keep_count:]
            
            for config in deployments_to_remove:
                # Remove from Kubernetes
                try:
                    self.k8s_manager.apps_v1.delete_namespaced_deployment(
                        name=config.name,
                        namespace=self.k8s_manager.namespace
                    )
                except:
                    pass  # Deployment might not exist
                
                # Remove from local tracking
                del self.deployment_configs[config.deployment_id]
                if config.deployment_id in self.deployments:
                    del self.deployments[config.deployment_id]
            
            logger.info(f"Cleaned up {len(deployments_to_remove)} old deployments")
            
        except Exception as e:
            logger.error(f"Deployment cleanup failed: {e}")
    
    def get_deployment_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        try:
            total_deployments = len(self.deployment_configs)
            successful_deployments = len([r for r in self.deployments.values() 
                                        if r.status == DeploymentStatus.COMPLETED])
            failed_deployments = len([r for r in self.deployments.values() 
                                    if r.status == DeploymentStatus.FAILED])
            
            # Calculate average deployment time
            completed_deployments = [r for r in self.deployments.values() 
                                   if r.status == DeploymentStatus.COMPLETED and r.duration]
            avg_deployment_time = (
                sum(r.duration for r in completed_deployments) / len(completed_deployments)
                if completed_deployments else 0
            )
            
            return {
                "total_deployments": total_deployments,
                "successful_deployments": successful_deployments,
                "failed_deployments": failed_deployments,
                "success_rate": (successful_deployments / total_deployments * 100) if total_deployments > 0 else 0,
                "average_deployment_time": round(avg_deployment_time, 2),
                "active_deployments": len([c for c in self.deployment_configs.values() 
                                         if c.environment == EnvironmentType.PRODUCTION])
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment metrics: {e}")
            return {}


# Global deployment manager
deployment_manager = DeploymentManager()





























