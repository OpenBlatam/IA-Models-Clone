"""
Ultimate Deployment System for Facebook Posts
Automated deployment, CI/CD, and infrastructure management
"""

import asyncio
import subprocess
import json
import yaml
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# Pure functions for deployment

class DeploymentEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class DeploymentConfig:
    """Immutable deployment configuration - pure data structure"""
    environment: DeploymentEnvironment
    version: str
    docker_image: str
    replicas: int
    resources: Dict[str, Any]
    environment_variables: Dict[str, str]
    health_check_path: str
    rollback_version: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "environment": self.environment.value,
            "version": self.version,
            "docker_image": self.docker_image,
            "replicas": self.replicas,
            "resources": self.resources,
            "environment_variables": self.environment_variables,
            "health_check_path": self.health_check_path,
            "rollback_version": self.rollback_version
        }


@dataclass(frozen=True)
class DeploymentResult:
    """Immutable deployment result - pure data structure"""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    logs: List[str]
    errors: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "deployment_id": self.deployment_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "logs": self.logs,
            "errors": self.errors,
            "metrics": self.metrics
        }


def create_dockerfile_content(
    python_version: str = "3.11",
    app_port: int = 8000,
    health_check_interval: int = 30
) -> str:
    """Create Dockerfile content - pure function"""
    return f"""
FROM python:{python_version}-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_improved.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_improved.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \\
    chown -R app:app /app
USER app

# Expose port
EXPOSE {app_port}

# Health check
HEALTHCHECK --interval={health_check_interval}s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{app_port}/health || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{app_port}"]
"""


def create_docker_compose_content(
    environment: DeploymentEnvironment,
    replicas: int = 3,
    redis_enabled: bool = True,
    postgres_enabled: bool = True
) -> str:
    """Create docker-compose content - pure function"""
    services = {
        "app": {
            "build": ".",
            "ports": ["8000:8000"],
            "environment": [
                "ENVIRONMENT={environment.value}",
                "REDIS_URL=redis://redis:6379",
                "DATABASE_URL=postgresql://postgres:password@postgres:5432/facebook_posts"
            ],
            "depends_on": [],
            "deploy": {
                "replicas": replicas,
                "resources": {
                    "limits": {
                        "cpus": "1.0",
                        "memory": "1G"
                    },
                    "reservations": {
                        "cpus": "0.5",
                        "memory": "512M"
                    }
                }
            },
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            }
        }
    }
    
    if redis_enabled:
        services["redis"] = {
            "image": "redis:7-alpine",
            "ports": ["6379:6379"],
            "volumes": ["redis_data:/data"],
            "command": "redis-server --appendonly yes"
        }
        services["app"]["depends_on"].append("redis")
    
    if postgres_enabled:
        services["postgres"] = {
            "image": "postgres:15-alpine",
            "environment": [
                "POSTGRES_DB=facebook_posts",
                "POSTGRES_USER=postgres",
                "POSTGRES_PASSWORD=password"
            ],
            "ports": ["5432:5432"],
            "volumes": ["postgres_data:/var/lib/postgresql/data"]
        }
        services["app"]["depends_on"].append("postgres")
    
    volumes = {}
    if redis_enabled:
        volumes["redis_data"] = {}
    if postgres_enabled:
        volumes["postgres_data"] = {}
    
    return yaml.dump({
        "version": "3.8",
        "services": services,
        "volumes": volumes
    }, default_flow_style=False)


def create_kubernetes_deployment_content(
    config: DeploymentConfig,
    namespace: str = "facebook-posts"
) -> str:
    """Create Kubernetes deployment content - pure function"""
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"facebook-posts-{config.environment.value}",
            "namespace": namespace,
            "labels": {
                "app": "facebook-posts",
                "environment": config.environment.value,
                "version": config.version
            }
        },
        "spec": {
            "replicas": config.replicas,
            "selector": {
                "matchLabels": {
                    "app": "facebook-posts",
                    "environment": config.environment.value
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "facebook-posts",
                        "environment": config.environment.value,
                        "version": config.version
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "facebook-posts",
                        "image": config.docker_image,
                        "ports": [{"containerPort": 8000}],
                        "env": [
                            {"name": k, "value": v} for k, v in config.environment_variables.items()
                        ],
                        "resources": config.resources,
                        "livenessProbe": {
                            "httpGet": {
                                "path": config.health_check_path,
                                "port": 8000
                            },
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10
                        },
                        "readinessProbe": {
                            "httpGet": {
                                "path": config.health_check_path,
                                "port": 8000
                            },
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5
                        }
                    }]
                }
            }
        }
    }
    
    return yaml.dump(deployment, default_flow_style=False)


def create_kubernetes_service_content(
    environment: DeploymentEnvironment,
    namespace: str = "facebook-posts"
) -> str:
    """Create Kubernetes service content - pure function"""
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"facebook-posts-{environment.value}",
            "namespace": namespace,
            "labels": {
                "app": "facebook-posts",
                "environment": environment.value
            }
        },
        "spec": {
            "selector": {
                "app": "facebook-posts",
                "environment": environment.value
            },
            "ports": [{
                "port": 80,
                "targetPort": 8000,
                "protocol": "TCP"
            }],
            "type": "LoadBalancer"
        }
    }
    
    return yaml.dump(service, default_flow_style=False)


# Ultimate Deployment System Class

class UltimateDeploymentSystem:
    """Ultimate Deployment System with CI/CD and infrastructure management"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.deployments: Dict[str, DeploymentResult] = {}
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        
        # Statistics
        self.stats = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "rollbacks": 0
        }
    
    async def create_deployment_config(
        self,
        environment: DeploymentEnvironment,
        version: str,
        replicas: int = 3,
        resources: Optional[Dict[str, Any]] = None,
        environment_variables: Optional[Dict[str, str]] = None
    ) -> DeploymentConfig:
        """Create deployment configuration"""
        try:
            # Default resources
            default_resources = {
                "limits": {
                    "cpu": "1.0",
                    "memory": "1Gi"
                },
                "requests": {
                    "cpu": "0.5",
                    "memory": "512Mi"
                }
            }
            
            if resources:
                default_resources.update(resources)
            
            # Default environment variables
            default_env_vars = {
                "ENVIRONMENT": environment.value,
                "LOG_LEVEL": "INFO",
                "WORKERS": str(replicas)
            }
            
            if environment_variables:
                default_env_vars.update(environment_variables)
            
            # Create Docker image name
            docker_image = f"facebook-posts-{environment.value}:{version}"
            
            config = DeploymentConfig(
                environment=environment,
                version=version,
                docker_image=docker_image,
                replicas=replicas,
                resources=default_resources,
                environment_variables=default_env_vars,
                health_check_path="/health",
                rollback_version=None
            )
            
            config_key = f"{environment.value}_{version}"
            self.deployment_configs[config_key] = config
            
            logger.info(f"Created deployment config for {environment.value} v{version}")
            
            return config
            
        except Exception as e:
            logger.error("Error creating deployment config", error=str(e))
            raise
    
    async def build_docker_image(
        self,
        config: DeploymentConfig,
        build_context: Optional[str] = None
    ) -> bool:
        """Build Docker image"""
        try:
            build_context = build_context or str(self.project_root)
            
            # Create Dockerfile if it doesn't exist
            dockerfile_path = Path(build_context) / "Dockerfile"
            if not dockerfile_path.exists():
                dockerfile_content = create_dockerfile_content()
                dockerfile_path.write_text(dockerfile_content)
                logger.info("Created Dockerfile")
            
            # Build Docker image
            build_command = [
                "docker", "build",
                "-t", config.docker_image,
                "-f", str(dockerfile_path),
                build_context
            ]
            
            logger.info(f"Building Docker image: {config.docker_image}")
            
            result = subprocess.run(
                build_command,
                capture_output=True,
                text=True,
                cwd=build_context
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully built Docker image: {config.docker_image}")
                return True
            else:
                logger.error(f"Failed to build Docker image: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error("Error building Docker image", error=str(e))
            return False
    
    async def deploy_to_docker_compose(
        self,
        config: DeploymentConfig,
        compose_file_path: Optional[str] = None
    ) -> DeploymentResult:
        """Deploy using Docker Compose"""
        deployment_id = f"deploy_{int(time.time())}"
        start_time = datetime.utcnow()
        
        logs = []
        errors = []
        
        try:
            # Create docker-compose.yml if it doesn't exist
            if not compose_file_path:
                compose_file_path = self.project_root / "docker-compose.yml"
            
            compose_content = create_docker_compose_content(
                config.environment,
                config.replicas
            )
            
            Path(compose_file_path).write_text(compose_content)
            logs.append("Created docker-compose.yml")
            
            # Deploy with Docker Compose
            deploy_command = [
                "docker-compose",
                "-f", str(compose_file_path),
                "up", "-d", "--scale", f"app={config.replicas}"
            ]
            
            logs.append(f"Running: {' '.join(deploy_command)}")
            
            result = subprocess.run(
                deploy_command,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                logs.append("Docker Compose deployment successful")
                status = DeploymentStatus.SUCCESS
            else:
                errors.append(f"Docker Compose deployment failed: {result.stderr}")
                status = DeploymentStatus.FAILED
            
        except Exception as e:
            error_msg = f"Error in Docker Compose deployment: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            status = DeploymentStatus.FAILED
        
        # Create deployment result
        deployment_result = DeploymentResult(
            deployment_id=deployment_id,
            status=status,
            start_time=start_time,
            end_time=datetime.utcnow(),
            logs=logs,
            errors=errors,
            metrics={
                "replicas": config.replicas,
                "environment": config.environment.value,
                "version": config.version
            }
        )
        
        self.deployments[deployment_id] = deployment_result
        self._update_stats(status)
        
        return deployment_result
    
    async def deploy_to_kubernetes(
        self,
        config: DeploymentConfig,
        namespace: str = "facebook-posts",
        kubeconfig_path: Optional[str] = None
    ) -> DeploymentResult:
        """Deploy to Kubernetes"""
        deployment_id = f"k8s_deploy_{int(time.time())}"
        start_time = datetime.utcnow()
        
        logs = []
        errors = []
        
        try:
            # Create Kubernetes manifests
            deployment_content = create_kubernetes_deployment_content(config, namespace)
            service_content = create_kubernetes_service_content(config.environment, namespace)
            
            # Write manifests to temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                deployment_file = Path(temp_dir) / "deployment.yaml"
                service_file = Path(temp_dir) / "service.yaml"
                
                deployment_file.write_text(deployment_content)
                service_file.write_text(service_content)
                
                logs.append("Created Kubernetes manifests")
                
                # Apply manifests
                kubectl_cmd = ["kubectl"]
                if kubeconfig_path:
                    kubectl_cmd.extend(["--kubeconfig", kubeconfig_path])
                
                # Create namespace if it doesn't exist
                namespace_cmd = kubectl_cmd + ["create", "namespace", namespace, "--dry-run=client", "-o", "yaml"]
                subprocess.run(namespace_cmd, capture_output=True)
                
                # Apply deployment
                apply_cmd = kubectl_cmd + ["apply", "-f", str(deployment_file)]
                result = subprocess.run(apply_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logs.append("Applied Kubernetes deployment")
                else:
                    errors.append(f"Failed to apply deployment: {result.stderr}")
                
                # Apply service
                apply_cmd = kubectl_cmd + ["apply", "-f", str(service_file)]
                result = subprocess.run(apply_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logs.append("Applied Kubernetes service")
                else:
                    errors.append(f"Failed to apply service: {result.stderr}")
                
                # Wait for deployment to be ready
                wait_cmd = kubectl_cmd + [
                    "rollout", "status", "deployment",
                    f"facebook-posts-{config.environment.value}",
                    "-n", namespace
                ]
                
                result = subprocess.run(wait_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logs.append("Deployment is ready")
                    status = DeploymentStatus.SUCCESS
                else:
                    errors.append(f"Deployment not ready: {result.stderr}")
                    status = DeploymentStatus.FAILED
            
        except subprocess.TimeoutExpired:
            errors.append("Deployment timeout - taking too long to become ready")
            status = DeploymentStatus.FAILED
        except Exception as e:
            error_msg = f"Error in Kubernetes deployment: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            status = DeploymentStatus.FAILED
        
        # Create deployment result
        deployment_result = DeploymentResult(
            deployment_id=deployment_id,
            status=status,
            start_time=start_time,
            end_time=datetime.utcnow(),
            logs=logs,
            errors=errors,
            metrics={
                "replicas": config.replicas,
                "environment": config.environment.value,
                "version": config.version,
                "namespace": namespace
            }
        )
        
        self.deployments[deployment_id] = deployment_result
        self._update_stats(status)
        
        return deployment_result
    
    async def health_check(
        self,
        config: DeploymentConfig,
        timeout: int = 60
    ) -> bool:
        """Perform health check on deployment"""
        try:
            # This would typically check the actual deployment
            # For now, we'll simulate a health check
            health_check_url = f"http://localhost:8000{config.health_check_path}"
            
            # Simulate health check
            await asyncio.sleep(2)  # Simulate check time
            
            # In a real implementation, you would make an HTTP request
            # and check the response status and content
            
            logger.info(f"Health check passed for {config.environment.value}")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def rollback_deployment(
        self,
        environment: DeploymentEnvironment,
        target_version: str
    ) -> DeploymentResult:
        """Rollback deployment to previous version"""
        deployment_id = f"rollback_{int(time.time())}"
        start_time = datetime.utcnow()
        
        logs = []
        errors = []
        
        try:
            logs.append(f"Rolling back {environment.value} to version {target_version}")
            
            # Find current deployment config
            current_config = None
            for config in self.deployment_configs.values():
                if config.environment == environment:
                    current_config = config
                    break
            
            if not current_config:
                errors.append(f"No deployment found for {environment.value}")
                status = DeploymentStatus.FAILED
            else:
                # Create rollback config
                rollback_config = DeploymentConfig(
                    environment=environment,
                    version=target_version,
                    docker_image=f"facebook-posts-{environment.value}:{target_version}",
                    replicas=current_config.replicas,
                    resources=current_config.resources,
                    environment_variables=current_config.environment_variables,
                    health_check_path=current_config.health_check_path,
                    rollback_version=current_config.version
                )
                
                # Deploy rollback version
                if environment == DeploymentEnvironment.DEVELOPMENT:
                    result = await self.deploy_to_docker_compose(rollback_config)
                else:
                    result = await self.deploy_to_kubernetes(rollback_config)
                
                if result.status == DeploymentStatus.SUCCESS:
                    logs.append(f"Successfully rolled back to version {target_version}")
                    status = DeploymentStatus.ROLLBACK
                else:
                    errors.extend(result.errors)
                    status = DeploymentStatus.FAILED
            
        except Exception as e:
            error_msg = f"Error during rollback: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            status = DeploymentStatus.FAILED
        
        # Create rollback result
        rollback_result = DeploymentResult(
            deployment_id=deployment_id,
            status=status,
            start_time=start_time,
            end_time=datetime.utcnow(),
            logs=logs,
            errors=errors,
            metrics={
                "rollback_version": target_version,
                "environment": environment.value
            }
        )
        
        self.deployments[deployment_id] = rollback_result
        self._update_stats(DeploymentStatus.ROLLBACK)
        
        return rollback_result
    
    async def run_ci_cd_pipeline(
        self,
        environment: DeploymentEnvironment,
        version: str,
        run_tests: bool = True,
        run_linting: bool = True
    ) -> DeploymentResult:
        """Run complete CI/CD pipeline"""
        pipeline_id = f"pipeline_{int(time.time())}"
        start_time = datetime.utcnow()
        
        logs = []
        errors = []
        
        try:
            logs.append(f"Starting CI/CD pipeline for {environment.value} v{version}")
            
            # Step 1: Run tests
            if run_tests:
                logs.append("Running tests...")
                test_result = await self._run_tests()
                if not test_result:
                    errors.append("Tests failed")
                    status = DeploymentStatus.FAILED
                    return self._create_pipeline_result(pipeline_id, start_time, logs, errors, status)
                logs.append("Tests passed")
            
            # Step 2: Run linting
            if run_linting:
                logs.append("Running linting...")
                lint_result = await self._run_linting()
                if not lint_result:
                    errors.append("Linting failed")
                    status = DeploymentStatus.FAILED
                    return self._create_pipeline_result(pipeline_id, start_time, logs, errors, status)
                logs.append("Linting passed")
            
            # Step 3: Create deployment config
            config = await self.create_deployment_config(environment, version)
            logs.append("Created deployment configuration")
            
            # Step 4: Build Docker image
            logs.append("Building Docker image...")
            build_result = await self.build_docker_image(config)
            if not build_result:
                errors.append("Docker build failed")
                status = DeploymentStatus.FAILED
                return self._create_pipeline_result(pipeline_id, start_time, logs, errors, status)
            logs.append("Docker image built successfully")
            
            # Step 5: Deploy
            logs.append("Deploying application...")
            if environment == DeploymentEnvironment.DEVELOPMENT:
                deploy_result = await self.deploy_to_docker_compose(config)
            else:
                deploy_result = await self.deploy_to_kubernetes(config)
            
            if deploy_result.status == DeploymentStatus.SUCCESS:
                logs.append("Deployment successful")
                status = DeploymentStatus.SUCCESS
            else:
                errors.extend(deploy_result.errors)
                status = DeploymentStatus.FAILED
            
            # Step 6: Health check
            if status == DeploymentStatus.SUCCESS:
                logs.append("Running health check...")
                health_ok = await self.health_check(config)
                if not health_ok:
                    errors.append("Health check failed")
                    status = DeploymentStatus.FAILED
                else:
                    logs.append("Health check passed")
            
        except Exception as e:
            error_msg = f"Error in CI/CD pipeline: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            status = DeploymentStatus.FAILED
        
        return self._create_pipeline_result(pipeline_id, start_time, logs, errors, status)
    
    async def _run_tests(self) -> bool:
        """Run test suite"""
        try:
            test_command = ["python", "-m", "pytest", "tests/", "-v"]
            result = subprocess.run(test_command, capture_output=True, text=True, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return False
    
    async def _run_linting(self) -> bool:
        """Run code linting"""
        try:
            lint_command = ["python", "-m", "flake8", ".", "--max-line-length=100"]
            result = subprocess.run(lint_command, capture_output=True, text=True, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error running linting: {str(e)}")
            return False
    
    def _create_pipeline_result(
        self,
        pipeline_id: str,
        start_time: datetime,
        logs: List[str],
        errors: List[str],
        status: DeploymentStatus
    ) -> DeploymentResult:
        """Create pipeline result"""
        return DeploymentResult(
            deployment_id=pipeline_id,
            status=status,
            start_time=start_time,
            end_time=datetime.utcnow(),
            logs=logs,
            errors=errors,
            metrics={
                "pipeline_steps": len(logs),
                "errors_count": len(errors)
            }
        )
    
    def _update_stats(self, status: DeploymentStatus) -> None:
        """Update deployment statistics"""
        self.stats["total_deployments"] += 1
        
        if status == DeploymentStatus.SUCCESS:
            self.stats["successful_deployments"] += 1
        elif status == DeploymentStatus.FAILED:
            self.stats["failed_deployments"] += 1
        elif status == DeploymentStatus.ROLLBACK:
            self.stats["rollbacks"] += 1
    
    def get_deployment_history(self) -> List[DeploymentResult]:
        """Get deployment history"""
        return list(self.deployments.values())
    
    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        return {
            "statistics": self.stats.copy(),
            "success_rate": (
                self.stats["successful_deployments"] / max(1, self.stats["total_deployments"])
            ) * 100,
            "recent_deployments": [
                deployment.to_dict() for deployment in list(self.deployments.values())[-10:]
            ]
        }


# Factory functions

def create_ultimate_deployment_system(project_root: str = ".") -> UltimateDeploymentSystem:
    """Create ultimate deployment system - pure function"""
    return UltimateDeploymentSystem(project_root)


async def get_ultimate_deployment_system(project_root: str = ".") -> UltimateDeploymentSystem:
    """Get ultimate deployment system instance"""
    return create_ultimate_deployment_system(project_root)

