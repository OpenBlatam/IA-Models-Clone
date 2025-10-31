#!/usr/bin/env python3
"""
Deployment and CI/CD Scripts for Frontier Model Training
Provides automated deployment, containerization, and CI/CD pipeline management.
"""

import os
import sys
import json
import yaml
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import docker
from docker.errors import DockerException
import kubernetes
from kubernetes import client, config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import requests
import time
import logging

console = Console()

class DeploymentType(Enum):
    """Deployment types."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD = "cloud"
    EDGE = "edge"

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    deployment_type: DeploymentType
    environment: Environment
    image_name: str
    image_tag: str
    replicas: int = 1
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    cpu_request: str = "1"
    memory_request: str = "2Gi"
    gpu_enabled: bool = False
    gpu_count: int = 0
    ports: List[int] = None
    environment_vars: Dict[str, str] = None
    volumes: List[Dict[str, str]] = None
    health_check: Dict[str, Any] = None
    auto_scaling: Dict[str, Any] = None
    secrets: List[str] = None
    config_maps: List[str] = None

@dataclass
class CIConfig:
    """CI/CD configuration."""
    pipeline_name: str
    trigger_branches: List[str]
    build_stages: List[str]
    test_stages: List[str]
    deploy_stages: List[str]
    notification_channels: List[str]
    artifact_registry: str
    deployment_targets: List[str]

class DockerManager:
    """Docker container management."""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
        except DockerException as e:
            console.print(f"[red]Docker not available: {e}[/red]")
            self.client = None
    
    def build_image(self, 
                   dockerfile_path: str,
                   image_name: str,
                   image_tag: str = "latest",
                   build_args: Dict[str, str] = None,
                   context_path: str = ".") -> bool:
        """Build Docker image."""
        if not self.client:
            console.print("[red]Docker client not available[/red]")
            return False
        
        try:
            console.print(f"[blue]Building Docker image: {image_name}:{image_tag}[/blue]")
            
            # Build the image
            image, build_logs = self.client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=f"{image_name}:{image_tag}",
                buildargs=build_args or {},
                rm=True,
                forcerm=True
            )
            
            console.print(f"[green]Successfully built image: {image_name}:{image_tag}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to build image: {e}[/red]")
            return False
    
    def push_image(self, image_name: str, image_tag: str = "latest", registry: str = None) -> bool:
        """Push Docker image to registry."""
        if not self.client:
            console.print("[red]Docker client not available[/red]")
            return False
        
        try:
            full_name = f"{registry}/{image_name}:{image_tag}" if registry else f"{image_name}:{image_tag}"
            console.print(f"[blue]Pushing image: {full_name}[/blue]")
            
            self.client.images.push(full_name)
            console.print(f"[green]Successfully pushed image: {full_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to push image: {e}[/red]")
            return False
    
    def run_container(self, 
                    image_name: str,
                    image_tag: str = "latest",
                    container_name: str = None,
                    ports: Dict[str, str] = None,
                    environment_vars: Dict[str, str] = None,
                    volumes: Dict[str, str] = None,
                    detach: bool = True) -> Optional[str]:
        """Run Docker container."""
        if not self.client:
            console.print("[red]Docker client not available[/red]")
            return None
        
        try:
            full_name = f"{image_name}:{image_tag}"
            console.print(f"[blue]Running container: {full_name}[/blue]")
            
            container = self.client.containers.run(
                image=full_name,
                name=container_name,
                ports=ports,
                environment=environment_vars,
                volumes=volumes,
                detach=detach
            )
            
            container_id = container.id
            console.print(f"[green]Container started: {container_id}[/green]")
            return container_id
            
        except Exception as e:
            console.print(f"[red]Failed to run container: {e}[/red]")
            return None
    
    def stop_container(self, container_id: str) -> bool:
        """Stop Docker container."""
        if not self.client:
            return False
        
        try:
            container = self.client.containers.get(container_id)
            container.stop()
            console.print(f"[green]Container stopped: {container_id}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to stop container: {e}[/red]")
            return False

class KubernetesManager:
    """Kubernetes deployment management."""
    
    def __init__(self, config_path: str = None):
        try:
            if config_path:
                config.load_kube_config(config_file=config_path)
            else:
                config.load_incluster_config()
            
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.batch_v1 = client.BatchV1Api()
            
        except Exception as e:
            console.print(f"[red]Kubernetes not available: {e}[/red]")
            self.v1 = None
            self.apps_v1 = None
            self.batch_v1 = None
    
    def create_deployment(self, config: DeploymentConfig) -> bool:
        """Create Kubernetes deployment."""
        if not self.apps_v1:
            console.print("[red]Kubernetes client not available[/red]")
            return False
        
        try:
            # Create deployment manifest
            deployment_manifest = self._create_deployment_manifest(config)
            
            # Create the deployment
            response = self.apps_v1.create_namespaced_deployment(
                namespace="default",
                body=deployment_manifest
            )
            
            console.print(f"[green]Deployment created: {response.metadata.name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to create deployment: {e}[/red]")
            return False
    
    def create_service(self, config: DeploymentConfig) -> bool:
        """Create Kubernetes service."""
        if not self.v1:
            console.print("[red]Kubernetes client not available[/red]")
            return False
        
        try:
            # Create service manifest
            service_manifest = self._create_service_manifest(config)
            
            # Create the service
            response = self.v1.create_namespaced_service(
                namespace="default",
                body=service_manifest
            )
            
            console.print(f"[green]Service created: {response.metadata.name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to create service: {e}[/red]")
            return False
    
    def _create_deployment_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create deployment manifest."""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "labels": {
                    "app": config.name,
                    "environment": config.environment.value
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.name,
                            "environment": config.environment.value
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.name,
                            "image": f"{config.image_name}:{config.image_tag}",
                            "ports": [{"containerPort": port} for port in (config.ports or [8080])],
                            "resources": {
                                "limits": {
                                    "cpu": config.cpu_limit,
                                    "memory": config.memory_limit
                                },
                                "requests": {
                                    "cpu": config.cpu_request,
                                    "memory": config.memory_request
                                }
                            },
                            "env": [
                                {"name": k, "value": v} 
                                for k, v in (config.environment_vars or {}).items()
                            ]
                        }]
                    }
                }
            }
        }
        
        # Add GPU support if enabled
        if config.gpu_enabled and config.gpu_count > 0:
            manifest["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = str(config.gpu_count)
        
        return manifest
    
    def _create_service_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.name}-service",
                "labels": {
                    "app": config.name
                }
            },
            "spec": {
                "selector": {
                    "app": config.name
                },
                "ports": [
                    {
                        "port": port,
                        "targetPort": port,
                        "protocol": "TCP"
                    } for port in (config.ports or [8080])
                ],
                "type": "LoadBalancer"
            }
        }

class DeploymentManager:
    """Main deployment manager."""
    
    def __init__(self):
        self.docker_manager = DockerManager()
        self.k8s_manager = KubernetesManager()
    
    def deploy(self, config: DeploymentConfig) -> bool:
        """Deploy application based on configuration."""
        console.print(f"[bold blue]Deploying {config.name} to {config.deployment_type.value}[/bold blue]")
        
        if config.deployment_type == DeploymentType.DOCKER:
            return self._deploy_docker(config)
        elif config.deployment_type == DeploymentType.KUBERNETES:
            return self._deploy_kubernetes(config)
        elif config.deployment_type == DeploymentType.LOCAL:
            return self._deploy_local(config)
        else:
            console.print(f"[red]Unsupported deployment type: {config.deployment_type.value}[/red]")
            return False
    
    def _deploy_docker(self, config: DeploymentConfig) -> bool:
        """Deploy using Docker."""
        # Build image
        if not self.docker_manager.build_image(
            dockerfile_path="Dockerfile",
            image_name=config.image_name,
            image_tag=config.image_tag
        ):
            return False
        
        # Run container
        container_id = self.docker_manager.run_container(
            image_name=config.image_name,
            image_tag=config.image_tag,
            container_name=config.name,
            ports={str(port): str(port) for port in (config.ports or [8080])},
            environment_vars=config.environment_vars
        )
        
        return container_id is not None
    
    def _deploy_kubernetes(self, config: DeploymentConfig) -> bool:
        """Deploy using Kubernetes."""
        # Create deployment
        if not self.k8s_manager.create_deployment(config):
            return False
        
        # Create service
        if not self.k8s_manager.create_service(config):
            return False
        
        return True
    
    def _deploy_local(self, config: DeploymentConfig) -> bool:
        """Deploy locally."""
        console.print("[blue]Deploying locally...[/blue]")
        
        # Set environment variables
        if config.environment_vars:
            for key, value in config.environment_vars.items():
                os.environ[key] = value
        
        # Run the application
        try:
            cmd = ["python", "run_training.py"]
            if config.environment == Environment.PRODUCTION:
                cmd.extend(["--config", "config/production.yaml"])
            elif config.environment == Environment.STAGING:
                cmd.extend(["--config", "config/staging.yaml"])
            else:
                cmd.extend(["--config", "config/development.yaml"])
            
            subprocess.run(cmd, check=True)
            console.print("[green]Local deployment successful[/green]")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Local deployment failed: {e}[/red]")
            return False

class CIManager:
    """CI/CD pipeline manager."""
    
    def __init__(self, config: CIConfig):
        self.config = config
    
    def create_github_workflow(self, output_path: str = ".github/workflows/ci.yml"):
        """Create GitHub Actions workflow."""
        workflow_content = f"""
name: {self.config.pipeline_name}

on:
  push:
    branches: {self.config.trigger_branches}
  pull_request:
    branches: {self.config.trigger_branches}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-html
    
    - name: Run tests
      run: |
        python test_framework.py --run-tests --coverage --parallel
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./test_results/coverage.xml
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: test_results/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: {self.config.artifact_registry}
        username: ${{{{ secrets.REGISTRY_USERNAME }}}}
        password: ${{{{ secrets.REGISTRY_PASSWORD }}}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          {self.config.artifact_registry}/frontier-model:${{{{ github.sha }}}}
          {self.config.artifact_registry}/frontier-model:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add staging deployment commands here

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add production deployment commands here
"""
        
        workflow_path = Path(output_path)
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(workflow_path, 'w') as f:
            f.write(workflow_content)
        
        console.print(f"[green]GitHub workflow created: {output_path}[/green]")
    
    def create_gitlab_ci(self, output_path: str = ".gitlab-ci.yml"):
        """Create GitLab CI configuration."""
        ci_content = f"""
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.9
  before_script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov pytest-html
  script:
    - python test_framework.py --run-tests --coverage --parallel
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: test_results/coverage.xml
    paths:
      - test_results/
    expire_in: 1 week

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker build -t $CI_REGISTRY_IMAGE:latest .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main
    - develop

deploy-staging:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to staging"
    # Add staging deployment commands
  only:
    - develop

deploy-production:
  stage: deploy
  image: alpine:latest
  script:
    - echo "Deploying to production"
    # Add production deployment commands
  only:
    - main
"""
        
        with open(output_path, 'w') as f:
            f.write(ci_content)
        
        console.print(f"[green]GitLab CI configuration created: {output_path}[/green]")
    
    def create_jenkins_pipeline(self, output_path: str = "Jenkinsfile"):
        """Create Jenkins pipeline."""
        pipeline_content = f"""
pipeline {{
    agent any
    
    environment {{
        DOCKER_REGISTRY = '{self.config.artifact_registry}'
        IMAGE_NAME = 'frontier-model'
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Test') {{
            steps {{
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-cov pytest-html'
                sh 'python test_framework.py --run-tests --coverage --parallel'
            }}
            post {{
                always {{
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'test_results',
                        reportFiles: 'test_report.html',
                        reportName: 'Test Report'
                    ])
                    publishCoverage adapters: [
                        coberturaAdapter('test_results/coverage.xml')
                    ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }}
            }}
        }}
        
        stage('Build') {{
            steps {{
                script {{
                    def image = docker.build("$DOCKER_REGISTRY/$IMAGE_NAME:$BUILD_NUMBER")
                    docker.withRegistry('https://$DOCKER_REGISTRY', 'docker-registry-credentials') {{
                        image.push()
                        image.push('latest')
                    }}
                }}
            }}
        }}
        
        stage('Deploy Staging') {{
            when {{
                branch 'develop'
            }}
            steps {{
                echo 'Deploying to staging environment'
                // Add staging deployment commands
            }}
        }}
        
        stage('Deploy Production') {{
            when {{
                branch 'main'
            }}
            steps {{
                echo 'Deploying to production environment'
                // Add production deployment commands
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        failure {{
            emailext (
                subject: "Build Failed: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                body: "Build failed. Please check the console output.",
                to: "{', '.join(self.config.notification_channels)}"
            )
        }}
    }}
}}
"""
        
        with open(output_path, 'w') as f:
            f.write(pipeline_content)
        
        console.print(f"[green]Jenkins pipeline created: {output_path}[/green]")

def create_dockerfile(output_path: str = "Dockerfile"):
    """Create Dockerfile for Frontier Model."""
    dockerfile_content = """
# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 frontier && chown -R frontier:frontier /app
USER frontier

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "run_training.py"]
"""
    
    with open(output_path, 'w') as f:
        f.write(dockerfile_content)
    
    console.print(f"[green]Dockerfile created: {output_path}[/green]")

def create_docker_compose(output_path: str = "docker-compose.yml"):
    """Create Docker Compose configuration."""
    compose_content = """
version: '3.8'

services:
  frontier-model:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  frontier-model-gpu:
    build: .
    ports:
      - "8081:8080"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  grafana-storage:
"""
    
    with open(output_path, 'w') as f:
        f.write(compose_content)
    
    console.print(f"[green]Docker Compose configuration created: {output_path}[/green]")

def create_requirements_txt(output_path: str = "requirements.txt"):
    """Create requirements.txt file."""
    requirements_content = """
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
trl>=0.7.0

# Configuration and utilities
pyyaml>=6.0
tyro>=0.5.0
rich>=13.0.0
loguru>=0.7.0
psutil>=5.9.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-html>=3.1.0
pytest-xdist>=3.0.0
coverage>=7.0.0

# Monitoring and logging
wandb>=0.15.0
mlflow>=2.5.0
tensorboard>=2.13.0
sentry-sdk>=1.30.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Deployment
docker>=6.1.0
kubernetes>=27.0.0
requests>=2.31.0

# Optional GPU support
# nvidia-ml-py3>=7.352.0
# GPUtil>=1.4.0
"""
    
    with open(output_path, 'w') as f:
        f.write(requirements_content)
    
    console.print(f"[green]Requirements file created: {output_path}[/green]")

def main():
    """Main function for deployment CLI."""
    parser = argparse.ArgumentParser(description="Frontier Model Deployment Manager")
    parser.add_argument("--deploy", action="store_true", help="Deploy application")
    parser.add_argument("--config", type=str, help="Deployment configuration file")
    parser.add_argument("--type", type=str, choices=["local", "docker", "kubernetes"], 
                       default="docker", help="Deployment type")
    parser.add_argument("--environment", type=str, choices=["development", "staging", "production"],
                       default="development", help="Deployment environment")
    parser.add_argument("--create-ci", action="store_true", help="Create CI/CD configuration")
    parser.add_argument("--ci-type", type=str, choices=["github", "gitlab", "jenkins"],
                       default="github", help="CI/CD platform")
    parser.add_argument("--create-docker", action="store_true", help="Create Docker configuration")
    
    args = parser.parse_args()
    
    if args.create_docker:
        console.print("[bold blue]Creating Docker configuration...[/bold blue]")
        create_dockerfile()
        create_docker_compose()
        create_requirements_txt()
        console.print("[green]Docker configuration created successfully[/green]")
    
    if args.create_ci:
        console.print(f"[bold blue]Creating {args.ci_type} CI/CD configuration...[/bold blue]")
        
        ci_config = CIConfig(
            pipeline_name="frontier-model-ci",
            trigger_branches=["main", "develop"],
            build_stages=["test", "build"],
            test_stages=["unit", "integration", "performance"],
            deploy_stages=["staging", "production"],
            notification_channels=["admin@example.com"],
            artifact_registry="your-registry.com",
            deployment_targets=["staging", "production"]
        )
        
        ci_manager = CIManager(ci_config)
        
        if args.ci_type == "github":
            ci_manager.create_github_workflow()
        elif args.ci_type == "gitlab":
            ci_manager.create_gitlab_ci()
        elif args.ci_type == "jenkins":
            ci_manager.create_jenkins_pipeline()
        
        console.print(f"[green]{args.ci_type} CI/CD configuration created successfully[/green]")
    
    if args.deploy:
        if not args.config:
            console.print("[red]Deployment configuration file required[/red]")
            return 1
        
        # Load deployment configuration
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        deployment_config = DeploymentConfig(**config_data)
        
        # Override with command line arguments
        if args.type:
            deployment_config.deployment_type = DeploymentType(args.type)
        if args.environment:
            deployment_config.environment = Environment(args.environment)
        
        # Deploy
        manager = DeploymentManager()
        success = manager.deploy(deployment_config)
        
        if success:
            console.print("[green]Deployment successful[/green]")
            return 0
        else:
            console.print("[red]Deployment failed[/red]")
            return 1

if __name__ == "__main__":
    exit(main())
