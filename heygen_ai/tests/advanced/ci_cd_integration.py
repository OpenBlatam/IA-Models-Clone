"""
Advanced CI/CD integration for HeyGen AI test system.
Comprehensive pipeline automation with GitHub Actions, Docker, and cloud integration.
"""

import os
import json
import yaml
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

@dataclass
class CIConfig:
    """CI/CD configuration."""
    project_name: str = "heygen-ai"
    python_version: str = "3.11"
    test_command: str = "python -m pytest tests/ -v"
    coverage_command: str = "python -m pytest tests/ --cov=. --cov-report=xml"
    lint_command: str = "python -m flake8 ."
    format_command: str = "python -m black ."
    docker_image: str = "python:3.11-slim"
    dockerfile_path: str = "Dockerfile"
    registry_url: str = "ghcr.io"
    environment: str = "production"
    secrets: Dict[str, str] = field(default_factory=dict)

class GitHubActionsGenerator:
    """Generates GitHub Actions workflows for CI/CD."""
    
    def __init__(self, config: CIConfig):
        self.config = config
        self.workflows_dir = Path(".github/workflows")
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_test_workflow(self) -> str:
        """Generate comprehensive test workflow."""
        workflow = {
            "name": "Test Suite",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]},
                "schedule": [{"cron": "0 2 * * *"}]  # Daily at 2 AM
            },
            "env": {
                "PYTHON_VERSION": self.config.python_version,
                "PROJECT_NAME": self.config.project_name
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "matrix": {
                            "python-version": ["3.9", "3.10", "3.11"],
                            "test-type": ["unit", "integration", "performance"]
                        }
                    },
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Python ${{ matrix.python-version }}",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "${{ matrix.python-version }}"
                            }
                        },
                        {
                            "name": "Cache dependencies",
                            "uses": "actions/cache@v3",
                            "with": {
                                "path": "~/.cache/pip",
                                "key": "${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run linting",
                            "run": self.config.lint_command
                        },
                        {
                            "name": "Run tests",
                            "run": f"{self.config.test_command} --type=${{ matrix.test-type }}"
                        },
                        {
                            "name": "Generate coverage report",
                            "run": self.config.coverage_command
                        },
                        {
                            "name": "Upload coverage to Codecov",
                            "uses": "codecov/codecov-action@v3",
                            "with": {
                                "file": "./coverage.xml"
                            }
                        }
                    ]
                },
                "security-scan": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Run security scan",
                            "uses": "securecodewarrior/github-action-add-sarif@v1",
                            "with": {
                                "sarif-file": "security-scan-results.sarif"
                            }
                        }
                    ]
                },
                "performance-test": {
                    "runs-on": "ubuntu-latest",
                    "if": "github.event_name == 'schedule'",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": self.config.python_version
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run performance tests",
                            "run": "python tests/advanced/benchmark_suite.py"
                        },
                        {
                            "name": "Upload performance results",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "performance-results",
                                "path": "benchmark_reports/"
                            }
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    def generate_deploy_workflow(self) -> str:
        """Generate deployment workflow."""
        workflow = {
            "name": "Deploy",
            "on": {
                "push": {"branches": ["main"]},
                "workflow_dispatch": {
                    "inputs": {
                        "environment": {
                            "description": "Environment to deploy to",
                            "required": True,
                            "default": "staging",
                            "type": "choice",
                            "options": ["staging", "production"]
                        }
                    }
                }
            },
            "env": {
                "REGISTRY": self.config.registry_url,
                "IMAGE_NAME": f"{self.config.registry_url}/{self.config.project_name}"
            },
            "jobs": {
                "build-and-push": {
                    "runs-on": "ubuntu-latest",
                    "permissions": {
                        "contents": "read",
                        "packages": "write"
                    },
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v3"
                        },
                        {
                            "name": "Log in to Container Registry",
                            "uses": "docker/login-action@v3",
                            "with": {
                                "registry": self.config.registry_url,
                                "username": "${{ github.actor }}",
                                "password": "${{ secrets.GITHUB_TOKEN }}"
                            }
                        },
                        {
                            "name": "Extract metadata",
                            "id": "meta",
                            "uses": "docker/metadata-action@v5",
                            "with": {
                                "images": "${{ env.IMAGE_NAME }}",
                                "tags": "type=ref,event=branch"
                            }
                        },
                        {
                            "name": "Build and push Docker image",
                            "uses": "docker/build-push-action@v5",
                            "with": {
                                "context": ".",
                                "file": self.config.dockerfile_path,
                                "push": True,
                                "tags": "${{ steps.meta.outputs.tags }}",
                                "labels": "${{ steps.meta.outputs.labels }}",
                                "cache-from": "type=gha",
                                "cache-to": "type=gha,mode=max"
                            }
                        }
                    ]
                },
                "deploy": {
                    "needs": "build-and-push",
                    "runs-on": "ubuntu-latest",
                    "environment": "${{ github.event.inputs.environment || 'staging' }}",
                    "steps": [
                        {
                            "name": "Deploy to environment",
                            "run": f"echo 'Deploying to ${{{{ github.event.inputs.environment || 'staging' }}}}'"
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    def generate_workflows(self):
        """Generate all CI/CD workflows."""
        # Test workflow
        test_workflow = self.generate_test_workflow()
        test_file = self.workflows_dir / "test.yml"
        with open(test_file, 'w') as f:
            f.write(test_workflow)
        print(f"✅ Generated test workflow: {test_file}")
        
        # Deploy workflow
        deploy_workflow = self.generate_deploy_workflow()
        deploy_file = self.workflows_dir / "deploy.yml"
        with open(deploy_file, 'w') as f:
            f.write(deploy_workflow)
        print(f"✅ Generated deploy workflow: {deploy_file}")

class DockerGenerator:
    """Generates Docker configuration for containerized testing."""
    
    def __init__(self, config: CIConfig):
        self.config = config
    
    def generate_dockerfile(self) -> str:
        """Generate optimized Dockerfile for testing."""
        dockerfile = f"""# Multi-stage Dockerfile for HeyGen AI Testing
FROM {self.config.docker_image} as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]
"""
        return dockerfile
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for local testing."""
        compose = f"""version: '3.8'

services:
  heygen-ai-tests:
    build:
      context: .
      dockerfile: {self.config.dockerfile_path}
    container_name: heygen-ai-test-runner
    environment:
      - PYTHONPATH=/app
      - TEST_ENVIRONMENT=docker
    volumes:
      - .:/app
      - ./test_reports:/app/test_reports
    command: python -m pytest tests/ -v --html=test_reports/report.html --self-contained-html
    
  heygen-ai-benchmark:
    build:
      context: .
      dockerfile: {self.config.dockerfile_path}
    container_name: heygen-ai-benchmark
    environment:
      - PYTHONPATH=/app
      - BENCHMARK_MODE=true
    volumes:
      - .:/app
      - ./benchmark_reports:/app/benchmark_reports
    command: python tests/advanced/benchmark_suite.py
    
  heygen-ai-monitoring:
    build:
      context: .
      dockerfile: {self.config.dockerfile_path}
    container_name: heygen-ai-monitoring
    environment:
      - PYTHONPATH=/app
      - MONITORING_MODE=true
    volumes:
      - .:/app
      - ./monitoring_data:/app/monitoring_data
    command: python tests/advanced/test_dashboard.py
    ports:
      - "8080:8080"

networks:
  default:
    name: heygen-ai-test-network
"""
        return compose
    
    def generate_docker_files(self):
        """Generate all Docker configuration files."""
        # Dockerfile
        dockerfile_content = self.generate_dockerfile()
        dockerfile_path = Path(self.config.dockerfile_path)
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        print(f"✅ Generated Dockerfile: {dockerfile_path}")
        
        # docker-compose.yml
        compose_content = self.generate_docker_compose()
        compose_path = Path("docker-compose.yml")
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        print(f"✅ Generated docker-compose.yml: {compose_path}")

class CloudIntegration:
    """Cloud integration for CI/CD pipelines."""
    
    def __init__(self, config: CIConfig):
        self.config = config
    
    def generate_aws_config(self) -> Dict[str, Any]:
        """Generate AWS configuration for cloud testing."""
        return {
            "version": "0.2",
            "phases": {
                "pre_build": {
                    "commands": [
                        "echo Logging in to Amazon ECR...",
                        "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"
                    ]
                },
                "build": {
                    "commands": [
                        "echo Build started on `date`",
                        "echo Building the Docker image...",
                        f"docker build -t {self.config.project_name}:$IMAGE_TAG .",
                        f"docker tag {self.config.project_name}:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/{self.config.project_name}:$IMAGE_TAG"
                    ]
                },
                "post_build": {
                    "commands": [
                        "echo Build completed on `date`",
                        "echo Pushing the Docker images...",
                        f"docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/{self.config.project_name}:$IMAGE_TAG",
                        "echo Writing image definitions file...",
                        f"printf '[{{\"name\":\"{self.config.project_name}\",\"imageUri\":\"%s\"}}]' $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/{self.config.project_name}:$IMAGE_TAG > imagedefinitions.json"
                    ]
                }
            },
            "artifacts": {
                "files": [
                    "imagedefinitions.json"
                ]
            }
        }
    
    def generate_azure_config(self) -> Dict[str, Any]:
        """Generate Azure DevOps configuration."""
        return {
            "trigger": ["main", "develop"],
            "pool": {
                "vmImage": "ubuntu-latest"
            },
            "variables": {
                "pythonVersion": self.config.python_version,
                "projectName": self.config.project_name
            },
            "stages": [
                {
                    "stage": "Test",
                    "jobs": [
                        {
                            "job": "RunTests",
                            "steps": [
                                {
                                    "task": "UsePythonVersion@0",
                                    "inputs": {
                                        "versionSpec": "$(pythonVersion)"
                                    }
                                },
                                {
                                    "task": "PipAuthenticate@1",
                                    "inputs": {
                                        "pythonDownloadServiceConnections": "true"
                                    }
                                },
                                {
                                    "script": "pip install -r requirements.txt",
                                    "displayName": "Install dependencies"
                                },
                                {
                                    "script": self.config.test_command,
                                    "displayName": "Run tests"
                                },
                                {
                                    "task": "PublishTestResults@2",
                                    "inputs": {
                                        "testResultsFormat": "Pytest",
                                        "testResultsFiles": "**/test-results.xml",
                                        "failTaskOnFailedTests": True
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    
    def generate_cloud_configs(self):
        """Generate cloud configuration files."""
        # AWS CodeBuild
        aws_config = self.generate_aws_config()
        aws_file = Path("buildspec.yml")
        with open(aws_file, 'w') as f:
            yaml.dump(aws_config, f, default_flow_style=False)
        print(f"✅ Generated AWS buildspec.yml: {aws_file}")
        
        # Azure DevOps
        azure_config = self.generate_azure_config()
        azure_file = Path("azure-pipelines.yml")
        with open(azure_file, 'w') as f:
            yaml.dump(azure_config, f, default_flow_style=False)
        print(f"✅ Generated Azure pipeline: {azure_file}")

class CICDManager:
    """Main CI/CD manager for orchestrating all integrations."""
    
    def __init__(self, config: Optional[CIConfig] = None):
        self.config = config or CIConfig()
        self.github_actions = GitHubActionsGenerator(self.config)
        self.docker = DockerGenerator(self.config)
        self.cloud = CloudIntegration(self.config)
    
    def setup_complete_cicd(self):
        """Set up complete CI/CD pipeline."""
        print("🚀 Setting up Complete CI/CD Pipeline")
        print("=" * 50)
        
        # Generate GitHub Actions workflows
        print("\n📋 Generating GitHub Actions workflows...")
        self.github_actions.generate_workflows()
        
        # Generate Docker configuration
        print("\n🐳 Generating Docker configuration...")
        self.docker.generate_docker_files()
        
        # Generate cloud configurations
        print("\n☁️  Generating cloud configurations...")
        self.cloud.generate_cloud_configs()
        
        # Generate additional configuration files
        self._generate_additional_configs()
        
        print("\n✅ Complete CI/CD pipeline setup completed!")
        print("\n📋 Next steps:")
        print("   1. Commit and push changes to trigger workflows")
        print("   2. Configure secrets in GitHub repository settings")
        print("   3. Set up cloud resources (AWS/Azure)")
        print("   4. Monitor pipeline execution in GitHub Actions")
    
    def _generate_additional_configs(self):
        """Generate additional configuration files."""
        # .gitignore updates
        gitignore_additions = """
# Test artifacts
test_reports/
benchmark_reports/
monitoring_data/
coverage.xml
.coverage
.pytest_cache/
.mypy_cache/

# Docker
.dockerignore

# CI/CD
buildspec.yml
azure-pipelines.yml
"""
        
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            with open(gitignore_path, 'a') as f:
                f.write(gitignore_additions)
        else:
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_additions)
        
        print(f"✅ Updated .gitignore")
        
        # .dockerignore
        dockerignore_content = """
# Git
.git
.gitignore

# Documentation
README.md
*.md

# CI/CD
.github/
buildspec.yml
azure-pipelines.yml

# Test artifacts
test_reports/
benchmark_reports/
monitoring_data/
.coverage
.pytest_cache/
.mypy_cache/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
"""
        
        dockerignore_path = Path(".dockerignore")
        with open(dockerignore_path, 'w') as f:
            f.write(dockerignore_content)
        print(f"✅ Generated .dockerignore")
        
        # requirements.txt for CI/CD
        requirements_content = """
# Core dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-html>=3.0.0
pytest-xdist>=3.0.0
pytest-benchmark>=4.0.0

# Code quality
flake8>=6.0.0
black>=23.0.0
mypy>=1.0.0

# Performance testing
psutil>=5.9.0
memory-profiler>=0.60.0

# Monitoring
matplotlib>=3.6.0
numpy>=1.24.0

# CI/CD
PyYAML>=6.0
"""
        
        requirements_path = Path("requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        print(f"✅ Generated requirements.txt")

def demo_cicd_setup():
    """Demonstrate CI/CD setup."""
    print("🔄 CI/CD Integration Demo")
    print("=" * 30)
    
    # Create configuration
    config = CIConfig(
        project_name="heygen-ai-tests",
        python_version="3.11",
        environment="production"
    )
    
    # Create CI/CD manager
    cicd_manager = CICDManager(config)
    
    # Set up complete CI/CD pipeline
    cicd_manager.setup_complete_cicd()
    
    return cicd_manager

if __name__ == "__main__":
    # Run demo
    demo_cicd_setup()
