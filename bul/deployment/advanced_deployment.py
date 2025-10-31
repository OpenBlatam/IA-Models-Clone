"""
Advanced Deployment Configuration for BUL API
============================================

Production-ready deployment configuration with:
- Docker multi-stage builds
- Kubernetes manifests
- CI/CD pipelines
- Monitoring and observability
- Security hardening
- Performance optimization
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# Docker Configuration
DOCKERFILE_CONTENT = """
# Multi-stage Docker build for BUL API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements-optimized.txt .
RUN pip install --no-cache-dir -r requirements-optimized.txt

# Copy application code
COPY . .

# Change ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""

# Docker Compose Configuration
DOCKER_COMPOSE_CONTENT = """
version: '3.8'

services:
  bul-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://bul_user:bul_password@postgres:5432/bul_db
      - REDIS_URL=redis://redis:6379/0
      - SECURITY_SECRET_KEY=${SECURITY_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=bul_db
      - POSTGRES_USER=bul_user
      - POSTGRES_PASSWORD=bul_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./deployment/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bul_user -d bul_db"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx.conf:/etc/nginx/nginx.conf
      - ./deployment/ssl:/etc/nginx/ssl
    depends_on:
      - bul-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./deployment/grafana/dashboards:/var/lib/grafana/dashboards
      - ./deployment/grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
"""

# Kubernetes Manifests
KUBERNETES_NAMESPACE = """
apiVersion: v1
kind: Namespace
metadata:
  name: bul-api
  labels:
    name: bul-api
"""

KUBERNETES_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bul-api
  namespace: bul-api
  labels:
    app: bul-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bul-api
  template:
    metadata:
      labels:
        app: bul-api
    spec:
      containers:
      - name: bul-api
        image: bul-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: bul-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: bul-secrets
              key: redis-url
        - name: SECURITY_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: bul-secrets
              key: secret-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: config-volume
        configMap:
          name: bul-config
"""

KUBERNETES_SERVICE = """
apiVersion: v1
kind: Service
metadata:
  name: bul-api-service
  namespace: bul-api
spec:
  selector:
    app: bul-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""

KUBERNETES_INGRESS = """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bul-api-ingress
  namespace: bul-api
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.bul.com
    secretName: bul-tls
  rules:
  - host: api.bul.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bul-api-service
            port:
              number: 80
"""

# Nginx Configuration
NGINX_CONFIG = """
events {
    worker_connections 1024;
}

http {
    upstream bul_api {
        server bul-api:8000;
    }

    server {
        listen 80;
        server_name api.bul.com;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
        limit_req zone=api burst=20 nodelay;

        # Proxy settings
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        location / {
            proxy_pass http://bul_api;
        }

        location /health {
            proxy_pass http://bul_api/health;
            access_log off;
        }

        location /metrics {
            proxy_pass http://bul_api/metrics;
            allow 10.0.0.0/8;
            deny all;
        }
    }
}
"""

# Prometheus Configuration
PROMETHEUS_CONFIG = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "bul_rules.yml"

scrape_configs:
  - job_name: 'bul-api'
    static_configs:
      - targets: ['bul-api:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
"""

# CI/CD Pipeline Configuration
GITHUB_ACTIONS_WORKFLOW = """
name: BUL API CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-optimized.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=bul --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          deployment/kubernetes/namespace.yaml
          deployment/kubernetes/deployment.yaml
          deployment/kubernetes/service.yaml
          deployment/kubernetes/ingress.yaml
        kubeconfig: ${{ secrets.KUBECONFIG }}
        namespace: bul-api
"""

# Advanced Deployment Configuration
@dataclass
class DeploymentConfig:
    """Advanced deployment configuration"""
    
    # Environment settings
    environment: str = "production"
    debug: bool = False
    
    # Application settings
    app_name: str = "bul-api"
    version: str = "3.0.0"
    port: int = 8000
    workers: int = 4
    
    # Database settings
    database_url: str = "postgresql://bul_user:bul_password@localhost:5432/bul_db"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    
    # Security settings
    secret_key: str = "your-secret-key-here"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Monitoring settings
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    
    # Performance settings
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30
    keepalive_timeout: int = 5
    
    # Scaling settings
    min_replicas: int = 3
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "app_name": self.app_name,
            "version": self.version,
            "port": self.port,
            "workers": self.workers,
            "database_url": self.database_url,
            "database_pool_size": self.database_pool_size,
            "database_max_overflow": self.database_max_overflow,
            "redis_url": self.redis_url,
            "redis_max_connections": self.redis_max_connections,
            "secret_key": "***",  # Hide sensitive data
            "jwt_algorithm": self.jwt_algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            "enable_logging": self.enable_logging,
            "max_request_size": self.max_request_size,
            "request_timeout": self.request_timeout,
            "keepalive_timeout": self.keepalive_timeout,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_utilization": self.target_cpu_utilization,
            "target_memory_utilization": self.target_memory_utilization
        }

# Deployment Factory
class DeploymentFactory:
    """Factory for creating deployment configurations"""
    
    @staticmethod
    def create_docker_deployment(config: DeploymentConfig) -> str:
        """Create Docker deployment configuration"""
        return DOCKERFILE_CONTENT
    
    @staticmethod
    def create_docker_compose_deployment(config: DeploymentConfig) -> str:
        """Create Docker Compose deployment configuration"""
        return DOCKER_COMPOSE_CONTENT
    
    @staticmethod
    def create_kubernetes_deployment(config: DeploymentConfig) -> Dict[str, str]:
        """Create Kubernetes deployment configuration"""
        return {
            "namespace.yaml": KUBERNETES_NAMESPACE,
            "deployment.yaml": KUBERNETES_DEPLOYMENT,
            "service.yaml": KUBERNETES_SERVICE,
            "ingress.yaml": KUBERNETES_INGRESS
        }
    
    @staticmethod
    def create_nginx_config(config: DeploymentConfig) -> str:
        """Create Nginx configuration"""
        return NGINX_CONFIG
    
    @staticmethod
    def create_prometheus_config(config: DeploymentConfig) -> str:
        """Create Prometheus configuration"""
        return PROMETHEUS_CONFIG
    
    @staticmethod
    def create_github_actions_workflow(config: DeploymentConfig) -> str:
        """Create GitHub Actions workflow"""
        return GITHUB_ACTIONS_WORKFLOW

# Deployment Manager
class DeploymentManager:
    """Advanced deployment manager"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.factory = DeploymentFactory()
    
    def create_all_deployments(self, output_dir: str = "deployment") -> None:
        """Create all deployment configurations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create Docker files
        dockerfile_path = output_path / "Dockerfile"
        dockerfile_path.write_text(self.factory.create_docker_deployment(self.config))
        
        docker_compose_path = output_path / "docker-compose.yml"
        docker_compose_path.write_text(self.factory.create_docker_compose_deployment(self.config))
        
        # Create Kubernetes manifests
        k8s_path = output_path / "kubernetes"
        k8s_path.mkdir(exist_ok=True)
        
        k8s_configs = self.factory.create_kubernetes_deployment(self.config)
        for filename, content in k8s_configs.items():
            (k8s_path / filename).write_text(content)
        
        # Create Nginx configuration
        nginx_path = output_path / "nginx.conf"
        nginx_path.write_text(self.factory.create_nginx_config(self.config))
        
        # Create Prometheus configuration
        prometheus_path = output_path / "prometheus.yml"
        prometheus_path.write_text(self.factory.create_prometheus_config(self.config))
        
        # Create GitHub Actions workflow
        github_path = Path(".github/workflows")
        github_path.mkdir(parents=True, exist_ok=True)
        ci_cd_path = github_path / "ci-cd.yml"
        ci_cd_path.write_text(self.factory.create_github_actions_workflow(self.config))
        
        # Create configuration files
        config_path = output_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        
        print(f"Deployment configurations created in {output_dir}/")
        print("Files created:")
        print(f"  - {dockerfile_path}")
        print(f"  - {docker_compose_path}")
        print(f"  - {k8s_path}/")
        print(f"  - {nginx_path}")
        print(f"  - {prometheus_path}")
        print(f"  - {ci_cd_path}")
        print(f"  - {config_path}")

# Export all deployment components
__all__ = [
    "DeploymentConfig",
    "DeploymentFactory", 
    "DeploymentManager",
    "DOCKERFILE_CONTENT",
    "DOCKER_COMPOSE_CONTENT",
    "KUBERNETES_NAMESPACE",
    "KUBERNETES_DEPLOYMENT",
    "KUBERNETES_SERVICE",
    "KUBERNETES_INGRESS",
    "NGINX_CONFIG",
    "PROMETHEUS_CONFIG",
    "GITHUB_ACTIONS_WORKFLOW"
]












