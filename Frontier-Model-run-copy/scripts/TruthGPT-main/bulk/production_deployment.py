#!/usr/bin/env python3
"""
Production Deployment - Production deployment scripts and configuration
Handles Docker, Kubernetes, and cloud deployment configurations
"""

import os
import json
import yaml
import subprocess
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    # Application settings
    app_name: str = "bulk-optimization"
    app_version: str = "1.0.0"
    app_port: int = 8000
    
    # Environment settings
    environment: str = "production"
    debug: bool = False
    
    # Resource settings
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    cpu_request: str = "1"
    memory_request: str = "2Gi"
    
    # Scaling settings
    replicas: int = 3
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Database settings
    database_host: str = "postgres"
    database_port: int = 5432
    database_name: str = "bulk_optimization"
    database_user: str = "bulk_user"
    
    # Redis settings
    redis_host: str = "redis"
    redis_port: int = 6379
    
    # Monitoring settings
    enable_monitoring: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Security settings
    enable_ssl: bool = True
    ssl_cert_path: str = "/etc/ssl/certs/bulk-optimization.crt"
    ssl_key_path: str = "/etc/ssl/private/bulk-optimization.key"
    
    # Storage settings
    storage_class: str = "fast-ssd"
    storage_size: str = "100Gi"
    storage_mount_path: str = "/var/lib/bulk-optimization"

class DockerDeployment:
    """Docker deployment configuration."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_dockerfile(self, output_path: str = "Dockerfile"):
        """Create Dockerfile."""
        dockerfile_content = f"""# Multi-stage build for production
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Copy application code
COPY . .

# Set ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE {self.config.app_port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.config.app_port}/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "production_api:app", "--host", "0.0.0.0", "--port", "{self.config.app_port}"]
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"Dockerfile created at {output_path}")
    
    def create_docker_compose(self, output_path: str = "docker-compose.yml"):
        """Create docker-compose.yml."""
        compose_content = f"""version: '3.8'

services:
  app:
    build: .
    ports:
      - "{self.config.app_port}:{self.config.app_port}"
    environment:
      - ENVIRONMENT={self.config.environment}
      - DEBUG={str(self.config.debug).lower()}
      - DB_HOST={self.config.database_host}
      - DB_PORT={self.config.database_port}
      - DB_NAME={self.config.database_name}
      - DB_USER={self.config.database_user}
      - REDIS_HOST={self.config.redis_host}
      - REDIS_PORT={self.config.redis_port}
    volumes:
      - ./data:/var/lib/bulk-optimization
      - ./logs:/var/log/bulk-optimization
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '{self.config.cpu_limit}'
          memory: {self.config.memory_limit}
        reservations:
          cpus: '{self.config.cpu_request}'
          memory: {self.config.memory_request}

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB={self.config.database_name}
      - POSTGRES_USER={self.config.database_user}
      - POSTGRES_PASSWORD=bulk_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
"""
        
        with open(output_path, 'w') as f:
            f.write(compose_content)
        
        self.logger.info(f"Docker Compose file created at {output_path}")
    
    def create_nginx_config(self, output_path: str = "nginx.conf"):
        """Create Nginx configuration."""
        nginx_content = f"""events {{
    worker_connections 1024;
}}

http {{
    upstream app {{
        server app:{self.config.app_port};
    }}
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {{
        listen 80;
        server_name _;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }}
    
    server {{
        listen 443 ssl http2;
        server_name _;
        
        # SSL configuration
        ssl_certificate /etc/ssl/bulk-optimization.crt;
        ssl_certificate_key /etc/ssl/bulk-optimization.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # API endpoints
        location / {{
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }}
        
        # Health check endpoint
        location /health {{
            proxy_pass http://app;
            access_log off;
        }}
    }}
}}
"""
        
        with open(output_path, 'w') as f:
            f.write(nginx_content)
        
        self.logger.info(f"Nginx configuration created at {output_path}")

class KubernetesDeployment:
    """Kubernetes deployment configuration."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_namespace(self, output_path: str = "namespace.yaml"):
        """Create Kubernetes namespace."""
        namespace_content = f"""apiVersion: v1
kind: Namespace
metadata:
  name: {self.config.app_name}
  labels:
    name: {self.config.app_name}
"""
        
        with open(output_path, 'w') as f:
            f.write(namespace_content)
        
        self.logger.info(f"Namespace created at {output_path}")
    
    def create_configmap(self, output_path: str = "configmap.yaml"):
        """Create Kubernetes ConfigMap."""
        configmap_content = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {self.config.app_name}-config
  namespace: {self.config.app_name}
data:
  ENVIRONMENT: "{self.config.environment}"
  DEBUG: "{str(self.config.debug).lower()}"
  DB_HOST: "{self.config.database_host}"
  DB_PORT: "{self.config.database_port}"
  DB_NAME: "{self.config.database_name}"
  DB_USER: "{self.config.database_user}"
  REDIS_HOST: "{self.config.redis_host}"
  REDIS_PORT: "{self.config.redis_port}"
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "4"
  MAX_MEMORY_GB: "16"
"""
        
        with open(output_path, 'w') as f:
            f.write(configmap_content)
        
        self.logger.info(f"ConfigMap created at {output_path}")
    
    def create_secret(self, output_path: str = "secret.yaml"):
        """Create Kubernetes Secret."""
        secret_content = f"""apiVersion: v1
kind: Secret
metadata:
  name: {self.config.app_name}-secret
  namespace: {self.config.app_name}
type: Opaque
data:
  DB_PASSWORD: YnVsa19wYXNzd29yZA==  # bulk_password
  JWT_SECRET: c2VjcmV0X2tleV9mb3Jfand0  # secret_key_for_jwt
  SECRET_KEY: c2VjcmV0X2tleV9mb3JfYXBw  # secret_key_for_app
"""
        
        with open(output_path, 'w') as f:
            f.write(secret_content)
        
        self.logger.info(f"Secret created at {output_path}")
    
    def create_deployment(self, output_path: str = "deployment.yaml"):
        """Create Kubernetes Deployment."""
        deployment_content = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {self.config.app_name}
  namespace: {self.config.app_name}
  labels:
    app: {self.config.app_name}
spec:
  replicas: {self.config.replicas}
  selector:
    matchLabels:
      app: {self.config.app_name}
  template:
    metadata:
      labels:
        app: {self.config.app_name}
    spec:
      containers:
      - name: {self.config.app_name}
        image: {self.config.app_name}:{self.config.app_version}
        ports:
        - containerPort: {self.config.app_port}
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        envFrom:
        - configMapRef:
            name: {self.config.app_name}-config
        - secretRef:
            name: {self.config.app_name}-secret
        resources:
          limits:
            cpu: {self.config.cpu_limit}
            memory: {self.config.memory_limit}
          requests:
            cpu: {self.config.cpu_request}
            memory: {self.config.memory_request}
        livenessProbe:
          httpGet:
            path: /health
            port: {self.config.app_port}
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: {self.config.app_port}
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: data-volume
          mountPath: /var/lib/bulk-optimization
        - name: logs-volume
          mountPath: /var/log/bulk-optimization
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: {self.config.app_name}-data-pvc
      - name: logs-volume
        emptyDir: {{}}
"""
        
        with open(output_path, 'w') as f:
            f.write(deployment_content)
        
        self.logger.info(f"Deployment created at {output_path}")
    
    def create_service(self, output_path: str = "service.yaml"):
        """Create Kubernetes Service."""
        service_content = f"""apiVersion: v1
kind: Service
metadata:
  name: {self.config.app_name}-service
  namespace: {self.config.app_name}
  labels:
    app: {self.config.app_name}
spec:
  selector:
    app: {self.config.app_name}
  ports:
  - name: http
    port: 80
    targetPort: {self.config.app_port}
    protocol: TCP
  type: ClusterIP
"""
        
        with open(output_path, 'w') as f:
            f.write(service_content)
        
        self.logger.info(f"Service created at {output_path}")
    
    def create_hpa(self, output_path: str = "hpa.yaml"):
        """Create Horizontal Pod Autoscaler."""
        hpa_content = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {self.config.app_name}-hpa
  namespace: {self.config.app_name}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {self.config.app_name}
  minReplicas: {self.config.min_replicas}
  maxReplicas: {self.config.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        
        with open(output_path, 'w') as f:
            f.write(hpa_content)
        
        self.logger.info(f"HPA created at {output_path}")
    
    def create_pvc(self, output_path: str = "pvc.yaml"):
        """Create Persistent Volume Claim."""
        pvc_content = f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {self.config.app_name}-data-pvc
  namespace: {self.config.app_name}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {self.config.storage_size}
  storageClassName: {self.config.storage_class}
"""
        
        with open(output_path, 'w') as f:
            f.write(pvc_content)
        
        self.logger.info(f"PVC created at {output_path}")
    
    def create_ingress(self, output_path: str = "ingress.yaml"):
        """Create Kubernetes Ingress."""
        ingress_content = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {self.config.app_name}-ingress
  namespace: {self.config.app_name}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - {self.config.app_name}.example.com
    secretName: {self.config.app_name}-tls
  rules:
  - host: {self.config.app_name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {self.config.app_name}-service
            port:
              number: 80
"""
        
        with open(output_path, 'w') as f:
            f.write(ingress_content)
        
        self.logger.info(f"Ingress created at {output_path}")

class ProductionDeployment:
    """Production deployment manager."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.docker = DockerDeployment(config)
        self.k8s = KubernetesDeployment(config)
    
    def create_docker_deployment(self, output_dir: str = "docker"):
        """Create Docker deployment files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create Docker files
        self.docker.create_dockerfile(output_path / "Dockerfile")
        self.docker.create_docker_compose(output_path / "docker-compose.yml")
        self.docker.create_nginx_config(output_path / "nginx.conf")
        
        # Create requirements.txt
        self._create_requirements_txt(output_path / "requirements.txt")
        
        # Create .dockerignore
        self._create_dockerignore(output_path / ".dockerignore")
        
        self.logger.info(f"Docker deployment files created in {output_dir}")
    
    def create_kubernetes_deployment(self, output_dir: str = "kubernetes"):
        """Create Kubernetes deployment files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create K8s files
        self.k8s.create_namespace(output_path / "namespace.yaml")
        self.k8s.create_configmap(output_path / "configmap.yaml")
        self.k8s.create_secret(output_path / "secret.yaml")
        self.k8s.create_deployment(output_path / "deployment.yaml")
        self.k8s.create_service(output_path / "service.yaml")
        self.k8s.create_hpa(output_path / "hpa.yaml")
        self.k8s.create_pvc(output_path / "pvc.yaml")
        self.k8s.create_ingress(output_path / "ingress.yaml")
        
        # Create deployment script
        self._create_k8s_deploy_script(output_path / "deploy.sh")
        
        self.logger.info(f"Kubernetes deployment files created in {output_dir}")
    
    def _create_requirements_txt(self, output_path: str):
        """Create requirements.txt."""
        requirements = """# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9

# Redis
redis==5.0.1
aioredis==2.0.1

# Monitoring
prometheus-client==0.19.0
psutil==5.9.6

# Machine Learning
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
scikit-learn==1.3.2

# Data processing
pandas==2.1.4
h5py==3.10.0
zarr==2.16.1

# Configuration
pyyaml==6.0.1
python-dotenv==1.0.0

# Security
cryptography==41.0.8
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# HTTP client
httpx==0.25.2
aiofiles==23.2.1

# Logging
structlog==23.2.0
"""
        
        with open(output_path, 'w') as f:
            f.write(requirements)
    
    def _create_dockerignore(self, output_path: str):
        """Create .dockerignore."""
        dockerignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Documentation
*.md
docs/

# Tests
tests/
test_*.py
*_test.py

# Logs
*.log
logs/

# Data
data/
*.csv
*.json
*.pkl
*.h5
*.zarr

# Temporary files
tmp/
temp/
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerignore)
    
    def _create_k8s_deploy_script(self, output_path: str):
        """Create Kubernetes deployment script."""
        script_content = f"""#!/bin/bash

# Kubernetes deployment script for {self.config.app_name}
set -e

NAMESPACE="{self.config.app_name}"
APP_NAME="{self.config.app_name}"

echo "🚀 Deploying {self.config.app_name} to Kubernetes"

# Create namespace
kubectl apply -f namespace.yaml

# Create ConfigMap
kubectl apply -f configmap.yaml

# Create Secret
kubectl apply -f secret.yaml

# Create PVC
kubectl apply -f pvc.yaml

# Create Deployment
kubectl apply -f deployment.yaml

# Create Service
kubectl apply -f service.yaml

# Create HPA
kubectl apply -f hpa.yaml

# Create Ingress
kubectl apply -f ingress.yaml

echo "✅ Deployment completed"

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/$APP_NAME -n $NAMESPACE

# Get deployment status
echo "📊 Deployment status:"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

echo "🎉 Deployment successful!"
"""
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_path, 0o755)

def create_production_deployment(config: Optional[Dict[str, Any]] = None) -> ProductionDeployment:
    """Create production deployment."""
    if config is None:
        config = {}
    
    deployment_config = DeploymentConfig(**config)
    return ProductionDeployment(deployment_config)

if __name__ == "__main__":
    # Example usage
    config = {
        "app_name": "bulk-optimization",
        "app_version": "1.0.0",
        "environment": "production",
        "replicas": 3,
        "cpu_limit": "2",
        "memory_limit": "4Gi"
    }
    
    deployment = create_production_deployment(config)
    
    # Create Docker deployment
    deployment.create_docker_deployment("docker")
    print("✅ Docker deployment files created")
    
    # Create Kubernetes deployment
    deployment.create_kubernetes_deployment("kubernetes")
    print("✅ Kubernetes deployment files created")
    
    print("🎉 Production deployment configuration completed!")

