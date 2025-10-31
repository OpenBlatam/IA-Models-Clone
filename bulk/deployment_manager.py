"""
BUL Deployment Manager
=====================

Advanced deployment management tool for the BUL system.
"""

import os
import sys
import json
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Advanced deployment management for BUL system."""
    
    def __init__(self):
        self.deployment_configs = {
            'development': {
                'host': 'localhost',
                'port': 8000,
                'debug': True,
                'workers': 1,
                'reload': True
            },
            'staging': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'workers': 2,
                'reload': False
            },
            'production': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'workers': 4,
                'reload': False
            }
        }
    
    def create_dockerfile(self, environment: str = 'production') -> bool:
        """Create Dockerfile for deployment."""
        print(f"🐳 Creating Dockerfile for {environment} environment...")
        
        dockerfile_content = f"""# BUL - Business Universal Language
# Optimized Dockerfile for {environment}

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_optimized.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_optimized.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p generated_documents logs

# Set environment variables
ENV PYTHONPATH=/app
ENV BUL_ENV={environment}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "start_optimized.py", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        try:
            with open('Dockerfile', 'w') as f:
                f.write(dockerfile_content)
            print("✅ Dockerfile created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating Dockerfile: {e}")
            return False
    
    def create_docker_compose(self, environment: str = 'production') -> bool:
        """Create docker-compose.yml for deployment."""
        print(f"🐳 Creating docker-compose.yml for {environment} environment...")
        
        config = self.deployment_configs.get(environment, self.deployment_configs['production'])
        
        compose_content = f"""version: '3.8'

services:
  bul-app:
    build: .
    container_name: bul-{environment}
    ports:
      - "{config['port']}:8000"
    environment:
      - BUL_ENV={environment}
      - BUL_DEBUG={str(config['debug']).lower()}
      - BUL_API_HOST=0.0.0.0
      - BUL_API_PORT=8000
    volumes:
      - ./generated_documents:/app/generated_documents
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add reverse proxy
  nginx:
    image: nginx:alpine
    container_name: bul-nginx-{environment}
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - bul-app
    restart: unless-stopped
    profiles:
      - proxy

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: bul-prometheus-{environment}
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
    profiles:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: bul-grafana-{environment}
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped
    profiles:
      - monitoring

volumes:
  grafana-storage:
"""
        
        try:
            with open('docker-compose.yml', 'w') as f:
                f.write(compose_content)
            print("✅ docker-compose.yml created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating docker-compose.yml: {e}")
            return False
    
    def create_nginx_config(self) -> bool:
        """Create nginx configuration."""
        print("🌐 Creating nginx configuration...")
        
        nginx_content = """events {
    worker_connections 1024;
}

http {
    upstream bul_backend {
        server bul-app:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://bul_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://bul_backend/health;
            access_log off;
        }
    }
}
"""
        
        try:
            with open('nginx.conf', 'w') as f:
                f.write(nginx_content)
            print("✅ nginx.conf created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating nginx.conf: {e}")
            return False
    
    def create_prometheus_config(self) -> bool:
        """Create Prometheus configuration."""
        print("📊 Creating Prometheus configuration...")
        
        prometheus_content = """global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bul-app'
    static_configs:
      - targets: ['bul-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
"""
        
        try:
            with open('prometheus.yml', 'w') as f:
                f.write(prometheus_content)
            print("✅ prometheus.yml created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating prometheus.yml: {e}")
            return False
    
    def create_deployment_scripts(self) -> bool:
        """Create deployment scripts."""
        print("📜 Creating deployment scripts...")
        
        # Create deploy.sh
        deploy_script = """#!/bin/bash
# BUL Deployment Script

set -e

ENVIRONMENT=${1:-production}
echo "🚀 Deploying BUL system in $ENVIRONMENT mode..."

# Build and start services
docker-compose --profile $ENVIRONMENT up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Health check
echo "🔍 Performing health check..."
curl -f http://localhost:8000/health || exit 1

echo "✅ Deployment completed successfully!"
echo "🌐 Application available at: http://localhost:8000"
echo "📚 API docs available at: http://localhost:8000/docs"
"""
        
        # Create stop.sh
        stop_script = """#!/bin/bash
# BUL Stop Script

echo "⏹️ Stopping BUL system..."

docker-compose down

echo "✅ BUL system stopped successfully!"
"""
        
        # Create logs.sh
        logs_script = """#!/bin/bash
# BUL Logs Script

SERVICE=${1:-bul-app}
echo "📋 Showing logs for $SERVICE..."

docker-compose logs -f $SERVICE
"""
        
        try:
            with open('deploy.sh', 'w') as f:
                f.write(deploy_script)
            os.chmod('deploy.sh', 0o755)
            
            with open('stop.sh', 'w') as f:
                f.write(stop_script)
            os.chmod('stop.sh', 0o755)
            
            with open('logs.sh', 'w') as f:
                f.write(logs_script)
            os.chmod('logs.sh', 0o755)
            
            print("✅ Deployment scripts created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating deployment scripts: {e}")
            return False
    
    def create_kubernetes_manifests(self) -> bool:
        """Create Kubernetes deployment manifests."""
        print("☸️ Creating Kubernetes manifests...")
        
        # Create deployment.yaml
        deployment_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: bul-app
  labels:
    app: bul-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bul-app
  template:
    metadata:
      labels:
        app: bul-app
    spec:
      containers:
      - name: bul-app
        image: bul-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: BUL_ENV
          value: "production"
        - name: BUL_DEBUG
          value: "false"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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
---
apiVersion: v1
kind: Service
metadata:
  name: bul-service
spec:
  selector:
    app: bul-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bul-ingress
spec:
  rules:
  - host: bul.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bul-service
            port:
              number: 80
"""
        
        try:
            with open('k8s-deployment.yaml', 'w') as f:
                f.write(deployment_yaml)
            print("✅ Kubernetes manifests created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating Kubernetes manifests: {e}")
            return False
    
    def create_ci_cd_pipeline(self) -> bool:
        """Create CI/CD pipeline configuration."""
        print("🔄 Creating CI/CD pipeline...")
        
        # GitHub Actions workflow
        github_actions = """name: BUL CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements_optimized.txt
    
    - name: Run tests
      run: |
        python bul_toolkit.py run test
    
    - name: Run security audit
      run: |
        python bul_toolkit.py run security
    
    - name: Run performance analysis
      run: |
        python bul_toolkit.py run performance --component all

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t bul-app:${{ github.sha }} .
    
    - name: Run container tests
      run: |
        docker run --rm bul-app:${{ github.sha }} python bul_toolkit.py run validate

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Add your deployment commands here
"""
        
        try:
            os.makedirs('.github/workflows', exist_ok=True)
            with open('.github/workflows/ci-cd.yml', 'w') as f:
                f.write(github_actions)
            print("✅ CI/CD pipeline created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating CI/CD pipeline: {e}")
            return False
    
    def create_monitoring_setup(self) -> bool:
        """Create monitoring setup."""
        print("📊 Creating monitoring setup...")
        
        # Create monitoring script
        monitoring_script = """#!/bin/bash
# BUL Monitoring Setup

echo "📊 Setting up BUL monitoring..."

# Start monitoring stack
docker-compose --profile monitoring up -d

echo "✅ Monitoring stack started!"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana: http://localhost:3000 (admin/admin)"
echo "🔍 BUL App: http://localhost:8000"
"""
        
        try:
            with open('setup-monitoring.sh', 'w') as f:
                f.write(monitoring_script)
            os.chmod('setup-monitoring.sh', 0o755)
            print("✅ Monitoring setup created successfully")
            return True
        except Exception as e:
            print(f"❌ Error creating monitoring setup: {e}")
            return False
    
    def generate_deployment_report(self) -> str:
        """Generate deployment readiness report."""
        report = f"""
BUL Deployment Readiness Report
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DEPLOYMENT OPTIONS
------------------
✅ Docker Deployment
  - Dockerfile created
  - docker-compose.yml created
  - nginx.conf created
  - Deployment scripts created

✅ Kubernetes Deployment
  - k8s-deployment.yaml created
  - Service and Ingress configured
  - Resource limits defined

✅ CI/CD Pipeline
  - GitHub Actions workflow created
  - Automated testing configured
  - Build and deployment pipeline ready

✅ Monitoring Setup
  - Prometheus configuration created
  - Grafana dashboard ready
  - Monitoring scripts created

DEPLOYMENT COMMANDS
------------------
# Docker deployment
./deploy.sh production

# Kubernetes deployment
kubectl apply -f k8s-deployment.yaml

# Monitoring setup
./setup-monitoring.sh

# View logs
./logs.sh bul-app

# Stop services
./stop.sh

ENVIRONMENT CONFIGURATIONS
-------------------------
Development: Local development with hot reload
Staging: Production-like environment for testing
Production: Full production deployment with monitoring

NEXT STEPS
----------
1. Configure environment variables
2. Set up domain and SSL certificates
3. Configure monitoring alerts
4. Set up backup strategies
5. Configure log aggregation
"""
        
        return report

def main():
    """Main deployment manager function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Deployment Manager")
    parser.add_argument("--environment", choices=['development', 'staging', 'production'],
                       default='production', help="Deployment environment")
    parser.add_argument("--component", choices=['all', 'docker', 'k8s', 'cicd', 'monitoring'],
                       default='all', help="Component to create")
    parser.add_argument("--report", action="store_true", help="Generate deployment report")
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    print("🚀 BUL Deployment Manager")
    print("=" * 50)
    print(f"Environment: {args.environment}")
    print(f"Component: {args.component}")
    
    success = True
    
    if args.component in ['all', 'docker']:
        success &= manager.create_dockerfile(args.environment)
        success &= manager.create_docker_compose(args.environment)
        success &= manager.create_nginx_config()
        success &= manager.create_deployment_scripts()
    
    if args.component in ['all', 'k8s']:
        success &= manager.create_kubernetes_manifests()
    
    if args.component in ['all', 'cicd']:
        success &= manager.create_ci_cd_pipeline()
    
    if args.component in ['all', 'monitoring']:
        success &= manager.create_prometheus_config()
        success &= manager.create_monitoring_setup()
    
    if success:
        print("\n✅ All deployment components created successfully!")
        
        if args.report:
            report = manager.generate_deployment_report()
            report_file = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📄 Deployment report saved to: {report_file}")
        
        print("\n🚀 Ready for deployment!")
        print("💡 Use './deploy.sh production' to deploy")
    else:
        print("\n❌ Some components failed to create")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
