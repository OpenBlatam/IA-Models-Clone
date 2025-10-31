"""
BUL System - Practical Docker Configuration
Real, practical Docker setup for the BUL system
"""

import os
from pathlib import Path

def create_dockerfile():
    """Create practical Dockerfile"""
    dockerfile_content = """# BUL System - Practical Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile created")

def create_docker_compose():
    """Create practical docker-compose.yml"""
    docker_compose_content = """# BUL System - Practical Docker Compose
version: '3.8'

services:
  # Main application
  bul-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://bul_user:bul_password@postgres:5432/bul_db
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=your-secret-key-here
      - DEBUG=False
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=bul_db
      - POSTGRES_USER=bul_user
      - POSTGRES_PASSWORD=bul_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bul_user -d bul_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - bul-api
    restart: unless-stopped

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Monitoring with Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("✅ docker-compose.yml created")

def create_nginx_config():
    """Create Nginx configuration"""
    nginx_config = """# BUL System - Nginx Configuration
events {
    worker_connections 1024;
}

http {
    upstream bul_api {
        server bul-api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # API routes
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://bul_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Health check
        location /health {
            proxy_pass http://bul_api/health;
            access_log off;
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    }
}
"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    
    print("✅ nginx.conf created")

def create_prometheus_config():
    """Create Prometheus configuration"""
    prometheus_config = """# BUL System - Prometheus Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'bul-api'
    static_configs:
      - targets: ['bul-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
"""
    
    with open("prometheus.yml", "w") as f:
        f.write(prometheus_config)
    
    print("✅ prometheus.yml created")

def create_init_sql():
    """Create database initialization SQL"""
    init_sql = """-- BUL System - Database Initialization
-- Create database and user
CREATE DATABASE bul_db;
CREATE USER bul_user WITH PASSWORD 'bul_password';
GRANT ALL PRIVILEGES ON DATABASE bul_db TO bul_user;

-- Connect to bul_db
\\c bul_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for performance
-- (Indexes will be created by SQLAlchemy migrations)
"""
    
    with open("init.sql", "w") as f:
        f.write(init_sql)
    
    print("✅ init.sql created")

def create_docker_ignore():
    """Create .dockerignore file"""
    dockerignore_content = """# BUL System - Docker Ignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

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

# Docker
Dockerfile
docker-compose.yml
.dockerignore

# Logs
logs/
*.log

# Database
*.db
*.sqlite3

# Environment
.env
.env.local
.env.production

# Documentation
README.md
docs/
*.md

# Tests
tests/
test_*
*_test.py
pytest.ini
.coverage
htmlcov/

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml
"""
    
    with open(".dockerignore", "w") as f:
        f.write(dockerignore_content)
    
    print("✅ .dockerignore created")

def create_requirements():
    """Create requirements.txt for Docker"""
    requirements_content = """# BUL System - Production Requirements
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9

# Cache
redis==5.0.1

# Security
bcrypt==4.1.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# AI Integration
openai==1.3.7

# HTTP Client
httpx==0.25.2
aiohttp==3.9.1

# Monitoring
prometheus-client==0.19.0

# Utilities
python-dotenv==1.0.0
python-multipart==0.0.6

# Production
gunicorn==21.2.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements.txt created")

def create_health_check():
    """Create health check script"""
    health_check_content = """#!/bin/bash
# BUL System - Health Check Script

# Check if API is responding
API_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ API is healthy"
    exit 0
else
    echo "❌ API is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
"""
    
    with open("health_check.sh", "w") as f:
        f.write(health_check_content)
    
    # Make executable
    os.chmod("health_check.sh", 0o755)
    
    print("✅ health_check.sh created")

def create_deployment_script():
    """Create deployment script"""
    deployment_script = """#!/bin/bash
# BUL System - Deployment Script

set -e

echo "🚀 Starting BUL System deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

# Build and start services
echo "📦 Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check health
echo "🔍 Checking service health..."
docker-compose ps

# Run health check
echo "🏥 Running health check..."
./health_check.sh

echo "✅ Deployment completed successfully!"
echo "🌐 API available at: http://localhost:8000"
echo "📊 Grafana available at: http://localhost:3000"
echo "📈 Prometheus available at: http://localhost:9090"
"""
    
    with open("deploy.sh", "w") as f:
        f.write(deployment_script)
    
    # Make executable
    os.chmod("deploy.sh", 0o755)
    
    print("✅ deploy.sh created")

def create_all_docker_files():
    """Create all Docker-related files"""
    print("🐳 Creating practical Docker configuration for BUL System...")
    
    create_dockerfile()
    create_docker_compose()
    create_nginx_config()
    create_prometheus_config()
    create_init_sql()
    create_docker_ignore()
    create_requirements()
    create_health_check()
    create_deployment_script()
    
    print("\n✅ All Docker files created successfully!")
    print("\n📋 Next steps:")
    print("1. Run: chmod +x deploy.sh health_check.sh")
    print("2. Run: ./deploy.sh")
    print("3. Check: http://localhost:8000/health")

if __name__ == "__main__":
    create_all_docker_files()













