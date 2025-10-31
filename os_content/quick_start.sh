#!/bin/bash

# Quick Start Script for OS Content Production System
# One-command deployment and startup

set -e

echo "🚀 OS Content Production System - Quick Start"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
mkdir -p {logs,data,cache,models,backup,ssl}

# Generate SSL certificates if they don't exist
if [ ! -f "ssl/cert.pem" ]; then
    echo "🔐 Generating SSL certificates..."
    mkdir -p ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/key.pem \
        -out ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# Create basic configuration files
echo "⚙️  Creating configuration files..."

# Redis config
cat > redis.conf << EOF
bind 0.0.0.0
port 6379
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
EOF

# Nginx config
cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream os_content_backend {
        server os-content-app:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://os_content_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        location /health {
            proxy_pass http://os_content_backend;
            access_log off;
        }
    }
}
EOF

# Prometheus config
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'os-content-app'
    static_configs:
      - targets: ['os-content-app:8000']
    metrics_path: '/metrics'
EOF

# Database init
cat > init.sql << EOF
CREATE DATABASE IF NOT EXISTS os_content;
USE os_content;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    prompt TEXT NOT NULL,
    duration INTEGER NOT NULL,
    output_path VARCHAR(500) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nlp_analyses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    text TEXT NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF

# Build and start services
echo "🔨 Building and starting services..."

# Build the application
docker build -f production_dockerfile -t os-content:latest .

# Start services
docker-compose -f production_compose.yml up -d postgres redis
echo "⏳ Waiting for database services..."
sleep 10

docker-compose -f production_compose.yml up -d prometheus grafana
echo "⏳ Waiting for monitoring services..."
sleep 5

docker-compose -f production_compose.yml up -d os-content-app
echo "⏳ Waiting for application..."
sleep 10

docker-compose -f production_compose.yml up -d nginx

# Wait for services to be ready
echo "⏳ Waiting for all services to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

# Check if services are running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "✅ OS Content Production System is ready!"
    echo ""
    echo "🌐 Services:"
    echo "   Application: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo "   Health Check: http://localhost:8000/health"
    echo "   Grafana: http://localhost:3000 (admin/admin)"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "🗄️  Databases:"
    echo "   PostgreSQL: localhost:5432"
    echo "   Redis: localhost:6379"
    echo ""
    echo "🔧 Management:"
    echo "   View logs: docker-compose -f production_compose.yml logs -f"
    echo "   Stop system: docker-compose -f production_compose.yml down"
    echo "   Restart: docker-compose -f production_compose.yml restart"
    echo ""
    echo "📊 Quick Test:"
    echo "   curl -X POST http://localhost:8000/api/nlp/analyze \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"text\": \"Hello world!\", \"analysis_type\": \"sentiment\"}'"
    echo ""
else
    echo "❌ Failed to start services. Check logs with:"
    echo "   docker-compose -f production_compose.yml logs"
    exit 1
fi 