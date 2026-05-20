#!/bin/bash
# EC2 User Data Script
# This script runs automatically when an EC2 instance is launched
# It installs and configures the Manuales Hogar AI application

set -e  # Exit on error
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=========================================="
echo "Manuales Hogar AI - EC2 Setup"
echo "=========================================="
echo "Started at: $(date)"

# Update system
echo "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

# Install required packages
echo "Installing required packages..."
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    nginx \
    postgresql-client \
    redis-tools \
    docker.io \
    docker-compose \
    htop \
    unzip \
    jq \
    certbot \
    python3-certbot-nginx \
    awscli

# Start and enable Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Install Docker Compose v2 if not present
if ! command -v docker compose &> /dev/null; then
    echo "Installing Docker Compose v2..."
    apt-get install -y docker-compose-plugin
fi

# Create application directory
APP_DIR="/opt/manuales-hogar-ai"
echo "Creating application directory: $APP_DIR"
mkdir -p $APP_DIR
cd $APP_DIR

# Clone repository (if using Git)
# Uncomment and configure if deploying from Git
# GIT_REPO="${git_repo}"
# GIT_BRANCH="${git_branch:-main}"
# if [ -n "$GIT_REPO" ]; then
#     echo "Cloning repository: $GIT_REPO"
#     git clone -b $GIT_BRANCH $GIT_REPO .
# fi

# Create systemd service
echo "Creating systemd service..."
cat > /etc/systemd/system/manuales-hogar-ai.service << 'EOF'
[Unit]
Description=Manuales Hogar AI Application
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/manuales-hogar-ai
ExecStart=/opt/manuales-hogar-ai/ec2/start-service.sh
ExecStop=/opt/manuales-hogar-ai/ec2/stop-service.sh
Restart=on-failure
RestartSec=10
User=ubuntu
Group=ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Create environment file template
echo "Creating environment file template..."
cat > $APP_DIR/.env.example << 'EOF'
# Application
ENVIRONMENT=prod
DEBUG=false
PORT=8000
WORKERS=4

# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=manuales_user
DB_PASSWORD=change_me
DB_NAME=manuales_db
DATABASE_URL=postgresql://manuales_user:change_me@localhost:5432/manuales_db

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenRouter API
OPENROUTER_API_KEY=your_api_key_here

# Security
ALLOWED_ORIGINS=https://yourdomain.com
SECRET_KEY=change_me_secret_key

# Monitoring
ENABLE_PROMETHEUS=true
ENABLE_TRACING=false
OTLP_ENDPOINT=

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
EOF

# Set permissions
chown -R ubuntu:ubuntu $APP_DIR
chmod +x $APP_DIR/ec2/*.sh 2>/dev/null || true

# Enable and start service (will fail if .env not configured, that's OK)
systemctl daemon-reload
# systemctl enable manuales-hogar-ai  # Enable but don't start yet

echo "=========================================="
echo "EC2 Setup Complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Copy your application files to: $APP_DIR"
echo "2. Configure .env file: $APP_DIR/.env"
echo "3. Start the service: sudo systemctl start manuales-hogar-ai"
echo "4. Check status: sudo systemctl status manuales-hogar-ai"
echo "Finished at: $(date)"




