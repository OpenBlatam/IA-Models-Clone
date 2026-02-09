#!/bin/bash
# Complete setup script for EC2
# Run this script to set up the application on a fresh EC2 instance

set -e

APP_DIR="/opt/manuales-hogar-ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Manuales Hogar AI - EC2 Setup"
echo "=========================================="

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo"
    exit 1
fi

# Update system
echo "Updating system..."
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

# Install Docker Compose v2
if ! command -v docker compose &> /dev/null; then
    echo "Installing Docker Compose v2..."
    apt-get install -y docker-compose-plugin
fi

# Start and enable Docker
echo "Configuring Docker..."
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu || usermod -aG docker $SUDO_USER || true

# Create application directory
echo "Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Copy application files (if running from source)
if [ -d "$SCRIPT_DIR/../" ]; then
    echo "Copying application files..."
    cp -r $SCRIPT_DIR/../* $APP_DIR/ 2>/dev/null || true
    # Exclude certain files
    rm -rf $APP_DIR/.git $APP_DIR/__pycache__ $APP_DIR/*/__pycache__ 2>/dev/null || true
fi

# Set permissions
chown -R ubuntu:ubuntu $APP_DIR || chown -R $SUDO_USER:$SUDO_USER $APP_DIR
chmod +x $APP_DIR/ec2/*.sh 2>/dev/null || true

# Create .env if it doesn't exist
if [ ! -f "$APP_DIR/.env" ]; then
    echo "Creating .env file from template..."
    if [ -f "$APP_DIR/.env.example" ]; then
        cp $APP_DIR/.env.example $APP_DIR/.env
        echo "Please edit $APP_DIR/.env with your configuration"
    fi
fi

# Setup Nginx
echo "Configuring Nginx..."
if [ -f "$APP_DIR/ec2/nginx.conf" ]; then
    cp $APP_DIR/ec2/nginx.conf /etc/nginx/sites-available/manuales-hogar-ai
    ln -sf /etc/nginx/sites-available/manuales-hogar-ai /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl reload nginx
fi

# Setup systemd service
echo "Setting up systemd service..."
cp $APP_DIR/ec2/manuales-hogar-ai.service /etc/systemd/system/ 2>/dev/null || \
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

systemctl daemon-reload
systemctl enable manuales-hogar-ai

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit configuration: sudo nano $APP_DIR/.env"
echo "2. Start the service: sudo systemctl start manuales-hogar-ai"
echo "3. Check status: sudo systemctl status manuales-hogar-ai"
echo "4. View logs: sudo journalctl -u manuales-hogar-ai -f"
echo "5. Setup SSL: sudo certbot --nginx -d yourdomain.com"
echo ""




