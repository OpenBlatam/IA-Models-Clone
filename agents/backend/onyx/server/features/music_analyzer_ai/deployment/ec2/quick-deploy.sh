#!/bin/bash
# Quick deployment script - Works on any EC2 instance
# Detects OS and deploys Music Analyzer AI

set -e

echo "=========================================="
echo "Music Analyzer AI - Quick EC2 Deployment"
echo "=========================================="

# Detect OS and user
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    USER_NAME=$(whoami)
elif [ -f /etc/redhat-release ]; then
    OS="amzn"
    USER_NAME="ec2-user"
else
    OS="ubuntu"
    USER_NAME="ubuntu"
fi

echo "Detected OS: $OS"
echo "User: $USER_NAME"

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        sudo usermod -aG docker $USER_NAME
    else
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER_NAME
    fi
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
fi

# Create app directory
APP_DIR="/opt/music-analyzer-ai"
sudo mkdir -p $APP_DIR
sudo chown $USER_NAME:$USER_NAME $APP_DIR

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Deploy your code to: $APP_DIR"
echo "2. Create .env file with your credentials"
echo "3. Run: cd $APP_DIR && docker-compose -f deployment/docker-compose.prod.yml up -d"
echo ""
echo "Or use the full deploy script:"
echo "  ./deployment/ec2/deploy.sh"
echo "=========================================="




