#!/bin/bash
# Deployment script for EC2
# This script helps deploy the application to an EC2 instance

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EC2_HOST="${EC2_HOST:-}"
EC2_USER="${EC2_USER:-ubuntu}"
APP_DIR="/opt/manuales-hogar-ai"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${GREEN}=========================================="
echo "Manuales Hogar AI - EC2 Deployment"
echo "==========================================${NC}"

# Check if EC2_HOST is set
if [ -z "$EC2_HOST" ]; then
    echo -e "${YELLOW}EC2_HOST environment variable not set${NC}"
    read -p "Enter EC2 hostname or IP: " EC2_HOST
fi

# Check SSH connection
echo -e "${GREEN}Testing SSH connection...${NC}"
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes $EC2_USER@$EC2_HOST exit 2>/dev/null; then
    echo -e "${RED}ERROR: Cannot connect to $EC2_USER@$EC2_HOST${NC}"
    echo "Please ensure:"
    echo "1. EC2 instance is running"
    echo "2. SSH key is configured"
    echo "3. Security group allows SSH (port 22)"
    exit 1
fi

echo -e "${GREEN}SSH connection successful!${NC}"

# Create remote directory
echo -e "${GREEN}Creating remote directory...${NC}"
ssh $EC2_USER@$EC2_HOST "sudo mkdir -p $APP_DIR && sudo chown $EC2_USER:$EC2_USER $APP_DIR"

# Copy application files
echo -e "${GREEN}Copying application files...${NC}"
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '*.log' \
    --exclude '.pytest_cache' \
    --exclude 'node_modules' \
    $LOCAL_DIR/ $EC2_USER@$EC2_HOST:$APP_DIR/

# Run setup script on remote
echo -e "${GREEN}Running setup script on remote...${NC}"
ssh $EC2_USER@$EC2_HOST "cd $APP_DIR && sudo bash ec2/setup.sh"

# Create .env if it doesn't exist
echo -e "${GREEN}Checking .env file...${NC}"
if ! ssh $EC2_USER@$EC2_HOST "test -f $APP_DIR/.env"; then
    echo -e "${YELLOW}.env file not found. Creating from template...${NC}"
    ssh $EC2_USER@$EC2_HOST "cd $APP_DIR && cp .env.example .env"
    echo -e "${YELLOW}Please configure $APP_DIR/.env on the remote server${NC}"
fi

echo -e "${GREEN}=========================================="
echo "Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps on EC2:"
echo "1. SSH to server: ssh $EC2_USER@$EC2_HOST"
echo "2. Edit .env: sudo nano $APP_DIR/.env"
echo "3. Start service: sudo systemctl start manuales-hogar-ai"
echo "4. Check status: sudo systemctl status manuales-hogar-ai"
echo "5. View logs: sudo journalctl -u manuales-hogar-ai -f"
echo ""




