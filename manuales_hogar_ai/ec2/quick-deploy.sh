#!/bin/bash
# Quick deployment script for EC2
# Simplifies the deployment process

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Manuales Hogar AI - Quick EC2 Deployment${NC}"
echo ""

# Get EC2 details
read -p "EC2 Hostname/IP: " EC2_HOST
read -p "EC2 User [ubuntu]: " EC2_USER
EC2_USER=${EC2_USER:-ubuntu}

# Check if files exist
if [ ! -f "ec2/setup.sh" ]; then
    echo -e "${YELLOW}Error: ec2/setup.sh not found${NC}"
    echo "Please run this script from the project root"
    exit 1
fi

# Deploy
echo -e "${GREEN}Deploying to $EC2_USER@$EC2_HOST...${NC}"

# Test connection
if ! ssh -o ConnectTimeout=5 $EC2_USER@$EC2_HOST exit 2>/dev/null; then
    echo -e "${YELLOW}Error: Cannot connect to EC2 instance${NC}"
    exit 1
fi

# Create directory
ssh $EC2_USER@$EC2_HOST "sudo mkdir -p /opt/manuales-hogar-ai && sudo chown $EC2_USER:$EC2_USER /opt/manuales-hogar-ai"

# Copy files
echo "Copying files..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '*.log' \
    . $EC2_USER@$EC2_HOST:/opt/manuales-hogar-ai/

# Run setup
echo "Running setup..."
ssh $EC2_USER@$EC2_HOST "cd /opt/manuales-hogar-ai && sudo bash ec2/setup.sh"

echo -e "${GREEN}Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "1. SSH: ssh $EC2_USER@$EC2_HOST"
echo "2. Configure: sudo nano /opt/manuales-hogar-ai/.env"
echo "3. Start: sudo systemctl start manuales-hogar-ai"




