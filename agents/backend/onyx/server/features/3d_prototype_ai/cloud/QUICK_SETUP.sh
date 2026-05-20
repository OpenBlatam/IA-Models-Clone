#!/bin/bash
# Quick setup script - One command to set up everything
# This script ensures all bash scripts are executable and environment is ready

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "3D Prototype AI - Quick Setup"
echo "=========================================="
echo ""

# Make this script executable
chmod +x "${0}" 2>/dev/null || true

# Make all scripts executable
echo "Making scripts executable..."
find scripts -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
chmod +x user_data/init.sh 2>/dev/null || true
echo "✓ Scripts made executable"
echo ""

# Run setup script if it exists
if [ -f "scripts/setup.sh" ]; then
    echo "Running setup script..."
    ./scripts/setup.sh
else
    echo "Setup script not found, creating basic structure..."
    
    # Create directories
    mkdir -p backups logs tmp
    
    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# AWS Configuration
AWS_REGION=us-east-1
AWS_INSTANCE_TYPE=t3.large
AWS_KEY_NAME=your-key-pair-name
AWS_KEY_PATH=~/.ssh/your-key-pair-name.pem

# Application Configuration
APP_PORT=8030
APP_HOST=0.0.0.0

# Deployment Configuration
DEPLOYMENT_METHOD=terraform
SKIP_APP_DEPLOY=false
USE_DOCKER=true
EOF
        echo "✓ Created .env file (please edit with your settings)"
    fi
    
    echo "✓ Basic structure created"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: ./scripts/check_environment.sh"
echo "3. Run: ./scripts/validate.sh"
echo "4. Run: ./scripts/deploy.sh"
echo ""

