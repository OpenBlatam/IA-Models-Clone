#!/bin/bash
# Initial EC2 setup script for unified_ai_model
# Run this script once on a fresh EC2 instance (Amazon Linux 2 or Ubuntu)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    log_warn "Cannot detect OS, assuming Amazon Linux"
    OS="amzn"
fi

log_info "Detected OS: $OS"

# Install Docker based on OS
install_docker() {
    if command -v docker &> /dev/null; then
        log_info "Docker is already installed"
        return
    fi

    log_info "Installing Docker..."
    
    if [[ "$OS" == "amzn" ]]; then
        # Amazon Linux 2
        sudo yum update -y
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -a -G docker ec2-user
    elif [[ "$OS" == "ubuntu" ]]; then
        # Ubuntu
        sudo apt-get update
        sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io
        sudo usermod -a -G docker ubuntu
    else
        log_warn "Unsupported OS. Installing Docker using get.docker.com script..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
    fi
}

# Install Docker Compose
install_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        log_info "Docker Compose is already installed"
        return
    fi

    log_info "Installing Docker Compose..."
    
    DOCKER_COMPOSE_VERSION="v2.24.0"
    sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
}

# Install Git
install_git() {
    if command -v git &> /dev/null; then
        log_info "Git is already installed"
        return
    fi

    log_info "Installing Git..."
    
    if [[ "$OS" == "amzn" ]]; then
        sudo yum install -y git
    elif [[ "$OS" == "ubuntu" ]]; then
        sudo apt-get install -y git
    fi
}

# Clone repository
setup_repository() {
    APP_DIR="/home/$(whoami)/unified_ai_model"
    REPO_URL="${REPO_URL:-https://github.com/OpenBlatam/onyx.git}"
    
    if [ -d "$APP_DIR" ]; then
        log_info "Repository already exists at $APP_DIR"
        cd "$APP_DIR"
        git pull origin main
    else
        log_info "Cloning repository..."
        git clone "$REPO_URL" "$APP_DIR"
    fi
}

# Create environment file
setup_environment() {
    ENV_FILE="/home/$(whoami)/unified_ai_model/backend/unified_ai_model/aws/.env"
    
    if [ -f "$ENV_FILE" ]; then
        log_info "Environment file already exists"
        return
    fi

    log_info "Creating environment file template..."
    cat > "$ENV_FILE" << 'EOF'
# Unified AI Model Environment Variables
# Fill in your API keys and configuration

# Required: At least one API key
DEEPSEEK_API_KEY=your-deepseek-api-key
OPENROUTER_API_KEY=

# Optional configuration
CORS_ORIGINS=http://localhost:3000,https://your-frontend-domain.com
DEBUG=false
EOF

    log_warn "Please edit $ENV_FILE with your actual API keys!"
}

# Configure firewall
setup_firewall() {
    log_info "Configuring firewall rules..."
    
    if command -v ufw &> /dev/null; then
        sudo ufw allow 80/tcp
        sudo ufw allow 443/tcp
        sudo ufw allow 8050/tcp
    elif command -v firewall-cmd &> /dev/null; then
        sudo firewall-cmd --permanent --add-port=80/tcp
        sudo firewall-cmd --permanent --add-port=443/tcp
        sudo firewall-cmd --permanent --add-port=8050/tcp
        sudo firewall-cmd --reload
    else
        log_warn "No firewall detected. Make sure your security group allows ports 80, 443, and 8050"
    fi
}

# Main execution
main() {
    log_info "Starting EC2 setup for unified_ai_model..."
    
    install_docker
    install_docker_compose
    install_git
    setup_repository
    setup_environment
    setup_firewall
    
    log_info "============================================"
    log_info "EC2 setup completed successfully!"
    log_info "============================================"
    log_info ""
    log_info "Next steps:"
    log_info "1. Edit the .env file with your API keys:"
    log_info "   nano /home/$(whoami)/unified_ai_model/backend/unified_ai_model/aws/.env"
    log_info ""
    log_info "2. Start the service:"
    log_info "   cd /home/$(whoami)/unified_ai_model/backend/unified_ai_model"
    log_info "   docker-compose -f aws/docker-compose.yml up -d"
    log_info ""
    log_info "3. Check the service status:"
    log_info "   curl http://localhost:8050/api/v1/health"
    log_info ""
    log_warn "Remember to log out and log back in for Docker group permissions to take effect!"
}

main "$@"
