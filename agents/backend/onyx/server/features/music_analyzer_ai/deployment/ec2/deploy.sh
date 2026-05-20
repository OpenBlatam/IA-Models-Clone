#!/bin/bash
# Universal deployment script for EC2 instances
# Refactored with modular functions and better structure

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

# Source libraries
source "$LIB_DIR/common.sh" || error_exit "Failed to load common.sh"
source "$LIB_DIR/docker.sh" || error_exit "Failed to load docker.sh"
source "$LIB_DIR/deployment.sh" || error_exit "Failed to load deployment.sh"
source "$LIB_DIR/performance.sh" || error_exit "Failed to load performance.sh"

# Configuration
readonly APP_DIR="/opt/music-analyzer-ai"
readonly LOG_FILE="/var/log/music-analyzer-deploy.log"
readonly HEALTH_URL="http://localhost:8010/health"

# Export for use in functions
export LOG_FILE APP_DIR

# Main deployment function
main() {
    log_step "Music Analyzer AI - EC2 Deployment"
    
    # Pre-flight checks (run in parallel where possible)
    check_sudo
    detect_os
    
    log_info "OS: $OS_NAME"
    log_info "User: ${DEFAULT_USER:-$(whoami)}"
    log_info "Parallel jobs: $PARALLEL_JOBS"
    
    # Run resource checks and optimizations in parallel
    check_resources &
    local resources_pid=$!
    
    # Apply performance optimizations in background
    apply_all_optimizations &
    local optimizations_pid=$!
    
    # Ensure Docker and Docker Compose (can run in parallel)
    ensure_docker &
    local docker_pid=$!
    
    ensure_docker_compose &
    local compose_pid=$!
    
    # Wait for all background tasks
    wait $resources_pid
    wait $docker_pid
    wait $compose_pid
    wait $optimizations_pid
    
    # Get project root
    local project_root
    project_root=$(cd "$SCRIPT_DIR/../.." && pwd)
    
    # Deploy application
    copy_application_files "$project_root" "$APP_DIR"
    create_env_file "$APP_DIR"
    create_management_scripts "$APP_DIR"
    
    # Pre-warm images in background while copying files
    prewarm_images "$APP_DIR" &
    local prewarm_pid=$!
    
    # Build and start (with parallel builds)
    wait $prewarm_pid
    build_images "$APP_DIR" 1800 true  # Enable parallel builds
    start_services "$APP_DIR"
    
    # Wait for services and validate
    if wait_for_service "$HEALTH_URL" 30 2; then
        print_deployment_info
        print_commands "$APP_DIR"
        log_success "Deployment completed successfully!"
    else
        log_warning "Health check failed. Services may still be starting."
        log_info "Recent logs:"
        get_service_logs "$APP_DIR" "" 50
        print_commands "$APP_DIR"
    fi
}

# Run main function
main "$@"

# Enhanced OS detection
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
        OS_NAME="$NAME"
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
        VER=$(lsb_release -sr)
        OS_NAME="$OS $VER"
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        OS=$DISTRIB_ID
        VER=$DISTRIB_RELEASE
        OS_NAME="$DISTRIB_DESCRIPTION"
    elif [ -f /etc/debian_version ]; then
        OS=debian
        VER=$(cat /etc/debian_version)
        OS_NAME="Debian $VER"
    elif [ -f /etc/redhat-release ]; then
        OS=$(cat /etc/redhat-release | awk '{print $1}' | tr '[:upper:]' '[:lower:]')
        VER=$(cat /etc/redhat-release | sed 's/.*release \([0-9.]*\).*/\1/')
        OS_NAME=$(cat /etc/redhat-release)
    else
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        VER=$(uname -r)
        OS_NAME="$OS $VER"
    fi
    
    # Normalize OS names
    case $OS in
        amzn|amazon)
            OS="amzn"
            USER_NAME="ec2-user"
            ;;
        ubuntu|debian)
            USER_NAME="ubuntu"
            ;;
        rhel|centos|fedora)
            USER_NAME="ec2-user"
            ;;
        *)
            USER_NAME=$(whoami)
            ;;
    esac
}

detect_os
log "${YELLOW}📦 Detected OS: $OS_NAME${NC}"
log "${YELLOW}👤 User: $USER_NAME${NC}"

# Application directory
APP_DIR="/opt/music-analyzer-ai"
mkdir -p $APP_DIR
cd $APP_DIR

# Install Docker with retry logic
install_docker() {
    log "${BLUE}🐳 Installing Docker...${NC}"
    
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        case $OS in
            ubuntu|debian)
                if curl -fsSL https://get.docker.com -o /tmp/get-docker.sh; then
                    sh /tmp/get-docker.sh || error_exit "Failed to install Docker"
                    sudo usermod -aG docker $USER_NAME
                    sudo systemctl start docker
                    sudo systemctl enable docker
                    break
                fi
                ;;
            amzn|rhel|centos|fedora)
                if sudo yum install -y docker; then
                    sudo systemctl start docker
                    sudo systemctl enable docker
                    sudo usermod -aG docker $USER_NAME
                    break
                fi
                ;;
            *)
                error_exit "Unsupported OS: $OS. Please install Docker manually."
                ;;
        esac
        
        if [ $attempt -lt $max_attempts ]; then
            log "${YELLOW}⚠️  Attempt $attempt failed. Retrying...${NC}"
            sleep 5
        else
            error_exit "Failed to install Docker after $max_attempts attempts"
        fi
        attempt=$((attempt + 1))
    done
    
    # Verify Docker installation
    if ! docker --version > /dev/null 2>&1; then
        error_exit "Docker installation verification failed"
    fi
    
    log "${GREEN}✅ Docker installed successfully${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null || ! docker info > /dev/null 2>&1; then
    install_docker
else
    log "${GREEN}✅ Docker is already installed${NC}"
    # Ensure user is in docker group
    if ! groups | grep -q docker; then
        sudo usermod -aG docker $USER_NAME
        log "${YELLOW}⚠️  Added $USER_NAME to docker group. You may need to log out and back in.${NC}"
    fi
fi

# Install Docker Compose with version detection
install_docker_compose() {
    log "${BLUE}📦 Installing Docker Compose...${NC}"
    
    local compose_version=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)
    local arch=$(uname -m)
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    
    # Handle architecture mapping
    case $arch in
        x86_64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) arch="x86_64" ;;
    esac
    
    local download_url="https://github.com/docker/compose/releases/download/${compose_version}/docker-compose-${os}-${arch}"
    
    if curl -L "$download_url" -o /tmp/docker-compose; then
        sudo mv /tmp/docker-compose /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
        
        # Verify installation
        if docker-compose --version > /dev/null 2>&1; then
            log "${GREEN}✅ Docker Compose installed: $(docker-compose --version)${NC}"
        else
            error_exit "Docker Compose installation verification failed"
        fi
    else
        error_exit "Failed to download Docker Compose"
    fi
}

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    install_docker_compose
else
    if command -v docker-compose &> /dev/null; then
        log "${GREEN}✅ Docker Compose is already installed: $(docker-compose --version)${NC}"
    else
        log "${GREEN}✅ Docker Compose plugin is available${NC}"
    fi
fi

# Get the script directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Application directory
APP_DIR="/opt/music-analyzer-ai"
sudo mkdir -p $APP_DIR
sudo chown -R $USER_NAME:$USER_NAME $APP_DIR

log "${GREEN}📦 Copying application files...${NC}"

# Check if source directory exists
if [ ! -d "$PROJECT_ROOT" ]; then
    error_exit "Source directory not found: $PROJECT_ROOT"
fi

# Copy application files with better error handling
if command -v rsync &> /dev/null; then
    rsync -av --progress \
          --exclude='.git' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.env' \
          --exclude='node_modules' \
          --exclude='.next' \
          --exclude='*.log' \
          --exclude='.DS_Store' \
          "$PROJECT_ROOT/" "$APP_DIR/" || error_exit "Failed to copy application files"
else
    # Fallback to cp if rsync not available
    log "${YELLOW}⚠️  rsync not found, using cp (slower)...${NC}"
    cp -r "$PROJECT_ROOT"/* "$APP_DIR/" 2>/dev/null || error_exit "Failed to copy application files"
fi

log "${GREEN}✅ Application files copied${NC}"

# Create .env if it doesn't exist
if [ ! -f "$APP_DIR/.env" ]; then
    echo -e "${YELLOW}Creating .env file template...${NC}"
    cat > "$APP_DIR/.env" << 'EOF'
ENVIRONMENT=production
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
LOG_LEVEL=INFO
CACHE_ENABLED=true
POSTGRES_PASSWORD=changeme
REDIS_PASSWORD=changeme
GRAFANA_PASSWORD=admin
DATABASE_URL=postgresql://music_analyzer:changeme@postgres:5432/music_analyzer_db
EOF
    echo -e "${YELLOW}⚠️  Please update $APP_DIR/.env with your credentials!${NC}"
fi

# Create startup script
cat > "$APP_DIR/start.sh" << 'EOF'
#!/bin/bash
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml up -d
EOF

chmod +x "$APP_DIR/start.sh"

# Create stop script
cat > "$APP_DIR/stop.sh" << 'EOF'
#!/bin/bash
cd /opt/music-analyzer-ai
docker-compose -f deployment/docker-compose.prod.yml down
EOF

chmod +x "$APP_DIR/stop.sh"

# Build and start with retry logic
build_and_start() {
    log "${GREEN}🔨 Building Docker images...${NC}"
    cd "$APP_DIR"
    
    # Build with timeout
    if timeout 1800 docker-compose -f deployment/docker-compose.prod.yml build; then
        log "${GREEN}✅ Docker images built successfully${NC}"
    else
        error_exit "Docker build failed or timed out"
    fi
    
    log "${GREEN}🚀 Starting services...${NC}"
    if docker-compose -f deployment/docker-compose.prod.yml up -d; then
        log "${GREEN}✅ Services started${NC}"
    else
        error_exit "Failed to start services"
    fi
}

# Health check with retries
health_check() {
    local max_retries=30
    local retry_count=0
    local health_url="http://localhost:8010/health"
    
    log "${GREEN}⏳ Waiting for services to be ready...${NC}"
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -sf "$health_url" > /dev/null 2>&1; then
            log "${GREEN}✅ Health check passed!${NC}"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $((retry_count % 5)) -eq 0 ]; then
            log "${YELLOW}⏳ Still waiting... ($retry_count/$max_retries)${NC}"
        fi
        sleep 2
    done
    
    log "${YELLOW}⚠️  Health check timeout. Services may still be starting.${NC}"
    return 1
}

# Build and start
build_and_start

# Wait and check health
if health_check; then
    # Get instance IP
    INSTANCE_IP=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
    
    log "${GREEN}✅ Deployment successful!${NC}"
    log "${CYAN}========================================${NC}"
    log "${GREEN}📊 Service URLs:${NC}"
    log "  🌐 API:          http://$INSTANCE_IP:8010"
    log "  ❤️  Health:       http://$INSTANCE_IP:8010/health"
    log "  📖 Docs:          http://$INSTANCE_IP:8010/docs"
    log "  📈 Grafana:       http://$INSTANCE_IP:3000"
    log "  📊 Prometheus:    http://$INSTANCE_IP:9090"
    log ""
    log "${GREEN}📝 Useful commands:${NC}"
    log "  View logs:    cd $APP_DIR && docker-compose -f deployment/docker-compose.prod.yml logs -f"
    log "  Stop:         cd $APP_DIR && ./stop.sh"
    log "  Restart:      cd $APP_DIR && ./start.sh"
    log "  Status:       cd $APP_DIR && docker-compose -f deployment/docker-compose.prod.yml ps"
    log "${CYAN}========================================${NC}"
else
    log "${YELLOW}⚠️  Services may still be starting. Check logs:${NC}"
    log "  cd $APP_DIR && docker-compose -f deployment/docker-compose.prod.yml logs"
    log ""
    log "${BLUE}📋 Recent logs:${NC}"
    cd "$APP_DIR"
    docker-compose -f deployment/docker-compose.prod.yml logs --tail=50
fi

log "${GREEN}✨ Deployment script completed!${NC}"

