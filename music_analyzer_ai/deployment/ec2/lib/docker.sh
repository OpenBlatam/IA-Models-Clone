#!/bin/bash
# Docker installation and management functions
# Source common.sh first: source lib/common.sh && source lib/docker.sh

# Install Docker based on OS
install_docker() {
    log_step "Installing Docker"
    
    detect_os
    
    case "$OS_ID" in
        ubuntu|debian)
            install_docker_ubuntu
            ;;
        amzn|amazon|rhel|centos|fedora)
            install_docker_amazon_linux
            ;;
        *)
            error_exit "Unsupported OS: $OS_ID. Please install Docker manually."
            ;;
    esac
    
    # Verify installation
    if ! docker_is_running; then
        error_exit "Docker installation verification failed"
    fi
    
    # Add user to docker group
    if [ -n "${DEFAULT_USER:-}" ] && [ "$DEFAULT_USER" != "$(whoami)" ]; then
        sudo usermod -aG docker "$DEFAULT_USER" 2>/dev/null || true
        log_warning "Added $DEFAULT_USER to docker group. You may need to log out and back in."
    fi
    
    log_success "Docker installed: $(docker --version)"
}

install_docker_ubuntu() {
    log_info "Installing Docker on Ubuntu/Debian..."
    
    # Use cached script if available
    local script_path="/tmp/get-docker.sh"
    if [ ! -f "$script_path" ] || [ $(find "$script_path" -mmin +60 2>/dev/null | wc -l) -gt 0 ]; then
        if retry 3 2 30 curl -fsSL --connect-timeout 10 --max-time 30 \
            https://get.docker.com -o "$script_path"; then
            log_info "Docker installation script downloaded"
        else
            error_exit "Failed to download Docker installation script"
        fi
    else
        log_info "Using cached Docker installation script"
    fi
    
    # Install with optimized flags
    sudo sh "$script_path" --mirror Aliyun 2>/dev/null || \
    sudo sh "$script_path" || error_exit "Failed to install Docker"
    
    # Start services in parallel
    sudo systemctl start docker &
    sudo systemctl enable docker &
    wait
    
    # Cleanup
    rm -f "$script_path"
}

install_docker_amazon_linux() {
    log_info "Installing Docker on Amazon Linux/RHEL/CentOS..."
    
    if retry 3 2 sudo yum install -y docker; then
        sudo systemctl start docker
        sudo systemctl enable docker
    else
        error_exit "Failed to install Docker via yum"
    fi
}

# Optimized Docker Compose installation with caching
install_docker_compose() {
    log_step "Installing Docker Compose"
    
    # Get latest version (cached)
    local compose_version
    if ! compose_version=$(cache_get "compose_version"); then
        compose_version=$(curl -s --max-time 10 --connect-timeout 5 \
            https://api.github.com/repos/docker/compose/releases/latest | \
            grep -o '"tag_name":"[^"]*"' | cut -d'"' -f4) || {
            log_warning "Could not fetch latest version, using default"
            compose_version="v2.24.0"
        }
        cache_set "compose_version" "$compose_version"
    fi
    
    local arch=$(uname -m)
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    
    # Handle architecture mapping
    case "$arch" in
        x86_64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) arch="x86_64" ;;
    esac
    
    local download_url="https://github.com/docker/compose/releases/download/${compose_version}/docker-compose-${os}-${arch}"
    local install_path="/usr/local/bin/docker-compose"
    local temp_file="/tmp/docker-compose-${compose_version}"
    
    # Check if already installed with correct version
    if [ -f "$install_path" ]; then
        local current_version
        if current_version=$(docker-compose --version 2>/dev/null | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+'); then
            if [ "$current_version" = "${compose_version#v}" ]; then
                log_success "Docker Compose already installed: $current_version"
                return 0
            fi
        fi
    fi
    
    log_info "Downloading Docker Compose ${compose_version}..."
    
    # Use cached download if available and recent
    if [ ! -f "$temp_file" ] || [ $(find "$temp_file" -mmin +60 2>/dev/null | wc -l) -gt 0 ]; then
        if retry 3 2 60 curl -L --connect-timeout 10 --max-time 60 \
            --progress-bar "$download_url" -o "$temp_file"; then
            log_info "Download complete"
        else
            error_exit "Failed to download Docker Compose"
        fi
    else
        log_info "Using cached download"
    fi
    
    # Install
    sudo mv "$temp_file" "$install_path"
    sudo chmod +x "$install_path"
    sudo ln -sf "$install_path" /usr/bin/docker-compose
    
    # Verify installation
    if docker-compose --version >/dev/null 2>&1; then
        log_success "Docker Compose installed: $(docker-compose --version)"
    else
        error_exit "Docker Compose installation verification failed"
    fi
}

# Ensure Docker is installed and running
ensure_docker() {
    if ! command_exists docker; then
        install_docker
    elif ! docker_is_running; then
        log_warning "Docker is installed but not running. Starting..."
        sudo systemctl start docker || error_exit "Failed to start Docker"
    else
        log_success "Docker is already installed and running"
    fi
    
    # Ensure user is in docker group
    if ! groups | grep -q docker; then
        sudo usermod -aG docker "$(whoami)" 2>/dev/null || true
        log_warning "Added $(whoami) to docker group"
    fi
}

# Ensure Docker Compose is installed
ensure_docker_compose() {
    if ! docker_compose_cmd >/dev/null 2>&1; then
        install_docker_compose
    else
        local cmd=$(docker_compose_cmd)
        log_success "Docker Compose is already installed: $($cmd --version)"
    fi
}

# Clean Docker resources
docker_cleanup() {
    log_info "Cleaning up Docker resources..."
    docker system prune -f --volumes 2>/dev/null || true
    log_success "Docker cleanup complete"
}

# Get Docker stats
docker_stats() {
    if docker_is_running; then
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    else
        log_error "Docker is not running"
        return 1
    fi
}

