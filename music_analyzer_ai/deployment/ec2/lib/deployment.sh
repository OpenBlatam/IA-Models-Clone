#!/bin/bash
# Deployment functions
# Source common.sh first: source lib/common.sh && source lib/deployment.sh

# Optimized file copy with parallel processing and checksums
copy_application_files() {
    local source_dir="${1}"
    local target_dir="${2:-/opt/music-analyzer-ai}"
    
    log_step "Copying Application Files"
    
    if [ ! -d "$source_dir" ]; then
        error_exit "Source directory not found: $source_dir"
    fi
    
    sudo mkdir -p "$target_dir"
    sudo chown -R "${DEFAULT_USER:-$(whoami)}:${DEFAULT_USER:-$(whoami)}" "$target_dir"
    
    # Check if target already exists and is up to date
    if [ -d "$target_dir" ] && [ -f "$target_dir/.deployment-checksum" ]; then
        local source_checksum
        local target_checksum
        
        if command_exists find && command_exists md5sum; then
            source_checksum=$(find "$source_dir" -type f -exec md5sum {} \; 2>/dev/null | \
                sort | md5sum | cut -d' ' -f1)
            target_checksum=$(cat "$target_dir/.deployment-checksum" 2>/dev/null)
            
            if [ "$source_checksum" = "$target_checksum" ]; then
                log_info "Files are up to date, skipping copy"
                return 0
            fi
        fi
    fi
    
    # Use rsync with optimizations
    if command_exists rsync; then
        log_info "Using rsync for efficient file transfer..."
        
        # Build exclude list
        local exclude_file="/tmp/rsync-excludes-$$"
        cat > "$exclude_file" << 'EOF'
.git
__pycache__
*.pyc
.env
node_modules
.next
*.log
.DS_Store
.vscode
.idea
*.swp
*.swo
*.tmp
EOF
        
        # Use rsync with compression and checksums
        rsync -avz --delete \
              --exclude-from="$exclude_file" \
              --checksum \
              --info=progress2 \
              "$source_dir/" "$target_dir/" || error_exit "Failed to copy application files"
        
        rm -f "$exclude_file"
        
        # Save checksum for future comparisons
        if command_exists find && command_exists md5sum; then
            find "$source_dir" -type f -exec md5sum {} \; 2>/dev/null | \
                sort | md5sum | cut -d' ' -f1 > "$target_dir/.deployment-checksum" 2>/dev/null || true
        fi
    else
        log_warning "rsync not found, using tar (faster than cp)..."
        # Use tar for better performance than cp
        (cd "$source_dir" && tar --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.env' \
            --exclude='node_modules' \
            --exclude='.next' \
            -cf - .) | (cd "$target_dir" && tar -xf -) || \
            error_exit "Failed to copy application files"
    fi
    
    log_success "Application files copied to $target_dir"
}

# Create environment file
create_env_file() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    local env_file="$app_dir/.env"
    
    if [ -f "$env_file" ]; then
        log_info "Environment file already exists: $env_file"
        return 0
    fi
    
    log_info "Creating environment file template..."
    
    cat > "$env_file" << 'EOF'
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
    
    log_warning "Please update $env_file with your actual credentials!"
}

# Create management scripts
create_management_scripts() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    
    log_step "Creating Management Scripts"
    
    # Start script
    cat > "$app_dir/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker-compose -f deployment/docker-compose.prod.yml up -d
EOF
    
    # Stop script
    cat > "$app_dir/stop.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker-compose -f deployment/docker-compose.prod.yml down
EOF
    
    # Restart script
    cat > "$app_dir/restart.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker-compose -f deployment/docker-compose.prod.yml restart
EOF
    
    # Status script
    cat > "$app_dir/status.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker-compose -f deployment/docker-compose.prod.yml ps
EOF
    
    # Make scripts executable
    chmod +x "$app_dir"/*.sh
    
    log_success "Management scripts created"
}

# Optimized Docker build with parallel builds and cache
build_images() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    local timeout="${2:-1800}"  # 30 minutes default
    local parallel="${3:-true}"
    
    log_step "Building Docker Images"
    
    cd "$app_dir" || error_exit "Cannot change to directory: $app_dir"
    
    local compose_cmd=$(docker_compose_cmd) || error_exit "Docker Compose not available"
    
    # Build with optimizations
    local build_args=""
    if [ "$parallel" = "true" ]; then
        build_args="--parallel"
    fi
    
    # Use BuildKit for better performance
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    log_info "Building with optimizations (BuildKit, parallel: $parallel)..."
    
    if timeout "$timeout" $compose_cmd -f deployment/docker-compose.prod.yml \
        build $build_args --progress=plain; then
        log_success "Docker images built successfully"
    else
        error_exit "Docker build failed or timed out"
    fi
}

# Start services
start_services() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    
    log_step "Starting Services"
    
    cd "$app_dir" || error_exit "Cannot change to directory: $app_dir"
    
    local compose_cmd=$(docker_compose_cmd) || error_exit "Docker Compose not available"
    
    if $compose_cmd -f deployment/docker-compose.prod.yml up -d; then
        log_success "Services started"
    else
        error_exit "Failed to start services"
    fi
}

# Stop services
stop_services() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    
    log_step "Stopping Services"
    
    cd "$app_dir" || error_exit "Cannot change to directory: $app_dir"
    
    local compose_cmd=$(docker_compose_cmd) || error_exit "Docker Compose not available"
    
    $compose_cmd -f deployment/docker-compose.prod.yml down
    log_success "Services stopped"
}

# Get service status
get_service_status() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    
    cd "$app_dir" || return 1
    
    local compose_cmd=$(docker_compose_cmd) || return 1
    $compose_cmd -f deployment/docker-compose.prod.yml ps
}

# Get service logs
get_service_logs() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    local service="${2:-}"
    local lines="${3:-50}"
    
    cd "$app_dir" || return 1
    
    local compose_cmd=$(docker_compose_cmd) || return 1
    
    if [ -n "$service" ]; then
        $compose_cmd -f deployment/docker-compose.prod.yml logs --tail="$lines" "$service"
    else
        $compose_cmd -f deployment/docker-compose.prod.yml logs --tail="$lines"
    fi
}

