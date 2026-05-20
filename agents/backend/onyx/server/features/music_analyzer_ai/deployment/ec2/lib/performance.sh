#!/bin/bash
# Performance optimization functions
# Source common.sh first: source lib/common.sh && source lib/performance.sh

# Optimize system settings
optimize_system() {
    log_step "Optimizing System Performance"
    
    # Increase file descriptor limits
    if [ -f /etc/security/limits.conf ]; then
        if ! grep -q "music-analyzer" /etc/security/limits.conf; then
            log_info "Increasing file descriptor limits..."
            sudo tee -a /etc/security/limits.conf > /dev/null << 'EOF'
# Music Analyzer AI optimizations
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF
        fi
    fi
    
    # Optimize kernel parameters for Docker
    if [ -f /etc/sysctl.conf ]; then
        if ! grep -q "music-analyzer" /etc/sysctl.conf; then
            log_info "Optimizing kernel parameters..."
            sudo tee -a /etc/sysctl.conf > /dev/null << 'EOF'
# Music Analyzer AI kernel optimizations
net.core.somaxconn = 1024
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.ip_local_port_range = 10000 65535
vm.swappiness = 10
vm.overcommit_memory = 1
EOF
            sudo sysctl -p > /dev/null 2>&1 || true
        fi
    fi
    
    log_success "System optimizations applied"
}

# Optimize Docker daemon
optimize_docker() {
    log_step "Optimizing Docker Configuration"
    
    local docker_daemon="/etc/docker/daemon.json"
    
    if [ ! -f "$docker_daemon" ]; then
        log_info "Creating Docker daemon configuration..."
        sudo mkdir -p /etc/docker
    fi
    
    # Backup existing config
    if [ -f "$docker_daemon" ]; then
        sudo cp "$docker_daemon" "${docker_daemon}.bak"
    fi
    
    # Create optimized configuration
    sudo tee "$docker_daemon" > /dev/null << 'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  },
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5,
  "experimental": false
}
EOF
    
    # Restart Docker if running
    if systemctl is-active --quiet docker; then
        log_info "Restarting Docker to apply optimizations..."
        sudo systemctl restart docker
    fi
    
    log_success "Docker optimizations applied"
}

# Pre-warm Docker images
prewarm_images() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    
    log_info "Pre-warming Docker images..."
    
    cd "$app_dir" || return 1
    
    local compose_cmd=$(docker_compose_cmd) || return 1
    
    # Pull images in parallel
    $compose_cmd -f deployment/docker-compose.prod.yml pull --parallel --quiet &
    local pull_pid=$!
    
    # Build in background if needed
    $compose_cmd -f deployment/docker-compose.prod.yml build --parallel --quiet &
    local build_pid=$!
    
    # Wait for both
    wait $pull_pid
    wait $build_pid
    
    log_success "Images pre-warmed"
}

# Optimize disk I/O
optimize_disk_io() {
    log_info "Checking disk I/O performance..."
    
    # Check if we're on EBS
    if [ -b /dev/nvme0n1 ] || [ -b /dev/xvda ]; then
        log_info "EBS volume detected, enabling optimizations..."
        
        # Enable writeback caching if available
        if command_exists blockdev; then
            for device in /dev/nvme* /dev/xvd*; do
                if [ -b "$device" ]; then
                    sudo blockdev --setra 8192 "$device" 2>/dev/null || true
                fi
            done
        fi
    fi
    
    log_success "Disk I/O optimizations applied"
}

# Set CPU governor to performance
optimize_cpu() {
    log_info "Optimizing CPU settings..."
    
    if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -f "$cpu" ]; then
                echo performance | sudo tee "$cpu" > /dev/null 2>&1 || true
            fi
        done
        log_success "CPU governor set to performance"
    else
        log_info "CPU frequency scaling not available"
    fi
}

# Apply all optimizations
apply_all_optimizations() {
    optimize_system
    optimize_docker
    optimize_disk_io
    optimize_cpu
    log_success "All performance optimizations applied"
}




