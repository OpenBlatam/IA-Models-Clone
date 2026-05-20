#!/bin/bash
# Common functions library for EC2 deployment scripts
# Optimized for performance with caching and parallelization
# Source this file in other scripts: source lib/common.sh

# Color codes
readonly COLOR_RESET='\033[0m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_RED='\033[0;31m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_CYAN='\033[0;36m'
readonly COLOR_MAGENTA='\033[0;35m'

# Performance optimizations
readonly CACHE_DIR="${TMPDIR:-/tmp}/music-analyzer-cache"
readonly CACHE_TTL=300  # 5 minutes
readonly PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Initialize cache directory
mkdir -p "$CACHE_DIR" 2>/dev/null || true

# Cache management functions
cache_get() {
    local key="$1"
    local cache_file="$CACHE_DIR/${key}.cache"
    
    if [ -f "$cache_file" ]; then
        local age=$(($(date +%s) - $(stat -c %Y "$cache_file" 2>/dev/null || echo 0)))
        if [ $age -lt $CACHE_TTL ]; then
            cat "$cache_file"
            return 0
        fi
    fi
    return 1
}

cache_set() {
    local key="$1"
    local value="$2"
    local cache_file="$CACHE_DIR/${key}.cache"
    echo "$value" > "$cache_file" 2>/dev/null || true
}

cache_clear() {
    rm -rf "$CACHE_DIR"/*.cache 2>/dev/null || true
}

# Optimized command execution with caching
cached_command() {
    local cache_key="$1"
    shift
    local cmd="$*"
    
    # Try cache first
    if cache_get "$cache_key"; then
        return 0
    fi
    
    # Execute and cache result
    local result
    if result=$($cmd 2>/dev/null); then
        cache_set "$cache_key" "$result"
        echo "$result"
        return 0
    fi
    return 1
}

# Optimized logging with buffering
LOG_BUFFER="${LOG_BUFFER:-}"
LOG_BUFFER_SIZE=100

flush_log_buffer() {
    if [ -n "$LOG_BUFFER" ]; then
        echo -e "$LOG_BUFFER" | tee -a "${LOG_FILE:-/var/log/music-analyzer.log}" >/dev/null
        LOG_BUFFER=""
    fi
}

log_buffered() {
    local message="$1"
    LOG_BUFFER="${LOG_BUFFER}${message}\n"
    
    if [ $(echo -e "$LOG_BUFFER" | wc -l) -ge $LOG_BUFFER_SIZE ]; then
        flush_log_buffer
    fi
}

# Logging functions (optimized)
log_info() {
    local msg="${COLOR_BLUE}ℹ️  $1${COLOR_RESET}"
    echo -e "$msg"
    log_buffered "$msg"
}

log_success() {
    local msg="${COLOR_GREEN}✅ $1${COLOR_RESET}"
    echo -e "$msg"
    log_buffered "$msg"
}

log_warning() {
    local msg="${COLOR_YELLOW}⚠️  $1${COLOR_RESET}"
    echo -e "$msg"
    log_buffered "$msg"
}

log_error() {
    local msg="${COLOR_RED}❌ $1${COLOR_RESET}"
    echo -e "$msg" >&2
    log_buffered "$msg"
    flush_log_buffer  # Always flush on errors
}

log_step() {
    local msg="${COLOR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${COLOR_RESET}\n${COLOR_CYAN}📌 $1${COLOR_RESET}\n${COLOR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${COLOR_RESET}"
    echo -e "$msg"
    log_buffered "$msg"
}

# Flush buffer on exit
trap flush_log_buffer EXIT

# Error handling
error_exit() {
    log_error "$1"
    exit "${2:-1}"
}

# Optimized command existence check with caching
_command_cache=""
command_exists() {
    local cmd="$1"
    
    # Check cache first
    if echo "$_command_cache" | grep -q "^${cmd}:.*$"; then
        echo "$_command_cache" | grep "^${cmd}:" | cut -d: -f2
        return 0
    fi
    
    # Check command
    if command -v "$cmd" >/dev/null 2>&1; then
        _command_cache="${_command_cache}${cmd}:1\n"
        return 0
    else
        _command_cache="${_command_cache}${cmd}:0\n"
        return 1
    fi
}

# Check if running as root or with sudo
check_sudo() {
    if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
        error_exit "This script requires sudo privileges. Please run with sudo."
    fi
}

# Optimized retry with connection pooling and timeout
retry() {
    local max_attempts="${1:-3}"
    local delay="${2:-1}"
    local timeout="${3:-30}"
    local attempt=1
    shift 3
    
    while [ $attempt -le $max_attempts ]; do
        # Use timeout to prevent hanging
        if timeout "$timeout" "$@" 2>/dev/null; then
            return 0
        fi
        
        local exit_code=$?
        
        if [ $attempt -lt $max_attempts ]; then
            if [ $exit_code -eq 124 ]; then
                log_warning "Attempt $attempt/$max_attempts timed out. Retrying in ${delay}s..."
            else
                log_warning "Attempt $attempt/$max_attempts failed. Retrying in ${delay}s..."
            fi
            sleep "$delay"
            delay=$((delay * 2))  # Exponential backoff
        fi
        attempt=$((attempt + 1))
    done
    
    return 1
}

# Parallel execution helper
run_parallel() {
    local max_jobs="${1:-$PARALLEL_JOBS}"
    shift
    local pids=()
    local commands=("$@")
    local total=${#commands[@]}
    local completed=0
    
    for cmd in "${commands[@]}"; do
        # Wait if we've reached max jobs
        while [ ${#pids[@]} -ge $max_jobs ]; do
            for pid in "${pids[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    # Process finished
                    pids=("${pids[@]/$pid}")
                    completed=$((completed + 1))
                fi
            done
            sleep 0.1
        done
        
        # Start new process
        eval "$cmd" &
        pids+=($!)
    done
    
    # Wait for all to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
        completed=$((completed + 1))
    done
}

# Optimized wait for service with connection reuse
wait_for_service() {
    local url="${1}"
    local max_attempts="${2:-30}"
    local delay="${3:-2}"
    local attempt=1
    local curl_opts="-sf --max-time 5 --connect-timeout 2"
    
    log_info "Waiting for service at $url..."
    
    # Use persistent connection if possible
    if command_exists curl; then
        while [ $attempt -le $max_attempts ]; do
            if curl $curl_opts "$url" >/dev/null 2>&1; then
                log_success "Service is ready!"
                return 0
            fi
            
            if [ $((attempt % 5)) -eq 0 ]; then
                log_info "Still waiting... ($attempt/$max_attempts)"
            fi
            
            sleep "$delay"
            attempt=$((attempt + 1))
        done
    else
        log_warning "curl not available, skipping health check"
        return 1
    fi
    
    log_warning "Service did not become ready after $max_attempts attempts"
    return 1
}

# Optimized EC2 metadata with caching
get_instance_metadata() {
    local key="${1}"
    local cache_key="metadata_${key}"
    
    # Try cache first
    if cached_command "$cache_key" get_instance_metadata_uncached "$key"; then
        return 0
    fi
    
    # Fallback
    echo ""
}

get_instance_metadata_uncached() {
    local key="${1}"
    curl -s --max-time 2 --connect-timeout 1 \
         "http://169.254.169.254/latest/meta-data/${key}" 2>/dev/null || echo ""
}

# Get public IP (cached)
get_public_ip() {
    local ip
    ip=$(cache_get "public_ip" || get_instance_metadata_uncached "public-ipv4")
    if [ -z "$ip" ]; then
        ip="localhost"
    fi
    cache_set "public_ip" "$ip"
    echo "$ip"
}

# Get instance ID (cached)
get_instance_id() {
    local id
    id=$(cache_get "instance_id" || get_instance_metadata_uncached "instance-id")
    if [ -z "$id" ]; then
        id="unknown"
    fi
    cache_set "instance_id" "$id"
    echo "$id"
}

# Check system resources
# Optimized resource checks with parallel execution
check_memory() {
    local min_gb="${1:-2}"
    local total_mem
    
    # Use cached value if available
    if ! total_mem=$(cache_get "total_memory"); then
        if command_exists free; then
            total_mem=$(free -g | awk '/^Mem:/{print $2}')
            cache_set "total_memory" "$total_mem"
        else
            log_warning "Cannot check memory (free command not available)"
            return 0
        fi
    fi
    
    if [ "$total_mem" -lt "$min_gb" ]; then
        log_warning "System has ${total_mem}GB RAM. Recommended: ${min_gb}GB+"
        return 1
    else
        log_success "Memory: ${total_mem}GB"
        return 0
    fi
}

check_disk_space() {
    local min_gb="${1:-10}"
    local free_space
    
    # Use cached value if available
    if ! free_space=$(cache_get "free_disk"); then
        if command_exists df; then
            free_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
            cache_set "free_disk" "$free_space"
        else
            log_warning "Cannot check disk space (df command not available)"
            return 0
        fi
    fi
    
    if [ "$free_space" -lt "$min_gb" ]; then
        error_exit "Insufficient disk space. Need at least ${min_gb}GB free, have ${free_space}GB"
    else
        log_success "Disk space: ${free_space}GB free"
        return 0
    fi
}

check_cpu() {
    local cpu_count
    if ! cpu_count=$(cache_get "cpu_count"); then
        if command_exists nproc; then
            cpu_count=$(nproc)
            cache_set "cpu_count" "$cpu_count"
        else
            cpu_count="unknown"
        fi
    fi
    log_success "CPU cores: ${cpu_count}"
}

check_resources() {
    log_step "Checking System Resources"
    
    # Run checks in parallel for better performance
    check_memory 2 &
    local mem_pid=$!
    
    check_disk_space 10 &
    local disk_pid=$!
    
    check_cpu &
    local cpu_pid=$!
    
    # Wait for all checks
    wait $mem_pid
    wait $disk_pid
    wait $cpu_pid
}

# OS detection
detect_os() {
    local os_file="/etc/os-release"
    
    if [ -f "$os_file" ]; then
        . "$os_file"
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
        OS_NAME="$NAME"
    elif [ -f "/etc/redhat-release" ]; then
        OS_ID=$(cat /etc/redhat-release | awk '{print $1}' | tr '[:upper:]' '[:lower:]')
        OS_VERSION=$(cat /etc/redhat-release | sed 's/.*release \([0-9.]*\).*/\1/')
        OS_NAME=$(cat /etc/redhat-release)
    elif [ -f "/etc/debian_version" ]; then
        OS_ID="debian"
        OS_VERSION=$(cat /etc/debian_version)
        OS_NAME="Debian $OS_VERSION"
    else
        OS_ID=$(uname -s | tr '[:upper:]' '[:lower:]')
        OS_VERSION=$(uname -r)
        OS_NAME="$OS_ID $OS_VERSION"
    fi
    
    # Determine default user
    case "$OS_ID" in
        amzn|amazon|rhel|centos|fedora)
            DEFAULT_USER="ec2-user"
            ;;
        ubuntu|debian)
            DEFAULT_USER="ubuntu"
            ;;
        *)
            DEFAULT_USER=$(whoami)
            ;;
    esac
    
    export OS_ID OS_VERSION OS_NAME DEFAULT_USER
}

# Docker functions
docker_is_running() {
    docker info >/dev/null 2>&1
}

docker_compose_cmd() {
    if command_exists docker-compose; then
        echo "docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    else
        return 1
    fi
}

# Validate deployment
validate_deployment() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    local health_url="${2:-http://localhost:8010/health}"
    
    log_step "Validating Deployment"
    
    # Check if app directory exists
    if [ ! -d "$app_dir" ]; then
        error_exit "Application directory not found: $app_dir"
    fi
    
    # Check Docker Compose file
    if [ ! -f "$app_dir/deployment/docker-compose.prod.yml" ]; then
        error_exit "Docker Compose file not found"
    fi
    
    # Check health endpoint
    if wait_for_service "$health_url" 30 2; then
        log_success "Deployment validation passed"
        return 0
    else
        log_warning "Health check failed, but deployment may still be starting"
        return 1
    fi
}

# Print deployment info
print_deployment_info() {
    local instance_ip=$(get_public_ip)
    local instance_id=$(get_instance_id)
    
    log_step "Deployment Information"
    echo ""
    log_success "Instance ID: $instance_id"
    log_success "Public IP: $instance_ip"
    echo ""
    log_info "Service URLs:"
    echo "  🌐 API:          http://$instance_ip:8010"
    echo "  ❤️  Health:       http://$instance_ip:8010/health"
    echo "  📖 Docs:          http://$instance_ip:8010/docs"
    echo "  📈 Grafana:       http://$instance_ip:3000"
    echo "  📊 Prometheus:    http://$instance_ip:9090"
    echo ""
}

# Print useful commands
print_commands() {
    local app_dir="${1:-/opt/music-analyzer-ai}"
    
    log_info "Useful Commands:"
    echo "  View logs:    cd $app_dir && docker-compose -f deployment/docker-compose.prod.yml logs -f"
    echo "  Stop:         cd $app_dir && ./stop.sh"
    echo "  Restart:      cd $app_dir && ./start.sh"
    echo "  Status:       cd $app_dir && docker-compose -f deployment/docker-compose.prod.yml ps"
    echo "  Monitor:      ./deployment/ec2/monitor.sh"
    echo "  Update:       ./deployment/ec2/update.sh"
    echo ""
}

