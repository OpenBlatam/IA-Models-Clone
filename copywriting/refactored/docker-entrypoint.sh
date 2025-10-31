#!/bin/bash
# Production Docker Entrypoint Script
# ===================================
# Intelligent startup with optimization detection and health checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Banner
print_banner() {
    echo -e "${PURPLE}"
    echo "============================================================================"
    echo "🚀 PRODUCTION OPTIMIZED COPYWRITING SERVICE"
    echo "============================================================================"
    echo -e "${NC}"
}

# Pre-flight checks
run_preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check Python version
    python_version=$(python --version 2>&1)
    log_info "Python version: $python_version"
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        memory_info=$(free -h | grep "Mem:")
        log_info "Memory: $memory_info"
    fi
    
    # Check disk space
    if command -v df >/dev/null 2>&1; then
        disk_info=$(df -h / | tail -1)
        log_info "Disk space: $disk_info"
    fi
    
    # Check required environment variables
    required_vars=()
    
    # Check AI provider keys
    if [[ -z "${OPENROUTER_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
        log_warn "No AI provider API keys found. At least one is required."
        log_warn "Set OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
    fi
    
    # Check database URL
    if [[ -z "${DATABASE_URL:-}" ]]; then
        log_warn "DATABASE_URL not set. Using default PostgreSQL connection."
        export DATABASE_URL="postgresql://localhost/copywriting"
    fi
    
    # Check Redis URL
    if [[ -z "${REDIS_URL:-}" ]]; then
        log_warn "REDIS_URL not set. Using default Redis connection."
        export REDIS_URL="redis://localhost:6379/0"
    fi
    
    log_info "Pre-flight checks completed"
}

# Optimization detection
check_optimizations() {
    log_info "Checking optimization libraries..."
    
    # Run optimization check
    if python -c "
import sys
sys.path.insert(0, '/app')
try:
    from production_optimized import UltraOptimizationDetector
    detector = UltraOptimizationDetector()
    report = detector.get_optimization_report()
    summary = report['summary']
    print(f'Optimization Score: {summary[\"optimization_score\"]:.1f}/100')
    print(f'Performance Multiplier: {summary[\"performance_multiplier\"]:.1f}x')
    print(f'Available Libraries: {summary[\"available_count\"]}/{summary[\"total_count\"]}')
    if summary['gpu_available']:
        print('GPU Acceleration: AVAILABLE')
    
    # Show missing critical libraries
    if report.get('missing_critical'):
        print('Missing Critical Libraries:')
        for lib in report['missing_critical'][:3]:
            print(f'  - {lib}')
except Exception as e:
    print(f'Optimization check failed: {e}')
    exit(1)
"; then
        log_info "Optimization check completed successfully"
    else
        log_error "Optimization check failed"
        exit 1
    fi
}

# Wait for dependencies
wait_for_dependencies() {
    log_info "Waiting for dependencies..."
    
    # Wait for Redis if configured
    if [[ "${REDIS_URL:-}" =~ redis://([^:]+):?([0-9]+)?/.* ]]; then
        redis_host="${BASH_REMATCH[1]}"
        redis_port="${BASH_REMATCH[2]:-6379}"
        
        if [[ "$redis_host" != "localhost" && "$redis_host" != "127.0.0.1" ]]; then
            log_info "Waiting for Redis at $redis_host:$redis_port..."
            timeout=30
            while ! nc -z "$redis_host" "$redis_port" 2>/dev/null; do
                timeout=$((timeout - 1))
                if [[ $timeout -le 0 ]]; then
                    log_warn "Redis connection timeout. Continuing anyway..."
                    break
                fi
                sleep 1
            done
            
            if [[ $timeout -gt 0 ]]; then
                log_info "Redis is available"
            fi
        fi
    fi
    
    # Wait for PostgreSQL if configured
    if [[ "${DATABASE_URL:-}" =~ postgresql://.*@([^:]+):?([0-9]+)?/.* ]]; then
        db_host="${BASH_REMATCH[1]}"
        db_port="${BASH_REMATCH[2]:-5432}"
        
        if [[ "$db_host" != "localhost" && "$db_host" != "127.0.0.1" ]]; then
            log_info "Waiting for PostgreSQL at $db_host:$db_port..."
            timeout=30
            while ! nc -z "$db_host" "$db_port" 2>/dev/null; do
                timeout=$((timeout - 1))
                if [[ $timeout -le 0 ]]; then
                    log_warn "PostgreSQL connection timeout. Continuing anyway..."
                    break
                fi
                sleep 1
            done
            
            if [[ $timeout -gt 0 ]]; then
                log_info "PostgreSQL is available"
            fi
        fi
    fi
}

# Setup memory optimization
setup_memory_optimization() {
    log_info "Setting up memory optimizations..."
    
    # Use jemalloc if available
    if [[ -f "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2" ]]; then
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
        log_info "jemalloc memory allocator enabled"
    elif [[ -f "/usr/lib/libjemalloc.so" ]]; then
        export LD_PRELOAD="/usr/lib/libjemalloc.so:${LD_PRELOAD:-}"
        log_info "jemalloc memory allocator enabled"
    fi
    
    # Set Python memory optimization
    export PYTHONMALLOC=malloc
    export MALLOC_TRIM_THRESHOLD_=100000
    export MALLOC_MMAP_THRESHOLD_=131072
    
    log_info "Memory optimizations configured"
}

# Performance tuning
setup_performance_tuning() {
    log_info "Applying performance tuning..."
    
    # CPU optimization
    if [[ -n "${CPU_COUNT:-}" ]]; then
        export OMP_NUM_THREADS="$CPU_COUNT"
        export MKL_NUM_THREADS="$CPU_COUNT"
        export OPENBLAS_NUM_THREADS="$CPU_COUNT"
    else
        # Auto-detect CPU count
        cpu_count=$(nproc 2>/dev/null || echo "1")
        export OMP_NUM_THREADS="$cpu_count"
        export MKL_NUM_THREADS="$cpu_count"
        export OPENBLAS_NUM_THREADS="$cpu_count"
    fi
    
    # Network optimization
    export PYTHONHASHSEED=random
    
    # Disable Python bytecode generation in production
    if [[ "${ENVIRONMENT:-}" == "production" ]]; then
        export PYTHONDONTWRITEBYTECODE=1
    fi
    
    log_info "Performance tuning applied"
}

# Health check function
run_health_check() {
    log_info "Running health check..."
    
    # Start service in background for health check
    timeout 30 python run_production.py health > /tmp/health_check.json 2>&1
    
    if [[ $? -eq 0 ]]; then
        log_info "Health check passed"
        if [[ "${DEBUG:-false}" == "true" ]]; then
            cat /tmp/health_check.json
        fi
    else
        log_error "Health check failed"
        cat /tmp/health_check.json
        exit 1
    fi
}

# Graceful shutdown handler
cleanup() {
    log_info "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    # Wait for processes to terminate
    sleep 2
    
    log_info "Cleanup completed"
    exit 0
}

# Setup signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    print_banner
    
    # Change to app directory
    cd /app
    
    # Run startup sequence
    run_preflight_checks
    setup_memory_optimization
    setup_performance_tuning
    wait_for_dependencies
    check_optimizations
    
    # Run health check if not in development mode
    if [[ "${ENVIRONMENT:-}" == "production" ]]; then
        run_health_check
    fi
    
    log_info "Starting application..."
    log_info "Command: $*"
    
    # Execute the command
    exec "$@"
}

# Special handling for different commands
case "${1:-}" in
    "python")
        # Normal Python execution
        main "$@"
        ;;
    "bash"|"sh")
        # Shell access
        exec "$@"
        ;;
    "check")
        # Run optimization check only
        print_banner
        cd /app
        python run_production.py check
        ;;
    "benchmark")
        # Run benchmark only
        print_banner
        cd /app
        python run_production.py benchmark
        ;;
    "health")
        # Run health check only
        print_banner
        cd /app
        python run_production.py health
        ;;
    *)
        # Default to main execution
        main "$@"
        ;;
esac 