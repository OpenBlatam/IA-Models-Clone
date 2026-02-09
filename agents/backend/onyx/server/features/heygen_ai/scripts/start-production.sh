#!/bin/bash

# =============================================================================
# Production Startup Script for Next-Level HeyGen AI FastAPI
# Handles startup, health checks, monitoring, and graceful shutdown
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/.."
LOG_DIR="${APP_DIR}/logs"
PID_FILE="${APP_DIR}/app.pid"
HEALTH_CHECK_TIMEOUT=30
MAX_STARTUP_TIME=120

# Environment defaults
ENVIRONMENT="${ENVIRONMENT:-production}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-info}"
OPTIMIZATION_TIER="${OPTIMIZATION_TIER:-3}"
RELOAD="${RELOAD:-false}"

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# =============================================================================
# Pre-startup Checks
# =============================================================================

check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check required environment variables
    if [[ -z "${SECRET_KEY:-}" ]]; then
        error "SECRET_KEY environment variable is not set"
        exit 1
    fi
    
    if [[ -z "${JWT_SECRET:-}" ]]; then
        error "JWT_SECRET environment variable is not set"
        exit 1
    fi
    
    # Check if running in container
    if [[ -f /.dockerenv ]]; then
        info "Running in Docker container"
    else
        info "Running on host system"
    fi
    
    log "✅ Dependencies check passed"
}

check_system_resources() {
    log "Checking system resources..."
    
    # Check available memory (minimum 2GB recommended)
    AVAILABLE_MEMORY=$(python -c "import psutil; print(psutil.virtual_memory().available // (1024**3))")
    if [[ $AVAILABLE_MEMORY -lt 2 ]]; then
        warn "Available memory is ${AVAILABLE_MEMORY}GB, recommended minimum is 2GB"
    else
        log "✅ Available memory: ${AVAILABLE_MEMORY}GB"
    fi
    
    # Check available disk space (minimum 5GB recommended)
    AVAILABLE_DISK=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $AVAILABLE_DISK -lt 5 ]]; then
        warn "Available disk space is ${AVAILABLE_DISK}GB, recommended minimum is 5GB"
    else
        log "✅ Available disk space: ${AVAILABLE_DISK}GB"
    fi
    
    # Check CPU cores
    CPU_CORES=$(python -c "import multiprocessing; print(multiprocessing.cpu_count())")
    log "✅ CPU cores available: ${CPU_CORES}"
    
    # Adjust workers based on CPU cores if not explicitly set
    if [[ "${WORKERS}" == "4" && $CPU_CORES -lt 4 ]]; then
        WORKERS=$CPU_CORES
        warn "Adjusted workers to ${WORKERS} based on available CPU cores"
    fi
}

setup_directories() {
    log "Setting up directories..."
    
    # Create necessary directories
    mkdir -p "${LOG_DIR}"
    mkdir -p "${APP_DIR}/outputs/videos"
    mkdir -p "${APP_DIR}/outputs/images"
    mkdir -p "${APP_DIR}/cache"
    mkdir -p "${APP_DIR}/temp"
    mkdir -p "${APP_DIR}/monitoring"
    
    # Set permissions
    chmod 755 "${LOG_DIR}"
    chmod 755 "${APP_DIR}/outputs"
    chmod 755 "${APP_DIR}/cache"
    chmod 755 "${APP_DIR}/temp"
    
    log "✅ Directories setup completed"
}

check_external_services() {
    log "Checking external service connections..."
    
    # Check Redis connection if enabled
    if [[ "${ENABLE_REDIS:-true}" == "true" ]]; then
        REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
        if python -c "
import sys
try:
    import redis
    r = redis.from_url('${REDIS_URL}')
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
            log "✅ Redis connection successful"
        else
            warn "Redis connection failed, caching will be limited"
        fi
    fi
    
    # Check database connection if configured
    if [[ -n "${DATABASE_URL:-}" ]]; then
        if python -c "
import sys
try:
    import asyncio
    import asyncpg
    
    async def check_db():
        conn = await asyncpg.connect('${DATABASE_URL}')
        await conn.close()
        print('Database connection successful')
    
    asyncio.run(check_db())
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
            log "✅ Database connection successful"
        else
            warn "Database connection failed, using fallback configuration"
        fi
    fi
}

# =============================================================================
# Application Startup
# =============================================================================

start_application() {
    log "Starting Next-Level HeyGen AI FastAPI application..."
    
    # Change to application directory
    cd "${APP_DIR}"
    
    # Set Python path
    export PYTHONPATH="${APP_DIR}:${PYTHONPATH:-}"
    
    # Determine startup command based on environment
    if [[ "${ENVIRONMENT}" == "production" ]]; then
        # Production: Use Gunicorn with Uvicorn workers
        STARTUP_CMD="gunicorn main_next_level:app \
            --worker-class uvicorn.workers.UvicornWorker \
            --workers ${WORKERS} \
            --bind ${HOST}:${PORT} \
            --log-level ${LOG_LEVEL} \
            --access-logfile ${LOG_DIR}/access.log \
            --error-logfile ${LOG_DIR}/error.log \
            --capture-output \
            --enable-stdio-inheritance \
            --preload \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --timeout 120 \
            --keep-alive 2 \
            --pid ${PID_FILE}"
    else
        # Development: Use Uvicorn directly
        STARTUP_CMD="uvicorn main_next_level:app \
            --host ${HOST} \
            --port ${PORT} \
            --log-level ${LOG_LEVEL} \
            --access-log \
            --reload=${RELOAD}"
    fi
    
    # Add loop configuration for Unix systems
    if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "win32" ]]; then
        STARTUP_CMD="${STARTUP_CMD} --loop uvloop"
    fi
    
    log "Startup command: ${STARTUP_CMD}"
    log "Environment: ${ENVIRONMENT}"
    log "Optimization tier: ${OPTIMIZATION_TIER}"
    log "Host: ${HOST}:${PORT}"
    log "Workers: ${WORKERS}"
    log "Log level: ${LOG_LEVEL}"
    
    # Start the application
    exec ${STARTUP_CMD}
}

# =============================================================================
# Health Check Functions
# =============================================================================

wait_for_startup() {
    log "Waiting for application to start..."
    
    local count=0
    local max_attempts=$((MAX_STARTUP_TIME / 5))
    
    while [[ $count -lt $max_attempts ]]; do
        if curl -f -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
            log "✅ Application started successfully"
            return 0
        fi
        
        count=$((count + 1))
        info "Waiting for startup... (${count}/${max_attempts})"
        sleep 5
    done
    
    error "Application failed to start within ${MAX_STARTUP_TIME} seconds"
    return 1
}

# =============================================================================
# Signal Handlers
# =============================================================================

cleanup() {
    log "Received shutdown signal, performing cleanup..."
    
    # Kill application if PID file exists
    if [[ -f "${PID_FILE}" ]]; then
        local pid=$(cat "${PID_FILE}")
        if kill -0 "$pid" 2>/dev/null; then
            log "Sending SIGTERM to application (PID: ${pid})"
            kill -TERM "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [[ $count -lt 30 ]]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                warn "Forcing application shutdown"
                kill -KILL "$pid"
            fi
        fi
        
        rm -f "${PID_FILE}"
    fi
    
    log "Cleanup completed"
    exit 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log "🚀 Starting Next-Level HeyGen AI FastAPI Production Server"
    log "=================================================="
    
    # Set up signal handlers
    trap cleanup SIGTERM SIGINT
    
    # Pre-startup checks
    check_dependencies
    check_system_resources
    setup_directories
    check_external_services
    
    # Display configuration
    info "Configuration Summary:"
    info "  Environment: ${ENVIRONMENT}"
    info "  Optimization Tier: ${OPTIMIZATION_TIER}"
    info "  Host: ${HOST}:${PORT}"
    info "  Workers: ${WORKERS}"
    info "  Log Level: ${LOG_LEVEL}"
    info "  GPU Optimization: ${ENABLE_GPU_OPTIMIZATION:-true}"
    info "  Redis Enabled: ${ENABLE_REDIS:-true}"
    info "  Request Batching: ${ENABLE_REQUEST_BATCHING:-true}"
    info "  Performance Profiling: ${ENABLE_PERFORMANCE_PROFILING:-true}"
    
    log "✅ Pre-startup checks completed successfully"
    log "=================================================="
    
    # Start the application
    start_application
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 