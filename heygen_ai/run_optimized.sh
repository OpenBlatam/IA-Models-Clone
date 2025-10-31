#!/bin/bash

# =============================================================================
# HeyGen AI FastAPI - Optimized Startup Script
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        REQUIRED_VERSION="3.9"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
        else
            print_error "Python $PYTHON_VERSION found, but $REQUIRED_VERSION or higher is required"
            exit 1
        fi
    else
        print_error "Python 3.9+ is required but not installed"
        exit 1
    fi
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment activation script not found"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [ -f "requirements-optimized.txt" ]; then
        pip install -r requirements-optimized.txt
        print_success "Dependencies installed from requirements-optimized.txt"
    elif [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_error "No requirements file found"
        exit 1
    fi
}

# Function to check environment variables
check_environment() {
    print_status "Checking environment configuration..."
    
    # Set default values if not provided
    export ENVIRONMENT=${ENVIRONMENT:-"development"}
    export HOST=${HOST:-"0.0.0.0"}
    export PORT=${PORT:-"8000"}
    export WORKERS=${WORKERS:-"1"}
    export LOG_LEVEL=${LOG_LEVEL:-"info"}
    
    print_success "Environment: $ENVIRONMENT"
    print_success "Host: $HOST"
    print_success "Port: $PORT"
    print_success "Workers: $WORKERS"
    print_success "Log Level: $LOG_LEVEL"
    
    # Check for required environment variables based on environment
    if [ "$ENVIRONMENT" = "production" ]; then
        if [ -z "$DATABASE_URL" ]; then
            print_warning "DATABASE_URL not set for production environment"
        fi
        if [ -z "$REDIS_URL" ]; then
            print_warning "REDIS_URL not set for production environment"
        fi
        if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "your-secret-key-change-in-production" ]; then
            print_warning "SECRET_KEY should be changed for production environment"
        fi
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p outputs/videos
    mkdir -p temp
    mkdir -p cache
    
    print_success "Directories created"
}

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check available memory
    if command_exists free; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 2 ]; then
            print_warning "Low memory detected: ${MEMORY_GB}GB. Consider using basic optimization level."
        else
            print_success "Memory: ${MEMORY_GB}GB available"
        fi
    fi
    
    # Check available disk space
    if command_exists df; then
        DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
        if [ "$DISK_GB" -lt 5 ]; then
            print_warning "Low disk space: ${DISK_GB}GB available"
        else
            print_success "Disk space: ${DISK_GB}GB available"
        fi
    fi
}

# Function to start the server
start_server() {
    print_status "Starting optimized HeyGen AI FastAPI server..."
    
    # Determine which startup script to use
    if [ -f "start_optimized.py" ]; then
        STARTUP_SCRIPT="start_optimized.py"
        print_success "Using optimized startup script"
    elif [ -f "main_optimized.py" ]; then
        STARTUP_SCRIPT="main_optimized.py"
        print_success "Using optimized main script"
    elif [ -f "main.py" ]; then
        STARTUP_SCRIPT="main.py"
        print_warning "Using standard main script (optimized version not found)"
    else
        print_error "No startup script found"
        exit 1
    fi
    
    # Start the server
    python3 "$STARTUP_SCRIPT" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
}

# Function to show help
show_help() {
    echo "HeyGen AI FastAPI - Optimized Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -e, --environment   Set environment (development, staging, production)"
    echo "  -H, --host          Set host (default: 0.0.0.0)"
    echo "  -p, --port          Set port (default: 8000)"
    echo "  -w, --workers       Set number of workers (default: 1)"
    echo "  -l, --log-level     Set log level (debug, info, warning, error)"
    echo "  --skip-install      Skip dependency installation"
    echo "  --skip-check        Skip system checks"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT         Application environment"
    echo "  HOST               Server host"
    echo "  PORT               Server port"
    echo "  WORKERS            Number of worker processes"
    echo "  LOG_LEVEL          Logging level"
    echo "  DATABASE_URL       Database connection URL"
    echo "  REDIS_URL          Redis connection URL"
    echo "  SECRET_KEY         Application secret key"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Start with defaults"
    echo "  $0 -e production -p 8080 -w 4        # Start production with 4 workers"
    echo "  $0 --environment development --reload # Start development with reload"
}

# Parse command line arguments
SKIP_INSTALL=false
SKIP_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            export ENVIRONMENT="$2"
            shift 2
            ;;
        -H|--host)
            export HOST="$2"
            shift 2
            ;;
        -p|--port)
            export PORT="$2"
            shift 2
            ;;
        -w|--workers)
            export WORKERS="$2"
            shift 2
            ;;
        -l|--log-level)
            export LOG_LEVEL="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "=============================================================================="
    echo "                    HeyGen AI FastAPI - Optimized Startup"
    echo "=============================================================================="
    echo ""
    
    # System checks
    if [ "$SKIP_CHECK" = false ]; then
        check_python_version
        check_system_resources
    fi
    
    # Environment setup
    check_environment
    create_directories
    
    # Virtual environment setup
    check_venv
    activate_venv
    
    # Install dependencies
    if [ "$SKIP_INSTALL" = false ]; then
        install_dependencies
    fi
    
    # Start server
    start_server
}

# Run main function
main "$@" 