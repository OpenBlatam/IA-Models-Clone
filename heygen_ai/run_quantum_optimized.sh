#!/bin/bash

# Quantum-Optimized HeyGen AI FastAPI Runner
# Advanced GPU utilization and mixed precision training

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_NAME="quantum_venv"
PYTHON_VERSION="3.11"
PORT=8000
HOST="0.0.0.0"
WORKERS=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION_ACTUAL=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Python version: $PYTHON_VERSION_ACTUAL"
    else
        log_error "Python3 is not installed"
        exit 1
    fi
    
    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        log_warning "NVIDIA GPU not detected - will use CPU"
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    log_info "Total system memory: ${TOTAL_MEM}GB"
    
    if [ "$TOTAL_MEM" -lt 8 ]; then
        log_warning "Low memory system detected (< 8GB)"
    fi
}

# Setup virtual environment
setup_virtual_environment() {
    log_info "Setting up virtual environment..."
    
    if [ ! -d "$VENV_NAME" ]; then
        log_info "Creating virtual environment: $VENV_NAME"
        python3 -m venv "$VENV_NAME"
    else
        log_info "Virtual environment already exists: $VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    log_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install quantum-level requirements
    if [ -f "requirements-quantum.txt" ]; then
        log_info "Installing quantum-level dependencies..."
        pip install -r requirements-quantum.txt
    else
        log_error "requirements-quantum.txt not found"
        exit 1
    fi
    
    # Install additional optimization libraries
    log_info "Installing additional optimization libraries..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers diffusers accelerate bitsandbytes
    
    log_success "Dependencies installed successfully"
}

# Configure environment
configure_environment() {
    log_info "Configuring environment..."
    
    # Set environment variables for optimization
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export TOKENIZERS_PARALLELISM=false
    
    # Set PyTorch optimization flags
    export TORCH_CUDNN_V8_API_ENABLED=1
    export TORCH_CUDNN_V8_API_DISABLED=0
    
    # Set memory optimization
    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    
    log_success "Environment configured"
}

# Run pre-flight checks
run_preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check if main application exists
    if [ ! -f "main_quantum_optimized.py" ]; then
        log_error "main_quantum_optimized.py not found"
        exit 1
    fi
    
    # Check if optimization modules exist
    if [ ! -d "api/optimization" ]; then
        log_error "api/optimization directory not found"
        exit 1
    fi
    
    # Check GPU memory
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        log_info "Available GPU memory: ${GPU_MEM}MB"
        
        if [ "$GPU_MEM" -lt 4000 ]; then
            log_warning "Low GPU memory detected (< 4GB)"
        fi
    fi
    
    log_success "Pre-flight checks completed"
}

# Start the application
start_application() {
    log_info "Starting Quantum-Optimized HeyGen AI..."
    
    # Set optimization flags
    export QUANTUM_OPTIMIZATION_ENABLED=1
    export MIXED_PRECISION_TRAINING=1
    export GPU_OPTIMIZATION_LEVEL=quantum
    
    # Start with uvicorn
    exec uvicorn main_quantum_optimized:fastapi_application \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info \
        --access-log \
        --use-colors \
        --reload-dir . \
        --reload-dir api \
        --reload-dir api/optimization
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Deactivate virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    # Clear GPU memory
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi
    
    log_success "Cleanup completed"
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Interrupted by user"; exit 1' INT TERM

# Main execution
main() {
    log_info "🚀 Starting Quantum-Optimized HeyGen AI Setup"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Run setup steps
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    configure_environment
    run_preflight_checks
    
    log_success "✅ Setup completed successfully"
    log_info "🌐 Starting application on http://$HOST:$PORT"
    
    # Start the application
    start_application
}

# Run main function
main "$@" 