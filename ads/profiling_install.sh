#!/bin/bash

# Profiling and Optimization System Installation Script
# This script installs all dependencies for the Onyx Ads Backend profiling system

set -e  # Exit on any error

echo "🚀 Installing Profiling and Optimization System Dependencies"
echo "=========================================================="

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

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python not found. Please install Python 3.8+ first."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
        print_success "pip3 found"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
        print_success "pip found"
    else
        print_error "pip not found. Please install pip first."
        exit 1
    fi
}

# Check if virtual environment exists
check_venv() {
    print_status "Checking virtual environment..."
    if [ -d "venv" ]; then
        print_success "Virtual environment found"
        source venv/bin/activate
    else
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv venv
        source venv/bin/activate
        print_success "Virtual environment created and activated"
    fi
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    $PIP_CMD install --upgrade pip
    print_success "pip upgraded"
}

# Install core dependencies
install_core_deps() {
    print_status "Installing core dependencies..."
    
    # Core PyTorch ecosystem
    print_status "Installing PyTorch ecosystem..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Core profiling tools
    print_status "Installing profiling tools..."
    $PIP_CMD install psutil memory-profiler line-profiler pyinstrument
    
    # Data processing
    print_status "Installing data processing tools..."
    $PIP_CMD install numpy pandas dask vaex
    
    # I/O optimization
    print_status "Installing I/O optimization tools..."
    $PIP_CMD install aiofiles lz4 h5py tables
    
    print_success "Core dependencies installed"
}

# Install optional dependencies
install_optional_deps() {
    print_status "Installing optional dependencies..."
    
    # Parallel processing
    $PIP_CMD install joblib concurrent-futures
    
    # System monitoring
    $PIP_CMD install prometheus-client statsd GPUtil
    
    # Visualization
    $PIP_CMD install matplotlib seaborn plotly
    
    # Database and caching
    $PIP_CMD install redis aioredis diskcache
    
    # Configuration and logging
    $PIP_CMD install pydantic structlog rich colorama
    
    # Testing
    $PIP_CMD install pytest pytest-asyncio pytest-cov pytest-benchmark
    
    # Development tools
    $PIP_CMD install ipython jupyter notebook debugpy
    
    # Web framework
    $PIP_CMD install fastapi uvicorn websockets aiohttp
    
    # Machine learning
    $PIP_CMD install transformers diffusers accelerate peft
    
    # Data validation
    $PIP_CMD install marshmallow cerberus jsonschema
    
    # Security
    $PIP_CMD install cryptography bcrypt passlib
    
    # Network
    $PIP_CMD install requests httpx websocket-client
    
    # File formats
    $PIP_CMD install pyyaml toml configparser python-dotenv
    
    # Performance
    $PIP_CMD install numba cython mypy black isort flake8
    
    # Utilities
    $PIP_CMD install click tqdm tenacity retry
    
    # Monitoring
    $PIP_CMD install sentry-sdk loguru
    
    print_success "Optional dependencies installed"
}

# Install GPU-specific dependencies
install_gpu_deps() {
    print_status "Checking GPU support..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        print_status "Installing GPU-specific dependencies..."
        $PIP_CMD install nvidia-ml-py
        print_success "GPU dependencies installed"
    else
        print_warning "No NVIDIA GPU detected. Skipping GPU-specific dependencies."
    fi
}

# Install advanced profiling tools
install_advanced_profiling() {
    print_status "Installing advanced profiling tools..."
    
    # Check if we can install py-spy (requires Rust)
    if command -v cargo &> /dev/null; then
        $PIP_CMD install py-spy
        print_success "py-spy installed"
    else
        print_warning "Rust not found. Skipping py-spy installation."
        print_status "To install py-spy, install Rust first: https://rustup.rs/"
    fi
    
    # Install scalene
    $PIP_CMD install scalene
    print_success "scalene installed"
}

# Install distributed computing tools
install_distributed_tools() {
    print_status "Installing distributed computing tools..."
    
    # Dask distributed
    $PIP_CMD install "dask[distributed]"
    
    # Ray
    $PIP_CMD install "ray[default]"
    
    print_success "Distributed computing tools installed"
}

# Install cloud monitoring (optional)
install_cloud_monitoring() {
    print_status "Installing cloud monitoring tools..."
    
    # Google Cloud
    $PIP_CMD install google-cloud-monitoring
    
    # AWS
    $PIP_CMD install boto3
    
    # Azure
    $PIP_CMD install azure-monitor
    
    print_success "Cloud monitoring tools installed"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test core imports
    python3 -c "
import torch
import psutil
import numpy as np
import pandas as pd
import fastapi
import pytest
print('✅ Core dependencies verified')
"
    
    # Test profiling imports
    python3 -c "
import memory_profiler
import line_profiler
import pyinstrument
print('✅ Profiling tools verified')
"
    
    # Test optimization imports
    python3 -c "
import dask
import vaex
import ray
print('✅ Optimization tools verified')
"
    
    print_success "Installation verification completed"
}

# Create configuration files
create_config_files() {
    print_status "Creating configuration files..."
    
    # Create profiling config
    cat > profiling_config.yaml << EOF
# Profiling Configuration
profiling:
  enabled: true
  profile_cpu: true
  profile_memory: true
  profile_gpu: true
  profile_data_loading: true
  profile_preprocessing: true
  profile_depth: 10
  min_time_threshold: 0.001
  min_memory_threshold: 1048576
  save_profiles: true
  profile_dir: "profiles"
  auto_optimize: true
  optimization_threshold: 0.1
  real_time_monitoring: true
  alert_threshold: 5.0
  monitoring_interval: 1.0

data_optimization:
  optimize_loading: true
  prefetch_factor: 2
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  memory_efficient: true
  max_memory_usage: 0.8
  chunk_size: 1000
  enable_caching: true
  cache_dir: "cache"
  cache_size: 1000
  optimize_preprocessing: true
  batch_preprocessing: true
  parallel_preprocessing: true
  preprocessing_workers: 2
  optimize_io: true
  compression: "gzip"
  buffer_size: 8192
  async_io: true
  monitor_performance: true
  log_metrics: true
  alert_threshold: 5.0
EOF
    
    print_success "Configuration files created"
}

# Main installation function
main() {
    echo "Starting installation process..."
    
    check_python
    check_pip
    check_venv
    upgrade_pip
    install_core_deps
    install_optional_deps
    install_gpu_deps
    install_advanced_profiling
    install_distributed_tools
    install_cloud_monitoring
    verify_installation
    create_config_files
    
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run tests: python -m pytest test_profiling_optimization.py -v"
    echo "3. Start profiling: python -c \"from profiling_optimizer import ProfilingOptimizer; print('Ready!')\""
    echo ""
    echo "Documentation: PROFILING_OPTIMIZATION_GUIDE.md"
    echo "Configuration: profiling_config.yaml"
}

# Handle command line arguments
case "${1:-}" in
    --core-only)
        check_python
        check_pip
        check_venv
        upgrade_pip
        install_core_deps
        verify_installation
        ;;
    --gpu-only)
        install_gpu_deps
        ;;
    --advanced-only)
        install_advanced_profiling
        ;;
    --distributed-only)
        install_distributed_tools
        ;;
    --cloud-only)
        install_cloud_monitoring
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --core-only        Install only core dependencies"
        echo "  --gpu-only         Install only GPU-specific dependencies"
        echo "  --advanced-only    Install only advanced profiling tools"
        echo "  --distributed-only Install only distributed computing tools"
        echo "  --cloud-only       Install only cloud monitoring tools"
        echo "  --help, -h         Show this help message"
        echo ""
        echo "Default: Install all dependencies"
        exit 0
        ;;
    *)
        main
        ;;
esac 