# Dependencies Guide for Profiling and Optimization System

A comprehensive guide for managing dependencies for the Onyx Ads Backend profiling and optimization system.

## 📦 Overview

This guide covers all dependencies required for the profiling and optimization system, including:
- **Core dependencies** - Essential packages for basic functionality
- **Development dependencies** - Tools for development and debugging
- **Production dependencies** - Optimized packages for deployment
- **Optional dependencies** - Advanced features and integrations
- **Installation scripts** - Automated setup and configuration

## 🎯 Dependency Categories

### 1. Core Dependencies (`profiling_requirements_minimal.txt`)

Essential packages required for basic profiling functionality:

```bash
# Core PyTorch ecosystem
torch>=2.0.0
torchvision>=0.15.0

# Essential profiling tools
psutil>=5.9.0
memory-profiler>=0.60.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# I/O optimization
aiofiles>=23.0.0

# Configuration and logging
pydantic>=2.0.0
structlog>=23.1.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Web framework
fastapi>=0.100.0
uvicorn>=0.23.0

# Machine learning
transformers>=4.30.0
accelerate>=0.20.0

# Utilities
click>=8.1.0
tqdm>=4.65.0
```

### 2. Development Dependencies (`profiling_requirements_dev.txt`)

Additional packages for development, debugging, and testing:

```bash
# Advanced profiling tools
line-profiler>=4.0.0
pyinstrument>=5.0.0
py-spy>=0.3.0
scalene>=1.5.0

# Development and debugging
ipython>=8.14.0
jupyter>=1.0.0
notebook>=7.0.0
debugpy>=1.6.0
pdbpp>=0.10.0

# Code quality and formatting
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0

# Testing and validation
pytest-cov>=4.1.0
pytest-benchmark>=4.0.0
hypothesis>=6.75.0

# Performance optimization
numba>=0.57.0
cython>=3.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Monitoring and logging
rich>=13.0.0
colorama>=0.4.6
loguru>=0.7.0

# Database and caching
redis>=4.5.0
aioredis>=2.0.0
diskcache>=5.6.0

# Parallel processing
joblib>=1.3.0
concurrent-futures>=3.1.0

# System monitoring
prometheus-client>=0.17.0
statsd>=4.0.0
GPUtil>=1.4.0

# Network and communication
requests>=2.31.0
httpx>=0.24.0
websockets>=11.0.0
aiohttp>=3.8.0

# File formats
pyyaml>=6.0.0
toml>=0.10.0
configparser>=5.3.0
python-dotenv>=1.0.0

# Utilities
tenacity>=8.2.0
retry>=0.9.2

# Security
cryptography>=41.0.0
bcrypt>=4.0.0
passlib>=1.7.0

# Data validation
marshmallow>=3.20.0
cerberus>=1.3.0
jsonschema>=4.17.0
```

### 3. Production Dependencies (`profiling_requirements_production.txt`)

Optimized packages for production deployment:

```bash
# Production-grade profiling
psutil>=5.9.0
memory-profiler>=0.60.0
line-profiler>=4.0.0

# Production monitoring and alerting
prometheus-client>=0.17.0
sentry-sdk>=1.28.0
structlog>=23.1.0
loguru>=0.7.0

# Production web framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
gunicorn>=21.2.0

# Production database and caching
redis>=4.5.0
aioredis>=2.0.0
diskcache>=5.6.0

# Production security
cryptography>=41.0.0
bcrypt>=4.0.0
passlib>=1.7.0
python-jose[cryptography]>=3.3.0

# Production data processing
numpy>=1.24.0
pandas>=2.0.0
dask>=2023.0.0

# Production I/O optimization
aiofiles>=23.0.0
lz4>=4.0.0
h5py>=3.8.0

# Production parallel processing
joblib>=1.3.0
concurrent-futures>=3.1.0

# Production configuration
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Production validation
marshmallow>=3.20.0
jsonschema>=4.17.0

# Production utilities
click>=8.1.0
tqdm>=4.65.0
tenacity>=8.2.0

# Production testing (minimal)
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Production machine learning
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.4.0

# Production network
requests>=2.31.0
httpx>=0.24.0
aiohttp>=3.8.0

# Production file formats
pyyaml>=6.0.0
toml>=0.10.0

# Production performance (optional)
numba>=0.57.0

# Production monitoring (optional)
# google-cloud-monitoring>=2.15.0  # GCP
# boto3>=1.28.0  # AWS
# azure-monitor>=0.1.0  # Azure

# Production GPU support (optional)
# nvidia-ml-py>=12.535.0  # NVIDIA GPUs
```

### 4. Complete Dependencies (`profiling_requirements.txt`)

All dependencies including optional packages:

```bash
# Core PyTorch ecosystem
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0
torch-geometric>=2.3.0

# Profiling and monitoring tools
psutil>=5.9.0
memory-profiler>=0.60.0
line-profiler>=4.0.0
pyinstrument>=5.0.0
py-spy>=0.3.0
scalene>=1.5.0

# Data processing and optimization
numpy>=1.24.0
pandas>=2.0.0
dask>=2023.0.0
vaex>=4.15.0
ray>=2.5.0
modin>=0.20.0

# I/O and storage optimization
aiofiles>=23.0.0
lz4>=4.0.0
zstandard>=0.21.0
snappy>=1.1.0
h5py>=3.8.0
tables>=3.8.0

# Parallel processing and concurrency
joblib>=1.3.0
multiprocessing-logging>=0.3.0
concurrent-futures>=3.1.0
asyncio-mqtt>=0.13.0

# System monitoring and metrics
prometheus-client>=0.17.0
datadog>=0.44.0
statsd>=4.0.0
psutil>=5.9.0
GPUtil>=1.4.0

# Visualization and reporting
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
bokeh>=3.0.0
dash>=2.10.0

# Database and caching
redis>=4.5.0
aioredis>=2.0.0
memcached>=1.59.0
diskcache>=5.6.0

# Configuration and logging
pydantic>=2.0.0
structlog>=23.1.0
rich>=13.0.0
colorama>=0.4.6

# Testing and validation
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-benchmark>=4.0.0
hypothesis>=6.75.0

# Development and debugging
ipython>=8.14.0
jupyter>=1.0.0
notebook>=7.0.0
debugpy>=1.6.0
pdbpp>=0.10.0

# Web framework integration
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0.0
aiohttp>=3.8.0

# Machine learning and AI
transformers>=4.30.0
diffusers>=0.20.0
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.41.0

# Data validation and serialization
marshmallow>=3.20.0
cerberus>=1.3.0
jsonschema>=4.17.0
pydantic>=2.0.0

# Security and authentication
cryptography>=41.0.0
bcrypt>=4.0.0
passlib>=1.7.0

# Network and communication
requests>=2.31.0
httpx>=0.24.0
websocket-client>=1.6.0

# File and data formats
pyyaml>=6.0.0
toml>=0.10.0
configparser>=5.3.0
python-dotenv>=1.0.0

# Performance and optimization
numba>=0.57.0
cython>=3.0.0
mypy>=1.5.0
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0

# System utilities
click>=8.1.0
tqdm>=4.65.0
tenacity>=8.2.0
retry>=0.9.2

# Monitoring and alerting
sentry-sdk>=1.28.0
loguru>=0.7.0
structlog>=23.1.0

# Optional: GPU monitoring (CUDA specific)
# nvidia-ml-py>=12.535.0

# Optional: Advanced profiling
# py-spy>=0.3.0
# scalene>=1.5.0

# Optional: Distributed computing
# dask[distributed]>=2023.0.0
# ray[default]>=2.5.0

# Optional: Cloud monitoring
# google-cloud-monitoring>=2.15.0
# boto3>=1.28.0
# azure-monitor>=0.1.0
```

## 🚀 Installation Methods

### 1. Automated Installation Script

Use the provided installation script for complete setup:

```bash
# Make script executable
chmod +x profiling_install.sh

# Install all dependencies
./profiling_install.sh

# Install only core dependencies
./profiling_install.sh --core-only

# Install only GPU dependencies
./profiling_install.sh --gpu-only

# Install only advanced profiling tools
./profiling_install.sh --advanced-only

# Install only distributed computing tools
./profiling_install.sh --distributed-only

# Install only cloud monitoring tools
./profiling_install.sh --cloud-only

# Show help
./profiling_install.sh --help
```

### 2. Manual Installation

Install dependencies manually based on your needs:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install minimal dependencies
pip install -r profiling_requirements_minimal.txt

# Install development dependencies
pip install -r profiling_requirements_dev.txt

# Install production dependencies
pip install -r profiling_requirements_production.txt

# Install all dependencies
pip install -r profiling_requirements.txt
```

### 3. Conda Installation

For Conda users:

```bash
# Create conda environment
conda create -n profiling-env python=3.9
conda activate profiling-env

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r profiling_requirements_minimal.txt
```

## 🔧 Platform-Specific Installation

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config

# Install profiling dependencies
./profiling_install.sh
```

### CentOS/RHEL

```bash
# Install system dependencies
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    python3-devel \
    python3-pip \
    openssl-devel \
    libffi-devel \
    atlas-devel \
    lapack-devel \
    blas-devel \
    gcc-gfortran

# Install profiling dependencies
./profiling_install.sh
```

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install \
    python3 \
    openssl \
    libffi \
    pkg-config \
    rust

# Install profiling dependencies
./profiling_install.sh
```

### Windows

```bash
# Install Chocolatey if not installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install system dependencies
choco install python3 git rust

# Install profiling dependencies
./profiling_install.sh
```

## 🐳 Docker Installation

### Dockerfile

```dockerfile
# Use official Python image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY profiling_requirements_minimal.txt .
COPY profiling_requirements_production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r profiling_requirements_production.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  profiling-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./profiles:/app/profiles
      - ./cache:/app/cache
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

## 🔍 Dependency Verification

### Verify Installation

```bash
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
```

### Run Tests

```bash
# Run all tests
python -m pytest test_profiling_optimization.py -v

# Run specific test categories
python -m pytest test_profiling_optimization.py::TestProfilingOptimizer -v
python -m pytest test_profiling_optimization.py::TestDataLoadingOptimizer -v
python -m pytest test_profiling_optimization.py::TestIntegrationWithFineTuning -v

# Run with coverage
python -m pytest test_profiling_optimization.py --cov=profiling_optimizer --cov=data_optimization -v
```

## 🚨 Troubleshooting

### Common Issues

1. **PyTorch Installation Issues**
   ```bash
   # For CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Memory Profiler Issues**
   ```bash
   # Install system dependencies
   sudo apt-get install python3-dev
   
   # Reinstall memory-profiler
   pip uninstall memory-profiler
   pip install memory-profiler
   ```

3. **Line Profiler Issues**
   ```bash
   # Install Cython first
   pip install cython
   
   # Reinstall line-profiler
   pip uninstall line-profiler
   pip install line-profiler
   ```

4. **GPU Monitoring Issues**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Install NVIDIA ML Python bindings
   pip install nvidia-ml-py
   ```

5. **Permission Issues**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER venv/
   chmod +x profiling_install.sh
   ```

### Version Conflicts

```bash
# Check for conflicts
pip check

# Resolve conflicts
pip install --upgrade pip
pip install -r profiling_requirements_minimal.txt --force-reinstall

# Use specific versions
pip install torch==2.0.0 torchvision==0.15.0
```

### Environment Issues

```bash
# Create fresh environment
rm -rf venv/
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r profiling_requirements_minimal.txt
```

## 📊 Dependency Management

### Version Pinning

For production deployments, pin specific versions:

```bash
# Generate requirements with exact versions
pip freeze > requirements_frozen.txt

# Install from frozen requirements
pip install -r requirements_frozen.txt
```

### Dependency Updates

```bash
# Check for updates
pip list --outdated

# Update specific packages
pip install --upgrade torch psutil numpy

# Update all packages
pip install --upgrade -r profiling_requirements_minimal.txt
```

### Security Scanning

```bash
# Install safety
pip install safety

# Scan for vulnerabilities
safety check -r profiling_requirements_minimal.txt

# Fix vulnerabilities
safety check -r profiling_requirements_minimal.txt --full-report
```

## 📈 Performance Considerations

### Memory Usage

- **Minimal installation**: ~500MB
- **Development installation**: ~2GB
- **Production installation**: ~1.5GB
- **Complete installation**: ~3GB

### Installation Time

- **Minimal**: 5-10 minutes
- **Development**: 15-30 minutes
- **Production**: 10-20 minutes
- **Complete**: 30-60 minutes

### Disk Space

- **Minimal**: ~1GB
- **Development**: ~3GB
- **Production**: ~2GB
- **Complete**: ~5GB

## 🔒 Security Considerations

### Package Verification

```bash
# Verify package signatures
pip install --require-hashes -r profiling_requirements_minimal.txt

# Use trusted sources
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

### Environment Isolation

```bash
# Use virtual environments
python3 -m venv venv
source venv/bin/activate

# Use conda environments
conda create -n profiling-env python=3.9
conda activate profiling-env
```

### Access Control

```bash
# Restrict package installation
pip install --user -r profiling_requirements_minimal.txt

# Use package managers
sudo apt-get install python3-psutil python3-numpy
```

This comprehensive dependencies guide ensures proper installation and configuration of the profiling and optimization system across different platforms and use cases. 