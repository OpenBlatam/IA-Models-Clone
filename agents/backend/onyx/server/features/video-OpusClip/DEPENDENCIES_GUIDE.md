# Dependencies Guide for Video-OpusClip

This guide provides comprehensive information about managing dependencies for the Video-OpusClip system, including installation, configuration, and troubleshooting.

## Table of Contents

1. [Overview](#overview)
2. [Installation Options](#installation-options)
3. [Dependency Categories](#dependency-categories)
4. [Installation Scripts](#installation-scripts)
5. [Environment Setup](#environment-setup)
6. [Platform-Specific Notes](#platform-specific-notes)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Overview

The Video-OpusClip system has comprehensive dependencies covering:

- **Core ML/AI**: PyTorch, Transformers, Diffusers
- **Video Processing**: OpenCV, MoviePy, FFmpeg
- **Profiling**: Line profiler, Memory profiler, GPU monitoring
- **Web Interface**: Gradio, FastAPI
- **Optimization**: Mixed precision, Multi-GPU, Distributed training
- **Development**: Testing, Code quality, Documentation

### Requirements Files

- **`requirements_basic.txt`**: Essential dependencies for basic functionality
- **`requirements_complete.txt`**: Full dependencies with all features
- **`install_dependencies.py`**: Automated installation script

## Installation Options

### 1. Basic Installation

For users who want to get started quickly with core functionality:

```bash
# Using the installation script
python install_dependencies.py --type basic

# Manual installation
pip install -r requirements_basic.txt
```

**Includes:**
- PyTorch ecosystem
- Basic video processing
- Gradio interface
- Essential utilities

### 2. Full Installation

For users who want all features:

```bash
# Using the installation script
python install_dependencies.py --type full

# Manual installation
pip install -r requirements_complete.txt
```

**Includes:**
- All basic dependencies
- Profiling tools
- Mixed precision training
- Multi-GPU support
- Advanced ML tools
- Cloud integration

### 3. GPU-Optimized Installation

For users with GPU hardware:

```bash
# Using the installation script
python install_dependencies.py --type gpu

# Manual installation
pip install -r requirements_complete.txt[gpu]
```

**Includes:**
- CUDA-enabled PyTorch
- GPU monitoring tools
- Mixed precision training
- Multi-GPU support

### 4. Development Installation

For developers:

```bash
# Using the installation script
python install_dependencies.py --type dev

# Manual installation
pip install -r requirements_complete.txt[dev]
```

**Includes:**
- All full dependencies
- Testing frameworks
- Code quality tools
- Documentation generators

### 5. Production Installation

For production deployments:

```bash
# Using the installation script
python install_dependencies.py --type prod

# Manual installation
pip install -r requirements_complete.txt[prod]
```

**Includes:**
- All full dependencies
- Production web server
- Performance optimizations
- Monitoring tools

## Dependency Categories

### 1. Core Dependencies

#### PyTorch Ecosystem
```txt
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0
torch-geometric>=2.3.0
```

#### Deep Learning and AI
```txt
transformers>=4.30.0
diffusers>=0.18.0
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.41.0
```

#### Computer Vision
```txt
opencv-python>=4.8.0
Pillow>=9.5.0
imageio>=2.31.0
scikit-image>=0.21.0
albumentations>=1.3.0
```

### 2. Video Processing Dependencies

#### Video Libraries
```txt
moviepy>=1.0.3
ffmpeg-python>=0.2.0
pytube>=15.0.0
yt-dlp>=2023.7.6
decord>=0.6.0
av>=10.0.0
```

#### Video Analysis
```txt
mediapipe>=0.10.0
face-recognition>=1.3.0
dlib>=19.24.0
opencv-contrib-python>=4.8.0
```

### 3. Profiling and Monitoring Dependencies

#### Code Profiling
```txt
line-profiler>=4.1.0
memory-profiler>=0.61.0
psutil>=5.9.0
py-spy>=0.3.14
pyinstrument>=4.6.0
```

#### Performance Monitoring
```txt
structlog>=23.1.0
prometheus-client>=0.17.0
grafana-api>=1.0.3
influxdb-client>=1.36.0
```

#### GPU Monitoring
```txt
GPUtil>=1.4.0
nvidia-ml-py3>=7.352.0
pynvml>=11.5.0
```

### 4. Optimization Dependencies

#### Mixed Precision Training
```txt
apex>=0.1.0
amp>=0.1.0
pytorch-memlab>=0.2.4
```

#### Multi-GPU and Distributed Training
```txt
torch-distributed>=2.0.0
horovod>=0.28.0
deepspeed>=0.9.0
fairscale>=0.4.13
ray>=2.6.0
```

### 5. Web Interface Dependencies

#### Gradio Interface
```txt
gradio>=3.40.0
gradio-client>=0.6.0
```

#### Web Framework
```txt
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
starlette>=0.27.0
```

### 6. Data Processing Dependencies

#### Scientific Computing
```txt
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
```

#### Data Storage
```txt
h5py>=3.9.0
zarr>=2.15.0
lmdb>=1.4.0
pyarrow>=12.0.0
fastparquet>=2023.7.0
```

### 7. Development Dependencies

#### Testing Framework
```txt
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-asyncio>=0.21.0
pytest-mock>=3.11.0
pytest-benchmark>=4.0.0
```

#### Code Quality
```txt
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0
pylint>=2.17.0
```

## Installation Scripts

### Automated Installation

The `install_dependencies.py` script provides automated installation:

```bash
# Basic installation
python install_dependencies.py --type basic

# Full installation with GPU support
python install_dependencies.py --type gpu

# Development installation
python install_dependencies.py --type dev

# Production installation
python install_dependencies.py --type prod

# Custom requirements file
python install_dependencies.py --requirements custom_requirements.txt

# Upgrade existing packages
python install_dependencies.py --type full --upgrade

# Skip dependency verification
python install_dependencies.py --type basic --no-deps

# Create environment file
python install_dependencies.py --type basic --create-env
```

### Script Features

- **System Detection**: Automatically detects Python version, CUDA availability
- **Platform Support**: Handles Windows, Linux, and macOS differences
- **Error Handling**: Provides clear error messages and recovery options
- **Verification**: Verifies installation success
- **Environment Setup**: Creates configuration files

## Environment Setup

### 1. Environment Variables

Create a `.env` file with your configuration:

```bash
# Model Settings
MODEL_CACHE_DIR=./models
OUTPUT_DIR=./outputs
TEMP_DIR=./temp

# GPU Settings
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=7.5,8.0,8.6

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/video_opusclip.log

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Profiling
ENABLE_PROFILING=true
PROFILE_LEVEL=basic
SAVE_PROFILES=true

# Mixed Precision
ENABLE_MIXED_PRECISION=true
MIXED_PRECISION_DTYPE=float16

# Multi-GPU
ENABLE_MULTI_GPU=false
DISTRIBUTED_BACKEND=nccl

# Data Loading
NUM_WORKERS=4
PIN_MEMORY=true
PREFETCH_FACTOR=2
```

### 2. Virtual Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv video_opusclip_env

# Activate on Windows
video_opusclip_env\Scripts\activate

# Activate on Linux/macOS
source video_opusclip_env/bin/activate

# Install dependencies
python install_dependencies.py --type basic
```

### 3. Conda Environment

Using Conda for dependency management:

```bash
# Create conda environment
conda create -n video_opusclip python=3.9

# Activate environment
conda activate video_opusclip

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements_basic.txt
```

## Platform-Specific Notes

### Windows

#### Prerequisites
- Python 3.8+ (from python.org or Microsoft Store)
- Visual Studio Build Tools (for some packages)
- CUDA Toolkit (for GPU support)

#### Installation
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Install dependencies
python install_dependencies.py --type basic
```

#### Common Issues
- **Build Tools**: Some packages require Visual Studio Build Tools
- **Path Issues**: Ensure Python and pip are in PATH
- **CUDA**: Install CUDA Toolkit before PyTorch

### Linux

#### Prerequisites
- Python 3.8+ (from package manager or pyenv)
- Build essentials: `sudo apt-get install build-essential`
- CUDA Toolkit (for GPU support)

#### Installation
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev python3-pip

# Install Python dependencies
python install_dependencies.py --type basic
```

#### Common Issues
- **Permission Issues**: Use `--user` flag or virtual environment
- **Missing Libraries**: Install system packages for OpenCV, FFmpeg
- **CUDA**: Ensure CUDA drivers and toolkit are installed

### macOS

#### Prerequisites
- Python 3.8+ (from Homebrew or pyenv)
- Xcode Command Line Tools
- Homebrew (recommended)

#### Installation
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install dependencies
python install_dependencies.py --type basic
```

#### Common Issues
- **Xcode**: Install Xcode Command Line Tools
- **M1 Macs**: Some packages may need Rosetta 2
- **CUDA**: Not available on macOS, use CPU or cloud

## Troubleshooting

### Common Installation Issues

#### 1. PyTorch Installation Failures

**Problem**: PyTorch installation fails
**Solution**:
```bash
# Clear pip cache
pip cache purge

# Install with specific index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 2. OpenCV Installation Issues

**Problem**: OpenCV fails to install
**Solution**:
```bash
# Install system dependencies (Linux)
sudo apt-get install libopencv-dev python3-opencv

# Or use conda
conda install opencv

# Or install pre-built wheel
pip install opencv-python-headless
```

#### 3. CUDA Issues

**Problem**: CUDA not detected
**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Memory Issues

**Problem**: Out of memory during installation
**Solution**:
```bash
# Use pip with memory optimization
pip install --no-cache-dir -r requirements_basic.txt

# Install packages one by one
pip install torch
pip install transformers
# ... etc
```

#### 5. Version Conflicts

**Problem**: Package version conflicts
**Solution**:
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows

# Install with upgrade
pip install --upgrade pip
pip install -r requirements_basic.txt --upgrade
```

### Performance Issues

#### 1. Slow Installation

**Solutions**:
- Use faster pip mirrors
- Use conda for large packages
- Install packages in parallel
- Use pre-built wheels

#### 2. Large Download Sizes

**Solutions**:
- Use `--no-deps` for selective installation
- Download packages offline
- Use conda for better compression

### GPU-Specific Issues

#### 1. CUDA Version Mismatch

**Problem**: PyTorch CUDA version doesn't match system
**Solution**:
```bash
# Check system CUDA version
nvcc --version

# Install matching PyTorch version
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### 2. GPU Memory Issues

**Problem**: GPU out of memory
**Solution**:
```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use mixed precision
export ENABLE_MIXED_PRECISION=true
```

## Best Practices

### 1. Environment Management

#### Use Virtual Environments
```bash
# Always use virtual environments
python -m venv my_env
source my_env/bin/activate  # Linux/macOS
my_env\Scripts\activate     # Windows
```

#### Pin Versions
```bash
# Generate requirements with exact versions
pip freeze > requirements_exact.txt

# Use for reproducible builds
pip install -r requirements_exact.txt
```

### 2. Installation Strategy

#### Staged Installation
```bash
# 1. Install core dependencies
pip install torch torchvision torchaudio

# 2. Install ML libraries
pip install transformers diffusers accelerate

# 3. Install video processing
pip install opencv-python moviepy ffmpeg-python

# 4. Install optional dependencies
pip install gradio fastapi uvicorn
```

#### Dependency Groups
```bash
# Install specific groups
pip install -r requirements_complete.txt[dev]
pip install -r requirements_complete.txt[prod]
pip install -r requirements_complete.txt[gpu]
```

### 3. Maintenance

#### Regular Updates
```bash
# Update pip
pip install --upgrade pip

# Update packages
pip list --outdated
pip install --upgrade package_name

# Update all packages
pip install --upgrade -r requirements_basic.txt
```

#### Cleanup
```bash
# Remove unused packages
pip uninstall package_name

# Clean pip cache
pip cache purge

# Remove old virtual environments
rm -rf old_env/
```

### 4. Production Deployment

#### Containerization
```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_basic.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_basic.txt

# Copy application
COPY . .

# Run application
CMD ["python", "main.py"]
```

#### Environment Variables
```bash
# Use environment variables for configuration
export MODEL_CACHE_DIR=/app/models
export CUDA_VISIBLE_DEVICES=0
export LOG_LEVEL=INFO
```

## Summary

This guide provides comprehensive information for managing Video-OpusClip dependencies:

1. **Multiple Installation Options**: Basic, full, GPU, development, production
2. **Automated Installation**: Use `install_dependencies.py` script
3. **Platform Support**: Windows, Linux, macOS specific instructions
4. **Troubleshooting**: Common issues and solutions
5. **Best Practices**: Environment management and maintenance

The dependency system is designed to be flexible and accommodate different use cases, from basic video processing to full-scale production deployments with GPU optimization and profiling capabilities. 