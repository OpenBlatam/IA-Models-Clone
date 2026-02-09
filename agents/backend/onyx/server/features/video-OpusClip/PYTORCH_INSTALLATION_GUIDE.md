# PyTorch Installation Guide for Video-OpusClip

Complete guide to installing PyTorch and setting up your Video-OpusClip system for optimal performance.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Python Installation](#python-installation)
3. [PyTorch Installation](#pytorch-installation)
4. [Video-OpusClip Dependencies](#video-opusclip-dependencies)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher (3.11+ recommended)
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 10GB free space minimum
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### GPU Requirements (Optional)

For GPU acceleration, you need:
- **NVIDIA GPU**: GTX 1060 or better (RTX series recommended)
- **CUDA Toolkit**: 11.8 or 12.1
- **NVIDIA Drivers**: Latest stable version

## Python Installation

### Windows Installation

#### Option 1: Microsoft Store (Recommended)
1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Click "Install"
4. Python will be automatically added to PATH

#### Option 2: Official Python Installer
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or 3.12
3. Run the installer
4. **Important**: Check "Add Python to PATH"
5. Choose "Install Now" or "Customize Installation"

#### Option 3: Anaconda/Miniconda
1. Download Miniconda from https://docs.conda.io/en/latest/miniconda.html
2. Run the installer
3. Add to PATH during installation
4. Create a new environment:
   ```bash
   conda create -n video-opusclip python=3.11
   conda activate video-opusclip
   ```

### Linux Installation

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### CentOS/RHEL
```bash
sudo yum install python3 python3-pip
```

#### Using pyenv (Recommended)
```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell profile
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python
pyenv install 3.11.5
pyenv global 3.11.5
```

### macOS Installation

#### Using Homebrew
```bash
brew install python@3.11
```

#### Using pyenv
```bash
brew install pyenv
pyenv install 3.11.5
pyenv global 3.11.5
```

## PyTorch Installation

### Verify Python Installation

First, verify Python is properly installed:

```bash
python --version
# or
python3 --version
```

### Install PyTorch

#### CPU Only Installation
```bash
pip install torch torchvision torchaudio
```

#### CUDA Installation (Recommended)

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Using Conda
```bash
# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Verify PyTorch Installation

Create a test script to verify PyTorch:

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
```

## Video-OpusClip Dependencies

### Navigate to Video-OpusClip Directory

```bash
cd agents/backend/onyx/server/features/video-OpusClip
```

### Install Dependencies

#### Option 1: Basic Dependencies
```bash
pip install -r requirements_basic.txt
```

#### Option 2: Complete Dependencies (Recommended)
```bash
pip install -r requirements_complete.txt
```

#### Option 3: Automated Installation
```bash
python install_dependencies.py
```

### Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv video-opusclip-env

# Activate virtual environment
# Windows:
video-opusclip-env\Scripts\activate
# Linux/macOS:
source video-opusclip-env/bin/activate

# Install dependencies
pip install -r requirements_complete.txt
```

## Verification

### Run Installation Tests

```bash
# Test PyTorch installation
python test_pytorch_install.py

# Run comprehensive setup check
python torch_setup_check.py
```

### Test Video-OpusClip Features

```bash
# Test mixed precision training
python quick_start_mixed_precision.py

# Test multi-GPU training
python quick_start_multi_gpu.py

# Test gradient accumulation
python quick_start_gradient_accumulation.py

# Test PyTorch debugging
python quick_start_pytorch_debugging.py

# Test profiling
python quick_start_profiling.py
```

## Troubleshooting

### Common Issues

#### 1. Python Not Found
**Problem**: `python` command not recognized
**Solution**:
```bash
# Check if Python is installed
where python
# or
which python3

# Add Python to PATH manually
# Windows: Add Python installation directory to PATH environment variable
# Linux/macOS: Add to ~/.bashrc or ~/.zshrc
```

#### 2. PyTorch Installation Fails
**Problem**: PyTorch installation fails with errors
**Solution**:
```bash
# Upgrade pip
pip install --upgrade pip

# Clear pip cache
pip cache purge

# Try alternative installation
pip install torch torchvision torchaudio --no-cache-dir
```

#### 3. CUDA Not Available
**Problem**: `torch.cuda.is_available()` returns False
**Solution**:
1. Install NVIDIA drivers
2. Install CUDA toolkit
3. Reinstall PyTorch with CUDA support
4. Check GPU compatibility

#### 4. Memory Issues
**Problem**: Out of memory errors
**Solution**:
```python
# Reduce batch size
config = TrainingConfig(batch_size=16)  # Instead of 32

# Use gradient accumulation
config = GradientAccumulationConfig(accumulation_steps=4)

# Enable mixed precision
config = MixedPrecisionConfig(enabled=True)
```

#### 5. Import Errors
**Problem**: Module import errors
**Solution**:
```bash
# Install missing dependencies
pip install -r requirements_basic.txt

# Check for conflicts
pip check

# Reinstall problematic packages
pip uninstall package_name
pip install package_name
```

### GPU-Specific Issues

#### NVIDIA Driver Issues
```bash
# Check NVIDIA driver version
nvidia-smi

# Update drivers from NVIDIA website
# https://www.nvidia.com/Download/index.aspx
```

#### CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Performance Optimization

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Windows
set CUDA_LAUNCH_BLOCKING=1
set TORCH_CUDNN_V8_API_ENABLED=1

# Linux/macOS
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDNN_V8_API_ENABLED=1
```

### PyTorch Configuration

```python
import torch

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# Enable cuDNN deterministic mode (for reproducibility)
torch.backends.cudnn.deterministic = True

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)
```

### System Optimization

#### Windows
1. Enable "High Performance" power plan
2. Disable unnecessary background processes
3. Update GPU drivers regularly

#### Linux
```bash
# Install performance tools
sudo apt install linux-tools-common linux-tools-generic

# Monitor performance
htop
nvidia-smi -l 1
```

## Quick Start Commands

### Complete Setup (One Command)
```bash
# Windows
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install -r requirements_complete.txt

# Linux/macOS
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install -r requirements_complete.txt
```

### Verification Commands
```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test Video-OpusClip
python test_pytorch_install.py

# Run setup check
python torch_setup_check.py
```

### Development Setup
```bash
# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Linux/macOS
# or
dev-env\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements_complete.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/
```

## Next Steps

After successful installation:

1. **Read the Guides**:
   - `PYTORCH_QUICK_REFERENCE.md` - Quick reference
   - `MIXED_PRECISION_GUIDE.md` - Mixed precision training
   - `MULTI_GPU_TRAINING_GUIDE.md` - Multi-GPU training
   - `GRADIENT_ACCUMULATION_GUIDE.md` - Gradient accumulation
   - `PYTORCH_DEBUGGING_GUIDE.md` - Debugging tools

2. **Run Examples**:
   - `quick_start_mixed_precision.py`
   - `quick_start_multi_gpu.py`
   - `quick_start_gradient_accumulation.py`

3. **Explore Features**:
   - Mixed precision training
   - Multi-GPU training
   - Gradient accumulation
   - Performance profiling
   - PyTorch debugging

4. **Production Setup**:
   - Configure logging
   - Set up monitoring
   - Optimize performance
   - Deploy models

---

This guide covers all aspects of PyTorch installation for your Video-OpusClip system. For specific issues, refer to the troubleshooting section or check the detailed guides in this directory. 