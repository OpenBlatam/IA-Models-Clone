#!/usr/bin/env python3
"""
Video-OpusClip Dependency Installation Script

This script helps install dependencies for the Video-OpusClip system
with different options for basic, full, or GPU-optimized installations.
"""

import subprocess
import sys
import os
import platform
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: pip is not available")
        return False

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA is not available (will use CPU)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check CUDA")
        return False

def install_requirements(requirements_file, extra_flags=None):
    """Install requirements from a file."""
    cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
    
    if extra_flags:
        cmd.extend(extra_flags)
    
    print(f"Installing from {requirements_file}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Installation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_pytorch_gpu():
    """Install PyTorch with GPU support."""
    print("Installing PyTorch with GPU support...")
    
    # Detect CUDA version and install appropriate PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"Detected CUDA version: {cuda_version}")
            
            if cuda_version.startswith("11"):
                cmd = [sys.executable, "-m", "pip", "install", 
                       "torch", "torchvision", "torchaudio", 
                       "--index-url", "https://download.pytorch.org/whl/cu118"]
            elif cuda_version.startswith("12"):
                cmd = [sys.executable, "-m", "pip", "install", 
                       "torch", "torchvision", "torchaudio", 
                       "--index-url", "https://download.pytorch.org/whl/cu121"]
            else:
                print("⚠️  Unknown CUDA version, installing CPU version")
                cmd = [sys.executable, "-m", "pip", "install", 
                       "torch", "torchvision", "torchaudio"]
            
            try:
                subprocess.run(cmd, check=True)
                print("✅ PyTorch GPU installation completed")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ PyTorch GPU installation failed: {e}")
                return False
    except ImportError:
        print("⚠️  PyTorch not available, installing CPU version")
        cmd = [sys.executable, "-m", "pip", "install", 
               "torch", "torchvision", "torchaudio"]
        try:
            subprocess.run(cmd, check=True)
            print("✅ PyTorch CPU installation completed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ PyTorch installation failed: {e}")
            return False

def install_optional_dependencies():
    """Install optional dependencies."""
    print("Installing optional dependencies...")
    
    optional_packages = [
        "jupyter",
        "ipywidgets",
        "wandb",
        "mlflow",
        "tensorboard",
        "optuna",
        "ray[tune]",
        "boto3",
        "google-cloud-storage",
        "azure-storage-blob"
    ]
    
    for package in optional_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package} (optional)")

def verify_installation():
    """Verify that key packages are installed correctly."""
    print("\nVerifying installation...")
    
    key_packages = [
        "torch",
        "torchvision",
        "transformers",
        "opencv-python",
        "numpy",
        "gradio",
        "fastapi"
    ]
    
    for package in key_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            return False
    
    return True

def create_environment_file():
    """Create a .env file with basic configuration."""
    env_content = """# Video-OpusClip Environment Configuration

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

# Cache Settings
CACHE_ENABLED=true
CACHE_SIZE=1000
CACHE_TTL=3600

# Cloud Storage (optional)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_S3_BUCKET=

GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_BUCKET=

AZURE_STORAGE_ACCOUNT=
AZURE_STORAGE_KEY=
AZURE_CONTAINER_NAME=
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
    else:
        print("⚠️  .env file already exists")

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install Video-OpusClip dependencies")
    parser.add_argument(
        "--type",
        choices=["basic", "full", "gpu", "dev", "prod"],
        default="basic",
        help="Installation type"
    )
    parser.add_argument(
        "--requirements",
        type=str,
        help="Custom requirements file path"
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade existing packages"
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip dependency verification"
    )
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create .env file"
    )
    
    args = parser.parse_args()
    
    print("Video-OpusClip Dependency Installation")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    if not check_pip():
        sys.exit(1)
    
    # Check CUDA availability
    cuda_available = check_cuda()
    
    # Determine requirements file
    if args.requirements:
        requirements_file = args.requirements
    elif args.type == "basic":
        requirements_file = "requirements_basic.txt"
    elif args.type == "full":
        requirements_file = "requirements_complete.txt"
    elif args.type == "gpu":
        requirements_file = "requirements_complete.txt"
    elif args.type == "dev":
        requirements_file = "requirements_complete.txt"
    elif args.type == "prod":
        requirements_file = "requirements_complete.txt"
    else:
        requirements_file = "requirements_basic.txt"
    
    # Check if requirements file exists
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file {requirements_file} not found")
        sys.exit(1)
    
    # Prepare installation flags
    extra_flags = []
    if args.upgrade:
        extra_flags.append("--upgrade")
    
    # Install PyTorch with GPU support if requested
    if args.type == "gpu" and cuda_available:
        if not install_pytorch_gpu():
            print("⚠️  GPU installation failed, continuing with CPU")
    
    # Install requirements
    if not install_requirements(requirements_file, extra_flags):
        sys.exit(1)
    
    # Install optional dependencies for full installation
    if args.type in ["full", "gpu", "dev", "prod"]:
        install_optional_dependencies()
    
    # Install extra dependencies based on type
    if args.type == "dev":
        print("Installing development dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "-r", requirements_file, "[dev]"], check=True)
    
    elif args.type == "prod":
        print("Installing production dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "-r", requirements_file, "[prod]"], check=True)
    
    elif args.type == "gpu":
        print("Installing GPU dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "-r", requirements_file, "[gpu]"], check=True)
    
    # Verify installation
    if not args.no_deps:
        if not verify_installation():
            print("❌ Installation verification failed")
            sys.exit(1)
    
    # Create environment file
    if args.create_env:
        create_environment_file()
    
    print("\n" + "=" * 50)
    print("✅ Installation completed successfully!")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Set up your environment variables (see .env file)")
    print("2. Download required models")
    print("3. Run the quick start scripts:")
    print("   - python quick_start_profiling.py --mode basic")
    print("   - python quick_start_mixed_precision.py --mode basic")
    print("   - python quick_start_ui.py")
    
    if args.type == "gpu":
        print("\nGPU-specific notes:")
        print("- Ensure CUDA drivers are installed")
        print("- Set CUDA_VISIBLE_DEVICES if needed")
        print("- Monitor GPU memory usage")
    
    print("\nFor more information, see the documentation in the docs/ directory")

if __name__ == "__main__":
    main() 