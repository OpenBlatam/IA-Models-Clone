#!/usr/bin/env python3
"""
PyTorch Setup Checker for Video-OpusClip

This script checks your current PyTorch installation and provides
recommendations for optimal setup with the Video-OpusClip system.
"""

import sys
import platform
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Python Version Check")
    print("=" * 40)
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible with PyTorch")
    else:
        print("❌ Python version may have compatibility issues")
        print("   Recommended: Python 3.8+")
    
    print()

def check_pytorch_installation():
    """Check PyTorch installation and version."""
    print("🔥 PyTorch Installation Check")
    print("=" * 40)
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA not available - training will be slower on CPU")
        
        # Check other PyTorch components
        try:
            import torchvision
            print(f"TorchVision Version: {torchvision.__version__}")
        except ImportError:
            print("❌ TorchVision not installed")
        
        try:
            import torchaudio
            print(f"TorchAudio Version: {torchaudio.__version__}")
        except ImportError:
            print("❌ TorchAudio not installed")
        
        print("✅ PyTorch installation check completed")
        
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install torch torchvision torchaudio")
        return False
    
    print()
    return True

def check_system_info():
    """Check system information."""
    print("💻 System Information")
    print("=" * 40)
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total Memory: {memory.total / 1024**3:.1f}GB")
        print(f"Available Memory: {memory.available / 1024**3:.1f}GB")
    except ImportError:
        print("⚠️  psutil not installed - cannot check memory")
    
    print()

def check_dependencies():
    """Check Video-OpusClip dependencies."""
    print("📦 Video-OpusClip Dependencies Check")
    print("=" * 40)
    
    dependencies = [
        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
        'opencv-python', 'Pillow', 'imageio', 'scikit-image',
        'moviepy', 'ffmpeg-python', 'yt-dlp',
        'transformers', 'diffusers', 'accelerate',
        'gradio', 'fastapi', 'uvicorn',
        'structlog', 'tqdm', 'pyyaml'
    ]
    
    missing_deps = []
    installed_deps = []
    
    for dep in dependencies:
        try:
            module = __import__(dep.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            installed_deps.append(f"✅ {dep} ({version})")
        except ImportError:
            missing_deps.append(f"❌ {dep}")
    
    print("Installed Dependencies:")
    for dep in installed_deps:
        print(f"  {dep}")
    
    if missing_deps:
        print("\nMissing Dependencies:")
        for dep in missing_deps:
            print(f"  {dep}")
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements_basic.txt")
    
    print()

def check_optimization_features():
    """Check optimization features availability."""
    print("⚡ Optimization Features Check")
    print("=" * 40)
    
    features = {
        'Mixed Precision': 'torch.cuda.amp',
        'Multi-GPU': 'torch.nn.parallel',
        'Distributed Training': 'torch.distributed',
        'Memory Profiling': 'memory_profiler',
        'Line Profiling': 'line_profiler',
        'GPU Monitoring': 'GPUtil',
        'Performance Monitoring': 'structlog'
    }
    
    for feature, module in features.items():
        try:
            __import__(module)
            print(f"✅ {feature}: Available")
        except ImportError:
            print(f"❌ {feature}: Not available")
    
    print()

def run_basic_torch_test():
    """Run basic PyTorch functionality test."""
    print("🧪 Basic PyTorch Functionality Test")
    print("=" * 40)
    
    try:
        import torch
        import torch.nn as nn
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("✅ Basic tensor operations: OK")
        
        # Test neural network
        model = nn.Linear(10, 1)
        input_tensor = torch.randn(5, 10)
        output = model(input_tensor)
        print("✅ Neural network forward pass: OK")
        
        # Test backward pass
        loss = output.mean()
        loss.backward()
        print("✅ Backward pass: OK")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model_cuda = model.to(device)
            input_cuda = input_tensor.to(device)
            output_cuda = model_cuda(input_cuda)
            print("✅ CUDA operations: OK")
        
        print("✅ All basic PyTorch tests passed!")
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False
    
    print()
    return True

def provide_recommendations():
    """Provide setup recommendations."""
    print("💡 Setup Recommendations")
    print("=" * 40)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("🔧 For GPU acceleration:")
            print("   1. Install NVIDIA drivers")
            print("   2. Install CUDA toolkit")
            print("   3. Install PyTorch with CUDA support:")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n🔧 For optimal Video-OpusClip performance:")
        print("   1. Install all dependencies: pip install -r requirements_complete.txt")
        print("   2. Enable mixed precision training in your config")
        print("   3. Use gradient accumulation for large models")
        print("   4. Monitor GPU memory usage")
        print("   5. Enable PyTorch debugging when needed")
        
        print("\n🔧 For development:")
        print("   1. Install development dependencies")
        print("   2. Set up proper logging")
        print("   3. Use profiling tools for optimization")
        
    except ImportError:
        print("❌ PyTorch not installed - install first")
    
    print()

def main():
    """Main function to run all checks."""
    print("🚀 Video-OpusClip PyTorch Setup Checker")
    print("=" * 50)
    print()
    
    # Run all checks
    check_python_version()
    pytorch_ok = check_pytorch_installation()
    check_system_info()
    check_dependencies()
    check_optimization_features()
    
    if pytorch_ok:
        run_basic_torch_test()
    
    provide_recommendations()
    
    print("🎉 Setup check completed!")
    print("\nNext steps:")
    print("1. Review any missing dependencies")
    print("2. Install required packages")
    print("3. Run the quick start scripts in this directory")
    print("4. Check the guides for detailed setup instructions")

if __name__ == "__main__":
    main() 