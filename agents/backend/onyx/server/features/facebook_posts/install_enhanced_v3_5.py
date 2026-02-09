#!/usr/bin/env python3
"""
Enhanced Installation Script for Unified AI Interface v3.5
Comprehensive dependency management and system optimization
"""
import sys
import os
import platform
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print the enhanced installation banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║  🚀 ENHANCED UNIFIED AI INTERFACE v3.5 - INSTALLATION SCRIPT 🚀          ║
    ║                                                                              ║
    ║  ⚛️  Quantum Hybrid Intelligence System                                     ║
    ║  🚀  Autonomous Extreme Optimization Engine                                ║
    ║  🧠  Conscious Evolutionary Learning System                                ║
    ║  ⚡  Advanced Performance Optimization                                      ║
    ║                                                                              ║
    ║  Installing the Future of AI Technology                                     ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_system_info():
    """Print comprehensive system information"""
    print("🔍 SYSTEM ANALYSIS:")
    print(f"   • Operating System: {platform.system()} {platform.release()}")
    print(f"   • Python Version: {sys.version}")
    print(f"   • Architecture: {platform.architecture()[0]}")
    print(f"   • Processor: {platform.processor()}")
    print(f"   • Current Directory: {os.getcwd()}")
    print(f"   • Python Executable: {sys.executable}")
    print()

def check_python_version():
    """Check if Python version meets requirements"""
    print("🐍 PYTHON VERSION CHECK:")
    
    version = sys.version_info
    required_version = (3, 9)
    
    if version >= required_version:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} meets requirements")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is too old")
        print(f"   💡 Required: Python {required_version[0]}.{required_version[1]}+")
        print(f"   💡 Please upgrade Python and try again")
        return False

def check_pip():
    """Check if pip is available and working"""
    print("📦 PIP AVAILABILITY CHECK:")
    
    try:
        import pip
        print(f"   ✅ pip {pip.__version__} is available")
        return True
    except ImportError:
        print("   ❌ pip is not available")
        print("   💡 Please install pip and try again")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("📦 UPGRADING PIP:")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        print("   ✅ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to upgrade pip: {e}")
        return False

def install_core_dependencies():
    """Install core dependencies"""
    print("📦 INSTALLING CORE DEPENDENCIES:")
    
    core_packages = [
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'gradio>=3.40.0',
        'plotly>=5.15.0',
        'psutil>=5.9.0'
    ]
    
    for package in core_packages:
        try:
            print(f"   📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    print("   ✅ All core dependencies installed!")
    return True

def install_advanced_dependencies():
    """Install advanced dependencies"""
    print("📦 INSTALLING ADVANCED DEPENDENCIES:")
    
    advanced_packages = [
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'accelerate>=0.20.0',
        'optuna>=3.2.0',
        'ray[tune]>=2.6.0'
    ]
    
    for package in advanced_packages:
        try:
            print(f"   📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Failed to install {package}: {e}")
            print(f"   💡 This is optional, continuing...")
    
    print("   ✅ Advanced dependencies installation completed!")
    return True

def install_optional_dependencies():
    """Install optional dependencies"""
    print("📦 INSTALLING OPTIONAL DEPENDENCIES:")
    
    optional_packages = [
        'qiskit>=0.44.0',
        'cirq>=1.2.0',
        'pennylane>=0.30.0',
        'nvidia-ml-py>=11.495.0',
        'pynvml>=11.5.0',
        'gputil>=1.4.0'
    ]
    
    for package in optional_packages:
        try:
            print(f"   📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Failed to install {package}: {e}")
            print(f"   💡 This is optional, continuing...")
    
    print("   ✅ Optional dependencies installation completed!")
    return True

def verify_installation():
    """Verify that all dependencies are properly installed"""
    print("🔍 VERIFYING INSTALLATION:")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'gradio', 'plotly', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} is available")
        except ImportError:
            print(f"   ❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("   ✅ All required packages are available!")
    return True

def test_pytorch():
    """Test PyTorch installation and capabilities"""
    print("🔥 TESTING PYTORCH:")
    
    try:
        import torch
        
        print(f"   ✅ PyTorch Version: {torch.__version__}")
        print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Version: {torch.version.cuda}")
            print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   ✅ GPU {i}: {gpu_name}")
        else:
            print("   ⚠️  CUDA not available - Using CPU")
        
        # Test basic operations
        print("   🔍 Testing basic operations...")
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.mm(x, y)
        print("   ✅ Basic tensor operations working")
        
        return True
        
    except ImportError:
        print("   ❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 CREATING DIRECTORIES:")
    
    directories = [
        'logs',
        'cache',
        'models',
        'data',
        'configs',
        'exports'
    ]
    
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"   ✅ Created directory: {directory}")
            else:
                print(f"   ℹ️  Directory already exists: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create directory {directory}: {e}")
    
    print("   ✅ Directory creation completed!")

def create_config_files():
    """Create default configuration files"""
    print("⚙️  CREATING CONFIGURATION FILES:")
    
    # Create main config
    config_content = """# Enhanced Unified AI Interface v3.5 Configuration
# Generated on: {timestamp}

[system]
name = "Enhanced Unified AI Interface v3.5"
version = "3.5.0"
port = 7865
debug = true

[performance]
enable_real_time_monitoring = true
enable_adaptive_optimization = true
max_memory_usage = 0.8
max_cpu_usage = 0.9
gpu_memory_fraction = 0.8

[ai]
quantum_enabled = true
optimization_enabled = true
learning_enabled = true
confidence_threshold = 0.85

[logging]
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
file = "logs/system.log"
""".format(timestamp=datetime.now().isoformat())
    
    try:
        with open('config.ini', 'w') as f:
            f.write(config_content)
        print("   ✅ Created config.ini")
    except Exception as e:
        print(f"   ❌ Failed to create config.ini: {e}")
    
    # Create requirements file
    requirements_content = """# Enhanced Unified AI Interface v3.5 Requirements
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
gradio>=3.40.0
plotly>=5.15.0
psutil>=5.9.0

# Advanced dependencies
scipy>=1.10.0
scikit-learn>=1.3.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
optuna>=3.2.0
ray[tune]>=2.6.0

# Optional dependencies
qiskit>=0.44.0
cirq>=1.2.0
pennylane>=0.30.0
nvidia-ml-py>=11.495.0
pynvml>=11.5.0
gputil>=1.4.0
"""
    
    try:
        with open('requirements.txt', 'w') as f:
            f.write(requirements_content)
        print("   ✅ Created requirements.txt")
    except Exception as e:
        print(f"   ❌ Failed to create requirements.txt: {e}")
    
    print("   ✅ Configuration files created!")

def run_system_tests():
    """Run basic system tests"""
    print("🧪 RUNNING SYSTEM TESTS:")
    
    try:
        # Test imports
        print("   🔍 Testing imports...")
        import torch
        import numpy as np
        import pandas as pd
        import gradio as gr
        import plotly.graph_objects as go
        import psutil
        print("   ✅ All imports successful")
        
        # Test basic operations
        print("   🔍 Testing basic operations...")
        x = np.random.randn(100, 100)
        y = np.random.randn(100, 100)
        z = np.dot(x, y)
        print("   ✅ NumPy operations working")
        
        # Test PyTorch
        print("   🔍 Testing PyTorch...")
        x_torch = torch.randn(100, 100)
        y_torch = torch.randn(100, 100)
        z_torch = torch.mm(x_torch, y_torch)
        print("   ✅ PyTorch operations working")
        
        # Test system monitoring
        print("   🔍 Testing system monitoring...")
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"   ✅ CPU: {cpu_percent}%, Memory: {memory.percent}%")
        
        print("   ✅ All system tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ System test failed: {e}")
        return False

def print_installation_summary():
    """Print installation summary"""
    print("\n" + "="*80)
    print("🎉 ENHANCED UNIFIED AI INTERFACE v3.5 - INSTALLATION COMPLETED! 🎉")
    print("="*80)
    print()
    print("🚀 Your enhanced AI system is ready to launch!")
    print()
    print("📁 Files created:")
    print("   • enhanced_unified_ai_interface_v3_5.py - Main interface")
    print("   • launch_enhanced_unified_ai_v3_5.py - Launch script")
    print("   • requirements_enhanced_v3_5.txt - Dependencies")
    print("   • README_ENHANCED_UNIFIED_AI_v3_5.md - Documentation")
    print("   • config.ini - Configuration file")
    print("   • requirements.txt - Basic requirements")
    print()
    print("🚀 To launch your enhanced AI system:")
    print("   python launch_enhanced_unified_ai_v3_5.py")
    print()
    print("🌐 Access your system at: http://localhost:7865")
    print()
    print("📚 For more information, see: README_ENHANCED_UNIFIED_AI_v3_5.md")
    print()
    print("⚡ Enjoy your enhanced AI experience!")
    print("="*80)

def main():
    """Main installation function"""
    try:
        # Print banner
        print_banner()
        
        # System checks
        print_system_info()
        
        if not check_python_version():
            return False
        
        if not check_pip():
            return False
        
        # Upgrade pip
        upgrade_pip()
        
        # Install dependencies
        if not install_core_dependencies():
            print("❌ Core dependencies installation failed")
            return False
        
        install_advanced_dependencies()
        install_optional_dependencies()
        
        # Verify installation
        if not verify_installation():
            print("❌ Installation verification failed")
            return False
        
        # Test PyTorch
        if not test_pytorch():
            print("❌ PyTorch test failed")
            return False
        
        # Create directories and config files
        create_directories()
        create_config_files()
        
        # Run system tests
        if not run_system_tests():
            print("❌ System tests failed")
            return False
        
        # Installation summary
        print_installation_summary()
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Installation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Installation failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        print("\n✅ Installation completed successfully!")
