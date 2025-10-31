#!/usr/bin/env python3
"""
Launch Script for Enhanced Unified AI Interface v3.5
The Ultimate AI Revolution with Advanced Performance Optimization
"""
import sys
import os
import platform
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print the revolutionary banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║  🚀 ENHANCED UNIFIED AI INTERFACE v3.5 - THE ULTIMATE REVOLUTION 🚀      ║
    ║                                                                              ║
    ║  ⚛️  Quantum Hybrid Intelligence System                                     ║
    ║  🚀  Autonomous Extreme Optimization Engine                                ║
    ║  🧠  Conscious Evolutionary Learning System                                ║
    ║  ⚡  Advanced Performance Optimization                                      ║
    ║                                                                              ║
    ║  The Future of AI is Here - Beyond Human Imagination                       ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_system_info():
    """Print enhanced system information"""
    print("🔍 ENHANCED SYSTEM INFORMATION:")
    print(f"   • Operating System: {platform.system()} {platform.release()}")
    print(f"   • Python Version: {sys.version}")
    print(f"   • Architecture: {platform.architecture()[0]}")
    print(f"   • Processor: {platform.processor()}")
    print(f"   • Current Directory: {os.getcwd()}")
    print(f"   • Available Memory: {get_available_memory()}")
    print()

def get_available_memory():
    """Get available system memory"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return f"{memory.available // (1024**3)} GB available"
    except ImportError:
        return "Unknown (psutil not available)"

def check_dependencies():
    """Check if all required dependencies are available"""
    print("📦 CHECKING ENHANCED DEPENDENCIES:")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'gradio', 'plotly', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Installing missing packages...")
        install_missing_packages(missing_packages)
        return False
    
    print("   ✅ All dependencies available!")
    print()
    return True

def install_missing_packages(packages):
    """Install missing packages"""
    try:
        for package in packages:
            print(f"   📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ {package} installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install packages: {e}")

def check_pytorch_compatibility():
    """Check PyTorch compatibility and capabilities"""
    print("🔥 ENHANCED PYTORCH COMPATIBILITY CHECK:")
    
    try:
        import torch
        
        print(f"   ✅ PyTorch Version: {torch.__version__}")
        print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Version: {torch.version.cuda}")
            print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"   ✅ GPU {i}: {gpu_name} ({gpu_memory} GB)")
        else:
            print("   ⚠️  CUDA not available - Using CPU")
        
        # Check PyTorch performance
        print("   🔍 Testing PyTorch performance...")
        test_pytorch_performance()
        
        print("   ✅ PyTorch compatibility confirmed!")
        print()
        return True
        
    except ImportError:
        print("   ❌ PyTorch not available")
        print()
        return False
    except Exception as e:
        print(f"   ❌ PyTorch compatibility check failed: {e}")
        print()
        return False

def test_pytorch_performance():
    """Test PyTorch performance with simple operations"""
    try:
        import torch
        import time
        
        # Test tensor operations
        start_time = time.time()
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.mm(x, y)
        operation_time = time.time() - start_time
        
        print(f"   ⚡ Matrix multiplication (1000x1000): {operation_time:.4f}s")
        
        # Test memory usage
        memory_used = x.element_size() * x.nelement() / (1024**2)
        print(f"   💾 Memory usage: {memory_used:.2f} MB")
        
    except Exception as e:
        print(f"   ⚠️  Performance test failed: {e}")

def check_system_performance():
    """Check system performance capabilities"""
    print("⚡ SYSTEM PERFORMANCE CHECK:")
    
    try:
        import psutil
        
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"   🖥️  CPU Cores: {cpu_count}")
        if cpu_freq:
            print(f"   🚀 CPU Frequency: {cpu_freq.current:.1f} MHz")
        
        # Memory information
        memory = psutil.virtual_memory()
        print(f"   💾 Total Memory: {memory.total // (1024**3)} GB")
        print(f"   📊 Available Memory: {memory.available // (1024**3)} GB")
        print(f"   📈 Memory Usage: {memory.percent}%")
        
        # Disk information
        disk = psutil.disk_usage('/')
        print(f"   💿 Disk Space: {disk.free // (1024**3)} GB available")
        
        print("   ✅ System performance check completed!")
        print()
        
    except ImportError:
        print("   ⚠️  psutil not available - skipping performance check")
        print()
    except Exception as e:
        print(f"   ❌ Performance check failed: {e}")
        print()

def print_feature_overview():
    """Print comprehensive feature overview"""
    print("🚀 ENHANCED FEATURES OVERVIEW:")
    print("   ⚛️  Quantum Hybrid Intelligence System")
    print("      • Advanced quantum-inspired algorithms")
    print("      • Neural network optimization")
    print("      • Real-time data processing")
    print()
    print("   🚀 Autonomous Extreme Optimization Engine")
    print("      • Multi-objective optimization")
    print("      • Performance monitoring")
    print("      • Adaptive learning")
    print()
    print("   🧠 Conscious Evolutionary Learning System")
    print("      • Continuous improvement")
    print("      • Pattern recognition")
    print("      • Knowledge evolution")
    print()
    print("   ⚡ Advanced Performance Optimization")
    print("      • Real-time monitoring")
    print("      • Resource optimization")
    print("      • Adaptive scaling")
    print()

def check_port_availability():
    """Check if the required port is available"""
    print("🔌 PORT AVAILABILITY CHECK:")
    
    port = 7865
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"   ❌ Port {port} is already in use")
            print(f"   💡 Please stop any existing services on port {port}")
            return False
        else:
            print(f"   ✅ Port {port} is available")
            return True
            
    except Exception as e:
        print(f"   ⚠️  Could not check port availability: {e}")
        return True

def launch_enhanced_interface():
    """Launch the enhanced unified AI interface"""
    print("🚀 LAUNCHING ENHANCED UNIFIED AI INTERFACE v3.5...")
    print()
    
    try:
        # Import and launch the enhanced interface
        from enhanced_unified_ai_interface_v3_5 import EnhancedUnifiedAIInterfaceV35
        
        print("✅ Enhanced interface imported successfully!")
        print("🚀 Creating interface...")
        
        interface = EnhancedUnifiedAIInterfaceV35()
        app = interface.create_interface()
        
        print("✅ Interface created successfully!")
        print("🚀 Launching web interface...")
        print()
        print("🌐 Access your enhanced AI system at: http://localhost:7865")
        print("⚡ Press Ctrl+C to stop the system")
        print()
        
        # Launch the interface
        app.launch(
            server_name="0.0.0.0",
            server_port=7865,
            share=False,
            debug=True,
            show_error=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure all required files are in the correct directory")
        return False
    except Exception as e:
        print(f"❌ Launch error: {e}")
        return False

def main():
    """Main function"""
    try:
        # Print banner
        print_banner()
        
        # System checks
        print_system_info()
        check_system_performance()
        
        if not check_dependencies():
            print("❌ Dependency check failed. Please install missing packages.")
            return False
        
        if not check_pytorch_compatibility():
            print("❌ PyTorch compatibility check failed.")
            return False
        
        if not check_port_availability():
            print("❌ Port availability check failed.")
            return False
        
        # Feature overview
        print_feature_overview()
        
        # Launch interface
        launch_enhanced_interface()
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
