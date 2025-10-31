#!/usr/bin/env python3
"""
Advanced Learning Facebook Content Optimization System v3.1 - Launcher
This script launches the revolutionary AI-powered learning system with federated learning,
transfer learning, and active learning capabilities
"""

import sys
import os
import time
from datetime import datetime

def main():
    """Launch the advanced learning system v3.1"""
    print("🚀 Advanced Learning Facebook Content Optimization System v3.1")
    print("=" * 80)
    print("🎯 Revolutionary AI-Powered Learning Capabilities")
    print("=" * 80)
    
    try:
        # Import and launch the advanced learning system
        from advanced_learning_interface_v3_1 import AdvancedLearningInterfaceV31
        
        print("✅ Advanced Learning System v3.1 imported successfully!")
        print("🖥️ Launching Revolutionary Learning Interface...")
        print("\n🔮 Revolutionary v3.1 Features:")
        print("   - 🤝 Federated Learning Network (Multi-organization collaboration)")
        print("   - 🔄 Transfer Learning & Domain Adaptation (Cross-platform optimization)")
        print("   - 🧠 Active Learning & Data Intelligence (Intelligent data collection)")
        print("   - 🔮 Enhanced Content Analysis v3.0 (Advanced predictions)")
        print("   - 📊 Advanced Analytics & Insights (Predictive intelligence)")
        print("   - 🏥 System Health & Performance (Comprehensive monitoring)")
        
        print("\n🌐 Interface Tabs:")
        print("   - 🔮 Advanced Content Analysis v3.0: Enhanced viral prediction & sentiment analysis")
        print("   - 🤝 Federated Learning Network: Collaborative AI training across organizations")
        print("   - 🔄 Transfer Learning & Adaptation: Cross-platform knowledge transfer")
        print("   - 🧠 Active Learning & Intelligence: Smart data collection & continuous improvement")
        print("   - 📊 Advanced Analytics & Insights: Predictive analytics & trend forecasting")
        print("   - 🏥 System Health & Performance: Real-time monitoring & optimization")
        
        print("\n🚀 Revolutionary Learning Capabilities:")
        print("   - Multi-organization AI collaboration")
        print("   - Cross-platform knowledge transfer")
        print("   - Intelligent data collection & labeling")
        print("   - Continuous model improvement")
        print("   - Privacy-preserving learning")
        print("   - Real-time adaptation & optimization")
        
        print("\n⏰ Initializing Advanced AI Learning Systems...")
        
        # Create and launch the interface
        interface = AdvancedLearningInterfaceV31()
        
        print("✅ Advanced Learning Interface v3.1 created successfully!")
        print("🌐 The interface will open in your browser...")
        print("   - URL: http://localhost:7863")
        print("   - Port: 7863 (different from v3.0 to avoid conflicts)")
        print("   - Features: 6 revolutionary tabs with AI learning capabilities")
        
        print("\n🎯 Ready to launch! Starting the interface...")
        time.sleep(2)
        
        # Initialize systems and create the interface
        if interface.initialize_systems():
            print("✅ All systems initialized successfully!")
            gradio_interface = interface.create_interface()
            
            # Launch the interface
            gradio_interface.launch(
                server_name="0.0.0.0",  # Allow external connections
                server_port=7863,        # Use port 7863 for v3.1
                share=False,             # Set to True to create a public link
                debug=True,              # Enable debug mode
                show_error=True,         # Show error details
                quiet=False              # Show all output
            )
        else:
            print("❌ Failed to initialize systems")
            sys.exit(1)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Please ensure all dependencies are installed:")
        print("   pip install -r requirements_enhanced_system.txt")
        print("   pip install scikit-learn pandas plotly torch")
        print("\n📚 Required packages:")
        print("   - torch (PyTorch)")
        print("   - gradio")
        print("   - scikit-learn")
        print("   - pandas")
        print("   - plotly")
        print("   - numpy")
        print("   - matplotlib")
        print("   - seaborn")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error launching system: {e}")
        print("Please check the logs for more details.")
        print("\n🔍 Common issues:")
        print("   - Port 7863 already in use (try different port)")
        print("   - Missing dependencies")
        print("   - System resources insufficient")
        print("   - PyTorch compatibility issues")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    required_packages = [
        'torch', 'gradio', 'sklearn', 'pandas', 
        'plotly', 'numpy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'plotly':
                import plotly
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'seaborn':
                import seaborn
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing dependencies before running the system.")
        return False
    
    print("✅ All dependencies are available!")
    return True

def check_pytorch_compatibility():
    """Check PyTorch compatibility and features"""
    print("\n🔍 Checking PyTorch compatibility...")
    
    try:
        import torch
        
        print(f"   ✅ PyTorch Version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Available: {torch.version.cuda}")
            print(f"   ✅ GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️ CUDA Not Available (CPU mode)")
        
        # Check PyTorch features
        if hasattr(torch, 'autocast'):
            print("   ✅ Mixed Precision Training: Available")
        else:
            print("   ⚠️ Mixed Precision Training: Not Available")
        
        return True
        
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False

def system_info():
    """Display system information"""
    print("\n💻 System Information:")
    print(f"   - Python Version: {sys.version}")
    print(f"   - Working Directory: {os.getcwd()}")
    print(f"   - Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   - GPU Available: ✅ {torch.cuda.get_device_name(0)}")
            print(f"   - CUDA Version: {torch.version.cuda}")
        else:
            print("   - GPU Available: ❌ (CPU mode)")
    except ImportError:
        print("   - GPU Check: ❌ PyTorch not available")

def display_v3_1_features():
    """Display detailed v3.1 features"""
    print("\n🚀 **Advanced Learning System v3.1 - Revolutionary Features**")
    print("=" * 80)
    
    print("\n🤝 **Federated Learning Network**")
    print("   • Multi-organization AI collaboration")
    print("   • Privacy-preserving model training")
    print("   • Distributed learning across clients")
    print("   • Secure model aggregation")
    print("   • Real-time network health monitoring")
    
    print("\n🔄 **Transfer Learning & Domain Adaptation**")
    print("   • Cross-platform knowledge transfer")
    print("   • Adversarial domain adaptation")
    print("   • Multi-task learning capabilities")
    print("   • Progressive layer unfreezing")
    print("   • Knowledge distillation")
    
    print("\n🧠 **Active Learning & Data Intelligence**")
    print("   • Intelligent data sampling strategies")
    print("   • Uncertainty-based sample selection")
    print("   • Diversity-aware data collection")
    print("   • Human-in-the-loop labeling")
    print("   • Continuous model improvement")
    
    print("\n🔮 **Enhanced Content Analysis v3.0**")
    print("   • Neural network viral prediction")
    print("   • Multi-emotion sentiment analysis")
    print("   • Context-aware optimization")
    print("   • Engagement forecasting")
    print("   • AI-powered recommendations")
    
    print("\n📊 **Advanced Analytics & Insights**")
    print("   • Predictive performance analytics")
    print("   • Trend forecasting & analysis")
    print("   • Comparative content analysis")
    print("   • ROI optimization insights")
    print("   • Competitive intelligence")
    
    print("\n🏥 **System Health & Performance**")
    print("   • Real-time system monitoring")
    print("   • Performance optimization")
    print("   • Resource management")
    print("   • Error detection & recovery")
    print("   • Automated optimization recommendations")

def display_technical_specs():
    """Display technical specifications"""
    print("\n⚙️ **Technical Specifications v3.1**")
    print("=" * 60)
    
    print("\n🏗️ **Architecture**")
    print("   • Modular microservices architecture")
    print("   • RESTful API endpoints")
    print("   • Real-time WebSocket communication")
    print("   • Distributed computing support")
    print("   • Scalable cloud deployment ready")
    
    print("\n🧠 **AI Models**")
    print("   • PyTorch-based neural networks")
    print("   • Transformer architectures")
    print("   • Multi-modal learning capabilities")
    print("   • Federated learning algorithms")
    print("   • Transfer learning frameworks")
    
    print("\n📊 **Data Processing**")
    print("   • Real-time streaming analytics")
    print("   • Big data processing capabilities")
    print("   • Automated data validation")
    print("   • Intelligent data augmentation")
    print("   • Synthetic data generation")
    
    print("\n🔒 **Security & Privacy**")
    print("   • End-to-end encryption")
    print("   • Differential privacy protection")
    print("   • Secure multi-party computation")
    print("   • GDPR compliance features")
    print("   • Audit logging & monitoring")

def display_use_cases():
    """Display use cases and applications"""
    print("\n🎯 **Use Cases & Applications v3.1**")
    print("=" * 60)
    
    print("\n🏢 **Enterprise Organizations**")
    print("   • Multi-brand content optimization")
    print("   • Cross-department collaboration")
    print("   • Secure data sharing")
    print("   • Scalable AI deployment")
    print("   • ROI optimization")
    
    print("\n📱 **Social Media Agencies**")
    print("   • Multi-client management")
    print("   • Content performance optimization")
    print("   • Audience targeting")
    print("   • Campaign optimization")
    print("   • Competitive analysis")
    
    print("\n🎨 **Content Creators**")
    print("   • Viral content optimization")
    print("   • Audience engagement maximization")
    print("   • Trend prediction")
    print("   • Performance analytics")
    print("   • Growth optimization")
    
    print("\n🔬 **Research & Development**")
    print("   • AI model research")
    print("   • Federated learning experiments")
    print("   • Transfer learning studies")
    print("   • Active learning research")
    print("   • Performance benchmarking")

if __name__ == "__main__":
    print("🚀 Advanced Learning System v3.1 - Initialization")
    print("=" * 60)
    
    # Display v3.1 features
    display_v3_1_features()
    
    # Display technical specifications
    display_technical_specs()
    
    # Display use cases
    display_use_cases()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check PyTorch compatibility
    if not check_pytorch_compatibility():
        print("⚠️ PyTorch compatibility issues detected. System may run in limited mode.")
    
    # Display system info
    system_info()
    
    print("\n" + "=" * 60)
    print("🎯 Starting Advanced Learning System v3.1...")
    print("=" * 60)
    
    # Launch the system
    main()
