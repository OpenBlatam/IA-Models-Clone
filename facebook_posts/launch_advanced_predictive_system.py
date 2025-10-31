#!/usr/bin/env python3
"""
Advanced Predictive Facebook Content Optimization System v3.0 - Launcher
This script launches the next-generation AI-powered prediction and forecasting system
"""

import sys
import os
import time
from datetime import datetime

def main():
    """Launch the advanced predictive system"""
    print("🚀 Advanced Predictive Facebook Content Optimization System v3.0")
    print("=" * 80)
    print("🎯 Next-generation AI-powered prediction and forecasting capabilities")
    print("=" * 80)
    
    try:
        # Import and launch the advanced predictive system
        from advanced_predictive_interface import AdvancedPredictiveInterface
        
        print("✅ Advanced Predictive System imported successfully!")
        print("🖥️ Launching Advanced Gradio Interface...")
        print("\n🔮 New v3.0 Features:")
        print("   - Advanced Viral Prediction with Neural Networks")
        print("   - Context-Aware Sentiment Analysis with Emotion Detection")
        print("   - Intelligent Audience Segmentation using ML")
        print("   - Real-time Engagement Forecasting")
        print("   - Temporal Pattern Analysis")
        print("   - Advanced Context Analysis")
        print("   - AI-Powered Recommendations Engine")
        
        print("\n🌐 Interface Tabs:")
        print("   - 🔮 Advanced Content Analysis: Multi-model content analysis")
        print("   - 📊 Predictive Analytics: Real-time predictions and forecasting")
        print("   - 🎯 Audience Intelligence: Advanced segmentation and behavioral analysis")
        print("   - 📈 Performance Forecasting: Future performance predictions")
        print("   - 🏥 System Health & Metrics: Comprehensive monitoring")
        
        print("\n🚀 System Capabilities:")
        print("   - Neural Network-based Viral Prediction")
        print("   - Multi-modal Sentiment Analysis")
        print("   - Machine Learning Audience Segmentation")
        print("   - Temporal Pattern Recognition")
        print("   - Context-Aware Optimization")
        print("   - Real-time Learning and Adaptation")
        
        print("\n⏰ Initializing Advanced AI Models...")
        
        # Create and launch the interface
        interface = AdvancedPredictiveInterface()
        
        print("✅ Advanced Predictive Interface created successfully!")
        print("🌐 The interface will open in your browser...")
        print("   - URL: http://localhost:7862")
        print("   - Port: 7862 (different from v2.0 to avoid conflicts)")
        print("   - Features: 5 specialized tabs with advanced AI capabilities")
        
        print("\n🎯 Ready to launch! Starting the interface...")
        time.sleep(2)
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7862,        # Use port 7862 for v3.0
            share=False,             # Set to True to create a public link
            debug=True,              # Enable debug mode
            show_error=True,         # Show error details
            quiet=False              # Show all output
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Please ensure all dependencies are installed:")
        print("   pip install -r requirements_enhanced_system.txt")
        print("   pip install scikit-learn pandas plotly")
        print("\n📚 Required packages:")
        print("   - torch (PyTorch)")
        print("   - gradio")
        print("   - scikit-learn")
        print("   - pandas")
        print("   - plotly")
        print("   - numpy")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error launching system: {e}")
        print("Please check the logs for more details.")
        print("\n🔍 Common issues:")
        print("   - Port 7862 already in use (try different port)")
        print("   - Missing dependencies")
        print("   - System resources insufficient")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    required_packages = [
        'torch', 'gradio', 'sklearn', 'pandas', 
        'plotly', 'numpy', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'plotly':
                import plotly
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

if __name__ == "__main__":
    print("🚀 Advanced Predictive System v3.0 - Initialization")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Display system info
    system_info()
    
    print("\n" + "=" * 60)
    print("🎯 Starting Advanced Predictive System...")
    print("=" * 60)
    
    # Launch the system
    main()

