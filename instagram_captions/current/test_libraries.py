#!/usr/bin/env python3
"""
Test script to check library installations
"""

def test_libraries():
    """Test if required libraries are installed"""
    
    print("🔍 Testing Library Installations...")
    print("=" * 50)
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch: Not installed")
    
    # Test Transformers
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers: Not installed")
    
    # Test Diffusers
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        print("❌ Diffusers: Not installed")
    
    # Test Gradio
    try:
        import gradio
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio: Not installed")
    
    # Test FastAPI
    try:
        import fastapi
        print(f"✅ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI: Not installed")
    
    # Test NumPy
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy: Not installed")
    
    # Test Pandas
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas: Not installed")
    
    # Test Scikit-learn
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn: Not installed")
    
    print("=" * 50)
    print("🎯 Library Test Complete!")

if __name__ == "__main__":
    test_libraries()





