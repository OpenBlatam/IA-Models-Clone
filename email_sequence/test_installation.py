from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import sys
from pathlib import Path
        import torch
        import transformers
        import datasets
        import numpy as np
        import pandas as pd
        import sklearn
        import gradio
        import yaml
        import dotenv
        from loguru import logger
        from tqdm import tqdm
        from pydantic import BaseModel
        import torch
        import numpy as np
        import pandas as pd
        from transformers import AutoTokenizer
        import gradio as gr
    import platform
        import torch
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Test script to verify Email Sequence AI System installation
"""


def test_imports():
    """Test all core imports."""
    print("🧪 Testing Email Sequence AI System Installation")
    print("=" * 50)
    
    # Test core ML/AI libraries
    try:
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets: {e}")
        return False
    
    # Test data processing libraries
    try:
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    # Test web interface
    try:
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError as e:
        print(f"❌ Gradio: {e}")
        return False
    
    # Test configuration and utilities
    try:
        print("✅ PyYAML")
    except ImportError as e:
        print(f"❌ PyYAML: {e}")
        return False
    
    try:
        print("✅ python-dotenv")
    except ImportError as e:
        print(f"❌ python-dotenv: {e}")
        return False
    
    try:
        print("✅ Loguru")
    except ImportError as e:
        print(f"❌ Loguru: {e}")
        return False
    
    try:
        print("✅ tqdm")
    except ImportError as e:
        print(f"❌ tqdm: {e}")
        return False
    
    try:
        print("✅ Pydantic")
    except ImportError as e:
        print(f"❌ Pydantic: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 50)
    
    # Test PyTorch tensor operations
    try:
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        print("✅ PyTorch tensor operations")
    except Exception as e:
        print(f"❌ PyTorch tensor operations: {e}")
        return False
    
    # Test NumPy operations
    try:
        arr = np.random.randn(3, 3)
        result = np.linalg.inv(arr)
        print("✅ NumPy operations")
    except Exception as e:
        print(f"❌ NumPy operations: {e}")
        return False
    
    # Test Pandas operations
    try:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = df.describe()
        print("✅ Pandas operations")
    except Exception as e:
        print(f"❌ Pandas operations: {e}")
        return False
    
    # Test Transformers
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        print("✅ Transformers tokenizer")
    except Exception as e:
        print(f"❌ Transformers tokenizer: {e}")
        return False
    
    # Test Gradio
    try:
        print("✅ Gradio import")
    except Exception as e:
        print(f"❌ Gradio import: {e}")
        return False
    
    return True

def test_system_info():
    """Display system information."""
    print("\n💻 System Information")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Processor: {platform.processor()}")
    
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available")

def main():
    """Main test function."""
    print("🚀 Email Sequence AI System - Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test functionality
        functionality_ok = test_basic_functionality()
        
        # Display system info
        test_system_info()
        
        if functionality_ok:
            print("\n🎉 All tests passed! Installation is successful.")
            print("\nNext steps:")
            print("1. Run the basic demo: python examples/basic_demo.py")
            print("2. Start training: python examples/training_example.py")
            print("3. Launch Gradio app: python examples/gradio_app.py")
            return True
        else:
            print("\n⚠️  Some functionality tests failed.")
            return False
    else:
        print("\n❌ Import tests failed. Please check your installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 