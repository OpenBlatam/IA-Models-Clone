#!/usr/bin/env python3
"""
Simple PyTorch Installation Test for Video-OpusClip

Run this script after installing Python and PyTorch to verify everything works.
"""

def test_pytorch_basic():
    """Test basic PyTorch functionality."""
    print("🔥 Testing PyTorch Installation")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("✅ Basic tensor operations: OK")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.version.cuda}")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            
            # Test GPU operations
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print("✅ GPU operations: OK")
        else:
            print("⚠️  CUDA not available - CPU only mode")
        
        # Test neural network
        import torch.nn as nn
        model = nn.Linear(10, 1)
        input_tensor = torch.randn(5, 10)
        output = model(input_tensor)
        print("✅ Neural network: OK")
        
        print("\n🎉 PyTorch installation successful!")
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch not installed: {e}")
        print("Install with: pip install torch torchvision torchaudio")
        return False
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_video_opusclip_imports():
    """Test Video-OpusClip specific imports."""
    print("\n📦 Testing Video-OpusClip Imports")
    print("=" * 40)
    
    modules_to_test = [
        'numpy',
        'torch',
        'torchvision',
        'opencv-python',
        'Pillow',
        'transformers',
        'diffusers',
        'gradio',
        'fastapi'
    ]
    
    failed_imports = []
    successful_imports = []
    
    for module in modules_to_test:
        try:
            if module == 'opencv-python':
                import cv2
                version = cv2.__version__
            else:
                imported_module = __import__(module.replace('-', '_'))
                version = getattr(imported_module, '__version__', 'unknown')
            
            successful_imports.append(f"✅ {module} ({version})")
        except ImportError:
            failed_imports.append(f"❌ {module}")
    
    print("Successful imports:")
    for imp in successful_imports:
        print(f"  {imp}")
    
    if failed_imports:
        print("\nFailed imports:")
        for imp in failed_imports:
            print(f"  {imp}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements_basic.txt")
        return False
    else:
        print("\n🎉 All Video-OpusClip dependencies available!")
        return True

def main():
    """Main test function."""
    print("🚀 Video-OpusClip PyTorch Installation Test")
    print("=" * 50)
    
    # Test PyTorch
    pytorch_ok = test_pytorch_basic()
    
    # Test Video-OpusClip imports
    imports_ok = test_video_opusclip_imports()
    
    print("\n" + "=" * 50)
    if pytorch_ok and imports_ok:
        print("🎉 All tests passed! Your Video-OpusClip system is ready.")
        print("\nNext steps:")
        print("1. Run: python torch_setup_check.py")
        print("2. Try: python quick_start_mixed_precision.py")
        print("3. Check the guides in this directory")
    else:
        print("❌ Some tests failed. Please install missing dependencies.")
        print("\nInstallation commands:")
        print("1. Install PyTorch: pip install torch torchvision torchaudio")
        print("2. Install dependencies: pip install -r requirements_basic.txt")
        print("3. Run this test again")

if __name__ == "__main__":
    main() 