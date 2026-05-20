"""
Health Check Script
===================

Check system health and configuration.
"""

import sys
from pathlib import Path


def check_imports():
    """Check if all imports work."""
    print("Checking imports...")
    
    try:
        from image_upscaling_ai.core.upscaling_service import UpscalingService
        print("  ✅ UpscalingService")
    except Exception as e:
        print(f"  ❌ UpscalingService: {e}")
        return False
    
    try:
        from image_upscaling_ai.models import RealESRGANModelManager
        print("  ✅ RealESRGANModelManager")
    except Exception as e:
        print(f"  ⚠️  RealESRGANModelManager: {e}")
    
    try:
        from image_upscaling_ai.utils import ImageUtils
        print("  ✅ ImageUtils")
    except Exception as e:
        print(f"  ❌ ImageUtils: {e}")
        return False
    
    return True


def check_configuration():
    """Check configuration."""
    print("\nChecking configuration...")
    
    try:
        from image_upscaling_ai.config.upscaling_config import UpscalingConfig
        
        config = UpscalingConfig.from_env()
        validation = config.validate()
        
        if validation:
            print("  ✅ Configuration valid")
        else:
            print("  ⚠️  Configuration has warnings")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def check_directories():
    """Check required directories."""
    print("\nChecking directories...")
    
    directories = ["output", "cache"]
    all_exist = True
    
    for dir_name in directories:
        path = Path(dir_name)
        if path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ⚠️  {dir_name}/ (will be created on first use)")
    
    return all_exist


def check_dependencies():
    """Check dependencies."""
    print("\nChecking dependencies...")
    
    required = {
        "PIL": "pillow",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "pydantic": "pydantic"
    }
    
    optional = {
        "cv2": "opencv-python-headless",
        "torch": "torch",
        "realesrgan": "realesrgan"
    }
    
    all_ok = True
    
    for module, package in required.items():
        try:
            if module == "PIL":
                __import__("PIL")
            else:
                __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (required)")
            all_ok = False
    
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            print(f"  ⚠️  {package} (optional, not installed)")
    
    return all_ok


def main():
    """Run health check."""
    print("="*60)
    print("Image Upscaling AI - Health Check")
    print("="*60)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("Configuration", check_configuration),
        ("Directories", check_directories),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Health Check Summary")
    print("="*60)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    if all_passed:
        print("\n✅ System is healthy!")
        return 0
    else:
        print("\n❌ Some checks failed. Please review above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


