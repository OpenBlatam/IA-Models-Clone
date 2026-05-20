"""
Quick Environment Check
Fast validation of environment and dependencies
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.7+)"

def check_directory():
    """Check if we're in the right directory"""
    core_exists = Path("core").exists()
    tests_exists = Path("tests").exists()
    runner_exists = Path("run_unified_tests.py").exists()
    
    if core_exists and tests_exists and runner_exists:
        return True, "Correct directory"
    return False, "Missing core/, tests/, or run_unified_tests.py"

def check_dependencies():
    """Check critical dependencies"""
    results = {}
    
    try:
        import torch
        results['torch'] = (True, f"PyTorch {torch.__version__}")
    except ImportError:
        results['torch'] = (False, "PyTorch not installed")
    
    try:
        import numpy
        results['numpy'] = (True, f"NumPy {numpy.__version__}")
    except ImportError:
        results['numpy'] = (False, "NumPy not installed")
    
    return results

def check_imports():
    """Check if core modules can be imported"""
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from core import OptimizationEngine, ModelManager, TrainingManager
        return True, "Core imports successful"
    except Exception as e:
        return False, f"Import error: {str(e)}"

def main():
    """Main check function"""
    print("=" * 60)
    print("TruthGPT Quick Environment Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("1. Python Version:")
    ok, msg = check_python_version()
    status = "✅" if ok else "❌"
    print(f"   {status} {msg}")
    if not ok:
        all_ok = False
    print()
    
    # Check directory
    print("2. Directory Structure:")
    ok, msg = check_directory()
    status = "✅" if ok else "❌"
    print(f"   {status} {msg}")
    if not ok:
        all_ok = False
    print()
    
    # Check dependencies
    print("3. Dependencies:")
    deps = check_dependencies()
    for name, (ok, msg) in deps.items():
        status = "✅" if ok else "❌"
        print(f"   {status} {name}: {msg}")
        if not ok:
            all_ok = False
    print()
    
    # Check imports
    print("4. Core Module Imports:")
    ok, msg = check_imports()
    status = "✅" if ok else "❌"
    print(f"   {status} {msg}")
    if not ok:
        all_ok = False
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ All checks passed! Ready to run tests.")
        print()
        print("Run tests with:")
        print("  python run_unified_tests.py")
        print("  or")
        print("  run_tests.bat (Windows)")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues above.")
        print()
        print("Quick fixes:")
        print("  - Install Python 3.7+: https://www.python.org/downloads/")
        print("  - Install dependencies: pip install torch numpy")
        print("  - Make sure you're in the TruthGPT-main directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())







