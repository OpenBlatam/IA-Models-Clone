"""
Dependency Checker for TruthGPT
Checks if all required dependencies are installed
"""

import sys
import importlib
from typing import List, Tuple, Dict

REQUIRED_PACKAGES = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'dataclasses': 'dataclasses (Python 3.7+)',
}

OPTIONAL_PACKAGES = {
    'pytest': 'pytest (for alternative test runner)',
    'coverage': 'coverage (for code coverage)',
}

def check_package(package_name: str, display_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed"""
    if display_name is None:
        display_name = package_name
    
    try:
        importlib.import_module(package_name)
        return True, f"✅ {display_name}"
    except ImportError:
        return False, f"❌ {display_name} (not installed)"

def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.7+)"

def main():
    """Main function"""
    print("=" * 60)
    print("TruthGPT Dependency Checker")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("Python Version:")
    ok, msg = check_python_version()
    print(f"  {msg}")
    if not ok:
        all_ok = False
    print()
    
    # Check required packages
    print("Required Packages:")
    for package, display_name in REQUIRED_PACKAGES.items():
        ok, msg = check_package(package, display_name)
        print(f"  {msg}")
        if not ok:
            all_ok = False
    print()
    
    # Check optional packages
    print("Optional Packages:")
    for package, display_name in OPTIONAL_PACKAGES.items():
        ok, msg = check_package(package, display_name)
        print(f"  {msg}")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ All required dependencies are installed!")
        print("You can run tests with: python run_unified_tests.py")
        return 0
    else:
        print("❌ Some required dependencies are missing!")
        print()
        print("Install missing packages with:")
        print("  pip install torch numpy")
        return 1

if __name__ == "__main__":
    sys.exit(main())







