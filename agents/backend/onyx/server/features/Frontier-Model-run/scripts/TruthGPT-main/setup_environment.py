"""
Environment Setup Script
Helps set up the testing environment
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python():
    """Check if Python is available"""
    try:
        version = sys.version_info
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} found")
        return True
    except Exception as e:
        print(f"❌ Python check failed: {e}")
        return False

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False
    except Exception as e:
        print(f"❌ Error installing {package}: {e}")
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required = ['torch', 'numpy']
    optional = ['psutil']
    
    print("\nChecking dependencies...")
    missing_required = []
    missing_optional = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_required.append(package)
    
    for package in optional:
        try:
            __import__(package)
            print(f"✅ {package} is installed (optional)")
        except ImportError:
            print(f"⚠️  {package} is missing (optional)")
            missing_optional.append(package)
    
    # Install missing required packages
    if missing_required:
        print(f"\nInstalling required packages: {', '.join(missing_required)}")
        for package in missing_required:
            if not install_package(package):
                return False
    
    # Ask about optional packages
    if missing_optional:
        print(f"\nOptional packages available: {', '.join(missing_optional)}")
        response = input("Install optional packages? (y/n): ").lower().strip()
        if response == 'y':
            for package in missing_optional:
                install_package(package)
    
    return True

def verify_setup():
    """Verify the setup is correct"""
    print("\nVerifying setup...")
    
    # Check directory
    if not Path("core").exists():
        print("❌ core/ directory not found")
        return False
    if not Path("tests").exists():
        print("❌ tests/ directory not found")
        return False
    if not Path("run_unified_tests.py").exists():
        print("❌ run_unified_tests.py not found")
        return False
    
    print("✅ Directory structure is correct")
    
    # Check imports
    try:
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from core import OptimizationEngine, ModelManager
        print("✅ Core modules can be imported")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("TruthGPT Environment Setup")
    print("=" * 60)
    print()
    
    # Check Python
    if not check_python():
        print("\n❌ Python is required. Please install Python 3.7+ from:")
        print("   https://www.python.org/downloads/")
        return 1
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        print("\n❌ Failed to install required dependencies")
        return 1
    
    # Verify setup
    if not verify_setup():
        print("\n❌ Setup verification failed")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! You can now run tests.")
    print("=" * 60)
    print()
    print("Run tests with:")
    print("  python run_unified_tests.py")
    print("  or")
    print("  python quick_check.py  # Quick environment check")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())







