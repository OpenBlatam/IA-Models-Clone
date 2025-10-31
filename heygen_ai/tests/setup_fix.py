"""
Setup Fix Script
===============

Script to fix and setup the Ultimate Breakthrough Test Case Generation System.
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_installation():
    """Verify that all required packages are installed"""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        "numpy",
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "pytest-mock",
        "pytest-benchmark",
        "pytest-xdist",
        "pytest-html",
        "pytest-json-report"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All packages installed successfully")
        return True

def run_tests():
    """Run the test system"""
    print("\n🧪 Running tests...")
    
    try:
        # Import and run the test
        from test_system_fix import test_imports, test_basic_functionality
        test_imports()
        test_basic_functionality()
        print("✅ Tests completed successfully")
        return True
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ULTIMATE BREAKTHROUGH SYSTEM FIX")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found. Please run this script from the tests directory.")
        return False
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        return False
    
    # Run tests
    if not run_tests():
        print("❌ Tests failed")
        return False
    
    print("\n🎉 SYSTEM FIX COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("The Ultimate Breakthrough Test Case Generation System is now ready to use!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
