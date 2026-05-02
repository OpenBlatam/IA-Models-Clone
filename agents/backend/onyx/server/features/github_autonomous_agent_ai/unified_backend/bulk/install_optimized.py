"""
BUL Optimized Installation Script
=================================

Automated installation and setup for the optimized BUL system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please install Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available."""
    print("📦 Checking pip availability...")
    
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not available")
        print("   Please install pip")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    requirements_file = "requirements_optimized.txt"
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    try:
        # Install dependencies
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_environment():
    """Setup environment configuration."""
    print("🔧 Setting up environment...")
    
    env_template = "env_optimized.txt"
    env_file = ".env"
    
    if not Path(env_template).exists():
        print(f"❌ Environment template not found: {env_template}")
        return False
    
    if Path(env_file).exists():
        print(f"⚠️  Environment file already exists: {env_file}")
        response = input("   Overwrite? (y/N): ").lower().strip()
        if response != 'y':
            print("   Keeping existing environment file")
            return True
    
    try:
        # Copy template to .env
        shutil.copy2(env_template, env_file)
        print(f"✅ Environment file created: {env_file}")
        print("   Please edit .env with your configuration")
        return True
        
    except Exception as e:
        print(f"❌ Error creating environment file: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "generated_documents",
        "logs",
        "test_output"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")
            return False
    
    return True

def validate_installation():
    """Validate the installation."""
    print("🔍 Validating installation...")
    
    try:
        # Test imports
        from modules import DocumentProcessor, QueryAnalyzer, BusinessAgentManager
        from config_optimized import BULConfig
        from bul_optimized import BULSystem
        
        print("✅ All modules imported successfully")
        
        # Test configuration
        config = BULConfig()
        print("✅ Configuration loaded successfully")
        
        # Test basic functionality
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("Test query")
        if analysis.primary_area:
            print("✅ Basic functionality working")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def show_next_steps():
    """Show next steps to the user."""
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your configuration:")
    print("   - Set API keys if needed")
    print("   - Adjust other settings as required")
    
    print("\n2. Start the system:")
    print("   python start_optimized.py")
    
    print("\n3. Access the API:")
    print("   http://localhost:8000/docs")
    
    print("\n4. Run tests:")
    print("   python test_optimized.py")
    
    print("\n5. Run demo:")
    print("   python demo_optimized.py")
    
    print("\n6. Validate system:")
    print("   python validate_system.py")
    
    print("\n📚 Documentation:")
    print("   README_OPTIMIZED.md - Complete documentation")
    print("   OPTIMIZATION_SUMMARY.md - Optimization details")

def main():
    """Main installation function."""
    print("🚀 BUL - Business Universal Language (Optimized)")
    print("=" * 60)
    print("Automated Installation Script")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    if not check_pip():
        return 1
    
    # Installation steps
    steps = [
        ("Installing Dependencies", install_dependencies),
        ("Setting up Environment", setup_environment),
        ("Creating Directories", create_directories),
        ("Validating Installation", validate_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ {step_name} failed")
            return 1
    
    # Show next steps
    show_next_steps()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
