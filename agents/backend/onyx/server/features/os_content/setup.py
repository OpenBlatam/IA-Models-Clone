#!/usr/bin/env python3
"""
Setup script for OS Content System
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import json
import yaml

class OSContentSetup:
    """Setup manager for OS Content System"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.venv_dir = base_dir / "venv"
        self.requirements_file = base_dir / "requirements.txt"
        self.env_file = base_dir / ".env"
        self.env_example = base_dir / "env.example"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ is required")
            return False
        
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
        return True
    
    def check_dependencies(self) -> List[str]:
        """Check system dependencies"""
        missing_deps = []
        
        # Check for essential system packages
        essential_packages = [
            ("git", "git --version"),
            ("docker", "docker --version"),
            ("docker-compose", "docker-compose --version"),
        ]
        
        for package, command in essential_packages:
            try:
                subprocess.run(command.split(), capture_output=True, check=True)
                print(f"✅ {package} found")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"⚠️  {package} not found")
                missing_deps.append(package)
        
        return missing_deps
    
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        try:
            if self.venv_dir.exists():
                print(f"📁 Virtual environment already exists at {self.venv_dir}")
                return True
            
            print(f"🔧 Creating virtual environment at {self.venv_dir}")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
            
            print("✅ Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def get_python_executable(self) -> Path:
        """Get Python executable path"""
        if os.name == "nt":  # Windows
            return self.venv_dir / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            return self.venv_dir / "bin" / "python"
    
    def get_pip_executable(self) -> Path:
        """Get pip executable path"""
        if os.name == "nt":  # Windows
            return self.venv_dir / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            return self.venv_dir / "bin" / "pip"
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        try:
            pip_exe = self.get_pip_executable()
            print("🔧 Upgrading pip...")
            subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
            print("✅ Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upgrade pip: {e}")
            return False
    
    def install_requirements(self) -> bool:
        """Install Python requirements"""
        try:
            if not self.requirements_file.exists():
                print("❌ Requirements file not found")
                return False
            
            pip_exe = self.get_pip_executable()
            print("🔧 Installing Python requirements...")
            
            # Install requirements with verbose output
            result = subprocess.run([
                str(pip_exe), "install", "-r", str(self.requirements_file),
                "--verbose"
            ], check=True)
            
            print("✅ Requirements installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    
    def setup_environment_file(self) -> bool:
        """Setup environment configuration file"""
        try:
            if self.env_file.exists():
                print(f"📁 Environment file already exists at {self.env_file}")
                return True
            
            if not self.env_example.exists():
                print("❌ Environment example file not found")
                return False
            
            # Copy example to .env
            shutil.copy2(self.env_example, self.env_file)
            print(f"✅ Environment file created at {self.env_file}")
            print("⚠️  Please edit .env file with your configuration values")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup environment file: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        try:
            directories = [
                "logs",
                "data",
                "temp",
                "models",
                "uploads",
                "cache"
            ]
            
            for directory in directories:
                dir_path = self.base_dir / directory
                dir_path.mkdir(exist_ok=True)
                print(f"📁 Created directory: {directory}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create directories: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run the test suite"""
        try:
            python_exe = self.get_python_executable()
            test_file = self.base_dir / "test_suite.py"
            
            if not test_file.exists():
                print("⚠️  Test suite not found, skipping tests")
                return True
            
            print("🧪 Running test suite...")
            result = subprocess.run([
                str(python_exe), str(test_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Tests passed successfully")
                return True
            else:
                print("❌ Tests failed")
                print("Test output:")
                print(result.stdout)
                print("Test errors:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Failed to run tests: {e}")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create startup scripts for different platforms"""
        try:
            # Windows batch file
            if os.name == "nt":
                bat_file = self.base_dir / "start.bat"
                with open(bat_file, 'w') as f:
                    f.write(f"""@echo off
echo Starting OS Content System...
cd /d "{self.base_dir}"
call "{self.venv_dir}\\Scripts\\activate.bat"
python main.py
pause
""")
                print("✅ Created start.bat for Windows")
            
            # Unix/Linux/macOS shell script
            else:
                sh_file = self.base_dir / "start.sh"
                with open(sh_file, 'w') as f:
                    f.write(f"""#!/bin/bash
echo "Starting OS Content System..."
cd "{self.base_dir}"
source "{self.venv_dir}/bin/activate"
python main.py
""")
                
                # Make executable
                os.chmod(sh_file, 0o755)
                print("✅ Created start.sh for Unix/Linux/macOS")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create startup scripts: {e}")
            return False
    
    def setup_database(self) -> bool:
        """Setup database (placeholder for future implementation)"""
        print("⚠️  Database setup not implemented yet")
        print("   Please configure your database manually in the .env file")
        return True
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 OS Content System Setup Complete!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. Edit the .env file with your configuration")
        print("2. Configure your database connection")
        print("3. Set up Redis if you plan to use caching")
        print("4. Configure external API keys if needed")
        print("\n🚀 To start the system:")
        
        if os.name == "nt":
            print("   Windows: Double-click start.bat or run 'start.bat'")
        else:
            print("   Unix/Linux/macOS: ./start.sh or python main.py")
        
        print("\n📚 Documentation:")
        print("   - Check the docs/ directory for detailed guides")
        print("   - API documentation will be available at http://localhost:8000/docs")
        print("   - Health check at http://localhost:8000/health")
        
        print("\n🔧 Development:")
        print("   - Activate virtual environment: source venv/bin/activate (Unix) or venv\\Scripts\\activate (Windows)")
        print("   - Run tests: python test_suite.py")
        print("   - Install new packages: pip install package_name")
        
        print("\n📞 Support:")
        print("   - Check logs in the logs/ directory")
        print("   - Review error logs for troubleshooting")
        print("   - Check the README.md for common issues")
    
    def run_setup(self, skip_tests: bool = False) -> bool:
        """Run the complete setup process"""
        print("🚀 Starting OS Content System Setup...")
        print("="*60)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check system dependencies
        missing_deps = self.check_dependencies()
        if missing_deps:
            print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
            print("   Some features may not work correctly")
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Upgrade pip
        if not self.upgrade_pip():
            return False
        
        # Install requirements
        if not self.install_requirements():
            return False
        
        # Setup environment file
        if not self.setup_environment_file():
            return False
        
        # Create directories
        if not self.create_directories():
            return False
        
        # Setup database
        if not self.setup_database():
            return False
        
        # Create startup scripts
        if not self.create_startup_scripts():
            return False
        
        # Run tests (optional)
        if not skip_tests:
            if not self.run_tests():
                print("⚠️  Tests failed, but setup can continue")
        
        # Print next steps
        self.print_next_steps()
        
        return True

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="OS Content System Setup")
    parser.add_argument(
        "--skip-tests", 
        action="store_true", 
        help="Skip running tests during setup"
    )
    parser.add_argument(
        "--base-dir", 
        type=str, 
        default=".", 
        help="Base directory for installation"
    )
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = Path(args.base_dir).resolve()
    
    if not base_dir.exists():
        print(f"❌ Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Create setup manager
    setup = OSContentSetup(base_dir)
    
    # Run setup
    success = setup.run_setup(skip_tests=args.skip_tests)
    
    if success:
        print("\n✅ Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
