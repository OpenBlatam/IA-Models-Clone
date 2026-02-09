#!/usr/bin/env python3
"""
Setup Tools
===========
Setup and installation script for API tools.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any


class ToolSetup:
    """Tool setup manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_packages = [
            "requests",
            "pytest",
            "playwright",
            "pytest-playwright"
        ]
    
    def check_python_version(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ is required")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if dependencies are installed."""
        results = {}
        
        for package in self.required_packages:
            try:
                __import__(package.replace("-", "_"))
                results[package] = True
                print(f"✅ {package} is installed")
            except ImportError:
                results[package] = False
                print(f"❌ {package} is NOT installed")
        
        return results
    
    def install_dependencies(self):
        """Install required dependencies."""
        print("📦 Installing dependencies...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True
            )
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
        
        # Install Playwright browsers
        print("🌐 Installing Playwright browsers...")
        try:
            subprocess.run(
                [sys.executable, "-m", "playwright", "install"],
                check=True
            )
            print("✅ Playwright browsers installed")
        except subprocess.CalledProcessError:
            print("⚠️  Playwright browser installation failed (optional)")
        
        return True
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            "test_results",
            "debug_output",
            "analytics",
            "screenshots",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def setup_environment(self):
        """Setup environment variables."""
        env_file = self.project_root / ".env"
        env_example = self.project_root / "env.example"
        
        if not env_file.exists() and env_example.exists():
            print("📝 Creating .env file from example...")
            with open(env_example, "r") as f:
                content = f.read()
            with open(env_file, "w") as f:
                f.write(content)
            print("✅ .env file created")
        elif env_file.exists():
            print("✅ .env file already exists")
    
    def verify_setup(self) -> bool:
        """Verify setup is complete."""
        print("\n🔍 Verifying setup...")
        
        all_good = True
        
        # Check Python
        if not self.check_python_version():
            all_good = False
        
        # Check dependencies
        deps = self.check_dependencies()
        if not all(deps.values()):
            print("\n⚠️  Some dependencies are missing")
            all_good = False
        
        # Check directories
        required_dirs = ["test_results", "debug_output", "analytics"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                print(f"❌ Directory missing: {dir_name}")
                all_good = False
        
        if all_good:
            print("\n✅ Setup verification passed!")
        else:
            print("\n⚠️  Setup verification found issues")
        
        return all_good
    
    def run_full_setup(self):
        """Run full setup process."""
        print("=" * 70)
        print("🚀 API Tools Setup")
        print("=" * 70)
        print()
        
        # Check Python
        if not self.check_python_version():
            print("❌ Setup failed: Python version incompatible")
            return False
        
        print()
        
        # Check and install dependencies
        deps = self.check_dependencies()
        if not all(deps.values()):
            print("\n📦 Installing missing dependencies...")
            if not self.install_dependencies():
                return False
        else:
            print("\n✅ All dependencies are installed")
        
        print()
        
        # Create directories
        print("📁 Creating directories...")
        self.create_directories()
        
        print()
        
        # Setup environment
        print("⚙️  Setting up environment...")
        self.setup_environment()
        
        print()
        
        # Verify
        return self.verify_setup()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup API Tools")
    parser.add_argument("--check", action="store_true", help="Check setup only")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--verify", action="store_true", help="Verify setup")
    
    args = parser.parse_args()
    
    setup = ToolSetup()
    
    if args.check:
        setup.check_python_version()
        setup.check_dependencies()
    elif args.install:
        setup.install_dependencies()
    elif args.verify:
        setup.verify_setup()
    else:
        setup.run_full_setup()


if __name__ == "__main__":
    main()



