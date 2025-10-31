#!/usr/bin/env python3
"""
Setup Script for Copywriting Service
===================================

Script to set up the development environment and initialize the service.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

# Add the improved directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from services import get_copywriting_service, cleanup_copywriting_service


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def install_dependencies() -> bool:
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")


def setup_environment() -> bool:
    """Set up environment variables"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔄 Setting up environment file...")
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ Environment file created from example")
            print("⚠️  Please update .env with your actual configuration values")
            return True
        except Exception as e:
            print(f"❌ Failed to create environment file: {e}")
            return False
    else:
        print("✅ Environment file already exists")
        return True


def create_directories() -> bool:
    """Create necessary directories"""
    directories = ["logs", "data", "cache"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"🔄 Creating directory: {directory}")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Directory created: {directory}")
            except Exception as e:
                print(f"❌ Failed to create directory {directory}: {e}")
                return False
        else:
            print(f"✅ Directory already exists: {directory}")
    
    return True


async def test_service() -> bool:
    """Test the service startup"""
    print("🔄 Testing service startup...")
    try:
        service = await get_copywriting_service()
        health = await service.get_health_status()
        print(f"✅ Service health check: {health.status}")
        await cleanup_copywriting_service()
        return True
    except Exception as e:
        print(f"❌ Service test failed: {e}")
        return False


def run_tests() -> bool:
    """Run the test suite"""
    return run_command("python -m pytest tests/ -v", "Running test suite")


def check_code_quality() -> bool:
    """Check code quality with linting tools"""
    commands = [
        ("python -m black --check .", "Checking code formatting with Black"),
        ("python -m isort --check-only .", "Checking import sorting with isort"),
        ("python -m flake8 .", "Checking code style with flake8"),
        ("python -m mypy .", "Checking type hints with mypy")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def format_code() -> bool:
    """Format code with Black and isort"""
    commands = [
        ("python -m black .", "Formatting code with Black"),
        ("python -m isort .", "Sorting imports with isort")
    ]
    
    all_passed = True
    for command, description in commands:
        if not run_command(command, description):
            all_passed = False
    
    return all_passed


def main():
    """Main setup function"""
    print("🚀 Setting up Copywriting Service...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found. Please run this script from the improved directory.")
        sys.exit(1)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment),
        ("Creating directories", create_directories),
        ("Testing service", lambda: asyncio.run(test_service())),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            failed_steps.append(step_name)
    
    if failed_steps:
        print(f"\n❌ Setup failed for: {', '.join(failed_steps)}")
        print("Please fix the issues above and run the setup again.")
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update .env with your configuration values")
    print("2. Run 'python -m improved.main' to start the service")
    print("3. Visit http://localhost:8000/docs for API documentation")
    print("4. Run 'python -m pytest tests/ -v' to run tests")


def dev_setup():
    """Development setup with additional tools"""
    print("🛠️  Setting up development environment...")
    
    # Install development dependencies
    dev_deps = [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "black",
        "isort",
        "flake8",
        "mypy",
        "httpx"
    ]
    
    for dep in dev_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Run code quality checks
    print("\n🔍 Running code quality checks...")
    if not check_code_quality():
        print("\n⚠️  Code quality issues found. Run 'python scripts/setup.py format' to fix them.")
    
    # Run tests
    print("\n🧪 Running tests...")
    run_tests()


def format():
    """Format code"""
    print("🎨 Formatting code...")
    if format_code():
        print("✅ Code formatting completed successfully")
    else:
        print("❌ Code formatting failed")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "dev":
            dev_setup()
        elif command == "format":
            format()
        elif command == "test":
            run_tests()
        elif command == "quality":
            check_code_quality()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: dev, format, test, quality")
            sys.exit(1)
    else:
        main()






























