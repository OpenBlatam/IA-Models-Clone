#!/usr/bin/env python3
"""
Development Setup Script
========================

Script to set up development environment.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e.stderr}")
        sys.exit(1)


def main():
    """Main setup function."""
    print("Setting up development environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)
    
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install dependencies
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing dependencies"
    )
    
    # Install development dependencies
    dev_deps = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0"
    ]
    
    for dep in dev_deps:
        run_command(
            [sys.executable, "-m", "pip", "install", dep],
            f"Installing {dep}"
        )
    
    # Create necessary directories
    directories = ["debug", "error_logs", "tests/output"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n✓ Development environment setup complete!")
    print("\nNext steps:")
    print("  1. Activate virtual environment (if using one)")
    print("  2. Run tests: pytest")
    print("  3. Format code: black imagen_video_enhancer_ai/")
    print("  4. Check types: mypy imagen_video_enhancer_ai/")


if __name__ == "__main__":
    main()




