#!/usr/bin/env python3
"""
Development Setup Script
Automates the setup process for local development
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a shell command"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def setup_virtual_environment():
    """Create and setup virtual environment"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    return run_command(
        [sys.executable, "-m", "venv", ".venv"],
        "Creating virtual environment"
    )


def install_dependencies():
    """Install project dependencies"""
    # Determine pip command based on platform
    if os.name == 'nt':  # Windows
        pip_cmd = ".venv\\Scripts\\pip.exe"
    else:  # Unix/Linux/Mac
        pip_cmd = ".venv/bin/pip"
    
    commands = [
        ([pip_cmd, "install", "--upgrade", "pip"], "Upgrading pip"),
        ([pip_cmd, "install", "-r", "requirements.txt"], "Installing dependencies"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    return True


def setup_pre_commit():
    """Setup pre-commit hooks"""
    if os.name == 'nt':  # Windows
        pre_commit_cmd = ".venv\\Scripts\\pre-commit.exe"
    else:
        pre_commit_cmd = ".venv/bin/pre-commit"
    
    commands = [
        ([pre_commit_cmd, "install"], "Installing pre-commit hooks"),
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc, check=False)  # Don't fail if pre-commit not available
    
    return True


def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        print("Creating .env from template...")
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file (please update with your values)")
        return True
    else:
        print("⚠️  No env.example found, skipping .env creation")
        return True


def setup_redis():
    """Check Redis availability"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        print("✅ Redis is running")
        return True
    except Exception:
        print("⚠️  Redis is not available (optional for development)")
        print("   Run: docker run -d -p 6379:6379 redis:alpine")
        return True  # Don't fail setup if Redis is not available


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🚀 Setting up Content Redundancy Detector")
    print("="*60)
    
    # Change to script directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating virtual environment", setup_virtual_environment),
        ("Creating .env file", create_env_file),
        ("Installing dependencies", install_dependencies),
        ("Setting up pre-commit hooks", setup_pre_commit),
        ("Checking Redis", setup_redis),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update .env with your configuration")
    print("2. Activate virtual environment:")
    if os.name == 'nt':
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("3. Run the application:")
    print("   python -m uvicorn app:app --reload")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()


