"""
Setup Script
============

Setup script for the image upscaling system.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required = [
        "PIL",
        "numpy",
        "fastapi",
        "pydantic"
    ]
    
    optional = [
        "cv2",
        "torch",
        "realesrgan"
    ]
    
    missing = []
    for dep in required:
        try:
            __import__(dep.lower().replace("pil", "PIL"))
        except ImportError:
            missing.append(dep)
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        return False
    
    print("✅ Required dependencies installed")
    
    # Check optional
    missing_optional = []
    for dep in optional:
        try:
            if dep == "cv2":
                __import__("cv2")
            elif dep == "torch":
                __import__("torch")
            elif dep == "realesrgan":
                __import__("realesrgan")
        except ImportError:
            missing_optional.append(dep)
    
    if missing_optional:
        print("⚠️  Optional dependencies not installed:")
        for dep in missing_optional:
            print(f"   - {dep}")
        print("   (These are optional but recommended)")
    
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "output",
        "cache",
        "logs",
        "models"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")


def setup_environment():
    """Setup environment variables."""
    env_file = Path(".env")
    
    if not env_file.exists():
        env_content = """# Image Upscaling AI Configuration

# OpenRouter
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Upscaling
UPSCALING_QUALITY_MODE=high
UPSCALING_DEFAULT_SCALE=2.0
UPSCALING_USE_AI=true
UPSCALING_USE_REALESRGAN=false

# Performance
UPSCALING_MAX_WORKERS=4
UPSCALING_BATCH_SIZE=1
UPSCALING_ENABLE_CACHE=true
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")
        print("⚠️  Please update .env with your configuration")
    else:
        print("✅ .env file already exists")


def install_dependencies():
    """Install dependencies."""
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        print("Installing dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed")
    else:
        print("⚠️  requirements.txt not found")


def main():
    """Main setup function."""
    print("="*60)
    print("Image Upscaling AI - Setup")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling dependencies...")
        install_dependencies()
        if not check_dependencies():
            print("\n❌ Setup failed. Please install dependencies manually.")
            return
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Setup environment
    print("\nSetting up environment...")
    setup_environment()
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update .env with your configuration")
    print("2. Run: python examples/simple_example.py")
    print("3. Or start API: uvicorn image_upscaling_ai.api.upscaling_api:app")


if __name__ == "__main__":
    main()


