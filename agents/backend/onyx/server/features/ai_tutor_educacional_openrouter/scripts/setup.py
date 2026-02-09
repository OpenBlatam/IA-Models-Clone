"""
Setup script for initial configuration.
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/tutor_db/students",
        "data/tutor_db/conversations",
        "data/tutor_db/evaluations",
        "data/tutor_db/reports",
        "conversations",
        ".cache/tutor",
        "logs",
        "reports",
        "backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_env_file():
    """Create .env file from template."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("⚠ .env file already exists, skipping...")
        return
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✓ Created .env file from .env.example")
        print("⚠ Please update .env with your Open Router API key")
    else:
        # Create basic .env
        env_content = """# Open Router API Configuration
OPENROUTER_API_KEY=your-api-key-here

# Optional: Custom Open Router Settings
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_DEFAULT_MODEL=openai/gpt-4
OPENROUTER_TIMEOUT=60
OPENROUTER_MAX_RETRIES=3
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=2000

# Storage Configuration
CONVERSATION_HISTORY_PATH=conversations
MAX_HISTORY_LENGTH=50

# Feature Flags
CACHE_ENABLED=true
CACHE_TTL=3600
ADAPTIVE_LEARNING=true
PROVIDE_EXERCISES=true
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✓ Created .env file")
        print("⚠ Please update .env with your Open Router API key")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "httpx",
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✓ All required dependencies are installed")
    return True


def main():
    """Main setup function."""
    print("=" * 60)
    print("AI Tutor Educacional - Setup")
    print("=" * 60)
    print()
    
    print("Creating directories...")
    create_directories()
    print()
    
    print("Creating .env file...")
    create_env_file()
    print()
    
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("=" * 60)
    if deps_ok:
        print("✓ Setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Update .env with your OPENROUTER_API_KEY")
        print("2. Run: python main.py")
        print("3. Visit: http://localhost:8000/docs")
    else:
        print("⚠ Setup completed with warnings")
        print("Please install missing dependencies before running")
    print("=" * 60)


if __name__ == "__main__":
    main()






