#!/usr/bin/env python3
"""
Functional Facebook Posts System Installer
==========================================

Automated installation script following functional programming principles
"""

import os
import sys
import subprocess
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pure functions for installation

def get_project_root() -> Path:
    """Get project root directory - pure function"""
    return Path(__file__).parent


def get_requirements_file() -> Path:
    """Get requirements file path - pure function"""
    return get_project_root() / "requirements_improved.txt"


def get_env_example_file() -> Path:
    """Get environment example file path - pure function"""
    return get_project_root() / "env.example"


def get_env_file() -> Path:
    """Get environment file path - pure function"""
    return get_project_root() / ".env"


def check_python_version() -> bool:
    """Check Python version - pure function"""
    version = sys.version_info
    return version.major == 3 and version.minor >= 8


def get_installation_config() -> Dict[str, Any]:
    """Get installation configuration - pure function"""
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "project_root": str(get_project_root()),
        "requirements_file": str(get_requirements_file()),
        "env_example_file": str(get_env_example_file()),
        "env_file": str(get_env_file())
    }


def validate_environment() -> bool:
    """Validate environment - pure function"""
    if not check_python_version():
        logger.error("Python 3.8+ is required")
        return False
    
    if not get_requirements_file().exists():
        logger.error("Requirements file not found")
        return False
    
    if not get_env_example_file().exists():
        logger.error("Environment example file not found")
        return False
    
    return True


def create_directories() -> List[Path]:
    """Create necessary directories - pure function"""
    directories = [
        get_project_root() / "logs",
        get_project_root() / "data",
        get_project_root() / "cache",
        get_project_root() / "temp"
    ]
    
    created_dirs = []
    for directory in directories:
        directory.mkdir(exist_ok=True)
        created_dirs.append(directory)
    
    return created_dirs


def read_env_example() -> str:
    """Read environment example file - pure function"""
    try:
        return get_env_example_file().read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading env example: {e}")
        return ""


def create_env_file(env_content: str) -> bool:
    """Create environment file - pure function"""
    try:
        get_env_file().write_text(env_content, encoding='utf-8')
        return True
    except Exception as e:
        logger.error(f"Error creating env file: {e}")
        return False


def get_pip_install_command() -> List[str]:
    """Get pip install command - pure function"""
    return [sys.executable, "-m", "pip", "install", "-r", str(get_requirements_file())]


def get_pip_upgrade_command() -> List[str]:
    """Get pip upgrade command - pure function"""
    return [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]


def get_pip_check_command() -> List[str]:
    """Get pip check command - pure function"""
    return [sys.executable, "-m", "pip", "check"]


# Installation functions

def install_dependencies() -> bool:
    """Install Python dependencies"""
    try:
        logger.info("Upgrading pip...")
        subprocess.run(get_pip_upgrade_command(), check=True, capture_output=True)
        
        logger.info("Installing dependencies...")
        subprocess.run(get_pip_install_command(), check=True, capture_output=True)
        
        logger.info("Checking for conflicts...")
        subprocess.run(get_pip_check_command(), check=True, capture_output=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during installation: {e}")
        return False


def setup_environment() -> bool:
    """Setup environment configuration"""
    try:
        logger.info("Setting up environment...")
        
        # Read env example
        env_content = read_env_example()
        if not env_content:
            return False
        
        # Create env file if it doesn't exist
        if not get_env_file().exists():
            if not create_env_file(env_content):
                return False
            logger.info("Created .env file from template")
        else:
            logger.info(".env file already exists")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        return False


def create_startup_scripts() -> bool:
    """Create startup scripts"""
    try:
        logger.info("Creating startup scripts...")
        
        # Create start script
        start_script = get_project_root() / "start.py"
        start_content = '''#!/usr/bin/env python3
"""
Start Facebook Posts System
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
        start_script.write_text(start_content, encoding='utf-8')
        start_script.chmod(0o755)
        
        # Create production start script
        prod_script = get_project_root() / "start_prod.py"
        prod_content = '''#!/usr/bin/env python3
"""
Start Facebook Posts System in Production Mode
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
'''
        prod_script.write_text(prod_content, encoding='utf-8')
        prod_script.chmod(0o755)
        
        return True
    except Exception as e:
        logger.error(f"Error creating startup scripts: {e}")
        return False


def run_health_check() -> bool:
    """Run health check"""
    try:
        logger.info("Running health check...")
        
        # Import and test basic functionality
        sys.path.append(str(get_project_root()))
        
        from core.config import get_settings, validate_environment
        from api.schemas import PostRequest, ContentType, AudienceType
        
        # Test configuration
        settings = get_settings()
        if not validate_environment():
            logger.error("Environment validation failed")
            return False
        
        # Test schema creation
        test_request = PostRequest(
            topic="Test topic",
            content_type=ContentType.TEXT,
            audience_type=AudienceType.GENERAL
        )
        
        if not test_request.topic:
            logger.error("Schema validation failed")
            return False
        
        logger.info("Health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def create_documentation() -> bool:
    """Create documentation"""
    try:
        logger.info("Creating documentation...")
        
        # Create README
        readme_content = '''# Facebook Posts System - Functional Edition

## Quick Start

1. **Install dependencies:**
   ```bash
   python install_functional_system.py
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start the system:**
   ```bash
   python start.py
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

## Features

- ✅ Functional programming principles
- ✅ FastAPI best practices
- ✅ Async/await patterns
- ✅ Type safety with Pydantic
- ✅ Comprehensive error handling
- ✅ Health checks and monitoring
- ✅ Caching and optimization
- ✅ Batch processing
- ✅ Analytics and metrics

## Configuration

Edit `.env` file to configure:
- Database connection
- Redis cache
- AI service API key
- Security settings
- Performance tuning

## Development

```bash
# Install in development mode
python install_functional_system.py --dev

# Run tests
python -m pytest tests/

# Run with debug
python start.py --debug
```

## Production

```bash
# Install in production mode
python install_functional_system.py --prod

# Run production server
python start_prod.py
```
'''
        
        readme_file = get_project_root() / "README.md"
        readme_file.write_text(readme_content, encoding='utf-8')
        
        return True
    except Exception as e:
        logger.error(f"Error creating documentation: {e}")
        return False


def print_installation_summary(config: Dict[str, Any]) -> None:
    """Print installation summary - pure function"""
    print("\n" + "="*60)
    print("🚀 FACEBOOK POSTS SYSTEM - INSTALLATION COMPLETE")
    print("="*60)
    print(f"📁 Project Root: {config['project_root']}")
    print(f"🐍 Python Version: {config['python_version']}")
    print(f"📦 Requirements: {config['requirements_file']}")
    print(f"⚙️  Environment: {config['env_file']}")
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your configuration")
    print("2. Run: python start.py")
    print("3. Visit: http://localhost:8000/docs")
    print("\n🔧 Available Commands:")
    print("- python start.py          # Development mode")
    print("- python start_prod.py     # Production mode")
    print("- python -m pytest         # Run tests")
    print("\n📚 Documentation:")
    print("- README.md                # Quick start guide")
    print("- /docs                    # API documentation")
    print("- /health                  # Health check")
    print("="*60)


# Main installation function

def main() -> None:
    """Main installation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install Facebook Posts System")
    parser.add_argument("--dev", action="store_true", help="Install in development mode")
    parser.add_argument("--prod", action="store_true", help="Install in production mode")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-env", action="store_true", help="Skip environment setup")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_installation_config()
    
    print("🚀 Installing Facebook Posts System - Functional Edition")
    print(f"📁 Project Root: {config['project_root']}")
    print(f"🐍 Python Version: {config['python_version']}")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Create directories
    logger.info("Creating directories...")
    created_dirs = create_directories()
    logger.info(f"Created {len(created_dirs)} directories")
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("Dependency installation failed")
            sys.exit(1)
    else:
        logger.info("Skipping dependency installation")
    
    # Setup environment
    if not args.skip_env:
        if not setup_environment():
            logger.error("Environment setup failed")
            sys.exit(1)
    else:
        logger.info("Skipping environment setup")
    
    # Create startup scripts
    if not create_startup_scripts():
        logger.error("Startup script creation failed")
        sys.exit(1)
    
    # Create documentation
    if not create_documentation():
        logger.error("Documentation creation failed")
        sys.exit(1)
    
    # Run health check
    if not run_health_check():
        logger.error("Health check failed")
        sys.exit(1)
    
    # Print summary
    print_installation_summary(config)
    
    logger.info("Installation completed successfully!")


if __name__ == "__main__":
    main()

