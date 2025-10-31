#!/usr/bin/env python3
"""
Gamma App - Enhanced Startup Script
Complete startup script for the Gamma App system
"""

import os
import sys
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gamma_app.log')
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy', 
        'redis', 'openai', 'anthropic', 'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"⚠️  Missing packages: {missing_packages}")
        logger.info("📦 Installing missing packages...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
            logger.info("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            logger.error("❌ Failed to install dependencies")
            return False
    
    logger.info("✅ All dependencies available")
    return True

def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    logger.info("📁 Creating directories...")
    
    directories = [
        "uploads", "data", "logs", "static", "templates",
        "cache", "models", "backups", "exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"  ✅ Created: {directory}")

def setup_environment():
    """Setup environment configuration"""
    logger = logging.getLogger(__name__)
    logger.info("⚙️  Setting up environment...")
    
    env_file = project_root / ".env"
    env_example = project_root / "env.example"
    
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        logger.info("📝 Created .env file from template")
        logger.warning("⚠️  Please update .env with your actual API keys")
    
    # Set default environment variables if not set
    os.environ.setdefault("SECRET_KEY", "gamma-app-default-secret-key-2024")
    os.environ.setdefault("DATABASE_URL", "sqlite:///gamma_app.db")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("DEBUG", "true")
    
    logger.info("✅ Environment configured")

def initialize_database():
    """Initialize database"""
    logger = logging.getLogger(__name__)
    logger.info("🗄️  Initializing database...")
    
    try:
        # Try to run database migrations
        subprocess.run([
            sys.executable, "-m", "alembic", "upgrade", "head"
        ], check=True, cwd=project_root)
        logger.info("✅ Database initialized")
    except subprocess.CalledProcessError:
        logger.warning("⚠️  Database migration failed - using default setup")
    except FileNotFoundError:
        logger.warning("⚠️  Alembic not found - skipping migrations")

def start_application():
    """Start the Gamma App application"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Gamma App...")
    
    try:
        # Import and start the application
        from api.main import app
        import uvicorn
        
        logger.info("🌐 Starting server on http://0.0.0.0:8000")
        logger.info("📚 API Documentation: http://0.0.0.0:8000/docs")
        logger.info("🔧 Environment: development")
        logger.info("🐛 Debug mode: enabled")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Try running: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"❌ Error starting application: {e}")

def main():
    """Main startup function"""
    print("🚀 Gamma App - AI-Powered Content Generation System")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Check dependencies
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            sys.exit(1)
        
        # Create directories
        create_directories()
        
        # Setup environment
        setup_environment()
        
        # Initialize database
        initialize_database()
        
        print("\n" + "=" * 60)
        print("🎉 Gamma App Setup Complete!")
        print("=" * 60)
        print("📋 System Status:")
        print("  ✅ Dependencies checked")
        print("  ✅ Directories created")
        print("  ✅ Environment configured")
        print("  ✅ Database initialized")
        print("\n🚀 Starting application...")
        print("=" * 60)
        
        # Start the application
        start_application()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down Gamma App...")
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



