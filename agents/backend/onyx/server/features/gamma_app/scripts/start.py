#!/usr/bin/env python3
"""
Gamma App - Startup Script
Script to start the Gamma App with proper configuration
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.main import app
from utils.config import get_settings, validate_config
import uvicorn

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gamma_app.log')
        ]
    )

def check_environment():
    """Check environment configuration"""
    settings = get_settings()
    
    print("🔍 Checking environment configuration...")
    
    # Check required settings
    if not settings.secret_key or settings.secret_key == "your-super-secret-key-change-this-in-production":
        print("❌ SECRET_KEY is not properly configured")
        return False
    
    # Check AI API keys
    if not settings.openai_api_key and not settings.anthropic_api_key:
        print("⚠️  No AI API keys configured. Some features may not work.")
    
    # Check database
    if settings.database_url.startswith("sqlite"):
        print("📁 Using SQLite database")
    else:
        print("🗄️  Using external database")
    
    print("✅ Environment configuration looks good!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "uploads",
        "data", 
        "logs",
        "static",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    """Main startup function"""
    print("🚀 Starting Gamma App...")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Create directories
    create_directories()
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix configuration issues.")
        sys.exit(1)
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed.")
        sys.exit(1)
    
    # Get settings
    settings = get_settings()
    
    print(f"🌐 Starting server on {settings.api_host}:{settings.api_port}")
    print(f"📚 API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🔧 Environment: {settings.environment}")
    print(f"🐛 Debug mode: {settings.debug}")
    print("=" * 50)
    
    # Start the server
    try:
        uvicorn.run(
            "api.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Gamma App...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()





























