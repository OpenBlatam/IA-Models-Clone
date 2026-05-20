#!/usr/bin/env python3
"""
Run API with Advanced Debugging
Runs the API with enhanced debugging capabilities
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api-debug.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_debugging():
    """Setup debugging environment"""
    # Enable debug mode
    os.environ['DEBUG'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Enable Python debugging
    import sys
    if hasattr(sys, 'gettrace') and sys.gettrace() is None:
        logger.info("Python debugging enabled")
    
    # Enable FastAPI debugging
    os.environ['FASTAPI_DEBUG'] = 'true'
    
    logger.info("Debugging environment configured")

def check_dependencies():
    """Check if all dependencies are installed"""
    required = ['fastapi', 'uvicorn', 'pydantic']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements-optimized.txt")
        return False
    
    logger.info("All dependencies installed")
    return True

def run_api():
    """Run the API server"""
    import uvicorn
    
    setup_debugging()
    
    if not check_dependencies():
        sys.exit(1)
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    reload = os.getenv('RELOAD', 'true').lower() == 'true'
    log_level = os.getenv('LOG_LEVEL', 'debug')
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Reload: {reload}, Log Level: {log_level}")
    
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            reload_dirs=[str(project_root)],
            reload_includes=["*.py"],
            reload_excludes=["*.pyc", "__pycache__"],
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    run_api()



