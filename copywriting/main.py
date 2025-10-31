from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import uvicorn
import logging
from pathlib import Path
import sys
from src.api.app import create_app
from src.config.settings import get_api_config
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Production Copywriting System
============================

Main entry point for the production copywriting system.
"""


# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    
    try:
        # Get configuration
        api_config = get_api_config()
        
        # Create application
        app = create_app()
        
        # Start server
        logger.info(f"Starting Copywriting System on {api_config.host}:{api_config.port}")
        
        uvicorn.run(
            app,
            host=api_config.host,
            port=api_config.port,
            workers=api_config.workers,
            reload=api_config.reload,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 