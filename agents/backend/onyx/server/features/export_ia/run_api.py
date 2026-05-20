#!/usr/bin/env python3
"""
Export IA API Server
===================

Run the Export IA API server with proper configuration.
"""

import uvicorn
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.fastapi_app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the API server."""
    logger.info("Starting Export IA API Server...")
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )


if __name__ == "__main__":
    main()




