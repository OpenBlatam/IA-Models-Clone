#!/usr/bin/env python3
"""
Run Server Script

Script to run the Quality Control AI API server.
"""

import uvicorn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from quality_control_ai.config.app_settings import get_settings
from quality_control_ai.infrastructure.logging import setup_logging

def main():
    """Run the API server."""
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        level=settings.log_level,
        format=settings.log_format,
        file=settings.log_file
    )
    
    # Run server
    uvicorn.run(
        "quality_control_ai.presentation.api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if not settings.debug else 1,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

if __name__ == "__main__":
    main()



