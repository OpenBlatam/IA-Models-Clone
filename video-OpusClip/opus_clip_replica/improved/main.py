"""
Main Entry Point for OpusClip Improved
=====================================

Production-ready entry point for the OpusClip Improved API.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from opus_clip_improved.app import create_app, main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("opus_clip_improved.log")
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting OpusClip Improved API")
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)






























