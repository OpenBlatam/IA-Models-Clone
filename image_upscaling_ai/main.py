"""
Image Upscaling AI - Main Entry Point
======================================

Main entry point for the Image Upscaling AI service.
"""

import uvicorn
import logging
from pathlib import Path

from config.upscaling_config import UpscalingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    # Load configuration
    config = UpscalingConfig.from_env()
    config.validate()
    
    logger.info("Starting Image Upscaling AI service...")
    logger.info(f"API will be available at http://{config.api_host}:{config.api_port}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Quality mode: {config.quality_mode}")
    logger.info(f"AI enhancement: {config.use_ai_enhancement}")
    logger.info(f"optimization_core: {config.use_optimization_core}")
    
    # Run server
    uvicorn.run(
        "api.upscaling_api:app",
        host=config.api_host,
        port=config.api_port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()


