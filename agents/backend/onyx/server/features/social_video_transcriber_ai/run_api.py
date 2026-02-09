"""
Run the Social Video Transcriber AI API server
"""

import uvicorn
import logging

from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run the API server"""
    settings = get_settings()
    
    logger.info(f"Starting {settings.api_title}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Listening on: {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        workers=1 if settings.environment == "development" else 4,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()












