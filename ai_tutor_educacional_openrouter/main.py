"""
Main entry point for AI Tutor Educacional.
"""

import asyncio
import logging
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from api.tutor_api import create_tutor_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = create_tutor_app()


async def main():
    """Main function to run the API server."""
    logger.info("Starting AI Tutor Educacional API server...")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

