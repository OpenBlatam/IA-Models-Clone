"""
Server Runner
=============

Main server entry point with improved configuration and lifecycle management.
"""

import uvicorn
import logging
from pathlib import Path
import sys
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from character_clothing_changer_ai.config.clothing_changer_config import ClothingChangerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """
    Run the Clothing Changer API server.
    
    Args:
        host: Server host (defaults to config)
        port: Server port (defaults to config)
        reload: Enable auto-reload
        log_level: Log level
    """
    config = ClothingChangerConfig.from_env()
    
    server_host = host or config.api_host
    server_port = port or config.api_port
    
    logger.info(f"Starting Clothing Changer API server on {server_host}:{server_port}")
    logger.info(f"Auto-reload: {reload}, Log level: {log_level}")
    
    try:
        uvicorn.run(
            "character_clothing_changer_ai.api.clothing_changer_api:app",
            host=server_host,
            port=server_port,
            reload=reload,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Character Clothing Changer AI Server")
    parser.add_argument("--host", type=str, help="Server host")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()

