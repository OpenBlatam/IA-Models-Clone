"""
Main entry point for Robot Maintenance AI API server.
"""

import os
import uvicorn
from .api.maintenance_api import create_maintenance_app
from .utils.logger_config import setup_logging

# Setup enhanced logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE", None)
setup_logging(log_level=log_level, log_file=log_file)

if __name__ == "__main__":
    app = create_maintenance_app()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower()
    )
