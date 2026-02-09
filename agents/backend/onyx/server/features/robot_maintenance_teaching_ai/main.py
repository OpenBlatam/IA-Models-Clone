"""
Main entry point for Robot Maintenance Teaching AI API.
"""

import uvicorn
import logging
from .api.maintenance_api import create_maintenance_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = create_maintenance_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

