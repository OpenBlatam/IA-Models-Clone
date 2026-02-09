"""
Plastic Surgery Visualization AI - FastAPI Server
AI system that visualizes how you'll look after plastic surgery procedures.
"""

import uvicorn

from config.settings import settings
from core.app_factory import create_app
from utils.logger import setup_logging, get_logger
from utils.development import setup_dev_environment, print_debug_info, is_development

# Setup development environment
setup_dev_environment()

# Initialize logging
setup_logging(
    log_level=settings.log_level,
    use_json=settings.use_json_logging
)
logger = get_logger(__name__)

# Print debug info in development
if is_development():
    print_debug_info()

# Create application
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower()
    )

