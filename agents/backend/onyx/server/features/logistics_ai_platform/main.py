"""
Main FastAPI application for Logistics AI Platform
A comprehensive freight forwarding and logistics management system
"""

from config.settings import settings
from core import (
    create_app,
    setup_middleware,
    setup_exception_handlers,
    setup_routers,
    setup_root_endpoints,
)
from utils.logger import logger


app = create_app()
setup_middleware(app)
setup_exception_handlers(app)
setup_routers(app)
setup_root_endpoints(app)


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Logistics AI Platform server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

