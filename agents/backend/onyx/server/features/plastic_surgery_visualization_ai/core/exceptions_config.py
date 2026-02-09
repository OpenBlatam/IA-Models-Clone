"""Exception handlers configuration."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.exceptions import (
    PlasticSurgeryAIException,
    VisualizationNotFoundError,
    RateLimitExceededError
)
from utils.error_handler import handle_exception
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure exception handlers."""
    logger.info("Setting up exception handlers...")
    
    @app.exception_handler(PlasticSurgeryAIException)
    async def custom_exception_handler(
        request: Request,
        exc: PlasticSurgeryAIException
    ) -> JSONResponse:
        """Handle custom exceptions."""
        return await handle_exception(request, exc)
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions."""
        return await handle_exception(request, exc)
    
    logger.info("Exception handlers setup complete")

