"""
Model Handlers
==============

Request handlers for model-related endpoints.
"""

import logging
from typing import Dict, Any
from fastapi.responses import JSONResponse

from ..utils.error_handlers import handle_api_error

logger = logging.getLogger(__name__)


class ModelHandlers:
    """Handlers for model-related operations."""
    
    def __init__(self, service):
        """
        Initialize handlers.
        
        Args:
            service: CharacterConsistencyService instance
        """
        self.service = service
    
    async def get_model_info(self) -> JSONResponse:
        """
        Handle get model info request.
        
        Returns:
            JSONResponse with model information
        """
        try:
            info = self.service.get_model_info()
            return JSONResponse(content=info)
        except Exception as e:
            raise handle_api_error("get_model_info", e)
    
    async def initialize_model(self) -> JSONResponse:
        """
        Handle initialize model request.
        
        Returns:
            JSONResponse with initialization status
        """
        try:
            self.service.initialize_model()
            return JSONResponse(content={
                "status": "initialized",
                "message": "Model initialized successfully"
            })
        except Exception as e:
            raise handle_api_error("initialize_model", e)
    
    async def health_check(self) -> JSONResponse:
        """
        Handle health check request.
        
        Returns:
            JSONResponse with health status
        """
        try:
            model_info = self.service.get_model_info()
            return JSONResponse(content={
                "status": "healthy",
                "model_initialized": model_info.get("status") != "not_initialized",
            })
        except Exception as e:
            from fastapi.responses import JSONResponse as JSONResp
            return JSONResp(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
            )

