"""
API Error Handler Utility
=========================

Centralized error handling for API endpoints.
"""

import logging
from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ...core.exceptions import (
    ValidationError,
    ImageValidationError,
    ClothingChangerError
)

logger = logging.getLogger(__name__)


class APIErrorHandler:
    """Handles API errors consistently."""
    
    @staticmethod
    def handle_validation_error(error: ValidationError) -> HTTPException:
        """
        Handle validation errors.
        
        Args:
            error: ValidationError instance
            
        Returns:
            HTTPException with 400 status
        """
        logger.warning(f"Validation error: {error.message}", extra=error.details)
        return HTTPException(status_code=400, detail=error.to_dict())
    
    @staticmethod
    def handle_image_validation_error(error: ImageValidationError) -> HTTPException:
        """
        Handle image validation errors.
        
        Args:
            error: ImageValidationError instance
            
        Returns:
            HTTPException with 400 status
        """
        logger.warning(f"Image validation error: {error.message}", extra=error.details)
        return HTTPException(status_code=400, detail=error.to_dict())
    
    @staticmethod
    def handle_runtime_error(error: RuntimeError, context: str = "operation") -> HTTPException:
        """
        Handle runtime errors (e.g., model initialization failures).
        
        Args:
            error: RuntimeError instance
            context: Context description
            
        Returns:
            HTTPException with 500 status
        """
        logger.error(f"Error in {context}: {error}", exc_info=True)
        return HTTPException(status_code=500, detail=str(error))
    
    @staticmethod
    def handle_clothing_changer_error(error: ClothingChangerError) -> HTTPException:
        """
        Handle ClothingChangerError exceptions.
        
        Args:
            error: ClothingChangerError instance
            
        Returns:
            HTTPException with appropriate status code
        """
        status_code = 500
        if isinstance(error, ValidationError) or isinstance(error, ImageValidationError):
            status_code = 400
        
        logger.error(f"ClothingChangerError: {error.message}", extra=error.details)
        return HTTPException(status_code=status_code, detail=error.to_dict())
    
    @staticmethod
    def handle_generic_error(error: Exception, context: str = "operation") -> HTTPException:
        """
        Handle generic exceptions.
        
        Args:
            error: Exception instance
            context: Context description
            
        Returns:
            HTTPException with 500 status
        """
        logger.error(f"Error in {context}: {error}", exc_info=True)
        return HTTPException(status_code=500, detail=str(error))
    
    @staticmethod
    def handle_error(error: Exception, context: str = "operation") -> HTTPException:
        """
        Handle any exception with appropriate error handler.
        
        Args:
            error: Exception instance
            context: Context description
            
        Returns:
            HTTPException
        """
        if isinstance(error, ValidationError):
            return APIErrorHandler.handle_validation_error(error)
        elif isinstance(error, ImageValidationError):
            return APIErrorHandler.handle_image_validation_error(error)
        elif isinstance(error, RuntimeError):
            return APIErrorHandler.handle_runtime_error(error, context)
        elif isinstance(error, ClothingChangerError):
            return APIErrorHandler.handle_clothing_changer_error(error)
        else:
            return APIErrorHandler.handle_generic_error(error, context)

