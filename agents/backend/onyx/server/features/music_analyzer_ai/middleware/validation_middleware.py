"""
Validation Middleware - Validate requests
"""

from typing import Any, Callable
import logging

from .middleware import BaseMiddleware
from ..validators import IValidator, ValidationResult

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseMiddleware):
    """
    Middleware that validates requests
    """
    
    def __init__(self, validator: IValidator):
        super().__init__("ValidationMiddleware")
        self.validator = validator
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        """Process with validation"""
        # Validate request
        result = self.validator.validate(request)
        
        if not result.is_valid:
            error_msg = f"Validation failed: {', '.join(result.errors)}"
            logger.warning(error_msg)
            raise ValueError(error_msg)
        
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"Validation warning: {warning}")
        
        # Process if valid
        return next_handler(request)








