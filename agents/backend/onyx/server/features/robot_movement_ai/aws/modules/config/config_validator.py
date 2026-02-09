"""
Config Validator
================

Configuration validation.
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Configuration validator."""
    
    def __init__(self, schema: type[BaseModel]):
        self.schema = schema
    
    def validate(self, config: Dict[str, Any]) -> BaseModel:
        """Validate configuration."""
        try:
            return self.schema(**config)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def validate_partial(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate partial configuration."""
        try:
            validated = self.schema(**config)
            return validated.dict(exclude_unset=True)
        except ValidationError as e:
            logger.error(f"Partial configuration validation failed: {e}")
            raise















