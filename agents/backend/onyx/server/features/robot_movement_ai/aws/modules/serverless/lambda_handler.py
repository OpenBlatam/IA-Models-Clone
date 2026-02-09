"""
Lambda Handler
==============

Optimized Lambda handler for FastAPI.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI
import json

logger = logging.getLogger(__name__)


class LambdaHandler:
    """Optimized Lambda handler."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self._handler = None
    
    def _create_handler(self):
        """Create Lambda handler."""
        try:
            from mangum import Mangum
            return Mangum(self.app, lifespan="off")
        except ImportError:
            logger.warning("mangum not installed, using basic handler")
            return self._basic_handler
    
    def _basic_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Basic Lambda handler (fallback)."""
        # This is a simplified handler
        # In production, use mangum or similar
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Lambda handler not properly configured"})
        }
    
    def __call__(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Handle Lambda event."""
        if self._handler is None:
            self._handler = self._create_handler()
        
        return self._handler(event, context)
    
    @staticmethod
    def create_handler(app: FastAPI):
        """Create Lambda handler from FastAPI app."""
        return LambdaHandler(app)















