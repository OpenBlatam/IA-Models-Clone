"""
Presentation Layer
==================

Main presentation layer class that combines routers, builders, and responses.
"""

import logging
from typing import List, Optional
from fastapi import FastAPI
from aws.modules.presentation.api_router import APIRouter
from aws.modules.presentation.endpoint_builder import EndpointBuilder
from aws.modules.presentation.response_builder import ResponseBuilder

logger = logging.getLogger(__name__)


class PresentationLayer:
    """Presentation layer manager."""
    
    def __init__(self, prefix: str = "/api/v1", tags: Optional[List[str]] = None):
        self.prefix = prefix
        self.tags = tags or []
        self.router = APIRouter(prefix=prefix, tags=tags)
        self.builder = EndpointBuilder(self.router)
        self.response_builder = ResponseBuilder()
    
    def get_router(self) -> APIRouter:
        """Get API router."""
        return self.router
    
    def get_builder(self) -> EndpointBuilder:
        """Get endpoint builder."""
        return self.builder
    
    def get_response_builder(self) -> ResponseBuilder:
        """Get response builder."""
        return self.response_builder
    
    def include_in_app(self, app: FastAPI):
        """Include router in FastAPI app."""
        app.include_router(self.router.get_router())















