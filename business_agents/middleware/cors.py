"""
CORS Middleware Setup
====================

CORS middleware configuration for the Business Agents System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import List

from ..config import config, is_production

def setup_cors_middleware(app: FastAPI) -> None:
    """Setup CORS and security middleware."""
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=config.cors_methods,
        allow_headers=config.cors_headers,
    )
    
    # Add trusted host middleware for production
    if is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual hosts in production
        )
