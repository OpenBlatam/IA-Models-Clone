from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Callable
from fastapi import Request, Response
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Security Middleware - Security headers and protection
"""



def create_security_middleware() -> Callable:
    """Create security middleware."""
    
    async def security_middleware(request: Request, call_next: Callable) -> Response:
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    return security_middleware 