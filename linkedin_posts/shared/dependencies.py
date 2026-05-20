from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from fastapi import Depends, HTTPException, Header, Request
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from typing import Optional
import asyncio
from typing import Any, List, Dict, Optional
import logging
"""
Shared Dependencies
===================

FastAPI dependencies for authentication, authorization, and rate limiting.
"""


# OAuth2 Password Bearer (placeholder for real auth)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# In-memory rate limit store (should use Redis in production)
_rate_limit_store = {}
_RATE_LIMIT = 100  # requests
_RATE_LIMIT_WINDOW = timedelta(minutes=1)

class User:
    """Simple user model for demonstration."""
    def __init__(self, user_id: str, username: str):
        
    """__init__ function."""
self.id = user_id
        self.username = username

# Fake user database (for demo only)
_fake_users = {
    "demo-token": User(user_id="user-123", username="demo")
}

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from token."""
    user = _fake_users.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

async def rate_limiter(request: Request, user: User = Depends(get_current_user)) -> None:
    """Simple rate limiting dependency (per-user)."""
    key = f"{user.id}:{request.url.path}"
    now = datetime.utcnow()
    window_start = now - _RATE_LIMIT_WINDOW
    
    # Initialize if not present
    if key not in _rate_limit_store:
        _rate_limit_store[key] = []
    
    # Remove old timestamps
    _rate_limit_store[key] = [t for t in _rate_limit_store[key] if t > window_start]
    
    # Check rate limit
    if len(_rate_limit_store[key]) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Record request
    _rate_limit_store[key].append(now)

# Dependency shortcuts
async def get_command_handlers():
    """Placeholder for injecting command handlers."""
    # TODO: wire actual command handlers via container
    pass

async def get_query_handlers():
    """Placeholder for injecting query handlers."""
    pass 