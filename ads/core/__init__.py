"""
Core utilities for the ads feature (authentication and helpers).

This lightweight module provides `get_current_user` for dependency injection
in tests and simple helpers used across API modules.
"""

from typing import Any, Dict

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> Dict[str, Any]:
    """Minimal auth dependency for tests.

    - If a Bearer token is provided, return a stub user with that token as id.
    - If none is provided, return a deterministic stub user.
    """
    if credentials is None:
        # Return a stub user for tests/local use
        return {"id": 0, "sub": "test-user"}
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"id": 1, "sub": token}


def format_response(data: Any) -> Dict[str, Any]:
    return {"success": True, "data": data}


def handle_error(error: Exception) -> Dict[str, Any]:
    return {"success": False, "error": str(error)}


__all__ = ["get_current_user", "format_response", "handle_error"]








