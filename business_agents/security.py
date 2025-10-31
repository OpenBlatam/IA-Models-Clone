from typing import Optional
from fastapi import Header, HTTPException
from .settings import settings

try:
    import jwt  # type: ignore
except Exception:
    jwt = None  # type: ignore


async def auth_dependency(
    authorization: Optional[str] = Header(default=None, convert_underscores=False),
    x_api_key: Optional[str] = Header(default=None)
):
    if not settings.enforce_auth:
        return True

    # API Key check (fast path)
    if settings.api_key and x_api_key == settings.api_key:
        return True

    # Bearer JWT check
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ", 1)[1]
    if not settings.jwt_secret or jwt is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        options = {"verify_aud": bool(settings.jwt_audience)}
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience if settings.jwt_audience else None,
            options=options,
        )
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")







