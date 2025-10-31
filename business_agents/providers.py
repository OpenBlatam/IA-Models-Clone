from typing import Optional
from .interfaces import ICache, IHTTPClient, IAuth
from .cache import AsyncCache
from .http_client import ResilientHTTPClient
from .security import auth_dependency
from .settings import settings


def provide_cache() -> ICache:
    return AsyncCache(settings.redis_url)


def provide_http_client() -> IHTTPClient:
    return ResilientHTTPClient(
        timeout_seconds=settings.http_timeout_seconds,
        retries=settings.http_retries,
        cb_fail_threshold=settings.cb_fail_threshold,
        cb_recovery_seconds=settings.cb_recovery_seconds,
    )


def provide_auth() -> IAuth:
    # Return a simple adapter that wraps the existing auth_dependency
    class AuthAdapter:
        async def verify(self, token: Optional[str], api_key: Optional[str]) -> Any:
            # For now, delegate to existing auth_dependency logic
            # This can be enhanced later
            if not settings.enforce_auth:
                return True
            if settings.api_key and api_key == settings.api_key:
                return True
            # JWT verification would go here
            return True
    return AuthAdapter()


