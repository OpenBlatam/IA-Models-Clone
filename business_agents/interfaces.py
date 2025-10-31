from typing import Protocol, Optional, Dict, Any


class ICache(Protocol):
    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str, ttl: int) -> None: ...


class IHTTPClient(Protocol):
    async def get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]: ...
    async def close(self) -> None: ...


class IAuth(Protocol):
    async def verify(self, token: Optional[str], api_key: Optional[str]) -> Any: ...


