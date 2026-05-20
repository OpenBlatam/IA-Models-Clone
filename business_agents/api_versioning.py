"""API versioning utilities."""
from enum import Enum
from typing import Optional
from fastapi import Header, HTTPException


class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"


def get_api_version(
    api_version: Optional[str] = Header(default="v1", alias="API-Version")
) -> APIVersion:
    """
    Extract and validate API version from header.
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(version: APIVersion = Depends(get_api_version)):
            if version == APIVersion.V2:
                # V2 logic
            else:
                # V1 logic
    """
    try:
        return APIVersion(api_version.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid API version '{api_version}'. Supported versions: {[v.value for v in APIVersion]}"
        )


def version_router(base_router, version: APIVersion):
    """Apply version prefix to router."""
    base_router.prefix = f"/api/{version.value}{base_router.prefix}"
    return base_router


