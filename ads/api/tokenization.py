"""
Unified Tokenization API for the ads feature.

This module adapts and re-exports the legacy tokenization API router under the new
api package to support inclusion in the unified API router.
"""

from fastapi import APIRouter

# Import legacy router
from ..tokenization_api import router as legacy_tokenization_router  # type: ignore

router = APIRouter(prefix="/tokenization", tags=["ads-tokenization"])

# Mount legacy routes under the new prefix
router.include_router(legacy_tokenization_router)

__all__ = ["router"]






