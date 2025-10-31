"""
API Routes Module
Aggregates all route modules
"""

from fastapi import APIRouter
from .analysis import router as analysis_router

# Create main API router
api_router = APIRouter()

# Include domain-specific routers
api_router.include_router(analysis_router)

# Include improved webhooks router
try:
    from .webhooks_improved import router as webhooks_router
    api_router.include_router(webhooks_router)
except ImportError:
    pass

# Include policy guardrails router
try:
    from .policy import router as policy_router
    api_router.include_router(policy_router)
except ImportError:
    pass

# Future routers:
# from .batch import router as batch_router
# from .ai_ml import router as ai_ml_router
# from .export import router as export_router
# from .analytics import router as analytics_router

__all__ = ["api_router"]
