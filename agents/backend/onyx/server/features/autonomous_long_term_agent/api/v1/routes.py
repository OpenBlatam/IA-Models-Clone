"""
API Routes for Autonomous Long-Term Agent
"""

from fastapi import APIRouter
from .controllers.agent_controller import router as agent_router

router = APIRouter(prefix="/api/v1")

router.include_router(agent_router)




