"""Information endpoints."""

from fastapi import APIRouter
from datetime import datetime
import sys
import platform

from core.constants import API_VERSION, SERVICE_NAME
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/info")
async def get_service_info():
    """Get service information."""
    return {
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/version")
async def get_version():
    """Get service version."""
    return {
        "version": API_VERSION,
        "service": SERVICE_NAME
    }

