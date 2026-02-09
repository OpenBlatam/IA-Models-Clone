"""Health check endpoints."""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config.settings import settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {
        "PIL": False,
        "numpy": False,
        "opencv": False,
    }
    
    try:
        import PIL
        dependencies["PIL"] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies["opencv"] = True
    except ImportError:
        pass
    
    return dependencies


def check_storage() -> Dict[str, Any]:
    """Check storage directories."""
    upload_dir = Path(settings.upload_dir)
    output_dir = Path(settings.output_dir)
    
    def is_writable(path: Path) -> bool:
        """Check if directory is writable."""
        if not path.exists() or not path.is_dir():
            return False
        try:
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False
    
    return {
        "upload_dir": {
            "exists": upload_dir.exists(),
            "writable": is_writable(upload_dir)
        },
        "output_dir": {
            "exists": output_dir.exists(),
            "writable": is_writable(output_dir)
        }
    }


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with dependency verification."""
    dependencies = check_dependencies()
    storage = check_storage()
    
    # Check if critical dependencies are available
    critical_ok = dependencies.get("PIL", False)
    
    status = "healthy" if critical_ok else "degraded"
    
    return {
        "status": status,
        "service": "plastic_surgery_visualization_ai",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies,
        "storage": storage
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint with full system verification."""
    dependencies = check_dependencies()
    storage = check_storage()
    
    # Check critical dependencies
    if not dependencies.get("PIL", False):
        raise HTTPException(
            status_code=503,
            detail="Service not ready: PIL/Pillow not available"
        )
    
    # Check storage
    if not storage["output_dir"]["exists"] or not storage["output_dir"]["writable"]:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: Output directory not writable"
        )
    
    return {
        "status": "ready",
        "service": "plastic_surgery_visualization_ai",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies,
        "storage": storage
    }


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Liveness check endpoint (simple)."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

