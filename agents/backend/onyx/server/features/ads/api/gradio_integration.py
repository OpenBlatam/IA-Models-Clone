"""
Unified Gradio Integration for the ads feature.

Provides simple endpoints to report Gradio demo availability and optional launch helpers.
Demos remain in separate files (gradio_example.py, gradio_text_demo.py, gradio_image_demo.py).
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/gradio", tags=["ads-gradio"])

@router.get("/health")
async def gradio_health():
    return {
        "status": "available",
        "demos": [
            "gradio_example.py",
            "gradio_text_demo.py",
            "gradio_image_demo.py",
        ],
        "timestamp": datetime.now().isoformat(),
    }

__all__ = ["router"]






