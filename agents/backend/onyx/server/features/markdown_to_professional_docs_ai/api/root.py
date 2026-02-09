"""Root and info endpoints"""
from fastapi import APIRouter
from config import settings

router = APIRouter(tags=["Root"])


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": settings.app_name,
        "version": "2.3.0",
        "description": "Convert Markdown to professional document formats",
        "supported_formats": [
            "excel", "pdf", "word", "tableau", "powerbi", 
            "html", "ppt", "latex", "rtf", "epub", "odt"
        ],
        "endpoints": {
            "convert": "/convert",
            "convert_file": "/convert/file",
            "batch_convert": "/convert/batch",
            "health": "/health",
            "formats": "/formats",
            "docs": "/docs"
        }
    }

