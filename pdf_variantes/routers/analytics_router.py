"""Analytics router with functional approach."""

from fastapi import APIRouter, Depends, Query, Path, HTTPException
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from ..dependencies import get_pdf_service, get_current_user, get_admin_user
from ..exceptions import PDFNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/metrics")
async def get_metrics(
    time_range: str = Query("24h"),
    pdf_service = Depends(get_pdf_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get system metrics."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "time_range": time_range,
        "metrics": {
            "total_documents": 0,
            "total_variants": 0,
            "success_rate": 1.0,
            "average_processing_time": 0.0
        }
    }


@router.get("/health")
async def get_health(pdf_service = Depends(get_pdf_service)) -> Dict[str, Any]:
    """Get system health."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "pdf_upload": "healthy",
            "variant_generation": "healthy",
            "topic_extraction": "healthy"
        }
    }


@router.get("/reports/{file_id}")
async def get_document_analytics(
    file_id: str = Path(...),
    pdf_service = Depends(get_pdf_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get document analytics."""
    file_path = pdf_service.upload_handler.get_file_path(file_id)
    
    if not file_path.exists():
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    analytics = await pdf_service.advanced.generate_analytics_report(file_id)
    
    return {
        "file_id": file_id,
        "analytics": analytics,
        "generated_at": datetime.utcnow().isoformat()
    }