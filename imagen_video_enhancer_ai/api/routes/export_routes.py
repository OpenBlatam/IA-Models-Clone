"""
Export Routes
=============

API routes for exporting results.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..dependencies import get_agent
from ..models import ExportResultsRequest
from ...utils.compression import CompressionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["export"])


@router.post("/export-results")
async def export_results(request: ExportResultsRequest):
    """Export task results to file."""
    agent = get_agent()
    
    try:
        exported_path = await agent.export_results(
            task_ids=request.task_ids,
            format=request.format,
            output_path=request.output_path
        )
        
        # Compress if requested
        if request.compress:
            exported_path = CompressionManager.compress_file(exported_path)
        
        return JSONResponse({
            "success": True,
            "exported_path": exported_path,
            "format": request.format,
            "compressed": request.compress
        })
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




