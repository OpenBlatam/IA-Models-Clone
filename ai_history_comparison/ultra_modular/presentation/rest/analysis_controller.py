"""
Analysis Controller
==================

Single responsibility: Handle HTTP requests for content analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Optional, Dict, Any
import logging

from ...application.commands.analyze_content_command import AnalyzeContentCommand
from ...application.handlers.analyze_content_handler import AnalyzeContentHandler
from ...infrastructure.persistence.history_repository import HistoryRepository
from ..dto.analysis_dto import (
    AnalyzeContentRequest,
    AnalyzeContentResponse,
    BulkAnalysisRequest,
    BulkAnalysisResponse
)

logger = logging.getLogger(__name__)


class AnalysisController:
    """
    Controller for content analysis endpoints.
    
    Single Responsibility: Handle HTTP requests for content analysis operations.
    """
    
    def __init__(
        self,
        analyze_handler: AnalyzeContentHandler,
        history_repository: HistoryRepository
    ):
        """
        Initialize the controller.
        
        Args:
            analyze_handler: Handler for analyze content commands
            history_repository: Repository for history entries
        """
        self._analyze_handler = analyze_handler
        self._history_repository = history_repository
        self._router = APIRouter(prefix="/analysis", tags=["Analysis"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self._router.post("/analyze", response_model=AnalyzeContentResponse)
        async def analyze_content(request: AnalyzeContentRequest):
            """Analyze content and create history entry."""
            try:
                command = AnalyzeContentCommand.create(
                    content=request.content,
                    model_version=request.model_version,
                    metadata=request.metadata,
                    user_id=request.user_id
                )
                
                result = await self._analyze_handler.handle(command)
                
                return AnalyzeContentResponse(
                    success=result["success"],
                    entry_id=result["entry_id"],
                    metrics=result["metrics"],
                    quality_score=result["quality_score"],
                    timestamp=result["timestamp"],
                    command_id=result["command_id"]
                )
                
            except Exception as e:
                logger.error(f"Error in analyze_content endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self._router.post("/bulk-analyze", response_model=BulkAnalysisResponse)
        async def bulk_analyze_content(request: BulkAnalysisRequest):
            """Analyze multiple content pieces."""
            try:
                results = []
                successful = 0
                failed = 0
                
                for content in request.contents:
                    try:
                        command = AnalyzeContentCommand.create(
                            content=content,
                            model_version=request.model_version,
                            metadata=request.metadata,
                            user_id=request.user_id
                        )
                        
                        result = await self._analyze_handler.handle(command)
                        results.append(AnalyzeContentResponse(
                            success=result["success"],
                            entry_id=result["entry_id"],
                            metrics=result["metrics"],
                            quality_score=result["quality_score"],
                            timestamp=result["timestamp"],
                            command_id=result["command_id"]
                        ))
                        successful += 1
                        
                    except Exception as e:
                        logger.error(f"Error analyzing content: {e}")
                        results.append(AnalyzeContentResponse(
                            success=False,
                            error_message=str(e)
                        ))
                        failed += 1
                
                return BulkAnalysisResponse(
                    results=results,
                    total_processed=len(request.contents),
                    successful=successful,
                    failed=failed
                )
                
            except Exception as e:
                logger.error(f"Error in bulk_analyze_content endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self._router.get("/entries/{entry_id}")
        async def get_entry(entry_id: str = Path(..., description="Entry ID")):
            """Get history entry by ID."""
            try:
                entry = await self._history_repository.find_by_id(entry_id)
                
                if not entry:
                    raise HTTPException(status_code=404, detail="Entry not found")
                
                return {
                    "id": entry.id,
                    "content": entry.content,
                    "model_version": entry.model_version,
                    "timestamp": entry.timestamp.isoformat(),
                    "metrics": entry.metrics.to_dict(),
                    "quality_score": entry.calculate_quality_score(),
                    "metadata": entry.metadata
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting entry {entry_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self._router.get("/entries")
        async def search_entries(
            model_version: Optional[str] = Query(None, description="Filter by model version"),
            days: Optional[int] = Query(7, description="Number of recent days"),
            limit: Optional[int] = Query(100, description="Maximum number of entries")
        ):
            """Search and filter history entries."""
            try:
                if days:
                    entries = await self._history_repository.find_recent_entries(days, model_version)
                elif model_version:
                    entries = await self._history_repository.find_by_model_version(model_version, limit)
                else:
                    entries = await self._history_repository.find_recent_entries(7, limit=limit)
                
                return {
                    "entries": [
                        {
                            "id": entry.id,
                            "model_version": entry.model_version,
                            "timestamp": entry.timestamp.isoformat(),
                            "quality_score": entry.calculate_quality_score(),
                            "word_count": entry.metrics.word_count,
                            "readability_score": entry.metrics.readability_score
                        }
                        for entry in entries
                    ],
                    "total_count": len(entries)
                }
                
            except Exception as e:
                logger.error(f"Error searching entries: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self._router.delete("/entries/{entry_id}")
        async def delete_entry(entry_id: str = Path(..., description="Entry ID")):
            """Delete history entry."""
            try:
                deleted = await self._history_repository.delete(entry_id)
                
                if not deleted:
                    raise HTTPException(status_code=404, detail="Entry not found")
                
                return {"message": "Entry deleted successfully"}
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting entry {entry_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    @property
    def router(self) -> APIRouter:
        """Get the FastAPI router."""
        return self._router
    
    def get_controller_name(self) -> str:
        """Get controller name."""
        return "AnalysisController"
    
    def get_endpoints(self) -> List[str]:
        """Get list of endpoints."""
        return [
            "POST /analysis/analyze",
            "POST /analysis/bulk-analyze",
            "GET /analysis/entries/{entry_id}",
            "GET /analysis/entries",
            "DELETE /analysis/entries/{entry_id}"
        ]




