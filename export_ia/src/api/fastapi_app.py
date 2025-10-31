"""
FastAPI application for the Export IA system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import logging
import os

from .models import (
    ExportRequest, ExportResponse, TaskStatusResponse, 
    StatisticsResponse, FormatInfo, ErrorResponse
)
from ..core.engine import ExportIAEngine, get_global_export_engine
from ..core.models import ExportConfig

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Export IA API",
        description="Advanced AI-powered document export system",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global engine instance
    engine: Optional[ExportIAEngine] = None
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the engine on startup."""
        nonlocal engine
        engine = get_global_export_engine()
        await engine.initialize()
        logger.info("Export IA API started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown the engine on shutdown."""
        nonlocal engine
        if engine:
            await engine.shutdown()
        logger.info("Export IA API shutdown")
    
    def get_engine() -> ExportIAEngine:
        """Dependency to get the engine instance."""
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        return engine
    
    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Export IA API",
            "version": "2.0.0",
            "description": "Advanced AI-powered document export system",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=dict)
    async def health_check(engine: ExportIAEngine = Depends(get_engine)):
        """Health check endpoint."""
        stats = engine.get_export_statistics()
        return {
            "status": "healthy",
            "active_tasks": stats.active_tasks,
            "total_tasks": stats.total_tasks,
            "uptime": "running"
        }
    
    @app.post("/export", response_model=ExportResponse)
    async def export_document(
        request: ExportRequest,
        background_tasks: BackgroundTasks,
        engine: ExportIAEngine = Depends(get_engine)
    ):
        """Export a document in the specified format."""
        try:
            # Create export configuration
            config = ExportConfig(
                format=request.format,
                document_type=request.document_type,
                quality_level=request.quality_level,
                template=request.template,
                custom_styles=request.custom_styles,
                metadata=request.metadata,
                branding=request.branding,
                output_options=request.output_options
            )
            
            # Submit export task
            task_id = await engine.export_document(request.content, config)
            
            return ExportResponse(
                task_id=task_id,
                status="pending",
                message="Export task created successfully",
                created_at=engine.task_manager.active_tasks[task_id].created_at
            )
            
        except Exception as e:
            logger.error(f"Export request failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/export/{task_id}/status", response_model=TaskStatusResponse)
    async def get_task_status(
        task_id: str,
        engine: ExportIAEngine = Depends(get_engine)
    ):
        """Get the status of an export task."""
        try:
            status_data = await engine.get_task_status(task_id)
            if status_data is None:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return TaskStatusResponse(**status_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Status request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/export/{task_id}")
    async def cancel_task(
        task_id: str,
        engine: ExportIAEngine = Depends(get_engine)
    ):
        """Cancel an active export task."""
        try:
            success = await engine.cancel_task(task_id)
            if not success:
                raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
            
            return {"message": "Task cancelled successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Cancel request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/export/{task_id}/download")
    async def download_file(
        task_id: str,
        engine: ExportIAEngine = Depends(get_engine)
    ):
        """Download the exported file."""
        try:
            status_data = await engine.get_task_status(task_id)
            if status_data is None:
                raise HTTPException(status_code=404, detail="Task not found")
            
            if status_data.get("status") != "completed":
                raise HTTPException(status_code=400, detail="Task not completed")
            
            file_path = status_data.get("file_path")
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")
            
            return FileResponse(
                path=file_path,
                filename=os.path.basename(file_path),
                media_type='application/octet-stream'
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Download request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/statistics", response_model=StatisticsResponse)
    async def get_statistics(engine: ExportIAEngine = Depends(get_engine)):
        """Get system statistics."""
        try:
            stats = engine.get_export_statistics()
            return StatisticsResponse(
                total_tasks=stats.total_tasks,
                active_tasks=stats.active_tasks,
                completed_tasks=stats.completed_tasks,
                failed_tasks=stats.failed_tasks,
                format_distribution=stats.format_distribution,
                quality_distribution=stats.quality_distribution,
                average_quality_score=stats.average_quality_score,
                average_processing_time=stats.average_processing_time,
                total_processing_time=stats.total_processing_time
            )
            
        except Exception as e:
            logger.error(f"Statistics request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/formats", response_model=List[FormatInfo])
    async def get_supported_formats(engine: ExportIAEngine = Depends(get_engine)):
        """Get list of supported export formats."""
        try:
            formats = engine.list_supported_formats()
            return [FormatInfo(**fmt) for fmt in formats]
            
        except Exception as e:
            logger.error(f"Formats request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/templates/{document_type}")
    async def get_document_template(
        document_type: str,
        engine: ExportIAEngine = Depends(get_engine)
    ):
        """Get template for a specific document type."""
        try:
            from ..core.models import DocumentType
            doc_type = DocumentType(document_type)
            template = engine.get_document_template(doc_type)
            return template
            
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid document type: {document_type}")
        except Exception as e:
            logger.error(f"Template request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/validate")
    async def validate_content(
        request: ExportRequest,
        engine: ExportIAEngine = Depends(get_engine)
    ):
        """Validate content and return quality metrics."""
        try:
            # Create export configuration
            config = ExportConfig(
                format=request.format,
                document_type=request.document_type,
                quality_level=request.quality_level,
                template=request.template,
                custom_styles=request.custom_styles,
                metadata=request.metadata,
                branding=request.branding,
                output_options=request.output_options
            )
            
            # Validate content
            metrics = engine.validate_content(request.content, config)
            return metrics
            
        except Exception as e:
            logger.error(f"Validation request failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    return app




