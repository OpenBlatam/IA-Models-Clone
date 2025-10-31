"""
Quality Service - Microservice for quality validation and enhancement.
"""

import asyncio
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn

from services.core import BaseService
from services.communication import get_message_bus, EventPublisher
from services.discovery import get_service_discovery
from src.core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel
from src.core.quality_manager import QualityManager
from src.core.validation import get_validation_manager

logger = logging.getLogger(__name__)


class QualityService(BaseService):
    """Microservice for handling quality validation and enhancement."""
    
    def __init__(self, host: str = "localhost", port: int = 8002):
        super().__init__("quality-service", "1.0.0", host, port)
        self.app = FastAPI(title="Quality Service", version="1.0.0")
        self.message_bus = get_message_bus()
        self.service_discovery = get_service_discovery()
        self.event_publisher = EventPublisher(self.message_bus, "quality-service")
        
        # Quality components
        self.quality_manager = None
        self.validation_manager = None
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup API routes for the quality service."""
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.post("/validate")
        async def validate_content(request: Dict[str, Any]):
            """Validate document content."""
            try:
                content = request.get("content")
                config_data = request.get("config", {})
                
                if not content:
                    raise HTTPException(status_code=400, detail="Content is required")
                
                # Create export configuration
                config = ExportConfig(
                    format=ExportFormat(config_data.get("format", "pdf")),
                    document_type=DocumentType(config_data.get("document_type", "report")),
                    quality_level=QualityLevel(config_data.get("quality_level", "professional"))
                )
                
                # Validate content
                validation_results = self.validation_manager.validate_export_request(content, config)
                
                # Get quality metrics
                quality_metrics = self.quality_manager.get_quality_metrics(content, config)
                
                # Publish event
                await self.event_publisher.publish_event(
                    "quality.validated",
                    {
                        "has_errors": self.validation_manager.has_errors(validation_results),
                        "quality_score": quality_metrics.overall_score
                    }
                )
                
                return {
                    "validation_results": [
                        {
                            "field": r.field,
                            "message": r.message,
                            "severity": r.severity.value,
                            "suggestion": r.suggestion
                        }
                        for r in validation_results
                    ],
                    "quality_metrics": {
                        "overall_score": quality_metrics.overall_score,
                        "formatting_score": quality_metrics.formatting_score,
                        "content_score": quality_metrics.content_score,
                        "accessibility_score": quality_metrics.accessibility_score,
                        "professional_score": quality_metrics.professional_score,
                        "issues": quality_metrics.issues,
                        "suggestions": quality_metrics.suggestions
                    }
                }
                
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/enhance")
        async def enhance_content(request: Dict[str, Any]):
            """Enhance document content quality."""
            try:
                content = request.get("content")
                config_data = request.get("config", {})
                
                if not content:
                    raise HTTPException(status_code=400, detail="Content is required")
                
                # Create export configuration
                config = ExportConfig(
                    format=ExportFormat(config_data.get("format", "pdf")),
                    document_type=DocumentType(config_data.get("document_type", "report")),
                    quality_level=QualityLevel(config_data.get("quality_level", "professional"))
                )
                
                # Enhance content
                enhanced_content = await self.quality_manager.process_content_for_quality(content, config)
                
                # Publish event
                await self.event_publisher.publish_event(
                    "quality.enhanced",
                    {"enhancements_applied": len(enhanced_content.get("enhancements", []))}
                )
                
                return {"enhanced_content": enhanced_content}
                
            except Exception as e:
                logger.error(f"Enhancement failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/quality-levels")
        async def get_quality_levels():
            """Get available quality levels."""
            return [
                {"level": level.value, "description": f"{level.value.title()} quality level"}
                for level in QualityLevel
            ]
    
    async def _start(self) -> None:
        """Start the quality service."""
        # Initialize quality components
        from src.core.config import ConfigManager
        config_manager = ConfigManager()
        
        self.quality_manager = QualityManager(config_manager)
        self.validation_manager = get_validation_manager()
        
        # Start message bus
        await self.message_bus.start()
        
        # Start service discovery
        await self.service_discovery.start()
        
        # Register with service discovery
        await self.service_discovery.register_service(
            name=self.name,
            host=self.host,
            port=self.port,
            health_url="/health",
            api_url="/"
        )
        
        # Register message handlers
        await self.message_bus.register_request_handler("validate", self._handle_validate_request)
        await self.message_bus.register_request_handler("enhance", self._handle_enhance_request)
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        
        logger.info(f"Quality Service started on {self.host}:{self.port}")
    
    async def _stop(self) -> None:
        """Stop the quality service."""
        # Stop server
        if hasattr(self, '_server_task'):
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        # Stop service discovery
        await self.service_discovery.stop()
        
        # Stop message bus
        await self.message_bus.stop()
        
        logger.info("Quality Service stopped")
    
    async def _handle_validate_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation request via message bus."""
        try:
            content = payload.get("content")
            config_data = payload.get("config", {})
            
            # Create export configuration
            config = ExportConfig(
                format=ExportFormat(config_data.get("format", "pdf")),
                document_type=DocumentType(config_data.get("document_type", "report")),
                quality_level=QualityLevel(config_data.get("quality_level", "professional"))
            )
            
            # Validate content
            validation_results = self.validation_manager.validate_export_request(content, config)
            quality_metrics = self.quality_manager.get_quality_metrics(content, config)
            
            return {
                "validation_results": [
                    {
                        "field": r.field,
                        "message": r.message,
                        "severity": r.severity.value,
                        "suggestion": r.suggestion
                    }
                    for r in validation_results
                ],
                "quality_metrics": {
                    "overall_score": quality_metrics.overall_score,
                    "formatting_score": quality_metrics.formatting_score,
                    "content_score": quality_metrics.content_score,
                    "accessibility_score": quality_metrics.accessibility_score,
                    "professional_score": quality_metrics.professional_score,
                    "issues": quality_metrics.issues,
                    "suggestions": quality_metrics.suggestions
                }
            }
            
        except Exception as e:
            logger.error(f"Validation request failed: {e}")
            return {"error": str(e)}
    
    async def _handle_enhance_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle enhancement request via message bus."""
        try:
            content = payload.get("content")
            config_data = payload.get("config", {})
            
            # Create export configuration
            config = ExportConfig(
                format=ExportFormat(config_data.get("format", "pdf")),
                document_type=DocumentType(config_data.get("document_type", "report")),
                quality_level=QualityLevel(config_data.get("quality_level", "professional"))
            )
            
            # Enhance content
            enhanced_content = await self.quality_manager.process_content_for_quality(content, config)
            
            return {"enhanced_content": enhanced_content}
            
        except Exception as e:
            logger.error(f"Enhancement request failed: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Create and start the quality service
    service = QualityService()
    
    async def main():
        await service.start()
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await service.stop()
    
    asyncio.run(main())




