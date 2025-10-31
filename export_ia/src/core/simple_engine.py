"""
Simple, practical Export IA Engine - Real and functional.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from .models import ExportConfig, ExportFormat, DocumentType, QualityLevel
from .config import ConfigManager
from .task_manager import TaskManager
from .quality_manager import QualityManager
from ..exporters import ExporterFactory

logger = logging.getLogger(__name__)


class SimpleExportEngine:
    """
    Simple, practical export engine focused on real functionality.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the simple export engine."""
        self.config_manager = ConfigManager(config_path)
        self.task_manager = TaskManager(self.config_manager.system_config)
        self.quality_manager = QualityManager(self.config_manager)
        self.exporters = ExporterFactory()
        
        self._initialized = False
        logger.info("Simple Export Engine initialized")
    
    async def initialize(self):
        """Initialize the engine."""
        if not self._initialized:
            await self.task_manager.start()
            self._initialized = True
            logger.info("Simple Export Engine fully initialized")
    
    async def shutdown(self):
        """Shutdown the engine."""
        if self._initialized:
            await self.task_manager.stop()
            self._initialized = False
            logger.info("Simple Export Engine shutdown complete")
    
    async def export_document(
        self,
        content: Dict[str, Any],
        config: ExportConfig,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export a document - simple and direct.
        
        Args:
            content: Document content
            config: Export configuration
            output_path: Optional output path
            
        Returns:
            Task ID for tracking
        """
        if not self._initialized:
            await self.initialize()
        
        # Validate input
        if not content:
            raise ValueError("Content is required")
        
        if not config.format:
            raise ValueError("Export format is required")
        
        # Submit task
        task_id = await self.task_manager.submit_task(content, config, output_path)
        
        logger.info(f"Export task created: {task_id}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        return await self.task_manager.get_task_status(task_id)
    
    async def get_export_statistics(self) -> Dict[str, Any]:
        """Get simple export statistics."""
        stats = self.task_manager.get_statistics()
        return {
            "total_tasks": stats.total_tasks,
            "active_tasks": stats.active_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "average_processing_time": stats.average_processing_time
        }
    
    def list_supported_formats(self) -> List[Dict[str, Any]]:
        """List supported formats."""
        return self.exporters.list_supported_formats()
    
    def get_document_template(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Get document template."""
        return self.config_manager.get_template(doc_type)
    
    def validate_content(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validate content and return quality metrics."""
        return self.quality_manager.get_quality_metrics(content, config)
    
    async def wait_for_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for task completion."""
        start_time = datetime.now()
        
        while True:
            status = await self.get_task_status(task_id)
            if not status:
                raise ValueError(f"Task not found: {task_id}")
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(2)
    
    async def export_and_wait(
        self,
        content: Dict[str, Any],
        config: ExportConfig,
        output_path: Optional[str] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Export document and wait for completion."""
        task_id = await self.export_document(content, config, output_path)
        return await self.wait_for_completion(task_id, timeout)


# Global engine instance
_engine: Optional[SimpleExportEngine] = None


def get_export_engine() -> SimpleExportEngine:
    """Get the global export engine instance."""
    global _engine
    if _engine is None:
        _engine = SimpleExportEngine()
    return _engine




