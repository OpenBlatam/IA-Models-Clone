"""
Main Export IA Engine - Refactored and Modular with Advanced Features
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import ExportConfig, ExportFormat, DocumentType, QualityLevel, ExportStatistics
from .config import ConfigManager, SystemConfig
from .task_manager import TaskManager
from .quality_manager import QualityManager
from .cache import get_cache_manager
from .monitoring import get_monitoring_manager
from .validation import get_validation_manager
from ..exporters import ExporterFactory
from ..plugins import get_plugin_manager

logger = logging.getLogger(__name__)


class ExportIAEngine:
    """
    Advanced AI-powered export engine for professional document generation.
    Refactored with modular architecture and clear separation of concerns.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Export IA Engine."""
        self.config_manager = ConfigManager(config_path)
        self.system_config = self.config_manager.system_config
        self.task_manager = TaskManager(self.system_config)
        self.quality_manager = QualityManager(self.config_manager)
        
        # Advanced components
        self.cache_manager = get_cache_manager()
        self.monitoring_manager = get_monitoring_manager()
        self.validation_manager = get_validation_manager()
        self.plugin_manager = get_plugin_manager()
        
        # Initialize components
        self._initialized = False
        
        logger.info("Export IA Engine initialized with advanced modular architecture")
    
    async def initialize(self):
        """Initialize the engine and start background services."""
        if not self._initialized:
            # Initialize core components
            await self.task_manager.start()
            
            # Initialize advanced components
            await self.monitoring_manager.initialize()
            await self.plugin_manager.initialize()
            
            self._initialized = True
            logger.info("Export IA Engine fully initialized with advanced features")
    
    async def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        if self._initialized:
            # Shutdown advanced components
            await self.plugin_manager.cleanup()
            await self.monitoring_manager.shutdown()
            
            # Shutdown core components
            await self.task_manager.stop()
            
            self._initialized = False
            logger.info("Export IA Engine shutdown complete")
    
    async def export_document(
        self,
        content: Dict[str, Any],
        config: ExportConfig,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export a document in the specified format with professional quality.
        
        Args:
            content: Document content to export
            config: Export configuration
            output_path: Optional output file path
            
        Returns:
            Task ID for tracking the export process
        """
        if not self._initialized:
            await self.initialize()
        
        # Validate input
        validation_results = self.validation_manager.validate_export_request(
            content, config, output_path
        )
        
        if self.validation_manager.has_errors(validation_results):
            error_messages = [r.message for r in self.validation_manager.get_errors(validation_results)]
            raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
        
        # Log warnings if any
        warnings = self.validation_manager.get_warnings(validation_results)
        if warnings:
            for warning in warnings:
                logger.warning(f"Validation warning: {warning.message}")
        
        # Record metrics
        await self.monitoring_manager.metrics_collector.record_counter("exports.requested")
        
        # Submit task to task manager
        task_id = await self.task_manager.submit_task(content, config, output_path)
        
        logger.info(f"Export task created: {task_id} - {config.format.value} format")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an export task."""
        return await self.task_manager.get_task_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        return await self.task_manager.cancel_task(task_id)
    
    def get_export_statistics(self) -> ExportStatistics:
        """Get export statistics."""
        return self.task_manager.get_statistics()
    
    def list_supported_formats(self) -> List[Dict[str, Any]]:
        """List all supported export formats."""
        return [
            {
                "format": fmt.value,
                "name": fmt.value.upper(),
                "description": f"Export to {fmt.value.upper()} format",
                "professional_features": self.config_manager.get_format_features(fmt)
            }
            for fmt in ExportFormat
        ]
    
    def get_quality_config(self, level: QualityLevel) -> Any:
        """Get quality configuration for a specific level."""
        return self.config_manager.get_quality_config(level)
    
    def get_document_template(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Get template for a specific document type."""
        return self.config_manager.get_template(doc_type)
    
    def validate_content(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validate content and return quality metrics."""
        return self.quality_manager.get_quality_metrics(content, config)
    
    # Advanced features
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        return await self.monitoring_manager.get_health_status()
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get system metrics summary."""
        return await self.monitoring_manager.get_metrics_summary()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return asyncio.run(self.cache_manager.get_all_stats())
    
    def list_plugins(self) -> Dict[str, Any]:
        """List all registered plugins."""
        return self.plugin_manager.list_plugins()
    
    def register_plugin(self, plugin: Any) -> None:
        """Register a plugin."""
        self.plugin_manager.register_plugin(plugin)
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin."""
        return self.plugin_manager.configure_plugin(plugin_name, config)
    
    def _validate_config(self, config: ExportConfig):
        """Validate export configuration."""
        if not isinstance(config.format, ExportFormat):
            raise ValueError(f"Invalid export format: {config.format}")
        
        if not isinstance(config.document_type, DocumentType):
            raise ValueError(f"Invalid document type: {config.document_type}")
        
        if not isinstance(config.quality_level, QualityLevel):
            raise ValueError(f"Invalid quality level: {config.quality_level}")
        
        # Check if format is supported
        if not ExporterFactory.is_format_supported(config.format):
            raise ValueError(f"Unsupported export format: {config.format}")
    
    # Convenience methods for common operations
    async def export_to_pdf(
        self,
        content: Dict[str, Any],
        document_type: DocumentType = DocumentType.REPORT,
        quality_level: QualityLevel = QualityLevel.PROFESSIONAL,
        output_path: Optional[str] = None
    ) -> str:
        """Convenience method to export to PDF."""
        config = ExportConfig(
            format=ExportFormat.PDF,
            document_type=document_type,
            quality_level=quality_level
        )
        return await self.export_document(content, config, output_path)
    
    async def export_to_docx(
        self,
        content: Dict[str, Any],
        document_type: DocumentType = DocumentType.REPORT,
        quality_level: QualityLevel = QualityLevel.PROFESSIONAL,
        output_path: Optional[str] = None
    ) -> str:
        """Convenience method to export to DOCX."""
        config = ExportConfig(
            format=ExportFormat.DOCX,
            document_type=document_type,
            quality_level=quality_level
        )
        return await self.export_document(content, config, output_path)
    
    async def export_to_html(
        self,
        content: Dict[str, Any],
        document_type: DocumentType = DocumentType.REPORT,
        quality_level: QualityLevel = QualityLevel.PROFESSIONAL,
        output_path: Optional[str] = None
    ) -> str:
        """Convenience method to export to HTML."""
        config = ExportConfig(
            format=ExportFormat.HTML,
            document_type=document_type,
            quality_level=quality_level
        )
        return await self.export_document(content, config, output_path)
    
    async def export_to_markdown(
        self,
        content: Dict[str, Any],
        document_type: DocumentType = DocumentType.REPORT,
        quality_level: QualityLevel = QualityLevel.PROFESSIONAL,
        output_path: Optional[str] = None
    ) -> str:
        """Convenience method to export to Markdown."""
        config = ExportConfig(
            format=ExportFormat.MARKDOWN,
            document_type=document_type,
            quality_level=quality_level
        )
        return await self.export_document(content, config, output_path)
    
    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# Global export engine instance
_global_export_engine: Optional[ExportIAEngine] = None


def get_global_export_engine() -> ExportIAEngine:
    """Get the global export engine instance."""
    global _global_export_engine
    if _global_export_engine is None:
        _global_export_engine = ExportIAEngine()
    return _global_export_engine


async def get_async_export_engine() -> ExportIAEngine:
    """Get an initialized async export engine instance."""
    engine = ExportIAEngine()
    await engine.initialize()
    return engine
