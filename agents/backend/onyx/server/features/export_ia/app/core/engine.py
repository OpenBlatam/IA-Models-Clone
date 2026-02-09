"""
Export Engine - Motor principal de exportación
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import ExportConfig, ExportFormat, DocumentType, QualityLevel
from .config import ConfigManager
from .task_manager import TaskManager
from .quality_manager import QualityManager
from ..exporters import ExporterFactory

logger = logging.getLogger(__name__)


class ExportEngine:
    """
    Motor principal de exportación de documentos.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Inicializar el motor de exportación."""
        self.config_manager = ConfigManager(config_path)
        self.task_manager = TaskManager(self.config_manager.system_config)
        self.quality_manager = QualityManager(self.config_manager)
        self.exporters = ExporterFactory()
        
        self._initialized = False
        logger.info("Export Engine inicializado")
    
    async def initialize(self):
        """Inicializar el motor."""
        if not self._initialized:
            await self.task_manager.start()
            self._initialized = True
            logger.info("Export Engine completamente inicializado")
    
    async def shutdown(self):
        """Cerrar el motor."""
        if self._initialized:
            await self.task_manager.stop()
            self._initialized = False
            logger.info("Export Engine cerrado")
    
    async def export_document(
        self,
        content: Dict[str, Any],
        config: ExportConfig,
        output_path: Optional[str] = None
    ) -> str:
        """
        Exportar un documento.
        
        Args:
            content: Contenido del documento
            config: Configuración de exportación
            output_path: Ruta de salida opcional
            
        Returns:
            ID de la tarea para seguimiento
        """
        if not self._initialized:
            await self.initialize()
        
        # Validar entrada
        if not content:
            raise ValueError("El contenido es requerido")
        
        if not config.format:
            raise ValueError("El formato de exportación es requerido")
        
        # Enviar tarea
        task_id = await self.task_manager.submit_task(content, config, output_path)
        
        logger.info(f"Tarea de exportación creada: {task_id}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de una tarea."""
        return await self.task_manager.get_task_status(task_id)
    
    async def get_export_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de exportación."""
        stats = self.task_manager.get_statistics()
        return {
            "total_tasks": stats.total_tasks,
            "active_tasks": stats.active_tasks,
            "completed_tasks": stats.completed_tasks,
            "failed_tasks": stats.failed_tasks,
            "average_processing_time": stats.average_processing_time
        }
    
    def list_supported_formats(self) -> List[Dict[str, Any]]:
        """Listar formatos soportados."""
        return self.exporters.list_supported_formats()
    
    def get_document_template(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Obtener plantilla de documento."""
        return self.config_manager.get_template(doc_type)
    
    def validate_content(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validar contenido y obtener métricas de calidad."""
        return self.quality_manager.get_quality_metrics(content, config)
    
    async def wait_for_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Esperar a que se complete una tarea."""
        start_time = datetime.now()
        
        while True:
            status = await self.get_task_status(task_id)
            if not status:
                raise ValueError(f"Tarea no encontrada: {task_id}")
            
            if status["status"] in ["completed", "failed", "cancelled"]:
                return status
            
            # Verificar timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Tarea {task_id} expiró después de {timeout} segundos")
            
            await asyncio.sleep(2)
    
    async def export_and_wait(
        self,
        content: Dict[str, Any],
        config: ExportConfig,
        output_path: Optional[str] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Exportar documento y esperar a que se complete."""
        task_id = await self.export_document(content, config, output_path)
        return await self.wait_for_completion(task_id, timeout)


# Instancia global del motor
_engine: Optional[ExportEngine] = None


def get_export_engine() -> ExportEngine:
    """Obtener la instancia global del motor de exportación."""
    global _engine
    if _engine is None:
        _engine = ExportEngine()
    return _engine




