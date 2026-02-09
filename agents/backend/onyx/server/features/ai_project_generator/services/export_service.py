"""
Export Service - Servicio de exportación
=========================================

Servicio independiente para exportación de proyectos.
"""

import logging
from typing import Dict, Any

from ..utils.export_generator import ExportGenerator

logger = logging.getLogger(__name__)


class ExportService:
    """Servicio para exportación de proyectos"""
    
    def __init__(self, exporter: ExportGenerator = None):
        self.exporter = exporter or ExportGenerator()
    
    async def export_project(
        self,
        project_path: str,
        format: str = "zip"
    ) -> Dict[str, Any]:
        """
        Exporta un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            format: Formato de exportación (zip, tar, tar.gz)
        
        Returns:
            Información de exportación
        """
        try:
            if format == "zip":
                result = self.exporter.export_to_zip(project_path)
            elif format == "tar":
                result = self.exporter.export_to_tar(project_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return result
        except Exception as e:
            logger.error(f"Error exporting project: {e}", exc_info=True)
            raise















