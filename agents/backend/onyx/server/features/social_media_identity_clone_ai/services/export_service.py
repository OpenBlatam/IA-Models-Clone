"""
Servicio de exportación de datos
"""

import logging
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from io import StringIO

from ..core.models import IdentityProfile, GeneratedContent
from ..services.storage_service import StorageService

logger = logging.getLogger(__name__)


class ExportService:
    """Servicio para exportar datos en diferentes formatos"""
    
    def __init__(self):
        self.storage = StorageService()
    
    def export_identity_json(self, identity_id: str) -> str:
        """
        Exporta identidad en formato JSON
        
        Args:
            identity_id: ID de la identidad
            
        Returns:
            JSON string
        """
        identity = self.storage.get_identity(identity_id)
        if not identity:
            raise ValueError(f"Identidad no encontrada: {identity_id}")
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "identity": identity.model_dump(),
            "generated_content": [
                content.model_dump() 
                for content in self.storage.get_generated_content(identity_id, limit=1000)
            ]
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
    
    def export_identity_csv(self, identity_id: str) -> str:
        """
        Exporta contenido generado en formato CSV
        
        Args:
            identity_id: ID de la identidad
            
        Returns:
            CSV string
        """
        identity = self.storage.get_identity(identity_id)
        if not identity:
            raise ValueError(f"Identidad no encontrada: {identity_id}")
        
        content_list = self.storage.get_generated_content(identity_id, limit=1000)
        
        if not content_list:
            return ""
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow([
            "Content ID",
            "Platform",
            "Content Type",
            "Title",
            "Content",
            "Hashtags",
            "Confidence Score",
            "Generated At"
        ])
        
        # Data
        for content in content_list:
            writer.writerow([
                content.content_id,
                content.platform.value,
                content.content_type.value,
                content.title or "",
                content.content,
                ",".join(content.hashtags),
                content.confidence_score or "",
                content.generated_at.isoformat()
            ])
        
        return output.getvalue()
    
    def export_analytics_json(self, identity_id: str) -> str:
        """
        Exporta analytics de identidad en JSON
        
        Args:
            identity_id: ID de la identidad
            
        Returns:
            JSON string
        """
        from ..analytics.analytics_service import AnalyticsService
        
        analytics_service = AnalyticsService()
        analytics = analytics_service.get_identity_analytics(identity_id)
        
        if not analytics:
            raise ValueError(f"Identidad no encontrada: {identity_id}")
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "analytics": analytics
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
    
    def save_export_to_file(
        self, 
        content: str, 
        filename: str, 
        format: str = "json"
    ) -> Path:
        """
        Guarda exportación en archivo
        
        Args:
            content: Contenido a guardar
            filename: Nombre del archivo
            format: Formato (json, csv)
            
        Returns:
            Path del archivo guardado
        """
        from ..config import get_settings
        
        settings = get_settings()
        export_dir = Path(settings.storage_path) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Agregar extensión si no tiene
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        file_path = export_dir / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Exportación guardada en: {file_path}")
        return file_path




