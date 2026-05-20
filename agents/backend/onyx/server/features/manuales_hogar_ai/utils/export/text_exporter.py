"""
Text Exporter
============

Exportador especializado para formato texto plano.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from ...core.base.service_base import BaseService


class TextExporter(BaseService):
    """Exportador de manuales a texto plano."""
    
    def __init__(self):
        """Inicializar exportador."""
        super().__init__(logger_name=__name__)
    
    def export(
        self,
        manual_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Exportar manual a texto plano.
        
        Args:
            manual_content: Contenido del manual
            metadata: Metadata adicional
        
        Returns:
            Contenido en formato texto plano
        """
        text = ""
        
        if metadata:
            text += self._add_metadata(metadata)
            text += "\n" + "=" * 50 + "\n\n"
        
        text += manual_content
        text += self._add_footer()
        
        return text
    
    def _add_metadata(self, metadata: Dict[str, Any]) -> str:
        """Agregar metadata al texto."""
        txt = ""
        
        if metadata.get("title"):
            txt += f"{metadata['title']}\n"
            txt += "=" * len(metadata['title']) + "\n\n"
        if metadata.get("category"):
            txt += f"Categoría: {metadata['category']}\n"
        if metadata.get("difficulty"):
            txt += f"Dificultad: {metadata['difficulty']}\n"
        if metadata.get("estimated_time"):
            txt += f"Tiempo estimado: {metadata['estimated_time']}\n"
        
        return txt
    
    def _add_footer(self) -> str:
        """Agregar footer al texto."""
        return f"\n\n{'=' * 50}\nGenerado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

