"""
Markdown Exporter
================

Exportador especializado para formato Markdown.
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime
from ...core.base.service_base import BaseService


class MarkdownExporter(BaseService):
    """Exportador de manuales a Markdown."""
    
    def __init__(self):
        """Inicializar exportador."""
        super().__init__(logger_name=__name__)
    
    def export(
        self,
        manual_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Exportar manual a Markdown.
        
        Args:
            manual_content: Contenido del manual
            metadata: Metadata adicional
        
        Returns:
            Contenido en formato Markdown
        """
        markdown = ""
        
        if metadata:
            markdown += self._add_metadata(metadata)
            markdown += "---\n\n"
        
        markdown += self._convert_content(manual_content)
        markdown += self._add_footer()
        
        return markdown
    
    def _add_metadata(self, metadata: Dict[str, Any]) -> str:
        """Agregar metadata al markdown."""
        md = ""
        
        if metadata.get("title"):
            md += f"# {metadata['title']}\n\n"
        if metadata.get("category"):
            md += f"**Categoría:** {metadata['category']}\n\n"
        if metadata.get("difficulty"):
            md += f"**Dificultad:** {metadata['difficulty']}\n\n"
        if metadata.get("estimated_time"):
            md += f"**Tiempo estimado:** {metadata['estimated_time']}\n\n"
        
        return md
    
    def _convert_content(self, content: str) -> str:
        """Convertir contenido a markdown."""
        lines = content.split('\n')
        markdown = ""
        in_list = False
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown += "\n"
                in_list = False
                continue
            
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if in_list:
                    markdown += "\n"
                    in_list = False
                title_text = re.sub(r'^\d+\.\s*', '', line)
                markdown += f"## {title_text}\n\n"
            elif line.startswith('PASO'):
                if in_list:
                    markdown += "\n"
                markdown += f"### {line}\n\n"
                in_list = False
            elif line.startswith(('-', '•', '📋', '🔧', '📦', '⚠️')):
                if not in_list:
                    markdown += "\n"
                markdown += f"{line}\n"
                in_list = True
            else:
                if in_list:
                    markdown += "\n"
                    in_list = False
                markdown += f"{line}\n\n"
        
        return markdown
    
    def _add_footer(self) -> str:
        """Agregar footer al markdown."""
        return f"\n---\n\n*Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

