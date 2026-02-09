"""
Report Generator System
=======================

Sistema de generación de reportes avanzado.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Sección de reporte."""
    title: str
    content: str
    level: int = 1  # Nivel de encabezado (1-6)


@dataclass
class Report:
    """Reporte."""
    report_id: str
    title: str
    author: str
    created_at: str
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """
    Generador de reportes.
    
    Genera reportes en múltiples formatos.
    """
    
    def __init__(self):
        """Inicializar generador de reportes."""
        self.reports: List[Report] = []
        self.templates: Dict[str, str] = {}
    
    def create_report(
        self,
        title: str,
        author: str = "System",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Report:
        """
        Crear nuevo reporte.
        
        Args:
            title: Título del reporte
            author: Autor
            metadata: Metadata adicional
            
        Returns:
            Reporte creado
        """
        report = Report(
            report_id=f"report_{len(self.reports)}",
            title=title,
            author=author,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.reports.append(report)
        return report
    
    def add_section(
        self,
        report: Report,
        title: str,
        content: str,
        level: int = 1
    ) -> ReportSection:
        """
        Agregar sección al reporte.
        
        Args:
            report: Reporte
            title: Título de la sección
            content: Contenido
            level: Nivel de encabezado
            
        Returns:
            Sección creada
        """
        section = ReportSection(
            title=title,
            content=content,
            level=level
        )
        
        report.sections.append(section)
        return section
    
    def generate_markdown(self, report: Report) -> str:
        """
        Generar reporte en Markdown.
        
        Args:
            report: Reporte
            
        Returns:
            Contenido Markdown
        """
        lines = [f"# {report.title}\n"]
        lines.append(f"**Author:** {report.author}\n")
        lines.append(f"**Created:** {report.created_at}\n")
        
        if report.metadata:
            lines.append("\n## Metadata\n")
            for key, value in report.metadata.items():
                lines.append(f"- **{key}:** {value}\n")
        
        for section in report.sections:
            header = "#" * section.level
            lines.append(f"\n{header} {section.title}\n")
            lines.append(f"{section.content}\n")
        
        return '\n'.join(lines)
    
    def generate_html(self, report: Report) -> str:
        """
        Generar reporte en HTML.
        
        Args:
            report: Reporte
            
        Returns:
            Contenido HTML
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .metadata {{ background: #f5f5f5; padding: 10px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <div class="metadata">
        <p><strong>Author:</strong> {report.author}</p>
        <p><strong>Created:</strong> {report.created_at}</p>
    </div>
"""
        
        for section in report.sections:
            tag = f"h{section.level}"
            html += f"    <{tag}>{section.title}</{tag}>\n"
            html += f"    <p>{section.content}</p>\n"
        
        html += """</body>
</html>"""
        
        return html
    
    def generate_pdf(self, report: Report, output_file: str) -> str:
        """
        Generar reporte en PDF.
        
        Args:
            report: Reporte
            output_file: Archivo de salida
            
        Returns:
            Ruta del archivo generado
        """
        # Nota: Requiere biblioteca como reportlab o weasyprint
        # Por ahora, generamos HTML que puede convertirse a PDF
        html_content = self.generate_html(report)
        
        html_path = Path(output_file).with_suffix('.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"PDF report generated (as HTML): {html_path}")
        return str(html_path)
    
    def export_report(
        self,
        report: Report,
        output_file: str,
        format: str = "markdown"
    ) -> str:
        """
        Exportar reporte.
        
        Args:
            report: Reporte
            output_file: Archivo de salida
            format: Formato (markdown, html, pdf)
            
        Returns:
            Ruta del archivo generado
        """
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "markdown":
            content = self.generate_markdown(report)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format == "html":
            content = self.generate_html(report)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format == "pdf":
            return self.generate_pdf(report, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Report exported: {output_file}")
        return str(path)


# Instancia global
_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Obtener instancia global del generador de reportes."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator






