"""
Sistema de reportes avanzados
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json


class ReportFormat(str, Enum):
    """Formatos de reporte"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    EXCEL = "excel"
    CSV = "csv"
    MARKDOWN = "markdown"
    XML = "xml"
    YAML = "yaml"


@dataclass
class ReportConfig:
    """Configuración de reporte"""
    format: ReportFormat
    include_charts: bool = True
    include_images: bool = True
    include_recommendations: bool = True
    include_history: bool = False
    language: str = "es"
    theme: str = "default"
    custom_sections: List[str] = None
    
    def __post_init__(self):
        if self.custom_sections is None:
            self.custom_sections = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "format": self.format.value,
            "include_charts": self.include_charts,
            "include_images": self.include_images,
            "include_recommendations": self.include_recommendations,
            "include_history": self.include_history,
            "language": self.language,
            "theme": self.theme,
            "custom_sections": self.custom_sections
        }


class AdvancedReporting:
    """Sistema de reportes avanzados"""
    
    def __init__(self):
        """Inicializa el sistema"""
        pass
    
    def generate_advanced_report(self, analysis_data: Dict,
                                config: ReportConfig) -> bytes:
        """
        Genera reporte avanzado
        
        Args:
            analysis_data: Datos del análisis
            config: Configuración del reporte
            
        Returns:
            Bytes del reporte
        """
        if config.format == ReportFormat.PDF:
            return self._generate_pdf(analysis_data, config)
        elif config.format == ReportFormat.HTML:
            return self._generate_html(analysis_data, config)
        elif config.format == ReportFormat.JSON:
            return self._generate_json(analysis_data, config)
        elif config.format == ReportFormat.EXCEL:
            return self._generate_excel(analysis_data, config)
        elif config.format == ReportFormat.CSV:
            return self._generate_csv(analysis_data, config)
        elif config.format == ReportFormat.MARKDOWN:
            return self._generate_markdown(analysis_data, config)
        else:
            return self._generate_json(analysis_data, config)
    
    def _generate_pdf(self, data: Dict, config: ReportConfig) -> bytes:
        """Genera PDF"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            import io
            
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # Título
            c.setFont("Helvetica-Bold", 20)
            c.drawString(100, 750, "Reporte de Análisis de Piel")
            
            # Contenido
            y = 700
            c.setFont("Helvetica", 12)
            c.drawString(100, y, f"Score General: {data.get('quality_scores', {}).get('overall_score', 0):.1f}")
            
            c.save()
            buffer.seek(0)
            return buffer.read()
        except ImportError:
            return b"PDF generation requires reportlab"
    
    def _generate_html(self, data: Dict, config: ReportConfig) -> bytes:
        """Genera HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .score {{ font-size: 24px; color: #4CAF50; }}
            </style>
        </head>
        <body>
            <h1>Reporte de Análisis de Piel</h1>
            <div class="score">
                Score: {data.get('quality_scores', {}).get('overall_score', 0):.1f}
            </div>
        </body>
        </html>
        """
        return html.encode('utf-8')
    
    def _generate_json(self, data: Dict, config: ReportConfig) -> bytes:
        """Genera JSON"""
        return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
    
    def _generate_excel(self, data: Dict, config: ReportConfig) -> bytes:
        """Genera Excel"""
        try:
            import pandas as pd
            from io import BytesIO
            
            # Crear DataFrame
            df = pd.DataFrame([data])
            
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Analysis', index=False)
            
            buffer.seek(0)
            return buffer.read()
        except ImportError:
            return b"Excel generation requires pandas and openpyxl"
    
    def _generate_csv(self, data: Dict, config: ReportConfig) -> bytes:
        """Genera CSV"""
        import csv
        import io
        
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        
        return buffer.getvalue().encode('utf-8')
    
    def _generate_markdown(self, data: Dict, config: ReportConfig) -> bytes:
        """Genera Markdown"""
        md = f"""# Reporte de Análisis de Piel

## Score General
{data.get('quality_scores', {}).get('overall_score', 0):.1f}

## Métricas Detalladas
"""
        for key, value in data.get('quality_scores', {}).items():
            md += f"- **{key}**: {value:.1f}\n"
        
        return md.encode('utf-8')
    
    def get_supported_formats(self) -> List[str]:
        """Obtiene formatos soportados"""
        return [f.value for f in ReportFormat]






