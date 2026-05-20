"""
Generador de Reportes Avanzados
=================================

Sistema para generar reportes detallados y personalizados.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Tipos de reporte"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPARATIVE = "comparative"
    TREND = "trend"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"


@dataclass
class ReportSection:
    """Sección de reporte"""
    title: str
    content: str
    data: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict[str, Any]]] = None


class AdvancedReportGenerator:
    """
    Generador de reportes avanzados
    
    Genera reportes completos con:
    - Métricas y estadísticas
    - Gráficos y visualizaciones
    - Análisis comparativo
    - Tendencias temporales
    - Recomendaciones
    """
    
    def __init__(self):
        """Inicializar generador"""
        logger.info("AdvancedReportGenerator inicializado")
    
    def generate_report(
        self,
        data: Dict[str, Any],
        report_type: ReportType = ReportType.SUMMARY,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generar reporte
        
        Args:
            data: Datos para el reporte
            report_type: Tipo de reporte
            template: Plantilla personalizada
        
        Returns:
            Reporte generado
        """
        report = {
            "metadata": {
                "type": report_type.value,
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "sections": []
        }
        
        if report_type == ReportType.SUMMARY:
            report["sections"] = self._generate_summary_sections(data)
        elif report_type == ReportType.DETAILED:
            report["sections"] = self._generate_detailed_sections(data)
        elif report_type == ReportType.COMPARATIVE:
            report["sections"] = self._generate_comparative_sections(data)
        elif report_type == ReportType.TREND:
            report["sections"] = self._generate_trend_sections(data)
        elif report_type == ReportType.EXECUTIVE:
            report["sections"] = self._generate_executive_sections(data)
        elif report_type == ReportType.TECHNICAL:
            report["sections"] = self._generate_technical_sections(data)
        
        # Agregar recomendaciones
        report["recommendations"] = self._generate_recommendations(data)
        
        return report
    
    def _generate_summary_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generar secciones de resumen"""
        sections = []
        
        # Overview
        sections.append(ReportSection(
            title="Resumen Ejecutivo",
            content=f"Análisis completado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            data={
                "total_documents": data.get("total_documents", 0),
                "total_analyses": data.get("total_analyses", 0),
                "success_rate": data.get("success_rate", 0)
            }
        ))
        
        # Key Metrics
        sections.append(ReportSection(
            title="Métricas Clave",
            content="Métricas principales del análisis",
            data=data.get("metrics", {})
        ))
        
        return sections
    
    def _generate_detailed_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generar secciones detalladas"""
        sections = self._generate_summary_sections(data)
        
        # Agregar secciones adicionales
        if "performance" in data:
            sections.append(ReportSection(
                title="Análisis de Rendimiento",
                content="Métricas detalladas de rendimiento",
                data=data["performance"],
                charts=[{
                    "type": "line",
                    "title": "Tiempo de Procesamiento",
                    "data": data["performance"].get("timeline", [])
                }]
            ))
        
        if "errors" in data:
            sections.append(ReportSection(
                title="Errores y Advertencias",
                content="Resumen de errores encontrados",
                data={"errors": data["errors"]}
            ))
        
        return sections
    
    def _generate_comparative_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generar secciones comparativas"""
        sections = []
        
        if "comparisons" in data:
            sections.append(ReportSection(
                title="Análisis Comparativo",
                content="Comparación entre diferentes períodos o grupos",
                data=data["comparisons"],
                charts=[{
                    "type": "bar",
                    "title": "Comparación",
                    "data": data["comparisons"]
                }]
            ))
        
        return sections
    
    def _generate_trend_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generar secciones de tendencias"""
        sections = []
        
        if "trends" in data:
            sections.append(ReportSection(
                title="Análisis de Tendencias",
                content="Tendencias temporales identificadas",
                data=data["trends"],
                charts=[{
                    "type": "line",
                    "title": "Tendencias",
                    "data": data["trends"]
                }]
            ))
        
        return sections
    
    def _generate_executive_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generar secciones ejecutivas"""
        sections = []
        
        sections.append(ReportSection(
            title="Resumen Ejecutivo",
            content="Resumen de alto nivel para ejecutivos",
            data={
                "key_findings": data.get("key_findings", []),
                "impact": data.get("impact", {}),
                "next_steps": data.get("next_steps", [])
            }
        ))
        
        return sections
    
    def _generate_technical_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Generar secciones técnicas"""
        sections = []
        
        sections.append(ReportSection(
            title="Especificaciones Técnicas",
            content="Detalles técnicos del análisis",
            data={
                "models_used": data.get("models_used", []),
                "processing_time": data.get("processing_time", 0),
                "resources_used": data.get("resources_used", {})
            }
        ))
        
        return sections
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        # Recomendaciones basadas en métricas
        if data.get("success_rate", 1.0) < 0.9:
            recommendations.append("Considerar optimizar el proceso de análisis para mejorar la tasa de éxito")
        
        if data.get("avg_processing_time", 0) > 10:
            recommendations.append("Optimizar el tiempo de procesamiento mediante caching o paralelización")
        
        if data.get("error_count", 0) > 0:
            recommendations.append("Revisar y corregir los errores encontrados en el análisis")
        
        return recommendations
    
    def export_report(
        self,
        report: Dict[str, Any],
        format: str = "json"
    ) -> str:
        """
        Exportar reporte
        
        Args:
            report: Reporte generado
            format: Formato (json, markdown, html)
        
        Returns:
            Reporte en formato solicitado
        """
        if format == "json":
            import json
            return json.dumps(report, indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            return self._export_markdown(report)
        
        elif format == "html":
            return self._export_html(report)
        
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _export_markdown(self, report: Dict[str, Any]) -> str:
        """Exportar a Markdown"""
        md = f"# Reporte de Análisis\n\n"
        md += f"**Generado:** {report['metadata']['generated_at']}\n\n"
        
        for section in report.get("sections", []):
            md += f"## {section.title}\n\n"
            md += f"{section.content}\n\n"
            if section.data:
                md += "```json\n"
                import json
                md += json.dumps(section.data, indent=2)
                md += "\n```\n\n"
        
        if report.get("recommendations"):
            md += "## Recomendaciones\n\n"
            for rec in report["recommendations"]:
                md += f"- {rec}\n"
        
        return md
    
    def _export_html(self, report: Dict[str, Any]) -> str:
        """Exportar a HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Análisis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #eee; }}
                .section {{ margin: 20px 0; }}
                .recommendations {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Reporte de Análisis</h1>
            <p><strong>Generado:</strong> {report['metadata']['generated_at']}</p>
        """
        
        for section in report.get("sections", []):
            html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <p>{section.content}</p>
            </div>
            """
        
        if report.get("recommendations"):
            html += """
            <div class="recommendations">
                <h2>Recomendaciones</h2>
                <ul>
            """
            for rec in report["recommendations"]:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"
        
        html += "</body></html>"
        return html


# Instancia global
_report_generator: Optional[AdvancedReportGenerator] = None


def get_report_generator() -> AdvancedReportGenerator:
    """Obtener instancia global del generador"""
    global _report_generator
    if _report_generator is None:
        _report_generator = AdvancedReportGenerator()
    return _report_generator
















