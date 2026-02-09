"""
Sistema de plantillas para reportes
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ReportTemplate(str, Enum):
    """Tipos de plantillas de reporte"""
    BASIC = "basic"
    DETAILED = "detailed"
    PROFESSIONAL = "professional"
    SUMMARY = "summary"
    COMPARISON = "comparison"


@dataclass
class TemplateConfig:
    """Configuración de plantilla"""
    name: str
    description: str
    include_charts: bool = True
    include_recommendations: bool = True
    include_history: bool = False
    include_comparison: bool = False
    language: str = "es"
    style: str = "modern"


class ReportTemplateEngine:
    """Motor de plantillas para reportes"""
    
    def __init__(self):
        """Inicializa el motor de plantillas"""
        self.templates: Dict[str, TemplateConfig] = {
            ReportTemplate.BASIC: TemplateConfig(
                name="Básico",
                description="Reporte básico con información esencial",
                include_charts=False,
                include_recommendations=True,
                include_history=False
            ),
            ReportTemplate.DETAILED: TemplateConfig(
                name="Detallado",
                description="Reporte detallado con todas las métricas",
                include_charts=True,
                include_recommendations=True,
                include_history=True
            ),
            ReportTemplate.PROFESSIONAL: TemplateConfig(
                name="Profesional",
                description="Reporte profesional para dermatólogos",
                include_charts=True,
                include_recommendations=True,
                include_history=True,
                style="professional"
            ),
            ReportTemplate.SUMMARY: TemplateConfig(
                name="Resumen",
                description="Resumen ejecutivo del análisis",
                include_charts=False,
                include_recommendations=False,
                include_history=False
            ),
            ReportTemplate.COMPARISON: TemplateConfig(
                name="Comparación",
                description="Reporte de comparación entre análisis",
                include_charts=True,
                include_recommendations=True,
                include_history=True,
                include_comparison=True
            )
        }
    
    def get_template(self, template_name: str) -> Optional[TemplateConfig]:
        """Obtiene configuración de plantilla"""
        return self.templates.get(template_name)
    
    def generate_report_structure(self, template_name: str,
                                 analysis_data: Dict) -> Dict:
        """
        Genera estructura de reporte según plantilla
        
        Args:
            template_name: Nombre de la plantilla
            analysis_data: Datos del análisis
            
        Returns:
            Estructura del reporte
        """
        template = self.get_template(template_name)
        
        if not template:
            template = self.templates[ReportTemplate.BASIC]
        
        report_structure = {
            "template": template_name,
            "metadata": {
                "title": f"Reporte de Análisis de Piel - {template.name}",
                "generated_at": analysis_data.get("timestamp", ""),
                "language": template.language
            }
        }
        
        # Información básica
        report_structure["basic_info"] = {
            "overall_score": analysis_data.get("quality_scores", {}).get("overall_score", 0),
            "skin_type": analysis_data.get("skin_type", "unknown"),
            "conditions": analysis_data.get("conditions", [])
        }
        
        # Métricas detalladas (si está habilitado)
        if template.include_charts or template_name == ReportTemplate.DETAILED:
            report_structure["detailed_metrics"] = analysis_data.get("quality_scores", {})
        
        # Recomendaciones (si está habilitado)
        if template.include_recommendations:
            report_structure["recommendations"] = analysis_data.get("recommendations", {})
        
        # Historial (si está habilitado)
        if template.include_history:
            report_structure["history"] = analysis_data.get("history", [])
        
        # Comparación (si está habilitado)
        if template.include_comparison:
            report_structure["comparison"] = analysis_data.get("comparison", {})
        
        return report_structure
    
    def list_templates(self) -> List[Dict]:
        """Lista todas las plantillas disponibles"""
        return [
            {
                "name": name,
                "config": {
                    "name": config.name,
                    "description": config.description,
                    "include_charts": config.include_charts,
                    "include_recommendations": config.include_recommendations,
                    "include_history": config.include_history,
                    "include_comparison": config.include_comparison
                }
            }
            for name, config in self.templates.items()
        ]






