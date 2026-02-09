"""
Sistema de Plantillas de Reportes
==================================
Plantillas personalizables para reportes
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
from enum import Enum
import structlog
import json

from .models import ValidationReport, PsychologicalProfile

logger = structlog.get_logger()


class TemplateType(str, Enum):
    """Tipos de plantilla"""
    EXECUTIVE = "executive"
    DETAILED = "detailed"
    SUMMARY = "summary"
    CLINICAL = "clinical"
    PERSONAL = "personal"
    CUSTOM = "custom"


class ReportTemplate:
    """Plantilla de reporte"""
    
    def __init__(
        self,
        id: UUID,
        name: str,
        template_type: TemplateType,
        sections: List[Dict[str, Any]],
        style: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.name = name
        self.template_type = template_type
        self.sections = sections
        self.style = style or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.template_type.value,
            "sections": self.sections,
            "style": self.style,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class TemplateManager:
    """Gestor de plantillas"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._templates: Dict[UUID, ReportTemplate] = {}
        self._load_default_templates()
        logger.info("TemplateManager initialized")
    
    def _load_default_templates(self) -> None:
        """Cargar plantillas por defecto"""
        from uuid import uuid4
        
        # Plantilla ejecutiva
        executive_template = ReportTemplate(
            id=uuid4(),
            name="Executive Summary",
            template_type=TemplateType.EXECUTIVE,
            sections=[
                {
                    "name": "summary",
                    "title": "Resumen Ejecutivo",
                    "required": True
                },
                {
                    "name": "key_findings",
                    "title": "Hallazgos Clave",
                    "required": True
                },
                {
                    "name": "recommendations",
                    "title": "Recomendaciones",
                    "required": True
                }
            ],
            style={"format": "professional", "color_scheme": "blue"}
        )
        self._templates[executive_template.id] = executive_template
        
        # Plantilla detallada
        detailed_template = ReportTemplate(
            id=uuid4(),
            name="Detailed Analysis",
            template_type=TemplateType.DETAILED,
            sections=[
                {
                    "name": "summary",
                    "title": "Resumen",
                    "required": True
                },
                {
                    "name": "personality_analysis",
                    "title": "Análisis de Personalidad",
                    "required": True
                },
                {
                    "name": "emotional_analysis",
                    "title": "Análisis Emocional",
                    "required": True
                },
                {
                    "name": "behavioral_patterns",
                    "title": "Patrones de Comportamiento",
                    "required": True
                },
                {
                    "name": "social_media_insights",
                    "title": "Insights de Redes Sociales",
                    "required": True
                },
                {
                    "name": "recommendations",
                    "title": "Recomendaciones",
                    "required": True
                }
            ],
            style={"format": "comprehensive", "color_scheme": "green"}
        )
        self._templates[detailed_template.id] = detailed_template
        
        # Plantilla clínica
        clinical_template = ReportTemplate(
            id=uuid4(),
            name="Clinical Report",
            template_type=TemplateType.CLINICAL,
            sections=[
                {
                    "name": "clinical_summary",
                    "title": "Resumen Clínico",
                    "required": True
                },
                {
                    "name": "risk_assessment",
                    "title": "Evaluación de Riesgos",
                    "required": True
                },
                {
                    "name": "personality_profile",
                    "title": "Perfil de Personalidad",
                    "required": True
                },
                {
                    "name": "treatment_recommendations",
                    "title": "Recomendaciones de Tratamiento",
                    "required": True
                }
            ],
            style={"format": "clinical", "color_scheme": "red"}
        )
        self._templates[clinical_template.id] = clinical_template
    
    def create_template(
        self,
        name: str,
        template_type: TemplateType,
        sections: List[Dict[str, Any]],
        style: Optional[Dict[str, Any]] = None
    ) -> ReportTemplate:
        """
        Crear nueva plantilla
        
        Args:
            name: Nombre de la plantilla
            template_type: Tipo de plantilla
            sections: Secciones de la plantilla
            style: Estilo (opcional)
            
        Returns:
            Plantilla creada
        """
        from uuid import uuid4
        
        template = ReportTemplate(
            id=uuid4(),
            name=name,
            template_type=template_type,
            sections=sections,
            style=style
        )
        
        self._templates[template.id] = template
        
        logger.info("Template created", template_id=str(template.id), name=name)
        
        return template
    
    def get_template(self, template_id: UUID) -> Optional[ReportTemplate]:
        """
        Obtener plantilla por ID
        
        Args:
            template_id: ID de la plantilla
            
        Returns:
            Plantilla o None
        """
        return self._templates.get(template_id)
    
    def get_templates(
        self,
        template_type: Optional[TemplateType] = None
    ) -> List[ReportTemplate]:
        """
        Obtener plantillas
        
        Args:
            template_type: Filtrar por tipo (opcional)
            
        Returns:
            Lista de plantillas
        """
        templates = list(self._templates.values())
        
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
        
        return templates
    
    def generate_report_from_template(
        self,
        template: ReportTemplate,
        report: ValidationReport,
        profile: Optional[PsychologicalProfile] = None
    ) -> Dict[str, Any]:
        """
        Generar reporte desde plantilla
        
        Args:
            template: Plantilla a usar
            report: Reporte base
            profile: Perfil psicológico (opcional)
            
        Returns:
            Reporte generado desde plantilla
        """
        generated_sections = {}
        
        for section in template.sections:
            section_name = section["name"]
            
            if section_name == "summary":
                generated_sections[section_name] = {
                    "title": section["title"],
                    "content": report.summary
                }
            elif section_name == "personality_analysis" and profile:
                generated_sections[section_name] = {
                    "title": section["title"],
                    "content": {
                        "traits": profile.personality_traits,
                        "confidence": profile.confidence_score
                    }
                }
            elif section_name == "social_media_insights":
                generated_sections[section_name] = {
                    "title": section["title"],
                    "content": report.social_media_insights
                }
            elif section_name == "recommendations" and profile:
                generated_sections[section_name] = {
                    "title": section["title"],
                    "content": profile.recommendations
                }
            # Agregar más secciones según necesidad
        
        return {
            "template_id": str(template.id),
            "template_name": template.name,
            "generated_at": datetime.utcnow().isoformat(),
            "sections": generated_sections,
            "style": template.style
        }


# Instancia global del gestor de plantillas
template_manager = TemplateManager()




