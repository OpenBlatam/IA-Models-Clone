"""
Templates Service - Sistema de plantillas
==========================================

Sistema de plantillas para CVs, cartas de presentación, etc.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class TemplateType(str):
    """Tipos de plantillas"""
    CV = "cv"
    COVER_LETTER = "cover_letter"
    LINKEDIN_PROFILE = "linkedin_profile"
    EMAIL = "email"
    THANK_YOU_NOTE = "thank_you_note"


@dataclass
class Template:
    """Plantilla"""
    id: str
    name: str
    description: str
    template_type: str
    content: str
    variables: List[str] = None
    category: str = "general"
    created_at: datetime = None


class TemplatesService:
    """Servicio de plantillas"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.templates: Dict[str, Template] = {}
        self._initialize_default_templates()
        logger.info("TemplatesService initialized")
    
    def _initialize_default_templates(self):
        """Inicializar plantillas por defecto"""
        # Plantilla de CV
        cv_template = Template(
            id="cv_template_1",
            name="CV Moderno",
            description="Plantilla de CV moderna y profesional",
            template_type=TemplateType.CV,
            content="""
# {name}
{email} | {phone} | {location}

## Resumen Profesional
{summary}

## Experiencia
{experience}

## Educación
{education}

## Habilidades
{skills}
            """.strip(),
            variables=["name", "email", "phone", "location", "summary", "experience", "education", "skills"],
            created_at=datetime.now()
        )
        
        # Plantilla de carta de presentación
        cover_letter_template = Template(
            id="cover_letter_template_1",
            name="Carta de Presentación Estándar",
            description="Plantilla de carta de presentación profesional",
            template_type=TemplateType.COVER_LETTER,
            content="""
{date}

{company_name}
{company_address}

Estimado/a {hiring_manager_name},

Me dirijo a usted para expresar mi interés en el puesto de {job_title}...

{body}

Atentamente,
{your_name}
            """.strip(),
            variables=["date", "company_name", "company_address", "hiring_manager_name", "job_title", "body", "your_name"],
            created_at=datetime.now()
        )
        
        self.templates[cv_template.id] = cv_template
        self.templates[cover_letter_template.id] = cover_letter_template
    
    def get_template(self, template_id: str) -> Optional[Template]:
        """Obtener plantilla"""
        return self.templates.get(template_id)
    
    def get_templates_by_type(self, template_type: str) -> List[Template]:
        """Obtener plantillas por tipo"""
        return [
            t for t in self.templates.values()
            if t.template_type == template_type
        ]
    
    def render_template(self, template_id: str, variables: Dict[str, str]) -> str:
        """Renderizar plantilla con variables"""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        content = template.content
        
        # Reemplazar variables
        for key, value in variables.items():
            content = content.replace(f"{{{key}}}", str(value))
        
        return content
    
    def create_template(
        self,
        name: str,
        description: str,
        template_type: str,
        content: str,
        variables: List[str]
    ) -> Template:
        """Crear nueva plantilla"""
        template = Template(
            id=f"template_{int(datetime.now().timestamp())}",
            name=name,
            description=description,
            template_type=template_type,
            content=content,
            variables=variables,
            created_at=datetime.now()
        )
        
        self.templates[template.id] = template
        return template




