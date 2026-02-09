"""
Sistema de templates para generación de contenido
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ContentTemplate:
    """Template para generación de contenido"""
    template_id: str
    name: str
    description: str
    platform: str  # tiktok, instagram, youtube
    content_type: str  # post, video, etc.
    template: str  # Template con placeholders
    variables: List[str] = field(default_factory=list)  # Variables disponibles
    example: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "platform": self.platform,
            "content_type": self.content_type,
            "template": self.template,
            "variables": self.variables,
            "example": self.example,
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentTemplate":
        """Crea desde diccionario"""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
    
    def render(self, **kwargs) -> str:
        """Renderiza template con variables"""
        result = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result


class TemplateService:
    """Servicio de gestión de templates"""
    
    def __init__(self):
        self.settings = get_settings()
        self.templates_dir = Path(self.settings.storage_path) / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates: Dict[str, ContentTemplate] = {}
        self._load_templates()
        self._create_default_templates()
    
    def _load_templates(self):
        """Carga templates desde disco"""
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    template = ContentTemplate.from_dict(data)
                    self.templates[template.template_id] = template
            except Exception as e:
                logger.error(f"Error cargando template {template_file}: {e}")
    
    def _create_default_templates(self):
        """Crea templates por defecto si no existen"""
        if self.templates:
            return
        
        default_templates = [
            ContentTemplate(
                template_id=str(uuid.uuid4()),
                name="Instagram Motivational Post",
                description="Post motivacional para Instagram",
                platform="instagram",
                content_type="post",
                template="💪 {message}\n\n{hashtags}",
                variables=["message", "hashtags"],
                example="💪 Nunca te rindas\n\n#motivation #success #inspiration",
                tags=["motivational", "instagram", "post"]
            ),
            ContentTemplate(
                template_id=str(uuid.uuid4()),
                name="TikTok Hook Script",
                description="Script con hook para TikTok",
                platform="tiktok",
                content_type="video",
                template="¿Sabías que {hook}?\n\n{content}\n\n{call_to_action}",
                variables=["hook", "content", "call_to_action"],
                example="¿Sabías que el 90% de las personas no logran sus metas?\n\nAquí te explico por qué...\n\n¡Sigue para más tips!",
                tags=["tiktok", "hook", "script"]
            ),
            ContentTemplate(
                template_id=str(uuid.uuid4()),
                name="YouTube Description",
                description="Descripción completa para YouTube",
                platform="youtube",
                content_type="video",
                template="""{title}

{description}

📌 En este video:
{points}

🔔 Suscríbete para más contenido como este
👍 Si te gustó, dale like
💬 Déjame saber en los comentarios qué opinas

{tags}

#youtube #content""",
                variables=["title", "description", "points", "tags"],
                tags=["youtube", "description"]
            )
        ]
        
        for template in default_templates:
            self.create_template(template)
    
    def create_template(self, template: ContentTemplate) -> str:
        """Crea un nuevo template"""
        if not template.template_id:
            template.template_id = str(uuid.uuid4())
        
        self.templates[template.template_id] = template
        self._save_template(template)
        
        logger.info(f"Template creado: {template.name} ({template.template_id})")
        return template.template_id
    
    def get_template(self, template_id: str) -> Optional[ContentTemplate]:
        """Obtiene un template"""
        return self.templates.get(template_id)
    
    def list_templates(
        self,
        platform: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ContentTemplate]:
        """Lista templates con filtros"""
        results = list(self.templates.values())
        
        if platform:
            results = [t for t in results if t.platform == platform]
        
        if content_type:
            results = [t for t in results if t.content_type == content_type]
        
        if tags:
            results = [
                t for t in results
                if any(tag in t.tags for tag in tags)
            ]
        
        return results
    
    def delete_template(self, template_id: str) -> bool:
        """Elimina un template"""
        template = self.templates.pop(template_id, None)
        if template:
            template_file = self.templates_dir / f"{template_id}.json"
            if template_file.exists():
                template_file.unlink()
            logger.info(f"Template eliminado: {template_id}")
            return True
        return False
    
    def _save_template(self, template: ContentTemplate):
        """Guarda template en disco"""
        try:
            template_file = self.templates_dir / f"{template.template_id}.json"
            with open(template_file, "w", encoding="utf-8") as f:
                json.dump(template.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error guardando template {template.template_id}: {e}")


# Singleton global
_template_service: Optional[TemplateService] = None


def get_template_service() -> TemplateService:
    """Obtiene instancia singleton del servicio de templates"""
    global _template_service
    if _template_service is None:
        _template_service = TemplateService()
    return _template_service




