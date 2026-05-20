"""
Custom Templates Service
Manages user-created templates
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CustomTemplate:
    """User-created template"""
    
    def __init__(
        self,
        template_id: str,
        user_id: str,
        name: str,
        description: str,
        config: Dict[str, Any],
        is_public: bool = False,
        created_at: Optional[datetime] = None
    ):
        self.template_id = template_id
        self.user_id = user_id
        self.name = name
        self.description = description
        self.config = config
        self.is_public = is_public
        self.created_at = created_at or datetime.utcnow()
        self.usage_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "template_id": self.template_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat(),
            "usage_count": self.usage_count,
        }


class CustomTemplateService:
    """Manages custom templates"""
    
    def __init__(self):
        # In-memory storage (use database in production)
        self.templates: Dict[str, CustomTemplate] = {}
        self.user_templates: Dict[str, List[str]] = {}  # user_id -> template_ids
    
    def create_template(
        self,
        user_id: str,
        name: str,
        description: str,
        config: Dict[str, Any],
        is_public: bool = False
    ) -> CustomTemplate:
        """
        Create custom template
        
        Args:
            user_id: User ID
            name: Template name
            description: Template description
            config: Template configuration
            is_public: Whether template is public
            
        Returns:
            Created template
        """
        template_id = f"custom_{user_id}_{len(self.templates) + 1}"
        
        template = CustomTemplate(
            template_id=template_id,
            user_id=user_id,
            name=name,
            description=description,
            config=config,
            is_public=is_public
        )
        
        self.templates[template_id] = template
        
        if user_id not in self.user_templates:
            self.user_templates[user_id] = []
        self.user_templates[user_id].append(template_id)
        
        logger.info(f"Created custom template: {template_id}")
        return template
    
    def get_template(self, template_id: str) -> Optional[CustomTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def get_user_templates(self, user_id: str) -> List[CustomTemplate]:
        """Get templates for user"""
        template_ids = self.user_templates.get(user_id, [])
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def get_public_templates(self) -> List[CustomTemplate]:
        """Get all public templates"""
        return [t for t in self.templates.values() if t.is_public]
    
    def update_template(
        self,
        template_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        is_public: Optional[bool] = None
    ) -> CustomTemplate:
        """Update template"""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError("Template not found")
        
        if template.user_id != user_id:
            raise ValueError("Not authorized to update this template")
        
        if name:
            template.name = name
        if description:
            template.description = description
        if config:
            template.config = config
        if is_public is not None:
            template.is_public = is_public
        
        logger.info(f"Updated template: {template_id}")
        return template
    
    def delete_template(self, template_id: str, user_id: str) -> bool:
        """Delete template"""
        template = self.templates.get(template_id)
        if not template:
            return False
        
        if template.user_id != user_id:
            raise ValueError("Not authorized to delete this template")
        
        del self.templates[template_id]
        
        if user_id in self.user_templates:
            self.user_templates[user_id].remove(template_id)
        
        logger.info(f"Deleted template: {template_id}")
        return True
    
    def increment_usage(self, template_id: str):
        """Increment template usage count"""
        template = self.templates.get(template_id)
        if template:
            template.usage_count += 1


_custom_template_service: Optional[CustomTemplateService] = None


def get_custom_template_service() -> CustomTemplateService:
    """Get custom template service instance (singleton)"""
    global _custom_template_service
    if _custom_template_service is None:
        _custom_template_service = CustomTemplateService()
    return _custom_template_service

