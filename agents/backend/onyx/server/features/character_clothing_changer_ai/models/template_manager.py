"""
Template Manager for Flux2 Clothing Changer
===========================================

Advanced template management and rendering system.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """Template information."""
    template_id: str
    name: str
    content: str
    variables: List[str]
    created_at: float = time.time()
    updated_at: float = time.time()
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TemplateManager:
    """Advanced template management system."""
    
    def __init__(self):
        """Initialize template manager."""
        self.templates: Dict[str, Template] = {}
        self.renderers: Dict[str, Callable] = {}
    
    def register_renderer(
        self,
        template_type: str,
        renderer: Callable[[str, Dict[str, Any]], str],
    ) -> None:
        """
        Register template renderer.
        
        Args:
            template_type: Template type
            renderer: Renderer function
        """
        self.renderers[template_type] = renderer
        logger.info(f"Registered renderer for type: {template_type}")
    
    def create_template(
        self,
        template_id: str,
        name: str,
        content: str,
        template_type: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Template:
        """
        Create template.
        
        Args:
            template_id: Template identifier
            name: Template name
            content: Template content
            template_type: Template type
            metadata: Optional metadata
            
        Returns:
            Created template
        """
        # Extract variables from content (simple extraction)
        variables = self._extract_variables(content)
        
        template = Template(
            template_id=template_id,
            name=name,
            content=content,
            variables=variables,
            metadata=metadata or {"type": template_type},
        )
        
        self.templates[template_id] = template
        logger.info(f"Created template: {template_id}")
        return template
    
    def render(
        self,
        template_id: str,
        variables: Dict[str, Any],
    ) -> str:
        """
        Render template.
        
        Args:
            template_id: Template identifier
            variables: Template variables
            
        Returns:
            Rendered content
        """
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self.templates[template_id]
        template_type = template.metadata.get("type", "default")
        
        # Use custom renderer if available
        if template_type in self.renderers:
            renderer = self.renderers[template_type]
            return renderer(template.content, variables)
        
        # Default simple string replacement
        result = template.content
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
            result = result.replace(f"{{{{ {key} }}}}", str(value))
        
        return result
    
    def _extract_variables(self, content: str) -> List[str]:
        """Extract variables from template content."""
        import re
        # Simple extraction - find {{variable}} patterns
        pattern = r'\{\{(\w+)\}\}'
        variables = re.findall(pattern, content)
        return list(set(variables))
    
    def update_template(
        self,
        template_id: str,
        content: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update template.
        
        Args:
            template_id: Template identifier
            content: Optional new content
            name: Optional new name
            metadata: Optional metadata updates
            
        Returns:
            True if updated
        """
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        
        if content:
            template.content = content
            template.variables = self._extract_variables(content)
        
        if name:
            template.name = name
        
        if metadata:
            template.metadata.update(metadata)
        
        template.updated_at = time.time()
        logger.info(f"Updated template: {template_id}")
        return True
    
    def get_template(self, template_id: str) -> Optional[Template]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[Template]:
        """List all templates."""
        return list(self.templates.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get template manager statistics."""
        return {
            "total_templates": len(self.templates),
            "renderers_registered": len(self.renderers),
        }


