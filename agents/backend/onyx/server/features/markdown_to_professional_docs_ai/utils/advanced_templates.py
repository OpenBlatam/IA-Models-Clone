"""Advanced template system"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class AdvancedTemplateManager:
    """Advanced template management system"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize template manager
        
        Args:
            templates_dir: Directory for templates
        """
        if templates_dir is None:
            from config import settings
            templates_dir = settings.temp_dir + "/templates"
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load built-in templates
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in templates"""
        self.templates = {
            "professional": {
                "name": "Professional",
                "description": "Clean, professional document style",
                "styles": {
                    "font_family": "Arial, sans-serif",
                    "font_size": "12pt",
                    "heading_color": "#2c3e50",
                    "text_color": "#34495e",
                    "accent_color": "#3498db",
                    "background_color": "#ffffff",
                    "border_color": "#e0e0e0"
                },
                "layout": {
                    "header": True,
                    "footer": True,
                    "page_numbers": True,
                    "table_of_contents": True
                }
            },
            "modern": {
                "name": "Modern",
                "description": "Modern, minimalist design",
                "styles": {
                    "font_family": "Helvetica, sans-serif",
                    "font_size": "11pt",
                    "heading_color": "#1a1a1a",
                    "text_color": "#333333",
                    "accent_color": "#667eea",
                    "background_color": "#f8f9fa",
                    "border_color": "#dee2e6"
                },
                "layout": {
                    "header": True,
                    "footer": True,
                    "page_numbers": True,
                    "table_of_contents": False
                }
            },
            "corporate": {
                "name": "Corporate",
                "description": "Formal corporate document style",
                "styles": {
                    "font_family": "Times New Roman, serif",
                    "font_size": "12pt",
                    "heading_color": "#000000",
                    "text_color": "#000000",
                    "accent_color": "#1a1a1a",
                    "background_color": "#ffffff",
                    "border_color": "#000000"
                },
                "layout": {
                    "header": True,
                    "footer": True,
                    "page_numbers": True,
                    "table_of_contents": True
                }
            }
        }
    
    def create_template(
        self,
        template_name: str,
        template_config: Dict[str, Any]
    ) -> bool:
        """
        Create a new template
        
        Args:
            template_name: Template name
            template_config: Template configuration
            
        Returns:
            True if successful
        """
        try:
            template_file = self.templates_dir / f"{template_name}.json"
            with open(template_file, 'w') as f:
                json.dump(template_config, f, indent=2)
            
            self.templates[template_name] = template_config
            return True
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            return False
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get template configuration
        
        Args:
            template_name: Template name
            
        Returns:
            Template configuration or None
        """
        # Check built-in templates
        if template_name in self.templates:
            return self.templates[template_name]
        
        # Check custom templates
        template_file = self.templates_dir / f"{template_name}.json"
        if template_file.exists():
            try:
                with open(template_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading template: {e}")
        
        return None
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates
        
        Returns:
            List of template info
        """
        templates = []
        
        # Built-in templates
        for name, config in self.templates.items():
            templates.append({
                "name": name,
                "description": config.get("description", ""),
                "type": "builtin"
            })
        
        # Custom templates
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    config = json.load(f)
                    templates.append({
                        "name": template_file.stem,
                        "description": config.get("description", ""),
                        "type": "custom"
                    })
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def delete_template(self, template_name: str) -> bool:
        """
        Delete a template
        
        Args:
            template_name: Template name
            
        Returns:
            True if successful
        """
        # Cannot delete built-in templates
        if template_name in self.templates:
            return False
        
        template_file = self.templates_dir / f"{template_name}.json"
        if template_file.exists():
            try:
                template_file.unlink()
                return True
            except Exception as e:
                logger.error(f"Error deleting template: {e}")
                return False
        
        return False
    
    def apply_template(
        self,
        template_name: str,
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply template to content
        
        Args:
            template_name: Template name
            content: Content to apply template to
            
        Returns:
            Content with template applied
        """
        template = self.get_template(template_name)
        if not template:
            return content
        
        # Apply styles
        if "styles" in template:
            content["styles"] = template["styles"]
        
        # Apply layout
        if "layout" in template:
            content["layout"] = template["layout"]
        
        return content
    
    def generate_css(self, template_name: str) -> str:
        """
        Generate CSS from template
        
        Args:
            template_name: Template name
            
        Returns:
            CSS string
        """
        template = self.get_template(template_name)
        if not template or "styles" not in template:
            return ""
        
        styles = template["styles"]
        css = f"""
        body {{
            font-family: {styles.get('font_family', 'Arial, sans-serif')};
            font-size: {styles.get('font_size', '12pt')};
            color: {styles.get('text_color', '#333333')};
            background-color: {styles.get('background_color', '#ffffff')};
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {styles.get('heading_color', '#2c3e50')};
        }}
        
        .accent {{
            color: {styles.get('accent_color', '#3498db')};
        }}
        
        table, .table {{
            border-color: {styles.get('border_color', '#e0e0e0')};
        }}
        """
        
        return css


# Global template manager
_template_manager: Optional[AdvancedTemplateManager] = None


def get_advanced_template_manager() -> AdvancedTemplateManager:
    """Get global template manager"""
    global _template_manager
    if _template_manager is None:
        _template_manager = AdvancedTemplateManager()
    return _template_manager

