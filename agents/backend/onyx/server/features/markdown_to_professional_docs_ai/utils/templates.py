"""Templates for document styling"""
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os


class TemplateManager:
    """Manage document templates"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize template manager
        
        Args:
            templates_dir: Directory containing templates
        """
        if templates_dir is None:
            templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
        
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default templates"""
        # Default professional template
        self._templates["professional"] = {
            "name": "Professional",
            "colors": {
                "primary": "#366092",
                "secondary": "#2c4a6b",
                "accent": "#4a90e2",
                "text": "#1a1a1a",
                "background": "#ffffff",
                "header_bg": "#366092",
                "header_text": "#ffffff"
            },
            "fonts": {
                "heading": "Arial, sans-serif",
                "body": "Calibri, sans-serif",
                "monospace": "Courier New, monospace"
            },
            "spacing": {
                "paragraph": 12,
                "heading": 20,
                "section": 30
            },
            "table": {
                "header_bg": "#366092",
                "header_text": "#ffffff",
                "row_alt": "#f2f2f2",
                "border": "#dddddd"
            }
        }
        
        # Modern template
        self._templates["modern"] = {
            "name": "Modern",
            "colors": {
                "primary": "#6366f1",
                "secondary": "#4f46e5",
                "accent": "#818cf8",
                "text": "#1f2937",
                "background": "#ffffff",
                "header_bg": "#6366f1",
                "header_text": "#ffffff"
            },
            "fonts": {
                "heading": "Inter, system-ui, sans-serif",
                "body": "Inter, system-ui, sans-serif",
                "monospace": "Fira Code, monospace"
            },
            "spacing": {
                "paragraph": 14,
                "heading": 24,
                "section": 40
            },
            "table": {
                "header_bg": "#6366f1",
                "header_text": "#ffffff",
                "row_alt": "#f3f4f6",
                "border": "#e5e7eb"
            }
        }
        
        # Classic template
        self._templates["classic"] = {
            "name": "Classic",
            "colors": {
                "primary": "#000000",
                "secondary": "#333333",
                "accent": "#666666",
                "text": "#000000",
                "background": "#ffffff",
                "header_bg": "#000000",
                "header_text": "#ffffff"
            },
            "fonts": {
                "heading": "Times New Roman, serif",
                "body": "Times New Roman, serif",
                "monospace": "Courier New, monospace"
            },
            "spacing": {
                "paragraph": 10,
                "heading": 18,
                "section": 25
            },
            "table": {
                "header_bg": "#000000",
                "header_text": "#ffffff",
                "row_alt": "#f9f9f9",
                "border": "#cccccc"
            }
        }
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get template by name
        
        Args:
            template_name: Name of template
            
        Returns:
            Template dictionary
        """
        return self._templates.get(template_name, self._templates["professional"])
    
    def list_templates(self) -> list:
        """List all available templates"""
        return list(self._templates.keys())
    
    def save_template(self, name: str, template: Dict[str, Any]) -> None:
        """
        Save custom template
        
        Args:
            name: Template name
            template: Template dictionary
        """
        self._templates[name] = template
        
        # Save to file
        template_file = self.templates_dir / f"{name}.json"
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
    
    def load_template_from_file(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load template from file
        
        Args:
            name: Template name
            
        Returns:
            Template dictionary or None
        """
        template_file = self.templates_dir / f"{name}.json"
        
        if template_file.exists():
            with open(template_file, 'r') as f:
                template = json.load(f)
                self._templates[name] = template
                return template
        
        return None
    
    def merge_template(self, base_template: str, custom_styling: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base template with custom styling
        
        Args:
            base_template: Base template name
            custom_styling: Custom styling overrides
            
        Returns:
            Merged template
        """
        template = self.get_template(base_template).copy()
        
        # Deep merge
        def deep_merge(base: dict, override: dict) -> dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(template, custom_styling)


# Global template manager
_template_manager: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Get global template manager"""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager

