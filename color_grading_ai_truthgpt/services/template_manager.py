"""
Template Manager for Color Grading AI
======================================

Manages color grading templates and presets.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ColorGradingTemplate:
    """Color grading template data structure."""
    name: str
    description: str
    category: str  # cinematic, vintage, modern, etc.
    color_params: Dict[str, Any]
    preview_image: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorGradingTemplate":
        """Create template from dictionary."""
        return cls(**data)


class TemplateManager:
    """
    Manages color grading templates.
    
    Features:
    - Load and save templates
    - Search templates by category/tags
    - Apply templates
    - Create custom templates
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize template manager.
        
        Args:
            templates_dir: Directory to store templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, ColorGradingTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default color grading templates."""
        default_templates = [
            {
                "name": "Cinematic Warm",
                "description": "Warm cinematic look with enhanced contrast",
                "category": "cinematic",
                "color_params": {
                    "brightness": 0.05,
                    "contrast": 1.2,
                    "saturation": 1.1,
                    "color_balance": {"r": 0.1, "g": 0.0, "b": -0.1},
                },
                "tags": ["cinematic", "warm", "film"]
            },
            {
                "name": "Vintage Film",
                "description": "Vintage film look with desaturated colors",
                "category": "vintage",
                "color_params": {
                    "brightness": 0.0,
                    "contrast": 1.15,
                    "saturation": 0.7,
                    "color_balance": {"r": 0.05, "g": -0.05, "b": 0.0},
                },
                "tags": ["vintage", "film", "retro"]
            },
            {
                "name": "Cool Blue",
                "description": "Cool blue tone for moody scenes",
                "category": "modern",
                "color_params": {
                    "brightness": -0.1,
                    "contrast": 1.3,
                    "saturation": 0.9,
                    "color_balance": {"r": -0.15, "g": 0.0, "b": 0.2},
                },
                "tags": ["cool", "blue", "moody"]
            },
            {
                "name": "High Contrast B&W",
                "description": "High contrast black and white",
                "category": "monochrome",
                "color_params": {
                    "brightness": 0.0,
                    "contrast": 1.5,
                    "saturation": 0.0,
                    "color_balance": {"r": 0.0, "g": 0.0, "b": 0.0},
                },
                "tags": ["black", "white", "monochrome", "contrast"]
            },
            {
                "name": "Golden Hour",
                "description": "Warm golden hour lighting",
                "category": "natural",
                "color_params": {
                    "brightness": 0.1,
                    "contrast": 1.1,
                    "saturation": 1.2,
                    "color_balance": {"r": 0.2, "g": 0.1, "b": -0.1},
                },
                "tags": ["golden", "warm", "natural", "sunset"]
            },
            {
                "name": "Teal & Orange",
                "description": "Popular teal and orange cinematic look",
                "category": "cinematic",
                "color_params": {
                    "brightness": 0.0,
                    "contrast": 1.25,
                    "saturation": 1.15,
                    "color_balance": {"r": 0.15, "g": 0.0, "b": -0.2},
                },
                "tags": ["cinematic", "teal", "orange", "hollywood"]
            },
        ]
        
        for template_data in default_templates:
            template = ColorGradingTemplate.from_dict(template_data)
            self._templates[template.name] = template
        
        logger.info(f"Loaded {len(self._templates)} default templates")
    
    def get_template(self, name: str) -> Optional[ColorGradingTemplate]:
        """
        Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template or None if not found
        """
        return self._templates.get(name)
    
    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ColorGradingTemplate]:
        """
        List templates with optional filtering.
        
        Args:
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            List of matching templates
        """
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]
        
        return templates
    
    def save_template(self, template: ColorGradingTemplate) -> str:
        """
        Save template to file.
        
        Args:
            template: Template to save
            
        Returns:
            Path to saved template file
        """
        self._templates[template.name] = template
        
        template_file = self.templates_dir / f"{template.name.lower().replace(' ', '_')}.json"
        
        with open(template_file, "w") as f:
            json.dump(template.to_dict(), f, indent=2)
        
        logger.info(f"Saved template: {template.name}")
        return str(template_file)
    
    def load_template_from_file(self, template_path: str) -> ColorGradingTemplate:
        """
        Load template from file.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Loaded template
        """
        with open(template_path, "r") as f:
            data = json.load(f)
        
        template = ColorGradingTemplate.from_dict(data)
        self._templates[template.name] = template
        
        return template
    
    def create_template(
        self,
        name: str,
        description: str,
        category: str,
        color_params: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> ColorGradingTemplate:
        """
        Create a new template.
        
        Args:
            name: Template name
            description: Template description
            category: Template category
            color_params: Color grading parameters
            tags: Optional tags
            
        Returns:
            Created template
        """
        template = ColorGradingTemplate(
            name=name,
            description=description,
            category=category,
            color_params=color_params,
            tags=tags or []
        )
        
        self._templates[name] = template
        return template
    
    def get_template_color_params(self, template_name: str) -> Dict[str, Any]:
        """
        Get color parameters from template.
        
        Args:
            template_name: Template name
            
        Returns:
            Color grading parameters
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        return template.color_params




