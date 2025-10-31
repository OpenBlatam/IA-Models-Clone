"""
Export IA Configuration System
==============================

Configuration management for professional document export settings,
templates, and quality standards.
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Types of document templates."""
    BUSINESS = "business"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"

class BrandingStyle(Enum):
    """Branding style options."""
    MINIMAL = "minimal"
    MODERN = "modern"
    CLASSIC = "classic"
    CORPORATE = "corporate"
    CREATIVE = "creative"
    TECHNICAL = "technical"

@dataclass
class FontConfig:
    """Font configuration for professional documents."""
    primary_font: str = "Calibri"
    secondary_font: str = "Arial"
    heading_font: str = "Calibri"
    monospace_font: str = "Courier New"
    font_sizes: Dict[str, int] = field(default_factory=lambda: {
        "title": 18,
        "heading1": 16,
        "heading2": 14,
        "heading3": 12,
        "body": 11,
        "caption": 9
    })

@dataclass
class ColorScheme:
    """Professional color scheme configuration."""
    primary: str = "#2E2E2E"
    secondary: str = "#5A5A5A"
    accent: str = "#1F4E79"
    background: str = "#FFFFFF"
    highlight: str = "#F2F2F2"
    success: str = "#28A745"
    warning: str = "#FFC107"
    error: str = "#DC3545"
    info: str = "#17A2B8"

@dataclass
class LayoutConfig:
    """Layout configuration for professional documents."""
    page_size: str = "A4"
    orientation: str = "portrait"
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 1.0,
        "bottom": 1.0,
        "left": 1.0,
        "right": 1.0
    })
    line_spacing: float = 1.15
    paragraph_spacing: float = 6.0
    header_height: float = 0.5
    footer_height: float = 0.5

@dataclass
class BrandingConfig:
    """Branding configuration for professional documents."""
    company_name: str = ""
    logo_path: Optional[str] = None
    website: str = ""
    email: str = ""
    phone: str = ""
    address: str = ""
    style: BrandingStyle = BrandingStyle.MODERN
    custom_colors: Optional[ColorScheme] = None
    custom_fonts: Optional[FontConfig] = None

@dataclass
class QualityStandards:
    """Quality standards for professional document export."""
    min_quality_score: float = 0.7
    required_features: List[str] = field(default_factory=lambda: [
        "professional_typography",
        "consistent_formatting",
        "proper_spacing"
    ])
    accessibility_requirements: List[str] = field(default_factory=lambda: [
        "alt_text_for_images",
        "proper_heading_structure",
        "sufficient_color_contrast"
    ])
    validation_rules: Dict[str, Any] = field(default_factory=lambda: {
        "min_sections": 1,
        "max_line_length": 80,
        "min_paragraph_length": 20
    })

@dataclass
class ExportTemplate:
    """Professional document template configuration."""
    id: str
    name: str
    description: str
    template_type: TemplateType
    document_type: str
    sections: List[Dict[str, Any]]
    styling: Dict[str, Any]
    branding: Optional[BrandingConfig] = None
    quality_standards: Optional[QualityStandards] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExportPreferences:
    """User export preferences."""
    default_format: str = "pdf"
    default_quality: str = "professional"
    auto_optimize: bool = True
    include_metadata: bool = True
    compress_output: bool = False
    watermark: bool = False
    page_numbers: bool = True
    headers_footers: bool = True
    table_of_contents: bool = False
    bibliography: bool = False

class ExportConfigManager:
    """Manages export configurations and templates."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config/export_ia")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        self.default_fonts = FontConfig()
        self.default_colors = ColorScheme()
        self.default_layout = LayoutConfig()
        self.default_quality = QualityStandards()
        self.default_preferences = ExportPreferences()
        
        # Load configurations
        self.templates: Dict[str, ExportTemplate] = {}
        self.branding_configs: Dict[str, BrandingConfig] = {}
        
        self._load_default_templates()
        self._load_configurations()
        
        logger.info(f"Export Config Manager initialized with {len(self.templates)} templates")
    
    def _load_default_templates(self):
        """Load default professional templates."""
        default_templates = [
            {
                "id": "business_plan",
                "name": "Business Plan Template",
                "description": "Professional business plan template with all standard sections",
                "template_type": TemplateType.BUSINESS,
                "document_type": "business_plan",
                "sections": [
                    {"id": "executive_summary", "title": "Executive Summary", "required": True},
                    {"id": "company_description", "title": "Company Description", "required": True},
                    {"id": "market_analysis", "title": "Market Analysis", "required": True},
                    {"id": "organization", "title": "Organization & Management", "required": True},
                    {"id": "service_product", "title": "Service or Product Line", "required": True},
                    {"id": "marketing_sales", "title": "Marketing & Sales", "required": True},
                    {"id": "funding_request", "title": "Funding Request", "required": False},
                    {"id": "financial_projections", "title": "Financial Projections", "required": True},
                    {"id": "appendix", "title": "Appendix", "required": False}
                ],
                "styling": {
                    "font_family": "Calibri",
                    "font_size": 11,
                    "line_spacing": 1.15,
                    "colors": {
                        "primary": "#2E2E2E",
                        "accent": "#1F4E79"
                    }
                }
            },
            {
                "id": "professional_report",
                "name": "Professional Report Template",
                "description": "Comprehensive report template for business and technical documents",
                "template_type": TemplateType.TECHNICAL,
                "document_type": "report",
                "sections": [
                    {"id": "introduction", "title": "Introduction", "required": True},
                    {"id": "methodology", "title": "Methodology", "required": True},
                    {"id": "findings", "title": "Findings", "required": True},
                    {"id": "analysis", "title": "Analysis", "required": True},
                    {"id": "conclusions", "title": "Conclusions", "required": True},
                    {"id": "recommendations", "title": "Recommendations", "required": True},
                    {"id": "references", "title": "References", "required": False}
                ],
                "styling": {
                    "font_family": "Calibri",
                    "font_size": 11,
                    "line_spacing": 1.15,
                    "colors": {
                        "primary": "#2E2E2E",
                        "accent": "#1F4E79"
                    }
                }
            },
            {
                "id": "proposal_template",
                "name": "Professional Proposal Template",
                "description": "Comprehensive proposal template for business proposals",
                "template_type": TemplateType.BUSINESS,
                "document_type": "proposal",
                "sections": [
                    {"id": "executive_summary", "title": "Executive Summary", "required": True},
                    {"id": "problem_statement", "title": "Problem Statement", "required": True},
                    {"id": "proposed_solution", "title": "Proposed Solution", "required": True},
                    {"id": "implementation_plan", "title": "Implementation Plan", "required": True},
                    {"id": "budget", "title": "Budget", "required": True},
                    {"id": "timeline", "title": "Timeline", "required": True},
                    {"id": "team_qualifications", "title": "Team Qualifications", "required": True},
                    {"id": "next_steps", "title": "Next Steps", "required": True}
                ],
                "styling": {
                    "font_family": "Calibri",
                    "font_size": 11,
                    "line_spacing": 1.15,
                    "colors": {
                        "primary": "#2E2E2E",
                        "accent": "#1F4E79"
                    }
                }
            }
        ]
        
        for template_data in default_templates:
            template = ExportTemplate(**template_data)
            self.templates[template.id] = template
    
    def _load_configurations(self):
        """Load configurations from files."""
        # Load branding configurations
        branding_file = self.config_dir / "branding.yaml"
        if branding_file.exists():
            try:
                with open(branding_file, 'r') as f:
                    branding_data = yaml.safe_load(f)
                    for name, config_data in branding_data.items():
                        self.branding_configs[name] = BrandingConfig(**config_data)
            except Exception as e:
                logger.error(f"Failed to load branding configurations: {e}")
        
        # Load custom templates
        templates_file = self.config_dir / "templates.yaml"
        if templates_file.exists():
            try:
                with open(templates_file, 'r') as f:
                    templates_data = yaml.safe_load(f)
                    for template_data in templates_data:
                        template = ExportTemplate(**template_data)
                        self.templates[template.id] = template
            except Exception as e:
                logger.error(f"Failed to load custom templates: {e}")
    
    def save_configuration(self, config_type: str, name: str, config: Any):
        """Save a configuration to file."""
        config_file = self.config_dir / f"{config_type}.yaml"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    existing_data = yaml.safe_load(f) or {}
            else:
                existing_data = {}
            
            existing_data[name] = asdict(config) if hasattr(config, '__dataclass_fields__') else config
            
            with open(config_file, 'w') as f:
                yaml.dump(existing_data, f, default_flow_style=False)
            
            logger.info(f"Configuration saved: {config_type}/{name}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration {config_type}/{name}: {e}")
    
    def get_template(self, template_id: str) -> Optional[ExportTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, template_type: Optional[TemplateType] = None) -> List[ExportTemplate]:
        """List available templates, optionally filtered by type."""
        templates = list(self.templates.values())
        
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
        
        return sorted(templates, key=lambda x: x.name)
    
    def create_custom_template(
        self,
        template_id: str,
        name: str,
        description: str,
        template_type: TemplateType,
        document_type: str,
        sections: List[Dict[str, Any]],
        styling: Dict[str, Any],
        branding: Optional[BrandingConfig] = None
    ) -> ExportTemplate:
        """Create a custom template."""
        template = ExportTemplate(
            id=template_id,
            name=name,
            description=description,
            template_type=template_type,
            document_type=document_type,
            sections=sections,
            styling=styling,
            branding=branding
        )
        
        self.templates[template_id] = template
        self.save_configuration("templates", template_id, template)
        
        logger.info(f"Custom template created: {template_id}")
        return template
    
    def get_branding_config(self, name: str) -> Optional[BrandingConfig]:
        """Get a branding configuration by name."""
        return self.branding_configs.get(name)
    
    def create_branding_config(
        self,
        name: str,
        company_name: str,
        style: BrandingStyle = BrandingStyle.MODERN,
        **kwargs
    ) -> BrandingConfig:
        """Create a branding configuration."""
        config = BrandingConfig(
            company_name=company_name,
            style=style,
            **kwargs
        )
        
        self.branding_configs[name] = config
        self.save_configuration("branding", name, config)
        
        logger.info(f"Branding configuration created: {name}")
        return config
    
    def validate_export_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an export configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check required fields
        required_fields = ["format", "document_type"]
        for field in required_fields:
            if field not in config:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Validate format
        if "format" in config:
            valid_formats = ["pdf", "docx", "html", "markdown", "rtf", "txt", "json", "xml"]
            if config["format"] not in valid_formats:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid format: {config['format']}")
        
        # Validate quality level
        if "quality_level" in config:
            valid_quality_levels = ["basic", "standard", "professional", "premium", "enterprise"]
            if config["quality_level"] not in valid_quality_levels:
                validation_result["warnings"].append(f"Unknown quality level: {config['quality_level']}")
        
        # Check template compatibility
        if "template" in config:
            template = self.get_template(config["template"])
            if not template:
                validation_result["warnings"].append(f"Template not found: {config['template']}")
            elif "document_type" in config and template.document_type != config["document_type"]:
                validation_result["suggestions"].append(
                    f"Template '{config['template']}' is designed for '{template.document_type}', "
                    f"but document type is '{config['document_type']}'"
                )
        
        return validation_result
    
    def get_export_recommendations(
        self,
        document_type: str,
        content_length: int,
        has_images: bool = False,
        has_tables: bool = False
    ) -> Dict[str, Any]:
        """Get export format recommendations based on content characteristics."""
        recommendations = {
            "primary_format": "pdf",
            "alternative_formats": [],
            "quality_level": "professional",
            "features": []
        }
        
        # Format recommendations based on content
        if has_images or has_tables:
            recommendations["primary_format"] = "pdf"
            recommendations["alternative_formats"] = ["docx", "html"]
            recommendations["features"].extend(["image_optimization", "table_formatting"])
        elif content_length > 10000:  # Long documents
            recommendations["primary_format"] = "pdf"
            recommendations["alternative_formats"] = ["docx"]
            recommendations["features"].extend(["page_numbers", "table_of_contents"])
        else:  # Short documents
            recommendations["primary_format"] = "docx"
            recommendations["alternative_formats"] = ["pdf", "html"]
        
        # Quality level recommendations
        if document_type in ["business_plan", "proposal", "contract"]:
            recommendations["quality_level"] = "premium"
        elif document_type in ["report", "manual"]:
            recommendations["quality_level"] = "professional"
        else:
            recommendations["quality_level"] = "standard"
        
        return recommendations
    
    def export_config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        if hasattr(config, '__dataclass_fields__'):
            return asdict(config)
        elif isinstance(config, dict):
            return config
        else:
            return {"error": "Invalid configuration object"}
    
    def import_config_from_dict(self, config_dict: Dict[str, Any], config_type: str) -> Any:
        """Import configuration from dictionary."""
        try:
            if config_type == "template":
                return ExportTemplate(**config_dict)
            elif config_type == "branding":
                return BrandingConfig(**config_dict)
            elif config_type == "fonts":
                return FontConfig(**config_dict)
            elif config_type == "colors":
                return ColorScheme(**config_dict)
            elif config_type == "layout":
                return LayoutConfig(**config_dict)
            elif config_type == "quality":
                return QualityStandards(**config_dict)
            elif config_type == "preferences":
                return ExportPreferences(**config_dict)
            else:
                raise ValueError(f"Unknown configuration type: {config_type}")
        except Exception as e:
            logger.error(f"Failed to import {config_type} configuration: {e}")
            return None

# Global configuration manager instance
_global_config_manager: Optional[ExportConfigManager] = None

def get_global_config_manager() -> ExportConfigManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ExportConfigManager()
    return _global_config_manager



























