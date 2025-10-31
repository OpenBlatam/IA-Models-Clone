"""
Configuration management for the Export IA system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from .models import ExportFormat, DocumentType, QualityLevel


@dataclass
class SystemConfig:
    """System-wide configuration."""
    output_directory: str = "exports"
    temp_directory: str = "temp"
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # seconds
    cleanup_temp_files: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class QualityConfig:
    """Quality configuration for different levels."""
    font_family: str
    font_size: int
    line_spacing: float
    margins: Dict[str, float]
    colors: Dict[str, str]
    header_footer: bool = False
    page_numbers: bool = False
    table_styling: bool = False
    custom_branding: bool = False
    advanced_formatting: bool = False
    interactive_elements: bool = False
    accessibility_features: bool = False


class ConfigManager:
    """Manages configuration for the Export IA system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.system_config = SystemConfig()
        self.quality_configs: Dict[QualityLevel, QualityConfig] = {}
        self.templates: Dict[DocumentType, Dict[str, Any]] = {}
        self.format_features: Dict[ExportFormat, list] = {}
        
        self._load_config()
        self._initialize_default_configs()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return os.path.join(os.path.dirname(__file__), "..", "..", "config", "export_config.yaml")
    
    def _load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    self._parse_config(config_data)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_path}: {e}")
                print("Using default configuration.")
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """Parse configuration data."""
        # System configuration
        if 'system' in config_data:
            system_data = config_data['system']
            self.system_config = SystemConfig(
                output_directory=system_data.get('output_directory', 'exports'),
                temp_directory=system_data.get('temp_directory', 'temp'),
                max_concurrent_tasks=system_data.get('max_concurrent_tasks', 10),
                task_timeout=system_data.get('task_timeout', 300),
                cleanup_temp_files=system_data.get('cleanup_temp_files', True),
                enable_logging=system_data.get('enable_logging', True),
                log_level=system_data.get('log_level', 'INFO')
            )
        
        # Quality configurations
        if 'quality_levels' in config_data:
            for level_name, level_config in config_data['quality_levels'].items():
                try:
                    level = QualityLevel(level_name)
                    self.quality_configs[level] = QualityConfig(**level_config)
                except ValueError:
                    print(f"Warning: Unknown quality level: {level_name}")
        
        # Templates
        if 'templates' in config_data:
            for template_name, template_config in config_data['templates'].items():
                try:
                    doc_type = DocumentType(template_name)
                    self.templates[doc_type] = template_config
                except ValueError:
                    print(f"Warning: Unknown document type: {template_name}")
        
        # Format features
        if 'format_features' in config_data:
            for format_name, features in config_data['format_features'].items():
                try:
                    fmt = ExportFormat(format_name)
                    self.format_features[fmt] = features
                except ValueError:
                    print(f"Warning: Unknown export format: {format_name}")
    
    def _initialize_default_configs(self):
        """Initialize default configurations if not loaded from file."""
        if not self.quality_configs:
            self._initialize_default_quality_configs()
        
        if not self.templates:
            self._initialize_default_templates()
        
        if not self.format_features:
            self._initialize_default_format_features()
    
    def _initialize_default_quality_configs(self):
        """Initialize default quality configurations."""
        self.quality_configs = {
            QualityLevel.BASIC: QualityConfig(
                font_family="Arial",
                font_size=11,
                line_spacing=1.0,
                margins={"top": 1, "bottom": 1, "left": 1, "right": 1},
                colors={"primary": "#000000", "secondary": "#666666"}
            ),
            QualityLevel.STANDARD: QualityConfig(
                font_family="Calibri",
                font_size=11,
                line_spacing=1.15,
                margins={"top": 1, "bottom": 1, "left": 1, "right": 1},
                colors={"primary": "#2E2E2E", "secondary": "#5A5A5A", "accent": "#1F4E79"}
            ),
            QualityLevel.PROFESSIONAL: QualityConfig(
                font_family="Calibri",
                font_size=11,
                line_spacing=1.15,
                margins={"top": 1, "bottom": 1, "left": 1, "right": 1},
                colors={"primary": "#2E2E2E", "secondary": "#5A5A5A", "accent": "#1F4E79", "highlight": "#F2F2F2"},
                header_footer=True,
                page_numbers=True,
                table_styling=True
            ),
            QualityLevel.PREMIUM: QualityConfig(
                font_family="Calibri",
                font_size=11,
                line_spacing=1.15,
                margins={"top": 1, "bottom": 1, "left": 1, "right": 1},
                colors={"primary": "#2E2E2E", "secondary": "#5A5A5A", "accent": "#1F4E79", "highlight": "#F2F2F2"},
                header_footer=True,
                page_numbers=True,
                table_styling=True,
                custom_branding=True,
                advanced_formatting=True
            ),
            QualityLevel.ENTERPRISE: QualityConfig(
                font_family="Calibri",
                font_size=11,
                line_spacing=1.15,
                margins={"top": 1, "bottom": 1, "left": 1, "right": 1},
                colors={"primary": "#2E2E2E", "secondary": "#5A5A5A", "accent": "#1F4E79", "highlight": "#F2F2F2"},
                header_footer=True,
                page_numbers=True,
                table_styling=True,
                custom_branding=True,
                advanced_formatting=True,
                interactive_elements=True,
                accessibility_features=True
            )
        }
    
    def _initialize_default_templates(self):
        """Initialize default document templates."""
        self.templates = {
            DocumentType.BUSINESS_PLAN: {
                "title_style": "Title",
                "heading_styles": ["Heading1", "Heading2", "Heading3"],
                "body_style": "Normal",
                "sections": [
                    "Executive Summary",
                    "Company Description",
                    "Market Analysis",
                    "Organization & Management",
                    "Service or Product Line",
                    "Marketing & Sales",
                    "Funding Request",
                    "Financial Projections",
                    "Appendix"
                ]
            },
            DocumentType.REPORT: {
                "title_style": "Title",
                "heading_styles": ["Heading1", "Heading2", "Heading3"],
                "body_style": "Normal",
                "sections": [
                    "Introduction",
                    "Methodology",
                    "Findings",
                    "Analysis",
                    "Conclusions",
                    "Recommendations",
                    "References"
                ]
            },
            DocumentType.PROPOSAL: {
                "title_style": "Title",
                "heading_styles": ["Heading1", "Heading2", "Heading3"],
                "body_style": "Normal",
                "sections": [
                    "Executive Summary",
                    "Problem Statement",
                    "Proposed Solution",
                    "Implementation Plan",
                    "Budget",
                    "Timeline",
                    "Team Qualifications",
                    "Next Steps"
                ]
            }
        }
    
    def _initialize_default_format_features(self):
        """Initialize default format features."""
        self.format_features = {
            ExportFormat.PDF: ["High quality", "Print ready", "Professional layout", "Vector graphics"],
            ExportFormat.DOCX: ["Editable", "Professional formatting", "Table support", "Image embedding"],
            ExportFormat.HTML: ["Web ready", "Responsive", "Interactive elements", "SEO friendly"],
            ExportFormat.MARKDOWN: ["Version control friendly", "Lightweight", "Platform agnostic", "Easy to edit"],
            ExportFormat.RTF: ["Cross platform", "Rich formatting", "Legacy support"],
            ExportFormat.TXT: ["Universal compatibility", "Lightweight", "Fast processing"],
            ExportFormat.JSON: ["Structured data", "API friendly", "Machine readable"],
            ExportFormat.XML: ["Structured data", "Validation support", "Industry standard"]
        }
    
    def get_quality_config(self, level: QualityLevel) -> QualityConfig:
        """Get quality configuration for a specific level."""
        return self.quality_configs.get(level, self.quality_configs[QualityLevel.PROFESSIONAL])
    
    def get_template(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Get template for a specific document type."""
        return self.templates.get(doc_type, {})
    
    def get_format_features(self, format_type: ExportFormat) -> list:
        """Get features for a specific export format."""
        return self.format_features.get(format_type, [])
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = file_path or self.config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config_data = {
            'system': {
                'output_directory': self.system_config.output_directory,
                'temp_directory': self.system_config.temp_directory,
                'max_concurrent_tasks': self.system_config.max_concurrent_tasks,
                'task_timeout': self.system_config.task_timeout,
                'cleanup_temp_files': self.system_config.cleanup_temp_files,
                'enable_logging': self.system_config.enable_logging,
                'log_level': self.system_config.log_level
            },
            'quality_levels': {
                level.value: {
                    'font_family': config.font_family,
                    'font_size': config.font_size,
                    'line_spacing': config.line_spacing,
                    'margins': config.margins,
                    'colors': config.colors,
                    'header_footer': config.header_footer,
                    'page_numbers': config.page_numbers,
                    'table_styling': config.table_styling,
                    'custom_branding': config.custom_branding,
                    'advanced_formatting': config.advanced_formatting,
                    'interactive_elements': config.interactive_elements,
                    'accessibility_features': config.accessibility_features
                }
                for level, config in self.quality_configs.items()
            },
            'templates': {
                doc_type.value: template
                for doc_type, template in self.templates.items()
            },
            'format_features': {
                fmt.value: features
                for fmt, features in self.format_features.items()
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)




