"""
Professional Documents Configuration
===================================

Configuration settings for the Professional Documents feature.
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseSettings, Field
from pathlib import Path


class ProfessionalDocumentsConfig(BaseSettings):
    """Configuration for Professional Documents feature."""
    
    # Feature settings
    enabled: bool = Field(default=True, description="Enable professional documents feature")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    
    # AI Configuration
    ai_model_name: str = Field(default="gpt-4", description="AI model for content generation")
    ai_max_tokens: int = Field(default=4000, description="Maximum tokens for AI generation")
    ai_temperature: float = Field(default=0.7, description="AI temperature for creativity")
    ai_timeout: int = Field(default=60, description="AI request timeout in seconds")
    
    # Document generation settings
    max_document_length: int = Field(default=50000, description="Maximum document length in words")
    min_document_length: int = Field(default=100, description="Minimum document length in words")
    default_document_type: str = Field(default="report", description="Default document type")
    default_tone: str = Field(default="professional", description="Default document tone")
    default_language: str = Field(default="en", description="Default document language")
    
    # Export settings
    export_directory: str = Field(default="exports", description="Directory for exported files")
    max_export_file_size: int = Field(default=50 * 1024 * 1024, description="Maximum export file size in bytes")
    export_cleanup_hours: int = Field(default=24, description="Hours before cleaning up export files")
    supported_export_formats: List[str] = Field(
        default=["pdf", "docx", "md", "html"], 
        description="Supported export formats"
    )
    
    # Template settings
    custom_templates_enabled: bool = Field(default=True, description="Enable custom templates")
    template_cache_size: int = Field(default=100, description="Template cache size")
    template_validation_enabled: bool = Field(default=True, description="Enable template validation")
    
    # Styling settings
    default_font_family: str = Field(default="Arial", description="Default font family")
    default_font_size: int = Field(default=12, description="Default font size")
    default_line_spacing: float = Field(default=1.5, description="Default line spacing")
    default_margins: Dict[str, float] = Field(
        default={"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0},
        description="Default margins in inches"
    )
    
    # Security settings
    require_authentication: bool = Field(default=True, description="Require user authentication")
    max_documents_per_user: int = Field(default=100, description="Maximum documents per user")
    document_access_control: bool = Field(default=True, description="Enable document access control")
    
    # Performance settings
    async_generation: bool = Field(default=True, description="Use async document generation")
    background_export: bool = Field(default=True, description="Use background export processing")
    cache_generated_content: bool = Field(default=True, description="Cache generated content")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    log_generation_time: bool = Field(default=True, description="Log generation time")
    log_export_time: bool = Field(default=True, description="Log export time")
    log_user_activity: bool = Field(default=True, description="Log user activity")
    
    # Integration settings
    integrate_with_existing_api: bool = Field(default=True, description="Integrate with existing API")
    enable_webhook_notifications: bool = Field(default=False, description="Enable webhook notifications")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for notifications")
    
    # File handling settings
    allowed_file_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".md", ".html", ".txt"],
        description="Allowed file extensions for uploads"
    )
    max_upload_size: int = Field(default=10 * 1024 * 1024, description="Maximum upload size in bytes")
    temp_directory: str = Field(default="temp", description="Temporary directory for processing")
    
    # Rate limiting settings
    rate_limit_requests_per_minute: int = Field(default=10, description="Rate limit for requests per minute")
    rate_limit_documents_per_hour: int = Field(default=50, description="Rate limit for documents per hour")
    rate_limit_exports_per_hour: int = Field(default=100, description="Rate limit for exports per hour")
    
    class Config:
        env_prefix = "PROFESSIONAL_DOCUMENTS_"
        case_sensitive = False


class DocumentTypeConfig:
    """Configuration for specific document types."""
    
    # Document type specific settings
    DOCUMENT_TYPE_SETTINGS = {
        "report": {
            "default_sections": ["Executive Summary", "Introduction", "Findings", "Recommendations", "Conclusion"],
            "min_length": 500,
            "max_length": 10000,
            "default_tone": "professional"
        },
        "proposal": {
            "default_sections": ["Executive Summary", "Problem Statement", "Solution", "Timeline", "Budget"],
            "min_length": 300,
            "max_length": 8000,
            "default_tone": "professional"
        },
        "manual": {
            "default_sections": ["Getting Started", "Features", "Troubleshooting", "FAQ"],
            "min_length": 200,
            "max_length": 15000,
            "default_tone": "casual"
        },
        "technical_document": {
            "default_sections": ["Overview", "Architecture", "API Reference", "Examples"],
            "min_length": 400,
            "max_length": 20000,
            "default_tone": "technical"
        },
        "academic_paper": {
            "default_sections": ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Conclusion"],
            "min_length": 1000,
            "max_length": 25000,
            "default_tone": "academic"
        },
        "whitepaper": {
            "default_sections": ["Executive Summary", "Market Analysis", "Solution", "Case Studies", "Conclusion"],
            "min_length": 800,
            "max_length": 12000,
            "default_tone": "professional"
        },
        "business_plan": {
            "default_sections": ["Executive Summary", "Company Description", "Market Analysis", "Financial Projections"],
            "min_length": 1000,
            "max_length": 20000,
            "default_tone": "formal"
        }
    }
    
    @classmethod
    def get_document_type_config(cls, document_type: str) -> Dict[str, Any]:
        """Get configuration for a specific document type."""
        return cls.DOCUMENT_TYPE_SETTINGS.get(document_type, cls.DOCUMENT_TYPE_SETTINGS["report"])
    
    @classmethod
    def get_all_document_types(cls) -> List[str]:
        """Get all supported document types."""
        return list(cls.DOCUMENT_TYPE_SETTINGS.keys())


class ExportFormatConfig:
    """Configuration for export formats."""
    
    # Export format specific settings
    EXPORT_FORMAT_SETTINGS = {
        "pdf": {
            "mime_type": "application/pdf",
            "file_extension": ".pdf",
            "supports_styling": True,
            "supports_images": True,
            "max_file_size": 50 * 1024 * 1024,  # 50MB
            "default_page_size": "A4"
        },
        "docx": {
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "file_extension": ".docx",
            "supports_styling": True,
            "supports_images": True,
            "max_file_size": 25 * 1024 * 1024,  # 25MB
            "default_page_size": "Letter"
        },
        "md": {
            "mime_type": "text/markdown",
            "file_extension": ".md",
            "supports_styling": False,
            "supports_images": True,
            "max_file_size": 5 * 1024 * 1024,  # 5MB
            "default_page_size": None
        },
        "html": {
            "mime_type": "text/html",
            "file_extension": ".html",
            "supports_styling": True,
            "supports_images": True,
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "default_page_size": None
        }
    }
    
    @classmethod
    def get_export_format_config(cls, format_name: str) -> Dict[str, Any]:
        """Get configuration for a specific export format."""
        return cls.EXPORT_FORMAT_SETTINGS.get(format_name, cls.EXPORT_FORMAT_SETTINGS["pdf"])
    
    @classmethod
    def get_all_export_formats(cls) -> List[str]:
        """Get all supported export formats."""
        return list(cls.EXPORT_FORMAT_SETTINGS.keys())


class StylingConfig:
    """Configuration for document styling."""
    
    # Predefined color schemes
    COLOR_SCHEMES = {
        "professional": {
            "header_color": "#2c3e50",
            "body_color": "#34495e",
            "accent_color": "#3498db",
            "background_color": "#ffffff"
        },
        "corporate": {
            "header_color": "#1f4e79",
            "body_color": "#2f2f2f",
            "accent_color": "#4472c4",
            "background_color": "#ffffff"
        },
        "academic": {
            "header_color": "#000000",
            "body_color": "#000000",
            "accent_color": "#000000",
            "background_color": "#ffffff"
        },
        "creative": {
            "header_color": "#8e44ad",
            "body_color": "#2c3e50",
            "accent_color": "#e74c3c",
            "background_color": "#ffffff"
        },
        "minimal": {
            "header_color": "#333333",
            "body_color": "#666666",
            "accent_color": "#999999",
            "background_color": "#ffffff"
        }
    }
    
    # Predefined font combinations
    FONT_COMBINATIONS = {
        "professional": {
            "primary": "Arial",
            "secondary": "Calibri",
            "monospace": "Courier New"
        },
        "academic": {
            "primary": "Times New Roman",
            "secondary": "Georgia",
            "monospace": "Courier New"
        },
        "modern": {
            "primary": "Helvetica",
            "secondary": "Arial",
            "monospace": "Monaco"
        },
        "creative": {
            "primary": "Georgia",
            "secondary": "Verdana",
            "monospace": "Consolas"
        }
    }
    
    @classmethod
    def get_color_scheme(cls, scheme_name: str) -> Dict[str, str]:
        """Get a predefined color scheme."""
        return cls.COLOR_SCHEMES.get(scheme_name, cls.COLOR_SCHEMES["professional"])
    
    @classmethod
    def get_font_combination(cls, combination_name: str) -> Dict[str, str]:
        """Get a predefined font combination."""
        return cls.FONT_COMBINATIONS.get(combination_name, cls.FONT_COMBINATIONS["professional"])
    
    @classmethod
    def get_available_color_schemes(cls) -> List[str]:
        """Get all available color schemes."""
        return list(cls.COLOR_SCHEMES.keys())
    
    @classmethod
    def get_available_font_combinations(cls) -> List[str]:
        """Get all available font combinations."""
        return list(cls.FONT_COMBINATIONS.keys())


# Global configuration instance
config = ProfessionalDocumentsConfig()

# Utility functions
def get_config() -> ProfessionalDocumentsConfig:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update configuration settings."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Validate export directory exists or can be created
        export_dir = Path(config.export_directory)
        export_dir.mkdir(exist_ok=True)
        
        # Validate temp directory exists or can be created
        temp_dir = Path(config.temp_directory)
        temp_dir.mkdir(exist_ok=True)
        
        # Validate AI settings
        if config.ai_temperature < 0 or config.ai_temperature > 2:
            raise ValueError("AI temperature must be between 0 and 2")
        
        if config.ai_max_tokens < 100 or config.ai_max_tokens > 8000:
            raise ValueError("AI max tokens must be between 100 and 8000")
        
        # Validate document length settings
        if config.min_document_length >= config.max_document_length:
            raise ValueError("Min document length must be less than max document length")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {str(e)}")
        return False

def get_environment_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    env_config = {}
    
    # Map environment variables to config attributes
    env_mappings = {
        "PROFESSIONAL_DOCUMENTS_ENABLED": "enabled",
        "PROFESSIONAL_DOCUMENTS_AI_MODEL": "ai_model_name",
        "PROFESSIONAL_DOCUMENTS_EXPORT_DIR": "export_directory",
        "PROFESSIONAL_DOCUMENTS_DEBUG": "debug_mode"
    }
    
    for env_var, config_attr in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Convert string values to appropriate types
            if value.lower() in ("true", "false"):
                env_config[config_attr] = value.lower() == "true"
            elif value.isdigit():
                env_config[config_attr] = int(value)
            else:
                env_config[config_attr] = value
    
    return env_config

# Initialize configuration from environment
env_config = get_environment_config()
if env_config:
    update_config(**env_config)

# Validate configuration on import
if not validate_config():
    print("Warning: Professional Documents configuration validation failed")




























