"""
TruthGPT Bulk Processor Configuration
====================================

Configuration settings specifically for the TruthGPT-inspired bulk document
generation system.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TruthGPTProcessingConfig:
    """Configuration for TruthGPT processing behavior."""
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    continuous_mode: bool = True
    auto_save_interval: int = 30  # seconds
    max_documents_per_request: int = 1000
    min_documents_per_request: int = 1
    default_max_documents: int = 100
    
    # TruthGPT-specific settings
    enable_document_variations: bool = True
    max_variations_per_type: int = 10
    variation_creativity_level: float = 0.8  # 0.0 to 1.0
    enable_cross_referencing: bool = True
    enable_content_evolution: bool = True

@dataclass
class TruthGPTDocumentConfig:
    """Configuration for TruthGPT document generation."""
    output_directory: str = "truthgpt_generated_documents"
    supported_formats: List[str] = field(default_factory=lambda: ["md", "html", "pdf", "docx", "txt"])
    max_document_size: int = 2 * 1024 * 1024  # 2MB
    template_directory: str = "truthgpt_templates"
    backup_enabled: bool = True
    compression_enabled: bool = True
    
    # TruthGPT-specific document settings
    include_metadata: bool = True
    include_cross_references: bool = True
    include_evolution_tracking: bool = True
    auto_format_documents: bool = True
    enable_document_chaining: bool = True

@dataclass
class TruthGPTBusinessConfig:
    """Configuration for TruthGPT business areas and document types."""
    enabled_areas: List[str] = field(default_factory=lambda: [
        "marketing", "sales", "operations", "hr", "finance", 
        "legal", "technical", "content", "strategy", "customer_service",
        "product_development", "business_development", "management",
        "innovation", "quality_assurance", "risk_management"
    ])
    
    supported_document_types: List[str] = field(default_factory=lambda: [
        "business_plan", "marketing_strategy", "sales_presentation", 
        "financial_analysis", "operational_manual", "hr_policy",
        "technical_documentation", "content_strategy", "legal_document",
        "customer_service_guide", "product_description", "proposal",
        "report", "white_paper", "case_study", "best_practices_guide",
        "implementation_plan", "risk_assessment", "quality_manual",
        "innovation_framework", "compliance_guide", "training_material"
    ])
    
    area_priorities: Dict[str, int] = field(default_factory=lambda: {
        "strategy": 1,
        "marketing": 1,
        "sales": 1,
        "finance": 1,
        "operations": 2,
        "technical": 2,
        "legal": 2,
        "hr": 3,
        "content": 3,
        "customer_service": 3,
        "product_development": 2,
        "business_development": 2,
        "management": 1,
        "innovation": 2,
        "quality_assurance": 3,
        "risk_management": 2
    })
    
    document_type_priorities: Dict[str, int] = field(default_factory=lambda: {
        "business_plan": 1,
        "marketing_strategy": 1,
        "financial_analysis": 1,
        "operational_manual": 2,
        "technical_documentation": 2,
        "legal_document": 2,
        "hr_policy": 3,
        "content_strategy": 3,
        "customer_service_guide": 3,
        "proposal": 2,
        "report": 2,
        "white_paper": 2,
        "case_study": 3,
        "best_practices_guide": 3,
        "implementation_plan": 2,
        "risk_assessment": 2,
        "quality_manual": 3,
        "innovation_framework": 2,
        "compliance_guide": 2,
        "training_material": 3
    })
    
    max_documents_per_area: int = 50
    max_documents_per_type: int = 30
    document_quality_threshold: float = 0.85

@dataclass
class TruthGPTAPIConfig:
    """Configuration for TruthGPT API endpoints."""
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_workers: int = 4
    api_prefix: str = "/api/v1/truthgpt"
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 200  # Higher limit for bulk operations
    rate_limit_window: int = 3600  # 1 hour
    
    # Security
    api_key_required: bool = False
    api_key: Optional[str] = None
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Streaming
    enable_streaming: bool = True
    stream_timeout: int = 300  # 5 minutes
    max_concurrent_streams: int = 10

@dataclass
class TruthGPTModelConfig:
    """Configuration for TruthGPT model settings."""
    default_model: str = "openai/gpt-4"
    fallback_models: List[str] = field(default_factory=lambda: [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-sonnet",
        "google/gemini-pro"
    ])
    
    # Model parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # TruthGPT-specific model settings
    enable_model_switching: bool = True
    model_switch_threshold: float = 0.8  # Switch if success rate drops below this
    enable_model_optimization: bool = True
    optimization_interval: int = 100  # Optimize every 100 requests

@dataclass
class TruthGPTConfig:
    """Main TruthGPT configuration class."""
    
    def __init__(self):
        self.system_name = "TruthGPT Bulk Document Generator"
        self.version = "1.0.0"
        self.debug_mode = os.getenv("TRUTHGPT_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("TRUTHGPT_LOG_LEVEL", "INFO")
        
        # Component configurations
        self.processing = TruthGPTProcessingConfig()
        self.document = TruthGPTDocumentConfig()
        self.business = TruthGPTBusinessConfig()
        self.api = TruthGPTAPIConfig()
        self.model = TruthGPTModelConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Validation
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        
        # Processing config
        if os.getenv("TRUTHGPT_MAX_CONCURRENT_TASKS"):
            self.processing.max_concurrent_tasks = int(os.getenv("TRUTHGPT_MAX_CONCURRENT_TASKS"))
        
        if os.getenv("TRUTHGPT_DEFAULT_MAX_DOCUMENTS"):
            self.processing.default_max_documents = int(os.getenv("TRUTHGPT_DEFAULT_MAX_DOCUMENTS"))
        
        # Document config
        if os.getenv("TRUTHGPT_OUTPUT_DIRECTORY"):
            self.document.output_directory = os.getenv("TRUTHGPT_OUTPUT_DIRECTORY")
        
        # API config
        if os.getenv("TRUTHGPT_API_HOST"):
            self.api.api_host = os.getenv("TRUTHGPT_API_HOST")
        
        if os.getenv("TRUTHGPT_API_PORT"):
            self.api.api_port = int(os.getenv("TRUTHGPT_API_PORT"))
        
        # Model config
        if os.getenv("TRUTHGPT_DEFAULT_MODEL"):
            self.model.default_model = os.getenv("TRUTHGPT_DEFAULT_MODEL")
        
        if os.getenv("TRUTHGPT_TEMPERATURE"):
            self.model.temperature = float(os.getenv("TRUTHGPT_TEMPERATURE"))
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate processing config
        if self.processing.max_concurrent_tasks <= 0:
            errors.append("max_concurrent_tasks must be greater than 0")
        
        if self.processing.default_max_documents < self.processing.min_documents_per_request:
            errors.append("default_max_documents must be >= min_documents_per_request")
        
        # Validate document config
        if self.document.max_document_size <= 0:
            errors.append("max_document_size must be greater than 0")
        
        # Validate API config
        if self.api.api_port <= 0 or self.api.api_port > 65535:
            errors.append("api_port must be between 1 and 65535")
        
        # Validate model config
        if not (0.0 <= self.model.temperature <= 2.0):
            errors.append("temperature must be between 0.0 and 2.0")
        
        if self.model.max_tokens <= 0:
            errors.append("max_tokens must be greater than 0")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return Path(self.document.output_directory)
    
    def get_template_directory(self) -> Path:
        """Get the template directory path."""
        return Path(self.document.template_directory)
    
    def is_area_enabled(self, area: str) -> bool:
        """Check if a business area is enabled."""
        return area in self.business.enabled_areas
    
    def is_document_type_supported(self, doc_type: str) -> bool:
        """Check if a document type is supported."""
        return doc_type in self.business.supported_document_types
    
    def get_area_priority(self, area: str) -> int:
        """Get priority for a business area."""
        return self.business.area_priorities.get(area, 5)
    
    def get_document_type_priority(self, doc_type: str) -> int:
        """Get priority for a document type."""
        return self.business.document_type_priorities.get(doc_type, 5)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            "model": self.model.default_model,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "top_p": self.model.top_p,
            "frequency_penalty": self.model.frequency_penalty,
            "presence_penalty": self.model.presence_penalty
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "processing": {
                "max_concurrent_tasks": self.processing.max_concurrent_tasks,
                "task_timeout": self.processing.task_timeout,
                "retry_attempts": self.processing.retry_attempts,
                "retry_delay": self.processing.retry_delay,
                "continuous_mode": self.processing.continuous_mode,
                "auto_save_interval": self.processing.auto_save_interval,
                "max_documents_per_request": self.processing.max_documents_per_request,
                "min_documents_per_request": self.processing.min_documents_per_request,
                "default_max_documents": self.processing.default_max_documents,
                "enable_document_variations": self.processing.enable_document_variations,
                "max_variations_per_type": self.processing.max_variations_per_type,
                "variation_creativity_level": self.processing.variation_creativity_level,
                "enable_cross_referencing": self.processing.enable_cross_referencing,
                "enable_content_evolution": self.processing.enable_content_evolution
            },
            "document": {
                "output_directory": self.document.output_directory,
                "supported_formats": self.document.supported_formats,
                "max_document_size": self.document.max_document_size,
                "template_directory": self.document.template_directory,
                "backup_enabled": self.document.backup_enabled,
                "compression_enabled": self.document.compression_enabled,
                "include_metadata": self.document.include_metadata,
                "include_cross_references": self.document.include_cross_references,
                "include_evolution_tracking": self.document.include_evolution_tracking,
                "auto_format_documents": self.document.auto_format_documents,
                "enable_document_chaining": self.document.enable_document_chaining
            },
            "business": {
                "enabled_areas": self.business.enabled_areas,
                "supported_document_types": self.business.supported_document_types,
                "area_priorities": self.business.area_priorities,
                "document_type_priorities": self.business.document_type_priorities,
                "max_documents_per_area": self.business.max_documents_per_area,
                "max_documents_per_type": self.business.max_documents_per_type,
                "document_quality_threshold": self.business.document_quality_threshold
            },
            "api": {
                "api_host": self.api.api_host,
                "api_port": self.api.api_port,
                "api_workers": self.api.api_workers,
                "api_prefix": self.api.api_prefix,
                "rate_limit_enabled": self.api.rate_limit_enabled,
                "rate_limit_requests": self.api.rate_limit_requests,
                "rate_limit_window": self.api.rate_limit_window,
                "api_key_required": self.api.api_key_required,
                "cors_enabled": self.api.cors_enabled,
                "cors_origins": self.api.cors_origins,
                "enable_streaming": self.api.enable_streaming,
                "stream_timeout": self.api.stream_timeout,
                "max_concurrent_streams": self.api.max_concurrent_streams
            },
            "model": {
                "default_model": self.model.default_model,
                "fallback_models": self.model.fallback_models,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "top_p": self.model.top_p,
                "frequency_penalty": self.model.frequency_penalty,
                "presence_penalty": self.model.presence_penalty,
                "enable_model_switching": self.model.enable_model_switching,
                "model_switch_threshold": self.model.model_switch_threshold,
                "enable_model_optimization": self.model.enable_model_optimization,
                "optimization_interval": self.model.optimization_interval
            }
        }

# Global configuration instance
_global_truthgpt_config: Optional[TruthGPTConfig] = None

def get_global_truthgpt_config() -> TruthGPTConfig:
    """Get the global TruthGPT configuration instance."""
    global _global_truthgpt_config
    if _global_truthgpt_config is None:
        _global_truthgpt_config = TruthGPTConfig()
    return _global_truthgpt_config



























