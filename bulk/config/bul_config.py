"""
BUL System Configuration
========================

Main configuration for the Business Unlimited system.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    max_concurrent_tasks: int = 5
    task_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    continuous_mode: bool = True
    auto_save_interval: int = 60  # seconds

@dataclass
class DocumentConfig:
    """Configuration for document generation."""
    output_directory: str = "generated_documents"
    supported_formats: List[str] = field(default_factory=lambda: ["md", "html", "pdf", "docx", "txt"])
    max_document_size: int = 1024 * 1024  # 1MB
    template_directory: str = "templates"
    backup_enabled: bool = True
    compression_enabled: bool = True

@dataclass
class SMEConfig:
    """Configuration for SME business areas."""
    enabled_areas: List[str] = field(default_factory=lambda: [
        "marketing", "sales", "operations", "hr", "finance", 
        "legal", "technical", "content", "strategy", "customer_service"
    ])
    area_priorities: Dict[str, int] = field(default_factory=lambda: {
        "marketing": 1,
        "sales": 1,
        "operations": 2,
        "hr": 3,
        "finance": 1,
        "legal": 2,
        "technical": 2,
        "content": 3,
        "strategy": 1,
        "customer_service": 3
    })
    max_documents_per_area: int = 10
    document_quality_threshold: float = 0.8

@dataclass
class BULConfig:
    """Main BUL system configuration."""
    
    def __init__(self):
        self.system_name = "BUL - Business Unlimited"
        self.version = "1.0.0"
        self.debug_mode = os.getenv("BUL_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("BUL_LOG_LEVEL", "INFO")
        
        # Processing configuration
        self.processing = ProcessingConfig()
        
        # Document configuration
        self.document = DocumentConfig()
        
        # SME configuration
        self.sme = SMEConfig()
        
        # API configuration
        self.api_host = os.getenv("BUL_API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("BUL_API_PORT", "8000"))
        self.api_workers = int(os.getenv("BUL_API_WORKERS", "4"))
        
        # Database configuration
        self.database_url = os.getenv("BUL_DATABASE_URL", "sqlite:///bul.db")
        
        # Cache configuration
        self.cache_enabled = os.getenv("BUL_CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("BUL_CACHE_TTL", "3600"))  # 1 hour
        
        # Security configuration
        self.api_key_required = os.getenv("BUL_API_KEY_REQUIRED", "false").lower() == "true"
        self.api_key = os.getenv("BUL_API_KEY")
        
        # Rate limiting
        self.rate_limit_enabled = os.getenv("BUL_RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.rate_limit_requests = int(os.getenv("BUL_RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("BUL_RATE_LIMIT_WINDOW", "3600"))  # 1 hour
        
    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return Path(self.document.output_directory)
    
    def get_template_directory(self) -> Path:
        """Get the template directory path."""
        return Path(self.document.template_directory)
    
    def is_area_enabled(self, area: str) -> bool:
        """Check if a business area is enabled."""
        return area in self.sme.enabled_areas
    
    def get_area_priority(self, area: str) -> int:
        """Get priority for a business area."""
        return self.sme.area_priorities.get(area, 5)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if not self.api_key and self.api_key_required:
            errors.append("API key is required but not provided")
        
        if self.processing.max_concurrent_tasks <= 0:
            errors.append("max_concurrent_tasks must be greater than 0")
        
        if self.document.max_document_size <= 0:
            errors.append("max_document_size must be greater than 0")
        
        return errors

