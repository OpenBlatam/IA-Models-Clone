"""
Configuration for Business Agents System
========================================

Configuration settings and environment variables for the business agents system.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DatabaseType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"

class CacheType(str, Enum):
    REDIS = "redis"
    MEMORY = "memory"
    NONE = "none"

class BusinessAgentsConfig(BaseSettings):
    """Configuration for Business Agents System."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Database Configuration
    database_type: DatabaseType = Field(default=DatabaseType.SQLITE, env="DATABASE_TYPE")
    database_url: str = Field(default="sqlite:///./business_agents.db", env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Cache Configuration
    cache_type: CacheType = Field(default=CacheType.MEMORY, env="CACHE_TYPE")
    cache_url: str = Field(default="redis://localhost:6379/0", env="CACHE_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    # Workflow Engine Configuration
    workflow_max_concurrent: int = Field(default=10, env="WORKFLOW_MAX_CONCURRENT")
    workflow_default_timeout: int = Field(default=300, env="WORKFLOW_DEFAULT_TIMEOUT")
    workflow_retry_delay: int = Field(default=5, env="WORKFLOW_RETRY_DELAY")
    workflow_max_retries: int = Field(default=3, env="WORKFLOW_MAX_RETRIES")
    
    # Document Generation Configuration
    document_output_dir: str = Field(default="./generated_documents", env="DOCUMENT_OUTPUT_DIR")
    document_max_size: int = Field(default=10 * 1024 * 1024, env="DOCUMENT_MAX_SIZE")  # 10MB
    document_cleanup_interval: int = Field(default=86400, env="DOCUMENT_CLEANUP_INTERVAL")  # 24 hours
    document_retention_days: int = Field(default=30, env="DOCUMENT_RETENTION_DAYS")
    
    # Agent Configuration
    agent_execution_timeout: int = Field(default=600, env="AGENT_EXECUTION_TIMEOUT")  # 10 minutes
    agent_max_concurrent: int = Field(default=5, env="AGENT_MAX_CONCURRENT")
    agent_health_check_interval: int = Field(default=60, env="AGENT_HEALTH_CHECK_INTERVAL")  # 1 minute
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # 30 seconds
    
    # External Services Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # Email Configuration
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    
    # File Storage Configuration
    storage_type: str = Field(default="local", env="STORAGE_TYPE")  # local, s3, gcs
    storage_bucket: Optional[str] = Field(default=None, env="STORAGE_BUCKET")
    storage_region: Optional[str] = Field(default=None, env="STORAGE_REGION")
    storage_access_key: Optional[str] = Field(default=None, env="STORAGE_ACCESS_KEY")
    storage_secret_key: Optional[str] = Field(default=None, env="STORAGE_SECRET_KEY")
    
    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # CORS Configuration
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: list = Field(default=["GET", "POST", "PUT", "DELETE"], env="CORS_METHODS")
    cors_headers: list = Field(default=["*"], env="CORS_HEADERS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global configuration instance
config = BusinessAgentsConfig()

# Business area configurations
BUSINESS_AREA_CONFIGS = {
    "marketing": {
        "max_concurrent_workflows": 5,
        "default_timeout": 600,
        "supported_document_types": ["campaign_plan", "content_brief", "strategy_document"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "high"
    },
    "sales": {
        "max_concurrent_workflows": 3,
        "default_timeout": 900,
        "supported_document_types": ["proposal", "contract", "presentation"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "high"
    },
    "operations": {
        "max_concurrent_workflows": 8,
        "default_timeout": 1200,
        "supported_document_types": ["process_document", "manual", "sop"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "medium"
    },
    "hr": {
        "max_concurrent_workflows": 4,
        "default_timeout": 480,
        "supported_document_types": ["policy", "training_material", "job_description"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "medium"
    },
    "finance": {
        "max_concurrent_workflows": 3,
        "default_timeout": 1800,
        "supported_document_types": ["financial_report", "budget", "analysis"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "high"
    },
    "legal": {
        "max_concurrent_workflows": 2,
        "default_timeout": 2400,
        "supported_document_types": ["contract", "agreement", "policy"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "high"
    },
    "technical": {
        "max_concurrent_workflows": 6,
        "default_timeout": 1500,
        "supported_document_types": ["specification", "documentation", "manual"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "medium"
    },
    "content": {
        "max_concurrent_workflows": 10,
        "default_timeout": 300,
        "supported_document_types": ["blog_post", "article", "social_media"],
        "ai_models": ["gpt-4", "claude-3"],
        "priority": "low"
    }
}

# Document type configurations
DOCUMENT_TYPE_CONFIGS = {
    "business_plan": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 1800,  # 30 minutes
        "complexity": "high",
        "sections": ["executive_summary", "market_analysis", "financial_projections"]
    },
    "marketing_strategy": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 1200,  # 20 minutes
        "complexity": "high",
        "sections": ["market_analysis", "target_audience", "marketing_mix"]
    },
    "sales_proposal": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 900,  # 15 minutes
        "complexity": "medium",
        "sections": ["problem_statement", "solution", "pricing"]
    },
    "financial_report": {
        "template_required": True,
        "ai_enhancement": False,
        "estimated_duration": 2400,  # 40 minutes
        "complexity": "high",
        "sections": ["executive_summary", "financial_statements", "analysis"]
    },
    "operational_manual": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 3600,  # 60 minutes
        "complexity": "high",
        "sections": ["overview", "procedures", "troubleshooting"]
    },
    "hr_policy": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 600,  # 10 minutes
        "complexity": "medium",
        "sections": ["policy_statement", "procedures", "compliance"]
    },
    "technical_specification": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 1800,  # 30 minutes
        "complexity": "high",
        "sections": ["overview", "requirements", "implementation"]
    },
    "project_proposal": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 1200,  # 20 minutes
        "complexity": "medium",
        "sections": ["project_overview", "timeline", "budget"]
    },
    "contract": {
        "template_required": True,
        "ai_enhancement": False,
        "estimated_duration": 1800,  # 30 minutes
        "complexity": "high",
        "sections": ["parties", "terms", "conditions"]
    },
    "presentation": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 900,  # 15 minutes
        "complexity": "medium",
        "sections": ["introduction", "main_content", "conclusion"]
    },
    "email_template": {
        "template_required": False,
        "ai_enhancement": True,
        "estimated_duration": 300,  # 5 minutes
        "complexity": "low",
        "sections": ["subject", "body", "call_to_action"]
    },
    "social_media_post": {
        "template_required": False,
        "ai_enhancement": True,
        "estimated_duration": 180,  # 3 minutes
        "complexity": "low",
        "sections": ["content", "hashtags", "call_to_action"]
    },
    "blog_post": {
        "template_required": False,
        "ai_enhancement": True,
        "estimated_duration": 1200,  # 20 minutes
        "complexity": "medium",
        "sections": ["introduction", "main_content", "conclusion"]
    },
    "press_release": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 600,  # 10 minutes
        "complexity": "medium",
        "sections": ["headline", "body", "contact_info"]
    },
    "user_manual": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 3600,  # 60 minutes
        "complexity": "high",
        "sections": ["introduction", "instructions", "troubleshooting"]
    },
    "training_material": {
        "template_required": True,
        "ai_enhancement": True,
        "estimated_duration": 2400,  # 40 minutes
        "complexity": "high",
        "sections": ["objectives", "content", "assessment"]
    }
}

# Workflow step type configurations
WORKFLOW_STEP_CONFIGS = {
    "task": {
        "supports_parallel": True,
        "supports_retry": True,
        "default_timeout": 300,
        "requires_agent": True
    },
    "condition": {
        "supports_parallel": False,
        "supports_retry": False,
        "default_timeout": 30,
        "requires_agent": False
    },
    "parallel": {
        "supports_parallel": True,
        "supports_retry": True,
        "default_timeout": 600,
        "requires_agent": False
    },
    "sequence": {
        "supports_parallel": False,
        "supports_retry": True,
        "default_timeout": 900,
        "requires_agent": False
    },
    "loop": {
        "supports_parallel": False,
        "supports_retry": True,
        "default_timeout": 1800,
        "requires_agent": False
    },
    "api_call": {
        "supports_parallel": True,
        "supports_retry": True,
        "default_timeout": 120,
        "requires_agent": False
    },
    "document_generation": {
        "supports_parallel": True,
        "supports_retry": True,
        "default_timeout": 600,
        "requires_agent": True
    },
    "notification": {
        "supports_parallel": True,
        "supports_retry": True,
        "default_timeout": 60,
        "requires_agent": False
    }
}

def get_business_area_config(business_area: str) -> Dict[str, Any]:
    """Get configuration for a specific business area."""
    return BUSINESS_AREA_CONFIGS.get(business_area, {})

def get_document_type_config(document_type: str) -> Dict[str, Any]:
    """Get configuration for a specific document type."""
    return DOCUMENT_TYPE_CONFIGS.get(document_type, {})

def get_workflow_step_config(step_type: str) -> Dict[str, Any]:
    """Get configuration for a specific workflow step type."""
    return WORKFLOW_STEP_CONFIGS.get(step_type, {})

def is_development() -> bool:
    """Check if running in development mode."""
    return config.environment == Environment.DEVELOPMENT

def is_production() -> bool:
    """Check if running in production mode."""
    return config.environment == Environment.PRODUCTION

def get_database_url() -> str:
    """Get database URL based on configuration."""
    if config.database_type == DatabaseType.SQLITE:
        return config.database_url
    elif config.database_type == DatabaseType.POSTGRESQL:
        return config.database_url
    elif config.database_type == DatabaseType.MYSQL:
        return config.database_url
    else:
        raise ValueError(f"Unsupported database type: {config.database_type}")

def get_cache_config() -> Dict[str, Any]:
    """Get cache configuration."""
    return {
        "type": config.cache_type,
        "url": config.cache_url,
        "ttl": config.cache_ttl
    }





























