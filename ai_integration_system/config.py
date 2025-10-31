"""
AI Integration System Configuration
Centralized configuration management for all platform integrations
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    url: str = Field(default="postgresql://postgres:password@localhost:5432/ai_integration")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo: bool = Field(default=False)

class RedisConfig(BaseSettings):
    """Redis configuration"""
    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=10)

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None)

class SecurityConfig(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(default="your-secret-key-here")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

class SalesforceConfig(BaseSettings):
    """Salesforce configuration"""
    enabled: bool = Field(default=False)
    base_url: str = Field(default="https://your-instance.salesforce.com")
    client_id: Optional[str] = Field(default=None)
    client_secret: Optional[str] = Field(default=None)
    username: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)
    security_token: Optional[str] = Field(default=None)
    sandbox: bool = Field(default=False)

class MailchimpConfig(BaseSettings):
    """Mailchimp configuration"""
    enabled: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)
    server_prefix: Optional[str] = Field(default=None)
    list_id: Optional[str] = Field(default=None)
    audience_id: Optional[str] = Field(default=None)

class WordPressConfig(BaseSettings):
    """WordPress configuration"""
    enabled: bool = Field(default=False)
    base_url: Optional[str] = Field(default=None)
    username: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)
    application_password: Optional[str] = Field(default=None)

class HubSpotConfig(BaseSettings):
    """HubSpot configuration"""
    enabled: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)
    access_token: Optional[str] = Field(default=None)
    portal_id: Optional[str] = Field(default=None)

class SlackConfig(BaseSettings):
    """Slack configuration"""
    enabled: bool = Field(default=False)
    bot_token: Optional[str] = Field(default=None)
    app_token: Optional[str] = Field(default=None)
    signing_secret: Optional[str] = Field(default=None)

class GoogleConfig(BaseSettings):
    """Google Workspace configuration"""
    enabled: bool = Field(default=False)
    credentials_file: Optional[str] = Field(default=None)
    scopes: list = Field(default=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"])

class MicrosoftConfig(BaseSettings):
    """Microsoft 365 configuration"""
    enabled: bool = Field(default=False)
    client_id: Optional[str] = Field(default=None)
    client_secret: Optional[str] = Field(default=None)
    tenant_id: Optional[str] = Field(default=None)

class AIConfig(BaseSettings):
    """AI services configuration"""
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    default_model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=2000)
    temperature: float = Field(default=0.7)

class IntegrationConfig(BaseSettings):
    """Integration system configuration"""
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=5)  # seconds
    batch_size: int = Field(default=10)
    queue_timeout: int = Field(default=300)  # seconds
    webhook_timeout: int = Field(default=30)  # seconds

class MonitoringConfig(BaseSettings):
    """Monitoring configuration"""
    prometheus_enabled: bool = Field(default=True)
    sentry_dsn: Optional[str] = Field(default=None)
    health_check_interval: int = Field(default=60)  # seconds

class Settings(BaseSettings):
    """Main application settings"""
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Platform configurations
    salesforce: SalesforceConfig = Field(default_factory=SalesforceConfig)
    mailchimp: MailchimpConfig = Field(default_factory=MailchimpConfig)
    wordpress: WordPressConfig = Field(default_factory=WordPressConfig)
    hubspot: HubSpotConfig = Field(default_factory=HubSpotConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    google: GoogleConfig = Field(default_factory=GoogleConfig)
    microsoft: MicrosoftConfig = Field(default_factory=MicrosoftConfig)
    
    # AI and integration configurations
    ai: AIConfig = Field(default_factory=AIConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_platform_config(platform: str) -> Dict[str, Any]:
    """Get configuration for a specific platform"""
    platform_configs = {
        "salesforce": settings.salesforce,
        "mailchimp": settings.mailchimp,
        "wordpress": settings.wordpress,
        "hubspot": settings.hubspot,
        "slack": settings.slack,
        "google": settings.google,
        "microsoft": settings.microsoft
    }
    
    config = platform_configs.get(platform.lower())
    if not config:
        return {}
    
    return config.dict()

def is_platform_enabled(platform: str) -> bool:
    """Check if a platform is enabled"""
    config = get_platform_config(platform)
    return config.get("enabled", False)

def get_database_url() -> str:
    """Get database URL from environment or config"""
    return os.getenv("DATABASE_URL", settings.database.url)

def get_redis_url() -> str:
    """Get Redis URL from environment or config"""
    return os.getenv("REDIS_URL", settings.redis.url)

def get_log_level() -> str:
    """Get log level from environment or config"""
    return os.getenv("LOG_LEVEL", settings.logging.level)

# Environment-specific configurations
def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration"""
    env_configs = {
        Environment.DEVELOPMENT: {
            "debug": True,
            "log_level": "DEBUG",
            "database_echo": True,
            "cors_origins": ["*"]
        },
        Environment.STAGING: {
            "debug": False,
            "log_level": "INFO",
            "database_echo": False,
            "cors_origins": ["https://staging.example.com"]
        },
        Environment.PRODUCTION: {
            "debug": False,
            "log_level": "WARNING",
            "database_echo": False,
            "cors_origins": ["https://example.com"]
        }
    }
    
    return env_configs.get(settings.environment, env_configs[Environment.DEVELOPMENT])

# Validation functions
def validate_platform_config(platform: str) -> bool:
    """Validate platform configuration"""
    config = get_platform_config(platform)
    
    if not config.get("enabled", False):
        return False
    
    required_fields = {
        "salesforce": ["base_url", "client_id", "client_secret", "username", "password"],
        "mailchimp": ["api_key", "server_prefix"],
        "wordpress": ["base_url", "username"],
        "hubspot": ["api_key"],
        "slack": ["bot_token"],
        "google": ["credentials_file"],
        "microsoft": ["client_id", "client_secret", "tenant_id"]
    }
    
    required = required_fields.get(platform.lower(), [])
    return all(config.get(field) for field in required)

def get_enabled_platforms() -> list:
    """Get list of enabled and properly configured platforms"""
    platforms = ["salesforce", "mailchimp", "wordpress", "hubspot", "slack", "google", "microsoft"]
    return [platform for platform in platforms if validate_platform_config(platform)]

# Export main settings
__all__ = [
    "settings",
    "get_platform_config",
    "is_platform_enabled",
    "get_database_url",
    "get_redis_url",
    "get_log_level",
    "get_environment_config",
    "validate_platform_config",
    "get_enabled_platforms"
]



























