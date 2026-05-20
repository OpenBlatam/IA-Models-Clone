from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from .plugins import ValidationLevel
from .plugins.config import PluginSystemConfig
    import sys
from typing import Any, List, Dict, Optional
import asyncio
"""
Unified Configuration System for AI Video

This module provides a comprehensive configuration system that integrates
all components of the AI video system including plugins, workflow, and
existing components.
"""



logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class VideoQuality(Enum):
    """Video quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class WorkflowConfig:
    """Configuration for the video workflow."""
    # General settings
    max_concurrent_workflows: int = 5
    workflow_timeout: int = 300  # 5 minutes
    enable_retry: bool = True
    max_retries: int = 3
    
    # Content extraction
    extraction_timeout: int = 60
    max_content_length: int = 50000
    enable_language_detection: bool = True
    
    # Video generation
    default_duration: float = 30.0
    default_resolution: str = "1920x1080"
    default_quality: VideoQuality = VideoQuality.HIGH
    enable_avatar_selection: bool = True
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    enable_metrics: bool = True
    enable_monitoring: bool = True


@dataclass
class AIConfig:
    """Configuration for AI models and services."""
    # Model settings
    default_model: str = "gpt-4"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.7
    
    # API settings
    api_timeout: int = 30
    api_retries: int = 3
    enable_streaming: bool = False
    
    # Content optimization
    enable_content_optimization: bool = True
    enable_short_video_optimization: bool = True
    enable_langchain_analysis: bool = True
    
    # Suggestions
    suggestion_count: int = 3
    enable_music_suggestions: bool = True
    enable_visual_suggestions: bool = True
    enable_transition_suggestions: bool = True


@dataclass
class StorageConfig:
    """Configuration for storage and file management."""
    # Local storage
    local_storage_path: str = "./storage"
    temp_directory: str = "./temp"
    output_directory: str = "./output"
    
    # File management
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_formats: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "mkv"])
    enable_compression: bool = True
    
    # Cleanup
    auto_cleanup: bool = True
    cleanup_interval: int = 86400  # 24 hours
    max_age_days: int = 7


@dataclass
class SecurityConfig:
    """Configuration for security and access control."""
    # Authentication
    enable_auth: bool = False
    auth_token_expiry: int = 3600  # 1 hour
    
    # Input validation
    enable_url_validation: bool = True
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    
    # Content filtering
    enable_content_filtering: bool = True
    filter_inappropriate_content: bool = True
    enable_nsfw_detection: bool = False
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and observability."""
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    enable_structured_logging: bool = True
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_prometheus: bool = True
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: int = 300  # 5 minutes
    
    # Alerts
    enable_alerts: bool = False
    alert_webhook_url: Optional[str] = None


@dataclass
class AIVideoConfig:
    """
    Complete configuration for the AI Video system.
    
    This class integrates all configuration aspects including:
    - Plugin system configuration
    - Workflow configuration
    - AI model configuration
    - Storage configuration
    - Security configuration
    - Monitoring configuration
    """
    
    # Core configurations
    plugins: PluginSystemConfig = field(default_factory=PluginSystemConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # Version
    version: str = "1.0.0"
    
    def __post_init__(self) -> Any:
        """Post-initialization setup."""
        # Ensure directories exist
        self._ensure_directories()
        
        # Validate configuration
        self._validate_config()
    
    def _ensure_directories(self) -> Any:
        """Ensure required directories exist."""
        directories = [
            self.storage.local_storage_path,
            self.storage.temp_directory,
            self.storage.output_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self) -> bool:
        """Validate configuration values."""
        if self.workflow.max_concurrent_workflows <= 0:
            raise ValueError("max_concurrent_workflows must be positive")
        
        if self.workflow.workflow_timeout <= 0:
            raise ValueError("workflow_timeout must be positive")
        
        if self.ai.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0 <= self.ai.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'plugins': self.plugins.__dict__,
            'workflow': self.workflow.__dict__,
            'ai': self.ai.__dict__,
            'storage': self.storage.__dict__,
            'security': self.security.__dict__,
            'monitoring': self.monitoring.__dict__,
            'environment': self.environment,
            'debug': self.debug,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIVideoConfig':
        """Create configuration from dictionary."""
        # Reconstruct nested configurations
        plugins_config = PluginSystemConfig(**data.get('plugins', {}))
        workflow_config = WorkflowConfig(**data.get('workflow', {}))
        ai_config = AIConfig(**data.get('ai', {}))
        storage_config = StorageConfig(**data.get('storage', {}))
        security_config = SecurityConfig(**data.get('security', {}))
        monitoring_config = MonitoringConfig(**data.get('monitoring', {}))
        
        return cls(
            plugins=plugins_config,
            workflow=workflow_config,
            ai=ai_config,
            storage=storage_config,
            security=security_config,
            monitoring=monitoring_config,
            environment=data.get('environment', 'development'),
            debug=data.get('debug', False),
            version=data.get('version', '1.0.0')
        )
    
    def update(self, **kwargs) -> 'AIVideoConfig':
        """Update configuration with new values."""
        data = self.to_dict()
        data.update(kwargs)
        return self.from_dict(data)


class ConfigManager:
    """
    Configuration manager for the AI Video system.
    
    This class provides:
    - Configuration loading from multiple sources
    - Environment variable support
    - Configuration validation
    - Configuration saving and reloading
    """
    
    def __init__(self, config_file: Optional[str] = None):
        
    """__init__ function."""
self.config_file = config_file
        self.config = AIVideoConfig()
        self.sources: Dict[str, str] = {}
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> Any:
        """Load configuration from all sources."""
        # Load defaults
        self.sources['default'] = 'default'
        
        # Load from file
        if self.config_file:
            self._load_from_file(self.config_file)
        
        # Load from environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_from_file(self, file_path: str):
        """Load configuration from file."""
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return
        
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                with open(path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return
            
            # Update configuration
            self.config = AIVideoConfig.from_dict(data)
            self.sources['file'] = file_path
            
            logger.info(f"Configuration loaded from file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from file {file_path}: {e}")
    
    def _load_from_environment(self) -> Any:
        """Load configuration from environment variables."""
        env_mappings = {
            # Plugin configuration
            'AI_VIDEO_PLUGIN_AUTO_DISCOVER': ('plugins.auto_discover', bool),
            'AI_VIDEO_PLUGIN_AUTO_LOAD': ('plugins.auto_load', bool),
            'AI_VIDEO_PLUGIN_VALIDATION_LEVEL': ('plugins.validation_level', str),
            
            # Workflow configuration
            'AI_VIDEO_MAX_CONCURRENT_WORKFLOWS': ('workflow.max_concurrent_workflows', int),
            'AI_VIDEO_WORKFLOW_TIMEOUT': ('workflow.workflow_timeout', int),
            'AI_VIDEO_DEFAULT_DURATION': ('workflow.default_duration', float),
            'AI_VIDEO_DEFAULT_RESOLUTION': ('workflow.default_resolution', str),
            
            # AI configuration
            'AI_VIDEO_DEFAULT_MODEL': ('ai.default_model', str),
            'AI_VIDEO_MAX_TOKENS': ('ai.max_tokens', int),
            'AI_VIDEO_TEMPERATURE': ('ai.temperature', float),
            
            # Storage configuration
            'AI_VIDEO_STORAGE_PATH': ('storage.local_storage_path', str),
            'AI_VIDEO_TEMP_DIR': ('storage.temp_directory', str),
            'AI_VIDEO_OUTPUT_DIR': ('storage.output_directory', str),
            
            # Security configuration
            'AI_VIDEO_ENABLE_AUTH': ('security.enable_auth', bool),
            'AI_VIDEO_ENABLE_RATE_LIMITING': ('security.enable_rate_limiting', bool),
            
            # Monitoring configuration
            'AI_VIDEO_LOG_LEVEL': ('monitoring.log_level', str),
            'AI_VIDEO_ENABLE_METRICS': ('monitoring.enable_metrics', bool),
            
            # General configuration
            'AI_VIDEO_ENVIRONMENT': ('environment', str),
            'AI_VIDEO_DEBUG': ('debug', bool),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert value to appropriate type
                    if value_type == bool:
                        converted_value = env_value.lower() in ['true', '1', 'yes', 'on']
                    elif value_type == int:
                        converted_value = int(env_value)
                    elif value_type == float:
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    # Update configuration
                    self._set_nested_value(config_path, converted_value)
                    self.sources[config_path] = f"env:{env_var}"
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to convert environment variable {env_var}: {e}")
    
    def _set_nested_value(self, path: str, value: Any):
        """Set a nested configuration value."""
        parts = path.split('.')
        current = self.config
        
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return
        
        if hasattr(current, parts[-1]):
            setattr(current, parts[-1], value)
    
    def _validate_configuration(self) -> bool:
        """Validate the configuration."""
        try:
            self.config._validate_config()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_config(self) -> AIVideoConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            self.config = self.config.update(**updates)
            self._validate_configuration()
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """Save configuration to file."""
        try:
            path = Path(file_path or self.config_file or "ai_video_config.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = self.config.to_dict()
            
            if path.suffix.lower() == '.json':
                with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(config_dict, f, indent=2, default=str)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    yaml.dump(config_dict, f, default_flow_style=False)
            else:
                # Default to JSON
                path = path.with_suffix('.json')
                with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            'version': self.config.version,
            'environment': self.config.environment,
            'debug': self.config.debug,
            'sources': self.sources,
            'components': {
                'plugins': len(self.config.plugins.plugin_dirs),
                'workflow': self.config.workflow.max_concurrent_workflows,
                'ai': self.config.ai.default_model,
                'storage': self.config.storage.local_storage_path,
                'security': self.config.security.enable_auth,
                'monitoring': self.config.monitoring.enable_metrics
            }
        }


# Convenience functions

def load_config(config_file: Optional[str] = None) -> AIVideoConfig:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        AIVideoConfig instance
    """
    manager = ConfigManager(config_file)
    return manager.get_config()


# Alias for backward compatibility
load_configuration = load_config


def create_default_config(file_path: str = "ai_video_config.json") -> bool:
    """
    Create a default configuration file.
    
    Args:
        file_path: Path to save the configuration file
        
    Returns:
        True if successful
    """
    config = AIVideoConfig()
    manager = ConfigManager()
    manager.config = config
    return manager.save_config(file_path)


def get_config_from_env() -> AIVideoConfig:
    """
    Load configuration from environment variables only.
    
    Returns:
        AIVideoConfig instance
    """
    manager = ConfigManager()
    # Override to only use environment variables
    manager.config = AIVideoConfig()
    manager._load_from_environment()
    manager._validate_configuration()
    return manager.config


# Configuration templates

DEFAULT_CONFIG = {
    "plugins": {
        "auto_discover": True,
        "auto_load": True,
        "validation_level": "standard",
        "plugin_dirs": ["./plugins", "./ai_video/plugins", "./extensions"],
        "enable_events": True,
        "enable_metrics": True
    },
    "workflow": {
        "max_concurrent_workflows": 5,
        "workflow_timeout": 300,
        "enable_retry": True,
        "max_retries": 3,
        "extraction_timeout": 60,
        "max_content_length": 50000,
        "enable_language_detection": True,
        "default_duration": 30.0,
        "default_resolution": "1920x1080",
        "default_quality": "high",
        "enable_avatar_selection": True,
        "enable_caching": True,
        "cache_ttl": 3600,
        "enable_metrics": True,
        "enable_monitoring": True
    },
    "ai": {
        "default_model": "gpt-4",
        "fallback_model": "gpt-3.5-turbo",
        "max_tokens": 4000,
        "temperature": 0.7,
        "api_timeout": 30,
        "api_retries": 3,
        "enable_streaming": False,
        "enable_content_optimization": True,
        "enable_short_video_optimization": True,
        "enable_langchain_analysis": True,
        "suggestion_count": 3,
        "enable_music_suggestions": True,
        "enable_visual_suggestions": True,
        "enable_transition_suggestions": True
    },
    "storage": {
        "local_storage_path": "./storage",
        "temp_directory": "./temp",
        "output_directory": "./output",
        "max_file_size": 104857600,
        "allowed_formats": ["mp4", "avi", "mov", "mkv"],
        "enable_compression": True,
        "auto_cleanup": True,
        "cleanup_interval": 86400,
        "max_age_days": 7
    },
    "security": {
        "enable_auth": False,
        "auth_token_expiry": 3600,
        "enable_url_validation": True,
        "allowed_domains": [],
        "blocked_domains": [],
        "enable_content_filtering": True,
        "filter_inappropriate_content": True,
        "enable_nsfw_detection": False,
        "enable_rate_limiting": True,
        "max_requests_per_minute": 60,
        "max_requests_per_hour": 1000
    },
    "monitoring": {
        "log_level": "INFO",
        "log_file": None,
        "enable_structured_logging": True,
        "enable_metrics": True,
        "metrics_port": 9090,
        "enable_prometheus": True,
        "enable_health_checks": True,
        "health_check_interval": 300,
        "enable_alerts": False,
        "alert_webhook_url": None
    },
    "environment": "development",
    "debug": False,
    "version": "1.0.0"
}


if __name__ == "__main__":
    # Create default configuration file
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        file_path = sys.argv[2] if len(sys.argv) > 2 else "ai_video_config.json"
        if create_default_config(file_path):
            print(f"✅ Default configuration created: {file_path}")
        else:
            print("❌ Failed to create configuration file")
    else:
        # Show current configuration
        config = load_config()
        print("Current configuration:")
        summary = ConfigManager().get_config_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}") 