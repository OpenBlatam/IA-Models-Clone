from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging
from pydantic import BaseModel, Field, validator
from pydantic.types import DirectoryPath, FilePath
from ..core.exceptions import ConfigurationError
from ..core.models import VideoQuality, VideoFormat, PluginCategory
    from multiple sources including environment variables, config files,
                import onyx.core.functions
                import onyx.utils.logger
                import onyx.llm.factory
from typing import Any, List, Dict, Optional
import asyncio
"""
Onyx AI Video System - Configuration Manager

Configuration management for the Onyx AI Video system with support for
environment variables, config files, and Onyx integration.
"""





class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_size: int = Field(default=10, description="Max log file size in MB")
    backup_count: int = Field(default=5, description="Number of backup files")
    use_onyx_logging: bool = Field(default=True, description="Use Onyx logging system")


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4", description="LLM model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(default=4000, ge=1, le=32000, description="Maximum tokens")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Retry attempts")
    use_onyx_llm: bool = Field(default=True, description="Use Onyx LLM system")


class VideoConfig(BaseModel):
    """Video generation configuration."""
    default_quality: VideoQuality = Field(default=VideoQuality.MEDIUM, description="Default video quality")
    default_format: VideoFormat = Field(default=VideoFormat.MP4, description="Default video format")
    default_duration: int = Field(default=60, ge=5, le=600, description="Default duration in seconds")
    max_duration: int = Field(default=600, ge=10, le=3600, description="Maximum duration in seconds")
    output_directory: str = Field(default="./output", description="Output directory")
    temp_directory: str = Field(default="./temp", description="Temporary directory")
    cleanup_temp: bool = Field(default=True, description="Clean up temporary files")


class PluginConfig(BaseModel):
    """Plugin system configuration."""
    plugins_directory: str = Field(default="./plugins", description="Plugins directory")
    auto_load: bool = Field(default=True, description="Auto-load plugins")
    enable_all: bool = Field(default=False, description="Enable all plugins by default")
    max_workers: int = Field(default=10, ge=1, le=50, description="Maximum plugin workers")
    timeout: int = Field(default=300, ge=30, le=3600, description="Plugin timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Plugin retry attempts")


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    metrics_interval: int = Field(default=60, description="Metrics collection interval in seconds")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_size: int = Field(default=1000, description="Cache size limit")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    gpu_enabled: bool = Field(default=True, description="Enable GPU acceleration")
    max_concurrent_requests: int = Field(default=10, ge=1, le=100, description="Max concurrent requests")


class SecurityConfig(BaseModel):
    """Security configuration."""
    enable_encryption: bool = Field(default=True, description="Enable data encryption")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key")
    validate_input: bool = Field(default=True, description="Validate input data")
    max_input_length: int = Field(default=10000, ge=100, le=100000, description="Maximum input length")
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, le=10000, description="Requests per minute")
    use_onyx_security: bool = Field(default=True, description="Use Onyx security system")


class OnyxIntegrationConfig(BaseModel):
    """Onyx integration configuration."""
    use_onyx_logging: bool = Field(default=True, description="Use Onyx logging")
    use_onyx_llm: bool = Field(default=True, description="Use Onyx LLM")
    use_onyx_telemetry: bool = Field(default=True, description="Use Onyx telemetry")
    use_onyx_encryption: bool = Field(default=True, description="Use Onyx encryption")
    use_onyx_threading: bool = Field(default=True, description="Use Onyx threading")
    use_onyx_retry: bool = Field(default=True, description="Use Onyx retry")
    use_onyx_gpu: bool = Field(default=True, description="Use Onyx GPU utilities")
    onyx_config_path: Optional[str] = Field(default=None, description="Onyx config path")


class OnyxAIVideoConfig(BaseModel):
    """
    Main configuration for Onyx AI Video system.
    
    Combines all configuration sections into a single model.
    """
    
    # System configuration
    system_name: str = Field(default="Onyx AI Video System", description="System name")
    version: str = Field(default="1.0.0", description="System version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Configuration sections
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    video: VideoConfig = Field(default_factory=VideoConfig, description="Video configuration")
    plugins: PluginConfig = Field(default_factory=PluginConfig, description="Plugin configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    onyx: OnyxIntegrationConfig = Field(default_factory=OnyxIntegrationConfig, description="Onyx integration configuration")
    
    # Custom configuration
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration")
    
    @validator('environment')
    def validate_environment(cls, v) -> bool:
        """Validate environment value."""
        valid_environments = ['development', 'testing', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    @validator('security')
    def validate_security_config(cls, v) -> bool:
        """Validate security configuration."""
        if v.enable_encryption and not v.encryption_key:
            # Try to get from environment
            env_key = os.getenv('AI_VIDEO_ENCRYPTION_KEY')
            if env_key:
                v.encryption_key = env_key
            else:
                raise ValueError("Encryption key required when encryption is enabled")
        return v
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        env_prefix = "AI_VIDEO_"
        env_file = ".env"
        env_file_encoding = "utf-8"


class OnyxConfigManager:
    """
    Configuration manager for Onyx AI Video system.
    
    Handles loading, validation, and management of configuration
    and Onyx integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        
    """__init__ function."""
self.config_path = config_path
        self.config: Optional[OnyxAIVideoConfig] = None
        self.logger = logging.getLogger(__name__)
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self, config_path: Optional[str] = None) -> OnyxAIVideoConfig:
        """
        Load configuration from file and environment.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        config_path = config_path or self.config_path
        
        try:
            # Try to load from file first
            if config_path and os.path.exists(config_path):
                self.logger.info(f"Loading configuration from: {config_path}")
                config = self._load_from_file(config_path)
            else:
                self.logger.info("No config file found, using environment variables")
                config = OnyxAIVideoConfig()
            
            # Override with environment variables
            config = self._override_from_env(config)
            
            # Validate configuration
            self._validate_config(config)
            
            # Create directories
            self._create_directories(config)
            
            self.config = config
            self.logger.info("Configuration loaded successfully")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _load_from_file(self, config_path: str) -> OnyxAIVideoConfig:
        """Load configuration from file."""
        file_ext = Path(config_path).suffix.lower()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if file_ext in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_ext == '.json':
                    data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {file_ext}")
            
            return OnyxAIVideoConfig(**data)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_path}: {e}")
    
    def _override_from_env(self, config: OnyxAIVideoConfig) -> OnyxAIVideoConfig:
        """Override configuration with environment variables."""
        # System configuration
        if env_val := os.getenv('AI_VIDEO_ENVIRONMENT'):
            config.environment = env_val
        
        if env_val := os.getenv('AI_VIDEO_DEBUG'):
            config.debug = env_val.lower() in ['true', '1', 'yes']
        
        # Logging configuration
        if env_val := os.getenv('AI_VIDEO_LOGGING_LEVEL'):
            config.logging.level = env_val
        
        if env_val := os.getenv('AI_VIDEO_LOGGING_FILE_PATH'):
            config.logging.file_path = env_val
        
        # LLM configuration
        if env_val := os.getenv('AI_VIDEO_LLM_PROVIDER'):
            config.llm.provider = env_val
        
        if env_val := os.getenv('AI_VIDEO_LLM_MODEL'):
            config.llm.model = env_val
        
        if env_val := os.getenv('AI_VIDEO_LLM_TEMPERATURE'):
            config.llm.temperature = float(env_val)
        
        # Video configuration
        if env_val := os.getenv('AI_VIDEO_OUTPUT_DIRECTORY'):
            config.video.output_directory = env_val
        
        if env_val := os.getenv('AI_VIDEO_TEMP_DIRECTORY'):
            config.video.temp_directory = env_val
        
        # Plugin configuration
        if env_val := os.getenv('AI_VIDEO_PLUGINS_DIRECTORY'):
            config.plugins.plugins_directory = env_val
        
        if env_val := os.getenv('AI_VIDEO_MAX_WORKERS'):
            config.plugins.max_workers = int(env_val)
        
        # Performance configuration
        if env_val := os.getenv('AI_VIDEO_CACHE_ENABLED'):
            config.performance.cache_enabled = env_val.lower() in ['true', '1', 'yes']
        
        if env_val := os.getenv('AI_VIDEO_CACHE_SIZE'):
            config.performance.cache_size = int(env_val)
        
        # Security configuration
        if env_val := os.getenv('AI_VIDEO_ENCRYPTION_KEY'):
            config.security.encryption_key = env_val
        
        if env_val := os.getenv('AI_VIDEO_RATE_LIMIT_REQUESTS'):
            config.security.rate_limit_requests = int(env_val)
        
        return config
    
    def _validate_config(self, config: OnyxAIVideoConfig) -> None:
        """Validate configuration."""
        # Validate directories
        if not os.path.isabs(config.video.output_directory):
            config.video.output_directory = os.path.abspath(config.video.output_directory)
        
        if not os.path.isabs(config.video.temp_directory):
            config.video.temp_directory = os.path.abspath(config.video.temp_directory)
        
        if not os.path.isabs(config.plugins.plugins_directory):
            config.plugins.plugins_directory = os.path.abspath(config.plugins.plugins_directory)
        
        # Validate LLM configuration
        if config.llm.temperature < 0 or config.llm.temperature > 2:
            raise ConfigurationError("LLM temperature must be between 0 and 2")
        
        if config.llm.max_tokens < 1 or config.llm.max_tokens > 32000:
            raise ConfigurationError("LLM max_tokens must be between 1 and 32000")
        
        # Validate video configuration
        if config.video.default_duration > config.video.max_duration:
            raise ConfigurationError("Default duration cannot exceed max duration")
        
        # Validate plugin configuration
        if config.plugins.max_workers < 1 or config.plugins.max_workers > 50:
            raise ConfigurationError("Plugin max_workers must be between 1 and 50")
        
        # Validate performance configuration
        if config.performance.cache_size < 1:
            raise ConfigurationError("Cache size must be positive")
        
        if config.performance.max_concurrent_requests < 1:
            raise ConfigurationError("Max concurrent requests must be positive")
    
    def _create_directories(self, config: OnyxAIVideoConfig) -> None:
        """Create necessary directories."""
        directories = [
            config.video.output_directory,
            config.video.temp_directory,
            config.plugins.plugins_directory
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def get_config(self) -> OnyxAIVideoConfig:
        """Get current configuration."""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def reload_config(self) -> OnyxAIVideoConfig:
        """Reload configuration."""
        self.config = None
        return self.get_config()
    
    def get_section(self, section_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration section."""
        config = self.get_config()
        return getattr(config, section_name, None)
    
    def update_config(self, updates: Dict[str, Any]) -> OnyxAIVideoConfig:
        """Update configuration with new values."""
        config = self.get_config()
        
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Try nested updates
                parts = key.split('.')
                current = config
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        break
                else:
                    if hasattr(current, parts[-1]):
                        setattr(current, parts[-1], value)
        
        self.config = config
        return config
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save configuration to file."""
        config_path = config_path or self.config_path
        if not config_path:
            raise ConfigurationError("No config path specified")
        
        config = self.get_config()
        
        try:
            file_ext = Path(config_path).suffix.lower()
            
            with open(config_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if file_ext in ['.yaml', '.yml']:
                    yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
                elif file_ext == '.json':
                    json.dump(config.dict(), f, indent=2, default=str)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {file_ext}")
            
            self.logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def get_env_config(self) -> Dict[str, str]:
        """Get configuration as environment variables."""
        config = self.get_config()
        env_config = {}
        
        # Convert config to environment variables
        for key, value in config.dict().items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    env_key = f"AI_VIDEO_{key.upper()}_{sub_key.upper()}"
                    env_config[env_key] = str(sub_value)
            else:
                env_key = f"AI_VIDEO_{key.upper()}"
                env_config[env_key] = str(value)
        
        return env_config
    
    def validate_onyx_integration(self) -> bool:
        """Validate Onyx integration configuration."""
        try:
            config = self.get_config()
            
            # Check if Onyx modules are available
            try:
            except ImportError as e:
                self.logger.warning(f"Onyx modules not available: {e}")
                return False
            
            # Validate Onyx-specific configuration
            if config.onyx.use_onyx_llm and not config.onyx.use_onyx_logging:
                self.logger.warning("Onyx LLM enabled but logging disabled")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Onyx integration validation failed: {e}")
            return False


# Global configuration manager instance
_config_manager: Optional[OnyxConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> OnyxConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = OnyxConfigManager(config_path)
    return _config_manager


def get_config(config_path: Optional[str] = None) -> OnyxAIVideoConfig:
    """Get configuration."""
    manager = get_config_manager(config_path)
    return manager.get_config()


def save_config(config: OnyxAIVideoConfig, config_path: Optional[str] = None) -> None:
    """Save configuration."""
    manager = get_config_manager(config_path)
    manager.config = config
    manager.save_config(config_path)


def reload_config(config_path: Optional[str] = None) -> OnyxAIVideoConfig:
    """Reload configuration."""
    manager = get_config_manager(config_path)
    return manager.reload_config()


def update_config(updates: Dict[str, Any], config_path: Optional[str] = None) -> OnyxAIVideoConfig:
    """Update configuration."""
    manager = get_config_manager(config_path)
    return manager.update_config(updates)


# Configuration templates
def get_default_config() -> Dict[str, Any]:
    """Get default configuration template."""
    return {
        "system_name": "Onyx AI Video System",
        "version": "1.0.0",
        "environment": "development",
        "debug": False,
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_path": None,
            "max_size": 10,
            "backup_count": 5,
            "use_onyx_logging": True
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4000,
            "timeout": 60,
            "retry_attempts": 3,
            "use_onyx_llm": True
        },
        "video": {
            "default_quality": "medium",
            "default_format": "mp4",
            "default_duration": 60,
            "max_duration": 600,
            "output_directory": "./output",
            "temp_directory": "./temp",
            "cleanup_temp": True
        },
        "plugins": {
            "plugins_directory": "./plugins",
            "auto_load": True,
            "enable_all": False,
            "max_workers": 10,
            "timeout": 300,
            "retry_attempts": 3
        },
        "performance": {
            "enable_monitoring": True,
            "metrics_interval": 60,
            "cache_enabled": True,
            "cache_size": 1000,
            "cache_ttl": 3600,
            "gpu_enabled": True,
            "max_concurrent_requests": 10
        },
        "security": {
            "enable_encryption": True,
            "encryption_key": None,
            "validate_input": True,
            "max_input_length": 10000,
            "rate_limit_enabled": True,
            "rate_limit_requests": 100,
            "use_onyx_security": True
        },
        "onyx": {
            "use_onyx_logging": True,
            "use_onyx_llm": True,
            "use_onyx_telemetry": True,
            "use_onyx_encryption": True,
            "use_onyx_threading": True,
            "use_onyx_retry": True,
            "use_onyx_gpu": True,
            "onyx_config_path": None
        },
        "custom": {}
    }


def create_config_file(config_path: str, template: Optional[Dict[str, Any]] = None) -> None:
    """Create configuration file from template."""
    template = template or get_default_config()
    
    try:
        file_ext = Path(config_path).suffix.lower()
        
        with open(config_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if file_ext in ['.yaml', '.yml']:
                yaml.dump(template, f, default_flow_style=False, indent=2)
            elif file_ext == '.json':
                json.dump(template, f, indent=2, default=str)
            else:
                raise ConfigurationError(f"Unsupported config file format: {file_ext}")
        
        print(f"Configuration file created: {config_path}")
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create config file: {e}") 