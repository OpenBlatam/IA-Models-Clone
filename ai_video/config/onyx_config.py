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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from onyx.utils.logger import setup_logger
from onyx.core.config import get_config as get_onyx_config
from onyx.utils.file import get_file_extension, get_file_size
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Onyx AI Video Configuration

Configuration system that integrates with Onyx's configuration patterns
and utilities for seamless operation within the Onyx ecosystem.
"""


# Onyx imports

logger = setup_logger(__name__)


@dataclass
class OnyxAIVideoConfig:
    """
    Onyx AI Video Configuration.
    
    Integrates with Onyx's configuration system while providing
    AI Video specific configuration options.
    """
    
    # Onyx Integration Settings
    use_onyx_logging: bool = True
    use_onyx_llm: bool = True
    use_onyx_telemetry: bool = True
    use_onyx_encryption: bool = True
    use_onyx_threading: bool = True
    use_onyx_retry: bool = True
    use_onyx_gpu: bool = True
    use_onyx_file_processing: bool = True
    use_onyx_security: bool = True
    
    # Performance Settings
    max_workers: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Video Generation Settings
    default_quality: str = "medium"
    default_duration: int = 60
    default_output_format: str = "mp4"
    max_duration: int = 600
    min_duration: int = 5
    
    # LLM Settings
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    llm_timeout: int = 60
    llm_retry_attempts: int = 3
    
    # Plugin Settings
    plugin_timeout: int = 60
    plugin_max_workers: int = 5
    plugin_retry_attempts: int = 2
    enable_plugins: bool = True
    plugin_directories: List[str] = field(default_factory=lambda: [
        "plugins",
        "custom_plugins"
    ])
    
    # Security Settings
    max_input_length: int = 10000
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "gif", "mp4", "avi", "mov"
    ])
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    # Monitoring Settings
    enable_metrics: bool = True
    enable_telemetry: bool = True
    metrics_interval: int = 60
    telemetry_endpoint: Optional[str] = None
    
    # Storage Settings
    output_directory: str = "output"
    temp_directory: str = "temp"
    cache_directory: str = "cache"
    
    # Development Settings
    debug_mode: bool = False
    development_mode: bool = False
    test_mode: bool = False
    
    # Environment Settings
    environment: str = "production"
    log_level: str = "INFO"
    
    def __post_init__(self) -> Any:
        """Post-initialization processing."""
        # Validate configuration
        self._validate_config()
        
        # Set environment-specific defaults
        self._set_environment_defaults()
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        
        if self.default_quality not in ["low", "medium", "high", "ultra"]:
            raise ValueError("default_quality must be one of: low, medium, high, ultra")
        
        if self.default_duration < self.min_duration or self.default_duration > self.max_duration:
            raise ValueError(f"default_duration must be between {self.min_duration} and {self.max_duration}")
    
    def _set_environment_defaults(self) -> None:
        """Set environment-specific defaults."""
        if self.environment == "development":
            self.debug_mode = True
            self.log_level = "DEBUG"
            self.enable_telemetry = False
        
        elif self.environment == "testing":
            self.test_mode = True
            self.log_level = "DEBUG"
            self.enable_telemetry = False
            self.timeout_seconds = 30
        
        elif self.environment == "production":
            self.debug_mode = False
            self.log_level = "INFO"
            self.enable_telemetry = True


class OnyxConfigManager:
    """
    Onyx Configuration Manager.
    
    Manages configuration loading, validation, and integration with
    Onyx's configuration system.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_config_manager")
        self.config: Optional[OnyxAIVideoConfig] = None
        self.onyx_config: Optional[Dict[str, Any]] = None
        self.config_file: Optional[Path] = None
    
    def load_config(self, config_path: Optional[str] = None) -> OnyxAIVideoConfig:
        """Load configuration from file and environment."""
        try:
            # Load Onyx configuration
            self.onyx_config = get_onyx_config()
            
            # Determine config file path
            if config_path:
                self.config_file = Path(config_path)
            else:
                self.config_file = self._find_config_file()
            
            # Load AI Video configuration
            ai_video_config = self._load_ai_video_config()
            
            # Merge with environment variables
            ai_video_config = self._merge_environment_config(ai_video_config)
            
            # Merge with Onyx configuration
            ai_video_config = self._merge_onyx_config(ai_video_config)
            
            # Create configuration object
            self.config = OnyxAIVideoConfig(**ai_video_config)
            
            self.logger.info(f"Configuration loaded from: {self.config_file}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise
    
    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations."""
        config_locations = [
            Path("config/onyx_ai_video.json"),
            Path("onyx_ai_video.json"),
            Path("config/ai_video.json"),
            Path("ai_video.json"),
            Path.home() / ".onyx_ai_video.json"
        ]
        
        for location in config_locations:
            if location.exists():
                return location
        
        # Return default location
        return Path("config/onyx_ai_video.json")
    
    def _load_ai_video_config(self) -> Dict[str, Any]:
        """Load AI Video configuration from file."""
        if not self.config_file.exists():
            self.logger.warning(f"Configuration file not found: {self.config_file}")
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config = json.load(f)
            
            self.logger.info(f"Configuration loaded from file: {self.config_file}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            return {}
    
    def _merge_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with environment variables."""
        env_mappings = {
            "AI_VIDEO_USE_ONYX_LOGGING": "use_onyx_logging",
            "AI_VIDEO_USE_ONYX_LLM": "use_onyx_llm",
            "AI_VIDEO_USE_ONYX_TELEMETRY": "use_onyx_telemetry",
            "AI_VIDEO_USE_ONYX_ENCRYPTION": "use_onyx_encryption",
            "AI_VIDEO_USE_ONYX_THREADING": "use_onyx_threading",
            "AI_VIDEO_USE_ONYX_RETRY": "use_onyx_retry",
            "AI_VIDEO_USE_ONYX_GPU": "use_onyx_gpu",
            "AI_VIDEO_MAX_WORKERS": "max_workers",
            "AI_VIDEO_TIMEOUT": "timeout_seconds",
            "AI_VIDEO_RETRY_ATTEMPTS": "retry_attempts",
            "AI_VIDEO_DEFAULT_QUALITY": "default_quality",
            "AI_VIDEO_DEFAULT_DURATION": "default_duration",
            "AI_VIDEO_DEFAULT_OUTPUT_FORMAT": "default_output_format",
            "AI_VIDEO_LLM_TEMPERATURE": "llm_temperature",
            "AI_VIDEO_LLM_MAX_TOKENS": "llm_max_tokens",
            "AI_VIDEO_LLM_TIMEOUT": "llm_timeout",
            "AI_VIDEO_PLUGIN_TIMEOUT": "plugin_timeout",
            "AI_VIDEO_PLUGIN_MAX_WORKERS": "plugin_max_workers",
            "AI_VIDEO_MAX_INPUT_LENGTH": "max_input_length",
            "AI_VIDEO_MAX_FILE_SIZE": "max_file_size",
            "AI_VIDEO_ENABLE_METRICS": "enable_metrics",
            "AI_VIDEO_ENABLE_TELEMETRY": "enable_telemetry",
            "AI_VIDEO_OUTPUT_DIRECTORY": "output_directory",
            "AI_VIDEO_TEMP_DIRECTORY": "temp_directory",
            "AI_VIDEO_CACHE_DIRECTORY": "cache_directory",
            "AI_VIDEO_DEBUG_MODE": "debug_mode",
            "AI_VIDEO_DEVELOPMENT_MODE": "development_mode",
            "AI_VIDEO_TEST_MODE": "test_mode",
            "AI_VIDEO_ENVIRONMENT": "environment",
            "AI_VIDEO_LOG_LEVEL": "log_level"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert value to appropriate type
                if config_key in ["use_onyx_logging", "use_onyx_llm", "use_onyx_telemetry", 
                                 "use_onyx_encryption", "use_onyx_threading", "use_onyx_retry", 
                                 "use_onyx_gpu", "enable_plugins", "enable_metrics", 
                                 "enable_telemetry", "debug_mode", "development_mode", "test_mode"]:
                    config[config_key] = env_value.lower() in ["true", "1", "yes", "on"]
                elif config_key in ["max_workers", "timeout_seconds", "retry_attempts", 
                                   "cache_size", "cache_ttl", "default_duration", 
                                   "max_duration", "min_duration", "llm_max_tokens", 
                                   "llm_timeout", "llm_retry_attempts", "plugin_timeout", 
                                   "plugin_max_workers", "plugin_retry_attempts", 
                                   "max_input_length", "max_file_size", "metrics_interval"]:
                    config[config_key] = int(env_value)
                elif config_key in ["llm_temperature"]:
                    config[config_key] = float(env_value)
                else:
                    config[config_key] = env_value
        
        return config
    
    def _merge_onyx_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge with Onyx configuration."""
        if not self.onyx_config:
            return config
        
        # Map Onyx configuration to AI Video configuration
        onyx_mappings = {
            "LOG_LEVEL": "log_level",
            "ENVIRONMENT": "environment",
            "DEBUG_MODE": "debug_mode",
            "GPU_ENABLED": "use_onyx_gpu",
            "MAX_WORKERS": "max_workers",
            "TIMEOUT": "timeout_seconds",
            "RETRY_ATTEMPTS": "retry_attempts"
        }
        
        for onyx_key, ai_video_key in onyx_mappings.items():
            if onyx_key in self.onyx_config and ai_video_key not in config:
                config[ai_video_key] = self.onyx_config[onyx_key]
        
        return config
    
    def save_config(self, config: OnyxAIVideoConfig, config_path: Optional[str] = None) -> None:
        """Save configuration to file."""
        try:
            save_path = Path(config_path) if config_path else self.config_file
            
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = self._config_to_dict(config)
            
            # Save to file
            with open(save_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Configuration saving failed: {e}")
            raise
    
    def _config_to_dict(self, config: OnyxAIVideoConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        config_dict = {}
        
        for field_name, field_value in config.__dict__.items():
            if isinstance(field_value, (int, float, str, bool, list)):
                config_dict[field_name] = field_value
            elif isinstance(field_value, datetime):
                config_dict[field_name] = field_value.isoformat()
            else:
                config_dict[field_name] = str(field_value)
        
        return config_dict
    
    def validate_config(self, config: OnyxAIVideoConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required directories
        directories = [
            config.output_directory,
            config.temp_directory,
            config.cache_directory
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {directory}: {e}")
        
        # Check file permissions
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists():
                if not os.access(dir_path, os.W_OK):
                    issues.append(f"No write permission for directory: {directory}")
        
        # Validate numeric values
        if config.max_workers <= 0:
            issues.append("max_workers must be positive")
        
        if config.timeout_seconds <= 0:
            issues.append("timeout_seconds must be positive")
        
        if config.retry_attempts < 0:
            issues.append("retry_attempts must be non-negative")
        
        # Validate quality settings
        valid_qualities = ["low", "medium", "high", "ultra"]
        if config.default_quality not in valid_qualities:
            issues.append(f"default_quality must be one of: {valid_qualities}")
        
        # Validate duration settings
        if config.default_duration < config.min_duration:
            issues.append(f"default_duration ({config.default_duration}) cannot be less than min_duration ({config.min_duration})")
        
        if config.default_duration > config.max_duration:
            issues.append(f"default_duration ({config.default_duration}) cannot be greater than max_duration ({config.max_duration})")
        
        return issues
    
    def get_config_summary(self, config: OnyxAIVideoConfig) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "environment": config.environment,
            "debug_mode": config.debug_mode,
            "onyx_integration": {
                "logging": config.use_onyx_logging,
                "llm": config.use_onyx_llm,
                "telemetry": config.use_onyx_telemetry,
                "encryption": config.use_onyx_encryption,
                "threading": config.use_onyx_threading,
                "retry": config.use_onyx_retry,
                "gpu": config.use_onyx_gpu
            },
            "performance": {
                "max_workers": config.max_workers,
                "timeout_seconds": config.timeout_seconds,
                "retry_attempts": config.retry_attempts,
                "cache_size": config.cache_size
            },
            "video_generation": {
                "default_quality": config.default_quality,
                "default_duration": config.default_duration,
                "default_output_format": config.default_output_format,
                "max_duration": config.max_duration,
                "min_duration": config.min_duration
            },
            "llm": {
                "temperature": config.llm_temperature,
                "max_tokens": config.llm_max_tokens,
                "timeout": config.llm_timeout,
                "retry_attempts": config.llm_retry_attempts
            },
            "plugins": {
                "enabled": config.enable_plugins,
                "timeout": config.plugin_timeout,
                "max_workers": config.plugin_max_workers,
                "retry_attempts": config.plugin_retry_attempts
            },
            "security": {
                "max_input_length": config.max_input_length,
                "max_file_size": config.max_file_size,
                "allowed_extensions": config.allowed_file_extensions
            },
            "monitoring": {
                "enable_metrics": config.enable_metrics,
                "enable_telemetry": config.enable_telemetry,
                "metrics_interval": config.metrics_interval
            },
            "storage": {
                "output_directory": config.output_directory,
                "temp_directory": config.temp_directory,
                "cache_directory": config.cache_directory
            }
        }


# Global configuration manager
config_manager = OnyxConfigManager()


def get_config(config_path: Optional[str] = None) -> OnyxAIVideoConfig:
    """Get Onyx AI Video configuration."""
    if config_manager.config is None:
        config_manager.load_config(config_path)
    
    return config_manager.config


def save_config(config: OnyxAIVideoConfig, config_path: Optional[str] = None) -> None:
    """Save Onyx AI Video configuration."""
    config_manager.save_config(config, config_path)


def validate_config(config: OnyxAIVideoConfig) -> List[str]:
    """Validate Onyx AI Video configuration."""
    return config_manager.validate_config(config)


def get_config_summary(config: OnyxAIVideoConfig) -> Dict[str, Any]:
    """Get Onyx AI Video configuration summary."""
    return config_manager.get_config_summary(config)


# Default configuration
DEFAULT_CONFIG = {
    "use_onyx_logging": True,
    "use_onyx_llm": True,
    "use_onyx_telemetry": True,
    "use_onyx_encryption": True,
    "use_onyx_threading": True,
    "use_onyx_retry": True,
    "use_onyx_gpu": True,
    "use_onyx_file_processing": True,
    "use_onyx_security": True,
    "max_workers": 10,
    "timeout_seconds": 300,
    "retry_attempts": 3,
    "cache_size": 1000,
    "cache_ttl": 3600,
    "default_quality": "medium",
    "default_duration": 60,
    "default_output_format": "mp4",
    "max_duration": 600,
    "min_duration": 5,
    "llm_temperature": 0.7,
    "llm_max_tokens": 2000,
    "llm_timeout": 60,
    "llm_retry_attempts": 3,
    "plugin_timeout": 60,
    "plugin_max_workers": 5,
    "plugin_retry_attempts": 2,
    "enable_plugins": True,
    "plugin_directories": ["plugins", "custom_plugins"],
    "max_input_length": 10000,
    "allowed_file_extensions": ["jpg", "jpeg", "png", "gif", "mp4", "avi", "mov"],
    "max_file_size": 100 * 1024 * 1024,
    "enable_metrics": True,
    "enable_telemetry": True,
    "metrics_interval": 60,
    "telemetry_endpoint": None,
    "output_directory": "output",
    "temp_directory": "temp",
    "cache_directory": "cache",
    "debug_mode": False,
    "development_mode": False,
    "test_mode": False,
    "environment": "production",
    "log_level": "INFO"
}


def create_default_config(config_path: str = "config/onyx_ai_video.json") -> None:
    """Create default configuration file."""
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(DEFAULT_CONFIG, f, indent=2)
        
        logger.info(f"Default configuration created: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to create default configuration: {e}")
        raise 