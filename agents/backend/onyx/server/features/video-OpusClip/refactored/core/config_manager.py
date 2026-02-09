"""
Configuration Manager

Centralized configuration management for the Ultimate Opus Clip system.
Handles environment variables, configuration files, and runtime settings.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import structlog
from typing import TypeVar, Generic

logger = structlog.get_logger("config_manager")

T = TypeVar('T')

class ConfigSource(Enum):
    """Configuration source types."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DEFAULT = "default"
    RUNTIME = "runtime"

@dataclass
class ConfigValue(Generic[T]):
    """Configuration value with metadata."""
    value: T
    source: ConfigSource
    description: str = ""
    required: bool = False
    validation_func: Optional[callable] = None

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        self.config_values: Dict[str, ConfigValue] = {}
        self.logger = structlog.get_logger("config_manager")
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from all sources."""
        try:
            # Load from file if provided
            if self.config_file and Path(self.config_file).exists():
                self._load_from_file(self.config_file)
            
            # Load from environment variables
            self._load_from_environment()
            
            # Set default values
            self._set_defaults()
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise
    
    def _load_from_file(self, file_path: str):
        """Load configuration from file."""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
                with open(file_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    file_config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
            
            self._merge_config(file_config, ConfigSource.FILE)
            self.logger.info(f"Configuration loaded from file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config from file {file_path}: {e}")
            raise
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        try:
            env_config = {}
            
            # Map environment variables to config keys
            env_mappings = {
                'OPUS_CLIP_LOG_LEVEL': 'logging.level',
                'OPUS_CLIP_DEBUG': 'debug.enabled',
                'OPUS_CLIP_MAX_WORKERS': 'processing.max_workers',
                'OPUS_CLIP_CACHE_ENABLED': 'cache.enabled',
                'OPUS_CLIP_CACHE_TTL': 'cache.ttl',
                'OPUS_CLIP_DATABASE_URL': 'database.url',
                'OPUS_CLIP_REDIS_URL': 'redis.url',
                'OPUS_CLIP_API_KEY': 'api.key',
                'OPUS_CLIP_SECRET_KEY': 'api.secret_key',
                'OPUS_CLIP_OPENAI_API_KEY': 'openai.api_key',
                'OPUS_CLIP_STABILITY_API_KEY': 'stability.api_key',
                'OPUS_CLIP_ELEVENLABS_API_KEY': 'elevenlabs.api_key',
                'OPUS_CLIP_YOUTUBE_API_KEY': 'youtube.api_key',
                'OPUS_CLIP_TIKTOK_API_KEY': 'tiktok.api_key',
                'OPUS_CLIP_INSTAGRAM_API_KEY': 'instagram.api_key',
                'OPUS_CLIP_TWITTER_API_KEY': 'twitter.api_key',
                'OPUS_CLIP_FACEBOOK_API_KEY': 'facebook.api_key',
                'OPUS_CLIP_GOOGLE_TRENDS_API_KEY': 'google_trends.api_key',
                'OPUS_CLIP_AWS_ACCESS_KEY': 'aws.access_key',
                'OPUS_CLIP_AWS_SECRET_KEY': 'aws.secret_key',
                'OPUS_CLIP_AWS_REGION': 'aws.region',
                'OPUS_CLIP_AZURE_ACCOUNT_NAME': 'azure.account_name',
                'OPUS_CLIP_AZURE_ACCOUNT_KEY': 'azure.account_key',
                'OPUS_CLIP_GCP_PROJECT_ID': 'gcp.project_id',
                'OPUS_CLIP_GCP_CREDENTIALS': 'gcp.credentials',
            }
            
            for env_var, config_key in env_mappings.items():
                value = os.getenv(env_var)
                if value is not None:
                    self._set_nested_value(env_config, config_key, value)
            
            self._merge_config(env_config, ConfigSource.ENVIRONMENT)
            self.logger.info("Configuration loaded from environment variables")
            
        except Exception as e:
            self.logger.error(f"Failed to load config from environment: {e}")
            raise
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            'logging': {
                'level': 'INFO',
                'format': 'json',
                'file': None
            },
            'debug': {
                'enabled': False,
                'profiling': False
            },
            'processing': {
                'max_workers': 4,
                'timeout': 300.0,
                'retry_attempts': 3,
                'queue_size': 1000
            },
            'cache': {
                'enabled': True,
                'ttl': 3600.0,
                'max_size': 1000,
                'backend': 'memory'
            },
            'database': {
                'url': 'sqlite:///opus_clip.db',
                'pool_size': 10,
                'max_overflow': 20
            },
            'redis': {
                'url': 'redis://localhost:6379',
                'db': 0,
                'password': None
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1,
                'key': None,
                'secret_key': None,
                'cors_origins': ['*']
            },
            'external_apis': {
                'openai': {
                    'api_key': None,
                    'base_url': 'https://api.openai.com/v1',
                    'timeout': 30.0
                },
                'stability': {
                    'api_key': None,
                    'base_url': 'https://api.stability.ai',
                    'timeout': 30.0
                },
                'elevenlabs': {
                    'api_key': None,
                    'base_url': 'https://api.elevenlabs.io',
                    'timeout': 30.0
                },
                'youtube': {
                    'api_key': None,
                    'base_url': 'https://www.googleapis.com/youtube/v3',
                    'timeout': 30.0
                },
                'tiktok': {
                    'api_key': None,
                    'base_url': 'https://open-api.tiktok.com',
                    'timeout': 30.0
                },
                'instagram': {
                    'api_key': None,
                    'base_url': 'https://graph.instagram.com',
                    'timeout': 30.0
                },
                'twitter': {
                    'api_key': None,
                    'base_url': 'https://api.twitter.com/2',
                    'timeout': 30.0
                },
                'facebook': {
                    'api_key': None,
                    'base_url': 'https://graph.facebook.com',
                    'timeout': 30.0
                },
                'google_trends': {
                    'api_key': None,
                    'base_url': 'https://trends.google.com',
                    'timeout': 30.0
                }
            },
            'cloud_storage': {
                'aws': {
                    'access_key': None,
                    'secret_key': None,
                    'region': 'us-east-1',
                    'bucket': 'opus-clip-storage'
                },
                'azure': {
                    'account_name': None,
                    'account_key': None,
                    'container': 'opus-clip-storage'
                },
                'gcp': {
                    'project_id': None,
                    'credentials': None,
                    'bucket': 'opus-clip-storage'
                }
            },
            'features': {
                'content_curation': {
                    'enabled': True,
                    'max_clips': 10,
                    'min_duration': 5.0,
                    'max_duration': 60.0,
                    'engagement_threshold': 0.6
                },
                'speaker_tracking': {
                    'enabled': True,
                    'confidence_threshold': 0.7,
                    'tracking_threshold': 0.6,
                    'max_tracking_distance': 100
                },
                'broll_integration': {
                    'enabled': True,
                    'max_suggestions': 5,
                    'confidence_threshold': 0.7,
                    'enable_ai_generation': True,
                    'enable_stock_footage': True
                },
                'viral_scoring': {
                    'enabled': True,
                    'trend_weight': 0.15,
                    'engagement_weight': 0.25,
                    'novelty_weight': 0.15,
                    'audience_weight': 0.15,
                    'timing_weight': 0.05,
                    'shareability_weight': 0.03,
                    'controversy_weight': 0.02
                },
                'audio_processing': {
                    'enabled': True,
                    'enhancement_level': 'balanced',
                    'add_background_music': True,
                    'add_sound_effects': True,
                    'noise_reduction': True
                },
                'professional_export': {
                    'enabled': True,
                    'formats': ['premiere_pro', 'final_cut', 'xml', 'edl'],
                    'quality': 'high',
                    'include_metadata': True
                },
                'analytics': {
                    'enabled': True,
                    'track_performance': True,
                    'generate_reports': True,
                    'retention_days': 30
                }
            },
            'platforms': {
                'tiktok': {
                    'aspect_ratio': [9, 16],
                    'resolution': [1080, 1920],
                    'max_duration': 15.0,
                    'min_duration': 8.0,
                    'frame_rate': 30.0
                },
                'youtube': {
                    'aspect_ratio': [16, 9],
                    'resolution': [1920, 1080],
                    'max_duration': 60.0,
                    'min_duration': 10.0,
                    'frame_rate': 30.0
                },
                'instagram': {
                    'aspect_ratio': [1, 1],
                    'resolution': [1080, 1080],
                    'max_duration': 30.0,
                    'min_duration': 5.0,
                    'frame_rate': 30.0
                },
                'twitter': {
                    'aspect_ratio': [16, 9],
                    'resolution': [1280, 720],
                    'max_duration': 30.0,
                    'min_duration': 5.0,
                    'frame_rate': 30.0
                },
                'facebook': {
                    'aspect_ratio': [16, 9],
                    'resolution': [1920, 1080],
                    'max_duration': 60.0,
                    'min_duration': 10.0,
                    'frame_rate': 30.0
                }
            }
        }
        
        self._merge_config(defaults, ConfigSource.DEFAULT)
        self.logger.info("Default configuration set")
    
    def _merge_config(self, new_config: Dict[str, Any], source: ConfigSource):
        """Merge new configuration into existing config."""
        def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            result = d1.copy()
            for key, value in d2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        self.config_data = merge_dicts(self.config_data, new_config)
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set a nested configuration value."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to get config value for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.RUNTIME):
        """Set configuration value."""
        try:
            self._set_nested_value(self.config_data, key, value)
            self.logger.info(f"Set config value: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to set config value for key {key}: {e}")
            raise
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate required values
        required_keys = [
            'api.host',
            'api.port',
            'processing.max_workers'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                errors.append(f"Required configuration key missing: {key}")
        
        # Validate API keys if features are enabled
        if self.get('features.content_curation.enabled', False):
            if not self.get('external_apis.openai.api_key'):
                errors.append("OpenAI API key required for content curation")
        
        if self.get('features.viral_scoring.enabled', False):
            if not self.get('external_apis.google_trends.api_key'):
                errors.append("Google Trends API key required for viral scoring")
        
        # Validate platform configurations
        platforms = self.get('platforms', {})
        for platform, config in platforms.items():
            if not config.get('resolution') or len(config['resolution']) != 2:
                errors.append(f"Invalid resolution for platform {platform}")
            
            if not config.get('aspect_ratio') or len(config['aspect_ratio']) != 2:
                errors.append(f"Invalid aspect ratio for platform {platform}")
        
        return errors
    
    def get_processor_config(self, processor_name: str) -> Dict[str, Any]:
        """Get configuration for a specific processor."""
        return self.get(f'features.{processor_name}', {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self.get(f'features.{feature_name}.enabled', False)
    
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get configuration for a specific platform."""
        return self.get(f'platforms.{platform}', {})
    
    def export_config(self, file_path: str, format: str = 'yaml'):
        """Export current configuration to file."""
        try:
            file_path = Path(file_path)
            
            if format.lower() == 'yaml' or format.lower() == 'yml':
                with open(file_path, 'w') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(self.config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Configuration exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            raise
    
    def reload(self):
        """Reload configuration from sources."""
        self.config_data = {}
        self._load_configuration()
        self.logger.info("Configuration reloaded")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data."""
        return self.config_data.copy()

# Global configuration instance
config_manager = ConfigManager()

# Export the manager and convenience functions
__all__ = [
    "ConfigManager",
    "ConfigValue", 
    "ConfigSource",
    "config_manager"
]


