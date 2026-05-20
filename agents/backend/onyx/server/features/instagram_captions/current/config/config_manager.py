"""
Configuration Manager for Instagram Captions API v10.0

Centralized configuration management with environment support.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    api_key_length: int = 32
    api_key_complexity: str = "high"
    password_hash_algorithm: str = "sha256"
    encryption_algorithm: str = "fernet"
    threat_detection_enabled: bool = True
    encryption_enabled: bool = True
    max_login_attempts: int = 5
    session_timeout_minutes: int = 60

@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""
    performance_monitoring_enabled: bool = True
    health_checks_enabled: bool = True
    metrics_collection_enabled: bool = True
    max_metrics_history: int = 10000
    health_check_interval_seconds: int = 30
    metrics_retention_hours: int = 24
    alerting_enabled: bool = True

@dataclass
class ResilienceConfig:
    """Resilience configuration settings."""
    circuit_breaker_enabled: bool = True
    error_handling_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    success_threshold: int = 3
    max_error_history: int = 1000

@dataclass
class PerformanceConfig:
    """Performance configuration settings."""
    caching_enabled: bool = True
    rate_limiting_enabled: bool = True
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 300
    max_requests_per_minute: int = 100
    request_timeout_seconds: int = 30

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    database_url: str = "sqlite:///instagram_captions.db"
    connection_pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = True
    workers: int = 1
    log_level: str = "INFO"
    cors_enabled: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])

class ConfigManager:
    """Centralized configuration management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config_data: Dict[str, Any] = {}
        
        # Initialize default configurations
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.resilience = ResilienceConfig()
        self.performance = PerformanceConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        
        # Load configuration
        self.load_config()
        self.load_environment_variables()
    
    def load_config(self):
        """Load configuration from file."""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                if config_file.suffix.lower() == '.yaml':
                    with open(config_file, 'r', encoding='utf-8') as f:
                        self.config_data = yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == '.json':
                    with open(config_file, 'r', encoding='utf-8') as f:
                        self.config_data = json.load(f)
                else:
                    print(f"Unsupported config file format: {config_file.suffix}")
                    return
                
                # Apply configuration to dataclasses
                self._apply_config()
                
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
    
    def load_environment_variables(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Security
            'SECURITY_API_KEY_LENGTH': ('security', 'api_key_length', int),
            'SECURITY_PASSWORD_HASH_ALGORITHM': ('security', 'password_hash_algorithm', str),
            'SECURITY_THREAT_DETECTION_ENABLED': ('security', 'threat_detection_enabled', bool),
            
            # Monitoring
            'MONITORING_ENABLED': ('monitoring', 'performance_monitoring_enabled', bool),
            'MONITORING_HEALTH_CHECKS_ENABLED': ('monitoring', 'health_checks_enabled', bool),
            'MONITORING_METRICS_ENABLED': ('monitoring', 'metrics_collection_enabled', bool),
            
            # Resilience
            'RESILIENCE_CIRCUIT_BREAKER_ENABLED': ('resilience', 'circuit_breaker_enabled', bool),
            'RESILIENCE_FAILURE_THRESHOLD': ('resilience', 'failure_threshold', int),
            'RESILIENCE_RECOVERY_TIMEOUT': ('resilience', 'recovery_timeout_seconds', int),
            
            # Performance
            'PERFORMANCE_CACHING_ENABLED': ('performance', 'caching_enabled', bool),
            'PERFORMANCE_RATE_LIMITING_ENABLED': ('performance', 'rate_limiting_enabled', bool),
            'PERFORMANCE_MAX_CACHE_SIZE': ('performance', 'max_cache_size', int),
            
            # Database
            'DATABASE_URL': ('database', 'database_url', str),
            'DATABASE_POOL_SIZE': ('database', 'connection_pool_size', int),
            
            # API
            'API_HOST': ('api', 'host', str),
            'API_PORT': ('api', 'port', int),
            'API_DEBUG': ('api', 'debug', bool),
            'API_LOG_LEVEL': ('api', 'log_level', str),
        }
        
        for env_var, (config_section, config_key, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        # Handle boolean values
                        if env_value.lower() in ('true', '1', 'yes', 'on'):
                            setattr(getattr(self, config_section), config_key, True)
                        elif env_value.lower() in ('false', '0', 'no', 'off'):
                            setattr(getattr(self, config_section), config_key, False)
                    else:
                        setattr(getattr(self, config_section), config_key, value_type(env_value))
                except (ValueError, TypeError) as e:
                    print(f"Error setting {env_var}={env_value}: {e}")
    
    def _apply_config(self):
        """Apply loaded configuration to dataclass instances."""
        for section_name, section_config in self.config_data.items():
            if hasattr(self, section_name):
                section_instance = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section_instance, key):
                        setattr(section_instance, key, value)
    
    def get_config(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if hasattr(self, section):
            section_instance = getattr(self, section)
            if hasattr(section_instance, key):
                return getattr(section_instance, key)
        return default
    
    def set_config(self, section: str, key: str, value: Any):
        """Set configuration value."""
        if hasattr(self, section):
            section_instance = getattr(self, section)
            if hasattr(section_instance, key):
                setattr(section_instance, key, value)
                return True
        return False
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        config = {}
        for section_name in ['security', 'monitoring', 'resilience', 'performance', 'database', 'api']:
            if hasattr(self, section_name):
                section_instance = getattr(self, section_name)
                config[section_name] = {
                    key: getattr(section_instance, key)
                    for key in section_instance.__dataclass_fields__.keys()
                }
        return config
    
    def save_config(self, format: str = "yaml"):
        """Save current configuration to file."""
        config_data = self.get_all_config()
        
        try:
            if format.lower() == "yaml":
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(self.config_path.replace('.yaml', '.json'), 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate security config
        if self.security.api_key_length < 16:
            validation_results['errors'].append("API key length must be at least 16 characters")
            validation_results['valid'] = False
        
        if self.security.max_login_attempts < 1:
            validation_results['errors'].append("Max login attempts must be at least 1")
            validation_results['valid'] = False
        
        # Validate monitoring config
        if self.monitoring.health_check_interval_seconds < 5:
            validation_results['warnings'].append("Health check interval is very short (< 5 seconds)")
        
        if self.monitoring.max_metrics_history > 100000:
            validation_results['warnings'].append("Very large metrics history may impact memory usage")
        
        # Validate resilience config
        if self.resilience.failure_threshold < 1:
            validation_results['errors'].append("Failure threshold must be at least 1")
            validation_results['valid'] = False
        
        if self.resilience.recovery_timeout_seconds < 10:
            validation_results['warnings'].append("Recovery timeout is very short (< 10 seconds)")
        
        # Validate performance config
        if self.performance.max_cache_size < 10:
            validation_results['warnings'].append("Cache size is very small (< 10 items)")
        
        if self.performance.max_requests_per_minute < 1:
            validation_results['errors'].append("Max requests per minute must be at least 1")
            validation_results['valid'] = False
        
        # Validate API config
        if self.api.port < 1 or self.api.port > 65535:
            validation_results['errors'].append("Port must be between 1 and 65535")
            validation_results['valid'] = False
        
        if self.api.workers < 1:
            validation_results['errors'].append("Number of workers must be at least 1")
            validation_results['valid'] = False
        
        return validation_results
    
    def create_default_config(self):
        """Create default configuration file."""
        default_config = self.get_all_config()
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            print(f"Default configuration created: {self.config_path}")
            return True
        except Exception as e:
            print(f"Error creating default config: {e}")
            return False






