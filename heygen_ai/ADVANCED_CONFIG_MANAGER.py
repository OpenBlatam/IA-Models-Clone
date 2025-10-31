#!/usr/bin/env python3
"""
🔧 HeyGen AI - Advanced Configuration Management System
======================================================

This module provides comprehensive configuration management for the HeyGen AI system:
- Environment-aware configuration
- Dynamic configuration updates
- Configuration validation and security
- Performance-optimized settings
- Centralized configuration management
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import secrets
import hashlib
from datetime import datetime, timedelta
import threading
import asyncio
from contextlib import asynccontextmanager
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class ConfigSecurityLevel(str, Enum):
    """Configuration security levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ENTERPRISE = "enterprise"
    MILITARY = "military"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "heygen_ai"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: str = "prefer"
    connection_timeout: int = 10

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    max_login_attempts: int = 5
    lockout_duration: int = 900
    session_timeout: int = 1800
    enable_2fa: bool = False
    encryption_key: str = ""

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_workers: int = 4
    worker_timeout: int = 300
    memory_limit_mb: int = 4096
    cpu_limit_percent: int = 80
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_compression: bool = True
    compression_level: int = 6
    enable_gzip: bool = True
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30

@dataclass
class AIModelConfig:
    """AI Model configuration"""
    model_cache_size: int = 100
    model_timeout: int = 300
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    enable_model_compilation: bool = True
    enable_quantization: bool = False
    enable_pruning: bool = False
    batch_size: int = 32
    max_sequence_length: int = 512

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    metrics_interval: int = 60
    enable_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_profiling: bool = False
    profiling_interval: int = 300
    enable_alerting: bool = True
    alert_threshold_cpu: float = 80.0
    alert_threshold_memory: float = 85.0
    alert_threshold_disk: float = 90.0

@dataclass
class HeyGenAIConfig:
    """Main HeyGen AI configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    app_name: str = "HeyGen AI"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ai_models: AIModelConfig = field(default_factory=AIModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    config_hash: str = ""

class ConfigurationValidator:
    """Configuration validation system"""
    
    def __init__(self, security_level: ConfigSecurityLevel = ConfigSecurityLevel.ENTERPRISE):
        self.security_level = security_level
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration validation rules"""
        return {
            'database': {
                'host': {'type': str, 'required': True, 'min_length': 1},
                'port': {'type': int, 'required': True, 'min': 1, 'max': 65535},
                'database': {'type': str, 'required': True, 'min_length': 1},
                'username': {'type': str, 'required': True, 'min_length': 1},
                'password': {'type': str, 'required': True, 'min_length': 8},
                'pool_size': {'type': int, 'min': 1, 'max': 100},
                'max_overflow': {'type': int, 'min': 0, 'max': 200}
            },
            'security': {
                'secret_key': {'type': str, 'required': True, 'min_length': 32},
                'jwt_secret': {'type': str, 'required': True, 'min_length': 32},
                'password_min_length': {'type': int, 'min': 8, 'max': 128},
                'max_login_attempts': {'type': int, 'min': 1, 'max': 10},
                'lockout_duration': {'type': int, 'min': 60, 'max': 3600}
            },
            'performance': {
                'max_workers': {'type': int, 'min': 1, 'max': 32},
                'memory_limit_mb': {'type': int, 'min': 512, 'max': 32768},
                'cpu_limit_percent': {'type': int, 'min': 10, 'max': 100},
                'max_request_size': {'type': int, 'min': 1024, 'max': 100 * 1024 * 1024}
            }
        }
    
    def validate_config(self, config: HeyGenAIConfig) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # Validate database config
        db_errors = self._validate_section(config.database, 'database')
        errors.extend(db_errors)
        
        # Validate security config
        security_errors = self._validate_section(config.security, 'security')
        errors.extend(security_errors)
        
        # Validate performance config
        perf_errors = self._validate_section(config.performance, 'performance')
        errors.extend(perf_errors)
        
        # Additional security validations
        if self.security_level in [ConfigSecurityLevel.ENTERPRISE, ConfigSecurityLevel.MILITARY]:
            security_errors = self._validate_security_enhanced(config)
            errors.extend(security_errors)
        
        return len(errors) == 0, errors
    
    def _validate_section(self, section_config: Any, section_name: str) -> List[str]:
        """Validate a configuration section"""
        errors = []
        rules = self.validation_rules.get(section_name, {})
        
        for field_name, rules in rules.items():
            if not hasattr(section_config, field_name):
                errors.append(f"Missing field: {section_name}.{field_name}")
                continue
            
            value = getattr(section_config, field_name)
            
            # Type validation
            if 'type' in rules and not isinstance(value, rules['type']):
                errors.append(f"Invalid type for {section_name}.{field_name}: expected {rules['type'].__name__}")
                continue
            
            # Required validation
            if rules.get('required', False) and not value:
                errors.append(f"Required field {section_name}.{field_name} is empty")
                continue
            
            # String length validation
            if isinstance(value, str):
                if 'min_length' in rules and len(value) < rules['min_length']:
                    errors.append(f"{section_name}.{field_name} too short: minimum {rules['min_length']} characters")
                if 'max_length' in rules and len(value) > rules['max_length']:
                    errors.append(f"{section_name}.{field_name} too long: maximum {rules['max_length']} characters")
            
            # Numeric range validation
            if isinstance(value, (int, float)):
                if 'min' in rules and value < rules['min']:
                    errors.append(f"{section_name}.{field_name} too small: minimum {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    errors.append(f"{section_name}.{field_name} too large: maximum {rules['max']}")
        
        return errors
    
    def _validate_security_enhanced(self, config: HeyGenAIConfig) -> List[str]:
        """Enhanced security validation"""
        errors = []
        
        # Validate secret keys are not default
        if config.security.secret_key in ['', 'default', 'secret']:
            errors.append("Security secret_key must be set to a secure value")
        
        if config.security.jwt_secret in ['', 'default', 'secret']:
            errors.append("JWT secret must be set to a secure value")
        
        # Validate password strength requirements
        if config.security.password_min_length < 12:
            errors.append("Password minimum length should be at least 12 characters for enhanced security")
        
        # Validate CORS configuration
        if '*' in config.cors_origins and config.environment == Environment.PRODUCTION:
            errors.append("CORS origins should not be '*' in production environment")
        
        return errors

class ConfigurationManager:
    """Advanced configuration management system"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 environment: Optional[Environment] = None,
                 security_level: ConfigSecurityLevel = ConfigSecurityLevel.ENTERPRISE):
        self.config_path = config_path or self._get_default_config_path()
        self.environment = environment or self._detect_environment()
        self.security_level = security_level
        self.validator = ConfigurationValidator(security_level)
        self.config: Optional[HeyGenAIConfig] = None
        self._lock = threading.RLock()
        self._watchers = []
        self._last_modified = None
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        base_path = Path(__file__).parent
        return str(base_path / "config" / "heygen_ai_config.yaml")
    
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env = os.getenv('HEYGEN_ENV', '').lower()
        if env in ['dev', 'development']:
            return Environment.DEVELOPMENT
        elif env in ['staging', 'stage']:
            return Environment.STAGING
        elif env in ['prod', 'production']:
            return Environment.PRODUCTION
        elif env in ['test', 'testing']:
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT
    
    def load_config(self, config_path: Optional[str] = None) -> HeyGenAIConfig:
        """Load configuration from file"""
        config_path = config_path or self.config_path
        
        with self._lock:
            try:
                # Check if file exists
                if not os.path.exists(config_path):
                    logger.warning(f"Configuration file not found: {config_path}")
                    return self._create_default_config()
                
                # Load configuration based on file extension
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = self._load_yaml_config(config_path)
                elif config_path.endswith('.json'):
                    config_data = self._load_json_config(config_path)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path}")
                
                # Create configuration object
                config = self._create_config_from_data(config_data)
                
                # Validate configuration
                is_valid, errors = self.validator.validate_config(config)
                if not is_valid:
                    logger.error(f"Configuration validation failed: {errors}")
                    raise ValueError(f"Invalid configuration: {errors}")
                
                # Generate configuration hash
                config.config_hash = self._generate_config_hash(config)
                
                # Update metadata
                config.updated_at = datetime.now()
                
                self.config = config
                self._last_modified = os.path.getmtime(config_path)
                
                logger.info(f"Configuration loaded successfully from {config_path}")
                return config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                return self._create_default_config()
    
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _load_json_config(self, config_path: str) -> Dict[str, Any]:
        """Load JSON configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_config_from_data(self, data: Dict[str, Any]) -> HeyGenAIConfig:
        """Create configuration object from data"""
        # Set environment
        if 'environment' in data:
            data['environment'] = Environment(data['environment'])
        
        # Create sub-configurations
        database_data = data.get('database', {})
        database_config = DatabaseConfig(**database_data)
        
        redis_data = data.get('redis', {})
        redis_config = RedisConfig(**redis_data)
        
        security_data = data.get('security', {})
        security_config = SecurityConfig(**security_data)
        
        performance_data = data.get('performance', {})
        performance_config = PerformanceConfig(**performance_data)
        
        ai_models_data = data.get('ai_models', {})
        ai_models_config = AIModelConfig(**ai_models_data)
        
        monitoring_data = data.get('monitoring', {})
        monitoring_config = MonitoringConfig(**monitoring_data)
        
        # Create main configuration
        config = HeyGenAIConfig(
            environment=data.get('environment', self.environment),
            debug=data.get('debug', False),
            app_name=data.get('app_name', 'HeyGen AI'),
            version=data.get('version', '1.0.0'),
            api_prefix=data.get('api_prefix', '/api/v1'),
            cors_origins=data.get('cors_origins', ['*']),
            cors_methods=data.get('cors_methods', ['GET', 'POST', 'PUT', 'DELETE']),
            cors_headers=data.get('cors_headers', ['*']),
            database=database_config,
            redis=redis_config,
            security=security_config,
            performance=performance_config,
            ai_models=ai_models_config,
            monitoring=monitoring_config
        )
        
        return config
    
    def _create_default_config(self) -> HeyGenAIConfig:
        """Create default configuration"""
        logger.info("Creating default configuration")
        
        # Generate secure secrets
        secret_key = secrets.token_urlsafe(32)
        jwt_secret = secrets.token_urlsafe(32)
        encryption_key = secrets.token_urlsafe(32)
        
        config = HeyGenAIConfig(
            environment=self.environment,
            debug=self.environment == Environment.DEVELOPMENT,
            security=SecurityConfig(
                secret_key=secret_key,
                jwt_secret=jwt_secret,
                encryption_key=encryption_key
            )
        )
        
        self.config = config
        return config
    
    def _generate_config_hash(self, config: HeyGenAIConfig) -> str:
        """Generate configuration hash for change detection"""
        config_str = json.dumps(config.__dict__, default=str, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def save_config(self, config: HeyGenAIConfig, config_path: Optional[str] = None) -> bool:
        """Save configuration to file"""
        config_path = config_path or self.config_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Convert to dictionary
            config_dict = self._config_to_dict(config)
            
            # Save based on file extension
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self._save_yaml_config(config_dict, config_path)
            elif config_path.endswith('.json'):
                self._save_json_config(config_dict, config_path)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path}")
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _config_to_dict(self, config: HeyGenAIConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        config_dict = {}
        
        # Main configuration
        for field in config.__dataclass_fields__:
            if field in ['database', 'redis', 'security', 'performance', 'ai_models', 'monitoring']:
                continue
            value = getattr(config, field)
            if isinstance(value, Enum):
                config_dict[field] = value.value
            else:
                config_dict[field] = value
        
        # Sub-configurations
        config_dict['database'] = self._dataclass_to_dict(config.database)
        config_dict['redis'] = self._dataclass_to_dict(config.redis)
        config_dict['security'] = self._dataclass_to_dict(config.security)
        config_dict['performance'] = self._dataclass_to_dict(config.performance)
        config_dict['ai_models'] = self._dataclass_to_dict(config.ai_models)
        config_dict['monitoring'] = self._dataclass_to_dict(config.monitoring)
        
        return config_dict
    
    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        result = {}
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            if isinstance(value, Enum):
                result[field] = value.value
            else:
                result[field] = value
        return result
    
    def _save_yaml_config(self, config_dict: Dict[str, Any], config_path: str):
        """Save YAML configuration"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _save_json_config(self, config_dict: Dict[str, Any], config_path: str):
        """Save JSON configuration"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def get_config(self) -> Optional[HeyGenAIConfig]:
        """Get current configuration"""
        with self._lock:
            return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        if not self.config:
            logger.error("No configuration loaded")
            return False
        
        try:
            with self._lock:
                # Apply updates
                for key, value in updates.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {key}")
                
                # Update metadata
                self.config.updated_at = datetime.now()
                self.config.config_hash = self._generate_config_hash(self.config)
                
                logger.info("Configuration updated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def watch_config_changes(self, callback):
        """Watch for configuration file changes"""
        self._watchers.append(callback)
    
    def check_for_changes(self) -> bool:
        """Check if configuration file has changed"""
        if not os.path.exists(self.config_path):
            return False
        
        current_modified = os.path.getmtime(self.config_path)
        if self._last_modified and current_modified > self._last_modified:
            return True
        
        return False
    
    def reload_config(self) -> bool:
        """Reload configuration if changed"""
        if self.check_for_changes():
            logger.info("Configuration file changed, reloading...")
            self.load_config()
            
            # Notify watchers
            for callback in self._watchers:
                try:
                    callback(self.config)
                except Exception as e:
                    logger.error(f"Error in config watcher: {e}")
            
            return True
        
        return False
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        if not self.config:
            return {}
        
        env_config = {
            'environment': self.config.environment.value,
            'debug': self.config.debug,
            'app_name': self.config.app_name,
            'version': self.config.version
        }
        
        # Add environment-specific overrides
        if self.config.environment == Environment.PRODUCTION:
            env_config.update({
                'debug': False,
                'cors_origins': ['https://yourdomain.com'],
                'performance': {
                    'max_workers': 8,
                    'memory_limit_mb': 8192
                }
            })
        elif self.config.environment == Environment.DEVELOPMENT:
            env_config.update({
                'debug': True,
                'cors_origins': ['*'],
                'performance': {
                    'max_workers': 2,
                    'memory_limit_mb': 2048
                }
            })
        
        return env_config

# Example usage and demonstration
def main():
    """Demonstrate the configuration management system"""
    print("🔧 HeyGen AI - Advanced Configuration Management Demo")
    print("=" * 60)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager(
        environment=Environment.DEVELOPMENT,
        security_level=ConfigSecurityLevel.ENTERPRISE
    )
    
    # Load configuration
    print("📁 Loading configuration...")
    config = config_manager.load_config()
    
    print(f"✅ Environment: {config.environment.value}")
    print(f"✅ Debug Mode: {config.debug}")
    print(f"✅ App Name: {config.app_name}")
    print(f"✅ Version: {config.version}")
    print(f"✅ Database Host: {config.database.host}")
    print(f"✅ Security Level: {config_manager.security_level.value}")
    
    # Demonstrate configuration validation
    print("\n🔍 Validating configuration...")
    is_valid, errors = config_manager.validator.validate_config(config)
    
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print(f"❌ Configuration validation failed: {errors}")
    
    # Demonstrate configuration updates
    print("\n🔄 Updating configuration...")
    updates = {
        'debug': True,
        'performance': PerformanceConfig(max_workers=8, memory_limit_mb=8192)
    }
    
    success = config_manager.update_config(updates)
    if success:
        print("✅ Configuration updated successfully")
    else:
        print("❌ Failed to update configuration")
    
    # Get environment-specific configuration
    print("\n🌍 Environment-specific configuration:")
    env_config = config_manager.get_environment_config()
    for key, value in env_config.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Configuration management demo completed")

if __name__ == "__main__":
    main()


