"""
Environment Manager for Instagram Captions API v10.0

Environment-specific configuration and deployment management.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    name: str
    description: str
    debug: bool
    log_level: str
    database_url: str
    redis_url: Optional[str]
    allowed_hosts: List[str]
    cors_origins: List[str]
    security_level: str
    monitoring_enabled: bool
    alerting_enabled: bool
    backup_enabled: bool
    ssl_enabled: bool
    rate_limiting: Dict[str, Any]
    cache_settings: Dict[str, Any]

class EnvironmentManager:
    """Environment-specific configuration and deployment management."""
    
    def __init__(self):
        self.environments = {
            'development': EnvironmentConfig(
                name='development',
                description='Development environment for local development',
                debug=True,
                log_level='DEBUG',
                database_url='sqlite:///dev_instagram_captions.db',
                redis_url=None,
                allowed_hosts=['localhost', '127.0.0.1', '0.0.0.0'],
                cors_origins=['*'],
                security_level='medium',
                monitoring_enabled=True,
                alerting_enabled=False,
                backup_enabled=False,
                ssl_enabled=False,
                rate_limiting={
                    'enabled': True,
                    'max_requests_per_minute': 1000,
                    'burst_limit': 200
                },
                cache_settings={
                    'enabled': True,
                    'max_size': 1000,
                    'ttl_seconds': 300
                }
            ),
            'staging': EnvironmentConfig(
                name='staging',
                description='Staging environment for testing',
                debug=False,
                log_level='INFO',
                database_url='postgresql://staging_user:staging_pass@staging_db:5432/instagram_captions',
                redis_url='redis://staging_redis:6379/0',
                allowed_hosts=['staging.example.com', 'staging-api.example.com'],
                cors_origins=['https://staging.example.com'],
                security_level='high',
                monitoring_enabled=True,
                alerting_enabled=True,
                backup_enabled=True,
                ssl_enabled=True,
                rate_limiting={
                    'enabled': True,
                    'max_requests_per_minute': 500,
                    'burst_limit': 100
                },
                cache_settings={
                    'enabled': True,
                    'max_size': 5000,
                    'ttl_seconds': 600
                }
            ),
            'production': EnvironmentConfig(
                name='production',
                description='Production environment for live service',
                debug=False,
                log_level='WARNING',
                database_url='postgresql://prod_user:prod_pass@prod_db:5432/instagram_captions',
                redis_url='redis://prod_redis:6379/0',
                allowed_hosts=['api.example.com', 'www.example.com'],
                cors_origins=['https://example.com', 'https://www.example.com'],
                security_level='maximum',
                monitoring_enabled=True,
                alerting_enabled=True,
                backup_enabled=True,
                ssl_enabled=True,
                rate_limiting={
                    'enabled': True,
                    'max_requests_per_minute': 100,
                    'burst_limit': 20
                },
                cache_settings={
                    'enabled': True,
                    'max_size': 10000,
                    'ttl_seconds': 1800
                }
            ),
            'testing': EnvironmentConfig(
                name='testing',
                description='Testing environment for automated tests',
                debug=True,
                log_level='DEBUG',
                database_url='sqlite:///test_instagram_captions.db',
                redis_url=None,
                allowed_hosts=['localhost', '127.0.0.1'],
                cors_origins=['*'],
                security_level='low',
                monitoring_enabled=False,
                alerting_enabled=False,
                backup_enabled=False,
                ssl_enabled=False,
                rate_limiting={
                    'enabled': False,
                    'max_requests_per_minute': 10000,
                    'burst_limit': 1000
                },
                cache_settings={
                    'enabled': False,
                    'max_size': 100,
                    'ttl_seconds': 60
                }
            )
        }
        
        self.current_environment = self._detect_environment()
        self.config_overrides = {}
    
    def _detect_environment(self) -> str:
        """Detect current environment from various sources."""
        # Check environment variable first
        env_var = os.getenv('ENVIRONMENT', '').lower()
        if env_var in self.environments:
            return env_var
        
        # Check for common environment indicators
        if os.getenv('FLASK_ENV') == 'development':
            return 'development'
        if os.getenv('FLASK_ENV') == 'production':
            return 'production'
        
        # Check for common file indicators
        if Path('.env.development').exists():
            return 'development'
        if Path('.env.production').exists():
            return 'production'
        if Path('.env.staging').exists():
            return 'staging'
        if Path('.env.testing').exists():
            return 'testing'
        
        # Check for common directory names
        cwd = Path.cwd().name.lower()
        if 'dev' in cwd or 'development' in cwd:
            return 'development'
        if 'prod' in cwd or 'production' in cwd:
            return 'production'
        if 'staging' in cwd:
            return 'staging'
        if 'test' in cwd:
            return 'testing'
        
        # Default to development for safety
        return 'development'
    
    def get_environment(self, name: Optional[str] = None) -> EnvironmentConfig:
        """Get environment configuration."""
        env_name = name or self.current_environment
        if env_name not in self.environments:
            raise ValueError(f"Environment '{env_name}' not found")
        
        return self.environments[env_name]
    
    def set_environment(self, name: str):
        """Set current environment."""
        if name not in self.environments:
            raise ValueError(f"Environment '{name}' not found")
        
        self.current_environment = name
        print(f"Environment set to: {name}")
    
    def get_current_environment(self) -> EnvironmentConfig:
        """Get current environment configuration."""
        return self.get_environment(self.current_environment)
    
    def override_config(self, section: str, key: str, value: Any):
        """Override configuration value for current environment."""
        if section not in self.config_overrides:
            self.config_overrides[section] = {}
        
        self.config_overrides[section][key] = value
    
    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value with overrides."""
        # Check overrides first
        if section in self.config_overrides and key in self.config_overrides[section]:
            return self.config_overrides[section][key]
        
        # Get from environment config
        env_config = self.get_current_environment()
        
        if hasattr(env_config, key):
            return getattr(env_config, key)
        
        return default
    
    def create_environment_file(self, environment: str, format: str = "env"):
        """Create environment configuration file."""
        if environment not in self.environments:
            raise ValueError(f"Environment '{environment}' not found")
        
        env_config = self.environments[environment]
        
        if format.lower() == "env":
            content = self._create_env_content(env_config)
            filename = f".env.{environment}"
        elif format.lower() == "json":
            content = self._create_json_content(env_config)
            filename = f"config.{environment}.json"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Environment file created: {filename}")
            return True
        except Exception as e:
            print(f"Error creating environment file: {e}")
            return False
    
    def _create_env_content(self, env_config: EnvironmentConfig) -> str:
        """Create .env file content."""
        content = f"# Environment: {env_config.name}\n"
        content += f"# Description: {env_config.description}\n\n"
        
        # Basic settings
        content += f"ENVIRONMENT={env_config.name}\n"
        content += f"DEBUG={str(env_config.debug).lower()}\n"
        content += f"LOG_LEVEL={env_config.log_level}\n\n"
        
        # Database
        content += f"DATABASE_URL={env_config.database_url}\n"
        if env_config.redis_url:
            content += f"REDIS_URL={env_config.redis_url}\n"
        content += "\n"
        
        # Security
        content += f"SECURITY_LEVEL={env_config.security_level}\n"
        content += f"SSL_ENABLED={str(env_config.ssl_enabled).lower()}\n"
        content += f"ALLOWED_HOSTS={','.join(env_config.allowed_hosts)}\n"
        content += f"CORS_ORIGINS={','.join(env_config.cors_origins)}\n\n"
        
        # Monitoring
        content += f"MONITORING_ENABLED={str(env_config.monitoring_enabled).lower()}\n"
        content += f"ALERTING_ENABLED={str(env_config.alerting_enabled).lower()}\n"
        content += f"BACKUP_ENABLED={str(env_config.backup_enabled).lower()}\n\n"
        
        # Performance
        content += f"RATE_LIMITING_ENABLED={str(env_config.rate_limiting['enabled']).lower()}\n"
        content += f"MAX_REQUESTS_PER_MINUTE={env_config.rate_limiting['max_requests_per_minute']}\n"
        content += f"BURST_LIMIT={env_config.rate_limiting['burst_limit']}\n\n"
        
        content += f"CACHE_ENABLED={str(env_config.cache_settings['enabled']).lower()}\n"
        content += f"MAX_CACHE_SIZE={env_config.cache_settings['max_size']}\n"
        content += f"CACHE_TTL_SECONDS={env_config.cache_settings['ttl_seconds']}\n"
        
        return content
    
    def _create_json_content(self, env_config: EnvironmentConfig) -> str:
        """Create JSON configuration content."""
        config_dict = {
            'environment': env_config.name,
            'description': env_config.description,
            'debug': env_config.debug,
            'log_level': env_config.log_level,
            'database': {
                'url': env_config.database_url,
                'redis_url': env_config.redis_url
            },
            'security': {
                'level': env_config.security_level,
                'ssl_enabled': env_config.ssl_enabled,
                'allowed_hosts': env_config.allowed_hosts,
                'cors_origins': env_config.cors_origins
            },
            'monitoring': {
                'enabled': env_config.monitoring_enabled,
                'alerting_enabled': env_config.alerting_enabled,
                'backup_enabled': env_config.backup_enabled
            },
            'performance': {
                'rate_limiting': env_config.rate_limiting,
                'cache_settings': env_config.cache_settings
            }
        }
        
        return json.dumps(config_dict, indent=2)
    
    def validate_environment(self, environment: str) -> Dict[str, Any]:
        """Validate environment configuration."""
        if environment not in self.environments:
            return {'valid': False, 'error': f"Environment '{environment}' not found"}
        
        env_config = self.environments[environment]
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate database URL
        if not env_config.database_url:
            validation_results['errors'].append("Database URL is required")
            validation_results['valid'] = False
        
        # Validate allowed hosts
        if not env_config.allowed_hosts:
            validation_results['warnings'].append("No allowed hosts specified")
        
        # Validate CORS origins
        if not env_config.cors_origins:
            validation_results['warnings'].append("No CORS origins specified")
        
        # Validate rate limiting
        if env_config.rate_limiting['enabled']:
            if env_config.rate_limiting['max_requests_per_minute'] < 1:
                validation_results['errors'].append("Rate limiting requests per minute must be at least 1")
                validation_results['valid'] = False
            
            if env_config.rate_limiting['burst_limit'] < 1:
                validation_results['errors'].append("Rate limiting burst limit must be at least 1")
                validation_results['valid'] = False
        
        # Validate cache settings
        if env_config.cache_settings['enabled']:
            if env_config.cache_settings['max_size'] < 1:
                validation_results['errors'].append("Cache max size must be at least 1")
                validation_results['valid'] = False
            
            if env_config.cache_settings['ttl_seconds'] < 1:
                validation_results['errors'].append("Cache TTL must be at least 1 second")
                validation_results['valid'] = False
        
        return validation_results
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of all environments."""
        summary = {
            'current_environment': self.current_environment,
            'environments': {}
        }
        
        for env_name, env_config in self.environments.items():
            validation = self.validate_environment(env_name)
            summary['environments'][env_name] = {
                'name': env_config.name,
                'description': env_config.description,
                'debug': env_config.debug,
                'security_level': env_config.security_level,
                'monitoring_enabled': env_config.monitoring_enabled,
                'ssl_enabled': env_config.ssl_enabled,
                'validation': validation
            }
        
        return summary
    
    def migrate_environment(self, from_env: str, to_env: str) -> Dict[str, Any]:
        """Migrate configuration from one environment to another."""
        if from_env not in self.environments:
            return {'success': False, 'error': f"Source environment '{from_env}' not found"}
        
        if to_env not in self.environments:
            return {'success': False, 'error': f"Target environment '{to_env}' not found"}
        
        try:
            # Create environment files for both environments
            self.create_environment_file(from_env, "env")
            self.create_environment_file(to_env, "env")
            
            # Create JSON configs for both environments
            self.create_environment_file(from_env, "json")
            self.create_environment_file(to_env, "json")
            
            return {
                'success': True,
                'message': f"Successfully migrated from {from_env} to {to_env}",
                'files_created': [
                    f".env.{from_env}",
                    f".env.{to_env}",
                    f"config.{from_env}.json",
                    f"config.{to_env}.json"
                ]
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Migration failed: {e}"}






