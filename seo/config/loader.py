from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
import orjson
                import boto3
                from botocore.exceptions import ClientError
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration loader for production environment.
Supports YAML, environment variables, and secure secrets management.
"""



class ConfigLoader:
    """Production configuration loader with environment support."""
    
    def __init__(self, config_path: Optional[str] = None):
        
    """__init__ function."""
self.config_path = config_path or "config/production.yml"
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> Any:
        """Load configuration from multiple sources."""
        # Load base config
        self._load_yaml_config()
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Load secrets
        self._load_secrets()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
    
    def _load_yaml_config(self) -> Any:
        """Load YAML configuration file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded YAML config from {self.config_path}")
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load YAML config: {e}")
            self.config = self._get_default_config()
    
    def _load_env_overrides(self) -> Any:
        """Override config with environment variables."""
        env_mappings = {
            # App settings
            'APP_ENVIRONMENT': ('app', 'environment'),
            'APP_DEBUG': ('app', 'debug'),
            'APP_VERSION': ('app', 'version'),
            
            # Server settings
            'SERVER_HOST': ('server', 'host'),
            'SERVER_PORT': ('server', 'port'),
            'SERVER_WORKERS': ('server', 'workers'),
            'SERVER_TIMEOUT': ('server', 'timeout'),
            
            # Database settings
            'DATABASE_URL': ('database', 'url'),
            'DATABASE_POOL_SIZE': ('database', 'pool_size'),
            'DATABASE_MAX_OVERFLOW': ('database', 'max_overflow'),
            
            # Redis settings
            'REDIS_URL': ('redis', 'url'),
            'REDIS_PASSWORD': ('redis', 'password'),
            'REDIS_DB': ('redis', 'db'),
            
            # OpenAI settings
            'OPENAI_API_KEY': ('openai', 'api_key'),
            'OPENAI_MODEL': ('openai', 'model'),
            'OPENAI_TEMPERATURE': ('openai', 'temperature'),
            'OPENAI_MAX_TOKENS': ('openai', 'max_tokens'),
            
            # Selenium settings
            'SELENIUM_HEADLESS': ('selenium', 'headless'),
            'SELENIUM_TIMEOUT': ('selenium', 'timeout'),
            'SELENIUM_WINDOW_SIZE': ('selenium', 'window_size'),
            
            # Cache settings
            'CACHE_TYPE': ('cache', 'type'),
            'CACHE_SIZE': ('cache', 'size'),
            'CACHE_TTL': ('cache', 'ttl'),
            'CACHE_COMPRESSION_LEVEL': ('cache', 'compression_level'),
            
            # HTTP client settings
            'HTTP_RATE_LIMIT': ('http_client', 'rate_limit'),
            'HTTP_MAX_CONNECTIONS': ('http_client', 'max_connections'),
            'HTTP_TIMEOUT': ('http_client', 'timeout'),
            'HTTP_ENABLE_HTTP2': ('http_client', 'enable_http2'),
            
            # Security settings
            'SECRET_KEY': ('security', 'secret_key'),
            'API_KEY': ('security', 'api_key'),
            'CORS_ORIGINS': ('security', 'cors_origins'),
            
            # Monitoring settings
            'PROMETHEUS_PORT': ('monitoring', 'prometheus_port'),
            'GRAFANA_PORT': ('monitoring', 'grafana_port'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(self.config, config_path, self._parse_env_value(env_value))
                logger.debug(f"Override {config_path} with {env_var}")
    
    def _load_secrets(self) -> Any:
        """Load secrets from secure sources."""
        # Load from Kubernetes secrets
        self._load_k8s_secrets()
        
        # Load from Docker secrets
        self._load_docker_secrets()
        
        # Load from AWS Secrets Manager
        self._load_aws_secrets()
    
    def _load_k8s_secrets(self) -> Any:
        """Load secrets from Kubernetes."""
        k8s_secret_path = "/var/run/secrets/kubernetes.io/serviceaccount"
        if os.path.exists(k8s_secret_path):
            try:
                # Load Kubernetes secrets
                secret_files = [
                    "openai-api-key",
                    "redis-password",
                    "database-password",
                    "secret-key"
                ]
                
                for secret_file in secret_files:
                    secret_path = f"/var/run/secrets/kubernetes.io/serviceaccount/{secret_file}"
                    if os.path.exists(secret_path):
                        with open(secret_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            secret_value = f.read().strip()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            
                        # Map to config
                        if secret_file == "openai-api-key":
                            self._set_nested_value(self.config, ('openai', 'api_key'), secret_value)
                        elif secret_file == "redis-password":
                            self._set_nested_value(self.config, ('redis', 'password'), secret_value)
                        elif secret_file == "database-password":
                            self._set_nested_value(self.config, ('database', 'password'), secret_value)
                        elif secret_file == "secret-key":
                            self._set_nested_value(self.config, ('security', 'secret_key'), secret_value)
                
                logger.info("Loaded Kubernetes secrets")
            except Exception as e:
                logger.warning(f"Failed to load Kubernetes secrets: {e}")
    
    def _load_docker_secrets(self) -> Any:
        """Load secrets from Docker secrets."""
        docker_secrets_path = "/run/secrets"
        if os.path.exists(docker_secrets_path):
            try:
                secret_files = os.listdir(docker_secrets_path)
                for secret_file in secret_files:
                    secret_path = os.path.join(docker_secrets_path, secret_file)
                    with open(secret_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        secret_value = f.read().strip()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    
                    # Map to config based on filename
                    if secret_file == "openai_api_key":
                        self._set_nested_value(self.config, ('openai', 'api_key'), secret_value)
                    elif secret_file == "redis_password":
                        self._set_nested_value(self.config, ('redis', 'password'), secret_value)
                    elif secret_file == "database_password":
                        self._set_nested_value(self.config, ('database', 'password'), secret_value)
                    elif secret_file == "secret_key":
                        self._set_nested_value(self.config, ('security', 'secret_key'), secret_value)
                
                logger.info("Loaded Docker secrets")
            except Exception as e:
                logger.warning(f"Failed to load Docker secrets: {e}")
    
    def _load_aws_secrets(self) -> Any:
        """Load secrets from AWS Secrets Manager."""
        aws_secret_name = os.getenv('AWS_SECRET_NAME')
        if aws_secret_name:
            try:
                
                session = boto3.session.Session()
                client = session.client(
                    service_name='secretsmanager',
                    region_name=os.getenv('AWS_REGION', 'us-east-1')
                )
                
                response = client.get_secret_value(SecretId=aws_secret_name)
                if 'SecretString' in response:
                    secrets = json.loads(response['SecretString'])
                    
                    # Map secrets to config
                    for key, value in secrets.items():
                        if key == 'openai_api_key':
                            self._set_nested_value(self.config, ('openai', 'api_key'), value)
                        elif key == 'redis_password':
                            self._set_nested_value(self.config, ('redis', 'password'), value)
                        elif key == 'database_password':
                            self._set_nested_value(self.config, ('database', 'password'), value)
                        elif key == 'secret_key':
                            self._set_nested_value(self.config, ('security', 'secret_key'), value)
                
                logger.info("Loaded AWS secrets")
            except Exception as e:
                logger.warning(f"Failed to load AWS secrets: {e}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # List values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Default to string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: tuple, value: Any):
        """Set nested value in configuration dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate_config(self) -> bool:
        """Validate configuration and set defaults."""
        # Ensure required sections exist
        required_sections = ['app', 'server', 'database', 'redis', 'openai']
        for section in required_sections:
            if section not in self.config:
                self.config[section] = {}
        
        # Set defaults for missing values
        defaults = {
            'app': {
                'name': 'SEO Service Ultra-Optimized',
                'version': '2.0.0',
                'environment': 'production',
                'debug': False
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout': 30
            },
            'database': {
                'url': 'postgresql://user:pass@localhost:5432/seo_db',
                'pool_size': 20,
                'max_overflow': 30
            },
            'redis': {
                'url': 'redis://localhost:6379/0',
                'password': None,
                'db': 0
            },
            'openai': {
                'api_key': None,
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.1,
                'max_tokens': 4000
            },
            'selenium': {
                'headless': True,
                'timeout': 30,
                'window_size': '1920x1080'
            },
            'cache': {
                'type': 'ultra_optimized',
                'size': 5000,
                'ttl': 7200,
                'compression_level': 3
            },
            'http_client': {
                'rate_limit': 200,
                'max_connections': 200,
                'timeout': 15.0,
                'enable_http2': True
            },
            'security': {
                'secret_key': None,
                'api_key': None,
                'cors_origins': ['*']
            },
            'monitoring': {
                'prometheus_port': 9090,
                'grafana_port': 3000
            },
            'logging': {
                'level': 'INFO',
                'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
                'file_enabled': True,
                'file_path': 'logs/seo-service.log',
                'rotation': '100 MB',
                'retention': '30 days',
                'compression': 'zip'
            }
        }
        
        for section, section_defaults in defaults.items():
            if section not in self.config:
                self.config[section] = section_defaults
            else:
                for key, default_value in section_defaults.items():
                    if key not in self.config[section]:
                        self.config[section][key] = default_value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'app': {
                'name': 'SEO Service Ultra-Optimized',
                'version': '2.0.0',
                'environment': 'production',
                'debug': False
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout': 30
            },
            'database': {
                'url': 'postgresql://user:pass@localhost:5432/seo_db',
                'pool_size': 20,
                'max_overflow': 30
            },
            'redis': {
                'url': 'redis://localhost:6379/0',
                'password': None,
                'db': 0
            },
            'openai': {
                'api_key': None,
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.1,
                'max_tokens': 4000
            },
            'selenium': {
                'headless': True,
                'timeout': 30,
                'window_size': '1920x1080'
            },
            'cache': {
                'type': 'ultra_optimized',
                'size': 5000,
                'ttl': 7200,
                'compression_level': 3
            },
            'http_client': {
                'rate_limit': 200,
                'max_connections': 200,
                'timeout': 15.0,
                'enable_http2': True
            },
            'security': {
                'secret_key': None,
                'api_key': None,
                'cors_origins': ['*']
            },
            'monitoring': {
                'prometheus_port': 9090,
                'grafana_port': 3000
            },
            'logging': {
                'level': 'INFO',
                'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
                'file_enabled': True,
                'file_path': 'logs/seo-service.log',
                'rotation': '100 MB',
                'retention': '30 days',
                'compression': 'zip'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get configuration value by key (dot notation supported)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return self.config.copy()
    
    def to_json(self) -> str:
        """Get configuration as JSON string."""
        return orjson.dumps(self.config, option=orjson.OPT_INDENT_2).decode()
    
    def validate_required(self, required_keys: list) -> bool:
        """Validate that required configuration keys exist."""
        missing_keys = []
        
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        return True


# Global configuration instance
_config_instance: Optional[ConfigLoader] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration (singleton pattern)."""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    
    return _config_instance.to_dict()


def get_config() -> ConfigLoader:
    """Get configuration loader instance."""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader()
    
    return _config_instance


def reload_config(config_path: Optional[str] = None):
    """Reload configuration."""
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    return _config_instance.to_dict() 