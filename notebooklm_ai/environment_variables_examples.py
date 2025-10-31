from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import os
import logging
import json
import yaml
import re
import hashlib
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple, Type, TypeVar
from enum import Enum
import threading
from contextlib import contextmanager
from collections import defaultdict
import secrets
import string
from datetime import datetime, timedelta
    import dotenv
    import pydantic
from typing import Any, List, Dict, Optional
import asyncio
"""
Environment Variables Management Examples
========================================

This module provides comprehensive environment variable management with
validation, type conversion, security, and best practices.

Features:
- Environment variable loading and validation
- Type conversion and default values
- Security validation and sanitization
- Configuration management and hot reload
- Environment-specific configurations
- Secret management and encryption
- Validation schemas and error handling
- Performance optimization and caching
- Documentation and logging
- Testing and debugging utilities

Author: AI Assistant
License: MIT
"""


try:
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class EnvironmentType(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    LOCAL = "local"


class VariableType(Enum):
    """Variable types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    YAML = "yaml"
    PATH = "path"
    URL = "url"
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    PORT = "port"
    SECRET = "secret"


class SecurityLevel(Enum):
    """Security levels for environment variables."""
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    SECRET = "secret"
    CRITICAL = "critical"


@dataclass
class EnvVarConfig:
    """Environment variable configuration."""
    name: str
    var_type: VariableType = VariableType.STRING
    default: Any = None
    required: bool = False
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    validation_regex: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    example: str = ""


@dataclass
class EnvVarResult:
    """Result of environment variable processing."""
    name: str
    value: Any
    success: bool = True
    error_message: str = ""
    source: str = "environment"
    timestamp: datetime = field(default_factory=datetime.now)
    validation_passed: bool = True
    security_checked: bool = True


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    env_type: EnvironmentType = EnvironmentType.DEVELOPMENT
    config_file: Optional[str] = None
    secrets_file: Optional[str] = None
    validation_strict: bool = True
    cache_enabled: bool = True
    hot_reload: bool = False
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    log_sensitive: bool = False


class EnvironmentVariableError(Exception):
    """Custom exception for environment variable errors."""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security errors."""
    pass


class TypeConversionError(Exception):
    """Custom exception for type conversion errors."""
    pass


class EnvironmentManager:
    """Comprehensive environment variable manager."""
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize environment manager."""
        self.config = config
        self.variables: Dict[str, EnvVarResult] = {}
        self.validators: Dict[str, Callable] = self._register_validators()
        self.type_converters: Dict[VariableType, Callable] = self._register_type_converters()
        self.cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._last_reload = datetime.now()
        
        # Load configuration files
        self._load_config_files()
    
    def _register_validators(self) -> Dict[str, Callable]:
        """Register validation functions."""
        return {
            'url': self._validate_url,
            'email': self._validate_email,
            'ip_address': self._validate_ip_address,
            'port': self._validate_port,
            'path': self._validate_path,
            'json': self._validate_json,
            'yaml': self._validate_yaml,
            'secret': self._validate_secret
        }
    
    def _register_type_converters(self) -> Dict[VariableType, Callable]:
        """Register type conversion functions."""
        return {
            VariableType.STRING: str,
            VariableType.INTEGER: int,
            VariableType.FLOAT: float,
            VariableType.BOOLEAN: self._convert_boolean,
            VariableType.JSON: self._convert_json,
            VariableType.YAML: self._convert_yaml,
            VariableType.PATH: Path,
            VariableType.URL: str,
            VariableType.EMAIL: str,
            VariableType.IP_ADDRESS: str,
            VariableType.PORT: int,
            VariableType.SECRET: str
        }
    
    def _load_config_files(self) -> Any:
        """Load configuration files."""
        if not self.config.config_file:
            return
        
        config_path = Path(self.config.config_file)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            if config_path.suffix.lower() == '.yaml':
                with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config_data = json.load(f)
            else:
                logger.error(f"Unsupported config file format: {config_path.suffix}")
                return
            
            # Load environment variables from config
            for key, value in config_data.items():
                if isinstance(value, dict):
                    # Handle nested configuration
                    self._load_nested_config(key, value)
                else:
                    # Set as environment variable
                    os.environ[key] = str(value)
            
            logger.info(f"Loaded configuration from {config_path}")
        
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")
    
    def _load_nested_config(self, prefix: str, config: Dict[str, Any]):
        """Load nested configuration as environment variables."""
        for key, value in config.items():
            env_key = f"{prefix}_{key}".upper()
            if isinstance(value, dict):
                self._load_nested_config(env_key, value)
            else:
                os.environ[env_key] = str(value)
    
    def _validate_url(self, value: str) -> bool:
        """Validate URL format."""
        if not value:
            return False
        
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(value))
    
    def _validate_email(self, value: str) -> bool:
        """Validate email format."""
        if not value:
            return False
        
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        return bool(email_pattern.match(value))
    
    def _validate_ip_address(self, value: str) -> bool:
        """Validate IP address format."""
        if not value:
            return False
        
        # IPv4 pattern
        ipv4_pattern = re.compile(
            r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
        
        # IPv6 pattern (simplified)
        ipv6_pattern = re.compile(
            r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$')
        
        return bool(ipv4_pattern.match(value) or ipv6_pattern.match(value))
    
    def _validate_port(self, value: Union[str, int]) -> bool:
        """Validate port number."""
        try:
            port = int(value)
            return 1 <= port <= 65535
        except (ValueError, TypeError):
            return False
    
    def _validate_path(self, value: str) -> bool:
        """Validate file path."""
        if not value:
            return False
        
        try:
            path = Path(value)
            return True
        except Exception:
            return False
    
    def _validate_json(self, value: str) -> bool:
        """Validate JSON format."""
        if not value:
            return False
        
        try:
            json.loads(value)
            return True
        except json.JSONDecodeError:
            return False
    
    def _validate_yaml(self, value: str) -> bool:
        """Validate YAML format."""
        if not value:
            return False
        
        try:
            yaml.safe_load(value)
            return True
        except yaml.YAMLError:
            return False
    
    def _validate_secret(self, value: str) -> bool:
        """Validate secret strength."""
        if not value:
            return False
        
        # Check minimum length
        if len(value) < 8:
            return False
        
        # Check for complexity
        has_upper = any(c.isupper() for c in value)
        has_lower = any(c.islower() for c in value)
        has_digit = any(c.isdigit() for c in value)
        has_special = any(c in string.punctuation for c in value)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _convert_boolean(self, value: str) -> bool:
        """Convert string to boolean."""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ('true', '1', 'yes', 'on', 'enabled'):
                return True
            elif value in ('false', '0', 'no', 'off', 'disabled'):
                return False
        
        raise TypeConversionError(f"Cannot convert '{value}' to boolean")
    
    def _convert_json(self, value: str) -> Any:
        """Convert string to JSON object."""
        if isinstance(value, (dict, list)):
            return value
        
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise TypeConversionError(f"Invalid JSON: {e}")
    
    def _convert_yaml(self, value: str) -> Any:
        """Convert string to YAML object."""
        if isinstance(value, (dict, list)):
            return value
        
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError as e:
            raise TypeConversionError(f"Invalid YAML: {e}")
    
    def _check_security(self, config: EnvVarConfig, value: Any) -> bool:
        """Check security requirements."""
        if config.security_level == SecurityLevel.PUBLIC:
            return True
        
        # Log sensitive variables (if enabled)
        if config.security_level in (SecurityLevel.SENSITIVE, SecurityLevel.SECRET, SecurityLevel.CRITICAL):
            if self.config.log_sensitive:
                logger.warning(f"Loading sensitive variable: {config.name}")
            else:
                logger.debug(f"Loading sensitive variable: {config.name}")
        
        # Additional security checks for critical variables
        if config.security_level == SecurityLevel.CRITICAL:
            if not self._validate_secret(str(value)):
                logger.error(f"Critical variable {config.name} does not meet security requirements")
                return False
        
        return True
    
    def _validate_value(self, config: EnvVarConfig, value: Any) -> bool:
        """Validate value according to configuration."""
        if not value and config.required:
            return False
        
        if value is None:
            return True
        
        value_str = str(value)
        
        # Check length constraints
        if config.min_length and len(value_str) < config.min_length:
            return False
        
        if config.max_length and len(value_str) > config.max_length:
            return False
        
        # Check value constraints
        if config.min_value is not None:
            try:
                if float(value) < config.min_value:
                    return False
            except (ValueError, TypeError):
                return False
        
        if config.max_value is not None:
            try:
                if float(value) > config.max_value:
                    return False
            except (ValueError, TypeError):
                return False
        
        # Check allowed values
        if config.allowed_values and value not in config.allowed_values:
            return False
        
        # Check regex pattern
        if config.validation_regex:
            if not re.match(config.validation_regex, value_str):
                return False
        
        # Run type-specific validation
        validator_name = config.var_type.value
        if validator_name in self.validators:
            if not self.validators[validator_name](value):
                return False
        
        return True
    
    def _convert_value(self, config: EnvVarConfig, value: str) -> Any:
        """Convert value to specified type."""
        if value is None:
            return config.default
        
        try:
            converter = self.type_converters.get(config.var_type, str)
            return converter(value)
        except Exception as e:
            raise TypeConversionError(f"Failed to convert {config.name}: {e}")
    
    def get(self, name: str, config: EnvVarConfig) -> EnvVarResult:
        """Get environment variable with validation and conversion."""
        with self._lock:
            # Check cache first
            cache_key = f"{name}_{config.var_type.value}"
            if self.config.cache_enabled and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get raw value from environment
            raw_value = os.environ.get(name)
            
            # Handle required variables
            if raw_value is None:
                if config.required:
                    error_msg = f"Required environment variable '{name}' not found"
                    logger.error(error_msg)
                    result = EnvVarResult(
                        name=name,
                        value=config.default,
                        success=False,
                        error_message=error_msg,
                        validation_passed=False
                    )
                else:
                    result = EnvVarResult(
                        name=name,
                        value=config.default,
                        source="default"
                    )
            else:
                # Validate and convert value
                try:
                    # Check security requirements
                    if not self._check_security(config, raw_value):
                        result = EnvVarResult(
                            name=name,
                            value=config.default,
                            success=False,
                            error_message="Security validation failed",
                            security_checked=False
                        )
                    else:
                        # Validate value
                        if not self._validate_value(config, raw_value):
                            error_msg = f"Validation failed for '{name}'"
                            logger.error(error_msg)
                            result = EnvVarResult(
                                name=name,
                                value=config.default,
                                success=False,
                                error_message=error_msg,
                                validation_passed=False
                            )
                        else:
                            # Convert value
                            converted_value = self._convert_value(config, raw_value)
                            result = EnvVarResult(
                                name=name,
                                value=converted_value,
                                success=True
                            )
                
                except Exception as e:
                    error_msg = f"Error processing '{name}': {e}"
                    logger.error(error_msg)
                    result = EnvVarResult(
                        name=name,
                        value=config.default,
                        success=False,
                        error_message=error_msg
                    )
            
            # Store in cache
            if self.config.cache_enabled:
                self.cache[cache_key] = result
            
            # Store in variables
            self.variables[name] = result
            
            return result
    
    def set(self, name: str, value: Any, config: EnvVarConfig = None) -> EnvVarResult:
        """Set environment variable with validation."""
        if config is None:
            config = EnvVarConfig(name=name)
        
        with self._lock:
            try:
                # Validate and convert value
                if not self._check_security(config, value):
                    raise SecurityError(f"Security validation failed for {name}")
                
                if not self._validate_value(config, value):
                    raise ValidationError(f"Validation failed for {name}")
                
                converted_value = self._convert_value(config, str(value))
                
                # Set in environment
                os.environ[name] = str(value)
                
                result = EnvVarResult(
                    name=name,
                    value=converted_value,
                    success=True,
                    source="set"
                )
                
                # Update cache and variables
                if self.config.cache_enabled:
                    cache_key = f"{name}_{config.var_type.value}"
                    self.cache[cache_key] = result
                
                self.variables[name] = result
                
                logger.info(f"Set environment variable: {name}")
                return result
            
            except Exception as e:
                error_msg = f"Failed to set '{name}': {e}"
                logger.error(error_msg)
                return EnvVarResult(
                    name=name,
                    value=None,
                    success=False,
                    error_message=error_msg
                )
    
    def reload(self) -> bool:
        """Reload environment variables from files."""
        with self._lock:
            try:
                # Clear cache
                self.cache.clear()
                self.variables.clear()
                
                # Reload configuration files
                self._load_config_files()
                
                # Load .env file if available
                if DOTENV_AVAILABLE:
                    dotenv.load_dotenv()
                
                self._last_reload = datetime.now()
                logger.info("Environment variables reloaded successfully")
                return True
            
            except Exception as e:
                logger.error(f"Failed to reload environment variables: {e}")
                return False
    
    def export(self, format: str = "json") -> str:
        """Export environment variables to specified format."""
        with self._lock:
            data = {}
            for name, result in self.variables.items():
                if result.success:
                    # Mask sensitive values
                    if result.name in self._get_sensitive_variables():
                        data[name] = "***MASKED***"
                    else:
                        data[name] = result.value
            
            if format.lower() == "json":
                return json.dumps(data, indent=2, default=str)
            elif format.lower() == "yaml":
                return yaml.dump(data, default_flow_style=False)
            elif format.lower() == "env":
                lines = []
                for name, value in data.items():
                    lines.append(f"{name}={value}")
                return "\n".join(lines)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def _get_sensitive_variables(self) -> Set[str]:
        """Get list of sensitive variable names."""
        sensitive = set()
        for name, result in self.variables.items():
            if hasattr(result, 'security_level') and result.security_level in (SecurityLevel.SENSITIVE, SecurityLevel.SECRET, SecurityLevel.CRITICAL):
                sensitive.add(name)
        return sensitive
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of environment variables."""
        with self._lock:
            total = len(self.variables)
            successful = sum(1 for r in self.variables.values() if r.success)
            failed = total - successful
            
            sensitive_count = len(self._get_sensitive_variables())
            
            return {
                'total_variables': total,
                'successful': successful,
                'failed': failed,
                'sensitive_count': sensitive_count,
                'last_reload': self._last_reload.isoformat(),
                'cache_enabled': self.config.cache_enabled,
                'cache_size': len(self.cache)
            }


class SecretManager:
    """Secure secret management for environment variables."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize secret manager."""
        self.encryption_key = encryption_key or self._generate_key()
        self.secrets: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def _generate_key(self) -> str:
        """Generate encryption key."""
        return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
    
    def _encrypt(self, value: str) -> str:
        """Encrypt a value."""
        # Simple XOR encryption for demonstration
        # In production, use proper encryption like AES
        key_bytes = self.encryption_key.encode('utf-8')
        value_bytes = value.encode('utf-8')
        
        encrypted = bytearray()
        for i, byte in enumerate(value_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    def _decrypt(self, encrypted_value: str) -> str:
        """Decrypt a value."""
        # Simple XOR decryption for demonstration
        key_bytes = self.encryption_key.encode('utf-8')
        encrypted_bytes = base64.b64decode(encrypted_value.encode('utf-8'))
        
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return decrypted.decode('utf-8')
    
    def store_secret(self, name: str, value: str) -> bool:
        """Store a secret securely."""
        with self._lock:
            try:
                encrypted = self._encrypt(value)
                self.secrets[name] = encrypted
                logger.info(f"Stored secret: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to store secret {name}: {e}")
                return False
    
    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret."""
        with self._lock:
            if name not in self.secrets:
                return None
            
            try:
                encrypted = self.secrets[name]
                decrypted = self._decrypt(encrypted)
                return decrypted
            except Exception as e:
                logger.error(f"Failed to retrieve secret {name}: {e}")
                return None
    
    def list_secrets(self) -> List[str]:
        """List all secret names."""
        with self._lock:
            return list(self.secrets.keys())
    
    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        with self._lock:
            if name in self.secrets:
                del self.secrets[name]
                logger.info(f"Deleted secret: {name}")
                return True
            return False


class EnvironmentValidator:
    """Environment validation and testing utilities."""
    
    def __init__(self, manager: EnvironmentManager):
        """Initialize environment validator."""
        self.manager = manager
    
    def validate_environment(self, required_vars: List[EnvVarConfig]) -> Dict[str, Any]:
        """Validate complete environment configuration."""
        results = {
            'valid': True,
            'missing': [],
            'invalid': [],
            'warnings': [],
            'summary': {}
        }
        
        for config in required_vars:
            result = self.manager.get(config.name, config)
            
            if not result.success:
                results['valid'] = False
                if 'not found' in result.error_message.lower():
                    results['missing'].append(config.name)
                else:
                    results['invalid'].append({
                        'name': config.name,
                        'error': result.error_message
                    })
            
            if result.value == config.default and config.required:
                results['warnings'].append(f"Using default value for required variable: {config.name}")
        
        # Generate summary
        results['summary'] = {
            'total': len(required_vars),
            'valid': len(required_vars) - len(results['missing']) - len(results['invalid']),
            'missing': len(results['missing']),
            'invalid': len(results['invalid']),
            'warnings': len(results['warnings'])
        }
        
        return results
    
    def test_environment(self, test_configs: List[EnvVarConfig]) -> Dict[str, Any]:
        """Test environment variables with various scenarios."""
        results = {
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        for config in test_configs:
            test_result = {
                'name': config.name,
                'passed': False,
                'error': None
            }
            
            try:
                result = self.manager.get(config.name, config)
                if result.success:
                    test_result['passed'] = True
                    results['passed'] += 1
                else:
                    test_result['error'] = result.error_message
                    results['failed'] += 1
            except Exception as e:
                test_result['error'] = str(e)
                results['failed'] += 1
            
            results['tests'].append(test_result)
        
        return results


# Example usage functions
def demonstrate_basic_usage():
    """Demonstrate basic environment variable usage."""
    config = EnvironmentConfig(
        env_type=EnvironmentType.DEVELOPMENT,
        validation_strict=True,
        cache_enabled=True
    )
    
    manager = EnvironmentManager(config)
    
    # Define environment variable configurations
    db_config = EnvVarConfig(
        name="DATABASE_URL",
        var_type=VariableType.URL,
        required=True,
        security_level=SecurityLevel.SENSITIVE,
        description="Database connection URL",
        example="postgresql://user:pass@localhost:5432/db"
    )
    
    port_config = EnvVarConfig(
        name="API_PORT",
        var_type=VariableType.PORT,
        default=8000,
        min_value=1024,
        max_value=65535,
        description="API server port"
    )
    
    debug_config = EnvVarConfig(
        name="DEBUG",
        var_type=VariableType.BOOLEAN,
        default=False,
        description="Enable debug mode"
    )
    
    # Get environment variables
    db_result = manager.get("DATABASE_URL", db_config)
    port_result = manager.get("API_PORT", port_config)
    debug_result = manager.get("DEBUG", debug_config)
    
    print("Environment Variables:")
    print(f"  Database URL: {'***MASKED***' if db_result.success else 'ERROR'}")
    print(f"  API Port: {port_result.value}")
    print(f"  Debug Mode: {debug_result.value}")
    
    # Get summary
    summary = manager.get_summary()
    print(f"\nSummary: {summary}")


def demonstrate_validation():
    """Demonstrate environment variable validation."""
    config = EnvironmentConfig(validation_strict=True)
    manager = EnvironmentManager(config)
    validator = EnvironmentValidator(manager)
    
    # Define validation configurations
    validation_configs = [
        EnvVarConfig(
            name="EMAIL_SERVER",
            var_type=VariableType.EMAIL,
            required=True,
            description="Email server address"
        ),
        EnvVarConfig(
            name="MAX_CONNECTIONS",
            var_type=VariableType.INTEGER,
            default=100,
            min_value=1,
            max_value=1000,
            description="Maximum database connections"
        ),
        EnvVarConfig(
            name="API_KEY",
            var_type=VariableType.SECRET,
            required=True,
            security_level=SecurityLevel.CRITICAL,
            description="API authentication key"
        )
    ]
    
    # Validate environment
    validation_result = validator.validate_environment(validation_configs)
    
    print("Environment Validation:")
    print(f"  Valid: {validation_result['valid']}")
    print(f"  Missing: {validation_result['missing']}")
    print(f"  Invalid: {validation_result['invalid']}")
    print(f"  Warnings: {validation_result['warnings']}")
    print(f"  Summary: {validation_result['summary']}")


def demonstrate_secret_management():
    """Demonstrate secret management."""
    secret_manager = SecretManager()
    
    # Store secrets
    secrets_to_store = {
        "DB_PASSWORD": "super_secret_password_123",
        "API_SECRET": "api_secret_key_456",
        "JWT_SECRET": "jwt_secret_key_789"
    }
    
    for name, value in secrets_to_store.items():
        success = secret_manager.store_secret(name, value)
        print(f"Stored {name}: {'SUCCESS' if success else 'FAILED'}")
    
    # Retrieve secrets
    print("\nRetrieved Secrets:")
    for name in secrets_to_store.keys():
        value = secret_manager.get_secret(name)
        print(f"  {name}: {value}")
    
    # List secrets
    secret_list = secret_manager.list_secrets()
    print(f"\nAll Secrets: {secret_list}")


def demonstrate_advanced_features():
    """Demonstrate advanced environment variable features."""
    config = EnvironmentConfig(
        env_type=EnvironmentType.PRODUCTION,
        config_file="config.yaml",
        hot_reload=True,
        encryption_enabled=True
    )
    
    manager = EnvironmentManager(config)
    
    # Complex configuration
    complex_config = EnvVarConfig(
        name="REDIS_CONFIG",
        var_type=VariableType.JSON,
        default={"host": "localhost", "port": 6379},
        validation_regex=r'^\{.*\}$',
        description="Redis configuration as JSON"
    )
    
    # Set complex value
    redis_config = {
        "host": "redis.example.com",
        "port": 6379,
        "password": "redis_pass",
        "ssl": True
    }
    
    result = manager.set("REDIS_CONFIG", json.dumps(redis_config), complex_config)
    print(f"Set Redis config: {'SUCCESS' if result.success else 'FAILED'}")
    
    # Export environment
    export_json = manager.export("json")
    print(f"\nEnvironment Export (JSON):\n{export_json}")
    
    # Reload environment
    reload_success = manager.reload()
    print(f"\nReload successful: {reload_success}")


def main():
    """Main function demonstrating environment variable management."""
    logger.info("Starting environment variable management examples")
    
    # Demonstrate basic usage
    try:
        demonstrate_basic_usage()
    except Exception as e:
        logger.error(f"Basic usage demonstration failed: {e}")
    
    # Demonstrate validation
    try:
        demonstrate_validation()
    except Exception as e:
        logger.error(f"Validation demonstration failed: {e}")
    
    # Demonstrate secret management
    try:
        demonstrate_secret_management()
    except Exception as e:
        logger.error(f"Secret management demonstration failed: {e}")
    
    # Demonstrate advanced features
    try:
        demonstrate_advanced_features()
    except Exception as e:
        logger.error(f"Advanced features demonstration failed: {e}")
    
    logger.info("Environment variable management examples completed")


match __name__:
    case "__main__":
    main() 