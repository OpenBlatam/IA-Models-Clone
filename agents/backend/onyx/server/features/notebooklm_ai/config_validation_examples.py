from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from enum import Enum
import hashlib
import threading
from contextlib import contextmanager
    import yaml
    from yaml import SafeLoader, Loader, Dumper, YAMLError
    from yaml.constructor import ConstructorError
    from yaml.scanner import ScannerError
    from yaml.parser import ParserError
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
    from jsonschema.validators import validator_for
    from jsonschema.exceptions import SchemaError
from typing import Any, List, Dict, Optional
"""
Configuration Loading and Validation Examples
============================================

This module provides robust configuration loading and validation capabilities using:
- PyYAML: YAML configuration loading and parsing
- python-jsonschema: JSON Schema validation
- Combined approach for comprehensive config management

Features:
- YAML/JSON configuration loading with validation
- JSON Schema validation with detailed error reporting
- Configuration inheritance and merging
- Environment variable substitution
- Type conversion and coercion
- Default value handling
- Configuration hot-reloading
- Validation caching and performance optimization
- Comprehensive error handling and logging

Author: AI Assistant
License: MIT
"""


try:
    PYAML_AVAILABLE = True
except ImportError:
    PYAML_AVAILABLE = False
    SafeLoader = None
    Loader = None
    Dumper = None
    YAMLError = Exception
    ConstructorError = Exception
    ScannerError = Exception
    ParserError = Exception

try:
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    validate = None
    ValidationError = Exception
    Draft7Validator = None
    validator_for = None
    SchemaError = Exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigType(Enum):
    """Configuration file types."""
    YAML = "yaml"
    JSON = "json"
    INI = "ini"
    ENV = "env"


class ValidationLevel(Enum):
    """Validation levels."""
    STRICT = "strict"
    WARN = "warn"
    LOOSE = "loose"


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_errors: List[Dict[str, Any]] = field(default_factory=list)
    validation_time: float = 0.0
    config_hash: str = ""
    schema_hash: str = ""


@dataclass
class ConfigLoadResult:
    """Result of configuration loading."""
    success: bool
    config: Optional[Dict[str, Any]] = None
    config_type: Optional[ConfigType] = None
    file_path: str = ""
    load_time: float = 0.0
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    config_hash: str = ""


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    schema: Dict[str, Any]
    schema_type: str = "draft-07"
    title: str = ""
    description: str = ""
    version: str = "1.0.0"
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    custom_validators: Dict[str, Callable] = field(default_factory=dict)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigLoadError(Exception):
    """Custom exception for configuration loading errors."""
    pass


class ConfigSchemaError(Exception):
    """Custom exception for configuration schema errors."""
    pass


class EnvironmentVariableError(Exception):
    """Custom exception for environment variable errors."""
    pass


class ConfigManager:
    """Main configuration manager with loading and validation capabilities."""
    
    def __init__(self, base_path: str = "", validation_level: ValidationLevel = ValidationLevel.STRICT):
        """Initialize configuration manager."""
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.validation_level = validation_level
        self.config_cache: Dict[str, ConfigLoadResult] = {}
        self.schema_cache: Dict[str, ConfigSchema] = {}
        self.validation_cache: Dict[str, ConfigValidationResult] = {}
        self._cache_lock = threading.Lock()
        self._file_watchers: Dict[str, float] = {}
        self._hot_reload_callbacks: Dict[str, List[Callable]] = {}
        
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash for cache invalidation."""
        try:
            with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _is_file_modified(self, file_path: str) -> bool:
        """Check if file has been modified since last check."""
        if file_path not in self._file_watchers:
            return True
        
        try:
            current_mtime = os.path.getmtime(file_path)
            return current_mtime > self._file_watchers[file_path]
        except Exception:
            return True
    
    def _update_file_watcher(self, file_path: str):
        """Update file modification time."""
        try:
            self._file_watchers[file_path] = os.path.getmtime(file_path)
        except Exception:
            pass
    
    def _detect_config_type(self, file_path: str) -> ConfigType:
        """Detect configuration file type based on extension."""
        if not file_path:
            return ConfigType.YAML
        
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.yaml', '.yml']:
            return ConfigType.YAML
        elif ext == '.json':
            return ConfigType.JSON
        elif ext == '.ini':
            return ConfigType.INI
        elif ext == '.env':
            return ConfigType.ENV
        else:
            # Default to YAML for unknown extensions
            return ConfigType.YAML
    
    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base path."""
        if not file_path:
            raise ConfigLoadError("File path is required")
        
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path
        
        return path
    
    def _substitute_env_vars(self, value: Any) -> Any:
        """Substitute environment variables in configuration values."""
        if isinstance(value, str):
            # Pattern to match ${VAR_NAME} or $VAR_NAME
            env_pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
            
            def replace_env_var(match) -> Any:
                var_name = match.group(1) or match.group(2)
                if not var_name:
                    return match.group(0)
                
                env_value = os.getenv(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {var_name} not found")
                    return match.group(0)
                
                return env_value
            
            return re.sub(env_pattern, replace_env_var, value)
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        else:
            return value
    
    def load_yaml_config(self, file_path: str, substitute_env: bool = True) -> ConfigLoadResult:
        """Load YAML configuration file."""
        if not PYAML_AVAILABLE:
            return ConfigLoadResult(
                success=False,
                file_path=file_path,
                error_message="PyYAML is not available. Install with: pip install PyYAML"
            )
        
        start_time = time.time()
        resolved_path = self._resolve_path(file_path)
        
        # Check cache first
        cache_key = f"yaml:{resolved_path}"
        if cache_key in self.config_cache:
            cached_result = self.config_cache[cache_key]
            if not self._is_file_modified(str(resolved_path)):
                return cached_result
        
        try:
            if not resolved_path.exists():
                return ConfigLoadResult(
                    success=False,
                    file_path=str(resolved_path),
                    error_message=f"Configuration file not found: {resolved_path}"
                )
            
            logger.info(f"Loading YAML configuration from {resolved_path}")
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Parse YAML
                try:
                    config = yaml.safe_load(content)
                except YAMLError as e:
                    return ConfigLoadResult(
                        success=False,
                        file_path=str(resolved_path),
                        error_message=f"YAML parsing error: {e}",
                        config_hash=config_hash
                    )
                
                if config is None:
                    config = {}
                
                # Substitute environment variables if requested
                if substitute_env:
                    config = self._substitute_env_vars(config)
                
                load_time = time.time() - start_time
                
                result = ConfigLoadResult(
                    success=True,
                    config=config,
                    config_type=ConfigType.YAML,
                    file_path=str(resolved_path),
                    load_time=load_time,
                    config_hash=config_hash
                )
                
                # Cache result
                with self._cache_lock:
                    self.config_cache[cache_key] = result
                    self._update_file_watcher(str(resolved_path))
                
                return result
                
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Error loading YAML config: {e}")
            return ConfigLoadResult(
                success=False,
                file_path=str(resolved_path),
                error_message=f"Unexpected error: {e}",
                load_time=load_time
            )
    
    def load_json_config(self, file_path: str, substitute_env: bool = True) -> ConfigLoadResult:
        """Load JSON configuration file."""
        start_time = time.time()
        resolved_path = self._resolve_path(file_path)
        
        # Check cache first
        cache_key = f"json:{resolved_path}"
        if cache_key in self.config_cache:
            cached_result = self.config_cache[cache_key]
            if not self._is_file_modified(str(resolved_path)):
                return cached_result
        
        try:
            if not resolved_path.exists():
                return ConfigLoadResult(
                    success=False,
                    file_path=str(resolved_path),
                    error_message=f"Configuration file not found: {resolved_path}"
                )
            
            logger.info(f"Loading JSON configuration from {resolved_path}")
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Parse JSON
                try:
                    config = json.loads(content)
                except json.JSONDecodeError as e:
                    return ConfigLoadResult(
                        success=False,
                        file_path=str(resolved_path),
                        error_message=f"JSON parsing error: {e}",
                        config_hash=config_hash
                    )
                
                # Substitute environment variables if requested
                if substitute_env:
                    config = self._substitute_env_vars(config)
                
                load_time = time.time() - start_time
                
                result = ConfigLoadResult(
                    success=True,
                    config=config,
                    config_type=ConfigType.JSON,
                    file_path=str(resolved_path),
                    load_time=load_time,
                    config_hash=config_hash
                )
                
                # Cache result
                with self._cache_lock:
                    self.config_cache[cache_key] = result
                    self._update_file_watcher(str(resolved_path))
                
                return result
                
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Error loading JSON config: {e}")
            return ConfigLoadResult(
                success=False,
                file_path=str(resolved_path),
                error_message=f"Unexpected error: {e}",
                load_time=load_time
            )
    
    def load_config(self, file_path: str, substitute_env: bool = True) -> ConfigLoadResult:
        """Load configuration file with automatic type detection."""
        if not file_path:
            return ConfigLoadResult(
                success=False,
                error_message="File path is required"
            )
        
        config_type = self._detect_config_type(file_path)
        
        if config_type == ConfigType.YAML:
            return self.load_yaml_config(file_path, substitute_env)
        elif config_type == ConfigType.JSON:
            return self.load_json_config(file_path, substitute_env)
        else:
            return ConfigLoadResult(
                success=False,
                file_path=file_path,
                error_message=f"Unsupported configuration type: {config_type}"
            )
    
    def validate_config(self, config: Dict[str, Any], schema: ConfigSchema) -> ConfigValidationResult:
        """Validate configuration against JSON schema."""
        if not JSONSCHEMA_AVAILABLE:
            return ConfigValidationResult(
                valid=False,
                errors=["python-jsonschema is not available. Install with: pip install jsonschema"]
            )
        
        start_time = time.time()
        
        # Generate hashes for caching
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        schema_str = json.dumps(schema.schema, sort_keys=True)
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()
        
        cache_key = f"{config_hash}:{schema_hash}"
        
        # Check cache first
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        errors = []
        warnings = []
        schema_errors = []
        
        try:
            # Validate against JSON schema
            validator = validator_for(schema.schema)
            validator.check_schema(schema.schema)
            
            # Perform validation
            validation_errors = list(validator(schema.schema).iter_errors(config))
            
            for error in validation_errors:
                error_info = {
                    'path': list(error.path),
                    'message': error.message,
                    'validator': error.validator,
                    'validator_value': error.validator_value,
                    'instance': error.instance,
                    'schema_path': list(error.schema_path)
                }
                
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"{'.'.join(map(str, error.path))}: {error.message}")
                    schema_errors.append(error_info)
                elif self.validation_level == ValidationLevel.WARN:
                    warnings.append(f"{'.'.join(map(str, error.path))}: {error.message}")
                    schema_errors.append(error_info)
                # In LOOSE mode, we ignore validation errors
            
            # Run custom validators
            for field_name, validator_func in schema.custom_validators.items():
                try:
                    if field_name in config:
                        validator_func(config[field_name], config)
                except Exception as e:
                    error_msg = f"Custom validation failed for {field_name}: {e}"
                    if self.validation_level == ValidationLevel.STRICT:
                        errors.append(error_msg)
                    else:
                        warnings.append(error_msg)
            
            validation_time = time.time() - start_time
            valid = len(errors) == 0
            
            result = ConfigValidationResult(
                valid=valid,
                errors=errors,
                warnings=warnings,
                schema_errors=schema_errors,
                validation_time=validation_time,
                config_hash=config_hash,
                schema_hash=schema_hash
            )
            
            # Cache result
            with self._cache_lock:
                self.validation_cache[cache_key] = result
            
            return result
            
        except SchemaError as e:
            validation_time = time.time() - start_time
            logger.error(f"Schema error: {e}")
            return ConfigValidationResult(
                valid=False,
                errors=[f"Schema error: {e}"],
                validation_time=validation_time,
                config_hash=config_hash,
                schema_hash=schema_hash
            )
        except Exception as e:
            validation_time = time.time() - start_time
            logger.error(f"Validation error: {e}")
            return ConfigValidationResult(
                valid=False,
                errors=[f"Validation error: {e}"],
                validation_time=validation_time,
                config_hash=config_hash,
                schema_hash=schema_hash
            )
    
    def merge_configs(self, configs: List[Dict[str, Any]], strategy: str = "deep") -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        if not configs:
            return {}
        
        if len(configs) == 1:
            return configs[0].copy()
        
        result = configs[0].copy()
        
        for config in configs[1:]:
            if strategy == "deep":
                result = self._deep_merge(result, config)
            else:
                result.update(config)
        
        return result
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: Dict[str, Any], file_path: str, config_type: ConfigType = None) -> bool:
        """Save configuration to file."""
        if not config:
            logger.error("Configuration is empty")
            return False
        
        if not file_path:
            logger.error("File path is required")
            return False
        
        resolved_path = self._resolve_path(file_path)
        
        if config_type is None:
            config_type = self._detect_config_type(file_path)
        
        try:
            # Create directory if it doesn't exist
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_type == ConfigType.YAML:
                if not PYAML_AVAILABLE:
                    logger.error("PyYAML is not available")
                    return False
                
                with open(resolved_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    yaml.dump(config, f, default_flow_style=False, indent=2, Dumper=Dumper)
            
            elif config_type == ConfigType.JSON:
                with open(resolved_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            else:
                logger.error(f"Unsupported config type for saving: {config_type}")
                return False
            
            logger.info(f"Configuration saved to {resolved_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def add_hot_reload_callback(self, file_path: str, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for hot reload when configuration file changes."""
        if not file_path:
            return
        
        resolved_path = str(self._resolve_path(file_path))
        
        if resolved_path not in self._hot_reload_callbacks:
            self._hot_reload_callbacks[resolved_path] = []
        
        self._hot_reload_callbacks[resolved_path].append(callback)
        logger.info(f"Added hot reload callback for {resolved_path}")
    
    def check_hot_reload(self) -> List[str]:
        """Check for configuration files that need reloading."""
        reloaded_files = []
        
        for file_path in self._hot_reload_callbacks.keys():
            if self._is_file_modified(file_path):
                try:
                    # Reload configuration
                    load_result = self.load_config(file_path)
                    
                    if load_result.success and load_result.config:
                        # Call all callbacks
                        for callback in self._hot_reload_callbacks[file_path]:
                            try:
                                callback(load_result.config)
                            except Exception as e:
                                logger.error(f"Hot reload callback error: {e}")
                        
                        reloaded_files.append(file_path)
                        logger.info(f"Hot reloaded configuration: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Hot reload error for {file_path}: {e}")
        
        return reloaded_files
    
    def clear_cache(self) -> Any:
        """Clear all caches."""
        with self._cache_lock:
            self.config_cache.clear()
            self.validation_cache.clear()
            self._file_watchers.clear()
        logger.info("Configuration caches cleared")


class SchemaBuilder:
    """Builder for creating JSON schemas programmatically."""
    
    def __init__(self) -> Any:
        """Initialize schema builder."""
        self.schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def add_string_field(self, name: str, required: bool = False, pattern: str = None, 
                        min_length: int = None, max_length: int = None, 
                        enum: List[str] = None) -> 'SchemaBuilder':
        """Add string field to schema."""
        field_schema = {"type": "string"}
        
        if pattern:
            field_schema["pattern"] = pattern
        if min_length is not None:
            field_schema["minLength"] = min_length
        if max_length is not None:
            field_schema["maxLength"] = max_length
        if enum:
            field_schema["enum"] = enum
        
        self.schema["properties"][name] = field_schema
        
        if required:
            self.schema["required"].append(name)
        
        return self
    
    def add_number_field(self, name: str, required: bool = False, minimum: float = None, 
                        maximum: float = None, multiple_of: float = None) -> 'SchemaBuilder':
        """Add number field to schema."""
        field_schema = {"type": "number"}
        
        if minimum is not None:
            field_schema["minimum"] = minimum
        if maximum is not None:
            field_schema["maximum"] = maximum
        if multiple_of is not None:
            field_schema["multipleOf"] = multiple_of
        
        self.schema["properties"][name] = field_schema
        
        if required:
            self.schema["required"].append(name)
        
        return self
    
    def add_integer_field(self, name: str, required: bool = False, minimum: int = None, 
                         maximum: int = None) -> 'SchemaBuilder':
        """Add integer field to schema."""
        field_schema = {"type": "integer"}
        
        if minimum is not None:
            field_schema["minimum"] = minimum
        if maximum is not None:
            field_schema["maximum"] = maximum
        
        self.schema["properties"][name] = field_schema
        
        if required:
            self.schema["required"].append(name)
        
        return self
    
    def add_boolean_field(self, name: str, required: bool = False) -> 'SchemaBuilder':
        """Add boolean field to schema."""
        self.schema["properties"][name] = {"type": "boolean"}
        
        if required:
            self.schema["required"].append(name)
        
        return self
    
    def add_array_field(self, name: str, item_schema: Dict[str, Any], required: bool = False,
                       min_items: int = None, max_items: int = None) -> 'SchemaBuilder':
        """Add array field to schema."""
        field_schema = {
            "type": "array",
            "items": item_schema
        }
        
        if min_items is not None:
            field_schema["minItems"] = min_items
        if max_items is not None:
            field_schema["maxItems"] = max_items
        
        self.schema["properties"][name] = field_schema
        
        if required:
            self.schema["required"].append(name)
        
        return self
    
    def add_object_field(self, name: str, properties: Dict[str, Any], required: bool = False,
                        required_properties: List[str] = None) -> 'SchemaBuilder':
        """Add object field to schema."""
        field_schema = {
            "type": "object",
            "properties": properties
        }
        
        if required_properties:
            field_schema["required"] = required_properties
        
        self.schema["properties"][name] = field_schema
        
        if required:
            self.schema["required"].append(name)
        
        return self
    
    def add_custom_field(self, name: str, field_schema: Dict[str, Any], required: bool = False) -> 'SchemaBuilder':
        """Add custom field schema."""
        self.schema["properties"][name] = field_schema
        
        if required:
            self.schema["required"].append(name)
        
        return self
    
    def build(self) -> ConfigSchema:
        """Build the final schema."""
        return ConfigSchema(schema=self.schema.copy())


# Example usage functions
def demonstrate_yaml_loading():
    """Demonstrate YAML configuration loading."""
    if not PYAML_AVAILABLE:
        logger.error("PyYAML not available")
        return
    
    # Create sample YAML config
    sample_config = """
app:
  name: "My Application"
  version: "1.0.0"
  debug: true

database:
  host: ${DB_HOST:-localhost}
  port: ${DB_PORT:-5432}
  name: ${DB_NAME:-myapp}
  user: ${DB_USER:-postgres}
  password: ${DB_PASSWORD}

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "app.log"
"""
    
    # Write sample config to file
    config_file = "sample_config.yaml"
    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write(sample_config)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        manager = ConfigManager()
        result = manager.load_yaml_config(config_file)
        
        if result.success:
            logger.info(f"Loaded config: {result.config}")
            logger.info(f"Load time: {result.load_time:.3f}s")
        else:
            logger.error(f"Failed to load config: {result.error_message}")
    
    finally:
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)


def demonstrate_json_schema_validation():
    """Demonstrate JSON schema validation."""
    if not JSONSCHEMA_AVAILABLE:
        logger.error("python-jsonschema not available")
        return
    
    # Create schema using builder
    schema_builder = SchemaBuilder()
    schema = (schema_builder
              .add_string_field("app_name", required=True, min_length=1, max_length=100)
              .add_string_field("version", required=True, pattern=r"^\d+\.\d+\.\d+$")
              .add_boolean_field("debug", required=False)
              .add_object_field("database", {
                  "host": {"type": "string"},
                  "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                  "name": {"type": "string"},
                  "user": {"type": "string"},
                  "password": {"type": "string"}
              }, required=True, required_properties=["host", "port", "name"])
              .add_object_field("server", {
                  "host": {"type": "string"},
                  "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                  "workers": {"type": "integer", "minimum": 1, "maximum": 100}
              }, required=True)
              .build())
    
    # Sample configuration
    config = {
        "app_name": "My Application",
        "version": "1.0.0",
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp",
            "user": "postgres",
            "password": "secret"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4
        }
    }
    
    # Validate configuration
    manager = ConfigManager()
    validation_result = manager.validate_config(config, schema)
    
    if validation_result.valid:
        logger.info("Configuration is valid")
        if validation_result.warnings:
            logger.warning(f"Warnings: {validation_result.warnings}")
    else:
        logger.error(f"Configuration validation failed:")
        for error in validation_result.errors:
            logger.error(f"  - {error}")


def demonstrate_config_merging():
    """Demonstrate configuration merging."""
    # Base configuration
    base_config = {
        "app": {
            "name": "My App",
            "version": "1.0.0"
        },
        "database": {
            "host": "localhost",
            "port": 5432
        }
    }
    
    # Development overrides
    dev_config = {
        "app": {
            "debug": True
        },
        "database": {
            "name": "dev_db"
        },
        "logging": {
            "level": "DEBUG"
        }
    }
    
    # Production overrides
    prod_config = {
        "app": {
            "debug": False
        },
        "database": {
            "host": "prod-server",
            "name": "prod_db"
        },
        "logging": {
            "level": "WARNING"
        }
    }
    
    manager = ConfigManager()
    
    # Merge configurations
    merged_config = manager.merge_configs([base_config, dev_config, prod_config], strategy="deep")
    
    logger.info("Merged configuration:")
    logger.info(json.dumps(merged_config, indent=2))


def demonstrate_hot_reload():
    """Demonstrate hot reload functionality."""
    if not PYAML_AVAILABLE:
        logger.error("PyYAML not available")
        return
    
    # Create initial config
    initial_config = """
app:
  name: "Initial App"
  version: "1.0.0"
"""
    
    config_file = "hot_reload_config.yaml"
    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write(initial_config)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        manager = ConfigManager()
        
        # Add hot reload callback
        def config_changed(new_config) -> Any:
            logger.info(f"Configuration changed: {new_config}")
        
        manager.add_hot_reload_callback(config_file, config_changed)
        
        # Load initial config
        result = manager.load_config(config_file)
        if result.success:
            logger.info(f"Initial config: {result.config}")
        
        # Simulate config change
        updated_config = """
app:
  name: "Updated App"
  version: "2.0.0"
"""
        
        time.sleep(1)  # Ensure file modification time changes
        
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(updated_config)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Check for hot reload
        reloaded_files = manager.check_hot_reload()
        if reloaded_files:
            logger.info(f"Hot reloaded files: {reloaded_files}")
    
    finally:
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)


def main():
    """Main function demonstrating configuration management."""
    logger.info("Starting configuration management examples")
    
    # Demonstrate YAML loading
    try:
        demonstrate_yaml_loading()
    except Exception as e:
        logger.error(f"YAML loading demonstration failed: {e}")
    
    # Demonstrate JSON schema validation
    try:
        demonstrate_json_schema_validation()
    except Exception as e:
        logger.error(f"JSON schema validation demonstration failed: {e}")
    
    # Demonstrate config merging
    try:
        demonstrate_config_merging()
    except Exception as e:
        logger.error(f"Config merging demonstration failed: {e}")
    
    # Demonstrate hot reload
    try:
        demonstrate_hot_reload()
    except Exception as e:
        logger.error(f"Hot reload demonstration failed: {e}")
    
    logger.info("Configuration management examples completed")


match __name__:
    case "__main__":
    main() 