from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import yaml
import json
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from datetime import datetime
from error_handling_core import (
from error_handling_core import validate_ip_address, validate_port_number
    from jsonschema import validate, ValidationError as JsonSchemaValidationError
    from jsonschema import Draft7Validator
from typing import Any, List, Dict, Optional
import logging
"""
Configuration Management - YAML/JSON config loading and validation
Uses PyYAML and python-jsonschema with proper error handling
"""


# Import error handling components
    ValidationError, ErrorContext, ValidationResult, OperationResult,
    log_error_with_context, create_error_context, handle_validation_errors
)

# Import validation functions

# Try to import jsonschema
try:
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("Warning: jsonschema not available, schema validation disabled")

@dataclass(frozen=True)
class ConfigRequest:
    """Immutable configuration request"""
    config_path: str
    config_type: str = "yaml"  # "yaml", "json"
    schema_path: Optional[str] = None
    validate_schema: bool = True
    environment_overrides: Dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class ConfigResponse:
    """Immutable configuration response"""
    is_successful: bool
    config_data: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    config_source: Optional[str] = None

@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration schema"""
    scan_timeout: int = 30
    max_concurrent_scans: int = 10
    allowed_targets: List[str] = field(default_factory=list)
    blocked_targets: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 100
    enable_logging: bool = True
    log_level: str = "INFO"
    ssl_verify: bool = True

@dataclass(frozen=True)
class NetworkConfig:
    """Network configuration schema"""
    default_ports: List[int] = field(default_factory=lambda: [80, 443, 22, 21, 25, 53])
    scan_timeout: int = 5
    connection_timeout: int = 10
    max_retries: int = 3
    user_agent: str = "SecurityScanner/1.0"

# ============================================================================
# VALIDATION FUNCTIONS (CPU-bound)
# ============================================================================

def validate_config_path(config_path: str) -> ValidationResult:
    """Validate configuration file path - guard clauses with happy path last"""
    # Guard clause: Check if input is string
    if not isinstance(config_path, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_path",
                error_type="type_error",
                error_message="Config path must be a string",
                value=config_path,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if input is empty
    if not config_path.strip():
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_path",
                error_type="empty_value",
                error_message="Config path cannot be empty",
                value=config_path
            )]
        )
    
    # Guard clause: Check if file exists
    if not os.path.exists(config_path):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_path",
                error_type="file_not_found",
                error_message="Configuration file does not exist",
                value=config_path,
                expected_format="existing file path"
            )]
        )
    
    # Guard clause: Check if file is readable
    if not os.access(config_path, os.R_OK):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_path",
                error_type="permission_error",
                error_message="Configuration file is not readable",
                value=config_path,
                expected_format="readable file path"
            )]
        )
    
    # Happy path: All validations passed
    return ValidationResult(is_valid=True)

def validate_config_type(config_type: str) -> ValidationResult:
    """Validate configuration file type - guard clauses with happy path last"""
    # Guard clause: Check if input is string
    if not isinstance(config_type, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_type",
                error_type="type_error",
                error_message="Config type must be a string",
                value=config_type,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if type is valid
    valid_types = ["yaml", "json"]
    if config_type.lower() not in valid_types:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_type",
                error_type="invalid_type",
                error_message=f"Config type must be one of: {valid_types}",
                value=config_type,
                expected_format="yaml or json"
            )]
        )
    
    # Happy path: All validations passed
    return ValidationResult(is_valid=True)

def validate_yaml_content(yaml_content: str) -> ValidationResult:
    """Validate YAML content - guard clauses with happy path last"""
    # Guard clause: Check if input is string
    if not isinstance(yaml_content, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="yaml_content",
                error_type="type_error",
                error_message="YAML content must be a string",
                value=yaml_content,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if input is empty
    if not yaml_content.strip():
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="yaml_content",
                error_type="empty_value",
                error_message="YAML content cannot be empty",
                value=yaml_content
            )]
        )
    
    # Guard clause: Try to parse YAML
    try:
        yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="yaml_content",
                error_type="yaml_parse_error",
                error_message=f"Invalid YAML format: {str(e)}",
                value=yaml_content,
                expected_format="valid YAML"
            )]
        )
    
    # Happy path: All validations passed
    return ValidationResult(is_valid=True)

def validate_json_content(json_content: str) -> ValidationResult:
    """Validate JSON content - guard clauses with happy path last"""
    # Guard clause: Check if input is string
    if not isinstance(json_content, str):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="json_content",
                error_type="type_error",
                error_message="JSON content must be a string",
                value=json_content,
                expected_format="string"
            )]
        )
    
    # Guard clause: Check if input is empty
    if not json_content.strip():
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="json_content",
                error_type="empty_value",
                error_message="JSON content cannot be empty",
                value=json_content
            )]
        )
    
    # Guard clause: Try to parse JSON
    try:
        json.loads(json_content)
    except json.JSONDecodeError as e:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="json_content",
                error_type="json_parse_error",
                error_message=f"Invalid JSON format: {str(e)}",
                value=json_content,
                expected_format="valid JSON"
            )]
        )
    
    # Happy path: All validations passed
    return ValidationResult(is_valid=True)

# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

SECURITY_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "scan_timeout": {
            "type": "integer",
            "minimum": 1,
            "maximum": 300,
            "default": 30
        },
        "max_concurrent_scans": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 10
        },
        "allowed_targets": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "default": []
        },
        "blocked_targets": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "default": []
        },
        "rate_limit_per_minute": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10000,
            "default": 100
        },
        "enable_logging": {
            "type": "boolean",
            "default": True
        },
        "log_level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "default": "INFO"
        },
        "ssl_verify": {
            "type": "boolean",
            "default": True
        }
    },
    "required": ["scan_timeout", "max_concurrent_scans"],
    "additionalProperties": False
}

NETWORK_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "default_ports": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65535
            },
            "default": [80, 443, 22, 21, 25, 53]
        },
        "scan_timeout": {
            "type": "integer",
            "minimum": 1,
            "maximum": 60,
            "default": 5
        },
        "connection_timeout": {
            "type": "integer",
            "minimum": 1,
            "maximum": 300,
            "default": 10
        },
        "max_retries": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "default": 3
        },
        "user_agent": {
            "type": "string",
            "minLength": 1,
            "default": "SecurityScanner/1.0"
        }
    },
    "required": ["default_ports", "scan_timeout"],
    "additionalProperties": False
}

# ============================================================================
# CONFIGURATION LOADING OPERATIONS (I/O-bound)
# ============================================================================

def load_yaml_config_sync(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration synchronously - CPU-bound (wrapped for async)"""
    with open(config_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return yaml.safe_load(file)

def load_json_config_sync(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration synchronously - CPU-bound (wrapped for async)"""
    with open(config_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return json.load(file)

async def load_config_async(config_path: str, config_type: str) -> Dict[str, Any]:
    """Load configuration asynchronously - I/O-bound"""
    loop = asyncio.get_event_loop()
    
    if config_type.lower() == "yaml":
        return await loop.run_in_executor(None, load_yaml_config_sync, config_path)
    else:
        return await loop.run_in_executor(None, load_json_config_sync, config_path)

def validate_config_against_schema(config_data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """Validate configuration against JSON schema - CPU-bound"""
    if not JSONSCHEMA_AVAILABLE:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="schema_validation",
                error_type="dependency_missing",
                error_message="jsonschema not available for validation",
                value="schema_validation_disabled"
            )]
        )
    
    try:
        validate(instance=config_data, schema=schema)
        return ValidationResult(is_valid=True)
    except JsonSchemaValidationError as e:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="schema_validation",
                error_type="schema_validation_error",
                error_message=str(e),
                value=config_data
            )]
        )

def apply_environment_overrides(config_data: Dict[str, Any], overrides: Dict[str, str]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration - CPU-bound"""
    if not overrides:
        return config_data
    
    # Create a copy to avoid modifying original
    modified_config = config_data.copy()
    
    for key, env_var in overrides.items():
        if env_var in os.environ:
            # Handle nested keys (e.g., "database.host")
            keys = key.split('.')
            current = modified_config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = os.environ[env_var]
    
    return modified_config

async def load_configuration_roro(request: ConfigRequest) -> ConfigResponse:
    """Load configuration using RORO pattern - guard clauses with happy path last"""
    # Guard clause: Validate config path
    path_validation = validate_config_path(request.config_path)
    if not path_validation.is_valid:
        return ConfigResponse(
            is_successful=False,
            error_message="Invalid configuration file path"
        )
    
    # Guard clause: Validate config type
    type_validation = validate_config_type(request.config_type)
    if not type_validation.is_valid:
        return ConfigResponse(
            is_successful=False,
            error_message="Invalid configuration file type"
        )
    
    # Guard clause: Validate schema path if provided
    if request.schema_path:
        schema_validation = validate_config_path(request.schema_path)
        if not schema_validation.is_valid:
            return ConfigResponse(
                is_successful=False,
                error_message="Invalid schema file path"
            )
    
    # Happy path: Load configuration
    try:
        # Load configuration file
        config_data = await load_config_async(request.config_path, request.config_type)
        
        # Load schema if provided
        schema_data = None
        if request.schema_path and request.validate_schema:
            schema_data = await load_config_async(request.schema_path, "json")
        
        # Apply environment overrides
        if request.environment_overrides:
            config_data = apply_environment_overrides(config_data, request.environment_overrides)
        
        # Validate against schema if provided
        validation_errors = []
        if schema_data and request.validate_schema:
            schema_validation = validate_config_against_schema(config_data, schema_data)
            if not schema_validation.is_valid:
                validation_errors = [error.error_message for error in schema_validation.errors]
        
        return ConfigResponse(
            is_successful=True,
            config_data=config_data,
            validation_errors=validation_errors,
            config_source=request.config_path
        )
        
    except Exception as e:
        context = create_error_context(
            module_name=__name__,
            function_name="load_configuration_roro",
            parameters={"config_path": request.config_path, "config_type": request.config_type}
        )
        log_error_with_context(e, context)
        
        return ConfigResponse(
            is_successful=False,
            error_message=str(e)
        )

# ============================================================================
# CONFIGURATION VALIDATION OPERATIONS
# ============================================================================

def validate_security_config_roro(config_data: Dict[str, Any]) -> ValidationResult:
    """Validate security configuration - guard clauses with happy path last"""
    # Guard clause: Check if input is dictionary
    if not isinstance(config_data, dict):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_data",
                error_type="type_error",
                error_message="Config data must be a dictionary",
                value=config_data,
                expected_format="dict"
            )]
        )
    
    # Guard clause: Check if jsonschema is available
    if not JSONSCHEMA_AVAILABLE:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="schema_validation",
                error_type="dependency_missing",
                error_message="jsonschema not available for validation",
                value="schema_validation_disabled"
            )]
        )
    
    # Happy path: Validate against schema
    return validate_config_against_schema(config_data, SECURITY_CONFIG_SCHEMA)

def validate_network_config_roro(config_data: Dict[str, Any]) -> ValidationResult:
    """Validate network configuration - guard clauses with happy path last"""
    # Guard clause: Check if input is dictionary
    if not isinstance(config_data, dict):
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="config_data",
                error_type="type_error",
                error_message="Config data must be a dictionary",
                value=config_data,
                expected_format="dict"
            )]
        )
    
    # Guard clause: Check if jsonschema is available
    if not JSONSCHEMA_AVAILABLE:
        return ValidationResult(
            is_valid=False,
            errors=[ValidationError(
                field_name="schema_validation",
                error_type="dependency_missing",
                error_message="jsonschema not available for validation",
                value="schema_validation_disabled"
            )]
        )
    
    # Happy path: Validate against schema
    return validate_config_against_schema(config_data, NETWORK_CONFIG_SCHEMA)

# ============================================================================
# CONFIGURATION TEMPLATE GENERATION
# ============================================================================

def generate_security_config_template() -> str:
    """Generate security configuration template - CPU-bound"""
    template = {
        "scan_timeout": 30,
        "max_concurrent_scans": 10,
        "allowed_targets": [
            "192.168.1.0/24",
            "10.0.0.0/8"
        ],
        "blocked_targets": [
            "127.0.0.1",
            "0.0.0.0"
        ],
        "rate_limit_per_minute": 100,
        "enable_logging": True,
        "log_level": "INFO",
        "ssl_verify": True
    }
    
    return yaml.dump(template, default_flow_style=False, indent=2)

def generate_network_config_template() -> str:
    """Generate network configuration template - CPU-bound"""
    template = {
        "default_ports": [80, 443, 22, 21, 25, 53, 3389, 1433],
        "scan_timeout": 5,
        "connection_timeout": 10,
        "max_retries": 3,
        "user_agent": "SecurityScanner/1.0"
    }
    
    return yaml.dump(template, default_flow_style=False, indent=2)

def generate_config_schema_template() -> str:
    """Generate JSON schema template - CPU-bound"""
    schema_template = {
        "type": "object",
        "properties": {
            "example_field": {
                "type": "string",
                "description": "Example configuration field"
            }
        },
        "required": ["example_field"],
        "additionalProperties": False
    }
    
    return json.dumps(schema_template, indent=2)

# ============================================================================
# CONFIGURATION MERGING OPERATIONS
# ============================================================================

def merge_configurations_roro(configs: List[Dict[str, Any]], merge_strategy: str = "deep") -> Dict[str, Any]:
    """Merge multiple configurations - guard clauses with happy path last"""
    # Guard clause: Check if input is list
    if not isinstance(configs, list):
        return {}
    
    # Guard clause: Check if list is empty
    if len(configs) == 0:
        return {}
    
    # Guard clause: Check if all items are dictionaries
    for i, config in enumerate(configs):
        if not isinstance(config, dict):
            return {}
    
    # Guard clause: Validate merge strategy
    valid_strategies = ["deep", "shallow", "replace"]
    if merge_strategy not in valid_strategies:
        return {}
    
    # Happy path: Merge configurations
    if merge_strategy == "deep":
        return deep_merge_configs(configs)
    elif merge_strategy == "shallow":
        return shallow_merge_configs(configs)
    else:  # replace
        return configs[-1] if configs else {}

def deep_merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep merge configurations - CPU-bound"""
    if not configs:
        return {}
    
    result = configs[0].copy()
    
    for config in configs[1:]:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_configs([result[key], value])
            else:
                result[key] = value
    
    return result

def shallow_merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow merge configurations - CPU-bound"""
    result = {}
    
    for config in configs:
        result.update(config)
    
    return result

# ============================================================================
# CONFIGURATION EXPORT OPERATIONS
# ============================================================================

def export_config_to_yaml_roro(config_data: Dict[str, Any], output_path: str) -> bool:
    """Export configuration to YAML file - guard clauses with happy path last"""
    # Guard clause: Check if config_data is dictionary
    if not isinstance(config_data, dict):
        return False
    
    # Guard clause: Check if output_path is string
    if not isinstance(output_path, str) or not output_path.strip():
        return False
    
    # Guard clause: Check if directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            return False
    
    # Happy path: Export configuration
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, file, default_flow_style=False, indent=2)
        return True
    except Exception:
        return False

def export_config_to_json_roro(config_data: Dict[str, Any], output_path: str) -> bool:
    """Export configuration to JSON file - guard clauses with happy path last"""
    # Guard clause: Check if config_data is dictionary
    if not isinstance(config_data, dict):
        return False
    
    # Guard clause: Check if output_path is string
    if not isinstance(output_path, str) or not output_path.strip():
        return False
    
    # Guard clause: Check if directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            return False
    
    # Happy path: Export configuration
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config_data, file, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Data structures
    "ConfigRequest",
    "ConfigResponse",
    "SecurityConfig",
    "NetworkConfig",
    
    # Schema definitions
    "SECURITY_CONFIG_SCHEMA",
    "NETWORK_CONFIG_SCHEMA",
    
    # Validation functions
    "validate_config_path",
    "validate_config_type",
    "validate_yaml_content",
    "validate_json_content",
    "validate_security_config_roro",
    "validate_network_config_roro",
    
    # Configuration loading
    "load_configuration_roro",
    "load_config_async",
    
    # Template generation
    "generate_security_config_template",
    "generate_network_config_template",
    "generate_config_schema_template",
    
    # Configuration merging
    "merge_configurations_roro",
    "deep_merge_configs",
    "shallow_merge_configs",
    
    # Configuration export
    "export_config_to_yaml_roro",
    "export_config_to_json_roro",
    
    # Dependency flags
    "JSONSCHEMA_AVAILABLE"
] 