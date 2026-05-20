from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
    import yaml
    from jsonschema import validate, ValidationError
    from jsonschema.validators import Draft7Validator
from ..core import BaseConfig
from typing import Any, List, Dict, Optional
"""
Enhanced configuration management with PyYAML and jsonschema.
Provides robust configuration loading, validation, and management.
"""


# Optional imports for configuration handling
try:
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


@dataclass
class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None

@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    schema: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)

@dataclass
class SecurityConfig:
    """Security configuration with validation."""
    timeout: float = 10.0
    max_workers: int = 50
    retry_count: int = 3
    verify_ssl: bool = True
    user_agent: str = "Security Scanner"
    log_level: str = "INFO"
    output_format: str = "json"
    enable_colors: bool = True
    
    def validate(self) -> bool:
        """Validate configuration values."""
        if self.timeout <= 0:
            raise ConfigValidationError("Timeout must be positive", "timeout", self.timeout)
        if self.max_workers <= 0:
            raise ConfigValidationError("Max workers must be positive", "max_workers", self.max_workers)
        if self.retry_count < 0:
            raise ConfigValidationError("Retry count must be non-negative", "retry_count", self.retry_count)
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigValidationError("Invalid log level", "log_level", self.log_level)
        if self.output_format not in ["json", "yaml", "xml", "csv"]:
            raise ConfigValidationError("Invalid output format", "output_format", self.output_format)
        return True

class ConfigManager:
    """Enhanced configuration manager with YAML and JSON Schema support."""
    
    def __init__(self, config_dir: str = "configs"):
        
    """__init__ function."""
self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.schemas: Dict[str, ConfigSchema] = {}
        self.configs: Dict[str, Any] = {}
        
        # Initialize default schemas
        self._init_default_schemas()
    
    def _init_default_schemas(self) -> Any:
        """Initialize default configuration schemas."""
        # Security scanner schema
        security_schema = {
            "type": "object",
            "properties": {
                "timeout": {"type": "number", "minimum": 0.1, "maximum": 300},
                "max_workers": {"type": "integer", "minimum": 1, "maximum": 1000},
                "retry_count": {"type": "integer", "minimum": 0, "maximum": 10},
                "verify_ssl": {"type": "boolean"},
                "user_agent": {"type": "string", "minLength": 1},
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                "output_format": {"type": "string", "enum": ["json", "yaml", "xml", "csv"]},
                "enable_colors": {"type": "boolean"}
            },
            "required": ["timeout", "max_workers", "verify_ssl"],
            "additionalProperties": False
        }
        
        self.schemas["security"] = ConfigSchema(
            name="security",
            version="1.0.0",
            description="Security scanner configuration",
            schema=security_schema,
            required_fields=["timeout", "max_workers", "verify_ssl"],
            optional_fields=["retry_count", "user_agent", "log_level", "output_format", "enable_colors"]
        )
        
        # Network scanner schema
        network_schema = {
            "type": "object",
            "properties": {
                "scan_type": {"type": "string", "enum": ["tcp", "udp", "syn"]},
                "port_range": {"type": "string", "pattern": r"^\d+(-\d+)?(,\d+(-\d+)?)*$"},
                "common_ports": {"type": "boolean"},
                "banner_grab": {"type": "boolean"},
                "ssl_check": {"type": "boolean"},
                "use_nmap": {"type": "boolean"},
                "nmap_arguments": {"type": "string"}
            },
            "required": ["scan_type"],
            "additionalProperties": False
        }
        
        self.schemas["network"] = ConfigSchema(
            name="network",
            version="1.0.0",
            description="Network scanner configuration",
            schema=network_schema,
            required_fields=["scan_type"],
            optional_fields=["port_range", "common_ports", "banner_grab", "ssl_check", "use_nmap", "nmap_arguments"]
        )
    
    def load_yaml_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ConfigValidationError("PyYAML not available")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML format: {e}")
        except FileNotFoundError:
            raise ConfigValidationError(f"Configuration file not found: {file_path}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading YAML config: {e}")
    
    def save_yaml_config(self, config: Dict[str, Any], file_path: str) -> bool:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ConfigValidationError("PyYAML not available")
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(config, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            raise ConfigValidationError(f"Error saving YAML config: {e}")
    
    def load_json_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config = json.load(f)
            return config or {}
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON format: {e}")
        except FileNotFoundError:
            raise ConfigValidationError(f"Configuration file not found: {file_path}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading JSON config: {e}")
    
    def save_json_config(self, config: Dict[str, Any], file_path: str) -> bool:
        """Save configuration to JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            raise ConfigValidationError(f"Error saving JSON config: {e}")
    
    def validate_config(self, config: Dict[str, Any], schema_name: str) -> bool:
        """Validate configuration against schema."""
        if not JSONSCHEMA_AVAILABLE:
            self.logger.warning("jsonschema not available, skipping validation")
            return True
        
        if schema_name not in self.schemas:
            raise ConfigValidationError(f"Schema '{schema_name}' not found")
        
        schema = self.schemas[schema_name]
        
        try:
            validate(instance=config, schema=schema.schema)
            return True
        except ValidationError as e:
            raise ConfigValidationError(
                f"Configuration validation failed: {e.message}",
                field=str(e.path),
                value=e.instance
            )
    
    def create_default_config(self, config_type: str) -> Dict[str, Any]:
        """Create default configuration for given type."""
        if config_type not in self.schemas:
            raise ConfigValidationError(f"Unknown config type: {config_type}")
        
        schema = self.schemas[config_type]
        default_config = {}
        
        # Add required fields with defaults
        for field in schema.required_fields:
            if field == "timeout":
                default_config[field] = 10.0
            elif field == "max_workers":
                default_config[field] = 50
            elif field == "verify_ssl":
                default_config[field] = True
            elif field == "scan_type":
                default_config[field] = "tcp"
            else:
                default_config[field] = None
        
        # Add optional fields with defaults
        for field in schema.optional_fields:
            if field == "retry_count":
                default_config[field] = 3
            elif field == "user_agent":
                default_config[field] = "Security Scanner"
            elif field == "log_level":
                default_config[field] = "INFO"
            elif field == "output_format":
                default_config[field] = "json"
            elif field == "enable_colors":
                default_config[field] = True
            elif field == "common_ports":
                default_config[field] = True
            elif field == "banner_grab":
                default_config[field] = True
            elif field == "ssl_check":
                default_config[field] = True
            elif field == "use_nmap":
                default_config[field] = True
            elif field == "nmap_arguments":
                default_config[field] = "-sS -sV -O"
            else:
                default_config[field] = None
        
        return default_config
    
    async def load_config_async(self, file_path: str, config_type: str = None) -> Dict[str, Any]:
        """Load configuration asynchronously."""
        try:
            # Determine file type and load
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                config = await asyncio.get_event_loop().run_in_executor(
                    None, self.load_yaml_config, file_path
                )
            elif file_path.endswith('.json'):
                config = await asyncio.get_event_loop().run_in_executor(
                    None, self.load_json_config, file_path
                )
            else:
                raise ConfigValidationError(f"Unsupported file format: {file_path}")
            
            # Validate if schema is provided
            if config_type and config_type in self.schemas:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.validate_config, config, config_type
                )
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config {file_path}: {e}")
            raise
    
    async def save_config_async(self, config: Dict[str, Any], file_path: str) -> bool:
        """Save configuration asynchronously."""
        try:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return await asyncio.get_event_loop().run_in_executor(
                    None, self.save_yaml_config, config, file_path
                )
            elif file_path.endswith('.json'):
                return await asyncio.get_event_loop().run_in_executor(
                    None, self.save_json_config, config, file_path
                )
            else:
                raise ConfigValidationError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving config {file_path}: {e}")
            raise
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations with override precedence."""
        merged = base_config.copy()
        
        def deep_merge(base: Dict, override: Dict):
            
    """deep_merge function."""
for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged, override_config)
        return merged
    
    def get_config_path(self, config_name: str, config_type: str = "yaml") -> str:
        """Get configuration file path."""
        if config_type not in ["yaml", "yml", "json"]:
            raise ConfigValidationError(f"Unsupported config type: {config_type}")
        
        extension = "yml" if config_type == "yaml" else config_type
        return str(self.config_dir / f"{config_name}.{extension}")
    
    def list_configs(self) -> List[str]:
        """List available configuration files."""
        configs = []
        for file_path in self.config_dir.glob("*.*"):
            if file_path.suffix in ['.yaml', '.yml', '.json']:
                configs.append(file_path.stem)
        return sorted(configs)
    
    def validate_config_file(self, file_path: str, schema_name: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        # Load config
        if file_path.endswith(('.yaml', '.yml')):
            config = self.load_yaml_config(file_path)
        elif file_path.endswith('.json'):
            config = self.load_json_config(file_path)
        else:
            raise ConfigValidationError(f"Unsupported file format: {file_path}")
        
        # Validate config
        self.validate_config(config, schema_name)
        
        return config
    
    def create_config_template(self, config_type: str, file_path: str) -> bool:
        """Create configuration template file."""
        try:
            default_config = self.create_default_config(config_type)
            
            if file_path.endswith(('.yaml', '.yml')):
                return self.save_yaml_config(default_config, file_path)
            elif file_path.endswith('.json'):
                return self.save_json_config(default_config, file_path)
            else:
                raise ConfigValidationError(f"Unsupported file format: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error creating config template: {e}")
            raise

# Utility functions for configuration management
def load_security_config(file_path: str) -> SecurityConfig:
    """Load and validate security configuration."""
    manager = ConfigManager()
    config_data = manager.validate_config_file(file_path, "security")
    
    return SecurityConfig(**config_data)

def create_default_security_config(file_path: str) -> bool:
    """Create default security configuration file."""
    manager = ConfigManager()
    return manager.create_config_template("security", file_path)

def validate_config_schema(config: Dict[str, Any], schema_name: str) -> bool:
    """Validate configuration against schema."""
    manager = ConfigManager()
    return manager.validate_config(config, schema_name) 