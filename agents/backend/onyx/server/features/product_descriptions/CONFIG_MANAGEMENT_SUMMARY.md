# Configuration Management with PyYAML and python-jsonschema

## Overview

This document summarizes the implementation of a comprehensive configuration management system using PyYAML for configuration file handling and python-jsonschema for validation. The system provides robust, type-safe configuration management with environment variable overrides, caching, and FastAPI integration.

## Key Features

### 1. YAML Configuration Loading
- **File-based configuration**: Load configurations from YAML files
- **Safe loading**: Uses `yaml.safe_load()` for security
- **Error handling**: Comprehensive error handling for file operations
- **Encoding support**: UTF-8 encoding for international character support

### 2. JSON Schema Validation
- **Type validation**: Ensures configuration values match expected types
- **Range validation**: Validates numeric values within acceptable ranges
- **Required fields**: Enforces required configuration fields
- **Enum validation**: Validates string values against allowed options

### 3. Configuration Classes
- **SecurityConfig**: Security-related settings (scan duration, rate limits, ports)
- **DatabaseConfig**: Database connection settings
- **LoggingConfig**: Logging configuration settings
- **Type safety**: Dataclasses with type hints for compile-time validation

### 4. Environment Variable Overrides
- **Runtime overrides**: Override configuration values via environment variables
- **Selective overrides**: Only override specific configuration sections
- **Type conversion**: Automatic type conversion for environment variable values
- **Validation**: Override values are validated against schemas

### 5. Caching and Performance
- **LRU caching**: Cached configuration retrieval for performance
- **Cache invalidation**: Automatic cache clearing on configuration updates
- **Memory efficiency**: Efficient memory usage with lazy loading

### 6. FastAPI Integration
- **Dependency injection**: FastAPI dependency functions for configuration
- **Type safety**: Pydantic models for API request/response validation
- **Error handling**: Proper HTTP error responses for configuration issues

## Architecture

### Core Components

#### ConfigManager Class
```python
class ConfigManager:
    """Configuration manager with YAML and JSON schema validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config_cache = {}
        self.schemas = {...}  # JSON schemas for validation
```

#### Configuration Dataclasses
```python
@dataclass
class SecurityConfig:
    max_scan_duration: int = 300
    rate_limit_per_minute: int = 60
    allowed_ports: List[int] = None
    blocked_ips: List[str] = None
```

#### JSON Schemas
```python
schemas = {
    "security": {
        "type": "object",
        "properties": {
            "max_scan_duration": {"type": "integer", "minimum": 1, "maximum": 3600},
            "rate_limit_per_minute": {"type": "integer", "minimum": 1, "maximum": 1000},
            # ... more properties
        },
        "required": ["max_scan_duration", "rate_limit_per_minute"]
    }
}
```

### File Structure

```
dependencies/
├── config_helpers.py          # Main configuration management
├── crypto_helpers.py          # Cryptographic utilities
├── http_helpers.py           # HTTP client management
├── nmap_helpers.py           # Nmap integration
├── scapy_helpers.py          # Packet crafting utilities
└── ssh_helpers.py            # SSH operations

config_demo.py                 # Configuration management demo
test_config_management.py     # Comprehensive test suite
CONFIG_MANAGEMENT_SUMMARY.md  # This documentation
```

## Usage Examples

### Basic Configuration Management

```python
# Initialize configuration manager
config_manager = ConfigManager("config.yaml")

# Load configuration
config = config_manager.get_config()

# Get specific section
security_config = config_manager.get_section("security")

# Update configuration
new_security_config = {
    "max_scan_duration": 600,
    "rate_limit_per_minute": 120
}
config_manager.update_section("security", new_security_config)
```

### Environment Variable Overrides

```python
# Set environment variables
os.environ["SECURITY_MAX_SCAN_DURATION"] = "900"
os.environ["DB_HOST"] = "production-db.example.com"

# Apply overrides
config_manager.apply_env_overrides()
```

### FastAPI Integration

```python
from fastapi import Depends

# Dependency functions
def get_config_manager() -> ConfigManager:
    return ConfigManager()

def get_security_config(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> SecurityConfig:
    security_data = config_manager.get_section("security")
    return SecurityConfig(**security_data)

# Use in routes
@router.get("/security/config")
async def get_security_config_endpoint(
    security_config: SecurityConfig = Depends(get_security_config)
) -> SecurityConfig:
    return security_config
```

### Configuration Validation

```python
# Validate configuration content
validation_result = await validate_config_content(
    ConfigValidationRequest(config_content=yaml_content),
    config_manager
)

if not validation_result.is_valid:
    for error in validation_result.errors:
        print(f"Validation error: {error}")
```

## Configuration Schema

### Security Configuration
```yaml
security:
  max_scan_duration: 300          # Maximum scan duration in seconds (1-3600)
  rate_limit_per_minute: 60       # Rate limit per minute (1-1000)
  allowed_ports: [22, 80, 443]    # List of allowed ports
  blocked_ips: []                 # List of blocked IP addresses
```

### Database Configuration
```yaml
database:
  host: localhost                 # Database host
  port: 5432                      # Database port (1-65535)
  database: security_tools        # Database name
  username: admin                 # Database username
  password: secret123             # Database password
  pool_size: 10                   # Connection pool size (1-100)
  max_overflow: 20                # Maximum overflow connections (0+)
```

### Logging Configuration
```yaml
logging:
  level: INFO                     # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: logs/app.log         # Log file path (optional)
  max_file_size: 10485760         # Maximum file size in bytes (1024+)
  backup_count: 5                 # Number of backup files (0+)
```

## Environment Variables

### Security Overrides
- `SECURITY_MAX_SCAN_DURATION`: Override maximum scan duration
- `SECURITY_RATE_LIMIT`: Override rate limit per minute

### Database Overrides
- `DB_HOST`: Override database host
- `DB_PORT`: Override database port
- `DB_NAME`: Override database name

### Logging Overrides
- `LOG_LEVEL`: Override log level

## Error Handling

### Configuration Loading Errors
- **FileNotFoundError**: Configuration file not found
- **yaml.YAMLError**: Invalid YAML format
- **ValueError**: Configuration validation failed

### Validation Errors
- **Type errors**: Invalid data types
- **Range errors**: Values outside acceptable ranges
- **Required field errors**: Missing required fields
- **Enum errors**: Invalid enum values

### Environment Variable Errors
- **Type conversion errors**: Invalid numeric values
- **Validation errors**: Environment values fail schema validation

## Best Practices

### 1. Configuration Design
- Use descriptive section names
- Group related settings together
- Provide sensible defaults
- Document all configuration options

### 2. Validation
- Define comprehensive JSON schemas
- Validate all configuration sections
- Provide clear error messages
- Test validation with edge cases

### 3. Environment Variables
- Use consistent naming conventions
- Document all environment variables
- Validate environment variable values
- Provide fallback values

### 4. Performance
- Use caching for frequently accessed configurations
- Implement lazy loading for large configurations
- Clear cache when configurations change
- Monitor memory usage

### 5. Security
- Use `yaml.safe_load()` to prevent code execution
- Validate all configuration inputs
- Sanitize environment variable values
- Use secure file permissions

## Testing

### Unit Tests
- Configuration loading and saving
- Schema validation
- Environment variable overrides
- Error handling scenarios

### Integration Tests
- FastAPI dependency injection
- Configuration updates
- Cache invalidation
- End-to-end workflows

### Performance Tests
- Configuration loading performance
- Cache efficiency
- Memory usage
- Concurrent access

## Monitoring and Logging

### Configuration Changes
- Log all configuration updates
- Track configuration validation errors
- Monitor environment variable overrides
- Alert on configuration issues

### Performance Metrics
- Configuration loading time
- Cache hit rates
- Memory usage
- Validation performance

## Security Considerations

### Input Validation
- Validate all configuration inputs
- Sanitize environment variable values
- Prevent code injection via YAML
- Use secure file handling

### Access Control
- Restrict configuration file access
- Use secure file permissions
- Implement configuration encryption for sensitive data
- Audit configuration changes

### Error Handling
- Don't expose sensitive information in error messages
- Log security-relevant events
- Implement rate limiting for configuration updates
- Validate configuration changes

## Future Enhancements

### 1. Configuration Encryption
- Encrypt sensitive configuration values
- Support for encrypted environment variables
- Key management integration

### 2. Dynamic Configuration
- Runtime configuration updates
- Configuration change notifications
- Hot reloading capabilities

### 3. Configuration Templates
- Template-based configuration generation
- Environment-specific templates
- Configuration inheritance

### 4. Advanced Validation
- Cross-field validation
- Custom validation rules
- Validation rule composition

### 5. Configuration Versioning
- Configuration version tracking
- Migration support
- Rollback capabilities

## Conclusion

The configuration management system provides a robust, type-safe, and performant solution for managing application configuration. With comprehensive validation, environment variable support, and FastAPI integration, it ensures reliable configuration handling in production environments.

The system follows security best practices, provides comprehensive error handling, and includes extensive testing coverage. It's designed to be extensible and maintainable, supporting future enhancements and evolving requirements. 