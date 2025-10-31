from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import tempfile
import os

from dependencies.config_helpers import (
        import yaml
    import uvicorn
from typing import Any, List, Dict, Optional
import logging
# Import our configuration manager
    ConfigManager, SecurityConfig, DatabaseConfig, LoggingConfig,
    get_config_manager, get_security_config, get_database_config, get_logging_config
)

# Data Models
class ConfigUpdateRequest(BaseModel):
    section: str = Field(..., min_length=1)
    config_data: Dict[str, Any] = Field(..., description="Configuration data to update")

class ConfigValidationRequest(BaseModel):
    config_content: str = Field(..., min_length=1, description="YAML configuration content")

class ConfigValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class ConfigSectionResponse(BaseModel):
    section: str
    config: Dict[str, Any]
    schema_validation: bool

# Router for configuration management
router = APIRouter(prefix="/config", tags=["Configuration Management"])

@router.get("/", response_model=Dict[str, Any])
async def get_full_config(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """Get full configuration"""
    return config_manager.get_config()

@router.get("/sections", response_model=List[str])
async def get_config_sections(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> List[str]:
    """Get available configuration sections"""
    config = config_manager.get_config()
    return list(config.keys())

@router.get("/{section}", response_model=ConfigSectionResponse)
async def get_config_section(
    section: str,
    config_manager: ConfigManager = Depends(get_config_manager)
) -> ConfigSectionResponse:
    """Get specific configuration section"""
    try:
        config_data = config_manager.get_section(section)
        is_valid = True
        try:
            config_manager.validate_config_section(config_data, section)
        except ValueError:
            is_valid = False
        
        return ConfigSectionResponse(
            section=section,
            config=config_data,
            schema_validation=is_valid
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get section {section}: {str(e)}"
        )

@router.put("/{section}", response_model=Dict[str, Any])
async def update_config_section(
    section: str,
    request: ConfigUpdateRequest,
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """Update configuration section"""
    try:
        config_manager.update_section(section, request.config_data)
        return {"message": f"Section {section} updated successfully", "section": section}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration validation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update section {section}: {str(e)}"
        )

@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_config_content(
    request: ConfigValidationRequest,
    config_manager: ConfigManager = Depends(get_config_manager)
) -> ConfigValidationResponse:
    """Validate YAML configuration content"""
    errors = []
    warnings = []
    
    try:
        # Parse YAML content
        config_data = yaml.safe_load(request.config_content)
        
        if not config_data:
            errors.append("Configuration is empty")
            return ConfigValidationResponse(is_valid=False, errors=errors)
        
        # Validate each section
        for section, section_config in config_data.items():
            try:
                config_manager.validate_config_section(section_config, section)
            except ValueError as e:
                errors.append(f"Section '{section}': {str(e)}")
            except Exception as e:
                warnings.append(f"Section '{section}': Unexpected validation issue - {str(e)}")
        
        is_valid = len(errors) == 0
        return ConfigValidationResponse(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
        
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML format: {str(e)}")
        return ConfigValidationResponse(is_valid=False, errors=errors)
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
        return ConfigValidationResponse(is_valid=False, errors=errors)

@router.post("/initialize")
async def initialize_config(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, str]:
    """Initialize configuration with defaults"""
    try:
        config_manager.initialize_config()
        return {"message": "Configuration initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize configuration: {str(e)}"
        )

@router.post("/reload")
async def reload_config(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """Reload configuration from file"""
    try:
        config = config_manager.reload_config()
        return {"message": "Configuration reloaded successfully", "sections": list(config.keys())}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload configuration: {str(e)}"
        )

@router.get("/security/config", response_model=SecurityConfig)
async def get_security_config_endpoint(
    security_config: SecurityConfig = Depends(get_security_config)
) -> SecurityConfig:
    """Get security configuration"""
    return security_config

@router.get("/database/config", response_model=DatabaseConfig)
async def get_database_config_endpoint(
    database_config: DatabaseConfig = Depends(get_database_config)
) -> DatabaseConfig:
    """Get database configuration"""
    return database_config

@router.get("/logging/config", response_model=LoggingConfig)
async def get_logging_config_endpoint(
    logging_config: LoggingConfig = Depends(get_logging_config)
) -> LoggingConfig:
    """Get logging configuration"""
    return logging_config

@router.get("/env-overrides", response_model=Dict[str, Any])
async def get_environment_overrides(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, Any]:
    """Get configuration overrides from environment variables"""
    return config_manager.get_env_overrides()

@router.post("/apply-env-overrides")
async def apply_environment_overrides(
    config_manager: ConfigManager = Depends(get_config_manager)
) -> Dict[str, str]:
    """Apply environment variable overrides to configuration"""
    try:
        config_manager.apply_env_overrides()
        return {"message": "Environment overrides applied successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply environment overrides: {str(e)}"
        )

# Demo functions
async def demo_config_management():
    """Demonstrate configuration management features"""
    print("=== Configuration Management Demo ===\n")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = """
security:
  max_scan_duration: 300
  rate_limit_per_minute: 60
  allowed_ports: [22, 80, 443, 8080, 8443]
  blocked_ips: []

database:
  host: localhost
  port: 5432
  database: security_tools
  username: admin
  password: secret123
  pool_size: 10
  max_overflow: 20

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: logs/app.log
  max_file_size: 10485760
  backup_count: 5
        """
        f.write(config_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        temp_config_path = f.name
    
    try:
        # Initialize config manager
        config_manager = ConfigManager(temp_config_path)
        
        print("1. Loading configuration...")
        config = config_manager.get_config()
        print(f"   Loaded {len(config)} sections: {list(config.keys())}")
        
        print("\n2. Validating configuration...")
        try:
            config_manager.validate_full_config(config)
            print("   ✓ Configuration is valid")
        except ValueError as e:
            print(f"   ✗ Configuration validation failed: {e}")
        
        print("\n3. Getting security configuration...")
        security_config = config_manager.get_section("security")
        print(f"   Max scan duration: {security_config.get('max_scan_duration')}s")
        print(f"   Rate limit: {security_config.get('rate_limit_per_minute')}/min")
        
        print("\n4. Updating configuration...")
        new_security_config = {
            "max_scan_duration": 600,
            "rate_limit_per_minute": 120,
            "allowed_ports": [22, 80, 443, 8080, 8443, 3306],
            "blocked_ips": ["192.168.1.100"]
        }
        config_manager.update_section("security", new_security_config)
        print("   ✓ Security configuration updated")
        
        print("\n5. Environment overrides...")
        # Set some environment variables
        os.environ["SECURITY_MAX_SCAN_DURATION"] = "900"
        os.environ["DB_HOST"] = "production-db.example.com"
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        overrides = config_manager.get_env_overrides()
        print(f"   Found {len(overrides)} environment overrides")
        for section, section_overrides in overrides.items():
            print(f"   {section}: {section_overrides}")
        
        print("\n6. Applying environment overrides...")
        config_manager.apply_env_overrides()
        updated_config = config_manager.get_config()
        print(f"   Updated max scan duration: {updated_config['security']['max_scan_duration']}")
        print(f"   Updated database host: {updated_config['database']['host']}")
        print(f"   Updated log level: {updated_config['logging']['level']}")
        
        print("\n7. Configuration validation demo...")
        invalid_config = """
security:
  max_scan_duration: -1  # Invalid: negative value
  rate_limit_per_minute: 2000  # Invalid: exceeds maximum
database:
  port: 99999  # Invalid: exceeds maximum port number
        """
        
        validation_result = await validate_config_content(
            ConfigValidationRequest(config_content=invalid_config),
            config_manager
        )
        
        if not validation_result.is_valid:
            print("   ✗ Invalid configuration detected:")
            for error in validation_result.errors:
                print(f"     - {error}")
        
        print("\n=== Demo completed successfully! ===")
        
    finally:
        # Clean up temporary file
        os.unlink(temp_config_path)

# FastAPI app setup
app = FastAPI(title="Configuration Management Demo", version="1.0.0")
app.include_router(router)

if __name__ == "__main__":
    print("Starting Configuration Management Demo...")
    print("Access the API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    
    # Run demo
    asyncio.run(demo_config_management())
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000) 