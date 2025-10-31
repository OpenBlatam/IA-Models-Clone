"""
Configuration Routes
Real, working configuration management endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, Dict, Any
import asyncio
import os
from config_system import config_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/config", tags=["Configuration Management"])

@router.get("/get-config")
async def get_config(
    section: Optional[str] = None,
    key: Optional[str] = None
):
    """Get configuration value(s)"""
    try:
        result = await config_system.get_config(section, key)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-config")
async def update_config(
    section: str = Form(...),
    key: str = Form(...),
    value: str = Form(...)
):
    """Update configuration value"""
    try:
        # Try to parse value as JSON for complex types
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # If not JSON, use as string
            parsed_value = value
        
        result = await config_system.update_config(section, key, parsed_value)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-config")
async def reset_config(
    section: Optional[str] = Form(None)
):
    """Reset configuration to defaults"""
    try:
        result = await config_system.reset_config(section)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error resetting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export-config")
async def export_config(
    format: str = Form("json")
):
    """Export configuration"""
    try:
        result = await config_system.export_config(format)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error exporting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/import-config")
async def import_config(
    config_data: Dict[str, Any]
):
    """Import configuration"""
    try:
        result = await config_system.import_config(config_data)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error importing configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config-history")
async def get_config_history(
    limit: int = 50
):
    """Get configuration history"""
    try:
        history = config_system.get_config_history(limit)
        return JSONResponse(content={
            "history": history,
            "total_entries": len(config_system.config_history)
        })
    except Exception as e:
        logger.error(f"Error getting configuration history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config-stats")
async def get_config_stats():
    """Get configuration statistics"""
    try:
        stats = config_system.get_config_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting configuration stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environment-config")
async def get_environment_config():
    """Get environment-specific configuration"""
    try:
        env_config = config_system.get_environment_config()
        return JSONResponse(content=env_config)
    except Exception as e:
        logger.error(f"Error getting environment configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-config")
async def download_config(
    format: str = "json"
):
    """Download configuration file"""
    try:
        # Export configuration first
        export_result = await config_system.export_config(format)
        
        if "error" in export_result:
            raise HTTPException(status_code=500, detail=export_result["error"])
        
        filepath = export_result["filepath"]
        filename = export_result["filename"]
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        media_type = "application/json" if format == "json" else "application/x-yaml"
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type=media_type
        )
    except Exception as e:
        logger.error(f"Error downloading configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config-sections")
async def get_config_sections():
    """Get all configuration sections"""
    try:
        config = await config_system.get_config()
        sections = {}
        
        for section_name, section_data in config.items():
            if isinstance(section_data, dict):
                sections[section_name] = {
                    "keys": list(section_data.keys()),
                    "key_count": len(section_data.keys())
                }
            else:
                sections[section_name] = {
                    "type": type(section_data).__name__,
                    "value": section_data
                }
        
        return JSONResponse(content={
            "sections": sections,
            "total_sections": len(sections)
        })
    except Exception as e:
        logger.error(f"Error getting configuration sections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config-validation")
async def validate_config():
    """Validate current configuration"""
    try:
        config = await config_system.get_config()
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "timestamp": config_system.get_config_stats()["last_modified"]
        }
        
        # Validate required sections
        required_sections = ["system", "api", "ai", "security", "monitoring"]
        for section in required_sections:
            if section not in config:
                validation_result["errors"].append(f"Missing required section: {section}")
                validation_result["valid"] = False
        
        # Validate system section
        if "system" in config:
            system_config = config["system"]
            if "name" not in system_config:
                validation_result["errors"].append("Missing system.name")
                validation_result["valid"] = False
            if "version" not in system_config:
                validation_result["errors"].append("Missing system.version")
                validation_result["valid"] = False
        
        # Validate API section
        if "api" in config:
            api_config = config["api"]
            if "port" in api_config:
                try:
                    port = int(api_config["port"])
                    if port < 1 or port > 65535:
                        validation_result["errors"].append("Invalid API port: must be between 1 and 65535")
                        validation_result["valid"] = False
                except (ValueError, TypeError):
                    validation_result["errors"].append("Invalid API port: must be a number")
                    validation_result["valid"] = False
        
        # Validate security section
        if "security" in config:
            security_config = config["security"]
            if "rate_limiting" in security_config:
                rate_limiting = security_config["rate_limiting"]
                if "max_requests_per_minute" in rate_limiting:
                    try:
                        max_req = int(rate_limiting["max_requests_per_minute"])
                        if max_req < 1:
                            validation_result["warnings"].append("Rate limiting max_requests_per_minute should be at least 1")
                    except (ValueError, TypeError):
                        validation_result["errors"].append("Invalid rate limiting max_requests_per_minute")
                        validation_result["valid"] = False
        
        return JSONResponse(content=validation_result)
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backup-config")
async def backup_config():
    """Create configuration backup"""
    try:
        # Export current configuration
        export_result = await config_system.export_config("json")
        
        if "error" in export_result:
            raise HTTPException(status_code=500, detail=export_result["error"])
        
        # Get backup info
        backup_info = {
            "backup_created": True,
            "timestamp": export_result["filename"].split("_")[-1].replace(".json", ""),
            "filename": export_result["filename"],
            "filepath": export_result["filepath"]
        }
        
        return JSONResponse(content=backup_info)
    except Exception as e:
        logger.error(f"Error creating configuration backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config-templates")
async def get_config_templates():
    """Get configuration templates for different environments"""
    try:
        templates = {
            "development": {
                "system": {
                    "environment": "development",
                    "debug": True,
                    "log_level": "DEBUG"
                },
                "api": {
                    "reload": True,
                    "workers": 1
                },
                "monitoring": {
                    "metrics_interval": 30
                }
            },
            "staging": {
                "system": {
                    "environment": "staging",
                    "debug": False,
                    "log_level": "INFO"
                },
                "api": {
                    "reload": False,
                    "workers": 2
                },
                "monitoring": {
                    "metrics_interval": 60
                }
            },
            "production": {
                "system": {
                    "environment": "production",
                    "debug": False,
                    "log_level": "WARNING"
                },
                "api": {
                    "reload": False,
                    "workers": 4
                },
                "monitoring": {
                    "metrics_interval": 120
                },
                "security": {
                    "rate_limiting": {
                        "max_requests_per_minute": 50,
                        "max_requests_per_hour": 500
                    }
                }
            }
        }
        
        return JSONResponse(content={
            "templates": templates,
            "available_environments": list(templates.keys())
        })
    except Exception as e:
        logger.error(f"Error getting configuration templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-config")
async def health_check_config():
    """Configuration system health check"""
    try:
        stats = config_system.get_config_stats()
        config = await config_system.get_config()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Configuration System",
            "version": "1.0.0",
            "features": {
                "config_management": True,
                "config_validation": True,
                "config_backup": True,
                "config_export": True,
                "config_import": True,
                "config_history": True,
                "environment_config": True,
                "config_templates": True
            },
            "config_stats": stats,
            "current_config": {
                "sections": list(config.keys()),
                "environment": config.get("system", {}).get("environment", "unknown"),
                "version": config.get("system", {}).get("version", "unknown")
            }
        })
    except Exception as e:
        logger.error(f"Error in configuration health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













