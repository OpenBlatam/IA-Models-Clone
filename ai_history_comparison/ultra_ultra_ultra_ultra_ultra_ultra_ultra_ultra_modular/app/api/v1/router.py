"""
Main API router with all endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ...core.config import get_settings
from ...core.exceptions import BaseAppException
from ...models.schemas import (
    BaseResponse, ErrorResponse, HealthCheckResponse,
    PluginInstallRequest, PluginActivationRequest,
    ContentAnalysisRequest, ContentAnalysisResponse,
    SystemStats
)
from ...services.plugin_service import (
    discover_plugins, install_plugin, activate_plugin, deactivate_plugin,
    list_plugins, get_plugin_stats
)
from ...services.analysis_service import analyze_content

router = APIRouter()
settings = get_settings()


@router.exception_handler(BaseAppException)
async def app_exception_handler(request, exc: BaseAppException):
    """Handle custom application exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.message,
            error_code=type(exc).__name__,
            details=exc.details
        ).dict()
    )


@router.get("/health", response_model=HealthCheckResponse, summary="Health check")
async def health_check():
    """Check the health of all systems."""
    return HealthCheckResponse(
        status="healthy",
        systems={
            "plugin_system": True,
            "analysis_system": True,
            "api_system": True
        },
        version=settings.app_version
    )


@router.get("/stats", response_model=SystemStats, summary="System statistics")
async def get_system_stats():
    """Get system statistics."""
    plugin_stats = await get_plugin_stats()
    
    return SystemStats(
        systems={
            "plugins": plugin_stats,
            "analysis": {
                "total_analyses": 0,  # Would be tracked in real implementation
                "average_processing_time": 0.0
            }
        },
        total_requests=0,  # Would be tracked in real implementation
        successful_requests=0,
        failed_requests=0,
        average_response_time=0.0
    )


# Plugin endpoints
@router.get("/plugins", response_model=BaseResponse, summary="List plugins")
async def list_available_plugins():
    """List all available plugins."""
    plugins = await list_plugins()
    return BaseResponse(
        message=f"Found {len(plugins)} plugins",
        data={"plugins": [plugin.dict() for plugin in plugins]}
    )


@router.post("/plugins/install", response_model=BaseResponse, summary="Install plugin")
async def install_plugin_endpoint(request: PluginInstallRequest):
    """Install a plugin."""
    success = await install_plugin(
        request.plugin_name,
        request.auto_install_dependencies
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to install plugin")
    
    return BaseResponse(
        message=f"Plugin {request.plugin_name} installed successfully"
    )


@router.post("/plugins/activate", response_model=BaseResponse, summary="Activate plugin")
async def activate_plugin_endpoint(request: PluginActivationRequest):
    """Activate a plugin."""
    success = await activate_plugin(request.plugin_name)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to activate plugin")
    
    return BaseResponse(
        message=f"Plugin {request.plugin_name} activated successfully"
    )


@router.post("/plugins/{plugin_name}/deactivate", response_model=BaseResponse, summary="Deactivate plugin")
async def deactivate_plugin_endpoint(plugin_name: str):
    """Deactivate a plugin."""
    success = await deactivate_plugin(plugin_name)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to deactivate plugin")
    
    return BaseResponse(
        message=f"Plugin {plugin_name} deactivated successfully"
    )


# Analysis endpoints
@router.post("/analyze", response_model=ContentAnalysisResponse, summary="Analyze content")
async def analyze_content_endpoint(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze content using the ultra modular system."""
    try:
        # Perform analysis
        result = await analyze_content(request)
        
        # Execute plugin hooks in background
        background_tasks.add_task(
            _execute_plugin_hooks,
            "post_analysis",
            result.dict()
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


async def _execute_plugin_hooks(hook_name: str, data: dict):
    """Execute plugin hooks in background."""
    try:
        from ...services.plugin_service import execute_plugin_hook
        await execute_plugin_hook(hook_name, data)
    except Exception as e:
        # Log error but don't fail the main request
        print(f"Error executing plugin hook {hook_name}: {e}")


# Utility endpoints
@router.get("/", response_model=BaseResponse, summary="API information")
async def api_info():
    """Get API information."""
    return BaseResponse(
        message="Ultra Modular AI History Comparison API",
        data={
            "version": settings.app_version,
            "environment": settings.environment,
            "endpoints": {
                "health": "/health",
                "stats": "/stats",
                "plugins": "/plugins",
                "analyze": "/analyze"
            }
        }
    )




