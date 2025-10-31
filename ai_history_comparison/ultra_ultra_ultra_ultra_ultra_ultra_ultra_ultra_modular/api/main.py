"""
Main API - API Principal
=======================

API principal del sistema ultra modular con integración de todos los componentes.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..core.plugin_system.plugin_manager import PluginManager
from ..core.extension_system.extension_manager import ExtensionManager
from ..core.middleware_system.middleware_pipeline import MiddlewarePipeline, MiddlewareType
from ..core.component_system.component_registry import ComponentManager, ComponentScope
from ..core.event_system.event_bus import EventBus, BaseEvent, EventType
from ..core.workflow_system.workflow_engine import WorkflowEngine, WorkflowDefinition, WorkflowStep, StepType

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instancias globales de los sistemas
plugin_manager: Optional[PluginManager] = None
extension_manager: Optional[ExtensionManager] = None
middleware_pipeline: Optional[MiddlewarePipeline] = None
component_manager: Optional[ComponentManager] = None
event_bus: Optional[EventBus] = None
workflow_engine: Optional[WorkflowEngine] = None

# Crear aplicación FastAPI
app = FastAPI(
    title="Ultra Ultra Ultra Ultra Ultra Ultra Ultra Ultra Modular API",
    description="API ultra modular con máxima flexibilidad y composición dinámica",
    version="8.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependencias
async def get_plugin_manager() -> PluginManager:
    """Obtener gestor de plugins."""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin manager not initialized")
    return plugin_manager


async def get_extension_manager() -> ExtensionManager:
    """Obtener gestor de extensiones."""
    if not extension_manager:
        raise HTTPException(status_code=503, detail="Extension manager not initialized")
    return extension_manager


async def get_middleware_pipeline() -> MiddlewarePipeline:
    """Obtener pipeline de middleware."""
    if not middleware_pipeline:
        raise HTTPException(status_code=503, detail="Middleware pipeline not initialized")
    return middleware_pipeline


async def get_component_manager() -> ComponentManager:
    """Obtener gestor de componentes."""
    if not component_manager:
        raise HTTPException(status_code=503, detail="Component manager not initialized")
    return component_manager


async def get_event_bus() -> EventBus:
    """Obtener bus de eventos."""
    if not event_bus:
        raise HTTPException(status_code=503, detail="Event bus not initialized")
    return event_bus


async def get_workflow_engine() -> WorkflowEngine:
    """Obtener motor de workflows."""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not initialized")
    return workflow_engine


# Eventos de aplicación
@app.on_event("startup")
async def startup_event():
    """Inicializar sistemas al arrancar."""
    global plugin_manager, extension_manager, middleware_pipeline
    global component_manager, event_bus, workflow_engine
    
    try:
        logger.info("Initializing ultra modular systems...")
        
        # Inicializar gestor de plugins
        plugin_manager = PluginManager("plugins/")
        await plugin_manager.initialize()
        
        # Inicializar gestor de extensiones
        extension_manager = ExtensionManager()
        await extension_manager.initialize()
        
        # Inicializar pipeline de middleware
        middleware_pipeline = MiddlewarePipeline("main")
        await middleware_pipeline.initialize()
        
        # Inicializar gestor de componentes
        component_manager = ComponentManager()
        await component_manager.initialize()
        
        # Inicializar bus de eventos
        event_bus = EventBus("main")
        await event_bus.initialize()
        
        # Inicializar motor de workflows
        workflow_engine = WorkflowEngine("main")
        await workflow_engine.initialize()
        
        logger.info("All ultra modular systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing systems: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cerrar sistemas al apagar."""
    global plugin_manager, extension_manager, middleware_pipeline
    global component_manager, event_bus, workflow_engine
    
    try:
        logger.info("Shutting down ultra modular systems...")
        
        # Cerrar en orden inverso
        if workflow_engine:
            await workflow_engine.shutdown()
        
        if event_bus:
            await event_bus.shutdown()
        
        if component_manager:
            await component_manager.shutdown()
        
        if middleware_pipeline:
            await middleware_pipeline.shutdown()
        
        if extension_manager:
            await extension_manager.shutdown()
        
        if plugin_manager:
            await plugin_manager.shutdown()
        
        logger.info("All ultra modular systems shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down systems: {e}")


# Endpoints de salud
@app.get("/health", summary="Health check")
async def health_check():
    """Verificar salud de todos los sistemas."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "systems": {}
        }
        
        # Verificar salud de cada sistema
        if plugin_manager:
            health_status["systems"]["plugin_manager"] = await plugin_manager.health_check()
        
        if extension_manager:
            health_status["systems"]["extension_manager"] = await extension_manager.health_check()
        
        if middleware_pipeline:
            health_status["systems"]["middleware_pipeline"] = await middleware_pipeline.health_check()
        
        if component_manager:
            health_status["systems"]["component_manager"] = await component_manager.health_check()
        
        if event_bus:
            health_status["systems"]["event_bus"] = await event_bus.health_check()
        
        if workflow_engine:
            health_status["systems"]["workflow_engine"] = await workflow_engine.health_check()
        
        # Verificar si todos los sistemas están saludables
        all_healthy = all(health_status["systems"].values())
        if not all_healthy:
            health_status["status"] = "unhealthy"
            return JSONResponse(status_code=503, content=health_status)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# Endpoints de plugins
@app.get("/plugins", summary="List available plugins")
async def list_plugins(plugin_mgr: PluginManager = Depends(get_plugin_manager)):
    """Listar plugins disponibles."""
    try:
        plugins = await plugin_mgr.list_available_plugins()
        return {
            "plugins": [
                {
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "author": plugin.author,
                    "dependencies": plugin.dependencies,
                    "status": plugin.status,
                    "installed_at": plugin.installed_at.isoformat()
                }
                for plugin in plugins
            ]
        }
    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plugins/{plugin_name}/install", summary="Install plugin")
async def install_plugin(
    plugin_name: str,
    plugin_mgr: PluginManager = Depends(get_plugin_manager)
):
    """Instalar plugin."""
    try:
        success = await plugin_mgr.install_plugin(plugin_name)
        if success:
            return {"message": f"Plugin {plugin_name} installed successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to install plugin {plugin_name}")
    except Exception as e:
        logger.error(f"Error installing plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plugins/{plugin_name}/activate", summary="Activate plugin")
async def activate_plugin(
    plugin_name: str,
    plugin_mgr: PluginManager = Depends(get_plugin_manager)
):
    """Activar plugin."""
    try:
        success = await plugin_mgr.activate_plugin(plugin_name)
        if success:
            return {"message": f"Plugin {plugin_name} activated successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to activate plugin {plugin_name}")
    except Exception as e:
        logger.error(f"Error activating plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plugins/{plugin_name}/deactivate", summary="Deactivate plugin")
async def deactivate_plugin(
    plugin_name: str,
    plugin_mgr: PluginManager = Depends(get_plugin_manager)
):
    """Desactivar plugin."""
    try:
        success = await plugin_mgr.deactivate_plugin(plugin_name)
        if success:
            return {"message": f"Plugin {plugin_name} deactivated successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to deactivate plugin {plugin_name}")
    except Exception as e:
        logger.error(f"Error deactivating plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints de extensiones
@app.get("/extensions/points", summary="List extension points")
async def list_extension_points(ext_mgr: ExtensionManager = Depends(get_extension_manager)):
    """Listar puntos de extensión."""
    try:
        points = await ext_mgr.list_extension_points()
        return {"extension_points": points}
    except Exception as e:
        logger.error(f"Error listing extension points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extensions/points/{point_name}/execute", summary="Execute extensions at point")
async def execute_extensions(
    point_name: str,
    data: Any,
    ext_mgr: ExtensionManager = Depends(get_extension_manager)
):
    """Ejecutar extensiones en un punto."""
    try:
        from ..core.extension_system.extension_manager import ExtensionContext
        
        context = ExtensionContext(data)
        result = await ext_mgr.execute_extensions(point_name, context)
        
        return {
            "point_name": point_name,
            "original_data": data,
            "processed_data": result.data,
            "modified": result.modified,
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Error executing extensions at point {point_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints de middleware
@app.get("/middleware/stats", summary="Get middleware pipeline stats")
async def get_middleware_stats(middleware: MiddlewarePipeline = Depends(get_middleware_pipeline)):
    """Obtener estadísticas del pipeline de middleware."""
    try:
        stats = await middleware.get_pipeline_stats()
        return {"middleware_stats": stats}
    except Exception as e:
        logger.error(f"Error getting middleware stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/middleware/execute", summary="Execute middleware pipeline")
async def execute_middleware_pipeline(
    request_data: Any,
    middleware: MiddlewarePipeline = Depends(get_middleware_pipeline)
):
    """Ejecutar pipeline de middleware."""
    try:
        from ..core.middleware_system.middleware_pipeline import MiddlewareContext
        
        context = MiddlewareContext(request=request_data)
        result = await middleware.execute_pipeline(context)
        
        return {
            "original_request": request_data,
            "processed_request": result.request,
            "response": result.response,
            "modified": result.modified,
            "errors": [str(e) for e in result.errors],
            "warnings": result.warnings,
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Error executing middleware pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints de componentes
@app.get("/components", summary="List registered components")
async def list_components(comp_mgr: ComponentManager = Depends(get_component_manager)):
    """Listar componentes registrados."""
    try:
        components = await comp_mgr.list_components()
        stats = await comp_mgr.get_component_stats()
        
        return {
            "components": components,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error listing components: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/components/{component_name}", summary="Get component")
async def get_component(
    component_name: str,
    comp_mgr: ComponentManager = Depends(get_component_manager)
):
    """Obtener componente."""
    try:
        component = await comp_mgr.get_component(component_name)
        if component:
            return {
                "name": component_name,
                "type": type(component).__name__,
                "available": True
            }
        else:
            raise HTTPException(status_code=404, detail=f"Component {component_name} not found")
    except Exception as e:
        logger.error(f"Error getting component {component_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints de eventos
@app.post("/events/publish", summary="Publish event")
async def publish_event(
    event_type: str,
    event_data: Any,
    event_bus: EventBus = Depends(get_event_bus)
):
    """Publicar evento."""
    try:
        success = await event_bus.publish(event_type, event_data)
        if success:
            return {"message": f"Event {event_type} published successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to publish event {event_type}")
    except Exception as e:
        logger.error(f"Error publishing event {event_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events/stats", summary="Get event bus stats")
async def get_event_stats(event_bus: EventBus = Depends(get_event_bus)):
    """Obtener estadísticas del bus de eventos."""
    try:
        stats = await event_bus.get_event_stats()
        return {"event_stats": stats}
    except Exception as e:
        logger.error(f"Error getting event stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints de workflows
@app.get("/workflows", summary="List workflow definitions")
async def list_workflows(workflow_eng: WorkflowEngine = Depends(get_workflow_engine)):
    """Listar definiciones de workflow."""
    try:
        workflows = await workflow_eng.list_workflow_definitions()
        return {"workflows": workflows}
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/{workflow_id}/start", summary="Start workflow")
async def start_workflow(
    workflow_id: str,
    initial_data: Any = None,
    workflow_eng: WorkflowEngine = Depends(get_workflow_engine)
):
    """Iniciar workflow."""
    try:
        instance_id = await workflow_eng.start_workflow(workflow_id, initial_data)
        return {
            "workflow_id": workflow_id,
            "instance_id": instance_id,
            "message": f"Workflow {workflow_id} started successfully"
        }
    except Exception as e:
        logger.error(f"Error starting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows/instances/{instance_id}/status", summary="Get workflow status")
async def get_workflow_status(
    instance_id: str,
    workflow_eng: WorkflowEngine = Depends(get_workflow_engine)
):
    """Obtener estado de workflow."""
    try:
        instance = await workflow_eng.get_workflow_instance(instance_id)
        if instance:
            return {
                "instance_id": instance_id,
                "status": instance.status.value,
                "stats": instance.get_stats()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Workflow instance {instance_id} not found")
    except Exception as e:
        logger.error(f"Error getting workflow status {instance_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints de estadísticas generales
@app.get("/stats", summary="Get system statistics")
async def get_system_stats():
    """Obtener estadísticas generales del sistema."""
    try:
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "systems": {}
        }
        
        # Estadísticas de plugins
        if plugin_manager:
            plugin_stats = await plugin_manager.get_execution_stats()
            stats["systems"]["plugins"] = plugin_stats
        
        # Estadísticas de extensiones
        if extension_manager:
            extension_stats = await extension_manager.get_execution_stats()
            stats["systems"]["extensions"] = extension_stats
        
        # Estadísticas de middleware
        if middleware_pipeline:
            middleware_stats = await middleware_pipeline.get_pipeline_stats()
            stats["systems"]["middleware"] = middleware_stats
        
        # Estadísticas de componentes
        if component_manager:
            component_stats = await component_manager.get_component_stats()
            stats["systems"]["components"] = component_stats
        
        # Estadísticas de eventos
        if event_bus:
            event_stats = await event_bus.get_event_stats()
            stats["systems"]["events"] = event_stats
        
        # Estadísticas de workflows
        if workflow_engine:
            workflow_stats = await workflow_engine.get_workflow_stats()
            stats["systems"]["workflows"] = workflow_stats
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint de ejemplo de uso integrado
@app.post("/analyze", summary="Analyze content with all systems")
async def analyze_content(
    content: str,
    model_version: str = "1.0.0",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analizar contenido usando todos los sistemas ultra modulares."""
    try:
        # Crear evento de análisis
        if event_bus:
            await event_bus.publish("content_analysis_started", {
                "content_length": len(content),
                "model_version": model_version
            })
        
        # Ejecutar middleware pipeline
        if middleware_pipeline:
            from ..core.middleware_system.middleware_pipeline import MiddlewareContext
            context = MiddlewareContext(request={"content": content, "model_version": model_version})
            middleware_result = await middleware_pipeline.execute_pipeline(context)
            processed_data = middleware_result.request
        else:
            processed_data = {"content": content, "model_version": model_version}
        
        # Ejecutar extensiones pre-análisis
        if extension_manager:
            from ..core.extension_system.extension_manager import ExtensionContext
            ext_context = ExtensionContext(processed_data)
            pre_result = await extension_manager.execute_extensions("pre_analysis", ext_context)
            processed_data = pre_result.data
        
        # Ejecutar plugins de análisis
        analysis_results = {}
        if plugin_manager:
            # Ejecutar hooks de plugins
            plugin_results = await plugin_manager.execute_plugin_hook("pre_analysis", processed_data)
            analysis_results["plugin_results"] = plugin_results
        
        # Simular análisis (aquí se integrarían los componentes reales)
        analysis_result = {
            "content": processed_data.get("content", content),
            "model_version": processed_data.get("model_version", model_version),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "word_count": len(content.split()),
            "character_count": len(content),
            "analysis_results": analysis_results
        }
        
        # Ejecutar extensiones post-análisis
        if extension_manager:
            from ..core.extension_system.extension_manager import ExtensionContext
            ext_context = ExtensionContext(analysis_result)
            post_result = await extension_manager.execute_extensions("post_analysis", ext_context)
            analysis_result = post_result.data
        
        # Crear evento de finalización
        if event_bus:
            background_tasks.add_task(
                event_bus.publish,
                "content_analysis_completed",
                analysis_result
            )
        
        return {
            "success": True,
            "analysis_result": analysis_result,
            "systems_used": {
                "middleware_pipeline": middleware_pipeline is not None,
                "extension_manager": extension_manager is not None,
                "plugin_manager": plugin_manager is not None,
                "event_bus": event_bus is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error in integrated analysis: {e}")
        
        # Crear evento de error
        if event_bus:
            background_tasks.add_task(
                event_bus.publish,
                "content_analysis_failed",
                {"error": str(e), "content_length": len(content)}
            )
        
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )




