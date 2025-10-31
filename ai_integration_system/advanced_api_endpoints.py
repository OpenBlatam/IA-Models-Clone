"""
AI Integration System - Advanced API Endpoints
Enhanced API endpoints with workflow, analytics, and advanced features
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import uuid
import json
import csv
import io
from datetime import datetime, timedelta

from .integration_engine import (
    AIIntegrationEngine, 
    IntegrationRequest, 
    IntegrationResult, 
    ContentType, 
    IntegrationStatus,
    integration_engine
)
from .workflow_engine import (
    WorkflowDefinition, WorkflowStep, WorkflowExecution, 
    StepType, workflow_engine, initialize_workflow_engine
)
from .analytics import (
    analytics_engine, PerformanceMetrics, PlatformAnalytics, 
    ContentAnalytics, MetricType, TimeRange
)
from .middleware import get_current_user, require_permissions
from .monitoring import get_health_status

# Create router
router = APIRouter(prefix="/ai-integration/v2", tags=["AI Integration Advanced"])

# Pydantic models for advanced API
class WorkflowDefinitionModel(BaseModel):
    """Workflow definition model for API"""
    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    variables: Optional[Dict[str, Any]] = Field(default={}, description="Workflow variables")
    triggers: Optional[List[Dict[str, Any]]] = Field(default=[], description="Workflow triggers")

class WorkflowExecutionModel(BaseModel):
    """Workflow execution model for API"""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow")
    variables: Optional[Dict[str, Any]] = Field(default={}, description="Additional variables")

class AnalyticsQueryModel(BaseModel):
    """Analytics query model"""
    time_range: str = Field(default="day", description="Time range for analytics")
    metric_type: Optional[str] = Field(default=None, description="Specific metric type")
    platform: Optional[str] = Field(default=None, description="Filter by platform")
    content_type: Optional[str] = Field(default=None, description="Filter by content type")

class AdvancedIntegrationRequestModel(BaseModel):
    """Advanced integration request model"""
    content_id: str = Field(..., description="Unique identifier for the content")
    content_type: str = Field(..., description="Type of content")
    content_data: Dict[str, Any] = Field(..., description="Content data")
    target_platforms: List[str] = Field(..., description="Target platforms")
    workflow_id: Optional[str] = Field(default=None, description="Workflow to execute")
    priority: int = Field(default=1, ge=1, le=10, description="Priority level")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    schedule_at: Optional[datetime] = Field(default=None, description="Schedule execution time")

# Workflow Management Endpoints

@router.post("/workflows", response_model=Dict[str, str])
async def create_workflow(
    workflow: WorkflowDefinitionModel,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new workflow definition"""
    try:
        # Convert API model to workflow definition
        steps = []
        for step_data in workflow.steps:
            step = WorkflowStep(
                id=step_data["id"],
                name=step_data["name"],
                step_type=StepType(step_data["step_type"]),
                config=step_data["config"],
                dependencies=step_data.get("dependencies", []),
                max_retries=step_data.get("max_retries", 3),
                timeout=step_data.get("timeout", 300)
            )
            steps.append(step)
        
        workflow_def = WorkflowDefinition(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            steps=steps,
            variables=workflow.variables,
            triggers=workflow.triggers
        )
        
        # Register workflow
        workflow_engine.register_workflow(workflow_def)
        
        return {
            "message": "Workflow created successfully",
            "workflow_id": workflow.id,
            "status": "created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows", response_model=List[Dict[str, Any]])
async def list_workflows(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all available workflows"""
    try:
        workflows = workflow_engine.list_workflows()
        return workflows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get workflow definition by ID"""
    try:
        if workflow_id not in workflow_engine.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = workflow_engine.workflows[workflow_id]
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "step_type": step.step_type.value,
                    "config": step.config,
                    "dependencies": step.dependencies,
                    "max_retries": step.max_retries,
                    "timeout": step.timeout
                }
                for step in workflow.steps
            ],
            "variables": workflow.variables,
            "triggers": workflow.triggers,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/execute", response_model=Dict[str, str])
async def execute_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    execution: WorkflowExecutionModel = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute a workflow"""
    try:
        if workflow_id not in workflow_engine.workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(
            workflow_id, 
            execution.input_data
        )
        
        return {
            "message": "Workflow execution started",
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/executions/{execution_id}", response_model=Dict[str, Any])
async def get_workflow_execution_status(
    execution_id: str = Path(..., description="Execution ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get workflow execution status"""
    try:
        status = workflow_engine.get_workflow_status(execution_id)
        if not status:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/executions", response_model=List[Dict[str, Any]])
async def list_workflow_executions(
    workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List workflow executions"""
    try:
        executions = workflow_engine.list_executions(workflow_id)
        return executions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Integration Endpoints

@router.post("/integrate/advanced", response_model=Dict[str, str])
async def create_advanced_integration(
    request: AdvancedIntegrationRequestModel,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create an advanced integration request with workflow support"""
    try:
        # Validate content type
        try:
            content_type = ContentType(request.content_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid content type. Valid types: {[ct.value for ct in ContentType]}"
            )
        
        # Create integration request
        integration_request = IntegrationRequest(
            content_id=request.content_id,
            content_type=content_type,
            content_data=request.content_data,
            target_platforms=request.target_platforms,
            priority=request.priority,
            max_retries=request.max_retries,
            metadata={
                **request.metadata,
                "user_id": current_user.get("user_id"),
                "workflow_id": request.workflow_id,
                "api_request": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Add to queue
        await integration_engine.add_integration_request(integration_request)
        
        # Execute workflow if specified
        if request.workflow_id:
            workflow_input = {
                "content_id": request.content_id,
                "content_type": request.content_type,
                "content_data": request.content_data,
                "target_platforms": request.target_platforms,
                "user_id": current_user.get("user_id")
            }
            
            execution_id = await workflow_engine.execute_workflow(
                request.workflow_id, 
                workflow_input
            )
            
            return {
                "message": "Advanced integration request created with workflow",
                "content_id": request.content_id,
                "workflow_execution_id": execution_id,
                "status": "queued"
            }
        else:
            # Process integration directly
            background_tasks.add_task(integration_engine.process_single_request, integration_request)
            
            return {
                "message": "Advanced integration request created",
                "content_id": request.content_id,
                "status": "queued"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints

@router.get("/analytics/performance", response_model=Dict[str, Any])
async def get_performance_analytics(
    time_range: str = Query("day", description="Time range for analytics"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get performance analytics"""
    try:
        time_range_enum = TimeRange(time_range)
        metrics = await analytics_engine.get_performance_metrics(time_range_enum)
        
        return {
            "time_range": time_range,
            "metrics": {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": round(metrics.success_rate, 2),
                "error_rate": round(metrics.error_rate, 2),
                "average_response_time": round(metrics.average_response_time, 2),
                "median_response_time": round(metrics.median_response_time, 2),
                "p95_response_time": round(metrics.p95_response_time, 2),
                "p99_response_time": round(metrics.p99_response_time, 2),
                "throughput_per_hour": round(metrics.throughput_per_hour, 2)
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/platforms", response_model=List[Dict[str, Any]])
async def get_platform_analytics(
    time_range: str = Query("day", description="Time range for analytics"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get platform-specific analytics"""
    try:
        time_range_enum = TimeRange(time_range)
        analytics = await analytics_engine.get_platform_analytics(time_range_enum)
        
        return [
            {
                "platform": pa.platform,
                "total_integrations": pa.total_integrations,
                "successful_integrations": pa.successful_integrations,
                "failed_integrations": pa.failed_integrations,
                "success_rate": round(pa.success_rate, 2),
                "average_response_time": round(pa.average_response_time, 2),
                "health_status": pa.health_status,
                "last_health_check": pa.last_health_check.isoformat() if pa.last_health_check else None,
                "error_types": pa.error_types
            }
            for pa in analytics
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/content", response_model=List[Dict[str, Any]])
async def get_content_analytics(
    time_range: str = Query("day", description="Time range for analytics"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get content-specific analytics"""
    try:
        time_range_enum = TimeRange(time_range)
        analytics = await analytics_engine.get_content_analytics(time_range_enum)
        
        return [
            {
                "content_type": ca.content_type,
                "total_created": ca.total_created,
                "platforms_distributed": ca.platforms_distributed,
                "average_processing_time": round(ca.average_processing_time, 2),
                "success_rate": round(ca.success_rate, 2),
                "popular_tags": [{"tag": tag, "count": count} for tag, count in ca.popular_tags]
            }
            for ca in analytics
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/trends/{metric_type}", response_model=Dict[str, Any])
async def get_trend_analytics(
    metric_type: str = Path(..., description="Metric type for trend analysis"),
    time_range: str = Query("day", description="Time range for analytics"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get trend analysis for specific metrics"""
    try:
        metric_type_enum = MetricType(metric_type)
        time_range_enum = TimeRange(time_range)
        
        trend_data = await analytics_engine.get_trend_analysis(metric_type_enum, time_range_enum)
        
        return {
            "metric_type": metric_type,
            "time_range": time_range,
            "data_points": trend_data.data_points,
            "summary": trend_data.summary,
            "generated_at": trend_data.generated_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/report", response_model=Dict[str, Any])
async def get_comprehensive_report(
    time_range: str = Query("day", description="Time range for report"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get comprehensive analytics report"""
    try:
        time_range_enum = TimeRange(time_range)
        report = await analytics_engine.generate_comprehensive_report(time_range_enum)
        
        return report
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/export")
async def export_analytics_data(
    time_range: str = Query("day", description="Time range for export"),
    format: str = Query("csv", description="Export format (csv, json)"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Export analytics data"""
    try:
        time_range_enum = TimeRange(time_range)
        
        if format.lower() == "csv":
            # Generate CSV export
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Get analytics data
            performance_metrics = await analytics_engine.get_performance_metrics(time_range_enum)
            platform_analytics = await analytics_engine.get_platform_analytics(time_range_enum)
            
            # Write performance metrics
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Requests", performance_metrics.total_requests])
            writer.writerow(["Success Rate", f"{performance_metrics.success_rate:.2f}%"])
            writer.writerow(["Average Response Time", f"{performance_metrics.average_response_time:.2f}s"])
            writer.writerow([])
            
            # Write platform analytics
            writer.writerow(["Platform", "Total Integrations", "Success Rate", "Health Status"])
            for pa in platform_analytics:
                writer.writerow([
                    pa.platform,
                    pa.total_integrations,
                    f"{pa.success_rate:.2f}%",
                    "Healthy" if pa.health_status else "Unhealthy"
                ])
            
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=analytics_{time_range}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        
        elif format.lower() == "json":
            # Generate JSON export
            report = await analytics_engine.generate_comprehensive_report(time_range_enum)
            
            return JSONResponse(
                content=report,
                headers={"Content-Disposition": f"attachment; filename=analytics_{time_range}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'csv' or 'json'")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Monitoring Endpoints

@router.get("/monitoring/dashboard", response_model=Dict[str, Any])
async def get_monitoring_dashboard(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get comprehensive monitoring dashboard data"""
    try:
        # Get health status
        health_status = get_health_status()
        
        # Get performance metrics
        performance_metrics = await analytics_engine.get_performance_metrics(TimeRange.HOUR)
        
        # Get platform analytics
        platform_analytics = await analytics_engine.get_platform_analytics(TimeRange.HOUR)
        
        # Get recent workflow executions
        recent_executions = workflow_engine.list_executions()
        recent_executions = sorted(recent_executions, key=lambda x: x.get('started_at', ''), reverse=True)[:10]
        
        return {
            "health_status": health_status,
            "performance_metrics": {
                "total_requests": performance_metrics.total_requests,
                "success_rate": round(performance_metrics.success_rate, 2),
                "average_response_time": round(performance_metrics.average_response_time, 2),
                "throughput_per_hour": round(performance_metrics.throughput_per_hour, 2)
            },
            "platform_status": [
                {
                    "platform": pa.platform,
                    "health_status": pa.health_status,
                    "success_rate": round(pa.success_rate, 2),
                    "total_integrations": pa.total_integrations
                }
                for pa in platform_analytics
            ],
            "recent_workflows": recent_executions,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/alerts", response_model=List[Dict[str, Any]])
async def get_active_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get active system alerts"""
    try:
        from .monitoring import monitoring_service
        
        active_alerts = [alert for alert in monitoring_service.alerts if not alert.resolved]
        
        return [
            {
                "id": alert.id,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "platform": alert.platform,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in active_alerts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System Management Endpoints

@router.post("/system/initialize")
async def initialize_system(
    current_user: Dict[str, Any] = Depends(require_permissions(["admin"]))
):
    """Initialize system components"""
    try:
        # Initialize workflow engine
        initialize_workflow_engine()
        
        # Initialize integration engine
        from .integration_engine import initialize_engine
        await initialize_engine()
        
        return {
            "message": "System initialized successfully",
            "components": ["workflow_engine", "integration_engine"],
            "status": "initialized"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status", response_model=Dict[str, Any])
async def get_system_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get comprehensive system status"""
    try:
        # Get health status
        health_status = get_health_status()
        
        # Get system metrics
        from .monitoring import monitoring_service
        system_metrics = monitoring_service.metrics_cache.get("system", {})
        
        # Get database status
        from .database import check_database_health
        db_health = check_database_health()
        
        return {
            "overall_status": health_status["status"],
            "health_checks": health_status["checks"],
            "system_metrics": system_metrics,
            "database_health": db_health,
            "workflow_engine": {
                "registered_workflows": len(workflow_engine.workflows),
                "active_executions": len([e for e in workflow_engine.executions.values() if e.status.value == "running"])
            },
            "integration_engine": {
                "available_platforms": len(integration_engine.get_available_platforms()),
                "queue_length": len(integration_engine.integration_queue),
                "total_results": len(integration_engine.results)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Export the router
__all__ = ["router"]



























