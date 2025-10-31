"""
Edge Computing Routes
Real, working edge computing endpoints for AI document processing
"""

import logging
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import asyncio
from edge_computing_system import edge_computing_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/edge", tags=["Edge Computing"])

@router.post("/deploy-task")
async def deploy_task_to_edge(
    task_id: str = Form(...),
    task_data: dict = Form(...),
    target_node: Optional[str] = Form(None),
    priority: str = Form("normal")
):
    """Deploy task to edge node"""
    try:
        result = await edge_computing_system.deploy_task_to_edge(task_id, task_data, target_node, priority)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error deploying task to edge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-task")
async def execute_edge_task(
    task_id: str = Form(...)
):
    """Execute edge task"""
    try:
        result = await edge_computing_system.execute_edge_task(task_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error executing edge task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-network")
async def optimize_edge_network():
    """Optimize edge network performance"""
    try:
        result = await edge_computing_system.optimize_edge_network()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error optimizing edge network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitor-resources")
async def monitor_edge_resources():
    """Monitor edge computing resources"""
    try:
        result = await edge_computing_system.monitor_edge_resources()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error monitoring edge resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/edge-nodes")
async def get_edge_nodes():
    """Get all edge nodes"""
    try:
        result = edge_computing_system.get_edge_nodes()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting edge nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/edge-tasks")
async def get_edge_tasks():
    """Get all edge tasks"""
    try:
        result = edge_computing_system.get_edge_tasks()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting edge tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/edge-computing-stats")
async def get_edge_computing_stats():
    """Get edge computing statistics"""
    try:
        result = edge_computing_system.get_edge_computing_stats()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error getting edge computing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/edge-dashboard")
async def get_edge_dashboard():
    """Get comprehensive edge computing dashboard"""
    try:
        # Get all edge computing data
        nodes = edge_computing_system.get_edge_nodes()
        tasks = edge_computing_system.get_edge_tasks()
        stats = edge_computing_system.get_edge_computing_stats()
        resources = await edge_computing_system.monitor_edge_resources()
        
        # Calculate additional metrics
        total_nodes = nodes["node_count"]
        active_nodes = nodes["active_nodes"]
        total_tasks = tasks["task_count"]
        running_tasks = tasks["running_tasks"]
        completed_tasks = tasks["completed_tasks"]
        failed_tasks = tasks["failed_tasks"]
        
        # Calculate success rate
        success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        dashboard_data = {
            "timestamp": stats["uptime_seconds"],
            "overview": {
                "total_nodes": total_nodes,
                "active_nodes": active_nodes,
                "inactive_nodes": total_nodes - active_nodes,
                "total_tasks": total_tasks,
                "running_tasks": running_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": round(success_rate, 2),
                "uptime_hours": stats["uptime_hours"]
            },
            "edge_metrics": {
                "total_edge_nodes": stats["stats"]["total_edge_nodes"],
                "active_edge_nodes": stats["stats"]["active_edge_nodes"],
                "inactive_edge_nodes": stats["stats"]["inactive_edge_nodes"],
                "total_edge_tasks": stats["stats"]["total_edge_tasks"],
                "completed_edge_tasks": stats["stats"]["completed_edge_tasks"],
                "failed_edge_tasks": stats["stats"]["failed_edge_tasks"]
            },
            "edge_nodes": nodes["edge_nodes"],
            "edge_tasks": tasks["edge_tasks"],
            "resource_metrics": resources
        }
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error getting edge dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/edge-performance")
async def get_edge_performance():
    """Get edge computing performance analysis"""
    try:
        stats = edge_computing_system.get_edge_computing_stats()
        nodes = edge_computing_system.get_edge_nodes()
        tasks = edge_computing_system.get_edge_tasks()
        resources = await edge_computing_system.monitor_edge_resources()
        
        # Calculate performance metrics
        total_tasks = stats["stats"]["total_edge_tasks"]
        completed_tasks = stats["stats"]["completed_edge_tasks"]
        failed_tasks = stats["stats"]["failed_edge_tasks"]
        active_nodes = stats["stats"]["active_edge_nodes"]
        
        # Calculate metrics
        success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        failure_rate = (failed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        task_throughput = (completed_tasks / stats["uptime_hours"]) if stats["uptime_hours"] > 0 else 0
        
        # Get resource utilization
        overall_metrics = resources.get("overall_metrics", {})
        avg_cpu = overall_metrics.get("average_cpu_usage", 0)
        avg_memory = overall_metrics.get("average_memory_usage", 0)
        avg_disk = overall_metrics.get("average_disk_usage", 0)
        
        performance_data = {
            "timestamp": stats["uptime_seconds"],
            "performance_metrics": {
                "success_rate": round(success_rate, 2),
                "failure_rate": round(failure_rate, 2),
                "task_throughput": round(task_throughput, 2),
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "active_nodes": active_nodes
            },
            "resource_utilization": {
                "average_cpu_usage": avg_cpu,
                "average_memory_usage": avg_memory,
                "average_disk_usage": avg_disk,
                "resource_efficiency": round((100 - (avg_cpu + avg_memory + avg_disk) / 3), 2)
            },
            "node_performance": {
                "total_nodes": nodes["node_count"],
                "active_nodes": nodes["active_nodes"],
                "node_utilization": round((nodes["active_nodes"] / nodes["node_count"] * 100), 2) if nodes["node_count"] > 0 else 0
            }
        }
        
        return JSONResponse(content=performance_data)
    except Exception as e:
        logger.error(f"Error getting edge performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/edge-optimization")
async def get_edge_optimization():
    """Get edge computing optimization recommendations"""
    try:
        optimization = await edge_computing_system.optimize_edge_network()
        nodes = edge_computing_system.get_edge_nodes()
        tasks = edge_computing_system.get_edge_tasks()
        
        # Generate additional optimization recommendations
        recommendations = []
        
        # Check for load balancing opportunities
        if tasks["running_tasks"] > 0:
            recommendations.append({
                "type": "load_balancing",
                "priority": "medium",
                "description": "Consider distributing tasks across multiple edge nodes",
                "impact": "Improved performance and reliability"
            })
        
        # Check for resource optimization
        if nodes["active_nodes"] < nodes["node_count"]:
            recommendations.append({
                "type": "resource_utilization",
                "priority": "low",
                "description": f"Only {nodes['active_nodes']}/{nodes['node_count']} nodes are active",
                "impact": "Consider activating more nodes for better performance"
            })
        
        # Check for task optimization
        if tasks["failed_tasks"] > 0:
            recommendations.append({
                "type": "task_optimization",
                "priority": "high",
                "description": f"{tasks['failed_tasks']} tasks have failed",
                "impact": "Review task configurations and node capabilities"
            })
        
        optimization_data = {
            "timestamp": optimization.get("timestamp", ""),
            "optimizations": optimization.get("optimizations", []),
            "performance_improvements": optimization.get("performance_improvements", {}),
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
        
        return JSONResponse(content=optimization_data)
    except Exception as e:
        logger.error(f"Error getting edge optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-edge")
async def health_check_edge():
    """Edge computing system health check"""
    try:
        stats = edge_computing_system.get_edge_computing_stats()
        nodes = edge_computing_system.get_edge_nodes()
        tasks = edge_computing_system.get_edge_tasks()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Edge Computing System",
            "version": "1.0.0",
            "features": {
                "edge_node_management": True,
                "task_deployment": True,
                "task_execution": True,
                "resource_monitoring": True,
                "network_optimization": True,
                "load_balancing": True,
                "real_time_processing": True,
                "distributed_computing": True
            },
            "edge_computing_stats": stats["stats"],
            "system_status": {
                "total_edge_nodes": stats["stats"]["total_edge_nodes"],
                "active_edge_nodes": stats["stats"]["active_edge_nodes"],
                "total_edge_tasks": stats["stats"]["total_edge_tasks"],
                "completed_edge_tasks": stats["stats"]["completed_edge_tasks"],
                "failed_edge_tasks": stats["stats"]["failed_edge_tasks"],
                "uptime_hours": stats["uptime_hours"]
            },
            "available_edge_nodes": list(nodes["edge_nodes"].keys()),
            "node_capabilities": {
                "local_edge": ["document_processing", "ai_inference", "data_analysis", "real_time_processing"],
                "cloud_edge": ["heavy_computation", "machine_learning", "data_storage", "batch_processing"],
                "mobile_edge": ["lightweight_processing", "real_time_inference", "data_synchronization", "offline_processing"]
            }
        })
    except Exception as e:
        logger.error(f"Error in edge health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))













