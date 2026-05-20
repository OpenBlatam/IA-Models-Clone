"""
Automation Router - Workflow automation endpoints
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

try:
    from automation_engine import automation_engine
except ImportError:
    logging.warning("automation_engine module not available")
    automation_engine = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/automation", tags=["Automation"])


@router.post("/workflow/create", response_model=Dict[str, Any])
async def create_automation_workflow(workflow_data: Dict[str, Any]) -> JSONResponse:
    """Create automation workflow"""
    logger.info("Automation workflow creation requested")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        name = workflow_data.get("name")
        description = workflow_data.get("description", "")
        steps = workflow_data.get("steps", [])
        
        if not all([name, steps]):
            raise ValueError("Name and steps are required")
        
        workflow = automation_engine.create_workflow(name, description, steps)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Workflow created successfully",
                "workflow_id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "steps_count": len(workflow.steps)
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create automation workflow error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/workflow/{workflow_id}/execute", response_model=Dict[str, Any])
async def execute_automation_workflow(
    workflow_id: str = Path(...),
    execution_data: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Execute automation workflow"""
    logger.info(f"Automation workflow execution requested: {workflow_id}")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        context = execution_data.get("context", {}) if execution_data else {}
        
        result = await automation_engine.execute_workflow(workflow_id, context)
        
        if not result:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Workflow executed successfully",
                "workflow_id": workflow_id,
                "execution_id": result.execution_id,
                "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                "steps_completed": result.steps_completed,
                "total_steps": result.total_steps,
                "result": result.result
            },
            "error": None
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execute automation workflow error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/workflow/{workflow_id}", response_model=Dict[str, Any])
async def get_automation_workflow(workflow_id: str = Path(...)) -> JSONResponse:
    """Get automation workflow"""
    logger.info(f"Automation workflow requested: {workflow_id}")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        workflow = automation_engine.get_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "workflow_id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "type": step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type),
                        "config": step.config
                    }
                    for step in workflow.steps
                ],
                "created_at": workflow.created_at,
                "is_active": workflow.is_active
            },
            "error": None
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get automation workflow error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/workflows", response_model=Dict[str, Any])
async def get_automation_workflows() -> JSONResponse:
    """Get all automation workflows"""
    logger.info("All automation workflows requested")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        workflows = automation_engine.get_all_workflows()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "workflows": [
                    {
                        "workflow_id": workflow.id,
                        "name": workflow.name,
                        "description": workflow.description,
                        "steps_count": len(workflow.steps),
                        "created_at": workflow.created_at,
                        "is_active": workflow.is_active
                    }
                    for workflow in workflows
                ],
                "count": len(workflows)
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get automation workflows error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rule/create", response_model=Dict[str, Any])
async def create_automation_rule(rule_data: Dict[str, Any]) -> JSONResponse:
    """Create automation rule"""
    logger.info("Automation rule creation requested")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        name = rule_data.get("name")
        condition = rule_data.get("condition")
        action = rule_data.get("action")
        
        if not all([name, condition, action]):
            raise ValueError("Name, condition, and action are required")
        
        rule = automation_engine.create_rule(name, condition, action)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Rule created successfully",
                "rule_id": rule.id,
                "name": rule.name,
                "condition": rule.condition,
                "action": rule.action
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create automation rule error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/rules", response_model=Dict[str, Any])
async def get_automation_rules() -> JSONResponse:
    """Get all automation rules"""
    logger.info("All automation rules requested")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        rules = automation_engine.get_all_rules()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "rules": [
                    {
                        "rule_id": rule.id,
                        "name": rule.name,
                        "condition": rule.condition,
                        "action": rule.action,
                        "is_active": rule.is_active
                    }
                    for rule in rules
                ],
                "count": len(rules)
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get automation rules error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/executions", response_model=Dict[str, Any])
async def get_automation_executions() -> JSONResponse:
    """Get automation execution history"""
    logger.info("Automation executions requested")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        executions = automation_engine.get_execution_history()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "executions": [
                    {
                        "execution_id": exec.execution_id,
                        "workflow_id": exec.workflow_id,
                        "status": exec.status.value if hasattr(exec.status, 'value') else str(exec.status),
                        "started_at": exec.started_at,
                        "completed_at": exec.completed_at,
                        "steps_completed": exec.steps_completed
                    }
                    for exec in executions
                ],
                "count": len(executions)
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get automation executions error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=Dict[str, Any])
async def get_automation_stats() -> JSONResponse:
    """Get automation statistics"""
    logger.info("Automation stats requested")
    
    if not automation_engine:
        raise HTTPException(status_code=503, detail="Automation engine not available")
    
    try:
        stats = automation_engine.get_stats()
        return JSONResponse(content={
            "success": True,
            "data": stats,
            "error": None
        })
    except Exception as e:
        logger.error(f"Get automation stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






