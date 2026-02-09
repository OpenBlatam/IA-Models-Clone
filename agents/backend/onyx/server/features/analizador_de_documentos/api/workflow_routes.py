"""
Rutas para Workflows
====================

Endpoints para automatización de workflows.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.workflow_automation import WorkflowAutomator
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/workflows",
    tags=["Workflow Automation"]
)


class CreateWorkflowRequest(BaseModel):
    """Request para crear workflow"""
    name: str = Field(..., description="Nombre del workflow")
    description: str = Field(..., description="Descripción")
    steps: List[Dict[str, Any]] = Field(..., description="Lista de pasos")


class ExecuteWorkflowRequest(BaseModel):
    """Request para ejecutar workflow"""
    workflow_name: str = Field(..., description="Nombre del workflow")
    document_id: str = Field(..., description="ID del documento")
    document_content: str = Field(..., description="Contenido del documento")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata adicional")


# Instancia global del automatizador
_workflow_automator: Optional[WorkflowAutomator] = None


def get_workflow_automator() -> WorkflowAutomator:
    """Dependency para obtener automatizador"""
    global _workflow_automator
    if _workflow_automator is None:
        analyzer = get_analyzer()
        _workflow_automator = WorkflowAutomator(analyzer)
    return _workflow_automator


@router.get("/")
async def list_workflows(
    automator: WorkflowAutomator = Depends(get_workflow_automator)
):
    """Listar todos los workflows"""
    return {"workflows": automator.list_workflows()}


@router.post("/")
async def create_workflow(
    request: CreateWorkflowRequest,
    automator: WorkflowAutomator = Depends(get_workflow_automator)
):
    """Crear nuevo workflow"""
    try:
        workflow = automator.create_workflow(
            request.name,
            request.description,
            request.steps
        )
        
        return {
            "status": "created",
            "workflow": {
                "name": workflow.name,
                "description": workflow.description,
                "steps_count": len(workflow.steps)
            }
        }
    except Exception as e:
        logger.error(f"Error creando workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_workflow(
    request: ExecuteWorkflowRequest,
    automator: WorkflowAutomator = Depends(get_workflow_automator)
):
    """Ejecutar workflow"""
    try:
        execution = await automator.execute_workflow(
            request.workflow_name,
            request.document_id,
            request.document_content,
            request.metadata
        )
        
        return {
            "status": execution.status,
            "workflow_name": execution.workflow_name,
            "document_id": execution.document_id,
            "current_step": execution.current_step,
            "results": execution.results,
            "error": execution.error,
            "started_at": execution.started_at,
            "completed_at": execution.completed_at
        }
    except Exception as e:
        logger.error(f"Error ejecutando workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_name}")
async def get_workflow(
    workflow_name: str,
    automator: WorkflowAutomator = Depends(get_workflow_automator)
):
    """Obtener workflow específico"""
    workflow = automator.get_workflow(workflow_name)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow no encontrado: {workflow_name}")
    
    return {
        "name": workflow.name,
        "description": workflow.description,
        "steps": [
            {
                "step_id": s.step_id,
                "step_type": s.step_type.value,
                "config": s.config,
                "on_success": s.on_success,
                "on_failure": s.on_failure
            }
            for s in workflow.steps
        ],
        "enabled": workflow.enabled,
        "created_at": workflow.created_at
    }

