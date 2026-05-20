"""
Document Workflow - Sistema de Workflow
========================================

Sistema de workflow para procesamiento de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Estado de workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Paso de workflow."""
    step_id: str
    name: str
    processor: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 3
    timeout: Optional[float] = None
    required: bool = True


@dataclass
class WorkflowExecution:
    """Ejecución de workflow."""
    execution_id: str
    workflow_id: str
    document_id: str
    status: WorkflowStatus
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class WorkflowManager:
    """Gestor de workflows."""
    
    def __init__(self, analyzer):
        """Inicializar gestor."""
        self.analyzer = analyzer
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
    
    def register_workflow(
        self,
        workflow_id: str,
        steps: List[WorkflowStep]
    ):
        """Registrar workflow."""
        self.workflows[workflow_id] = steps
        logger.info(f"Workflow registrado: {workflow_id} con {len(steps)} pasos")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        document_id: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Ejecutar workflow.
        
        Args:
            workflow_id: ID del workflow
            document_id: ID del documento
            initial_data: Datos iniciales
        
        Returns:
            WorkflowExecution con resultado
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} no encontrado")
        
        execution_id = f"exec_{workflow_id}_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            document_id=document_id,
            status=WorkflowStatus.RUNNING
        )
        
        self.executions[execution_id] = execution
        
        steps = self.workflows[workflow_id]
        data = initial_data or {}
        
        try:
            for step in steps:
                execution.current_step = step.step_id
                
                try:
                    # Ejecutar paso con timeout si está definido
                    if step.timeout:
                        result = await asyncio.wait_for(
                            self._execute_step(step, data),
                            timeout=step.timeout
                        )
                    else:
                        result = await self._execute_step(step, data)
                    
                    # Guardar resultado
                    data[step.step_id] = result
                    execution.results[step.step_id] = result
                    execution.completed_steps.append(step.step_id)
                    
                    logger.info(f"Paso {step.step_id} completado")
                    
                except asyncio.TimeoutError:
                    error_msg = f"Timeout en paso {step.step_id}"
                    execution.errors.append(error_msg)
                    execution.failed_steps.append(step.step_id)
                    
                    if step.required:
                        execution.status = WorkflowStatus.FAILED
                        break
                    
                    logger.warning(error_msg)
                    
                except Exception as e:
                    error_msg = f"Error en paso {step.step_id}: {str(e)}"
                    execution.errors.append(error_msg)
                    execution.failed_steps.append(step.step_id)
                    
                    if step.required:
                        execution.status = WorkflowStatus.FAILED
                        break
                    
                    logger.error(error_msg)
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now()
                logger.info(f"Workflow {workflow_id} completado")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append(str(e))
            logger.error(f"Error en workflow {workflow_id}: {e}")
        
        return execution
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        data: Dict[str, Any]
    ) -> Any:
        """Ejecutar paso de workflow."""
        processor = step.processor
        
        # Retry logic
        last_error = None
        for attempt in range(step.retry_count):
            try:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(data, **step.config)
                else:
                    return processor(data, **step.config)
            except Exception as e:
                last_error = e
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_error
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Obtener ejecución."""
        return self.executions.get(execution_id)
    
    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None
    ) -> List[WorkflowExecution]:
        """Listar ejecuciones."""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions


# Workflows predefinidos
def create_analysis_workflow(analyzer) -> List[WorkflowStep]:
    """Crear workflow de análisis completo."""
    return [
        WorkflowStep(
            step_id="extract",
            name="Extraer Contenido",
            processor=lambda data, **kwargs: analyzer.analyze_document(document_content=data.get("content")),
            required=True
        ),
        WorkflowStep(
            step_id="quality",
            name="Analizar Calidad",
            processor=lambda data, **kwargs: analyzer.analyze_quality(data["extract"].content),
            required=False
        ),
        WorkflowStep(
            step_id="grammar",
            name="Analizar Gramática",
            processor=lambda data, **kwargs: analyzer.analyze_grammar(data["extract"].content),
            required=False
        ),
        WorkflowStep(
            step_id="recommendations",
            name="Generar Recomendaciones",
            processor=lambda data, **kwargs: analyzer.generate_recommendations(
                data["extract"],
                data.get("quality"),
                data.get("grammar")
            ),
            required=False
        )
    ]


__all__ = [
    "WorkflowManager",
    "WorkflowStep",
    "WorkflowExecution",
    "WorkflowStatus",
    "create_analysis_workflow"
]
















