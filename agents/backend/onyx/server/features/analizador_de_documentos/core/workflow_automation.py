"""
Automatización de Workflows
============================

Sistema para automatizar flujos de trabajo de análisis de documentos.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class WorkflowStepType(Enum):
    """Tipos de pasos de workflow"""
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    COMPARISON = "comparison"
    EXTRACTION = "extraction"
    NOTIFICATION = "notification"
    EXPORT = "export"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """Paso de workflow"""
    step_id: str
    step_type: WorkflowStepType
    config: Dict[str, Any]
    condition: Optional[Callable] = None
    on_success: Optional[str] = None  # ID del siguiente paso
    on_failure: Optional[str] = None


@dataclass
class Workflow:
    """Workflow de análisis"""
    name: str
    description: str
    steps: List[WorkflowStep]
    enabled: bool = True
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()


@dataclass
class WorkflowExecution:
    """Ejecución de workflow"""
    workflow_name: str
    document_id: str
    status: str  # running, completed, failed
    results: Dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    started_at: str = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now().isoformat()


class WorkflowAutomator:
    """
    Automatizador de workflows
    
    Permite crear y ejecutar workflows automatizados
    para análisis de documentos.
    """
    
    def __init__(self, analyzer):
        """
        Inicializar automatizador
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        logger.info("WorkflowAutomator inicializado")
    
    def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]]
    ) -> Workflow:
        """
        Crear workflow
        
        Args:
            name: Nombre del workflow
            description: Descripción
            steps: Lista de pasos
        
        Returns:
            Workflow creado
        """
        workflow_steps = []
        for step_data in steps:
            step = WorkflowStep(
                step_id=step_data["step_id"],
                step_type=WorkflowStepType(step_data["step_type"]),
                config=step_data.get("config", {}),
                on_success=step_data.get("on_success"),
                on_failure=step_data.get("on_failure")
            )
            workflow_steps.append(step)
        
        workflow = Workflow(
            name=name,
            description=description,
            steps=workflow_steps
        )
        
        self.workflows[name] = workflow
        logger.info(f"Workflow creado: {name}")
        
        return workflow
    
    async def execute_workflow(
        self,
        workflow_name: str,
        document_id: str,
        document_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Ejecutar workflow
        
        Args:
            workflow_name: Nombre del workflow
            document_id: ID del documento
            document_content: Contenido del documento
            metadata: Metadata adicional
        
        Returns:
            WorkflowExecution con resultados
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow no encontrado: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        if not workflow.enabled:
            raise ValueError(f"Workflow deshabilitado: {workflow_name}")
        
        execution = WorkflowExecution(
            workflow_name=workflow_name,
            document_id=document_id,
            status="running"
        )
        
        execution_id = f"{workflow_name}_{document_id}_{datetime.now().timestamp()}"
        self.executions[execution_id] = execution
        
        try:
            # Ejecutar pasos
            current_step_id = workflow.steps[0].step_id if workflow.steps else None
            
            while current_step_id:
                step = next((s for s in workflow.steps if s.step_id == current_step_id), None)
                if not step:
                    break
                
                execution.current_step = current_step_id
                
                # Ejecutar paso
                result = await self._execute_step(
                    step,
                    document_content,
                    execution.results,
                    metadata
                )
                
                execution.results[step.step_id] = result
                
                # Determinar siguiente paso
                if result.get("success", False):
                    current_step_id = step.on_success
                else:
                    current_step_id = step.on_failure
                    if not current_step_id:
                        # Si no hay paso de fallo, terminar
                        execution.status = "failed"
                        execution.error = result.get("error", "Unknown error")
                        break
            
            if execution.status == "running":
                execution.status = "completed"
                execution.completed_at = datetime.now().isoformat()
        
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            logger.error(f"Error ejecutando workflow {workflow_name}: {e}")
        
        return execution
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        content: str,
        previous_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ejecutar un paso del workflow"""
        try:
            if step.step_type == WorkflowStepType.ANALYSIS:
                from .document_analyzer import AnalysisTask
                tasks = step.config.get("tasks", ["classification", "summarization"])
                result = await self.analyzer.analyze_document(
                    document_content=content,
                    tasks=[AnalysisTask(t) for t in tasks]
                )
                return {"success": True, "result": result.__dict__ if hasattr(result, "__dict__") else result}
            
            elif step.step_type == WorkflowStepType.VALIDATION:
                from .document_validator import DocumentValidator
                validator = DocumentValidator()
                result = await validator.validate(content)
                return {"success": result.is_valid, "result": result.__dict__}
            
            elif step.step_type == WorkflowStepType.EXTRACTION:
                from .structured_extractor import StructuredExtractor
                extractor = StructuredExtractor(self.analyzer)
                schema = extractor.create_schema(step.config.get("schema", []))
                result = await extractor.extract_structured_data(content, schema)
                return {"success": True, "result": result}
            
            elif step.step_type == WorkflowStepType.NOTIFICATION:
                from ..utils.notifications import get_webhook_manager
                webhook_manager = get_webhook_manager()
                await webhook_manager.notify_analysis_complete(
                    document_id=metadata.get("document_id", "unknown"),
                    analysis_result=previous_results,
                    webhook_url=step.config.get("webhook_url")
                )
                return {"success": True, "message": "Notification sent"}
            
            elif step.step_type == WorkflowStepType.EXPORT:
                from ..utils.exporters import ResultExporter
                output_path = step.config.get("output_path", "./exports")
                format_type = step.config.get("format", "json")
                
                data_to_export = previous_results.get(step.config.get("source_step"), {})
                ResultExporter.export_json(data_to_export, f"{output_path}/export.json")
                return {"success": True, "exported_to": output_path}
            
            else:
                return {"success": False, "error": f"Step type not implemented: {step.step_type}"}
        
        except Exception as e:
            logger.error(f"Error ejecutando paso {step.step_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """Obtener workflow por nombre"""
        return self.workflows.get(name)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """Listar todos los workflows"""
        return [
            {
                "name": w.name,
                "description": w.description,
                "steps_count": len(w.steps),
                "enabled": w.enabled,
                "created_at": w.created_at
            }
            for w in self.workflows.values()
        ]

