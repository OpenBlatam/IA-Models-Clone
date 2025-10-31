"""
Workflow Engine
===============

Motor de flujos de trabajo para automatización de procesos empresariales.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Estados del flujo de trabajo"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    """Estados de tareas"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskType(Enum):
    """Tipos de tareas"""
    DOCUMENT_GENERATION = "document_generation"
    CONTENT_REVIEW = "content_review"
    APPROVAL = "approval"
    NOTIFICATION = "notification"
    DATA_PROCESSING = "data_processing"
    INTEGRATION = "integration"
    CUSTOM = "custom"

class TriggerType(Enum):
    """Tipos de disparadores"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_BASED = "event_based"
    CONDITIONAL = "conditional"
    WEBHOOK = "webhook"

@dataclass
class WorkflowTask:
    """Tarea del flujo de trabajo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: TaskType = TaskType.CUSTOM
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1  # 1-10, 10 es más alta
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error_message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # segundos
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowTrigger:
    """Disparador del flujo de trabajo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger_type: TriggerType = TriggerType.MANUAL
    conditions: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Workflow:
    """Flujo de trabajo"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0"
    status: WorkflowStatus = WorkflowStatus.PENDING
    tasks: List[WorkflowTask] = field(default_factory=list)
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskExecutor(ABC):
    """Ejecutor de tareas abstracto"""
    
    @abstractmethod
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Ejecutar tarea"""
        pass
    
    @abstractmethod
    def can_handle(self, task_type: TaskType) -> bool:
        """Verificar si puede manejar el tipo de tarea"""
        pass

class DocumentGenerationExecutor(TaskExecutor):
    """Ejecutor para generación de documentos"""
    
    def __init__(self, bul_engine):
        self.bul_engine = bul_engine
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Ejecutar generación de documento"""
        try:
            from ..core import DocumentRequest, BusinessArea, DocumentType
            
            # Crear solicitud de documento
            request = DocumentRequest(
                query=task.parameters.get("query", ""),
                business_area=BusinessArea(task.parameters.get("business_area", "strategy")),
                document_type=DocumentType(task.parameters.get("document_type", "business_plan")),
                company_name=task.parameters.get("company_name", ""),
                industry=task.parameters.get("industry", ""),
                language=task.parameters.get("language", "es")
            )
            
            # Generar documento
            response = await self.bul_engine.generate_document(request)
            
            return {
                "document_id": response.id,
                "title": response.title,
                "content": response.content,
                "word_count": response.word_count,
                "processing_time": response.processing_time,
                "confidence_score": response.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error executing document generation task: {e}")
            raise
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type == TaskType.DOCUMENT_GENERATION

class ContentReviewExecutor(TaskExecutor):
    """Ejecutor para revisión de contenido"""
    
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Ejecutar revisión de contenido"""
        try:
            from ..ai import AIEnhancementType
            
            content = task.parameters.get("content", "")
            review_type = task.parameters.get("review_type", "comprehensive")
            
            if review_type == "comprehensive":
                analysis = await self.ai_engine.analyze_content_comprehensive(content)
                return {
                    "sentiment_score": analysis.sentiment_score,
                    "sentiment_label": analysis.sentiment_label,
                    "readability_score": analysis.readability_score,
                    "coherence_score": analysis.coherence_score,
                    "seo_score": analysis.seo_score,
                    "keywords": analysis.keywords,
                    "processing_time": analysis.processing_time
                }
            else:
                enhancement = await self.ai_engine.enhance_content(
                    content, 
                    AIEnhancementType.CONTENT_OPTIMIZATION
                )
                return {
                    "enhanced_content": enhancement.enhanced_content,
                    "confidence_score": enhancement.confidence_score,
                    "processing_time": enhancement.processing_time
                }
                
        except Exception as e:
            logger.error(f"Error executing content review task: {e}")
            raise
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type == TaskType.CONTENT_REVIEW

class ApprovalExecutor(TaskExecutor):
    """Ejecutor para aprobaciones"""
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Ejecutar proceso de aprobación"""
        try:
            # Simular proceso de aprobación
            approver = task.parameters.get("approver", "admin")
            auto_approve = task.parameters.get("auto_approve", False)
            
            if auto_approve:
                return {
                    "approved": True,
                    "approver": "system",
                    "approved_at": datetime.now().isoformat(),
                    "comments": "Auto-approved"
                }
            else:
                # En una implementación real, esto enviaría notificaciones
                return {
                    "approved": False,
                    "approver": approver,
                    "status": "pending_approval",
                    "notification_sent": True
                }
                
        except Exception as e:
            logger.error(f"Error executing approval task: {e}")
            raise
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type == TaskType.APPROVAL

class NotificationExecutor(TaskExecutor):
    """Ejecutor para notificaciones"""
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Any:
        """Ejecutar notificación"""
        try:
            notification_type = task.parameters.get("type", "email")
            recipients = task.parameters.get("recipients", [])
            subject = task.parameters.get("subject", "BUL Workflow Notification")
            message = task.parameters.get("message", "")
            
            # Simular envío de notificación
            result = {
                "notification_type": notification_type,
                "recipients": recipients,
                "subject": subject,
                "message": message,
                "sent_at": datetime.now().isoformat(),
                "status": "sent"
            }
            
            logger.info(f"Notification sent: {subject} to {len(recipients)} recipients")
            return result
            
        except Exception as e:
            logger.error(f"Error executing notification task: {e}")
            raise
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type == TaskType.NOTIFICATION

class WorkflowEngine:
    """
    Motor de Flujos de Trabajo
    
    Maneja la ejecución de flujos de trabajo automatizados para el sistema BUL.
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.executors: List[TaskExecutor] = []
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.is_initialized = False
        
        logger.info("Workflow Engine initialized")
    
    async def initialize(self, bul_engine=None, ai_engine=None) -> bool:
        """Inicializar el motor de flujos de trabajo"""
        try:
            # Registrar ejecutores por defecto
            if bul_engine:
                self.executors.append(DocumentGenerationExecutor(bul_engine))
            
            if ai_engine:
                self.executors.append(ContentReviewExecutor(ai_engine))
            
            self.executors.append(ApprovalExecutor())
            self.executors.append(NotificationExecutor())
            
            # Crear flujos de trabajo por defecto
            await self._create_default_workflows()
            
            self.is_initialized = True
            logger.info("Workflow Engine fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Workflow Engine: {e}")
            return False
    
    async def _create_default_workflows(self):
        """Crear flujos de trabajo por defecto"""
        # Flujo de trabajo para generación de documentos con revisión
        doc_review_workflow = Workflow(
            name="Document Generation with Review",
            description="Genera un documento y lo somete a revisión automática",
            version="1.0",
            created_by="system",
            tasks=[
                WorkflowTask(
                    name="Generate Document",
                    description="Generar documento empresarial",
                    task_type=TaskType.DOCUMENT_GENERATION,
                    priority=10,
                    parameters={
                        "query": "{{workflow_variables.query}}",
                        "business_area": "{{workflow_variables.business_area}}",
                        "document_type": "{{workflow_variables.document_type}}",
                        "company_name": "{{workflow_variables.company_name}}"
                    }
                ),
                WorkflowTask(
                    name="Review Content",
                    description="Revisar contenido generado",
                    task_type=TaskType.CONTENT_REVIEW,
                    priority=8,
                    dependencies=["Generate Document"],
                    parameters={
                        "content": "{{tasks.Generate_Document.result.content}}",
                        "review_type": "comprehensive"
                    }
                ),
                WorkflowTask(
                    name="Send Notification",
                    description="Notificar completación",
                    task_type=TaskType.NOTIFICATION,
                    priority=5,
                    dependencies=["Review Content"],
                    parameters={
                        "type": "email",
                        "recipients": ["{{workflow_variables.user_email}}"],
                        "subject": "Documento generado y revisado",
                        "message": "Su documento ha sido generado y revisado exitosamente."
                    }
                )
            ],
            triggers=[
                WorkflowTrigger(
                    name="Manual Trigger",
                    trigger_type=TriggerType.MANUAL,
                    is_active=True
                )
            ]
        )
        
        self.workflows[doc_review_workflow.id] = doc_review_workflow
        
        # Flujo de trabajo para aprobación de documentos
        approval_workflow = Workflow(
            name="Document Approval Process",
            description="Proceso de aprobación para documentos importantes",
            version="1.0",
            created_by="system",
            tasks=[
                WorkflowTask(
                    name="Generate Document",
                    description="Generar documento",
                    task_type=TaskType.DOCUMENT_GENERATION,
                    priority=10,
                    parameters={
                        "query": "{{workflow_variables.query}}",
                        "business_area": "{{workflow_variables.business_area}}"
                    }
                ),
                WorkflowTask(
                    name="Review Content",
                    description="Revisar contenido",
                    task_type=TaskType.CONTENT_REVIEW,
                    priority=8,
                    dependencies=["Generate Document"],
                    parameters={
                        "content": "{{tasks.Generate_Document.result.content}}",
                        "review_type": "comprehensive"
                    }
                ),
                WorkflowTask(
                    name="Request Approval",
                    description="Solicitar aprobación",
                    task_type=TaskType.APPROVAL,
                    priority=6,
                    dependencies=["Review Content"],
                    parameters={
                        "approver": "{{workflow_variables.approver}}",
                        "auto_approve": False
                    }
                ),
                WorkflowTask(
                    name="Notify Approval",
                    description="Notificar aprobación",
                    task_type=TaskType.NOTIFICATION,
                    priority=4,
                    dependencies=["Request Approval"],
                    parameters={
                        "type": "email",
                        "recipients": ["{{workflow_variables.user_email}}"],
                        "subject": "Documento aprobado",
                        "message": "Su documento ha sido aprobado y está listo para usar."
                    }
                )
            ],
            triggers=[
                WorkflowTrigger(
                    name="Manual Trigger",
                    trigger_type=TriggerType.MANUAL,
                    is_active=True
                )
            ]
        )
        
        self.workflows[approval_workflow.id] = approval_workflow
        
        logger.info("Default workflows created")
    
    async def create_workflow(self, workflow: Workflow) -> str:
        """Crear nuevo flujo de trabajo"""
        self.workflows[workflow.id] = workflow
        logger.info(f"Workflow created: {workflow.name}")
        return workflow.id
    
    async def execute_workflow(self, workflow_id: str, variables: Dict[str, Any] = None) -> str:
        """Ejecutar flujo de trabajo"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        workflow.variables.update(variables or {})
        
        # Crear tarea asíncrona para ejecutar el flujo
        task = asyncio.create_task(self._execute_workflow_tasks(workflow))
        self.running_workflows[workflow_id] = task
        
        logger.info(f"Workflow {workflow.name} started")
        return workflow_id
    
    async def _execute_workflow_tasks(self, workflow: Workflow):
        """Ejecutar tareas del flujo de trabajo"""
        try:
            completed_tasks = set()
            context = {"workflow_variables": workflow.variables, "tasks": {}}
            
            while len(completed_tasks) < len(workflow.tasks):
                # Encontrar tareas listas para ejecutar
                ready_tasks = []
                for task in workflow.tasks:
                    if (task.id not in completed_tasks and 
                        task.status == TaskStatus.PENDING and
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # No hay tareas listas, verificar si hay tareas fallidas
                    failed_tasks = [t for t in workflow.tasks if t.status == TaskStatus.FAILED]
                    if failed_tasks:
                        workflow.status = WorkflowStatus.FAILED
                        logger.error(f"Workflow {workflow.name} failed due to failed tasks")
                        return
                    else:
                        # Esperar un poco y reintentar
                        await asyncio.sleep(1)
                        continue
                
                # Ejecutar tareas listas en paralelo
                tasks_to_run = []
                for task in ready_tasks:
                    if task.priority >= 5:  # Solo tareas de alta prioridad en paralelo
                        tasks_to_run.append(self._execute_task(task, context))
                
                if tasks_to_run:
                    await asyncio.gather(*tasks_to_run, return_exceptions=True)
                
                # Actualizar tareas completadas
                for task in workflow.tasks:
                    if task.status == TaskStatus.COMPLETED and task.id not in completed_tasks:
                        completed_tasks.add(task.id)
                        context["tasks"][task.name.replace(" ", "_")] = {"result": task.result}
            
            # Marcar flujo como completado
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            logger.info(f"Workflow {workflow.name} completed successfully")
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Workflow {workflow.name} failed: {e}")
        finally:
            # Limpiar tarea en ejecución
            if workflow.id in self.running_workflows:
                del self.running_workflows[workflow.id]
    
    async def _execute_task(self, task: WorkflowTask, context: Dict[str, Any]):
        """Ejecutar tarea individual"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Encontrar ejecutor apropiado
            executor = None
            for exec in self.executors:
                if exec.can_handle(task.task_type):
                    executor = exec
                    break
            
            if not executor:
                raise ValueError(f"No executor found for task type: {task.task_type}")
            
            # Ejecutar tarea con timeout
            result = await asyncio.wait_for(
                executor.execute(task, context),
                timeout=task.timeout
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.name} completed successfully")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error_message = f"Task timed out after {task.timeout} seconds"
            logger.error(f"Task {task.name} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                logger.warning(f"Task {task.name} failed, retrying ({task.retry_count}/{task.max_retries})")
            else:
                logger.error(f"Task {task.name} failed permanently: {e}")
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Obtener estado del flujo de trabajo"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED]) / len(workflow.tasks) * 100,
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status.value,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "error_message": task.error_message
                }
                for task in workflow.tasks
            ]
        }
    
    async def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Obtener flujos de trabajo disponibles"""
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "status": workflow.status.value,
                "created_at": workflow.created_at.isoformat(),
                "created_by": workflow.created_by,
                "task_count": len(workflow.tasks)
            }
            for workflow in self.workflows.values()
        ]
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancelar flujo de trabajo"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        
        # Cancelar tarea en ejecución si existe
        if workflow_id in self.running_workflows:
            self.running_workflows[workflow_id].cancel()
            del self.running_workflows[workflow_id]
        
        logger.info(f"Workflow {workflow.name} cancelled")
        return True

# Global workflow engine instance
_workflow_engine: Optional[WorkflowEngine] = None

async def get_global_workflow_engine() -> WorkflowEngine:
    """Obtener la instancia global del motor de flujos de trabajo"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
        # Se inicializará cuando se necesite con las dependencias
    return _workflow_engine
























