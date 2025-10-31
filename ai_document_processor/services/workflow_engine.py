"""
Motor de Workflows
=================

Motor para crear y ejecutar workflows automatizados de procesamiento de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class WorkflowStatus(str, Enum):
    """Estados de workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(str, Enum):
    """Estados de tarea"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TriggerType(str, Enum):
    """Tipos de trigger"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    FILE_UPLOAD = "file_upload"
    API_CALL = "api_call"
    WEBHOOK = "webhook"

@dataclass
class WorkflowTask:
    """Tarea de workflow"""
    id: str
    name: str
    task_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class WorkflowDefinition:
    """Definición de workflow"""
    id: str
    name: str
    description: str
    version: str
    tasks: List[WorkflowTask]
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowExecution:
    """Ejecución de workflow"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    tasks: List[WorkflowTask] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    triggered_by: str = "manual"
    triggered_at: datetime = field(default_factory=datetime.now)

class WorkflowEngine:
    """Motor de workflows"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.running_executions: Dict[str, asyncio.Task] = {}
        self.scheduled_workflows: Dict[str, asyncio.Task] = {}
        
    async def initialize(self):
        """Inicializa el motor de workflows"""
        logger.info("Inicializando motor de workflows...")
        
        # Registrar handlers de tareas por defecto
        await self._register_default_handlers()
        
        # Cargar workflows predefinidos
        await self._load_predefined_workflows()
        
        # Iniciar scheduler
        asyncio.create_task(self._scheduler_worker())
        
        logger.info("Motor de workflows inicializado")
    
    async def _register_default_handlers(self):
        """Registra handlers de tareas por defecto"""
        try:
            # Importar servicios necesarios
            from services.document_processor import DocumentProcessor
            from services.ai_classifier import AIClassifier
            from services.professional_transformer import ProfessionalTransformer
            from services.translation_service import TranslationService
            from services.batch_processor import BatchProcessor
            
            # Inicializar servicios
            self.document_processor = DocumentProcessor()
            self.ai_classifier = AIClassifier()
            self.professional_transformer = ProfessionalTransformer()
            self.translation_service = TranslationService()
            self.batch_processor = BatchProcessor()
            
            await self.document_processor.initialize()
            await self.ai_classifier.initialize()
            await self.professional_transformer.initialize()
            await self.translation_service.initialize()
            await self.batch_processor.initialize()
            
            # Registrar handlers
            self.task_handlers = {
                "process_document": self._handle_process_document,
                "classify_document": self._handle_classify_document,
                "transform_document": self._handle_transform_document,
                "translate_document": self._handle_translate_document,
                "batch_process": self._handle_batch_process,
                "wait": self._handle_wait,
                "condition": self._handle_condition,
                "loop": self._handle_loop,
                "parallel": self._handle_parallel,
                "merge_results": self._handle_merge_results,
                "send_notification": self._handle_send_notification,
                "save_result": self._handle_save_result
            }
            
            logger.info(f"Registrados {len(self.task_handlers)} handlers de tareas")
            
        except Exception as e:
            logger.error(f"Error registrando handlers por defecto: {e}")
    
    async def _load_predefined_workflows(self):
        """Carga workflows predefinidos"""
        try:
            # Workflow de procesamiento completo
            complete_processing_workflow = WorkflowDefinition(
                id="complete_processing",
                name="Procesamiento Completo de Documento",
                description="Procesa un documento completo: clasifica, transforma y traduce",
                version="1.0.0",
                tasks=[
                    WorkflowTask(
                        id="classify",
                        name="Clasificar Documento",
                        task_type="classify_document",
                        parameters={"include_analysis": True}
                    ),
                    WorkflowTask(
                        id="transform",
                        name="Transformar a Profesional",
                        task_type="transform_document",
                        parameters={"target_format": "consultancy"},
                        dependencies=["classify"]
                    ),
                    WorkflowTask(
                        id="translate",
                        name="Traducir Documento",
                        task_type="translate_document",
                        parameters={"target_language": "en"},
                        dependencies=["transform"]
                    ),
                    WorkflowTask(
                        id="save_result",
                        name="Guardar Resultado",
                        task_type="save_result",
                        parameters={"output_format": "json"},
                        dependencies=["translate"]
                    )
                ]
            )
            
            # Workflow de procesamiento en lote
            batch_processing_workflow = WorkflowDefinition(
                id="batch_processing",
                name="Procesamiento en Lote",
                description="Procesa múltiples documentos en lote",
                version="1.0.0",
                tasks=[
                    WorkflowTask(
                        id="batch_process",
                        name="Procesar Lote",
                        task_type="batch_process",
                        parameters={"target_format": "consultancy", "language": "es"}
                    ),
                    WorkflowTask(
                        id="generate_report",
                        name="Generar Reporte",
                        task_type="merge_results",
                        parameters={"report_type": "batch_summary"},
                        dependencies=["batch_process"]
                    ),
                    WorkflowTask(
                        id="notify_completion",
                        name="Notificar Completado",
                        task_type="send_notification",
                        parameters={"type": "email", "template": "batch_complete"},
                        dependencies=["generate_report"]
                    )
                ]
            )
            
            # Workflow de análisis avanzado
            advanced_analysis_workflow = WorkflowDefinition(
                id="advanced_analysis",
                name="Análisis Avanzado",
                description="Realiza análisis avanzado de documentos",
                version="1.0.0",
                tasks=[
                    WorkflowTask(
                        id="classify",
                        name="Clasificar",
                        task_type="classify_document",
                        parameters={"include_analysis": True}
                    ),
                    WorkflowTask(
                        id="analyze_sentiment",
                        name="Analizar Sentimientos",
                        task_type="analyze_sentiment",
                        dependencies=["classify"]
                    ),
                    WorkflowTask(
                        id="extract_entities",
                        name="Extraer Entidades",
                        task_type="extract_entities",
                        dependencies=["classify"]
                    ),
                    WorkflowTask(
                        id="merge_analysis",
                        name="Combinar Análisis",
                        task_type="merge_results",
                        parameters={"analysis_type": "comprehensive"},
                        dependencies=["analyze_sentiment", "extract_entities"]
                    )
                ]
            )
            
            # Guardar workflows
            self.workflows["complete_processing"] = complete_processing_workflow
            self.workflows["batch_processing"] = batch_processing_workflow
            self.workflows["advanced_analysis"] = advanced_analysis_workflow
            
            logger.info(f"Cargados {len(self.workflows)} workflows predefinidos")
            
        except Exception as e:
            logger.error(f"Error cargando workflows predefinidos: {e}")
    
    async def create_workflow(self, definition: WorkflowDefinition) -> bool:
        """Crea un nuevo workflow"""
        try:
            self.workflows[definition.id] = definition
            logger.info(f"Workflow creado: {definition.id}")
            return True
        except Exception as e:
            logger.error(f"Error creando workflow: {e}")
            return False
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        variables: Dict[str, Any] = None,
        triggered_by: str = "manual"
    ) -> str:
        """Ejecuta un workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow no encontrado: {workflow_id}")
            
            workflow_def = self.workflows[workflow_id]
            
            # Crear ejecución
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                started_at=datetime.now(),
                variables=variables or {},
                triggered_by=triggered_by
            )
            
            # Clonar tareas
            execution.tasks = [
                WorkflowTask(
                    id=task.id,
                    name=task.name,
                    task_type=task.task_type,
                    parameters=task.parameters.copy(),
                    dependencies=task.dependencies.copy(),
                    max_retries=task.max_retries,
                    timeout_seconds=task.timeout_seconds
                )
                for task in workflow_def.tasks
            ]
            
            self.executions[execution_id] = execution
            
            # Ejecutar workflow
            task = asyncio.create_task(self._execute_workflow(execution))
            self.running_executions[execution_id] = task
            
            logger.info(f"Workflow ejecutado: {workflow_id} (execution: {execution_id})")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error ejecutando workflow: {e}")
            raise
    
    async def _execute_workflow(self, execution: WorkflowExecution):
        """Ejecuta un workflow específico"""
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Ejecutar tareas en orden de dependencias
            completed_tasks = set()
            
            while len(completed_tasks) < len(execution.tasks):
                # Encontrar tareas listas para ejecutar
                ready_tasks = []
                for task in execution.tasks:
                    if (task.id not in completed_tasks and 
                        task.status == TaskStatus.PENDING and
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # No hay tareas listas, verificar si hay errores
                    failed_tasks = [t for t in execution.tasks if t.status == TaskStatus.FAILED]
                    if failed_tasks:
                        execution.status = WorkflowStatus.FAILED
                        execution.error = f"Tareas fallidas: {[t.id for t in failed_tasks]}"
                        break
                    else:
                        # Esperar un poco y reintentar
                        await asyncio.sleep(1)
                        continue
                
                # Ejecutar tareas listas en paralelo
                tasks_to_run = []
                for task in ready_tasks:
                    task_coroutine = self._execute_task(task, execution)
                    tasks_to_run.append(task_coroutine)
                
                if tasks_to_run:
                    await asyncio.gather(*tasks_to_run, return_exceptions=True)
                
                # Actualizar tareas completadas
                for task in execution.tasks:
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]:
                        completed_tasks.add(task.id)
            
            # Finalizar ejecución
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now()
            
            # Limpiar ejecución en memoria después de un tiempo
            asyncio.create_task(self._cleanup_execution(execution.id))
            
        except Exception as e:
            logger.error(f"Error ejecutando workflow {execution.id}: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
    
    async def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution):
        """Ejecuta una tarea específica"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Obtener handler de tarea
            if task.task_type not in self.task_handlers:
                raise ValueError(f"Handler no encontrado para tipo de tarea: {task.task_type}")
            
            handler = self.task_handlers[task.task_type]
            
            # Ejecutar tarea con timeout
            try:
                result = await asyncio.wait_for(
                    handler(task, execution),
                    timeout=task.timeout_seconds
                )
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Timeout después de {task.timeout_seconds} segundos"
                task.completed_at = datetime.now()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # Reintentar si es posible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error = None
                task.started_at = None
                task.completed_at = None
                logger.info(f"Reintentando tarea {task.id} (intento {task.retry_count})")
    
    # Handlers de tareas
    async def _handle_process_document(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para procesar documento"""
        try:
            file_path = execution.variables.get("file_path")
            if not file_path:
                raise ValueError("file_path no encontrado en variables")
            
            from models.document_models import DocumentProcessingRequest, ProfessionalFormat
            
            request = DocumentProcessingRequest(
                filename=Path(file_path).name,
                target_format=ProfessionalFormat(task.parameters.get("target_format", "consultancy")),
                language=task.parameters.get("language", "es"),
                include_analysis=task.parameters.get("include_analysis", True)
            )
            
            result = await self.document_processor.process_document(file_path, request)
            
            # Guardar resultado en variables de ejecución
            execution.variables[f"{task.id}_result"] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            raise
    
    async def _handle_classify_document(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para clasificar documento"""
        try:
            # Obtener texto del resultado anterior o de variables
            text = execution.variables.get("document_text")
            if not text:
                # Intentar obtener de resultado anterior
                for prev_task in execution.tasks:
                    if prev_task.id in task.dependencies and prev_task.result:
                        if hasattr(prev_task.result, 'professional_document'):
                            text = prev_task.result.professional_document.content
                        elif hasattr(prev_task.result, 'content'):
                            text = prev_task.result.content
                        break
            
            if not text:
                raise ValueError("No se encontró texto para clasificar")
            
            analysis = await self.ai_classifier.classify_document(text)
            execution.variables[f"{task.id}_result"] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error clasificando documento: {e}")
            raise
    
    async def _handle_transform_document(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para transformar documento"""
        try:
            # Obtener texto y análisis
            text = execution.variables.get("document_text")
            analysis = None
            
            for prev_task in execution.tasks:
                if prev_task.id in task.dependencies and prev_task.result:
                    if hasattr(prev_task.result, 'professional_document'):
                        text = prev_task.result.professional_document.content
                    elif hasattr(prev_task.result, 'content'):
                        text = prev_task.result.content
                    elif hasattr(prev_task.result, 'area'):
                        analysis = prev_task.result
                    break
            
            if not text:
                raise ValueError("No se encontró texto para transformar")
            
            from models.document_models import ProfessionalFormat
            
            professional_doc = await self.professional_transformer.transform_to_professional(
                text,
                analysis,
                ProfessionalFormat(task.parameters.get("target_format", "consultancy")),
                task.parameters.get("language", "es")
            )
            
            execution.variables[f"{task.id}_result"] = professional_doc
            
            return professional_doc
            
        except Exception as e:
            logger.error(f"Error transformando documento: {e}")
            raise
    
    async def _handle_translate_document(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para traducir documento"""
        try:
            # Obtener contenido a traducir
            content = None
            for prev_task in execution.tasks:
                if prev_task.id in task.dependencies and prev_task.result:
                    if hasattr(prev_task.result, 'content'):
                        content = prev_task.result.content
                    break
            
            if not content:
                raise ValueError("No se encontró contenido para traducir")
            
            translation_result = await self.translation_service.translate_document(
                content,
                task.parameters.get("target_language", "en"),
                task.parameters.get("source_language"),
                task.parameters.get("preserve_formatting", True)
            )
            
            execution.variables[f"{task.id}_result"] = translation_result
            
            return translation_result
            
        except Exception as e:
            logger.error(f"Error traduciendo documento: {e}")
            raise
    
    async def _handle_batch_process(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para procesamiento en lote"""
        try:
            file_paths = execution.variables.get("file_paths", [])
            if not file_paths:
                raise ValueError("file_paths no encontrado en variables")
            
            from models.document_models import ProfessionalFormat
            
            result = await self.batch_processor.process_batch(
                file_paths,
                ProfessionalFormat(task.parameters.get("target_format", "consultancy")),
                task.parameters.get("language", "es"),
                task.parameters.get("output_dir")
            )
            
            execution.variables[f"{task.id}_result"] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error en procesamiento en lote: {e}")
            raise
    
    async def _handle_wait(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para esperar"""
        wait_seconds = task.parameters.get("seconds", 1)
        await asyncio.sleep(wait_seconds)
        return {"waited_seconds": wait_seconds}
    
    async def _handle_condition(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para condiciones"""
        condition = task.parameters.get("condition")
        if not condition:
            return {"result": True}
        
        # Evaluar condición simple
        # En una implementación completa, usar un evaluador de expresiones
        return {"result": True}
    
    async def _handle_loop(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para loops"""
        iterations = task.parameters.get("iterations", 1)
        results = []
        
        for i in range(iterations):
            # Ejecutar tareas del loop
            result = {"iteration": i, "result": f"Loop iteration {i}"}
            results.append(result)
        
        return {"loop_results": results}
    
    async def _handle_parallel(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para ejecución en paralelo"""
        parallel_tasks = task.parameters.get("tasks", [])
        results = []
        
        for parallel_task in parallel_tasks:
            # Ejecutar tarea en paralelo
            result = await self._execute_task(parallel_task, execution)
            results.append(result)
        
        return {"parallel_results": results}
    
    async def _handle_merge_results(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para combinar resultados"""
        merge_type = task.parameters.get("type", "simple")
        results = []
        
        # Recopilar resultados de tareas dependientes
        for prev_task in execution.tasks:
            if prev_task.id in task.dependencies and prev_task.result:
                results.append(prev_task.result)
        
        if merge_type == "batch_summary":
            return {
                "type": "batch_summary",
                "total_results": len(results),
                "results": results
            }
        else:
            return {
                "type": "merged",
                "results": results
            }
    
    async def _handle_send_notification(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para enviar notificaciones"""
        try:
            from services.notification_service import NotificationService, NotificationType, NotificationPriority
            
            notification_service = NotificationService()
            await notification_service.initialize()
            
            title = task.parameters.get("title", "Notificación de Workflow")
            content = task.parameters.get("content", f"Workflow {execution.workflow_id} completado")
            notification_type = NotificationType(task.parameters.get("type", "console"))
            priority = NotificationPriority(task.parameters.get("priority", "medium"))
            
            await notification_service.send_notification(
                title=title,
                content=content,
                priority=priority,
                notification_type=notification_type,
                metadata={"workflow_id": execution.workflow_id, "execution_id": execution.id}
            )
            
            return {"notification_sent": True}
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
            raise
    
    async def _handle_save_result(self, task: WorkflowTask, execution: WorkflowExecution):
        """Handler para guardar resultados"""
        try:
            output_format = task.parameters.get("output_format", "json")
            output_path = task.parameters.get("output_path", f"workflow_result_{execution.id}.{output_format}")
            
            # Recopilar resultados
            results = {}
            for prev_task in execution.tasks:
                if prev_task.result:
                    results[prev_task.id] = prev_task.result
            
            # Guardar según formato
            if output_format == "json":
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            else:
                # Otros formatos
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(str(results))
            
            return {"saved_to": output_path}
            
        except Exception as e:
            logger.error(f"Error guardando resultado: {e}")
            raise
    
    async def _scheduler_worker(self):
        """Worker para workflows programados"""
        while True:
            try:
                await asyncio.sleep(60)  # Verificar cada minuto
                
                # Verificar workflows programados
                for workflow_id, workflow_def in self.workflows.items():
                    for trigger in workflow_def.triggers:
                        if trigger.get("type") == "scheduled":
                            schedule = trigger.get("schedule")
                            if self._should_trigger_scheduled(schedule):
                                await self.execute_workflow(
                                    workflow_id,
                                    triggered_by="scheduler"
                                )
                
            except Exception as e:
                logger.error(f"Error en scheduler worker: {e}")
    
    def _should_trigger_scheduled(self, schedule: str) -> bool:
        """Verifica si un workflow programado debe ejecutarse"""
        # Implementación simple de cron
        # En producción, usar una librería como croniter
        return False
    
    async def _cleanup_execution(self, execution_id: str):
        """Limpia ejecución después de un tiempo"""
        try:
            await asyncio.sleep(3600)  # Esperar 1 hora
            
            if execution_id in self.running_executions:
                del self.running_executions[execution_id]
            
            if execution_id in self.executions:
                del self.executions[execution_id]
            
            logger.info(f"Ejecución limpiada: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error limpiando ejecución: {e}")
    
    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de un workflow"""
        try:
            if execution_id not in self.executions:
                return None
            
            execution = self.executions[execution_id]
            
            return {
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "triggered_by": execution.triggered_by,
                "tasks": [
                    {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status.value,
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "error": task.error
                    }
                    for task in execution.tasks
                ],
                "error": execution.error
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de workflow: {e}")
            return None
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancela un workflow en ejecución"""
        try:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.now()
            
            # Cancelar tarea de ejecución si está corriendo
            if execution_id in self.running_executions:
                self.running_executions[execution_id].cancel()
                del self.running_executions[execution_id]
            
            logger.info(f"Workflow cancelado: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelando workflow: {e}")
            return False
    
    async def get_workflow_definitions(self) -> List[Dict[str, Any]]:
        """Obtiene definiciones de workflows"""
        try:
            return [
                {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "version": workflow.version,
                    "task_count": len(workflow.tasks),
                    "created_at": workflow.created_at.isoformat(),
                    "updated_at": workflow.updated_at.isoformat()
                }
                for workflow in self.workflows.values()
            ]
        except Exception as e:
            logger.error(f"Error obteniendo definiciones de workflows: {e}")
            return []
    
    async def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene historial de ejecuciones"""
        try:
            executions = list(self.executions.values())
            executions.sort(key=lambda x: x.started_at, reverse=True)
            
            return [
                {
                    "execution_id": execution.id,
                    "workflow_id": execution.workflow_id,
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "triggered_by": execution.triggered_by,
                    "duration_seconds": (
                        (execution.completed_at - execution.started_at).total_seconds()
                        if execution.completed_at else None
                    )
                }
                for execution in executions[:limit]
            ]
        except Exception as e:
            logger.error(f"Error obteniendo historial de ejecuciones: {e}")
            return []


