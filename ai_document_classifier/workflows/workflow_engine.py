"""
Workflow Automation Engine
==========================

Advanced workflow automation system for document processing
with conditional logic, parallel execution, and integration capabilities.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import yaml
import networkx as nx

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ConditionType(Enum):
    """Condition types for workflow logic"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    CUSTOM = "custom"

@dataclass
class WorkflowCondition:
    """Workflow condition definition"""
    field: str
    operator: ConditionType
    value: Any
    custom_function: Optional[str] = None

@dataclass
class WorkflowTask:
    """Workflow task definition"""
    id: str
    name: str
    description: str
    task_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[WorkflowCondition] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    parallel: bool = False
    error_handling: str = "stop"  # stop, continue, retry

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    id: str
    name: str
    description: str
    version: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    notifications: List[Dict[str, Any]] = field(default_factory=list)

class WorkflowEngine:
    """
    Advanced workflow automation engine
    """
    
    def __init__(self, workflows_dir: Optional[str] = None):
        """
        Initialize workflow engine
        
        Args:
            workflows_dir: Directory for workflow definitions
        """
        self.workflows_dir = Path(workflows_dir) if workflows_dir else Path(__file__).parent / "definitions"
        self.workflows_dir.mkdir(exist_ok=True)
        
        # Workflow registry
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        self.condition_handlers: Dict[ConditionType, Callable] = {}
        
        # Initialize built-in handlers
        self._initialize_builtin_handlers()
        
        # Load workflow definitions
        self._load_workflow_definitions()
    
    def _initialize_builtin_handlers(self):
        """Initialize built-in task and condition handlers"""
        
        # Document processing tasks
        self.task_handlers["classify_document"] = self._handle_classify_document
        self.task_handlers["extract_text"] = self._handle_extract_text
        self.task_handlers["generate_template"] = self._handle_generate_template
        self.task_handlers["validate_document"] = self._handle_validate_document
        self.task_handlers["analyze_content"] = self._handle_analyze_content
        self.task_handlers["translate_document"] = self._handle_translate_document
        self.task_handlers["check_plagiarism"] = self._handle_check_plagiarism
        self.task_handlers["generate_summary"] = self._handle_generate_summary
        
        # Data processing tasks
        self.task_handlers["transform_data"] = self._handle_transform_data
        self.task_handlers["validate_data"] = self._handle_validate_data
        self.task_handlers["enrich_data"] = self._handle_enrich_data
        self.task_handlers["export_data"] = self._handle_export_data
        
        # Integration tasks
        self.task_handlers["send_notification"] = self._handle_send_notification
        self.task_handlers["call_api"] = self._handle_call_api
        self.task_handlers["save_to_database"] = self._handle_save_to_database
        self.task_handlers["upload_file"] = self._handle_upload_file
        
        # Condition handlers
        self.condition_handlers[ConditionType.EQUALS] = lambda field, value, context: context.get(field) == value
        self.condition_handlers[ConditionType.NOT_EQUALS] = lambda field, value, context: context.get(field) != value
        self.condition_handlers[ConditionType.GREATER_THAN] = lambda field, value, context: context.get(field, 0) > value
        self.condition_handlers[ConditionType.LESS_THAN] = lambda field, value, context: context.get(field, 0) < value
        self.condition_handlers[ConditionType.CONTAINS] = lambda field, value, context: value in str(context.get(field, ""))
        self.condition_handlers[ConditionType.NOT_CONTAINS] = lambda field, value, context: value not in str(context.get(field, ""))
    
    def _load_workflow_definitions(self):
        """Load workflow definitions from files"""
        for workflow_file in self.workflows_dir.glob("*.yaml"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                
                workflow = self._parse_workflow_definition(workflow_data)
                self.workflows[workflow.id] = workflow
                logger.info(f"Loaded workflow: {workflow.name}")
                
            except Exception as e:
                logger.error(f"Error loading workflow {workflow_file}: {e}")
    
    def _parse_workflow_definition(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Parse workflow definition from data"""
        tasks = []
        for task_data in data.get("tasks", []):
            conditions = []
            for condition_data in task_data.get("conditions", []):
                condition = WorkflowCondition(
                    field=condition_data["field"],
                    operator=ConditionType(condition_data["operator"]),
                    value=condition_data["value"],
                    custom_function=condition_data.get("custom_function")
                )
                conditions.append(condition)
            
            task = WorkflowTask(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data.get("description", ""),
                task_type=task_data["type"],
                parameters=task_data.get("parameters", {}),
                conditions=conditions,
                retry_count=task_data.get("retry_count", 0),
                max_retries=task_data.get("max_retries", 3),
                timeout=task_data.get("timeout"),
                dependencies=task_data.get("dependencies", []),
                parallel=task_data.get("parallel", False),
                error_handling=task_data.get("error_handling", "stop")
            )
            tasks.append(task)
        
        return WorkflowDefinition(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            tasks=tasks,
            triggers=data.get("triggers", []),
            variables=data.get("variables", {}),
            timeout=data.get("timeout"),
            retry_policy=data.get("retry_policy", {}),
            notifications=data.get("notifications", [])
        )
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a new workflow"""
        self.workflows[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.name}")
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a custom task handler"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered task handler: {task_type}")
    
    def register_condition_handler(self, condition_type: ConditionType, handler: Callable):
        """Register a custom condition handler"""
        self.condition_handlers[condition_type] = handler
        logger.info(f"Registered condition handler: {condition_type.value}")
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Execute a workflow
        
        Args:
            workflow_id: ID of workflow to execute
            context: Initial context for workflow execution
            
        Returns:
            Workflow execution instance
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now(),
            context=context or {}
        )
        
        self.executions[execution_id] = execution
        
        try:
            # Build task dependency graph
            task_graph = self._build_task_graph(workflow.tasks)
            
            # Execute tasks in dependency order
            await self._execute_tasks(workflow, execution, task_graph)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append(str(e))
            logger.error(f"Workflow execution failed: {e}")
        
        return execution
    
    def _build_task_graph(self, tasks: List[WorkflowTask]) -> nx.DiGraph:
        """Build task dependency graph"""
        graph = nx.DiGraph()
        
        # Add nodes
        for task in tasks:
            graph.add_node(task.id, task=task)
        
        # Add edges based on dependencies
        for task in tasks:
            for dependency in task.dependencies:
                if dependency in graph:
                    graph.add_edge(dependency, task.id)
        
        return graph
    
    async def _execute_tasks(
        self, 
        workflow: WorkflowDefinition, 
        execution: WorkflowExecution, 
        task_graph: nx.DiGraph
    ):
        """Execute tasks in dependency order"""
        # Get topological order
        try:
            task_order = list(nx.topological_sort(task_graph))
        except nx.NetworkXError:
            raise ValueError("Workflow has circular dependencies")
        
        # Group tasks by parallel execution
        parallel_groups = self._group_parallel_tasks(task_graph, task_order)
        
        for group in parallel_groups:
            if len(group) == 1:
                # Single task
                task_id = group[0]
                await self._execute_single_task(workflow, execution, task_id)
            else:
                # Parallel tasks
                await self._execute_parallel_tasks(workflow, execution, group)
    
    def _group_parallel_tasks(self, graph: nx.DiGraph, task_order: List[str]) -> List[List[str]]:
        """Group tasks for parallel execution"""
        groups = []
        current_group = []
        
        for task_id in task_order:
            task_data = graph.nodes[task_id]
            task = task_data["task"]
            
            if task.parallel and not current_group:
                current_group = [task_id]
            elif task.parallel and current_group:
                current_group.append(task_id)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([task_id])
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _execute_single_task(
        self, 
        workflow: WorkflowDefinition, 
        execution: WorkflowExecution, 
        task_id: str
    ):
        """Execute a single task"""
        task = next((t for t in workflow.tasks if t.id == task_id), None)
        if not task:
            return
        
        # Check conditions
        if not self._evaluate_conditions(task.conditions, execution.context):
            execution.tasks[task_id] = {
                "status": TaskStatus.SKIPPED.value,
                "message": "Conditions not met",
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            }
            return
        
        # Execute task
        execution.tasks[task_id] = {
            "status": TaskStatus.RUNNING.value,
            "started_at": datetime.now().isoformat()
        }
        
        try:
            if task.task_type in self.task_handlers:
                result = await self.task_handlers[task.task_type](task, execution.context)
                execution.tasks[task_id].update({
                    "status": TaskStatus.COMPLETED.value,
                    "completed_at": datetime.now().isoformat(),
                    "result": result
                })
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            execution.tasks[task_id].update({
                "status": TaskStatus.FAILED.value,
                "completed_at": datetime.now().isoformat(),
                "error": str(e)
            })
            
            if task.error_handling == "stop":
                raise
            elif task.error_handling == "continue":
                logger.warning(f"Task {task_id} failed but continuing: {e}")
    
    async def _execute_parallel_tasks(
        self, 
        workflow: WorkflowDefinition, 
        execution: WorkflowExecution, 
        task_ids: List[str]
    ):
        """Execute tasks in parallel"""
        tasks = [self._execute_single_task(workflow, execution, task_id) for task_id in task_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _evaluate_conditions(
        self, 
        conditions: List[WorkflowCondition], 
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate workflow conditions"""
        for condition in conditions:
            if condition.operator in self.condition_handlers:
                handler = self.condition_handlers[condition.operator]
                if not handler(condition.field, condition.value, context):
                    return False
            else:
                logger.warning(f"Unknown condition operator: {condition.operator}")
                return False
        
        return True
    
    # Built-in task handlers
    async def _handle_classify_document(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document classification task"""
        # This would integrate with the document classifier
        document_content = context.get("document_content", "")
        use_ai = task.parameters.get("use_ai", True)
        
        # Simulate classification
        return {
            "document_type": "novel",
            "confidence": 0.95,
            "keywords": ["story", "character", "plot"],
            "processing_time": 0.5
        }
    
    async def _handle_extract_text(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text extraction task"""
        file_path = context.get("file_path", "")
        
        # Simulate text extraction
        return {
            "extracted_text": "Sample extracted text content",
            "word_count": 150,
            "language": "en"
        }
    
    async def _handle_generate_template(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle template generation task"""
        document_type = context.get("document_type", "novel")
        template_format = task.parameters.get("format", "json")
        
        # Simulate template generation
        return {
            "template_id": f"template_{document_type}_{uuid.uuid4().hex[:8]}",
            "format": template_format,
            "sections": ["Introduction", "Body", "Conclusion"]
        }
    
    async def _handle_validate_document(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document validation task"""
        document_content = context.get("document_content", "")
        
        # Simulate validation
        return {
            "is_valid": True,
            "errors": [],
            "warnings": ["Consider adding more detail"],
            "score": 85
        }
    
    async def _handle_analyze_content(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content analysis task"""
        document_content = context.get("document_content", "")
        
        # Simulate content analysis
        return {
            "sentiment": "positive",
            "readability_score": 75,
            "key_topics": ["technology", "innovation"],
            "word_count": len(document_content.split())
        }
    
    async def _handle_translate_document(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document translation task"""
        document_content = context.get("document_content", "")
        target_language = task.parameters.get("target_language", "es")
        
        # Simulate translation
        return {
            "translated_content": f"Translated content to {target_language}",
            "source_language": "en",
            "target_language": target_language,
            "confidence": 0.92
        }
    
    async def _handle_check_plagiarism(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plagiarism check task"""
        document_content = context.get("document_content", "")
        
        # Simulate plagiarism check
        return {
            "plagiarism_score": 5.2,
            "is_original": True,
            "sources_found": 0,
            "similarity_percentage": 5.2
        }
    
    async def _handle_generate_summary(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle summary generation task"""
        document_content = context.get("document_content", "")
        max_length = task.parameters.get("max_length", 100)
        
        # Simulate summary generation
        return {
            "summary": "This document discusses important topics related to the subject matter.",
            "word_count": len(document_content.split()),
            "summary_length": max_length
        }
    
    async def _handle_transform_data(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data transformation task"""
        input_data = context.get("input_data", {})
        transformation_type = task.parameters.get("type", "format")
        
        # Simulate data transformation
        return {
            "transformed_data": input_data,
            "transformation_type": transformation_type,
            "records_processed": len(input_data) if isinstance(input_data, list) else 1
        }
    
    async def _handle_validate_data(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation task"""
        input_data = context.get("input_data", {})
        validation_rules = task.parameters.get("rules", [])
        
        # Simulate data validation
        return {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "records_validated": len(input_data) if isinstance(input_data, list) else 1
        }
    
    async def _handle_enrich_data(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data enrichment task"""
        input_data = context.get("input_data", {})
        enrichment_type = task.parameters.get("type", "metadata")
        
        # Simulate data enrichment
        return {
            "enriched_data": input_data,
            "enrichment_type": enrichment_type,
            "fields_added": ["timestamp", "source", "confidence"]
        }
    
    async def _handle_export_data(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data export task"""
        input_data = context.get("input_data", {})
        export_format = task.parameters.get("format", "json")
        file_path = task.parameters.get("file_path", "output.json")
        
        # Simulate data export
        return {
            "export_path": file_path,
            "format": export_format,
            "records_exported": len(input_data) if isinstance(input_data, list) else 1,
            "file_size": 1024
        }
    
    async def _handle_send_notification(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification sending task"""
        message = task.parameters.get("message", "Workflow notification")
        recipients = task.parameters.get("recipients", [])
        notification_type = task.parameters.get("type", "email")
        
        # Simulate notification sending
        return {
            "notification_sent": True,
            "recipients": recipients,
            "type": notification_type,
            "message_id": str(uuid.uuid4())
        }
    
    async def _handle_call_api(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API call task"""
        url = task.parameters.get("url", "")
        method = task.parameters.get("method", "GET")
        headers = task.parameters.get("headers", {})
        data = task.parameters.get("data", {})
        
        # Simulate API call
        return {
            "status_code": 200,
            "response_data": {"success": True},
            "response_time": 0.5,
            "url": url
        }
    
    async def _handle_save_to_database(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database save task"""
        data = context.get("data", {})
        table = task.parameters.get("table", "documents")
        
        # Simulate database save
        return {
            "records_saved": len(data) if isinstance(data, list) else 1,
            "table": table,
            "record_id": str(uuid.uuid4())
        }
    
    async def _handle_upload_file(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file upload task"""
        file_path = context.get("file_path", "")
        destination = task.parameters.get("destination", "cloud_storage")
        
        # Simulate file upload
        return {
            "upload_successful": True,
            "destination": destination,
            "file_url": f"https://storage.example.com/{uuid.uuid4()}",
            "file_size": 1024
        }
    
    def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID"""
        return self.executions.get(execution_id)
    
    def get_workflow_executions(self, workflow_id: Optional[str] = None) -> List[WorkflowExecution]:
        """Get workflow executions"""
        if workflow_id:
            return [exec for exec in self.executions.values() if exec.workflow_id == workflow_id]
        return list(self.executions.values())
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        total_executions = len(self.executions)
        completed_executions = len([e for e in self.executions.values() if e.status == WorkflowStatus.COMPLETED])
        failed_executions = len([e for e in self.executions.values() if e.status == WorkflowStatus.FAILED])
        
        return {
            "total_workflows": len(self.workflows),
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "success_rate": completed_executions / max(total_executions, 1),
            "total_tasks": sum(len(w.tasks) for w in self.workflows.values())
        }

# Example usage
if __name__ == "__main__":
    # Initialize workflow engine
    engine = WorkflowEngine()
    
    # Create a sample workflow
    sample_workflow = WorkflowDefinition(
        id="document_processing_workflow",
        name="Document Processing Workflow",
        description="Complete document processing workflow",
        version="1.0.0",
        tasks=[
            WorkflowTask(
                id="extract_text",
                name="Extract Text",
                description="Extract text from document",
                task_type="extract_text"
            ),
            WorkflowTask(
                id="classify_document",
                name="Classify Document",
                description="Classify document type",
                task_type="classify_document",
                dependencies=["extract_text"]
            ),
            WorkflowTask(
                id="generate_template",
                name="Generate Template",
                description="Generate template based on classification",
                task_type="generate_template",
                dependencies=["classify_document"]
            )
        ]
    )
    
    # Register workflow
    engine.register_workflow(sample_workflow)
    
    # Execute workflow
    context = {
        "document_content": "This is a sample document about technology and innovation.",
        "file_path": "/path/to/document.pdf"
    }
    
    # Note: This would be run with asyncio.run() in a real application
    print("Workflow engine initialized successfully")
    print(f"Registered workflows: {len(engine.workflows)}")
    print(f"Available task handlers: {len(engine.task_handlers)}")



























