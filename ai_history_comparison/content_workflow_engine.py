"""
Content Workflow Automation Engine - Advanced Content Pipeline Management
====================================================================

This module provides comprehensive content workflow automation including:
- Visual workflow builder
- Automated content pipelines
- Content approval workflows
- Multi-stage content processing
- Content scheduling and publishing
- Workflow monitoring and analytics
- Custom workflow templates
- Integration with external systems
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import networkx as nx
from collections import defaultdict, deque
import yaml
from jinja2 import Template, Environment, FileSystemLoader
import schedule
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from webhook import Webhook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class TaskType(Enum):
    """Task type enumeration"""
    CONTENT_GENERATION = "content_generation"
    CONTENT_ANALYSIS = "content_analysis"
    CONTENT_OPTIMIZATION = "content_optimization"
    CONTENT_APPROVAL = "content_approval"
    CONTENT_PUBLISHING = "content_publishing"
    EMAIL_NOTIFICATION = "email_notification"
    WEBHOOK_TRIGGER = "webhook_trigger"
    DATA_TRANSFORMATION = "data_transformation"
    CUSTOM_SCRIPT = "custom_script"
    CONDITIONAL_BRANCH = "conditional_branch"

class TriggerType(Enum):
    """Trigger type enumeration"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    FILE_UPLOAD = "file_upload"
    API_CALL = "api_call"
    CONTENT_UPDATE = "content_update"
    USER_ACTION = "user_action"

@dataclass
class WorkflowTask:
    """Workflow task data structure"""
    task_id: str
    name: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retry_count: int = 3
    retry_delay: int = 60  # seconds
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error_message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WorkflowTrigger:
    """Workflow trigger data structure"""
    trigger_id: str
    name: str
    trigger_type: TriggerType
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class WorkflowExecution:
    """Workflow execution data structure"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    tasks: List[WorkflowTask] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    execution_time: float = 0.0

@dataclass
class WorkflowTemplate:
    """Workflow template data structure"""
    template_id: str
    name: str
    description: str
    category: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WorkflowDefinition:
    """Workflow definition data structure"""
    workflow_id: str
    name: str
    description: str
    version: str
    status: WorkflowStatus
    tasks: List[WorkflowTask] = field(default_factory=list)
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class ContentWorkflowEngine:
    """
    Advanced Content Workflow Automation Engine
    
    Provides comprehensive workflow automation for content management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Workflow Engine"""
        self.config = config
        self.workflows = {}
        self.templates = {}
        self.executions = {}
        self.scheduler = schedule
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running_workflows = {}
        self.workflow_graphs = {}
        
        # Initialize default templates
        self._initialize_templates()
        
        # Start scheduler
        self._start_scheduler()
        
        logger.info("Content Workflow Engine initialized successfully")
    
    def _initialize_templates(self):
        """Initialize default workflow templates"""
        try:
            # Content Creation Workflow
            content_creation_template = WorkflowTemplate(
                template_id="content_creation_001",
                name="Content Creation Workflow",
                description="Automated content creation and publishing workflow",
                category="content_management",
                tasks=[
                    WorkflowTask(
                        task_id="generate_content",
                        name="Generate Content",
                        task_type=TaskType.CONTENT_GENERATION,
                        description="Generate content based on requirements",
                        parameters={
                            "topic": "{{topic}}",
                            "format": "{{format}}",
                            "tone": "{{tone}}",
                            "length": "{{length}}"
                        }
                    ),
                    WorkflowTask(
                        task_id="analyze_content",
                        name="Analyze Content",
                        task_type=TaskType.CONTENT_ANALYSIS,
                        description="Analyze generated content for quality and SEO",
                        dependencies=["generate_content"],
                        parameters={
                            "content_id": "{{generate_content.result.content_id}}"
                        }
                    ),
                    WorkflowTask(
                        task_id="optimize_content",
                        name="Optimize Content",
                        task_type=TaskType.CONTENT_OPTIMIZATION,
                        description="Optimize content based on analysis results",
                        dependencies=["analyze_content"],
                        parameters={
                            "content_id": "{{generate_content.result.content_id}}",
                            "optimization_goals": ["seo", "engagement", "readability"]
                        }
                    ),
                    WorkflowTask(
                        task_id="approve_content",
                        name="Approve Content",
                        task_type=TaskType.CONTENT_APPROVAL,
                        description="Send content for approval",
                        dependencies=["optimize_content"],
                        parameters={
                            "content_id": "{{generate_content.result.content_id}}",
                            "approvers": "{{approvers}}",
                            "approval_timeout": 24
                        }
                    ),
                    WorkflowTask(
                        task_id="publish_content",
                        name="Publish Content",
                        task_type=TaskType.CONTENT_PUBLISHING,
                        description="Publish approved content",
                        dependencies=["approve_content"],
                        parameters={
                            "content_id": "{{generate_content.result.content_id}}",
                            "publish_channels": "{{publish_channels}}"
                        }
                    )
                ],
                triggers=[
                    WorkflowTrigger(
                        trigger_id="manual_trigger",
                        name="Manual Trigger",
                        trigger_type=TriggerType.MANUAL
                    )
                ]
            )
            
            self.templates["content_creation"] = content_creation_template
            
            # Content Analysis Workflow
            content_analysis_template = WorkflowTemplate(
                template_id="content_analysis_001",
                name="Content Analysis Workflow",
                description="Comprehensive content analysis and reporting workflow",
                category="content_analysis",
                tasks=[
                    WorkflowTask(
                        task_id="analyze_content",
                        name="Analyze Content",
                        task_type=TaskType.CONTENT_ANALYSIS,
                        description="Perform comprehensive content analysis",
                        parameters={
                            "content_id": "{{content_id}}"
                        }
                    ),
                    WorkflowTask(
                        task_id="generate_insights",
                        name="Generate Insights",
                        task_type=TaskType.CUSTOM_SCRIPT,
                        description="Generate actionable insights from analysis",
                        dependencies=["analyze_content"],
                        parameters={
                            "analysis_result": "{{analyze_content.result}}"
                        }
                    ),
                    WorkflowTask(
                        task_id="send_report",
                        name="Send Report",
                        task_type=TaskType.EMAIL_NOTIFICATION,
                        description="Send analysis report via email",
                        dependencies=["generate_insights"],
                        parameters={
                            "recipients": "{{recipients}}",
                            "report_data": "{{generate_insights.result}}"
                        }
                    )
                ],
                triggers=[
                    WorkflowTrigger(
                        trigger_id="scheduled_trigger",
                        name="Scheduled Analysis",
                        trigger_type=TriggerType.SCHEDULED,
                        parameters={
                            "schedule": "daily",
                            "time": "09:00"
                        }
                    )
                ]
            )
            
            self.templates["content_analysis"] = content_analysis_template
            
            # Content Optimization Workflow
            content_optimization_template = WorkflowTemplate(
                template_id="content_optimization_001",
                name="Content Optimization Workflow",
                description="Automated content optimization workflow",
                category="content_optimization",
                tasks=[
                    WorkflowTask(
                        task_id="analyze_performance",
                        name="Analyze Performance",
                        task_type=TaskType.CONTENT_ANALYSIS,
                        description="Analyze content performance metrics",
                        parameters={
                            "content_id": "{{content_id}}"
                        }
                    ),
                    WorkflowTask(
                        task_id="identify_optimizations",
                        name="Identify Optimizations",
                        task_type=TaskType.CUSTOM_SCRIPT,
                        description="Identify optimization opportunities",
                        dependencies=["analyze_performance"],
                        parameters={
                            "performance_data": "{{analyze_performance.result}}"
                        }
                    ),
                    WorkflowTask(
                        task_id="apply_optimizations",
                        name="Apply Optimizations",
                        task_type=TaskType.CONTENT_OPTIMIZATION,
                        description="Apply identified optimizations",
                        dependencies=["identify_optimizations"],
                        parameters={
                            "content_id": "{{content_id}}",
                            "optimizations": "{{identify_optimizations.result}}"
                        }
                    ),
                    WorkflowTask(
                        task_id="test_optimizations",
                        name="Test Optimizations",
                        task_type=TaskType.CUSTOM_SCRIPT,
                        description="Test optimized content",
                        dependencies=["apply_optimizations"],
                        parameters={
                            "content_id": "{{content_id}}"
                        }
                    )
                ],
                triggers=[
                    WorkflowTrigger(
                        trigger_id="performance_trigger",
                        name="Performance Trigger",
                        trigger_type=TriggerType.CONDITIONAL_BRANCH,
                        parameters={
                            "condition": "performance_score < 0.7"
                        }
                    )
                ]
            )
            
            self.templates["content_optimization"] = content_optimization_template
            
            logger.info("Default workflow templates initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing templates: {e}")
            raise
    
    def _start_scheduler(self):
        """Start the workflow scheduler"""
        try:
            def run_scheduler():
                while True:
                    self.scheduler.run_pending()
                    time.sleep(1)
            
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            
            logger.info("Workflow scheduler started")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    
    async def create_workflow(self, template_id: str, name: str, 
                            description: str, variables: Dict[str, Any] = None) -> WorkflowDefinition:
        """Create a new workflow from template"""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
            
            template = self.templates[template_id]
            workflow_id = str(uuid.uuid4())
            
            # Create workflow definition
            workflow = WorkflowDefinition(
                workflow_id=workflow_id,
                name=name,
                description=description,
                version="1.0",
                status=WorkflowStatus.DRAFT,
                tasks=template.tasks.copy(),
                triggers=template.triggers.copy(),
                variables=variables or {}
            )
            
            # Build workflow graph
            self._build_workflow_graph(workflow)
            
            # Store workflow
            self.workflows[workflow_id] = workflow
            
            logger.info(f"Workflow {workflow_id} created successfully")
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            raise
    
    def _build_workflow_graph(self, workflow: WorkflowDefinition):
        """Build directed graph for workflow execution"""
        try:
            graph = nx.DiGraph()
            
            # Add nodes (tasks)
            for task in workflow.tasks:
                graph.add_node(task.task_id, task=task)
            
            # Add edges (dependencies)
            for task in workflow.tasks:
                for dependency in task.dependencies:
                    graph.add_edge(dependency, task.task_id)
            
            # Store graph
            self.workflow_graphs[workflow.workflow_id] = graph
            
        except Exception as e:
            logger.error(f"Error building workflow graph: {e}")
    
    async def execute_workflow(self, workflow_id: str, variables: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute a workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            execution_id = str(uuid.uuid4())
            
            # Create execution
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.ACTIVE,
                started_at=datetime.utcnow(),
                variables=variables or {}
            )
            
            # Store execution
            self.executions[execution_id] = execution
            
            # Start execution in background
            asyncio.create_task(self._execute_workflow_tasks(execution))
            
            logger.info(f"Workflow execution {execution_id} started")
            
            return execution
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            raise
    
    async def _execute_workflow_tasks(self, execution: WorkflowExecution):
        """Execute workflow tasks in order"""
        try:
            workflow = self.workflows[execution.workflow_id]
            graph = self.workflow_graphs[execution.workflow_id]
            
            # Get topological order of tasks
            try:
                task_order = list(nx.topological_sort(graph))
            except nx.NetworkXError:
                logger.error("Workflow has circular dependencies")
                execution.status = WorkflowStatus.FAILED
                execution.error_message = "Circular dependencies detected"
                return
            
            # Execute tasks in order
            for task_id in task_order:
                task = next(t for t in workflow.tasks if t.task_id == task_id)
                
                # Check if task should be skipped
                if await self._should_skip_task(task, execution):
                    task.status = TaskStatus.SKIPPED
                    continue
                
                # Execute task
                await self._execute_task(task, execution)
                
                # Check if task failed
                if task.status == TaskStatus.FAILED:
                    execution.status = WorkflowStatus.FAILED
                    execution.error_message = task.error_message
                    break
            
            # Mark execution as completed
            if execution.status == WorkflowStatus.ACTIVE:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Workflow execution {execution.execution_id} completed with status {execution.status.value}")
            
        except Exception as e:
            logger.error(f"Error executing workflow tasks: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
    
    async def _should_skip_task(self, task: WorkflowTask, execution: WorkflowExecution) -> bool:
        """Check if task should be skipped based on conditions"""
        try:
            # Check if all dependencies are completed
            for dependency_id in task.dependencies:
                dependency_task = next(
                    (t for t in execution.tasks if t.task_id == dependency_id), 
                    None
                )
                if not dependency_task or dependency_task.status != TaskStatus.COMPLETED:
                    return True
            
            # Check conditional logic
            if task.task_type == TaskType.CONDITIONAL_BRANCH:
                condition = task.parameters.get("condition", "")
                if condition and not await self._evaluate_condition(condition, execution):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if task should be skipped: {e}")
            return False
    
    async def _evaluate_condition(self, condition: str, execution: WorkflowExecution) -> bool:
        """Evaluate conditional logic"""
        try:
            # Simple condition evaluation
            # This is a simplified version - in production, use a proper expression evaluator
            
            # Replace variables with values
            for key, value in execution.variables.items():
                condition = condition.replace(f"{{{{{key}}}}}", str(value))
            
            # Evaluate simple conditions
            if "performance_score < 0.7" in condition:
                return True  # Simplified
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution):
        """Execute a single workflow task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            # Add task to execution
            execution.tasks.append(task)
            
            # Execute based on task type
            if task.task_type == TaskType.CONTENT_GENERATION:
                result = await self._execute_content_generation(task, execution)
            elif task.task_type == TaskType.CONTENT_ANALYSIS:
                result = await self._execute_content_analysis(task, execution)
            elif task.task_type == TaskType.CONTENT_OPTIMIZATION:
                result = await self._execute_content_optimization(task, execution)
            elif task.task_type == TaskType.CONTENT_APPROVAL:
                result = await self._execute_content_approval(task, execution)
            elif task.task_type == TaskType.CONTENT_PUBLISHING:
                result = await self._execute_content_publishing(task, execution)
            elif task.task_type == TaskType.EMAIL_NOTIFICATION:
                result = await self._execute_email_notification(task, execution)
            elif task.task_type == TaskType.WEBHOOK_TRIGGER:
                result = await self._execute_webhook_trigger(task, execution)
            elif task.task_type == TaskType.CUSTOM_SCRIPT:
                result = await self._execute_custom_script(task, execution)
            else:
                result = {"status": "completed", "message": "Task executed successfully"}
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
    
    async def _execute_content_generation(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute content generation task"""
        try:
            # This would integrate with the Content Generation Engine
            # For now, return a mock result
            return {
                "content_id": f"content_{int(datetime.utcnow().timestamp())}",
                "title": "Generated Content",
                "content": "This is generated content based on the workflow parameters.",
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error executing content generation: {e}")
            raise
    
    async def _execute_content_analysis(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute content analysis task"""
        try:
            # This would integrate with the Content Intelligence Engine
            # For now, return a mock result
            return {
                "analysis_id": f"analysis_{int(datetime.utcnow().timestamp())}",
                "metrics": {
                    "readability_score": 75.0,
                    "seo_score": 0.8,
                    "engagement_score": 0.7
                },
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error executing content analysis: {e}")
            raise
    
    async def _execute_content_optimization(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute content optimization task"""
        try:
            # This would integrate with the Content Generation Engine
            # For now, return a mock result
            return {
                "optimization_id": f"optimization_{int(datetime.utcnow().timestamp())}",
                "optimizations_applied": ["seo", "engagement", "readability"],
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error executing content optimization: {e}")
            raise
    
    async def _execute_content_approval(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute content approval task"""
        try:
            # This would integrate with an approval system
            # For now, return a mock result
            return {
                "approval_id": f"approval_{int(datetime.utcnow().timestamp())}",
                "status": "approved",
                "approved_by": "system",
                "approved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing content approval: {e}")
            raise
    
    async def _execute_content_publishing(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute content publishing task"""
        try:
            # This would integrate with publishing systems
            # For now, return a mock result
            return {
                "publishing_id": f"publishing_{int(datetime.utcnow().timestamp())}",
                "published_channels": ["website", "social_media"],
                "status": "published",
                "published_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing content publishing: {e}")
            raise
    
    async def _execute_email_notification(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute email notification task"""
        try:
            # This would send actual emails
            # For now, return a mock result
            return {
                "notification_id": f"notification_{int(datetime.utcnow().timestamp())}",
                "recipients": task.parameters.get("recipients", []),
                "status": "sent",
                "sent_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing email notification: {e}")
            raise
    
    async def _execute_webhook_trigger(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute webhook trigger task"""
        try:
            # This would trigger actual webhooks
            # For now, return a mock result
            return {
                "webhook_id": f"webhook_{int(datetime.utcnow().timestamp())}",
                "url": task.parameters.get("url", ""),
                "status": "triggered",
                "triggered_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing webhook trigger: {e}")
            raise
    
    async def _execute_custom_script(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute custom script task"""
        try:
            # This would execute custom scripts
            # For now, return a mock result
            return {
                "script_id": f"script_{int(datetime.utcnow().timestamp())}",
                "script_name": task.parameters.get("script_name", ""),
                "status": "completed",
                "executed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing custom script: {e}")
            raise
    
    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        try:
            if execution_id not in self.executions:
                raise ValueError(f"Execution {execution_id} not found")
            
            execution = self.executions[execution_id]
            
            return {
                "execution_id": execution_id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "execution_time": execution.execution_time,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "status": task.status.value,
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "error_message": task.error_message
                    }
                    for task in execution.tasks
                ],
                "error_message": execution.error_message
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            raise
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow"""
        try:
            if execution_id not in self.executions:
                raise ValueError(f"Execution {execution_id} not found")
            
            execution = self.executions[execution_id]
            
            if execution.status == WorkflowStatus.ACTIVE:
                execution.status = WorkflowStatus.PAUSED
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error pausing workflow: {e}")
            return False
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow"""
        try:
            if execution_id not in self.executions:
                raise ValueError(f"Execution {execution_id} not found")
            
            execution = self.executions[execution_id]
            
            if execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.ACTIVE
                # Resume execution
                asyncio.create_task(self._execute_workflow_tasks(execution))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resuming workflow: {e}")
            return False
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow"""
        try:
            if execution_id not in self.executions:
                raise ValueError(f"Execution {execution_id} not found")
            
            execution = self.executions[execution_id]
            
            if execution.status in [WorkflowStatus.ACTIVE, WorkflowStatus.PAUSED]:
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling workflow: {e}")
            return False
    
    async def get_workflow_analytics(self, workflow_id: str, 
                                   time_period: str = "30d") -> Dict[str, Any]:
        """Get workflow analytics and metrics"""
        try:
            # Get executions for the workflow
            executions = [
                exec for exec in self.executions.values() 
                if exec.workflow_id == workflow_id
            ]
            
            if not executions:
                return {"error": "No executions found for this workflow"}
            
            # Calculate analytics
            total_executions = len(executions)
            successful_executions = len([e for e in executions if e.status == WorkflowStatus.COMPLETED])
            failed_executions = len([e for e in executions if e.status == WorkflowStatus.FAILED])
            
            success_rate = successful_executions / total_executions if total_executions > 0 else 0
            
            # Calculate average execution time
            completed_executions = [e for e in executions if e.execution_time > 0]
            avg_execution_time = (
                sum(e.execution_time for e in completed_executions) / len(completed_executions)
                if completed_executions else 0
            )
            
            # Task performance
            task_performance = defaultdict(list)
            for execution in executions:
                for task in execution.tasks:
                    if task.completed_at and task.started_at:
                        task_time = (task.completed_at - task.started_at).total_seconds()
                        task_performance[task.task_id].append({
                            "execution_time": task_time,
                            "status": task.status.value
                        })
            
            # Calculate task averages
            task_averages = {}
            for task_id, performances in task_performance.items():
                successful_tasks = [p for p in performances if p["status"] == "completed"]
                if successful_tasks:
                    task_averages[task_id] = {
                        "avg_execution_time": sum(p["execution_time"] for p in successful_tasks) / len(successful_tasks),
                        "success_rate": len(successful_tasks) / len(performances)
                    }
            
            return {
                "workflow_id": workflow_id,
                "time_period": time_period,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time,
                "task_performance": task_averages,
                "execution_trends": {
                    "daily_executions": [],  # Would be calculated from actual data
                    "success_rate_trend": []  # Would be calculated from actual data
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow analytics: {e}")
            return {"error": str(e)}
    
    async def export_workflow_definition(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow definition"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            return {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "status": workflow.status.value,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "task_type": task.task_type.value,
                        "description": task.description,
                        "parameters": task.parameters,
                        "dependencies": task.dependencies,
                        "timeout": task.timeout,
                        "retry_count": task.retry_count
                    }
                    for task in workflow.tasks
                ],
                "triggers": [
                    {
                        "trigger_id": trigger.trigger_id,
                        "name": trigger.name,
                        "trigger_type": trigger.trigger_type.value,
                        "parameters": trigger.parameters,
                        "is_active": trigger.is_active
                    }
                    for trigger in workflow.triggers
                ],
                "variables": workflow.variables,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting workflow definition: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of the Content Workflow Engine"""
    try:
        # Initialize engine
        config = {
            "max_workers": 10,
            "scheduler_enabled": True,
            "email_config": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your-email@gmail.com",
                "password": "your-password"
            }
        }
        
        engine = ContentWorkflowEngine(config)
        
        # Create a workflow from template
        print("Creating workflow from template...")
        workflow = await engine.create_workflow(
            template_id="content_creation",
            name="My Content Creation Workflow",
            description="Custom content creation workflow",
            variables={
                "topic": "Artificial Intelligence",
                "format": "article",
                "tone": "professional",
                "length": "medium",
                "approvers": ["admin@example.com"],
                "publish_channels": ["website", "social_media"]
            }
        )
        
        print(f"Workflow created: {workflow.workflow_id}")
        print(f"Tasks: {len(workflow.tasks)}")
        print(f"Triggers: {len(workflow.triggers)}")
        
        # Execute workflow
        print("\nExecuting workflow...")
        execution = await engine.execute_workflow(
            workflow_id=workflow.workflow_id,
            variables={
                "topic": "Machine Learning in Healthcare",
                "format": "blog_post",
                "tone": "conversational"
            }
        )
        
        print(f"Execution started: {execution.execution_id}")
        
        # Wait for execution to complete
        await asyncio.sleep(2)
        
        # Get execution status
        print("\nGetting execution status...")
        status = await engine.get_workflow_status(execution.execution_id)
        print(f"Status: {status['status']}")
        print(f"Tasks completed: {len([t for t in status['tasks'] if t['status'] == 'completed'])}")
        
        # Get workflow analytics
        print("\nGetting workflow analytics...")
        analytics = await engine.get_workflow_analytics(workflow.workflow_id)
        print(f"Total executions: {analytics['total_executions']}")
        print(f"Success rate: {analytics['success_rate']:.2%}")
        print(f"Average execution time: {analytics['average_execution_time']:.2f} seconds")
        
        # Export workflow definition
        print("\nExporting workflow definition...")
        definition = await engine.export_workflow_definition(workflow.workflow_id)
        print(f"Workflow definition exported: {len(definition['tasks'])} tasks")
        
        print("\nContent Workflow Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























