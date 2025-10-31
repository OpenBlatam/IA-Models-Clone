"""
AI Workflow Automation
=====================

Advanced AI-powered workflow automation and intelligent process management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import json
import networkx as nx
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


class TriggerType(str, Enum):
    """Trigger type."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    AI_DECISION = "ai_decision"


class ActionType(str, Enum):
    """Action type."""
    DOCUMENT_CREATE = "document_create"
    DOCUMENT_UPDATE = "document_update"
    DOCUMENT_APPROVE = "document_approve"
    DOCUMENT_REJECT = "document_reject"
    NOTIFICATION_SEND = "notification_send"
    INTEGRATION_SYNC = "integration_sync"
    AI_ANALYSIS = "ai_analysis"
    CUSTOM_SCRIPT = "custom_script"
    CONDITION_CHECK = "condition_check"
    PARALLEL_EXECUTION = "parallel_execution"


@dataclass
class WorkflowTrigger:
    """Workflow trigger configuration."""
    trigger_id: str
    trigger_type: TriggerType
    name: str
    description: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression for scheduled triggers
    event_filters: Dict[str, Any] = field(default_factory=dict)
    ai_model: Optional[str] = None  # AI model for decision-based triggers


@dataclass
class WorkflowAction:
    """Workflow action definition."""
    action_id: str
    action_type: ActionType
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300  # seconds
    dependencies: List[str] = field(default_factory=list)
    ai_prompt: Optional[str] = None  # AI prompt for AI-based actions


@dataclass
class WorkflowTask:
    """Workflow task instance."""
    task_id: str
    workflow_id: str
    action: WorkflowAction
    status: TaskStatus
    assigned_to: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowInstance:
    """Workflow execution instance."""
    instance_id: str
    workflow_id: str
    status: WorkflowStatus
    context: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    tasks: List[WorkflowTask] = field(default_factory=list)
    current_task: Optional[str] = None
    execution_graph: Optional[nx.DiGraph] = None


@dataclass
class WorkflowDefinition:
    """Workflow definition."""
    workflow_id: str
    name: str
    description: str
    version: str
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    actions: List[WorkflowAction] = field(default_factory=list)
    execution_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    variables: Dict[str, Any] = field(default_factory=dict)
    ai_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


class AIWorkflowEngine:
    """AI-powered workflow execution engine."""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.instances: Dict[str, WorkflowInstance] = {}
        self.execution_queue: deque = deque()
        self.ai_models: Dict[str, Any] = {}
        self.event_listeners: Dict[str, List[str]] = defaultdict(list)
        self.scheduled_triggers: Dict[str, Dict[str, Any]] = {}
        self._initialize_ai_models()
    
    def _initialize_ai_models(self):
        """Initialize AI models for workflow automation."""
        
        # Mock AI models - in production, integrate with actual AI services
        self.ai_models = {
            "decision_maker": {
                "name": "Workflow Decision Maker",
                "description": "Makes intelligent decisions in workflow execution",
                "capabilities": ["condition_evaluation", "path_selection", "optimization"]
            },
            "content_analyzer": {
                "name": "Content Analysis AI",
                "description": "Analyzes document content for workflow decisions",
                "capabilities": ["sentiment_analysis", "topic_detection", "quality_assessment"]
            },
            "approval_predictor": {
                "name": "Approval Prediction AI",
                "description": "Predicts approval likelihood for documents",
                "capabilities": ["approval_prediction", "risk_assessment", "recommendation"]
            }
        }
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        triggers: List[WorkflowTrigger] = None,
        actions: List[WorkflowAction] = None,
        ai_config: Dict[str, Any] = None
    ) -> WorkflowDefinition:
        """Create a new workflow definition."""
        
        workflow_id = str(uuid4())
        
        # Create execution graph
        execution_graph = nx.DiGraph()
        
        # Add actions as nodes
        if actions:
            for action in actions:
                execution_graph.add_node(action.action_id, action=action)
        
        # Add dependencies as edges
        if actions:
            for action in actions:
                for dep in action.dependencies:
                    if dep in [a.action_id for a in actions]:
                        execution_graph.add_edge(dep, action.action_id)
        
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            version="1.0",
            triggers=triggers or [],
            actions=actions or [],
            execution_graph=execution_graph,
            ai_config=ai_config or {}
        )
        
        self.workflows[workflow_id] = workflow
        
        # Register event listeners for event-based triggers
        for trigger in workflow.triggers:
            if trigger.trigger_type == TriggerType.EVENT_BASED:
                for event_type in trigger.event_filters.get("event_types", []):
                    self.event_listeners[event_type].append(workflow_id)
        
        logger.info(f"Created workflow: {name} ({workflow_id})")
        
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Dict[str, Any] = None,
        variables: Dict[str, Any] = None
    ) -> WorkflowInstance:
        """Execute a workflow."""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        # Create workflow instance
        instance = WorkflowInstance(
            instance_id=str(uuid4()),
            workflow_id=workflow_id,
            status=WorkflowStatus.ACTIVE,
            context=context or {},
            variables=variables or {},
            execution_graph=workflow.execution_graph.copy()
        )
        
        # Create tasks from actions
        for action in workflow.actions:
            task = WorkflowTask(
                task_id=str(uuid4()),
                workflow_id=workflow_id,
                action=action,
                status=TaskStatus.PENDING,
                max_retries=action.retry_config.get("max_retries", 3)
            )
            instance.tasks.append(task)
        
        self.instances[instance.instance_id] = instance
        
        # Start execution
        asyncio.create_task(self._execute_workflow_instance(instance))
        
        logger.info(f"Started workflow execution: {workflow.name} ({instance.instance_id})")
        
        return instance
    
    async def _execute_workflow_instance(self, instance: WorkflowInstance):
        """Execute a workflow instance."""
        
        try:
            # Get topological order of tasks
            try:
                task_order = list(nx.topological_sort(instance.execution_graph))
            except nx.NetworkXError:
                # Handle cycles or invalid graph
                task_order = [task.task_id for task in instance.tasks]
            
            # Execute tasks in order
            for task_id in task_order:
                task = next((t for t in instance.tasks if t.task_id == task_id), None)
                if not task:
                    continue
                
                # Check if task dependencies are met
                if not await self._check_task_dependencies(task, instance):
                    continue
                
                # Execute task
                await self._execute_task(task, instance)
                
                # Check if workflow should continue
                if instance.status in [WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                    break
            
            # Mark workflow as completed if all tasks are done
            if all(task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] for task in instance.tasks):
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.completed_at = datetime.now()
            logger.error(f"Workflow execution failed: {str(e)}")
    
    async def _check_task_dependencies(self, task: WorkflowTask, instance: WorkflowInstance) -> bool:
        """Check if task dependencies are met."""
        
        for dep_id in task.action.dependencies:
            dep_task = next((t for t in instance.tasks if t.action.action_id == dep_id), None)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _execute_task(self, task: WorkflowTask, instance: WorkflowInstance):
        """Execute a single task."""
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Execute based on action type
            if task.action.action_type == ActionType.DOCUMENT_CREATE:
                result = await self._execute_document_create(task, instance)
            elif task.action.action_type == ActionType.DOCUMENT_UPDATE:
                result = await self._execute_document_update(task, instance)
            elif task.action.action_type == ActionType.DOCUMENT_APPROVE:
                result = await self._execute_document_approve(task, instance)
            elif task.action.action_type == ActionType.DOCUMENT_REJECT:
                result = await self._execute_document_reject(task, instance)
            elif task.action.action_type == ActionType.NOTIFICATION_SEND:
                result = await self._execute_notification_send(task, instance)
            elif task.action.action_type == ActionType.INTEGRATION_SYNC:
                result = await self._execute_integration_sync(task, instance)
            elif task.action.action_type == ActionType.AI_ANALYSIS:
                result = await self._execute_ai_analysis(task, instance)
            elif task.action.action_type == ActionType.CUSTOM_SCRIPT:
                result = await self._execute_custom_script(task, instance)
            elif task.action.action_type == ActionType.CONDITION_CHECK:
                result = await self._execute_condition_check(task, instance)
            elif task.action.action_type == ActionType.PARALLEL_EXECUTION:
                result = await self._execute_parallel_execution(task, instance)
            else:
                raise ValueError(f"Unknown action type: {task.action.action_type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            # Retry if configured
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                await asyncio.sleep(task.action.retry_config.get("delay", 5))
                await self._execute_task(task, instance)
            else:
                instance.status = WorkflowStatus.FAILED
    
    async def _execute_document_create(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute document creation action."""
        
        params = task.action.parameters
        
        # Mock document creation - in production, integrate with document service
        document_data = {
            "title": params.get("title", "AI Generated Document"),
            "content": params.get("content", ""),
            "template": params.get("template", "default"),
            "metadata": params.get("metadata", {})
        }
        
        # Use AI to enhance content if configured
        if task.action.ai_prompt:
            enhanced_content = await self._ai_enhance_content(
                document_data["content"],
                task.action.ai_prompt,
                instance.variables
            )
            document_data["content"] = enhanced_content
        
        return {
            "action": "document_create",
            "document_id": str(uuid4()),
            "document_data": document_data,
            "created_at": datetime.now().isoformat()
        }
    
    async def _execute_document_update(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute document update action."""
        
        params = task.action.parameters
        document_id = params.get("document_id")
        
        if not document_id:
            raise ValueError("Document ID is required for document update")
        
        # Mock document update
        update_data = {
            "content": params.get("content"),
            "metadata": params.get("metadata", {}),
            "version_comment": params.get("version_comment", "AI Workflow Update")
        }
        
        return {
            "action": "document_update",
            "document_id": document_id,
            "update_data": update_data,
            "updated_at": datetime.now().isoformat()
        }
    
    async def _execute_document_approve(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute document approval action."""
        
        params = task.action.parameters
        document_id = params.get("document_id")
        
        if not document_id:
            raise ValueError("Document ID is required for document approval")
        
        # Use AI to predict approval likelihood
        approval_prediction = await self._ai_predict_approval(document_id, instance.variables)
        
        return {
            "action": "document_approve",
            "document_id": document_id,
            "approved": True,
            "approval_prediction": approval_prediction,
            "approved_at": datetime.now().isoformat(),
            "approved_by": params.get("approver", "AI Workflow")
        }
    
    async def _execute_document_reject(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute document rejection action."""
        
        params = task.action.parameters
        document_id = params.get("document_id")
        
        if not document_id:
            raise ValueError("Document ID is required for document rejection")
        
        return {
            "action": "document_reject",
            "document_id": document_id,
            "rejected": True,
            "rejection_reason": params.get("reason", "AI Workflow Rejection"),
            "rejected_at": datetime.now().isoformat(),
            "rejected_by": params.get("rejector", "AI Workflow")
        }
    
    async def _execute_notification_send(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute notification sending action."""
        
        params = task.action.parameters
        
        # Use AI to personalize notification content
        personalized_content = await self._ai_personalize_notification(
            params.get("template", ""),
            params.get("recipients", []),
            instance.variables
        )
        
        return {
            "action": "notification_send",
            "recipients": params.get("recipients", []),
            "subject": params.get("subject", "Workflow Notification"),
            "content": personalized_content,
            "sent_at": datetime.now().isoformat()
        }
    
    async def _execute_integration_sync(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute integration sync action."""
        
        params = task.action.parameters
        integration_id = params.get("integration_id")
        
        if not integration_id:
            raise ValueError("Integration ID is required for integration sync")
        
        # Mock integration sync
        return {
            "action": "integration_sync",
            "integration_id": integration_id,
            "data_synced": params.get("data", []),
            "sync_result": "success",
            "synced_at": datetime.now().isoformat()
        }
    
    async def _execute_ai_analysis(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute AI analysis action."""
        
        params = task.action.parameters
        
        # Perform AI analysis based on parameters
        analysis_type = params.get("analysis_type", "general")
        content = params.get("content", "")
        
        if analysis_type == "sentiment":
            result = await self._ai_analyze_sentiment(content)
        elif analysis_type == "quality":
            result = await self._ai_analyze_quality(content)
        elif analysis_type == "compliance":
            result = await self._ai_analyze_compliance(content)
        else:
            result = await self._ai_analyze_general(content)
        
        return {
            "action": "ai_analysis",
            "analysis_type": analysis_type,
            "result": result,
            "analyzed_at": datetime.now().isoformat()
        }
    
    async def _execute_custom_script(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute custom script action."""
        
        params = task.action.parameters
        script = params.get("script", "")
        
        # Execute custom script with context
        # In production, use a secure sandbox for script execution
        try:
            # Mock script execution
            result = {"script_output": "Custom script executed successfully"}
        except Exception as e:
            raise ValueError(f"Script execution failed: {str(e)}")
        
        return {
            "action": "custom_script",
            "script": script,
            "result": result,
            "executed_at": datetime.now().isoformat()
        }
    
    async def _execute_condition_check(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute condition check action."""
        
        params = task.action.parameters
        conditions = params.get("conditions", [])
        
        # Use AI to evaluate complex conditions
        evaluation_result = await self._ai_evaluate_conditions(conditions, instance.variables)
        
        return {
            "action": "condition_check",
            "conditions": conditions,
            "result": evaluation_result,
            "evaluated_at": datetime.now().isoformat()
        }
    
    async def _execute_parallel_execution(self, task: WorkflowTask, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute parallel execution action."""
        
        params = task.action.parameters
        parallel_actions = params.get("actions", [])
        
        # Execute actions in parallel
        results = []
        tasks = []
        
        for action_config in parallel_actions:
            # Create temporary task for parallel execution
            temp_task = WorkflowTask(
                task_id=str(uuid4()),
                workflow_id=instance.workflow_id,
                action=WorkflowAction(
                    action_id=str(uuid4()),
                    action_type=ActionType(action_config["type"]),
                    name=action_config.get("name", "Parallel Action"),
                    description=action_config.get("description", ""),
                    parameters=action_config.get("parameters", {})
                ),
                status=TaskStatus.PENDING
            )
            tasks.append(self._execute_task(temp_task, instance))
        
        # Wait for all parallel tasks to complete
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(parallel_results):
            if isinstance(result, Exception):
                results.append({"error": str(result)})
            else:
                results.append(result)
        
        return {
            "action": "parallel_execution",
            "parallel_results": results,
            "executed_at": datetime.now().isoformat()
        }
    
    # AI Helper Methods
    async def _ai_enhance_content(self, content: str, prompt: str, variables: Dict[str, Any]) -> str:
        """Enhance content using AI."""
        
        # Mock AI enhancement - in production, integrate with AI service
        enhanced_content = f"[AI Enhanced] {content}"
        
        # Apply variable substitution
        for key, value in variables.items():
            enhanced_content = enhanced_content.replace(f"{{{key}}}", str(value))
        
        return enhanced_content
    
    async def _ai_predict_approval(self, document_id: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Predict document approval likelihood using AI."""
        
        # Mock AI prediction - in production, use trained model
        return {
            "approval_likelihood": 0.85,
            "confidence": 0.92,
            "factors": [
                "Content quality: High",
                "Compliance: Good",
                "Format: Professional"
            ],
            "recommendations": [
                "Document is ready for approval",
                "Consider minor formatting improvements"
            ]
        }
    
    async def _ai_personalize_notification(self, template: str, recipients: List[str], variables: Dict[str, Any]) -> str:
        """Personalize notification content using AI."""
        
        # Mock AI personalization
        personalized_content = template
        
        # Apply variable substitution
        for key, value in variables.items():
            personalized_content = personalized_content.replace(f"{{{key}}}", str(value))
        
        return personalized_content
    
    async def _ai_analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze content sentiment using AI."""
        
        # Mock sentiment analysis
        return {
            "sentiment": "positive",
            "confidence": 0.78,
            "emotions": ["satisfaction", "confidence"],
            "recommendations": ["Content has positive tone"]
        }
    
    async def _ai_analyze_quality(self, content: str) -> Dict[str, Any]:
        """Analyze content quality using AI."""
        
        # Mock quality analysis
        return {
            "quality_score": 8.5,
            "readability": "good",
            "structure": "well_organized",
            "recommendations": ["Consider adding more examples"]
        }
    
    async def _ai_analyze_compliance(self, content: str) -> Dict[str, Any]:
        """Analyze content compliance using AI."""
        
        # Mock compliance analysis
        return {
            "compliance_score": 9.2,
            "violations": [],
            "recommendations": ["Content is compliant"]
        }
    
    async def _ai_analyze_general(self, content: str) -> Dict[str, Any]:
        """Perform general AI analysis."""
        
        # Mock general analysis
        return {
            "analysis_type": "general",
            "key_topics": ["document", "management", "workflow"],
            "summary": "Document contains relevant content for workflow processing",
            "recommendations": ["Continue with workflow execution"]
        }
    
    async def _ai_evaluate_conditions(self, conditions: List[Dict[str, Any]], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate conditions using AI."""
        
        # Mock condition evaluation
        results = []
        for condition in conditions:
            # Simple condition evaluation
            if condition.get("type") == "equals":
                result = variables.get(condition.get("variable")) == condition.get("value")
            elif condition.get("type") == "greater_than":
                result = variables.get(condition.get("variable"), 0) > condition.get("value", 0)
            else:
                result = True
            
            results.append({
                "condition": condition,
                "result": result
            })
        
        return {
            "conditions": results,
            "overall_result": all(r["result"] for r in results)
        }
    
    # Workflow Management Methods
    async def get_workflow_status(self, instance_id: str) -> Dict[str, Any]:
        """Get workflow instance status."""
        
        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance {instance_id} not found")
        
        instance = self.instances[instance_id]
        workflow = self.workflows[instance.workflow_id]
        
        return {
            "instance_id": instance_id,
            "workflow_name": workflow.name,
            "status": instance.status.value,
            "progress": self._calculate_workflow_progress(instance),
            "started_at": instance.started_at.isoformat(),
            "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "action_name": task.action.name,
                    "status": task.status.value,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "error_message": task.error_message
                }
                for task in instance.tasks
            ]
        }
    
    def _calculate_workflow_progress(self, instance: WorkflowInstance) -> float:
        """Calculate workflow progress percentage."""
        
        if not instance.tasks:
            return 0.0
        
        completed_tasks = len([t for t in instance.tasks if t.status == TaskStatus.COMPLETED])
        total_tasks = len(instance.tasks)
        
        return (completed_tasks / total_tasks) * 100
    
    async def pause_workflow(self, instance_id: str) -> bool:
        """Pause workflow execution."""
        
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        if instance.status == WorkflowStatus.ACTIVE:
            instance.status = WorkflowStatus.PAUSED
            return True
        
        return False
    
    async def resume_workflow(self, instance_id: str) -> bool:
        """Resume paused workflow."""
        
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        if instance.status == WorkflowStatus.PAUSED:
            instance.status = WorkflowStatus.ACTIVE
            asyncio.create_task(self._execute_workflow_instance(instance))
            return True
        
        return False
    
    async def cancel_workflow(self, instance_id: str) -> bool:
        """Cancel workflow execution."""
        
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        if instance.status in [WorkflowStatus.ACTIVE, WorkflowStatus.PAUSED]:
            instance.status = WorkflowStatus.CANCELLED
            instance.completed_at = datetime.now()
            return True
        
        return False
    
    async def get_workflow_analytics(self) -> Dict[str, Any]:
        """Get workflow analytics."""
        
        total_workflows = len(self.workflows)
        total_instances = len(self.instances)
        
        # Calculate status distribution
        status_counts = defaultdict(int)
        for instance in self.instances.values():
            status_counts[instance.status.value] += 1
        
        # Calculate average execution time
        completed_instances = [i for i in self.instances.values() if i.completed_at]
        avg_execution_time = 0
        if completed_instances:
            total_time = sum(
                (i.completed_at - i.started_at).total_seconds()
                for i in completed_instances
            )
            avg_execution_time = total_time / len(completed_instances)
        
        return {
            "total_workflows": total_workflows,
            "total_instances": total_instances,
            "status_distribution": dict(status_counts),
            "average_execution_time": avg_execution_time,
            "success_rate": (status_counts["completed"] / total_instances * 100) if total_instances > 0 else 0,
            "most_used_actions": self._get_most_used_actions(),
            "ai_usage_stats": self._get_ai_usage_stats()
        }
    
    def _get_most_used_actions(self) -> List[Dict[str, Any]]:
        """Get most used action types."""
        
        action_counts = defaultdict(int)
        for workflow in self.workflows.values():
            for action in workflow.actions:
                action_counts[action.action_type.value] += 1
        
        return [
            {"action_type": action_type, "count": count}
            for action_type, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _get_ai_usage_stats(self) -> Dict[str, Any]:
        """Get AI usage statistics."""
        
        ai_actions = 0
        ai_triggers = 0
        
        for workflow in self.workflows.values():
            # Count AI actions
            ai_actions += len([a for a in workflow.actions if a.ai_prompt])
            
            # Count AI triggers
            ai_triggers += len([t for t in workflow.triggers if t.trigger_type == TriggerType.AI_DECISION])
        
        return {
            "ai_actions": ai_actions,
            "ai_triggers": ai_triggers,
            "ai_models_available": len(self.ai_models)
        }