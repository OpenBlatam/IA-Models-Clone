"""
AI Integration System - Workflow Engine
Advanced workflow automation and orchestration for content integration
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from .integration_engine import IntegrationRequest, IntegrationResult, ContentType, IntegrationStatus
from .models import IntegrationLog

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class StepType(Enum):
    """Step types in workflow"""
    INTEGRATION = "integration"
    CONDITION = "condition"
    TRANSFORM = "transform"
    DELAY = "delay"
    NOTIFICATION = "notification"
    WEBHOOK = "webhook"
    SCRIPT = "script"

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    id: str
    name: str
    step_type: StepType
    config: Dict[str, Any]
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = None
    triggers: List[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}
        if self.triggers is None:
            self.triggers = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    id: str
    workflow_id: str
    status: WorkflowStatus
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = None
    variables: Dict[str, Any] = None
    current_step: Optional[str] = None
    started_at: datetime = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.output_data is None:
            self.output_data = {}
        if self.variables is None:
            self.variables = {}
        if self.started_at is None:
            self.started_at = datetime.utcnow()

class WorkflowEngine:
    """Advanced workflow engine for content integration"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.step_handlers: Dict[StepType, Callable] = {}
        self.is_running = False
        
        # Register default step handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default step handlers"""
        self.step_handlers[StepType.INTEGRATION] = self._handle_integration_step
        self.step_handlers[StepType.CONDITION] = self._handle_condition_step
        self.step_handlers[StepType.TRANSFORM] = self._handle_transform_step
        self.step_handlers[StepType.DELAY] = self._handle_delay_step
        self.step_handlers[StepType.NOTIFICATION] = self._handle_notification_step
        self.step_handlers[StepType.WEBHOOK] = self._handle_webhook_step
        self.step_handlers[StepType.SCRIPT] = self._handle_script_step
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition"""
        self.workflows[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.name} (v{workflow.version})")
    
    def register_step_handler(self, step_type: StepType, handler: Callable):
        """Register a custom step handler"""
        self.step_handlers[step_type] = handler
        logger.info(f"Registered handler for step type: {step_type.value}")
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> str:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        # Create execution instance
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            input_data=input_data,
            variables=workflow.variables.copy()
        )
        
        self.executions[execution_id] = execution
        
        # Start execution
        asyncio.create_task(self._execute_workflow_async(execution))
        
        logger.info(f"Started workflow execution: {execution_id}")
        return execution_id
    
    async def _execute_workflow_async(self, execution: WorkflowExecution):
        """Execute workflow asynchronously"""
        try:
            execution.status = WorkflowStatus.RUNNING
            workflow = self.workflows[execution.workflow_id]
            
            # Execute steps in dependency order
            completed_steps = set()
            failed_steps = set()
            
            while len(completed_steps) + len(failed_steps) < len(workflow.steps):
                # Find steps that can be executed
                ready_steps = []
                for step in workflow.steps:
                    if (step.id not in completed_steps and 
                        step.id not in failed_steps and
                        all(dep in completed_steps for dep in step.dependencies)):
                        ready_steps.append(step)
                
                if not ready_steps:
                    # No ready steps, check if we're stuck
                    remaining_steps = [s for s in workflow.steps 
                                     if s.id not in completed_steps and s.id not in failed_steps]
                    if remaining_steps:
                        execution.status = WorkflowStatus.FAILED
                        execution.error_message = "Workflow stuck - no ready steps"
                        break
                    else:
                        break
                
                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    task = asyncio.create_task(self._execute_step(step, execution))
                    tasks.append((step, task))
                
                # Wait for all tasks to complete
                for step, task in tasks:
                    try:
                        result = await task
                        if result:
                            completed_steps.add(step.id)
                        else:
                            failed_steps.add(step.id)
                    except Exception as e:
                        logger.error(f"Step execution failed: {step.id} - {str(e)}")
                        failed_steps.add(step.id)
                
                # Check if workflow should continue
                if failed_steps:
                    # Check if any failed step is critical
                    critical_failed = any(
                        step.id in failed_steps and step.config.get("critical", False)
                        for step in workflow.steps
                    )
                    if critical_failed:
                        execution.status = WorkflowStatus.FAILED
                        execution.error_message = "Critical step failed"
                        break
            
            # Set final status
            if execution.status == WorkflowStatus.RUNNING:
                if failed_steps:
                    execution.status = WorkflowStatus.FAILED
                    execution.error_message = f"Some steps failed: {list(failed_steps)}"
                else:
                    execution.status = WorkflowStatus.COMPLETED
            
            execution.completed_at = datetime.utcnow()
            logger.info(f"Workflow execution completed: {execution.id} - {execution.status.value}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(f"Workflow execution failed: {execution.id} - {str(e)}")
    
    async def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution) -> bool:
        """Execute a single workflow step"""
        try:
            step.status = StepStatus.RUNNING
            step.started_at = datetime.utcnow()
            execution.current_step = step.id
            
            logger.info(f"Executing step: {step.name} ({step.step_type.value})")
            
            # Get step handler
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"No handler for step type: {step.step_type.value}")
            
            # Execute step with timeout
            result = await asyncio.wait_for(
                handler(step, execution),
                timeout=step.timeout
            )
            
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.utcnow()
            step.result = result
            
            logger.info(f"Step completed: {step.name}")
            return True
            
        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error_message = f"Step timed out after {step.timeout} seconds"
            step.completed_at = datetime.utcnow()
            logger.error(f"Step timed out: {step.name}")
            return False
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.completed_at = datetime.utcnow()
            logger.error(f"Step failed: {step.name} - {str(e)}")
            
            # Retry if possible
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                logger.info(f"Retrying step: {step.name} (attempt {step.retry_count})")
                return await self._execute_step(step, execution)
            
            return False
    
    async def _handle_integration_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle integration step"""
        from .integration_engine import integration_engine
        
        config = step.config
        content_data = self._resolve_variables(config.get("content_data", {}), execution)
        
        # Create integration request
        request = IntegrationRequest(
            content_id=f"{execution.id}_{step.id}",
            content_type=ContentType(config.get("content_type", "blog_post")),
            content_data=content_data,
            target_platforms=config.get("target_platforms", []),
            priority=config.get("priority", 1),
            max_retries=config.get("max_retries", 3),
            metadata={
                "workflow_execution_id": execution.id,
                "workflow_step_id": step.id
            }
        )
        
        # Execute integration
        await integration_engine.process_single_request(request)
        
        # Get results
        status = await integration_engine.get_integration_status(request.content_id)
        
        return {
            "integration_request_id": request.content_id,
            "status": status,
            "platforms": config.get("target_platforms", [])
        }
    
    async def _handle_condition_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle condition step"""
        config = step.config
        condition = config.get("condition")
        
        # Evaluate condition
        result = self._evaluate_condition(condition, execution)
        
        return {
            "condition": condition,
            "result": result,
            "next_step": config.get("next_step_if_true" if result else "next_step_if_false")
        }
    
    async def _handle_transform_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle data transformation step"""
        config = step.config
        transformation = config.get("transformation")
        
        # Apply transformation
        transformed_data = self._apply_transformation(transformation, execution)
        
        # Update execution variables
        execution.variables.update(transformed_data)
        
        return {
            "transformation": transformation,
            "transformed_data": transformed_data
        }
    
    async def _handle_delay_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle delay step"""
        config = step.config
        delay_seconds = config.get("delay_seconds", 0)
        
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        
        return {
            "delay_seconds": delay_seconds,
            "delayed_until": datetime.utcnow().isoformat()
        }
    
    async def _handle_notification_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle notification step"""
        config = step.config
        notification_type = config.get("type", "email")
        message = self._resolve_variables(config.get("message", ""), execution)
        
        # Send notification (implement based on your notification system)
        notification_result = await self._send_notification(notification_type, message, config)
        
        return {
            "notification_type": notification_type,
            "message": message,
            "result": notification_result
        }
    
    async def _handle_webhook_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle webhook step"""
        config = step.config
        url = config.get("url")
        method = config.get("method", "POST")
        payload = self._resolve_variables(config.get("payload", {}), execution)
        
        # Send webhook
        webhook_result = await self._send_webhook(url, method, payload, config)
        
        return {
            "url": url,
            "method": method,
            "payload": payload,
            "result": webhook_result
        }
    
    async def _handle_script_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Handle script execution step"""
        config = step.config
        script = config.get("script")
        
        # Execute script (implement safely)
        script_result = await self._execute_script(script, execution)
        
        return {
            "script": script,
            "result": script_result
        }
    
    def _resolve_variables(self, data: Any, execution: WorkflowExecution) -> Any:
        """Resolve variables in data"""
        if isinstance(data, str):
            # Replace variables in string
            for key, value in execution.variables.items():
                data = data.replace(f"${{{key}}}", str(value))
            return data
        elif isinstance(data, dict):
            return {k: self._resolve_variables(v, execution) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_variables(item, execution) for item in data]
        else:
            return data
    
    def _evaluate_condition(self, condition: str, execution: WorkflowExecution) -> bool:
        """Evaluate condition string"""
        try:
            # Simple condition evaluation (implement safely)
            # Replace variables in condition
            resolved_condition = self._resolve_variables(condition, execution)
            
            # Evaluate condition (implement safely)
            # This is a simplified version - implement proper expression evaluation
            return eval(resolved_condition)  # Use a safe expression evaluator in production
        except Exception as e:
            logger.error(f"Condition evaluation failed: {str(e)}")
            return False
    
    def _apply_transformation(self, transformation: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Apply data transformation"""
        try:
            transformation_type = transformation.get("type", "map")
            
            if transformation_type == "map":
                # Map transformation
                mapping = transformation.get("mapping", {})
                result = {}
                for target_key, source_key in mapping.items():
                    if source_key in execution.variables:
                        result[target_key] = execution.variables[source_key]
                return result
            
            elif transformation_type == "filter":
                # Filter transformation
                filter_condition = transformation.get("condition")
                filtered_data = {}
                for key, value in execution.variables.items():
                    if self._evaluate_condition(filter_condition.replace("value", str(value)), execution):
                        filtered_data[key] = value
                return filtered_data
            
            else:
                logger.warning(f"Unknown transformation type: {transformation_type}")
                return {}
                
        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            return {}
    
    async def _send_notification(self, notification_type: str, message: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification"""
        try:
            # Implement notification sending based on type
            if notification_type == "email":
                # Send email notification
                pass
            elif notification_type == "slack":
                # Send Slack notification
                pass
            elif notification_type == "webhook":
                # Send webhook notification
                pass
            
            return {"status": "sent", "type": notification_type}
        except Exception as e:
            logger.error(f"Notification sending failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def _send_webhook(self, url: str, method: str, payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook"""
        try:
            import aiohttp
            
            headers = config.get("headers", {})
            timeout = config.get("timeout", 30)
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "POST":
                    async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                        return {
                            "status": response.status,
                            "response": await response.text()
                        }
                elif method.upper() == "GET":
                    async with session.get(url, params=payload, headers=headers, timeout=timeout) as response:
                        return {
                            "status": response.status,
                            "response": await response.text()
                        }
                else:
                    return {"status": "error", "error": f"Unsupported method: {method}"}
                    
        except Exception as e:
            logger.error(f"Webhook sending failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _execute_script(self, script: str, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute script safely"""
        try:
            # Implement safe script execution
            # This is a placeholder - implement proper sandboxing in production
            exec_globals = {
                "execution": execution,
                "variables": execution.variables,
                "datetime": datetime,
                "json": json
            }
            
            exec(script, exec_globals)
            
            return {"status": "completed", "output": "Script executed successfully"}
        except Exception as e:
            logger.error(f"Script execution failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        workflow = self.workflows.get(execution.workflow_id)
        if not workflow:
            return None
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "workflow_name": workflow.name,
            "status": execution.status.value,
            "current_step": execution.current_step,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error_message": execution.error_message,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "type": step.step_type.value,
                    "status": step.status.value,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error_message": step.error_message,
                    "retry_count": step.retry_count
                }
                for step in workflow.steps
            ]
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows"""
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "steps_count": len(workflow.steps),
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            }
            for workflow in self.workflows.values()
        ]
    
    def list_executions(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List workflow executions"""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        return [
            {
                "id": execution.id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "error_message": execution.error_message
            }
            for execution in executions
        ]

# Global workflow engine instance
workflow_engine = WorkflowEngine()

# Predefined workflow templates
def create_content_distribution_workflow() -> WorkflowDefinition:
    """Create a content distribution workflow template"""
    steps = [
        WorkflowStep(
            id="validate_content",
            name="Validate Content",
            step_type=StepType.CONDITION,
            config={
                "condition": "len(content_data.get('content', '')) > 100",
                "next_step_if_true": "transform_content",
                "next_step_if_false": "end_workflow"
            }
        ),
        WorkflowStep(
            id="transform_content",
            name="Transform Content",
            step_type=StepType.TRANSFORM,
            config={
                "transformation": {
                    "type": "map",
                    "mapping": {
                        "title": "title",
                        "content": "content",
                        "author": "author",
                        "tags": "tags"
                    }
                }
            },
            dependencies=["validate_content"]
        ),
        WorkflowStep(
            id="integrate_wordpress",
            name="Integrate with WordPress",
            step_type=StepType.INTEGRATION,
            config={
                "content_type": "blog_post",
                "target_platforms": ["wordpress"],
                "content_data": {
                    "title": "${title}",
                    "content": "${content}",
                    "author": "${author}",
                    "tags": "${tags}"
                }
            },
            dependencies=["transform_content"]
        ),
        WorkflowStep(
            id="integrate_hubspot",
            name="Integrate with HubSpot",
            step_type=StepType.INTEGRATION,
            config={
                "content_type": "blog_post",
                "target_platforms": ["hubspot"],
                "content_data": {
                    "title": "${title}",
                    "content": "${content}",
                    "author": "${author}",
                    "tags": "${tags}"
                }
            },
            dependencies=["transform_content"]
        ),
        WorkflowStep(
            id="send_notification",
            name="Send Completion Notification",
            step_type=StepType.NOTIFICATION,
            config={
                "type": "slack",
                "message": "Content distribution completed for: ${title}"
            },
            dependencies=["integrate_wordpress", "integrate_hubspot"]
        )
    ]
    
    return WorkflowDefinition(
        id="content_distribution",
        name="Content Distribution Workflow",
        description="Distribute content to multiple platforms with validation and notifications",
        version="1.0.0",
        steps=steps,
        variables={
            "default_author": "AI Assistant",
            "default_tags": ["AI", "Generated"]
        }
    )

def create_conditional_publishing_workflow() -> WorkflowDefinition:
    """Create a conditional publishing workflow template"""
    steps = [
        WorkflowStep(
            id="analyze_content",
            name="Analyze Content",
            step_type=StepType.SCRIPT,
            config={
                "script": """
# Analyze content sentiment and quality
content_length = len(variables.get('content', ''))
word_count = len(variables.get('content', '').split())
variables['content_length'] = content_length
variables['word_count'] = word_count
variables['is_high_quality'] = word_count > 500 and content_length > 2000
"""
            }
        ),
        WorkflowStep(
            id="check_quality",
            name="Check Content Quality",
            step_type=StepType.CONDITION,
            config={
                "condition": "variables.get('is_high_quality', False)",
                "next_step_if_true": "publish_premium",
                "next_step_if_false": "publish_standard"
            },
            dependencies=["analyze_content"]
        ),
        WorkflowStep(
            id="publish_premium",
            name="Publish to Premium Platforms",
            step_type=StepType.INTEGRATION,
            config={
                "content_type": "blog_post",
                "target_platforms": ["wordpress", "hubspot", "salesforce"],
                "content_data": {
                    "title": "${title}",
                    "content": "${content}",
                    "author": "${author}",
                    "tags": "${tags}",
                    "category": "premium"
                }
            },
            dependencies=["check_quality"]
        ),
        WorkflowStep(
            id="publish_standard",
            name="Publish to Standard Platforms",
            step_type=StepType.INTEGRATION,
            config={
                "content_type": "blog_post",
                "target_platforms": ["wordpress"],
                "content_data": {
                    "title": "${title}",
                    "content": "${content}",
                    "author": "${author}",
                    "tags": "${tags}",
                    "category": "standard"
                }
            },
            dependencies=["check_quality"]
        ),
        WorkflowStep(
            id="schedule_followup",
            name="Schedule Follow-up",
            step_type=StepType.DELAY,
            config={
                "delay_seconds": 3600  # 1 hour delay
            },
            dependencies=["publish_premium", "publish_standard"]
        ),
        WorkflowStep(
            id="send_analytics",
            name="Send Analytics Webhook",
            step_type=StepType.WEBHOOK,
            config={
                "url": "https://analytics.example.com/webhook",
                "method": "POST",
                "payload": {
                    "content_id": "${content_id}",
                    "platforms": "${target_platforms}",
                    "quality_score": "${is_high_quality}",
                    "word_count": "${word_count}"
                }
            },
            dependencies=["schedule_followup"]
        )
    ]
    
    return WorkflowDefinition(
        id="conditional_publishing",
        name="Conditional Publishing Workflow",
        description="Publish content based on quality analysis with follow-up actions",
        version="1.0.0",
        steps=steps,
        variables={
            "analytics_endpoint": "https://analytics.example.com",
            "quality_threshold": 500
        }
    )

# Initialize with default workflows
def initialize_workflow_engine():
    """Initialize workflow engine with default workflows"""
    # Register default workflows
    workflow_engine.register_workflow(create_content_distribution_workflow())
    workflow_engine.register_workflow(create_conditional_publishing_workflow())
    
    logger.info("Workflow engine initialized with default workflows")

# Export main components
__all__ = [
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowExecution",
    "WorkflowStatus",
    "StepStatus",
    "StepType",
    "workflow_engine",
    "create_content_distribution_workflow",
    "create_conditional_publishing_workflow",
    "initialize_workflow_engine"
]



























