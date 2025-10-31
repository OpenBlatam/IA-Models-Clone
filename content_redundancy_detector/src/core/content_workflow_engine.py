"""
Content Workflow Engine - Advanced workflow automation for content processing
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from abc import ABC, abstractmethod

import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TriggerType(Enum):
    """Workflow trigger types"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    WEBHOOK = "webhook"
    CONDITION = "condition"


@dataclass
class WorkflowStep:
    """Individual workflow step"""
    step_id: str
    name: str
    step_type: str
    config: Dict[str, Any]
    dependencies: List[str]
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    context: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 1


@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    triggers: List[Dict[str, Any]]
    variables: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class WorkflowStepHandler(ABC):
    """Abstract base class for workflow step handlers"""
    
    @abstractmethod
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step"""
        pass
    
    @abstractmethod
    async def validate(self, step: WorkflowStep, context: Dict[str, Any]) -> bool:
        """Validate step configuration"""
        pass
    
    @abstractmethod
    async def rollback(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback step execution"""
        pass


class ContentAnalysisStepHandler(WorkflowStepHandler):
    """Handler for content analysis steps"""
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content analysis step"""
        try:
            content = context.get("content", "")
            analysis_type = step.config.get("analysis_type", "comprehensive")
            
            # Import here to avoid circular imports
            from .ai_content_analyzer import analyze_content_with_ai
            
            # Perform analysis
            analysis_result = await analyze_content_with_ai(content, step.step_id)
            
            return {
                "analysis_result": asdict(analysis_result),
                "analysis_type": analysis_type,
                "content_length": len(content),
                "processing_time": (datetime.now() - step.started_at).total_seconds() if step.started_at else 0
            }
            
        except Exception as e:
            logger.error(f"Content analysis step failed: {e}")
            raise
    
    async def validate(self, step: WorkflowStep, context: Dict[str, Any]) -> bool:
        """Validate content analysis step"""
        required_config = ["analysis_type"]
        return all(key in step.config for key in required_config)
    
    async def rollback(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback content analysis step"""
        return {"status": "rolled_back", "step_id": step.step_id}


class ContentOptimizationStepHandler(WorkflowStepHandler):
    """Handler for content optimization steps"""
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content optimization step"""
        try:
            content = context.get("content", "")
            optimization_goals = step.config.get("optimization_goals", ["readability", "seo"])
            
            # Import here to avoid circular imports
            from .content_optimizer import optimize_content
            
            # Perform optimization
            optimization_result = await optimize_content(content, step.step_id, optimization_goals)
            
            return {
                "optimization_result": asdict(optimization_result),
                "optimization_goals": optimization_goals,
                "improvement_score": optimization_result.optimization_score,
                "suggestions_count": len(optimization_result.suggestions)
            }
            
        except Exception as e:
            logger.error(f"Content optimization step failed: {e}")
            raise
    
    async def validate(self, step: WorkflowStep, context: Dict[str, Any]) -> bool:
        """Validate content optimization step"""
        required_config = ["optimization_goals"]
        return all(key in step.config for key in required_config)
    
    async def rollback(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback content optimization step"""
        return {"status": "rolled_back", "step_id": step.step_id}


class SimilarityAnalysisStepHandler(WorkflowStepHandler):
    """Handler for similarity analysis steps"""
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute similarity analysis step"""
        try:
            content_list = context.get("content_list", [])
            similarity_threshold = step.config.get("similarity_threshold", 0.7)
            
            # Import here to avoid circular imports
            from .advanced_analytics import generate_redundancy_report
            
            # Perform similarity analysis
            redundancy_report = await generate_redundancy_report(
                content_list, similarity_threshold, "comprehensive"
            )
            
            return {
                "redundancy_report": asdict(redundancy_report),
                "similarity_threshold": similarity_threshold,
                "duplicate_groups": len(redundancy_report.duplicate_groups),
                "total_items": redundancy_report.total_content_items
            }
            
        except Exception as e:
            logger.error(f"Similarity analysis step failed: {e}")
            raise
    
    async def validate(self, step: WorkflowStep, context: Dict[str, Any]) -> bool:
        """Validate similarity analysis step"""
        return "content_list" in context and len(context["content_list"]) >= 2
    
    async def rollback(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback similarity analysis step"""
        return {"status": "rolled_back", "step_id": step.step_id}


class NotificationStepHandler(WorkflowStepHandler):
    """Handler for notification steps"""
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification step"""
        try:
            message = step.config.get("message", "Workflow step completed")
            notification_type = step.config.get("type", "info")
            recipients = step.config.get("recipients", [])
            
            # Simulate notification sending
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                "notification_sent": True,
                "message": message,
                "type": notification_type,
                "recipients": recipients,
                "sent_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Notification step failed: {e}")
            raise
    
    async def validate(self, step: WorkflowStep, context: Dict[str, Any]) -> bool:
        """Validate notification step"""
        required_config = ["message"]
        return all(key in step.config for key in required_config)
    
    async def rollback(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback notification step"""
        return {"status": "rolled_back", "step_id": step.step_id}


class WorkflowEngine:
    """Advanced workflow engine for content processing"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.step_handlers: Dict[str, WorkflowStepHandler] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Register default step handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default step handlers"""
        self.step_handlers["content_analysis"] = ContentAnalysisStepHandler()
        self.step_handlers["content_optimization"] = ContentOptimizationStepHandler()
        self.step_handlers["similarity_analysis"] = SimilarityAnalysisStepHandler()
        self.step_handlers["notification"] = NotificationStepHandler()
    
    async def initialize(self) -> None:
        """Initialize the workflow engine"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                db=2, 
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Load existing workflows
            await self._load_workflows()
            
            logger.info("Workflow engine initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory workflow engine: {e}")
            self.redis_client = None
    
    async def start(self) -> None:
        """Start the workflow engine"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        worker_task = asyncio.create_task(self._workflow_worker())
        self.worker_tasks.append(worker_task)
        
        logger.info("Workflow engine started")
    
    async def stop(self) -> None:
        """Stop the workflow engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Workflow engine stopped")
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        triggers: List[Dict[str, Any]] = None,
        variables: Dict[str, Any] = None
    ) -> str:
        """Create a new workflow definition"""
        
        workflow_id = str(uuid.uuid4())
        
        # Convert step dictionaries to WorkflowStep objects
        workflow_steps = []
        for step_data in steps:
            step = WorkflowStep(
                step_id=step_data["step_id"],
                name=step_data["name"],
                step_type=step_data["step_type"],
                config=step_data.get("config", {}),
                dependencies=step_data.get("dependencies", []),
                max_retries=step_data.get("max_retries", 3),
                timeout=step_data.get("timeout", 300)
            )
            workflow_steps.append(step)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            version="1.0.0",
            steps=workflow_steps,
            triggers=triggers or [],
            variables=variables or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store workflow
        self.workflows[workflow_id] = workflow
        
        # Save to Redis if available
        if self.redis_client:
            await self._save_workflow(workflow)
        
        logger.info(f"Workflow created: {name} ({workflow_id})")
        return workflow_id
    
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Dict[str, Any],
        trigger_type: TriggerType = TriggerType.MANUAL
    ) -> str:
        """Execute a workflow"""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        # Create execution instance
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            steps=[step for step in workflow.steps],  # Copy steps
            context=context.copy(),
            created_at=datetime.now()
        )
        
        # Store execution
        self.executions[execution_id] = execution
        
        # Save to Redis if available
        if self.redis_client:
            await self._save_execution(execution)
        
        # Queue for execution
        if self.redis_client:
            await self.redis_client.lpush("workflow_queue", execution_id)
        
        logger.info(f"Workflow execution queued: {execution_id}")
        return execution_id
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "created_at": execution.created_at.isoformat(),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error": execution.error,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": step.status.value,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error": step.error,
                    "retry_count": step.retry_count
                }
                for step in execution.steps
            ]
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        
        if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            return False
        
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now()
        
        # Save to Redis if available
        if self.redis_client:
            await self._save_execution(execution)
        
        logger.info(f"Workflow execution cancelled: {execution_id}")
        return True
    
    async def _workflow_worker(self) -> None:
        """Background worker for processing workflows"""
        
        while self.is_running:
            try:
                if self.redis_client:
                    # Get next execution from queue
                    execution_id = await self.redis_client.brpop("workflow_queue", timeout=1)
                    if execution_id:
                        execution_id = execution_id[1]
                    else:
                        continue
                else:
                    # Process pending executions
                    pending_executions = [
                        exec_id for exec_id, execution in self.executions.items()
                        if execution.status == WorkflowStatus.PENDING
                    ]
                    
                    if not pending_executions:
                        await asyncio.sleep(1)
                        continue
                    
                    execution_id = pending_executions[0]
                
                # Process execution
                await self._process_execution(execution_id)
                
            except Exception as e:
                logger.error(f"Workflow worker error: {e}")
                await asyncio.sleep(5)
    
    async def _process_execution(self, execution_id: str) -> None:
        """Process a workflow execution"""
        
        if execution_id not in self.executions:
            return
        
        execution = self.executions[execution_id]
        workflow = self.workflows[execution.workflow_id]
        
        try:
            # Update status
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            
            # Process steps in dependency order
            completed_steps = set()
            
            while len(completed_steps) < len(execution.steps):
                # Find steps that can be executed
                ready_steps = [
                    step for step in execution.steps
                    if step.step_id not in completed_steps
                    and all(dep in completed_steps for dep in step.dependencies)
                    and step.status == StepStatus.PENDING
                ]
                
                if not ready_steps:
                    # Check if we're stuck
                    remaining_steps = [
                        step for step in execution.steps
                        if step.step_id not in completed_steps
                    ]
                    
                    if remaining_steps:
                        # Mark remaining steps as failed
                        for step in remaining_steps:
                            step.status = StepStatus.FAILED
                            step.error = "Dependency resolution failed"
                        
                        execution.status = WorkflowStatus.FAILED
                        execution.error = "Workflow execution stuck"
                        break
                    else:
                        break
                
                # Execute ready steps in parallel
                step_tasks = [
                    self._execute_step(step, execution.context)
                    for step in ready_steps
                ]
                
                await asyncio.gather(*step_tasks, return_exceptions=True)
                
                # Update completed steps
                for step in ready_steps:
                    if step.status == StepStatus.COMPLETED:
                        completed_steps.add(step.step_id)
                    elif step.status == StepStatus.FAILED and step.retry_count < step.max_retries:
                        # Retry failed step
                        step.retry_count += 1
                        step.status = StepStatus.RETRYING
                        await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                        step.status = StepStatus.PENDING
                    elif step.status == StepStatus.FAILED:
                        # Step failed permanently
                        execution.status = WorkflowStatus.FAILED
                        execution.error = f"Step {step.step_id} failed permanently"
                        break
            
            # Update final status
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now()
            
            # Save execution
            if self.redis_client:
                await self._save_execution(execution)
            
            logger.info(f"Workflow execution completed: {execution_id} - {execution.status.value}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            
            if self.redis_client:
                await self._save_execution(execution)
            
            logger.error(f"Workflow execution failed: {execution_id} - {e}")
    
    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> None:
        """Execute a single workflow step"""
        
        try:
            # Update step status
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now()
            
            # Get step handler
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"No handler found for step type: {step.step_type}")
            
            # Validate step
            if not await handler.validate(step, context):
                raise ValueError(f"Step validation failed: {step.step_id}")
            
            # Execute step with timeout
            result = await asyncio.wait_for(
                handler.execute(step, context),
                timeout=step.timeout
            )
            
            # Update step result
            step.result = result
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()
            
            # Update context with step result
            context[f"step_{step.step_id}_result"] = result
            
            logger.info(f"Step completed: {step.step_id}")
            
        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error = f"Step timeout after {step.timeout} seconds"
            step.completed_at = datetime.now()
            logger.error(f"Step timeout: {step.step_id}")
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now()
            logger.error(f"Step failed: {step.step_id} - {e}")
    
    async def _load_workflows(self) -> None:
        """Load workflows from Redis"""
        if not self.redis_client:
            return
        
        try:
            workflow_keys = await self.redis_client.keys("workflow:*")
            for key in workflow_keys:
                workflow_data = await self.redis_client.get(key)
                if workflow_data:
                    workflow_dict = json.loads(workflow_data)
                    # Convert back to WorkflowDefinition
                    # This is a simplified version - in production, you'd want proper deserialization
                    pass
        except Exception as e:
            logger.warning(f"Failed to load workflows from Redis: {e}")
    
    async def _save_workflow(self, workflow: WorkflowDefinition) -> None:
        """Save workflow to Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"workflow:{workflow.workflow_id}"
            workflow_data = json.dumps(asdict(workflow), default=str)
            await self.redis_client.set(key, workflow_data)
        except Exception as e:
            logger.warning(f"Failed to save workflow to Redis: {e}")
    
    async def _save_execution(self, execution: WorkflowExecution) -> None:
        """Save execution to Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"execution:{execution.execution_id}"
            execution_data = json.dumps(asdict(execution), default=str)
            await self.redis_client.set(key, execution_data)
        except Exception as e:
            logger.warning(f"Failed to save execution to Redis: {e}")
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow engine metrics"""
        
        total_executions = len(self.executions)
        completed_executions = len([e for e in self.executions.values() if e.status == WorkflowStatus.COMPLETED])
        failed_executions = len([e for e in self.executions.values() if e.status == WorkflowStatus.FAILED])
        running_executions = len([e for e in self.executions.values() if e.status == WorkflowStatus.RUNNING])
        
        return {
            "total_workflows": len(self.workflows),
            "total_executions": total_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "running_executions": running_executions,
            "success_rate": completed_executions / total_executions if total_executions > 0 else 0,
            "active_handlers": len(self.step_handlers),
            "is_running": self.is_running,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of workflow engine"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "is_running": self.is_running,
            "total_workflows": len(self.workflows),
            "total_executions": len(self.executions),
            "active_handlers": len(self.step_handlers),
            "redis_connected": self.redis_client is not None,
            "timestamp": datetime.now().isoformat()
        }


# Global workflow engine instance
workflow_engine = WorkflowEngine()


async def initialize_workflow_engine() -> None:
    """Initialize the global workflow engine"""
    await workflow_engine.initialize()
    await workflow_engine.start()


async def shutdown_workflow_engine() -> None:
    """Shutdown the global workflow engine"""
    await workflow_engine.stop()


async def create_workflow(
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]] = None,
    variables: Dict[str, Any] = None
) -> str:
    """Create a new workflow"""
    return await workflow_engine.create_workflow(name, description, steps, triggers, variables)


async def execute_workflow(
    workflow_id: str,
    context: Dict[str, Any],
    trigger_type: TriggerType = TriggerType.MANUAL
) -> str:
    """Execute a workflow"""
    return await workflow_engine.execute_workflow(workflow_id, context, trigger_type)


async def get_execution_status(execution_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow execution status"""
    return await workflow_engine.get_execution_status(execution_id)


async def get_workflow_metrics() -> Dict[str, Any]:
    """Get workflow engine metrics"""
    return await workflow_engine.get_workflow_metrics()


async def get_workflow_health() -> Dict[str, Any]:
    """Get workflow engine health status"""
    return await workflow_engine.health_check()




