"""
Advanced Workflow Service for Facebook Posts API
Automated workflows, content pipelines, and business process automation
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog

from ..core.config import get_settings
from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType, OptimizationLevel
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed
from ..infrastructure.database import get_db_manager, PostRepository
from ..services.ai_service import get_ai_service
from ..services.analytics_service import get_analytics_service
from ..services.ml_service import get_ml_service
from ..services.optimization_service import get_optimization_service
from ..services.recommendation_service import get_recommendation_service
from ..services.notification_service import get_notification_service
from ..services.security_service import get_security_service

logger = structlog.get_logger(__name__)


class WorkflowStatus(Enum):
    """Workflow status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowTrigger(Enum):
    """Workflow trigger enumeration"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    WEBHOOK = "webhook"
    API = "api"


class StepStatus(Enum):
    """Step status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Workflow step data structure"""
    id: str
    name: str
    description: str
    step_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Workflow data structure"""
    id: str
    name: str
    description: str
    version: str
    trigger: WorkflowTrigger
    steps: List[WorkflowStep]
    config: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution data structure"""
    id: str
    workflow_id: str
    workflow_version: str
    status: WorkflowStatus
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    executed_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowStepExecutor:
    """Workflow step executor"""
    
    def __init__(self):
        self.ai_service = get_ai_service()
        self.analytics_service = get_analytics_service()
        self.ml_service = get_ml_service()
        self.optimization_service = get_optimization_service()
        self.recommendation_service = get_recommendation_service()
        self.notification_service = get_notification_service()
        self.security_service = get_security_service()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("workflow_step_execution")
    async def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step"""
        try:
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now()
            
            # Execute step based on type
            if step.step_type == "ai_generate":
                result = await self._execute_ai_generate(step, context)
            elif step.step_type == "ai_analyze":
                result = await self._execute_ai_analyze(step, context)
            elif step.step_type == "ml_predict":
                result = await self._execute_ml_predict(step, context)
            elif step.step_type == "optimize_content":
                result = await self._execute_optimize_content(step, context)
            elif step.step_type == "recommend_content":
                result = await self._execute_recommend_content(step, context)
            elif step.step_type == "send_notification":
                result = await self._execute_send_notification(step, context)
            elif step.step_type == "schedule_post":
                result = await self._execute_schedule_post(step, context)
            elif step.step_type == "publish_post":
                result = await self._execute_publish_post(step, context)
            elif step.step_type == "track_analytics":
                result = await self._execute_track_analytics(step, context)
            elif step.step_type == "conditional":
                result = await self._execute_conditional(step, context)
            elif step.step_type == "loop":
                result = await self._execute_loop(step, context)
            elif step.step_type == "parallel":
                result = await self._execute_parallel(step, context)
            elif step.step_type == "wait":
                result = await self._execute_wait(step, context)
            elif step.step_type == "custom":
                result = await self._execute_custom(step, context)
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")
            
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()
            step.output = result
            
            logger.info("Workflow step completed", step_id=step.id, step_type=step.step_type)
            return result
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.completed_at = datetime.now()
            
            logger.error("Workflow step failed", step_id=step.id, error=str(e))
            raise
    
    async def _execute_ai_generate(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI content generation step"""
        config = step.config
        
        # Get parameters from context or config
        topic = config.get("topic") or context.get("topic")
        audience_type = config.get("audience_type") or context.get("audience_type")
        content_type = config.get("content_type") or context.get("content_type")
        
        if not all([topic, audience_type, content_type]):
            raise ValueError("Missing required parameters for AI generation")
        
        # Generate content
        from ..core.models import PostRequest
        request = PostRequest(
            topic=topic,
            content_type=ContentType(content_type),
            audience_type=AudienceType(audience_type),
            tone=config.get("tone", "professional")
        )
        
        result = await self.ai_service.generate_content(request)
        
        return {
            "content": result.content if result else None,
            "metadata": result.metadata if result else {},
            "success": result is not None
        }
    
    async def _execute_ai_analyze(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI content analysis step"""
        config = step.config
        
        # Get content from context
        content = config.get("content") or context.get("content")
        if not content:
            raise ValueError("Content is required for AI analysis")
        
        # Analyze content
        result = await self.ai_service.analyze_content(content)
        
        return {
            "analysis": result.analysis if result else {},
            "scores": result.scores if result else {},
            "suggestions": result.suggestions if result else [],
            "success": result is not None
        }
    
    async def _execute_ml_predict(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ML prediction step"""
        config = step.config
        
        # Get parameters
        content = config.get("content") or context.get("content")
        prediction_type = config.get("prediction_type", "engagement")
        
        if not content:
            raise ValueError("Content is required for ML prediction")
        
        # Make prediction
        if prediction_type == "engagement":
            result = await self.ml_service.predict_engagement(content, context)
        elif prediction_type == "reach":
            result = await self.ml_service.predict_reach(content, context)
        elif prediction_type == "clicks":
            result = await self.ml_service.predict_clicks(content, context)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        
        return {
            "prediction": result.predicted_value if result else 0,
            "confidence": result.confidence if result else 0,
            "prediction_type": prediction_type,
            "success": result is not None
        }
    
    async def _execute_optimize_content(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content optimization step"""
        config = step.config
        
        # Get parameters
        content = config.get("content") or context.get("content")
        strategy = config.get("strategy", "engagement")
        
        if not content:
            raise ValueError("Content is required for optimization")
        
        # Optimize content
        result = await self.optimization_service.optimize_content(
            content=content,
            strategy=strategy,
            context=context
        )
        
        return {
            "optimized_content": result.optimized_content if result else content,
            "optimizations": result.optimizations if result else [],
            "improvement_score": result.improvement_score if result else 0,
            "success": result is not None
        }
    
    async def _execute_recommend_content(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content recommendation step"""
        config = step.config
        
        # Get parameters
        user_id = config.get("user_id") or context.get("user_id")
        audience_type = config.get("audience_type") or context.get("audience_type")
        content_type = config.get("content_type") or context.get("content_type")
        
        if not all([user_id, audience_type]):
            raise ValueError("Missing required parameters for content recommendation")
        
        # Get recommendations
        result = await self.recommendation_service.get_comprehensive_recommendations(
            user_id=user_id,
            audience_type=AudienceType(audience_type),
            content_type=ContentType(content_type) if content_type else None,
            limit=config.get("limit", 5)
        )
        
        return {
            "recommendations": [
                {
                    "type": rec.type.value,
                    "title": rec.title,
                    "suggestion": rec.suggestion,
                    "confidence": rec.confidence
                }
                for rec in result.recommendations
            ],
            "personalization_score": result.personalization_score,
            "success": True
        }
    
    async def _execute_send_notification(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notification sending step"""
        config = step.config
        
        # Get parameters
        notification_type = config.get("notification_type", "email")
        priority = config.get("priority", "medium")
        title = config.get("title") or context.get("title")
        message = config.get("message") or context.get("message")
        recipient = config.get("recipient") or context.get("recipient")
        
        if not all([title, message, recipient]):
            raise ValueError("Missing required parameters for notification")
        
        # Send notification
        notification_id = await self.notification_service.send_notification(
            notification_type=notification_type,
            priority=priority,
            title=title,
            message=message,
            recipient=recipient,
            metadata=config.get("metadata", {})
        )
        
        return {
            "notification_id": notification_id,
            "notification_type": notification_type,
            "recipient": recipient,
            "success": True
        }
    
    async def _execute_schedule_post(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post scheduling step"""
        config = step.config
        
        # Get parameters
        content = config.get("content") or context.get("content")
        schedule_time = config.get("schedule_time") or context.get("schedule_time")
        
        if not content:
            raise ValueError("Content is required for post scheduling")
        
        # Mock post scheduling (in real implementation, integrate with scheduling service)
        post_id = f"post_{int(time.time())}"
        
        return {
            "post_id": post_id,
            "content": content,
            "schedule_time": schedule_time,
            "status": "scheduled",
            "success": True
        }
    
    async def _execute_publish_post(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post publishing step"""
        config = step.config
        
        # Get parameters
        content = config.get("content") or context.get("content")
        platform = config.get("platform", "facebook")
        
        if not content:
            raise ValueError("Content is required for post publishing")
        
        # Mock post publishing (in real implementation, integrate with social media APIs)
        post_id = f"published_post_{int(time.time())}"
        
        return {
            "post_id": post_id,
            "content": content,
            "platform": platform,
            "status": "published",
            "published_at": datetime.now().isoformat(),
            "success": True
        }
    
    async def _execute_track_analytics(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analytics tracking step"""
        config = step.config
        
        # Get parameters
        post_id = config.get("post_id") or context.get("post_id")
        metrics = config.get("metrics", ["engagement", "reach", "clicks"])
        
        if not post_id:
            raise ValueError("Post ID is required for analytics tracking")
        
        # Mock analytics tracking (in real implementation, integrate with analytics service)
        analytics_data = {
            "post_id": post_id,
            "metrics": {metric: 0 for metric in metrics},
            "tracked_at": datetime.now().isoformat()
        }
        
        return {
            "analytics_data": analytics_data,
            "success": True
        }
    
    async def _execute_conditional(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conditional step"""
        config = step.config
        
        # Get condition
        condition = config.get("condition")
        if not condition:
            raise ValueError("Condition is required for conditional step")
        
        # Evaluate condition
        condition_result = self._evaluate_condition(condition, context)
        
        return {
            "condition": condition,
            "result": condition_result,
            "success": True
        }
    
    async def _execute_loop(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute loop step"""
        config = step.config
        
        # Get loop parameters
        loop_count = config.get("loop_count", 1)
        loop_items = config.get("loop_items", [])
        
        results = []
        
        if loop_items:
            # Loop over items
            for item in loop_items:
                item_context = {**context, "loop_item": item}
                # Execute sub-steps or custom logic
                result = await self._execute_loop_iteration(step, item_context)
                results.append(result)
        else:
            # Simple count loop
            for i in range(loop_count):
                item_context = {**context, "loop_index": i}
                result = await self._execute_loop_iteration(step, item_context)
                results.append(result)
        
        return {
            "loop_count": len(results),
            "results": results,
            "success": True
        }
    
    async def _execute_loop_iteration(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single loop iteration"""
        # Mock loop iteration execution
        return {
            "iteration_result": "completed",
            "context": context
        }
    
    async def _execute_parallel(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel step"""
        config = step.config
        
        # Get parallel tasks
        tasks = config.get("tasks", [])
        
        if not tasks:
            raise ValueError("Tasks are required for parallel step")
        
        # Execute tasks in parallel
        results = await asyncio.gather(
            *[self._execute_parallel_task(task, context) for task in tasks],
            return_exceptions=True
        )
        
        return {
            "task_count": len(tasks),
            "results": results,
            "success": True
        }
    
    async def _execute_parallel_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single parallel task"""
        # Mock parallel task execution
        return {
            "task_id": task.get("id", "unknown"),
            "result": "completed"
        }
    
    async def _execute_wait(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute wait step"""
        config = step.config
        
        # Get wait duration
        wait_seconds = config.get("wait_seconds", 1)
        
        # Wait for specified duration
        await asyncio.sleep(wait_seconds)
        
        return {
            "wait_seconds": wait_seconds,
            "waited_at": datetime.now().isoformat(),
            "success": True
        }
    
    async def _execute_custom(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom step"""
        config = step.config
        
        # Get custom function or script
        custom_function = config.get("custom_function")
        custom_script = config.get("custom_script")
        
        if custom_function:
            # Execute custom function
            result = await self._execute_custom_function(custom_function, context)
        elif custom_script:
            # Execute custom script
            result = await self._execute_custom_script(custom_script, context)
        else:
            raise ValueError("Custom function or script is required for custom step")
        
        return {
            "custom_result": result,
            "success": True
        }
    
    async def _execute_custom_function(self, function_name: str, context: Dict[str, Any]) -> Any:
        """Execute custom function"""
        # Mock custom function execution
        return f"Custom function {function_name} executed"
    
    async def _execute_custom_script(self, script: str, context: Dict[str, Any]) -> Any:
        """Execute custom script"""
        # Mock custom script execution
        return f"Custom script executed: {script[:50]}..."
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context"""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            if "==" in condition:
                left, right = condition.split("==", 1)
                left = left.strip()
                right = right.strip().strip("'\"")
                return str(context.get(left, "")) == right
            elif "!=" in condition:
                left, right = condition.split("!=", 1)
                left = left.strip()
                right = right.strip().strip("'\"")
                return str(context.get(left, "")) != right
            elif ">" in condition:
                left, right = condition.split(">", 1)
                left = left.strip()
                right = right.strip()
                return float(context.get(left, 0)) > float(right)
            elif "<" in condition:
                left, right = condition.split("<", 1)
                left = left.strip()
                right = right.strip()
                return float(context.get(left, 0)) < float(right)
            else:
                return False
        except Exception as e:
            logger.error("Failed to evaluate condition", condition=condition, error=str(e))
            return False


class WorkflowEngine:
    """Workflow execution engine"""
    
    def __init__(self):
        self.step_executor = WorkflowStepExecutor()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.workflows: Dict[str, Workflow] = {}
        self._load_default_workflows()
    
    def _load_default_workflows(self):
        """Load default workflows"""
        default_workflows = [
            Workflow(
                id="content_creation_pipeline",
                name="Content Creation Pipeline",
                description="Automated content creation and optimization pipeline",
                version="1.0",
                trigger=WorkflowTrigger.MANUAL,
                steps=[
                    WorkflowStep(
                        id="generate_content",
                        name="Generate Content",
                        description="Generate content using AI",
                        step_type="ai_generate",
                        config={"topic": "{{topic}}", "audience_type": "{{audience_type}}", "content_type": "{{content_type}}"}
                    ),
                    WorkflowStep(
                        id="analyze_content",
                        name="Analyze Content",
                        description="Analyze generated content",
                        step_type="ai_analyze",
                        config={"content": "{{generate_content.content}}"},
                        dependencies=["generate_content"]
                    ),
                    WorkflowStep(
                        id="optimize_content",
                        name="Optimize Content",
                        description="Optimize content for better performance",
                        step_type="optimize_content",
                        config={"content": "{{generate_content.content}}", "strategy": "engagement"},
                        dependencies=["analyze_content"]
                    ),
                    WorkflowStep(
                        id="schedule_post",
                        name="Schedule Post",
                        description="Schedule optimized content for posting",
                        step_type="schedule_post",
                        config={"content": "{{optimize_content.optimized_content}}"},
                        dependencies=["optimize_content"]
                    )
                ]
            ),
            Workflow(
                id="engagement_monitoring",
                name="Engagement Monitoring",
                description="Monitor post engagement and send alerts",
                version="1.0",
                trigger=WorkflowTrigger.EVENT,
                steps=[
                    WorkflowStep(
                        id="track_analytics",
                        name="Track Analytics",
                        description="Track post analytics",
                        step_type="track_analytics",
                        config={"post_id": "{{post_id}}", "metrics": ["engagement", "reach", "clicks"]}
                    ),
                    WorkflowStep(
                        id="check_engagement",
                        name="Check Engagement",
                        description="Check if engagement is high",
                        step_type="conditional",
                        config={"condition": "engagement_rate > 0.8"}
                    ),
                    WorkflowStep(
                        id="send_alert",
                        name="Send Alert",
                        description="Send high engagement alert",
                        step_type="send_notification",
                        config={
                            "notification_type": "email",
                            "priority": "high",
                            "title": "High Engagement Alert",
                            "message": "Your post is performing exceptionally well!",
                            "recipient": "{{user_email}}"
                        },
                        dependencies=["check_engagement"]
                    )
                ]
            ),
            Workflow(
                id="content_optimization_loop",
                name="Content Optimization Loop",
                description="Continuously optimize content based on performance",
                version="1.0",
                trigger=WorkflowTrigger.SCHEDULED,
                steps=[
                    WorkflowStep(
                        id="get_posts",
                        name="Get Posts",
                        description="Get posts for optimization",
                        step_type="custom",
                        config={"custom_function": "get_posts_for_optimization"}
                    ),
                    WorkflowStep(
                        id="optimize_loop",
                        name="Optimize Loop",
                        description="Loop through posts and optimize",
                        step_type="loop",
                        config={"loop_items": "{{get_posts.posts}}"}
                    ),
                    WorkflowStep(
                        id="update_posts",
                        name="Update Posts",
                        description="Update optimized posts",
                        step_type="custom",
                        config={"custom_function": "update_optimized_posts"},
                        dependencies=["optimize_loop"]
                    )
                ]
            )
        ]
        
        for workflow in default_workflows:
            self.workflows[workflow.id] = workflow
    
    @timed("workflow_execution")
    async def execute_workflow(
        self,
        workflow_id: str,
        trigger_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        executed_by: Optional[str] = None
    ) -> WorkflowExecution:
        """Execute a workflow"""
        try:
            # Get workflow
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            # Create execution
            execution = WorkflowExecution(
                id=f"exec_{int(time.time())}_{workflow_id}",
                workflow_id=workflow_id,
                workflow_version=workflow.version,
                status=WorkflowStatus.RUNNING,
                trigger_data=trigger_data,
                context=context or {},
                executed_by=executed_by
            )
            
            self.active_executions[execution.id] = execution
            
            # Execute workflow steps
            await self._execute_workflow_steps(workflow, execution)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            logger.info("Workflow execution completed", execution_id=execution.id, workflow_id=workflow_id)
            return execution
            
        except Exception as e:
            if execution:
                execution.status = WorkflowStatus.FAILED
                execution.completed_at = datetime.now()
                execution.metadata["error"] = str(e)
            
            logger.error("Workflow execution failed", workflow_id=workflow_id, error=str(e))
            raise
    
    async def _execute_workflow_steps(self, workflow: Workflow, execution: WorkflowExecution):
        """Execute workflow steps"""
        # Create step context
        step_context = {**execution.context, **execution.trigger_data}
        
        # Execute steps in dependency order
        completed_steps = set()
        
        while len(completed_steps) < len(workflow.steps):
            # Find steps that can be executed
            executable_steps = []
            for step in workflow.steps:
                if step.id in completed_steps:
                    continue
                
                # Check if all dependencies are completed
                if all(dep in completed_steps for dep in step.dependencies):
                    executable_steps.append(step)
            
            if not executable_steps:
                raise ValueError("No executable steps found - possible circular dependency")
            
            # Execute executable steps
            for step in executable_steps:
                try:
                    # Resolve step context
                    resolved_context = self._resolve_context(step_context, step.config)
                    
                    # Execute step
                    result = await self.step_executor.execute_step(step, resolved_context)
                    
                    # Update context with step result
                    step_context[f"{step.id}_result"] = result
                    step_context.update(result)
                    
                    completed_steps.add(step.id)
                    
                except Exception as e:
                    # Handle step failure
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        logger.warning("Step failed, retrying", step_id=step.id, retry_count=step.retry_count)
                        await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                    else:
                        logger.error("Step failed after max retries", step_id=step.id)
                        raise
    
    def _resolve_context(self, context: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve context variables in config"""
        resolved_config = {}
        
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Resolve context variable
                var_name = value[2:-2].strip()
                resolved_config[key] = context.get(var_name, value)
            else:
                resolved_config[key] = value
        
        return resolved_config
    
    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "executed_by": execution.executed_by,
            "metadata": execution.metadata
        }
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return False
        
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now()
        
        logger.info("Workflow execution cancelled", execution_id=execution_id)
        return True


class WorkflowService:
    """Main workflow service orchestrator"""
    
    def __init__(self):
        self.workflow_engine = WorkflowEngine()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("workflow_service_execution")
    async def execute_workflow(
        self,
        workflow_id: str,
        trigger_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        executed_by: Optional[str] = None
    ) -> str:
        """Execute workflow and return execution ID"""
        try:
            execution = await self.workflow_engine.execute_workflow(
                workflow_id=workflow_id,
                trigger_data=trigger_data,
                context=context,
                executed_by=executed_by
            )
            
            return execution.id
            
        except Exception as e:
            logger.error("Workflow execution failed", workflow_id=workflow_id, error=str(e))
            raise
    
    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        return await self.workflow_engine.get_workflow_status(execution_id)
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        return await self.workflow_engine.cancel_workflow(execution_id)
    
    async def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get list of available workflows"""
        workflows = []
        for workflow_id, workflow in self.workflow_engine.workflows.items():
            workflows.append({
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "trigger": workflow.trigger.value,
                "step_count": len(workflow.steps),
                "created_at": workflow.created_at.isoformat()
            })
        
        return workflows


# Global workflow service instance
_workflow_service: Optional[WorkflowService] = None


def get_workflow_service() -> WorkflowService:
    """Get global workflow service instance"""
    global _workflow_service
    
    if _workflow_service is None:
        _workflow_service = WorkflowService()
    
    return _workflow_service


# Export all classes and functions
__all__ = [
    # Enums
    'WorkflowStatus',
    'WorkflowTrigger',
    'StepStatus',
    
    # Data classes
    'WorkflowStep',
    'Workflow',
    'WorkflowExecution',
    
    # Services
    'WorkflowStepExecutor',
    'WorkflowEngine',
    'WorkflowService',
    
    # Utility functions
    'get_workflow_service',
]






























